"""
read_pdf.py — Module 1: PDF text extraction.
Version 3.0 — 2026-10-01

Changes from v2:
    - Scanned-page processing is now parallel via
      ThreadPoolExecutor. Threads are used (not processes)
      because the bottleneck is the external Tesseract
      subprocess, and the GIL is released while waiting.
    - Result order is preserved by tagging each page result
      with its page number, then sorting before joining.
    - OCR_WORKERS is a module-level constant, easily tuned.

This module implements:
    - detect_pdf_type(pdf_path)      -> (type, diagnostics)
    - extract_digital_text(pdf_path) -> str   (pdfplumber, layout=True)
    - extract_scanned_text(pdf_path) -> dict  (PyMuPDF + Tesseract)
    - extract_pdf(pdf_path)          -> output contract dict

Config is read from config/config.json (produced by
utilities.csv_to_json from config/config.csv).
The target PDF is named in the "File" key and lives in input/.

Outputs (written on every successful run):
    output/<pdf_stem>_<engine>.txt   — extracted text
    output/<pdf_stem>_meta.json      — extraction metadata (audit)
"""

from __future__ import annotations

import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

try:
    import pdfplumber
except ImportError:
    print("[FATAL] pdfplumber is not installed. Run: pip install pdfplumber")
    raise SystemExit(1)

try:
    import fitz  # PyMuPDF
except ImportError:
    print("[FATAL] PyMuPDF is not installed. Run: pip install PyMuPDF")
    raise SystemExit(1)

try:
    from PIL import Image, ImageEnhance, ImageFilter, ImageStat
except ImportError:
    print("[FATAL] Pillow is not installed. Run: pip install Pillow")
    raise SystemExit(1)

try:
    import pytesseract
except ImportError:
    print("[FATAL] pytesseract is not installed. Run: pip install pytesseract")
    raise SystemExit(1)


# ---------------------------------------------------------------------------
# Windows: point pytesseract to the Tesseract binary if it is not on PATH.
# Comment out this line on macOS / Linux if tesseract is on PATH.
# ---------------------------------------------------------------------------
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_JSON  = PROJECT_ROOT / "config" / "config.json"
INPUT_DIR    = PROJECT_ROOT / "input"
OUTPUT_DIR   = PROJECT_ROOT / "output"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Type detection (spec section 3)
DIGITAL_THRESHOLD_CHARS_PER_PAGE = 50
SAMPLE_PAGES = 3

# Digital extraction (spec section 4)
PAGE_SEPARATOR = "\n\n--- PAGE BREAK ---\n\n"

# Scanned extraction (spec section 5)
OCR_DPI = 300
OCR_LANG = "eng+chi_tra"
OCR_PSM = 6
MIN_WIDTH_PX = 1500
BLANK_MEAN_THRESHOLD = 250
DARK_MEAN_THRESHOLD = 20

# Parallelism for scanned-page OCR.
# Tesseract is a single-threaded external process; each call
# briefly spawns a subprocess. 8 workers saturate a 24-core
# machine without memory pressure (approx 25 MB per page image).
# Raise cautiously; above 12 yields diminishing returns.
OCR_WORKERS = 8


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class Module1Error(Exception):
    """Base error for Module 1."""


class ConfigError(Module1Error):
    """Config file missing or malformed."""


class PdfError(Module1Error):
    """PDF unreadable, corrupted, or empty."""


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(config_path: Path = CONFIG_JSON) -> dict[str, Any]:
    """Load config.json produced by utilities.csv_to_json."""
    if not config_path.exists():
        raise ConfigError(f"Config not found: {config_path}")
    try:
        with config_path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
    except json.JSONDecodeError as e:
        raise ConfigError(f"Invalid JSON in {config_path}: {e}")

    if "File" not in cfg:
        raise ConfigError(f"Config missing required key 'File': {config_path}")
    return cfg


# ---------------------------------------------------------------------------
# Type detection (spec section 3)
# ---------------------------------------------------------------------------

def detect_pdf_type(pdf_path: str | Path) -> tuple[str, dict[str, Any]]:
    """
    Classify a PDF as 'digital' or 'scanned' by sampling its first
    few pages and measuring average characters per page.

    Returns:
        (pdf_type, diagnostics)
        pdf_type    : "digital" | "scanned"
        diagnostics : {
            "pages_sampled":      int,
            "per_page_chars":     list[int],
            "avg_chars_per_page": float,
            "threshold":          int,
            "total_pages":        int,
        }

    Raises:
        PdfError if the file is missing or unreadable.
    """
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise PdfError(f"PDF not found: {pdf_path}")
    if not pdf_path.is_file():
        raise PdfError(f"Not a file: {pdf_path}")

    try:
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            if total_pages == 0:
                raise PdfError(f"PDF has zero pages: {pdf_path}")

            n_sample = min(SAMPLE_PAGES, total_pages)
            per_page_chars: list[int] = []

            for i in range(n_sample):
                page = pdf.pages[i]
                text = page.extract_text() or ""
                per_page_chars.append(len(text))
    except PdfError:
        raise
    except Exception as e:
        raise PdfError(f"Failed to open {pdf_path}: {e}")

    avg = sum(per_page_chars) / len(per_page_chars)
    pdf_type = "digital" if avg > DIGITAL_THRESHOLD_CHARS_PER_PAGE else "scanned"

    diagnostics = {
        "pages_sampled":      n_sample,
        "per_page_chars":     per_page_chars,
        "avg_chars_per_page": round(avg, 1),
        "threshold":          DIGITAL_THRESHOLD_CHARS_PER_PAGE,
        "total_pages":        total_pages,
    }
    return pdf_type, diagnostics


# ---------------------------------------------------------------------------
# Digital extraction (spec section 4)
# ---------------------------------------------------------------------------

def extract_digital_text(pdf_path: str | Path) -> str:
    """
    Extract text from a digital PDF using pdfplumber with
    layout=True to preserve column alignment. Pages are joined
    with PAGE_SEPARATOR.
    """
    pdf_path = Path(pdf_path)
    chunks: list[str] = []

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text(layout=True) or ""
                chunks.append(text)
    except Exception as e:
        raise PdfError(f"Digital extraction failed on {pdf_path}: {e}")

    return PAGE_SEPARATOR.join(chunks)


# ---------------------------------------------------------------------------
# Scanned extraction helpers (spec section 5)
# ---------------------------------------------------------------------------

def render_pages_to_images(
    pdf_path: str | Path,
    dpi: int = OCR_DPI,
) -> list[Image.Image]:
    """
    Render each PDF page to a PIL Image at the given DPI using PyMuPDF.
    """
    pdf_path = Path(pdf_path)

    try:
        doc = fitz.open(str(pdf_path))
    except Exception as e:
        raise PdfError(f"PyMuPDF failed to open {pdf_path}: {e}")

    if doc.page_count == 0:
        doc.close()
        raise PdfError(f"No pages in {pdf_path}")

    images: list[Image.Image] = []
    try:
        for page in doc:
            pix = page.get_pixmap(dpi=dpi)
            img = Image.frombytes(
                "RGB", [pix.width, pix.height], pix.samples
            )
            images.append(img)
    finally:
        doc.close()

    if not images:
        raise PdfError(f"No pages rendered from {pdf_path}")
    return images


def preprocess_for_ocr(image: Image.Image) -> Image.Image:
    """
    Grayscale -> contrast x2 -> median filter -> sharpen.
    Per spec section 5.2. Does NOT binarize.
    """
    img = image.convert("L")
    img = ImageEnhance.Contrast(img).enhance(2.0)
    img = img.filter(ImageFilter.MedianFilter(size=3))
    img = img.filter(ImageFilter.SHARPEN)
    return img


def check_image_quality(
    image: Image.Image,
    page_no: int,
) -> tuple[bool, str | None]:
    """
    Return (should_skip, reason).
    Cheap heuristics to skip blank back-sides and scanner artefacts.
    """
    width, _ = image.size
    if width < MIN_WIDTH_PX:
        # Low resolution is worth a warning but not a skip.
        pass

    # Mean brightness: near 255 = blank, near 0 = black.
    # ImageStat.Stat replaces the deprecated Image.getdata().
    small = image.convert("L").resize((100, 100))
    mean = ImageStat.Stat(small).mean[0]

    if mean > BLANK_MEAN_THRESHOLD:
        return True, f"page {page_no} nearly blank (mean={mean:.0f})"
    if mean < DARK_MEAN_THRESHOLD:
        return True, f"page {page_no} nearly black (mean={mean:.0f})"
    return False, None


def run_tesseract(
    image: Image.Image,
    lang: str = OCR_LANG,
    psm: int = OCR_PSM,
) -> str:
    """
    Run Tesseract on a preprocessed image.
    lang : "eng+chi_tra" for HK statements
    psm  : 6 = single uniform block (spec section 5.4)
    """
    config = f"--psm {psm}"
    try:
        return pytesseract.image_to_string(image, lang=lang, config=config)
    except pytesseract.TesseractNotFoundError:
        raise PdfError(
            "Tesseract binary not found on PATH. "
            "Install tesseract-ocr and ensure it is on PATH."
        )
    except pytesseract.TesseractError as e:
        raise PdfError(f"Tesseract failed: {e}")


# ---------------------------------------------------------------------------
# Per-page worker (runs inside a thread)
# ---------------------------------------------------------------------------

def _process_one_page(
    page_no: int,
    raw_img: Image.Image,
) -> tuple[int, str | None, str | None]:
    """
    Process a single scanned page:
        quality check -> preprocess -> Tesseract OCR.

    Executed inside a ThreadPoolExecutor worker thread.

    Returns:
        (page_no, text, skip_reason)
        - text is None if the page was skipped
        - skip_reason is None if the page was processed
    """
    skip, reason = check_image_quality(raw_img, page_no)
    if skip:
        return page_no, None, reason

    processed = preprocess_for_ocr(raw_img)
    text = run_tesseract(processed)
    return page_no, text, None


# ---------------------------------------------------------------------------
# Scanned extraction — parallel
# ---------------------------------------------------------------------------

def extract_scanned_text(
    pdf_path: str | Path,
    max_workers: int = OCR_WORKERS,
) -> dict[str, Any]:
    """
    Scanned path: render -> preprocess -> tesseract, per page.
    Pages are processed in parallel using a thread pool.

    Threads (not processes) are used because the bottleneck is
    the external Tesseract subprocess; the GIL is released while
    waiting for it. ProcessPoolExecutor would add PIL image
    serialisation overhead for no gain.

    Handwriting (DeepSeek Vision) is NOT yet wired in.

    Returns:
        {
            "printed_ocr": str,
            "handwriting": None,
            "warnings":    list[str],
        }
    """
    images = render_pages_to_images(pdf_path)

    printed_chunks: list[tuple[int, str]] = []
    warnings: list[str] = []

    # Submit all pages to the pool. Results arrive out of order,
    # so each carries its page number; we sort before joining.
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_process_one_page, i, img): i
            for i, img in enumerate(images, start=1)
        }

        for future in as_completed(futures):
            page_no, text, skip_reason = future.result()
            if skip_reason is not None:
                warnings.append(skip_reason)
            else:
                body = f"--- Page {page_no} (OCR) ---\n{text.rstrip()}"
                printed_chunks.append((page_no, body))

    if not printed_chunks:
        raise PdfError(f"All pages skipped or empty after OCR: {pdf_path}")

    # Preserve original page order in the output text.
    printed_chunks.sort(key=lambda item: item[0])
    printed_text = "\n\n".join(body for _, body in printed_chunks)

    # Sort warnings by page number for stable, readable output.
    warnings.sort()

    return {
        "printed_ocr": printed_text,
        "handwriting": None,
        "warnings":    warnings,
    }


# ---------------------------------------------------------------------------
# Hashing (audit trail, spec section 11)
# ---------------------------------------------------------------------------

def sha256_of_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Main entry: extract_pdf (spec section 7)
# ---------------------------------------------------------------------------

def extract_pdf(pdf_path: str | Path) -> dict[str, Any]:
    """
    Detect type, route to the right extractor, return the output
    contract dict (spec section 1).
    """
    pdf_path = Path(pdf_path)
    pdf_type, diag = detect_pdf_type(pdf_path)

    warnings: list[str] = []
    handwriting: str | None = None

    if pdf_type == "digital":
        text = extract_digital_text(pdf_path)
        if not text.strip():
            raise PdfError(f"Digital PDF produced empty text: {pdf_path.name}")
        engine = "pdfplumber"
    else:
        scanned = extract_scanned_text(pdf_path)
        text = scanned["printed_ocr"]
        handwriting = scanned["handwriting"]
        warnings.extend(scanned["warnings"])
        engine = "tesseract"

    return {
        "pdf_path":    str(pdf_path),
        "pdf_type":    pdf_type,
        "text":        text,
        "handwriting": handwriting,
        "page_count":  diag["total_pages"],
        "extraction_meta": {
            "engine":    engine,
            "warnings":  warnings,
            "detection": diag,
            "sha256":    sha256_of_file(pdf_path),
            "ocr_lang":  OCR_LANG if pdf_type == "scanned" else None,
            "dpi":       OCR_DPI if pdf_type == "scanned" else None,
        },
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> int:
    try:
        cfg = load_config()
    except ConfigError as e:
        print(f"[FAILED] {e}")
        return 1

    pdf_name = cfg["File"]
    pdf_path = INPUT_DIR / pdf_name

    print(f"Config : {CONFIG_JSON}")
    print(f"File   : {pdf_path}")
    print("-" * 60)

    # --- Step 1: detect ---
    try:
        pdf_type, diag = detect_pdf_type(pdf_path)
    except PdfError as e:
        print(f'{pdf_name} is "failed": {e}')
        return 1
    except Exception as e:
        print(f'{pdf_name} is "failed": unexpected error: {e}')
        return 1

    print(
        f'{pdf_name} is "{pdf_type}" '
        f'(avg {diag["avg_chars_per_page"]} chars/page over '
        f'{diag["pages_sampled"]}/{diag["total_pages"]} pages, '
        f'threshold {diag["threshold"]}; '
        f'per-page={diag["per_page_chars"]})'
    )

    # --- Step 2: extract ---
    try:
        result = extract_pdf(pdf_path)
    except PdfError as e:
        print(f'{pdf_name} is "failed": {e}')
        return 1
    except Exception as e:
        print(f'{pdf_name} is "failed": unexpected error: {e}')
        return 1

    # --- Step 3: success summary ---
    text = result["text"]
    engine = result["extraction_meta"]["engine"]
    print(
        f'Extracted {len(text)} chars from '
        f'{result["page_count"]} page(s) via {engine}. '
        f'sha256={result["extraction_meta"]["sha256"][:12]}...'
    )

    # --- Step 4: write text and metadata to output/ ---
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    stem = pdf_path.stem
    text_path = OUTPUT_DIR / f"{stem}_{engine}.txt"
    text_path.write_text(text, encoding="utf-8")
    print(f"Text written to {text_path}")

    meta_path = OUTPUT_DIR / f"{stem}_meta.json"
    meta_path.write_text(
        json.dumps(result["extraction_meta"], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Meta written to {meta_path}")

    if result["extraction_meta"]["warnings"]:
        for w in result["extraction_meta"]["warnings"]:
            print(f'  [warn] {w}')

    return 0


if __name__ == "__main__":
    sys.exit(main())