"""
txt_to_json_API.py — Module 2: DeepSeek API connector.
Version 3.0 — 2026-10-01

Changes from v2:
    - OCR text is split into per-statement chunks before calling
      the API. This prevents output truncation when a single PDF
      contains many statements whose combined JSON exceeds the
      model's max_tokens budget.
    - Each statement's response is saved as a separate JSON file.
    - Diagnostics print reasoning_content length (if the model
      returns it) and reasoning_tokens (if the API reports it).
    - finish_reason == "length" now returns partial content with
      a warning rather than silently discarding it.

Pipeline position:
    read_pdf.py        -> output/<stem>_tesseract.txt
    txt_to_json_API.py -> output/json/<stem>_stmt{N}.json
    json_to_csv.py     -> output/csv/*.csv   (Module 3, future)
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path
from typing import Any

try:
    from openai import OpenAI, APIError, RateLimitError, APIConnectionError
except ImportError:
    print("[FATAL] openai is not installed. Run: pip install openai")
    raise SystemExit(1)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_JSON  = PROJECT_ROOT / "config" / "config.json"
STYLES_DIR   = PROJECT_ROOT / "config" / "styles"
KEY_DIR      = PROJECT_ROOT / "key"
OUTPUT_DIR   = PROJECT_ROOT / "output"
JSON_DIR     = OUTPUT_DIR / "json"


# ---------------------------------------------------------------------------
# DeepSeek client configuration
# ---------------------------------------------------------------------------

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL    = "deepseek-flash"

# max_tokens caps the response size. DeepSeek chat completion
# accepts up to 8192. 8000 leaves headroom.
MAX_TOKENS  = 8000
TEMPERATURE = 0

# Retry policy for transient network / rate-limit failures.
MAX_RETRIES        = 5
RETRY_BACKOFF_BASE = 2


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class Module2Error(Exception):
    """Base error for Module 2."""


class ConfigError(Module2Error):
    """Config, style, key file, or OCR text missing or malformed."""


class APIErrorWrapper(Module2Error):
    """DeepSeek API call failed after retries."""


# ---------------------------------------------------------------------------
# Loading helpers
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
    return cfg


def load_api_key(cfg: dict[str, Any], key_dir: Path = KEY_DIR) -> str:
    """
    Read the API key from key/<Key_file>.

    The key filename is read from config.json's 'Key_file' key.
    The file should contain a single line: the raw key.
    Whitespace and newlines are stripped.
    """
    key_name = cfg.get("Key_file")
    if not key_name:
        raise ConfigError(
            "config.json missing 'Key_file' key. "
            "Add a line like:  Key_file,deepseek_key.txt"
        )

    key_path = key_dir / key_name
    if not key_path.exists():
        raise ConfigError(f"Key file not found: {key_path}")
    if not key_path.is_file():
        raise ConfigError(f"Not a file: {key_path}")

    key = key_path.read_text(encoding="utf-8").strip()

    if not key:
        raise ConfigError(f"Key file is empty: {key_path}")
    if not key.startswith("sk-"):
        print(f"  [warn] key does not start with 'sk-', "
              f"this may be wrong")

    return key


def load_style_text(style_name: str) -> str:
    """Load config/styles/<style_name>.txt."""
    style_path = STYLES_DIR / f"{style_name}.txt"
    if not style_path.exists():
        raise ConfigError(f"Style file not found: {style_path}")
    return style_path.read_text(encoding="utf-8")


def locate_ocr_text(cfg: dict[str, Any]) -> Path:
    """
    Find the OCR text file produced by Module 1.
    Naming: output/<pdf_stem>_<engine>.txt
    """
    pdf_name = cfg.get("File")
    if not pdf_name:
        raise ConfigError("config.json missing 'File' key")

    stem = Path(pdf_name).stem
    candidates = [
        OUTPUT_DIR / f"{stem}_tesseract.txt",
        OUTPUT_DIR / f"{stem}_pdfplumber.txt",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise ConfigError(
        f"No OCR text found. Looked for: "
        f"{[str(c) for c in candidates]}"
    )


# ---------------------------------------------------------------------------
# Statement splitting
# ---------------------------------------------------------------------------

def split_by_statement(ocr_text: str, marker: str) -> list[str]:
    """
    Split OCR text into per-statement chunks.

    Each chunk starts at an occurrence of the marker and ends
    just before the next occurrence. Text before the first
    marker is discarded (it is usually noise or the tail of a
    previous statement's page).

    Returns:
        list of chunk strings, each beginning with the marker.
        Empty list if the marker is not found.
    """
    parts = ocr_text.split(marker)
    if len(parts) < 2:
        return []
    chunks = []
    for part in parts[1:]:
        chunk = (marker + part).strip()
        if chunk:
            chunks.append(chunk)
    return chunks


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def build_messages(
    style_text: str,
    statement_text: str,
    cfg: dict[str, Any],
) -> list[dict[str, str]]:
    """
    Build the messages array for DeepSeek for one statement.

    The system prompt fixes the output format AND the numeric
    formatting rules. Numeric rules are in the system prompt
    (not just the style file) because the model follows system
    instructions more reliably, and because the style file is
    long enough that embedded rules can be missed.

    The user prompt embeds the style instruction, task context,
    and one statement's OCR text.
    """
    system_prompt = (
        "You are a bank statement parsing assistant for a Hong Kong CPA. "
        "You follow the provided style instruction file precisely.\n"
        "\n"
        "=== OUTPUT FORMAT — MANDATORY ===\n"
        "Return a single JSON object with this exact structure:\n"
        "{\n"
        '  "files": [\n'
        "    {\n"
        '      "filename": "<name>.csv",\n'
        '      "content":  "<full CSV content as a string>"\n'
        "    }\n"
        "  ],\n"
        '  "warnings": ["<any issue you encountered>"]\n'
        "}\n"
        "\n"
        "Rules for the outer JSON:\n"
        "- Do NOT wrap the JSON in markdown fences.\n"
        "- Do NOT add commentary outside the JSON.\n"
        "- Each CSV file's content is a single string with newline "
        "characters (\\n) separating rows.\n"
        "- The first line of each CSV is the header row.\n"
        "\n"
        "=== NUMERIC FORMATTING — MANDATORY ===\n"
        "Every number in every CSV field MUST be written as a plain\n"
        "decimal number with:\n"
        "  - a period (.) as the decimal mark\n"
        "  - NO thousands separator of any kind\n"
        "  - NO currency symbol\n"
        "  - NO spaces inside the number\n"
        "\n"
        "CORRECT examples:\n"
        "  960000.00\n"
        "  1895743.42\n"
        "  988638.00\n"
        "  0.00\n"
        "  4250.00\n"
        "\n"
        "WRONG examples (never produce these):\n"
        "  960,000.00\n"
        "  1,895,743.42\n"
        "  988,638\n"
        "  HKD 960,000.00\n"
        "  960 000.00\n"
        "\n"
        "This rule applies to EVERY numeric value in EVERY column,\n"
        "including:\n"
        "  - deposit\n"
        "  - withdrawal\n"
        "  - balance\n"
        "  - numbers inside the verify_row column\n"
        "  - numbers inside the remark column\n"
        "  - numbers in the total_verification CSV\n"
        "  - numbers in the cross_month_continuity CSV\n"
        "  - numbers in the net_movement CSV\n"
        "\n"
        "For verify_row, write:\n"
        "  checked: 939993.42 + 960000.00 - 4250.00 = 1895743.42\n"
        "NOT:\n"
        "  checked: 939,993.42 + 960,000.00 - 4,250.00 = 1,895,743.42\n"
        "\n"
        "If the source statement shows a thousands separator, strip\n"
        "it before writing. A comma must NEVER appear inside a\n"
        "numeric field under any circumstance.\n"
        "\n"
        "=== HEADER ROW ===\n"
        "Each CSV's first line MUST be exactly the header specified\n"
        "in the style file, in the exact column order specified.\n"
        "Do not add, remove, or reorder columns."
    )

    task_context = {
        "Bank":       cfg.get("Bank"),
        "Style":      cfg.get("Style"),
        "File":       cfg.get("File"),
        "Language":   cfg.get("Language"),
        "Page_start": cfg.get("Page_start"),
        "Page_end":   cfg.get("Page_end"),
        "Currency":   cfg.get("Currency"),
    }

    user_prompt = (
        "=== TASK CONTEXT (from config.json) ===\n"
        f"{json.dumps(task_context, indent=2, ensure_ascii=False)}\n\n"
        "=== STYLE INSTRUCTION FILE ===\n"
        f"{style_text}\n\n"
        "=== OCR TEXT FOR ONE STATEMENT ===\n"
        f"{statement_text}\n\n"
        "=== END OF INPUT ===\n\n"
        "Now produce the JSON object described in the system prompt, "
        "following the style instruction file. Process only the "
        "statement above.\n\n"
        "REMINDER before you write:\n"
        "  - Numeric fields: NO thousands separators. Write "
        "960000.00 not 960,000.00.\n"
        "  - verify_row: NO thousands separators. Write "
        "939993.42 not 939,993.42.\n"
        "  - CSV header: exactly as specified in the style file."
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]

DEFAULT_STATEMENT_MARKER = "Statement"

# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------

def call_deepseek(
    client: OpenAI,
    messages: list[dict[str, str]],
    max_retries: int = MAX_RETRIES,
) -> str:
    """
    Call DeepSeek chat completion with retry on transient errors
    and empty-content responses.

    Retries on:
        - RateLimitError            (429)
        - APIConnectionError        (network failure)
        - APIError with status >= 500
        - Empty or whitespace-only content

    Does NOT retry on:
        - APIError with status 4xx other than 429 (deterministic)
        - finish_reason == "length" with partial content

    Prints diagnostics on every response.

    Raises:
        APIErrorWrapper after exhausting max_retries.
    """
    last_error: Exception | str | None = None

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                response_format={"type": "json_object"},
                extra_body={"thinking": {"type": "disabled"}},
            )

            choice = response.choices[0]
            content = choice.message.content or ""

            # --- Diagnostics ---
            print(f"  [debug] finish_reason={choice.finish_reason}")
            print(f"  [debug] content length={len(content)}")

            # reasoning_content is returned by the model when
            # thinking mode is active; empty string means off.
            reasoning = getattr(choice.message, "reasoning_content", None) or ""
            print(f"  [debug] reasoning_content length={len(reasoning)}")

            usage = getattr(response, "usage", None)
            if usage is not None:
                print(
                    f"  [debug] usage: "
                    f"prompt={getattr(usage, 'prompt_tokens', '?')} "
                    f"completion={getattr(usage, 'completion_tokens', '?')} "
                    f"total={getattr(usage, 'total_tokens', '?')}"
                )
                # reasoning_tokens may not be exposed by all
                # API versions; fall back to '?'.
                reasoning_tokens = getattr(usage, "reasoning_tokens", None)
                if reasoning_tokens is None:
                    extra = getattr(usage, "model_extra", None) or {}
                    reasoning_tokens = extra.get("reasoning_tokens", "?")
                print(f"  [debug] reasoning_tokens={reasoning_tokens}")
            else:
                print("  [debug] usage: not returned by API")

            # --- Empty content: retry ---
            if not content.strip():
                last_error = "Empty content from DeepSeek"
                wait = RETRY_BACKOFF_BASE ** attempt
                print(f"  [retry {attempt}/{max_retries}] "
                      f"empty content, waiting {wait}s...")
                time.sleep(wait)
                continue

            # --- Truncated output: return partial with warning ---
            if choice.finish_reason == "length":
                print(f"  [warn] response truncated at max_tokens="
                      f"{MAX_TOKENS}. Returning partial content.")
                return content

            # --- Success ---
            return content

        except RateLimitError as e:
            last_error = e
            wait = RETRY_BACKOFF_BASE ** attempt
            print(f"  [retry {attempt}/{max_retries}] "
                  f"rate limited, waiting {wait}s...")
            time.sleep(wait)

        except APIConnectionError as e:
            last_error = e
            wait = RETRY_BACKOFF_BASE ** attempt
            print(f"  [retry {attempt}/{max_retries}] "
                  f"connection error, waiting {wait}s...")
            time.sleep(wait)

        except APIError as e:
            status = getattr(e, "status_code", None)
            if status is not None and status >= 500:
                last_error = e
                wait = RETRY_BACKOFF_BASE ** attempt
                print(f"  [retry {attempt}/{max_retries}] "
                      f"server error {status}, waiting {wait}s...")
                time.sleep(wait)
            else:
                raise APIErrorWrapper(f"DeepSeek API error: {e}")

    raise APIErrorWrapper(
        f"DeepSeek call failed after {max_retries} attempts. "
        f"Last error: {last_error}"
    )


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save_raw_response(raw: str, pdf_name: str, stmt_no: int) -> Path:
    """
    Save one statement's raw JSON response to output/json/.

    The string is validated as JSON before writing, so a
    malformed response fails loudly.

    Filename: <pdf_stem>_stmt<N>.json
    """
    JSON_DIR.mkdir(parents=True, exist_ok=True)

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        raise Module2Error(
            f"DeepSeek returned invalid JSON for statement {stmt_no}: {e}\n"
            f"First 500 chars:\n{raw[:500]}"
        )

    stem = Path(pdf_name).stem
    out_path = JSON_DIR / f"{stem}_stmt{stmt_no}.json"

    out_path.write_text(
        json.dumps(parsed, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    # --- Load config ---
    try:
        cfg = load_config()
    except ConfigError as e:
        print(f"[FAILED] {e}")
        return 1

    pdf_name   = cfg.get("File")
    style_name = cfg.get("Style")
    if not pdf_name or not style_name:
        print("[FAILED] config.json missing 'File' or 'Style'")
        return 1

    # --- Statement marker (from config) ---
    marker = cfg.get("Statement_marker")
    if not marker:
        print("[FAILED] config.json missing 'Statement_marker'")
        print("         Add to config.csv:")
        print("         Statement_marker,HSBC Business Direct Statement")
        return 1

    # --- Load style ---
    try:
        style_text = load_style_text(style_name)
        print(f"Style   : {STYLES_DIR / (style_name + '.txt')} "
              f"({len(style_text)} chars)")
    except ConfigError as e:
        print(f"[FAILED] {e}")
        return 1

    # --- Locate OCR text ---
    try:
        ocr_path = locate_ocr_text(cfg)
        ocr_text = ocr_path.read_text(encoding="utf-8")
        print(f"OCR text: {ocr_path} ({len(ocr_text)} chars)")
    except ConfigError as e:
        print(f"[FAILED] {e}")
        return 1

    # --- Split by statement marker ---
    chunks = split_by_statement(ocr_text, marker)
    if not chunks:
        print(f"[FAILED] No statement marker {marker!r} "
              f"found in OCR text. Cannot split.")
        return 1
    print(f"Split   : {len(chunks)} statement(s) "
          f"(marker={marker!r})")
    for i, ch in enumerate(chunks, start=1):
        print(f"  statement {i}: {len(ch)} chars")

    # --- Load API key ---
    try:
        api_key = load_api_key(cfg)
        print(f"API key : loaded from {KEY_DIR} "
              f"({api_key[:7]}...{api_key[-4:]})")
    except ConfigError as e:
        print(f"[FAILED] {e}")
        return 1

    client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)

    # --- Process each statement ---
    print()
    saved: list[Path] = []
    failed: list[int] = []

    for idx, chunk in enumerate(chunks, start=1):
        print(f"--- Statement {idx}/{len(chunks)} ---")
        messages = build_messages(style_text, chunk, cfg)
        total_chars = sum(len(m["content"]) for m in messages)
        print(f"  Prompt: {total_chars} chars "
              f"(~{total_chars // 3} tokens est.)")

        t0 = time.perf_counter()
        try:
            raw = call_deepseek(client, messages)
        except APIErrorWrapper as e:
            print(f"  [FAILED] {e}")
            failed.append(idx)
            continue

        elapsed = time.perf_counter() - t0
        print(f"  Response: {len(raw)} chars in {elapsed:.1f}s")

        try:
            out_path = save_raw_response(raw, pdf_name, idx)
            saved.append(out_path)
            print(f"  Saved: {out_path}")
        except Module2Error as e:
            print(f"  [FAILED] {e}")
            failed.append(idx)

    # --- Summary ---
    print()
    print("=" * 60)
    print(f"Summary: {len(saved)}/{len(chunks)} statements saved")
    for p in saved:
        print(f"  OK  {p.name}")
    for i in failed:
        print(f"  FAILED  statement {i}")

    if failed:
        print()
        print("Re-run the program to retry failed statements "
              "(already-saved ones will be overwritten).")
        return 1

    print()
    print(f"Next step: run json_to_csv.py (Module 3) to convert "
          f"the JSON files in {JSON_DIR} to CSV.")
    return 0


if __name__ == "__main__":
    sys.exit(main())