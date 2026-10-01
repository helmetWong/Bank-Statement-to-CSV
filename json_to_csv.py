"""
json_to_csv.py — Module 3: JSON to CSV writer.
Version 4.0 — 2026-10-01

Changes from v3:
    - Added RESERVE_COLUMNS injected immediately after the
      "balance" column and before "remark". Two blank columns
      for manual use (category, tax code, audit note, etc.).
    - Added inject_reserve_columns().
    - Applied injection after sanitize_csv_content() so numeric
      cleaning sees the original column layout.

Purpose:
    Read the JSON envelope files produced by Module 2 and write
    the embedded CSV content to output/csv/.

    No API calls. Pure local file operation. Re-runnable.

Input:
    output/json/<pdf_stem>_stmt<N>.json

Output:
    output/csv/<filename>.csv
    Files from different statements are merged by filename.
"""

from __future__ import annotations

import csv
import io
import json
import re
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR   = PROJECT_ROOT / "output"
JSON_DIR     = OUTPUT_DIR / "json"
CSV_DIR      = OUTPUT_DIR / "csv"


# ---------------------------------------------------------------------------
# Reserve columns (v4.0)
# ---------------------------------------------------------------------------

# Blank columns inserted immediately after RESERVE_INSERT_AFTER.
# Rename these when their purpose becomes clear
# (e.g. "tax_code", "client_ref").
RESERVE_COLUMNS = ["reserve_col_1", "reserve_col_2"]

# The anchor column. Reserve columns are placed right after it.
RESERVE_INSERT_AFTER = "balance"


# ---------------------------------------------------------------------------
# Numeric cleaning (fix for ISSUE-001)
# ---------------------------------------------------------------------------

_NUMBER_WITH_SEPARATOR = re.compile(r"^\d{1,3}(,\d{3})+(\.\d+)?$")
_FULLWIDTH_COMMA = "\uff0c"


def _clean_number(s: str) -> str:
    """Strip thousands separators from a numeric-looking string."""
    s = s.strip()
    if not s:
        return s
    normalized = s.replace(_FULLWIDTH_COMMA, ",")
    if _NUMBER_WITH_SEPARATOR.match(normalized):
        return normalized.replace(",", "")
    return s


def sanitize_csv_content(content: str) -> str:
    """
    Parse a raw CSV string, clean numeric fields, and re-emit
    as well-formed CSV with proper quoting.

    Fixes ISSUE-001: unquoted "960,000.00" would otherwise split
    into two CSV columns.
    """
    reader = csv.reader(io.StringIO(content))
    rows = list(reader)
    if not rows:
        return content

    header = rows[0]
    expected_cols = len(header)

    out = io.StringIO()
    writer = csv.writer(out, quoting=csv.QUOTE_MINIMAL,
                        lineterminator="\n")

    for i, row in enumerate(rows):
        if i == 0:
            writer.writerow(row)
            continue
        cleaned = [_clean_number(field) for field in row]
        if len(cleaned) != expected_cols:
            print(f"    [warn] row {i}: {len(cleaned)} columns, "
                  f"expected {expected_cols}. "
                  f"First fields: {row[:3]}")
        writer.writerow(cleaned)

    return out.getvalue()


# ---------------------------------------------------------------------------
# Reserve-column injection (v4.0)
# ---------------------------------------------------------------------------

def inject_reserve_columns(content: str) -> str:
    """
    Insert blank reserve columns immediately after the anchor
    column (RESERVE_INSERT_AFTER, default "balance").

    - If the reserve columns are already present, the content
      is returned unchanged.
    - If the anchor column is not found, the content is returned
      unchanged with a warning.
    - Each data row gets empty strings in the reserve positions.

    Applied AFTER sanitize_csv_content, so numeric cleaning
    sees the original column layout.
    """
    reader = csv.reader(io.StringIO(content))
    rows = list(reader)
    if not rows:
        return content

    header = rows[0]

    # Already injected? Nothing to do.
    if all(col in header for col in RESERVE_COLUMNS):
        return content

    if RESERVE_INSERT_AFTER not in header:
        print(f"    [warn] anchor column '{RESERVE_INSERT_AFTER}' "
              f"not found; reserve columns not injected. "
              f"Header: {header}")
        return content

    anchor_idx = header.index(RESERVE_INSERT_AFTER) + 1
    n_reserve = len(RESERVE_COLUMNS)

    out = io.StringIO()
    writer = csv.writer(out, quoting=csv.QUOTE_MINIMAL,
                        lineterminator="\n")

    # Write new header.
    new_header = (
        header[:anchor_idx]
        + RESERVE_COLUMNS
        + header[anchor_idx:]
    )
    writer.writerow(new_header)

    # Write data rows with blank reserve fields.
    for row in rows[1:]:
        new_row = (
            row[:anchor_idx]
            + [""] * n_reserve
            + row[anchor_idx:]
        )
        writer.writerow(new_row)

    return out.getvalue()


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class Module3Error(Exception):
    """Base error for Module 3."""


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def _stmt_sort_key(path: Path) -> tuple[str, int]:
    """Sort by (pdf_stem, statement number)."""
    stem = path.stem
    if "_stmt" in stem:
        base, num = stem.rsplit("_stmt", 1)
        try:
            return base, int(num)
        except ValueError:
            return stem, 0
    return stem, 0


def find_json_files(json_dir: Path = JSON_DIR) -> list[Path]:
    """Return all <pdf_stem>_stmt<N>.json files, sorted by N."""
    if not json_dir.exists():
        raise Module3Error(f"JSON folder not found: {json_dir}")
    files = sorted(
        json_dir.glob("*_stmt*.json"),
        key=_stmt_sort_key,
    )
    if not files:
        raise Module3Error(
            f"No *_stmt*.json files found in {json_dir}"
        )
    return files


def check_missing_statements(json_files: list[Path]) -> list[int]:
    """Return statement numbers missing from the observed range."""
    if len(json_files) < 2:
        return []
    bases = {_stmt_sort_key(p)[0] for p in json_files}
    if len(bases) > 1:
        return []
    numbers = sorted(_stmt_sort_key(p)[1] for p in json_files)
    if not numbers:
        return []
    expected = set(range(numbers[0], numbers[-1] + 1))
    present  = set(numbers)
    return sorted(expected - present)


# ---------------------------------------------------------------------------
# Load JSON files
# ---------------------------------------------------------------------------

def load_json(path: Path) -> dict[str, Any]:
    """Load and validate one JSON envelope."""
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise Module3Error(f"Invalid JSON in {path.name}: {e}")
    if "files" not in data or not isinstance(data["files"], list):
        raise Module3Error(f"{path.name} missing 'files' array")
    return data


# ---------------------------------------------------------------------------
# Merge and write CSVs
# ---------------------------------------------------------------------------

def merge_csv_content(
    existing: str | None,
    new_content: str,
) -> str:
    """Merge a new CSV block into an existing one.
    Header from the first block wins; subsequent headers dropped.
    Warns on header mismatch.
    """
    if existing is None:
        return new_content.rstrip("\n") + "\n"

    old_lines = existing.rstrip("\n").split("\n")
    new_lines = new_content.rstrip("\n").split("\n")
    if not new_lines:
        return existing

    old_header = old_lines[0] if old_lines else ""
    new_header = new_lines[0]
    if old_header != new_header:
        print(f"    [warn] header mismatch while merging:")
        print(f"           existing: {old_header[:80]}")
        print(f"           new     : {new_header[:80]}")

    body = new_lines[1:] if len(new_lines) > 1 else []
    merged = old_lines + body
    return "\n".join(merged) + "\n"


def process_json_files(json_files: list[Path]) -> dict[str, str]:
    """
    Read all JSON files, sanitize each CSV block, inject
    reserve columns, merge by filename.
    """
    merged: "OrderedDict[str, str]" = OrderedDict()
    warnings: list[str] = []

    for jf in json_files:
        data = load_json(jf)
        print(f"  {jf.name}: {len(data['files'])} file(s)")

        for entry in data["files"]:
            filename = entry.get("filename")
            content  = entry.get("content")
            if not filename or content is None:
                warnings.append(
                    f"{jf.name}: malformed entry {entry!r}"
                )
                continue

            safe_name = Path(filename).name

            # 1) Clean numerics (sees original column layout).
            clean_content = sanitize_csv_content(content)

            # 2) Inject reserve columns right after "balance".
            clean_content = inject_reserve_columns(clean_content)

            if safe_name in merged:
                merged[safe_name] = merge_csv_content(
                    merged[safe_name], clean_content
                )
            else:
                merged[safe_name] = (
                    clean_content.rstrip("\n") + "\n"
                )

        for w in data.get("warnings", []):
            warnings.append(f"{jf.name}: {w}")

    if warnings:
        print()
        print(f"Warnings from DeepSeek ({len(warnings)}):")
        for w in warnings:
            print(f"  - {w}")

    return merged


def write_csv_files(
    merged: dict[str, str],
    csv_dir: Path = CSV_DIR,
) -> list[Path]:
    """Write each merged CSV to csv_dir. Returns list of paths."""
    csv_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for filename, content in merged.items():
        out_path = csv_dir / filename
        out_path.write_text(content, encoding="utf-8-sig")
        written.append(out_path)
        line_count = content.count("\n")
        print(f"  wrote {out_path.name}  "
              f"({line_count} lines, {len(content)} chars)")
    return written


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print(f"JSON dir: {JSON_DIR}")
    print(f"CSV  dir: {CSV_DIR}")
    print(f"Reserve : {RESERVE_COLUMNS} "
          f"after '{RESERVE_INSERT_AFTER}'")
    print("-" * 60)

    try:
        json_files = find_json_files()
    except Module3Error as e:
        print(f"[FAILED] {e}")
        return 1

    print(f"Found {len(json_files)} JSON file(s):")
    for jf in json_files:
        print(f"  {jf.name}")
    print()

    missing = check_missing_statements(json_files)
    if missing:
        print("!" * 60)
        print(f"[WARN] missing statement number(s): {missing}")
        print(f"       Present span: stmt"
              f"{_stmt_sort_key(json_files[0])[1]}.."
              f"stmt{_stmt_sort_key(json_files[-1])[1]}")
        print(f"       The merge will continue.")
        print("!" * 60)
        print()

    print("Merging CSV content by filename...")
    try:
        merged = process_json_files(json_files)
    except Module3Error as e:
        print(f"[FAILED] {e}")
        return 1

    if not merged:
        print("[FAILED] No CSV content found in any JSON file.")
        return 1

    print()
    print(f"Writing {len(merged)} CSV file(s):")
    written = write_csv_files(merged)

    print()
    print("=" * 60)
    print(f"Done. {len(written)} CSV file(s) in {CSV_DIR}")
    if missing:
        print(f"NOTE: {len(missing)} statement(s) missing: "
              f"{missing}")
    return 0


if __name__ == "__main__":
    sys.exit(main())