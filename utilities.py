"""
utilities.py — shared helpers for the bank statement pipeline.

Module 1 scope (this version):
    csv_to_json(csv_path, json_path) -> dict

Design notes:
    - Duplicate keys in the CSV are collected into a list.
    - Single-occurrence keys remain strings.
    - All values stay as strings. Numeric coercion, if needed,
      happens in the consuming module, not here.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class UtilitiesError(Exception):
    """Base error for utilities module."""


class ConfigFileError(UtilitiesError):
    """Raised when a config file is missing, empty, or malformed."""


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

def csv_to_json(
    csv_path: str | Path,
    json_path: str | Path | None = None,
) -> dict[str, Any]:
    """
    Read a two-column key/value CSV and return a dict.

    Rules:
        - Header row must be exactly: key,value  (case-insensitive).
        - Blank lines are skipped.
        - Blank keys raise ConfigFileError.
        - Duplicate keys are collected into a list, in order.
        - Values are stripped of surrounding whitespace.
        - All values remain strings (no int/float coercion).

    If json_path is given, the result is also written to disk
    as UTF-8 JSON with indent=2.

    Returns:
        dict mapping each key to either a str or a list[str].
    """
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise ConfigFileError(f"CSV file not found: {csv_path}")
    if not csv_path.is_file():
        raise ConfigFileError(f"Not a file: {csv_path}")

    # utf-8-sig strips the BOM that Excel adds on Windows/HK.
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows:
        raise ConfigFileError(f"CSV file is empty: {csv_path}")

    # --- Validate header ---
    header = [cell.strip().lower() for cell in rows[0]]
    if header != ["key", "value"]:
        raise ConfigFileError(
            f"Header must be 'key,value', got: {rows[0]}"
        )

    # --- Build the dict ---
    result: dict[str, Any] = {}

    for line_no, row in enumerate(rows[1:], start=2):
        # Skip fully blank lines
        if not row or all(cell.strip() == "" for cell in row):
            continue

        if len(row) != 2:
            raise ConfigFileError(
                f"Line {line_no}: expected 2 columns, got {len(row)}: {row}"
            )

        key = row[0].strip()
        value = row[1].strip()

        if key == "":
            raise ConfigFileError(f"Line {line_no}: empty key")

        if key in result:
            existing = result[key]
            if isinstance(existing, list):
                existing.append(value)
            else:
                result[key] = [existing, value]
        else:
            result[key] = value

    # --- Optionally write JSON ---
    if json_path is not None:
        json_path = Path(json_path)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
            f.write("\n")  # trailing newline, POSIX-friendly

    return result, 