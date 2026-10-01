"""
create_config_json.py — Configuration loader.
Version 2.0 — 2026-10-01

Changes from utilities.py v1:
    - Renamed from utilities.py.
    - Added a main() entry point so this file can be run
      directly:  python create_config_json.py
    - The csv_to_json() function is unchanged; other modules
      that import it will still work.

Purpose:
    Read config/config.csv and write config/config.json.

    Duplicate keys in the CSV become lists, in order.
    Single-occurrence keys remain strings.
    All values stay as strings; numeric coercion happens in
    the consuming module.

Usage:
    Command line:
        python create_config_json.py
    From another module:
        from create_config_json import csv_to_json
        cfg = csv_to_json("config/config.csv", "config/config.json")
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_CSV   = PROJECT_ROOT / "config" / "config.csv"
CONFIG_JSON  = PROJECT_ROOT / "config" / "config.json"


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class UtilitiesError(Exception):
    """Base error for this module."""


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
                f"Line {line_no}: expected 2 columns, "
                f"got {len(row)}: {row}"
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
            f.write("\n")  # trailing newline

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    """
    Read config/config.csv and write config/config.json.

    Returns 0 on success, 1 on failure.
    """
    print(f"Input : {CONFIG_CSV}")
    print(f"Output: {CONFIG_JSON}")
    print("-" * 60)

    try:
        config = csv_to_json(CONFIG_CSV, CONFIG_JSON)
    except ConfigFileError as e:
        print(f"[FAILED] {e}")
        return 1

    print(f"Wrote {CONFIG_JSON}")
    print()
    print("Parsed config:")
    for k, v in config.items():
        print(f"  {k!r}: {v!r}")

    return 0


if __name__ == "__main__":
    sys.exit(main())