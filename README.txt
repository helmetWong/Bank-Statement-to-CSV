# Bank Statement PDF to CSV Parser for CPAs

A Python-based tool for Hong Kong CPA practices to convert bank
statement PDFs into accounting-ready CSV files, using the
DeepSeek API for parsing.

## Features

- Handles both digital PDFs (text layer) and scanned PDFs (OCR)
- Parallel OCR processing for multi-page statements
- Natural-language style files for per-bank parsing rules
- Splits multi-statement PDFs into per-statement API calls
- Numeric formatting enforcement (no thousands separators)
- Five-layer arithmetic verification of transaction rows
- Bilingual English / Chinese column splitting
- Reserve columns for manual coding and audit notes
- Local file operation for CSV generation (no API re-calls)
- Full extraction metadata saved alongside output

## Pipeline

Four programs, run in order:

    create_config_json.py   config.csv -> config.json
    read_pdf.py             PDF -> text (digital or OCR)
    txt_to_json_API.py      text -> DeepSeek -> JSON
    json_to_csv.py          JSON -> final CSV

## Disclaimer

**Important:** This software is provided for informational
purposes only. Users are solely responsible for:

- Verifying all parsed data against the original bank statements
- Ensuring the accuracy of every number before use in accounting
- Proper handling and security of client data
- Any decisions made based on the software output

The author provides no warranties regarding accuracy or
completeness of results. See [LICENSE](LICENSE) for complete
terms.

## Data Privacy

Only the OCR text of the bank statement is transmitted to the
DeepSeek API for parsing. The original PDF, the API key, and
all output files remain on your local machine. You are solely
responsible for securing client data and complying with all
applicable data protection laws.

## Installation

See SETUP.txt for detailed installation instructions.

    git clone https://github.com/helmetWong/bank-statement-parser.git
    cd bank-statement-parser
    pip install -r requirements.txt

## License

See LICENSE for complete terms.

Copyright (c) 2025 Wong Yun Wah Micheal