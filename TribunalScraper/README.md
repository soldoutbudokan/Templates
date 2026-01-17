# Tribunal Scraper

A Python scraper for downloading PDF documents from Canadian Competition Tribunal case pages.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python3 ct_scraper.py <case_url>
```

### Example

```bash
python3 ct_scraper.py "https://decisions.ct-tc.gc.ca/ct-tc/cd/en/item/521373/index.do"
```

This will create a folder `output/{Case Name} Documents/` containing all PDFs from the case.

### Options

- `-o, --output` - Parent directory for case folder (default: `output`)
- `-d, --depth` - How many levels deep to follow links (default: 2)

## Output Structure

```
output/
└── Wonderland - CT-2025-001 Documents/
    ├── Case_Details_CT-2025-001_2025-10-22_EN.pdf
    ├── 1_-_2025-05-02_-_Notice_of_Application.pdf
    ├── 6_-_CT-2025-001_-_Order_granting_extension.pdf
    └── ...
```
