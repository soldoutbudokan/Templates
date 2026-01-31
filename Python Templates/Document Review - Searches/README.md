# Document Review - Searches

**This is for sniff tests only and should not be relied on.**

Quick keyword search tools for document review. Useful for initial scans to find potentially relevant documents, but results should be verified manually.

## Scripts

- `Doc Review PDF Search.py` - Search PDFs for keywords
- `Doc Review PPTX Search.py` - Search PowerPoint files for keywords

## Usage

1. Edit `INPUT_FOLDER` in the script to point to your documents
2. Run the script
3. Enter keywords to search
4. Type `export` to save results to CSV

## Important Notes

### PDF Search Requires OCR for Scanned Documents

The PDF search uses `pdfplumber` which extracts embedded text only. It does **not** perform OCR.

- Native/digital PDFs: Works out of the box
- Scanned PDFs: Will return no results unless OCR has already been run on them

If your PDFs are scanned images, you need to OCR them first using tools like Adobe Acrobat, ABBYY, or `pytesseract` before this search will find anything.

## Dependencies

```
pip install pdfplumber python-pptx
```
