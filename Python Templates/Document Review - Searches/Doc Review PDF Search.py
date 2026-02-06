# %%
################################################################################
# Title: Document Review Key Searches
# Author: Tirth Bhatt
#
# Project: [Project]
################################################################################
# Interactive PDF Keyword Search
# Run in terminal: python pdf_search_interactive.py

import os
from pathlib import Path
import pdfplumber
import re
import csv

# %%
# CONFIGURATION - Edit this path

INPUT_FOLDER = r"[PATH]"
CONTEXT_CHARS = 150

# %%
def load_pdfs(folder_path):
    """
    Extract text from all PDFs upfront. Returns dict of {filename: {page_num: text}}
    """
    folder = Path(folder_path)
    pdf_files = list(folder.rglob("*.pdf"))
    
    print(f"Loading {len(pdf_files)} PDFs...")
    
    pdf_texts = {}
    
    for i, pdf_path in enumerate(pdf_files, start=1):
        filename = pdf_path.name
        print(f"  [{i}/{len(pdf_files)}] {filename}", end="\r")
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                pages = {}
                for page_num, page in enumerate(pdf.pages, start=1):
                    text = page.extract_text()
                    if text:
                        pages[page_num] = re.sub(r'\s+', ' ', text)
                pdf_texts[filename] = pages
        except Exception as e:
            print(f"\n  Error loading {filename}: {e}")
    
    print(f"\nLoaded {len(pdf_texts)} PDFs successfully.")
    return pdf_texts


# %%
def search_keyword(pdf_texts, keyword, context_chars=150):
    """
    Search all loaded PDFs for a keyword.
    """
    results = []
    pattern = re.compile(re.escape(keyword), re.IGNORECASE)
    
    for filename, pages in pdf_texts.items():
        for page_num, text in pages.items():
            for match in pattern.finditer(text):
                start = max(0, match.start() - context_chars)
                end = min(len(text), match.end() + context_chars)
                
                context = text[start:end]
                if start > 0:
                    context = "..." + context
                if end < len(text):
                    context = context + "..."
                
                results.append({
                    'file': filename,
                    'page': page_num,
                    'context': context.strip()
                })
    
    return results


# %%
def display_results(results, keyword):
    """
    Print results in a readable format.
    """
    if not results:
        print(f"\nNo matches for '{keyword}'")
        return
    
    print(f"\n{'='*60}")
    print(f"Found {len(results)} matches for '{keyword}'")
    print('='*60)
    
    for r in results:
        print(f"\n[{r['file']}] Page {r['page']}")
        print(f"  {r['context']}")
    
    # Summary
    files = set(r['file'] for r in results)
    print(f"\n--- Summary: {len(results)} matches across {len(files)} files ---")


# %%
def export_results(results, keyword):
    """
    Save results to CSV.
    """
    filename = f"search_{keyword.replace(' ', '_')}.csv"
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['file', 'page', 'context'])
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved to: {filename}")


# %%
def main():
    print("="*60)
    print("PDF KEYWORD SEARCH")
    print("="*60)
    
    # Load PDFs once
    pdf_texts = load_pdfs(INPUT_FOLDER)
    
    if not pdf_texts:
        print("No PDFs found. Check your INPUT_FOLDER path.")
        return
    
    last_results = []
    last_keyword = ""
    
    print("\nCommands:")
    print("  Type a keyword to search")
    print("  'export' - save last results to CSV")
    print("  'quit' or 'exit' - close program")
    print()
    
    while True:
        try:
            query = input("Search> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye.")
            break
        
        if not query:
            continue
        
        if query.lower() in ('quit', 'exit', 'q'):
            print("Goodbye.")
            break
        
        if query.lower() == 'export':
            if last_results:
                export_results(last_results, last_keyword)
            else:
                print("No results to export. Run a search first.")
            continue
        
        # Run search
        last_keyword = query
        last_results = search_keyword(pdf_texts, query, CONTEXT_CHARS)
        display_results(last_results, query)


if __name__ == "__main__":
    main()
