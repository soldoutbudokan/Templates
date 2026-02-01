#!/usr/bin/env python3
"""
Orwell.ru Scraper - Collects George Orwell's public domain works for ML training.
Follows content links from index pages to get full text.
"""

import requests
from bs4 import BeautifulSoup
import time
import json
import re
from pathlib import Path
from typing import Optional, List, Dict

BASE_URL = "https://orwell.ru"
OUTPUT_DIR = Path(__file__).parent.parent / "corpus"
DELAY_SECONDS = 1.0  # Be respectful to the server

# All known English works from orwell.ru
WORKS = {
    "novels": [
        "Down_and_Out_in_Paris_and_London",
        "Burmese_Days",
        "A_Clergymans_Daughter",
        "Keep_the_Aspidistra_Flying",
        "The_Road_to_Wigan_Pier",
        "Homage_to_Catalonia",
        "Coming_up_for_Air",
        "Animal_Farm",
        "1984",
    ],
    "essays": [
        "wiw",
        "mine",
        "north",
        "whale",
        "boys",
        "lion",
        "Spanish_War",
        "nationalism",
        "politics",
        "prevention",
        "lear",
        "joys",
    ],
    "articles": [
        "hanging",
        "spike",
        "elephant",
        "bookshop",
        "novel",
        "beans",
        "niggers",
        "marrakech",
        "notes",
        "My_Country",
        "words",
        "totalitarianism",
        "frontiers",
        "rediscovery",
        "pacifism",
        "As_I_Please",
        "pamphlet",
        "criminals",
        "socialists",
        "poetry",
        "antisemitism",
        "funny",
        "ABomb",
        "revenge",
        "germany",
        "science",
        "cooking",
        "park",
        "spirit",
        "tea",
        "spots",
        "cigar",
        "nose",
        "Common_Toad",
        "reviewer",
        "Poor_Die",
        "decline",
        "European_Unity",
        "leviathan",
    ],
    "reviews": [
        "dickens",
        "reade",
        "fascism",
        "tolstoy",
        "wells",
        "McGill",
        "kipling",
        "yeats",
        "twain",
        "dali",
        "koestler",
        "smollett",
        "chase",
        "books",
        "nonsense",
        "zamyatin",
        "plum",
        "vicar",
        "swift",
        "burnham",
        "bangor",
        "gissing",
        "gandhi",
    ],
}


def fetch_page(url: str) -> Optional[str]:
    """Fetch a page with error handling."""
    try:
        headers = {
            "User-Agent": "OrwellCorpusCollector/1.0 (Educational ML Training Project)"
        }
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        return None


def extract_text_from_content_page(html: str) -> str:
    """Extract clean text from a content page."""
    soup = BeautifulSoup(html, "html.parser")

    # Remove unwanted elements
    for element in soup(["script", "style", "nav", "header", "footer", "noscript"]):
        element.decompose()

    # Remove navigation elements (breadcrumbs, menus)
    for nav in soup.find_all(class_=re.compile(r'nav|menu|breadcrumb', re.I)):
        nav.decompose()

    # Try to find the main text content
    # orwell.ru uses various containers
    content = None
    for selector in ["#text", ".text", "#content", ".content", "article", "main"]:
        content = soup.select_one(selector)
        if content and len(content.get_text(strip=True)) > 500:
            break

    # Fallback to body
    if not content or len(content.get_text(strip=True)) < 500:
        content = soup.body if soup.body else soup

    # Get text
    text = content.get_text(separator="\n")

    # Clean up
    lines = []
    for line in text.splitlines():
        line = line.strip()
        # Skip navigation-like lines
        if line and not re.match(r'^(Index|>|Library|Novels|Essays|Articles|Reviews|\[.*\]|©|\d+\.\d+ KiB)$', line):
            # Skip lines that are just file size indicators
            if not re.match(r'^\d+\.\d+ KiB$', line):
                lines.append(line)

    return "\n\n".join(lines)


def get_content_links(index_html: str, base_path: str) -> List[str]:
    """Extract links to actual content pages from an index page."""
    soup = BeautifulSoup(index_html, "html.parser")
    content_links = []

    for link in soup.find_all("a", href=True):
        href = link["href"]
        # Look for English content links (e_, en_p_, etc.)
        if re.search(r'/(e_|en_p_|en_c|en_app)', href):
            # Convert relative to absolute URL
            if href.startswith("/"):
                full_url = BASE_URL + href
            elif not href.startswith("http"):
                full_url = base_path + "/" + href
            else:
                full_url = href
            if full_url not in content_links:
                content_links.append(full_url)

    return sorted(content_links)


def scrape_work(category: str, slug: str) -> Optional[Dict]:
    """Scrape a single work, following links to get full content."""
    index_url = f"{BASE_URL}/library/{category}/{slug}/english/"
    base_path = f"{BASE_URL}/library/{category}/{slug}/english"

    # Fetch index page
    index_html = fetch_page(index_url)
    if not index_html:
        return None

    # Get title from index page
    soup = BeautifulSoup(index_html, "html.parser")
    title_tag = soup.find("h1") or soup.find("title")
    title = title_tag.get_text().strip() if title_tag else slug.replace("_", " ")
    # Clean up title
    title = re.sub(r'\s*\|.*$', '', title).strip()

    # Find content page links
    content_links = get_content_links(index_html, base_path)

    all_text = []

    if content_links:
        # Fetch each content page
        for link in content_links:
            time.sleep(DELAY_SECONDS)
            html = fetch_page(link)
            if html:
                text = extract_text_from_content_page(html)
                if len(text) > 100:  # Only add substantial content
                    all_text.append(text)
    else:
        # Some works might have content directly on the index page
        # or use a different pattern - try common alternatives
        alt_patterns = [
            f"{base_path}/e_text",
            f"{base_path}/e_{slug}",
        ]
        for alt_url in alt_patterns:
            html = fetch_page(alt_url)
            if html:
                text = extract_text_from_content_page(html)
                if len(text) > 500:
                    all_text.append(text)
                    break
            time.sleep(DELAY_SECONDS)

        # If still nothing, try extracting from index page itself
        if not all_text:
            text = extract_text_from_content_page(index_html)
            if len(text) > 500:
                all_text.append(text)

    if not all_text:
        return None

    combined_text = "\n\n---\n\n".join(all_text)

    return {
        "title": title,
        "category": category,
        "slug": slug,
        "url": index_url,
        "text": combined_text,
        "char_count": len(combined_text),
        "word_count": len(combined_text.split()),
        "num_parts": len(content_links) if content_links else 1,
    }


def save_corpus(works: List[Dict], output_dir: Path):
    """Save the corpus in multiple formats."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save individual files
    for work in works:
        category_dir = output_dir / work["category"]
        category_dir.mkdir(exist_ok=True)

        text_file = category_dir / f"{work['slug']}.txt"
        with open(text_file, "w", encoding="utf-8") as f:
            f.write(f"# {work['title']}\n\n")
            f.write(work["text"])

    # Save combined training file
    combined_file = output_dir / "orwell_complete.txt"
    with open(combined_file, "w", encoding="utf-8") as f:
        for work in works:
            f.write(f"\n\n{'='*80}\n")
            f.write(f"# {work['title']}\n")
            f.write(f"{'='*80}\n\n")
            f.write(work["text"])

    # Save metadata
    metadata = [{k: v for k, v in work.items() if k != "text"} for work in works]
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # Save JSONL for training
    with open(output_dir / "orwell_training.jsonl", "w", encoding="utf-8") as f:
        for work in works:
            record = {"text": work["text"], "title": work["title"], "category": work["category"]}
            f.write(json.dumps(record) + "\n")

    print(f"\nSaved corpus to {output_dir}/")
    print(f"  - Individual text files in category folders")
    print(f"  - Combined file: orwell_complete.txt")
    print(f"  - Training format: orwell_training.jsonl")


def main():
    """Main scraping function."""
    print("Orwell.ru Corpus Scraper (Full Text)")
    print("=" * 50)

    all_works = []
    total_works = sum(len(slugs) for slugs in WORKS.values())
    current = 0

    for category, slugs in WORKS.items():
        print(f"\nScraping {category}...")

        for slug in slugs:
            current += 1
            print(f"  [{current}/{total_works}] {slug}...", end=" ", flush=True)

            work = scrape_work(category, slug)

            if work:
                all_works.append(work)
                print(f"OK ({work['word_count']:,} words, {work['num_parts']} parts)")
            else:
                print("FAILED")

            time.sleep(DELAY_SECONDS)

    print(f"\n{'=' * 50}")
    print(f"Successfully scraped {len(all_works)}/{total_works} works")

    total_words = sum(w["word_count"] for w in all_works)
    total_chars = sum(w["char_count"] for w in all_works)
    print(f"Total: {total_words:,} words, {total_chars:,} characters")

    save_corpus(all_works, OUTPUT_DIR)

    print("\nDone! Your corpus is ready for training.")


if __name__ == "__main__":
    main()
