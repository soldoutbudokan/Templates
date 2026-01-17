#!/usr/bin/env python3
"""
Canadian Competition Tribunal Document Scraper

Scrapes all PDF documents from a case page and its linked pages.
Usage: python ct_scraper.py <case_url>
"""

import argparse
import os
import re
import sys
from urllib.parse import urljoin, urlparse
from collections import deque

import requests
from bs4 import BeautifulSoup


class CTScraper:
    def __init__(self, base_url: str, output_dir: str = "output"):
        self.base_url = base_url
        self.output_base = output_dir
        self.output_dir = output_dir  # Will be updated with case name
        self.case_name = None
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
        })
        self.visited_urls: set[str] = set()
        self.downloaded_pdfs: set[str] = set()
        self.base_domain = urlparse(base_url).netloc

    def extract_case_name(self, soup: BeautifulSoup) -> str:
        """Extract case name from page title or heading."""
        # Try h3 with class title first (more reliable)
        h3 = soup.find("h3", class_="title")
        if h3:
            case_name = h3.get_text().strip()
            case_name = re.sub(r'[<>:"/\\|?*]', "_", case_name)
            return case_name

        # Try to get from title tag
        title = soup.find("title")
        if title:
            title_text = title.get_text().strip()
            # Format: "Case Name - CT-XXXX-XXX - Competition Tribunal"
            match = re.match(r"(.+?)\s*-\s*Competition Tribunal", title_text)
            if match:
                case_name = match.group(1).strip()
                case_name = re.sub(r'[<>:"/\\|?*]', "_", case_name)
                return case_name

        return "Case_Documents"

    def is_same_domain(self, url: str) -> bool:
        """Check if URL belongs to the same domain."""
        parsed = urlparse(url)
        return parsed.netloc == self.base_domain or parsed.netloc == ""

    def normalize_url(self, url: str, current_page: str) -> str:
        """Convert relative URL to absolute URL."""
        return urljoin(current_page, url)

    def is_pdf_url(self, url: str) -> bool:
        """Check if URL points to a PDF file or document download."""
        parsed = urlparse(url.lower())
        path = parsed.path
        return (
            path.endswith(".pdf") or
            "/pdf/" in path or
            "format=pdf" in url.lower() or
            path.endswith("/document.do")
        )

    def is_document_page(self, url: str) -> bool:
        """Check if URL is likely a document/case page worth exploring."""
        patterns = [
            r"/item/\d+",
            r"/doc/",
            r"/decision",
            r"/order",
            r"/ruling",
            r"/cd/",
            r"index\.do",
        ]
        return any(re.search(p, url, re.IGNORECASE) for p in patterns)

    def get_safe_filename(self, url: str, response: requests.Response) -> str:
        """Extract a safe filename from URL or response headers."""
        cd = response.headers.get("Content-Disposition", "")
        if "filename=" in cd:
            match = re.search(r'filename[^;=\n]*=["\']?([^"\';\n]+)', cd)
            if match:
                return match.group(1).strip()

        parsed = urlparse(url)
        filename = os.path.basename(parsed.path)

        if not filename or not filename.endswith(".pdf"):
            filename = f"document_{hash(url) % 100000}.pdf"

        filename = re.sub(r'[<>:"/\\|?*]', "_", filename)
        return filename

    def download_pdf(self, url: str) -> bool:
        """Download a PDF file."""
        if url in self.downloaded_pdfs:
            return False

        try:
            print(f"  Downloading: {url}")
            response = self.session.get(url, timeout=30, stream=True)
            response.raise_for_status()

            content_type = response.headers.get("Content-Type", "")
            if "pdf" not in content_type.lower() and not url.lower().endswith(".pdf"):
                print(f"    Skipping (not a PDF): {content_type}")
                return False

            filename = self.get_safe_filename(url, response)
            filepath = os.path.join(self.output_dir, filename)

            base, ext = os.path.splitext(filepath)
            counter = 1
            while os.path.exists(filepath):
                filepath = f"{base}_{counter}{ext}"
                counter += 1

            with open(filepath, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            self.downloaded_pdfs.add(url)
            print(f"    Saved: {os.path.basename(filepath)}")
            return True

        except requests.RequestException as e:
            print(f"    Error downloading {url}: {e}")
            return False

    def should_skip_url(self, url: str) -> bool:
        """Check if URL should be skipped (navigation, language toggle, etc.)."""
        skip_patterns = [
            r"/nav\.do$",
            r"/nav_date\.do$",
            r"/l\.do$",
            r"/rss/",
            r"/q\.do$",
            r"alternatelocale=",
            r"/fr/",
        ]
        return any(re.search(p, url, re.IGNORECASE) for p in skip_patterns)

    def extract_links(self, soup: BeautifulSoup, current_url: str) -> tuple[list[str], list[str]]:
        """Extract PDF links and page links from HTML."""
        pdf_links = []
        page_links = []

        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"].strip()
            if not href or href.startswith("#") or href.startswith("javascript:"):
                continue

            full_url = self.normalize_url(href, current_url)

            if not self.is_same_domain(full_url):
                continue

            if self.should_skip_url(full_url):
                continue

            if self.is_pdf_url(full_url):
                pdf_links.append(full_url)
            elif self.is_document_page(full_url) and full_url not in self.visited_urls:
                page_links.append(full_url)

        return pdf_links, page_links

    def scrape_page(self, url: str) -> tuple[list[str], list[str]]:
        """Scrape a single page for PDFs and links."""
        if url in self.visited_urls:
            return [], []

        self.visited_urls.add(url)
        print(f"\nScraping: {url}")

        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")
            pdf_links, page_links = self.extract_links(soup, url)

            for iframe in soup.find_all("iframe"):
                iframe_src = iframe.get("src", "")
                if iframe_src:
                    iframe_url = self.normalize_url(iframe_src, url)
                    if self.is_same_domain(iframe_url) and iframe_url not in self.visited_urls:
                        print(f"  Found iframe: {iframe_url}")
                        iframe_pdfs, iframe_pages = self.scrape_page(iframe_url)
                        pdf_links.extend(iframe_pdfs)
                        page_links.extend(iframe_pages)

            if "iframe=true" not in url and "/item/" in url:
                iframe_url = url + ("&" if "?" in url else "?") + "iframe=true"
                if iframe_url not in self.visited_urls:
                    iframe_pdfs, iframe_pages = self.scrape_page(iframe_url)
                    pdf_links.extend(iframe_pdfs)
                    page_links.extend(iframe_pages)

            return pdf_links, page_links

        except requests.RequestException as e:
            print(f"  Error fetching page: {e}")
            return [], []

    def scrape(self, max_depth: int = 2) -> int:
        """
        Scrape all PDFs from the base URL and linked pages.

        Args:
            max_depth: How many levels deep to follow links (default 2)

        Returns:
            Number of PDFs downloaded
        """
        print(f"Fetching case information from: {self.base_url}")
        try:
            response = self.session.get(self.base_url, timeout=30)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")

            iframe_url = self.base_url + ("&" if "?" in self.base_url else "?") + "iframe=true"
            iframe_response = self.session.get(iframe_url, timeout=30)
            if iframe_response.ok:
                iframe_soup = BeautifulSoup(iframe_response.text, "html.parser")
                self.case_name = self.extract_case_name(iframe_soup)
            else:
                self.case_name = self.extract_case_name(soup)
        except requests.RequestException as e:
            print(f"Warning: Could not fetch case name: {e}")
            self.case_name = "Case_Documents"

        # Create output directory with case name
        self.output_dir = os.path.join(self.output_base, f"{self.case_name} Documents")
        os.makedirs(self.output_dir, exist_ok=True)

        print(f"\nCase: {self.case_name}")
        print(f"Starting scrape of: {self.base_url}")
        print(f"Output directory: {os.path.abspath(self.output_dir)}")
        print(f"Max depth: {max_depth}")
        print("=" * 60)

        queue: deque[tuple[str, int]] = deque([(self.base_url, 0)])

        while queue:
            url, depth = queue.popleft()

            if url in self.visited_urls:
                continue

            pdf_links, page_links = self.scrape_page(url)

            for pdf_url in pdf_links:
                self.download_pdf(pdf_url)

            if depth < max_depth:
                for page_url in page_links:
                    if page_url not in self.visited_urls:
                        queue.append((page_url, depth + 1))

        print("\n" + "=" * 60)
        print(f"Scraping complete!")
        print(f"Pages visited: {len(self.visited_urls)}")
        print(f"PDFs downloaded: {len(self.downloaded_pdfs)}")

        return len(self.downloaded_pdfs)


def main():
    parser = argparse.ArgumentParser(
        description="Scrape PDF documents from Canadian Competition Tribunal case pages"
    )
    parser.add_argument(
        "url",
        help="URL of the case page to scrape"
    )
    parser.add_argument(
        "-o", "--output",
        default="output",
        help="Output directory for case folder (default: output)"
    )
    parser.add_argument(
        "-d", "--depth",
        type=int,
        default=2,
        help="Max depth to follow links (default: 2)"
    )

    args = parser.parse_args()

    if not args.url.startswith(("http://", "https://")):
        print("Error: Please provide a valid URL starting with http:// or https://")
        sys.exit(1)

    scraper = CTScraper(args.url, args.output)
    count = scraper.scrape(max_depth=args.depth)

    if count == 0:
        print("\nNo PDFs were found. The page structure may have changed.")
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
