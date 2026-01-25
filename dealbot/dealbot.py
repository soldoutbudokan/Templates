#!/usr/bin/env python3
"""
DealBot - Price Tracker
Runs via GitHub Actions to check for deals daily.
"""

import hashlib
import json
import os
import re
import requests
from datetime import datetime
from urllib.parse import urlparse

# =============================================================================
# CONFIGURATION - Edit your products here!
# =============================================================================

# Sites to ignore (second-hand, sketchy, etc.)
BLOCKED_SITES = [
    "ebay.ca",
    "ebay.com",
    "facebook.com",
    "marketplace.facebook.com",
    "kijiji.ca",
    "craigslist.org",
    "poshmark.com",
    "mercari.com",
    "aliexpress.com",
    "temu.com",
    "wish.com",
]

PRODUCTS = [
    {
        "name": "Sony WH-1000XM5 Headphones",
        "keywords": "Sony WH-1000XM5 headphones",
        "url": "",  # Optional: direct product URL for Shopify stores
        "typical_price": 450.00,
        "target_price": 350.00,
    },
    {
        "name": "Example Product 2",
        "keywords": "Example Product keywords",
        "url": "",
        "typical_price": 100.00,
        "target_price": 80.00,
    },
]

# =============================================================================
# API KEYS - Set these as GitHub Secrets, not here!
# =============================================================================

BRAVE_API_KEY = os.environ.get("BRAVE_API_KEY", "")
NTFY_TOPIC = os.environ.get("NTFY_TOPIC", "your-unique-topic")

# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def send_notification(title, message, url=None, priority="default"):
    """Send push notification via Ntfy."""
    # Remove emojis from title for header encoding compatibility
    clean_title = title.encode('ascii', 'ignore').decode('ascii').strip()
    if not clean_title:
        clean_title = "DealBot"

    headers = {"Title": clean_title, "Priority": priority}
    if url:
        headers["Click"] = url

    try:
        resp = requests.post(
            f"https://ntfy.sh/{NTFY_TOPIC}",
            data=message.encode('utf-8'),
            headers=headers,
            timeout=10
        )
        resp.raise_for_status()
        print(f"📱 Notification sent: {title}")
        return True
    except Exception as e:
        print(f"❌ Notification failed: {e}")
        return False


def extract_price(text):
    """Extract price from text."""
    patterns = [
        r'\$\s*(\d{1,5}(?:[.,]\d{2})?)',
        r'CAD\s*\$?\s*(\d{1,5}(?:[.,]\d{2})?)',
        r'CA\$\s*(\d{1,5}(?:[.,]\d{2})?)',
        r'(\d{1,5}(?:[.,]\d{2})?)\s*(?:CAD|\$)',
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1).replace(',', '.'))
            except ValueError:
                continue
    return None


def extract_domain(url):
    """Get clean domain from URL."""
    try:
        domain = urlparse(url).netloc
        return domain.replace('www.', '')
    except:
        return "Unknown"


def search_brave(query):
    """Search using Brave Search API."""
    if not BRAVE_API_KEY:
        print("⚠️ No Brave API key set")
        return []

    headers = {
        "Accept": "application/json",
        "X-Subscription-Token": BRAVE_API_KEY,
    }
    params = {"q": query, "country": "CA", "count": 20}

    try:
        resp = requests.get(
            "https://api.search.brave.com/res/v1/web/search",
            headers=headers,
            params=params,
            timeout=30
        )
        resp.raise_for_status()
        results = resp.json().get("web", {}).get("results", [])
        print(f"  🌐 Brave search: {len(results)} results")
        return results
    except Exception as e:
        print(f"  ❌ Search error: {e}")
        return []


def scrape_shopify_price(url):
    """Try to scrape price from Shopify stores."""
    try:
        json_url = url.rstrip('/') + '.json'
        resp = requests.get(json_url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            variants = data.get('product', {}).get('variants', [])
            if variants:
                return float(variants[0].get('price', 0))
    except:
        pass

    # Fallback: scrape HTML
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code == 200:
            return extract_price(resp.text)
    except:
        pass

    return None


def get_deal_hash(product_name, price, url):
    """Generate unique hash for a deal."""
    content = f"{product_name}:{price}:{url}"
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def check_product(product):
    """Check a single product for deals."""
    name = product['name']
    keywords = product['keywords']
    typical_price = product['typical_price']
    target_price = product['target_price']
    direct_url = product.get('url')

    print(f"\n🔍 Checking: {name}")
    deals = []

    # Method 1: Direct URL scraping (Shopify)
    if direct_url:
        price = scrape_shopify_price(direct_url)
        if price:
            print(f"  📦 Direct price: ${price:.2f}")
            discount = ((typical_price - price) / typical_price) * 100
            if price <= target_price:
                deals.append({
                    'price': price,
                    'discount': discount,
                    'source': extract_domain(direct_url),
                    'url': direct_url,
                })
                print(f"  ✅ DEAL! ${price:.2f} <= ${target_price:.2f}")
            else:
                print(f"  ❌ No deal: ${price:.2f} > ${target_price:.2f}")

    # Method 2: Brave Search
    if BRAVE_API_KEY:
        query = f'{keywords} price canada sale'
        results = search_brave(query)

        for result in results[:10]:
            title = result.get('title', '')
            desc = result.get('description', '')
            url = result.get('url', '')

            # Skip blocked sites
            domain = extract_domain(url)
            if any(blocked in domain for blocked in BLOCKED_SITES):
                continue

            if direct_url and domain == extract_domain(direct_url):
                continue

            combined = f"{title} {desc}"
            price = extract_price(combined)

            if price and price <= target_price:
                discount = ((typical_price - price) / typical_price) * 100
                deals.append({
                    'price': price,
                    'discount': discount,
                    'source': extract_domain(url),
                    'url': url,
                })
                print(f"  ✅ DEAL at {extract_domain(url)}: ${price:.2f}")

    return deals


def main():
    """Main function."""
    print("=" * 50)
    print(f"🤖 DealBot - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 50)

    if not BRAVE_API_KEY:
        print("⚠️ BRAVE_API_KEY not set - only direct scraping will work")

    total_deals = 0
    notifications = 0
    seen_hashes = set()  # Simple in-memory dedup for this run

    for product in PRODUCTS:
        try:
            deals = check_product(product)

            for deal in deals:
                total_deals += 1
                deal_hash = get_deal_hash(product['name'], deal['price'], deal['url'])

                if deal_hash in seen_hashes:
                    continue
                seen_hashes.add(deal_hash)

                # Send notification
                title = f"🛒 Deal: {product['name']}"
                message = f"${deal['price']:.2f} ({deal['discount']:.0f}% off)\nSource: {deal['source']}"

                if send_notification(title, message, deal['url'], priority="high"):
                    notifications += 1

        except Exception as e:
            print(f"  ❌ Error: {e}")

    # Summary
    print("\n" + "=" * 50)
    print(f"📊 Summary:")
    print(f"   Products: {len(PRODUCTS)}")
    print(f"   Deals found: {total_deals}")
    print(f"   Notifications: {notifications}")
    print("=" * 50)

    # Always send a summary so you know it ran
    if notifications == 0:
        send_notification(
            "📊 DealBot: No deals today",
            f"Checked {len(PRODUCTS)} products",
            priority="low"
        )


if __name__ == "__main__":
    main()
