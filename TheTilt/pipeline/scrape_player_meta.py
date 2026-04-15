# %% Scrape batting hand and bowling style from ESPNcricinfo
"""Fetch player metadata (batting hand, bowling style) from ESPNcricinfo.

Uses Playwright to bypass Cloudflare protection. Extracts data from
the __NEXT_DATA__ JSON embedded in each player's page.

Output: data/processed/player_meta.json
    {cricinfo_key: {bat: str, bowl: str, role: str}, ...}
"""
import json
import re
import time
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import yaml


def load_config(config_path: str = "config/pipeline_config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_cricinfo_keys() -> Dict[str, int]:
    """Map player_id -> cricinfo_key for all IPL players."""
    people = pd.read_csv("data/raw/people.csv")
    config = load_config()
    processed_dir = Path(config["data"]["processed_dir"])
    pt = pd.read_parquet(processed_dir / "player_tilt.parquet")
    ipl_ids = set(pt["player_id"].dropna())
    ipl_people = people[people["identifier"].isin(ipl_ids) & people["key_cricinfo"].notna()]
    return {row["identifier"]: int(row["key_cricinfo"]) for _, row in ipl_people.iterrows()}


def scrape_player_meta(output_path: Optional[str] = None) -> Dict:
    """Scrape batting/bowling metadata for all IPL players."""
    from playwright.sync_api import sync_playwright

    config = load_config()
    processed_dir = Path(config["data"]["processed_dir"])
    output_path = Path(output_path or processed_dir / "player_meta.json")

    keys = get_cricinfo_keys()
    # Reverse map: cricinfo_key -> player_id
    reverse = {v: k for k, v in keys.items()}

    # Load existing results to resume if interrupted
    results: Dict[str, dict] = {}
    if output_path.exists():
        with open(output_path, "r") as f:
            results = json.load(f)
        print(f"  Loaded {len(results)} existing results from {output_path}")

    remaining = {k: v for k, v in keys.items() if k not in results}
    print(f"  {len(remaining)} players to scrape ({len(results)} already done)")

    if not remaining:
        print("  All players already scraped!")
        return results

    cricinfo_ids = list(remaining.values())
    player_ids = list(remaining.keys())

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        page = context.new_page()

        errors = []
        for i, (pid, cid) in enumerate(zip(player_ids, cricinfo_ids)):
            try:
                page.goto(f"https://www.espncricinfo.com/cricketers/x-{cid}", timeout=15000)
                # Extract __NEXT_DATA__
                el = page.query_selector("#__NEXT_DATA__")
                if el:
                    raw = el.inner_text()
                    data = json.loads(raw)
                    p = data["props"]["appPageProps"]["data"]["player"]
                    results[pid] = {
                        "bat": (p.get("longBattingStyles") or [None])[0],
                        "bowl": (p.get("longBowlingStyles") or [None])[0],
                        "role": (p.get("playingRoles") or [None])[0],
                    }
                else:
                    errors.append({"id": cid, "pid": pid, "error": "no __NEXT_DATA__"})
            except Exception as e:
                errors.append({"id": cid, "pid": pid, "error": str(e)})

            if (i + 1) % 50 == 0 or i == len(cricinfo_ids) - 1:
                # Save progress
                with open(output_path, "w") as f:
                    json.dump(results, f, indent=2)
                print(f"  Progress: {i + 1}/{len(cricinfo_ids)} ({len(results)} ok, {len(errors)} errors)")

            # Rate limit: ~200ms between requests
            if i % 10 == 9:
                time.sleep(1)

        browser.close()

    # Final save
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  Scraped {len(results)} players, {len(errors)} errors")
    if errors:
        print(f"  Errors: {errors[:5]}")

    return results


if __name__ == "__main__":
    scrape_player_meta()
