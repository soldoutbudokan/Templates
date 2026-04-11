# %% Imports
"""Download Cricsheet people register and resolve full player names via Wikidata."""
import csv
import io
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, Optional, Set

import yaml


# %% Configuration
def load_config(config_path: str = "config/pipeline_config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


PEOPLE_CSV_URL = "https://cricsheet.org/register/people.csv"
WIKIDATA_SPARQL_URL = "https://query.wikidata.org/sparql"
WIKIDATA_BATCH_SIZE = 150
WIKIDATA_RATE_LIMIT = 1.0  # seconds between batches


# %% Download and parse Cricsheet people register
def download_people_csv(output_path: Optional[str] = None) -> Path:
    """Download Cricsheet people register CSV."""
    config = load_config()
    output_path = Path(output_path or config["data"]["raw_dir"]) / "people.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading Cricsheet people register...")
    req = urllib.request.Request(PEOPLE_CSV_URL, headers={"User-Agent": "TheTilt/1.0"})
    resp = urllib.request.urlopen(req, timeout=30)
    data = resp.read()

    with open(output_path, "wb") as f:
        f.write(data)

    print(f"  Saved to {output_path} ({len(data):,} bytes)")
    return output_path


def parse_people_csv(csv_path: Optional[str] = None) -> Dict[str, Dict]:
    """Parse people.csv into {uuid: {name, cricinfo_key}} mapping."""
    if csv_path is None:
        config = load_config()
        csv_path = Path(config["data"]["raw_dir"]) / "people.csv"
    else:
        csv_path = Path(csv_path)

    players = {}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            uid = row["identifier"]
            players[uid] = {
                "name": row["name"],
                "cricinfo_key": row.get("key_cricinfo", "").strip(),
            }

    return players


# %% Resolve full names from Wikidata
def resolve_full_names(
    cricinfo_keys: Dict[str, str],
    cache_path: Optional[Path] = None,
) -> Dict[str, str]:
    """Resolve full player names from Wikidata using ESPNcricinfo IDs (P2697).

    Args:
        cricinfo_keys: {cricinfo_key: player_uuid} mapping
        cache_path: Path to cached results (will be updated incrementally)

    Returns:
        {player_uuid: full_name} mapping
    """
    # Load cache if exists
    full_names: Dict[str, str] = {}
    if cache_path and cache_path.exists():
        with open(cache_path, "r") as f:
            full_names = json.load(f)

    # Find keys that still need resolution
    cached_uuids = set(full_names.keys())
    to_resolve = {k: v for k, v in cricinfo_keys.items() if v not in cached_uuids}

    if not to_resolve:
        print(f"  All {len(full_names)} full names cached")
        return full_names

    print(f"  Resolving {len(to_resolve)} full names from Wikidata...")
    keys_list = list(to_resolve.keys())

    for i in range(0, len(keys_list), WIKIDATA_BATCH_SIZE):
        batch = keys_list[i : i + WIKIDATA_BATCH_SIZE]
        values = " ".join(f'"{k}"' for k in batch)

        query = f"""
        SELECT ?cricinfo_id ?itemLabel WHERE {{
          VALUES ?cricinfo_id {{ {values} }}
          ?item wdt:P2697 ?cricinfo_id .
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" }}
        }}
        """

        url = WIKIDATA_SPARQL_URL + "?" + urllib.parse.urlencode(
            {"query": query, "format": "json"}
        )
        req = urllib.request.Request(url, headers={"User-Agent": "TheTilt/1.0"})

        try:
            resp = urllib.request.urlopen(req, timeout=60)
            data = json.loads(resp.read())
            for r in data["results"]["bindings"]:
                ckey = r["cricinfo_id"]["value"]
                name = r["itemLabel"]["value"]
                if ckey in to_resolve:
                    uid = to_resolve[ckey]
                    full_names[uid] = name
        except Exception as e:
            print(f"    Batch {i // WIKIDATA_BATCH_SIZE + 1} failed: {e}")

        batch_num = i // WIKIDATA_BATCH_SIZE + 1
        total_batches = (len(keys_list) + WIKIDATA_BATCH_SIZE - 1) // WIKIDATA_BATCH_SIZE
        print(f"    Batch {batch_num}/{total_batches}: {len(full_names)} resolved")

        if i + WIKIDATA_BATCH_SIZE < len(keys_list):
            time.sleep(WIKIDATA_RATE_LIMIT)

    # Save cache
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(full_names, f, indent=2)

    return full_names


# %% Main
def download_and_resolve(qualified_ids: Optional[Set[str]] = None) -> Dict[str, str]:
    """Download people register and resolve full names for qualified players.

    Args:
        qualified_ids: Set of player UUIDs to resolve. If None, resolves all.

    Returns:
        {player_uuid: full_name} mapping
    """
    config = load_config()
    processed_dir = Path(config["data"]["processed_dir"])
    cache_path = processed_dir / "full_names.json"

    # Download people.csv
    csv_path = download_people_csv()

    # Parse and filter to qualified players with cricinfo keys
    people = parse_people_csv(csv_path)
    cricinfo_keys = {}
    for uid, info in people.items():
        if info["cricinfo_key"] and (qualified_ids is None or uid in qualified_ids):
            cricinfo_keys[info["cricinfo_key"]] = uid

    print(f"  {len(cricinfo_keys)} players with ESPNcricinfo keys to resolve")

    # Resolve via Wikidata
    full_names = resolve_full_names(cricinfo_keys, cache_path)
    print(f"  Resolved {len(full_names)} full names total")

    return full_names


# %% Script entry
if __name__ == "__main__":
    full_names = download_and_resolve()
