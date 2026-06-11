# %% Imports
import random
import shutil
import time
import zipfile
from pathlib import Path
from typing import Optional

import requests
import yaml

# Cricsheet's CDN intermittently 415s/503s automated requests (e.g. the
# 2026-06-03 21:52Z scheduled run failed all 3 bare retries inside 30s, while
# the same URL served fine minutes later). Retry over a few minutes so a
# transient blip self-heals instead of failing the whole refresh.
# The 2026-06-11 episode outlasted the original 5-attempt/~5-min window
# (issue #205), so the schedule now stretches to ~26 min before jitter:
# sleeps of 15/30/60/120/240/480/600s between 8 attempts.
MAX_DOWNLOAD_ATTEMPTS = 8
BACKOFF_BASE_SECONDS = 15
BACKOFF_CAP_SECONDS = 600

DOWNLOAD_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; TheTiltBot/1.0; +https://github.com/soldoutbudokan/Templates)",
    "Accept": "application/zip, application/octet-stream, */*",
}


class CricsheetDownloadError(RuntimeError):
    """Cricsheet stayed unreachable through the whole retry window.

    Distinguishable from real pipeline failures so the refresh workflow can
    soft-fail on it (skip the refresh, next cron catches up — issue #205)
    while the retrain workflow keeps hard-failing.
    """


# %% Configuration
def load_config(config_path: str = "config/pipeline_config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# %% Download function
def download_cricsheet_data(
    url: Optional[str] = None,
    output_dir: Optional[str] = None,
    force: bool = False,
) -> Path:
    """Download and extract Cricsheet IPL ball-by-ball JSON data."""
    config = load_config()
    url = url or config["data"]["source_url"]
    output_dir = Path(output_dir or config["data"]["raw_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    zip_path = output_dir / "ipl_json.zip.part"

    # Check if already downloaded
    json_files = list(output_dir.glob("*.json"))
    if json_files and not force:
        print(f"Found {len(json_files)} JSON files in {output_dir}, skipping download.")
        print("Use force=True to re-download.")
        return output_dir

    # Download (with retries — see MAX_DOWNLOAD_ATTEMPTS note above)
    print(f"Downloading IPL data from {url}...")
    last_err: Optional[Exception] = None
    response = None
    for attempt in range(1, MAX_DOWNLOAD_ATTEMPTS + 1):
        try:
            response = requests.get(url, stream=True, headers=DOWNLOAD_HEADERS, timeout=60)
            response.raise_for_status()
            break
        except requests.exceptions.RequestException as e:
            last_err = e
            if attempt < MAX_DOWNLOAD_ATTEMPTS:
                # Exponential backoff (15s doubling, capped at 10 min) plus
                # jitter to spread retries across a transient CDN block.
                backoff = min(BACKOFF_BASE_SECONDS * 2 ** (attempt - 1), BACKOFF_CAP_SECONDS)
                delay = backoff + random.uniform(0, 0.5 * backoff)
                print(f"  Attempt {attempt} failed ({e}); retrying in {delay:.0f}s...")
                time.sleep(delay)
            else:
                print(f"  Attempt {attempt} failed ({e}); giving up.")
    else:
        raise CricsheetDownloadError(
            f"Failed to download {url} after {MAX_DOWNLOAD_ATTEMPTS} attempts"
        ) from last_err

    total_size = int(response.headers.get("content-length", 0))
    downloaded = 0

    with open(zip_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            if total_size > 0:
                pct = downloaded / total_size * 100
                print(f"\r  {downloaded / 1024 / 1024:.1f} MB / {total_size / 1024 / 1024:.1f} MB ({pct:.0f}%)", end="", flush=True)

    print(f"\n  Saved to {zip_path}")

    # Validate, extract to a temp dir, then swap into place on success only —
    # a corrupt/partial download can no longer strand a half-extracted corpus
    # that the next non-force run silently parses (issue #196).
    if not zipfile.is_zipfile(zip_path):
        zip_path.unlink()
        raise RuntimeError(f"Downloaded file from {url} is not a valid zip archive")

    print("Extracting...")
    extract_dir = output_dir / ".extract_tmp"
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir()
    extract_root = extract_dir.resolve()
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.namelist():
            # Zip-slip guard (issue #196): every member must resolve inside
            # the extraction dir — a compromised CDN response must not be able
            # to write outside the tree on the runner.
            target = (extract_dir / member).resolve()
            if target != extract_root and extract_root not in target.parents:
                raise RuntimeError(f"Zip member escapes extraction dir (zip-slip): {member!r}")
        zf.extractall(extract_dir)

    # Swap: clear the old corpus, then move the fresh files in. Also drops
    # any matches removed upstream instead of parsing them forever.
    for stale in output_dir.glob("*.json"):
        stale.unlink()
    for p in sorted(extract_dir.iterdir()):
        dest = output_dir / p.name
        if dest.exists():
            dest.unlink()
        p.rename(dest)
    extract_dir.rmdir()
    zip_path.unlink()

    json_files = list(output_dir.glob("*.json"))
    print(f"  Extracted {len(json_files)} match files to {output_dir}")

    return output_dir


# %% Main
if __name__ == "__main__":
    download_cricsheet_data()
