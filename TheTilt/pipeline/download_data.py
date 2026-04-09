# %% Imports
import zipfile
from pathlib import Path
from typing import Optional

import requests
import yaml


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

    zip_path = output_dir / "ipl_json.zip"

    # Check if already downloaded
    json_files = list(output_dir.glob("*.json"))
    if json_files and not force:
        print(f"Found {len(json_files)} JSON files in {output_dir}, skipping download.")
        print("Use force=True to re-download.")
        return output_dir

    # Download
    print(f"Downloading IPL data from {url}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()

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

    # Extract
    print("Extracting...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(output_dir)

    json_files = list(output_dir.glob("*.json"))
    print(f"  Extracted {len(json_files)} match files to {output_dir}")

    # Clean up zip
    zip_path.unlink()

    return output_dir


# %% Main
if __name__ == "__main__":
    download_cricsheet_data()
