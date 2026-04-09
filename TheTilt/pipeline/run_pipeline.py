# %% The TILT — Full Pipeline Orchestrator
"""
Run the complete pipeline end-to-end:
1. Download Cricsheet IPL data
2. Parse ball-by-ball events
3. Build match state features
4. Train win probability model
5. Compute per-player TILT scores
6. Export JSON for website
"""

import sys
import time
from pathlib import Path

# Ensure project root is on path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pipeline.download_data import download_cricsheet_data
from pipeline.parse_matches import parse_all_matches
from pipeline.build_features import build_all_features
from pipeline.train_win_prob import train_and_evaluate
from pipeline.compute_tilt import compute_tilt
from pipeline.export_json import export_all


# %% Run pipeline
def run_pipeline() -> None:
    """Run the full TILT pipeline."""
    start = time.time()

    print("=" * 60)
    print("  THE TILT — Win Probability Added for IPL Cricket")
    print("=" * 60)

    # Step 1: Download data
    print("\n[1/6] Downloading Cricsheet IPL data...")
    download_cricsheet_data()

    # Step 2: Parse matches
    print("\n[2/6] Parsing ball-by-ball match data...")
    parse_all_matches()

    # Step 3: Build features
    print("\n[3/6] Building match state features...")
    build_all_features()

    # Step 4: Train model
    print("\n[4/6] Training win probability model...")
    train_and_evaluate()

    # Step 5: Compute TILT
    print("\n[5/6] Computing player TILT scores...")
    deltas_df, player_tilt = compute_tilt()

    # Save intermediate results for later use
    processed_dir = Path("data/processed")
    deltas_df.to_parquet(processed_dir / "deltas.parquet", index=False)
    player_tilt.to_parquet(processed_dir / "player_tilt.parquet", index=False)

    # Step 6: Export JSON
    print("\n[6/6] Exporting JSON for website...")
    export_all(deltas_df, player_tilt)

    elapsed = time.time() - start
    print(f"\n{'=' * 60}")
    print(f"  Pipeline complete in {elapsed:.0f}s")
    print(f"  Results exported to public/data/")
    print(f"{'=' * 60}")


# %% Main
if __name__ == "__main__":
    run_pipeline()
