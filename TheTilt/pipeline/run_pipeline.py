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
from pipeline.download_people import download_and_resolve
from pipeline.parse_matches import parse_all_matches
from pipeline.build_features import build_all_features
from pipeline.train_win_prob import train_and_evaluate
from pipeline.compute_tilt import (
    compute_tilt,
    aggregate_player_season_tilt,
    aggregate_team_tilt,
    aggregate_team_season_tilt,
)
from pipeline.export_json import export_all


# %% Data-only refresh (skips training, reuses committed pickle)
def refresh_tilt_data_only() -> None:
    """Refresh data and recompute TILT without retraining the model.

    Steps run: 1 (download, force), 2 (people), 3 (parse), 4 (features),
               6 (compute TILT using existing pickle), 7 (export JSON).
    Skips step 5 (training). Use the `retrain-tilt-model` workflow when
    features or data-coverage assumptions change.
    """
    import yaml

    project_root = Path(__file__).parent.parent

    # Fail loudly if the committed pickle is missing — the refresh workflow
    # has nothing to fall back to.
    with open(project_root / "config/pipeline_config.yaml") as f:
        cfg = yaml.safe_load(f)
    model_path = project_root / cfg["model"]["save_path"]
    if not model_path.exists():
        raise FileNotFoundError(
            f"{model_path} is missing. Run the retrain-tilt-model workflow "
            "(or `python pipeline/run_pipeline.py` locally) to generate it."
        )

    start = time.time()
    print("=" * 60)
    print("  THE TILT — Data-only refresh (skipping model training)")
    print("=" * 60)

    print("\n[1/6] Downloading Cricsheet IPL data (force=True)...")
    download_cricsheet_data(force=True)

    print("\n[2/6] Downloading player register + resolving full names...")
    download_and_resolve()

    print("\n[3/6] Parsing ball-by-ball match data...")
    parse_all_matches()

    print("\n[4/6] Building match state features...")
    build_all_features()

    print("\n[5/6] Computing player TILT scores with existing model...")
    deltas_df, player_tilt = compute_tilt()

    processed_dir = Path("data/processed")
    deltas_df.to_parquet(processed_dir / "deltas.parquet", index=False)
    player_tilt.to_parquet(processed_dir / "player_tilt.parquet", index=False)

    print("\n  Computing per-player-season + team aggregates...")
    player_season_tilt = aggregate_player_season_tilt(deltas_df)
    team_tilt = aggregate_team_tilt(deltas_df)
    team_season_tilt = aggregate_team_season_tilt(deltas_df)
    player_season_tilt.to_parquet(processed_dir / "player_season_tilt.parquet", index=False)
    team_tilt.to_parquet(processed_dir / "team_tilt.parquet", index=False)
    team_season_tilt.to_parquet(processed_dir / "team_season_tilt.parquet", index=False)

    print("\n[6/6] Exporting JSON for website...")
    export_all(
        deltas_df,
        player_tilt,
        player_season_tilt=player_season_tilt,
        team_tilt=team_tilt,
        team_season_tilt=team_season_tilt,
    )

    elapsed = time.time() - start
    print(f"\n{'=' * 60}")
    print(f"  Refresh complete in {elapsed:.0f}s")
    print(f"{'=' * 60}")


# %% Run pipeline
def run_pipeline() -> None:
    """Run the full TILT pipeline."""
    start = time.time()

    print("=" * 60)
    print("  THE TILT — Win Probability Added for IPL Cricket")
    print("=" * 60)

    # Step 1: Download data
    print("\n[1/7] Downloading Cricsheet IPL data...")
    download_cricsheet_data()

    # Step 2: Download people register + resolve full names
    print("\n[2/7] Downloading player register + resolving full names...")
    download_and_resolve()

    # Step 3: Parse matches
    print("\n[3/7] Parsing ball-by-ball match data...")
    parse_all_matches()

    # Step 4: Build features
    print("\n[4/7] Building match state features...")
    build_all_features()

    # Step 5: Train model
    print("\n[5/7] Training win probability model...")
    train_and_evaluate()

    # Step 6: Compute TILT
    print("\n[6/7] Computing player TILT scores...")
    deltas_df, player_tilt = compute_tilt()

    # Save intermediate results for later use
    processed_dir = Path("data/processed")
    deltas_df.to_parquet(processed_dir / "deltas.parquet", index=False)
    player_tilt.to_parquet(processed_dir / "player_tilt.parquet", index=False)

    print("\n  Computing per-player-season + team aggregates...")
    player_season_tilt = aggregate_player_season_tilt(deltas_df)
    team_tilt = aggregate_team_tilt(deltas_df)
    team_season_tilt = aggregate_team_season_tilt(deltas_df)
    player_season_tilt.to_parquet(processed_dir / "player_season_tilt.parquet", index=False)
    team_tilt.to_parquet(processed_dir / "team_tilt.parquet", index=False)
    team_season_tilt.to_parquet(processed_dir / "team_season_tilt.parquet", index=False)

    # Step 7: Export JSON
    print("\n[7/7] Exporting JSON for website...")
    export_all(
        deltas_df,
        player_tilt,
        player_season_tilt=player_season_tilt,
        team_tilt=team_tilt,
        team_season_tilt=team_season_tilt,
    )

    elapsed = time.time() - start
    print(f"\n{'=' * 60}")
    print(f"  Pipeline complete in {elapsed:.0f}s")
    print(f"  Results exported to public/data/")
    print(f"{'=' * 60}")


# %% Main
if __name__ == "__main__":
    run_pipeline()
