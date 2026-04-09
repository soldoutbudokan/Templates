# %% Imports
import pickle
import re
from pathlib import Path
from typing import Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml


# %% Configuration
FEATURE_COLS = [
    "innings",
    "balls_remaining",
    "wickets_in_hand",
    "runs_scored",
    "run_rate",
    "required_run_rate",
    "target",
    "runs_needed",
    "is_powerplay",
    "is_middle",
    "is_death",
    "recent_run_rate",
    "recent_wickets",
    "venue",
    "batting_team_chose_to_bat",
    "season_numeric",
]


def load_config(config_path: str = "config/pipeline_config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# %% Compute state after each ball
def compute_state_after(row: pd.Series) -> dict:
    """Compute match state features AFTER a ball is bowled."""
    after = {}
    after["innings"] = row["innings"]
    after["runs_scored"] = row["runs_scored"] + row["runs_total"]
    after["wickets_fallen"] = row["wickets_fallen"] + (1 if row["is_wicket"] else 0)
    after["wickets_in_hand"] = 10 - after["wickets_fallen"]
    after["balls_remaining"] = max(row["balls_remaining"] - 1, 1)

    overs_bowled = (row["ball_number"]) / 6  # After this ball
    after["run_rate"] = after["runs_scored"] / max(overs_bowled, 0.1)

    after["is_powerplay"] = row["is_powerplay"]
    after["is_middle"] = row["is_middle"]
    after["is_death"] = row["is_death"]

    after["target"] = row["target"]
    if row["target"] > 0:
        after["runs_needed"] = row["target"] - after["runs_scored"]
        after["required_run_rate"] = min(
            after["runs_needed"] / max(after["balls_remaining"] / 6, 0.1),
            36.0,
        )
    else:
        after["runs_needed"] = 0
        after["required_run_rate"] = 0.0

    # Recalculate recent stats after this ball (improved approximation)
    # recent_run_rate covers last 18 balls (3 overs). Adding this ball's runs
    # and recomputing is more accurate than copying the pre-ball value.
    recent_balls = min(row.get("ball_number", 18), 18)
    recent_overs_before = max((recent_balls - 1) / 6, 1 / 6)
    recent_overs_after = recent_balls / 6
    after["recent_run_rate"] = (
        row["recent_run_rate"] * recent_overs_before + row["runs_total"]
    ) / max(recent_overs_after, 1 / 6)
    after["recent_wickets"] = row["recent_wickets"] + (1 if row["is_wicket"] else 0)

    # Carry forward context features (unchanged within a ball)
    after["venue"] = row["venue"]
    after["batting_team_chose_to_bat"] = row["batting_team_chose_to_bat"]
    after["season_numeric"] = row["season_numeric"]

    return after


# %% Compute win probability deltas
def compute_ball_deltas(
    model: lgb.LGBMClassifier,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute win probability delta for each ball."""
    print("Computing win probability for each ball...")

    # Ensure venue is categorical for LightGBM
    if "venue" in df.columns:
        df["venue"] = df["venue"].astype("category")

    # Win prob BEFORE each ball
    X_before = df[FEATURE_COLS].copy()
    wp_before = model.predict_proba(X_before)[:, 1]

    # Win prob AFTER each ball
    print("Computing post-delivery states...")
    after_states = df.apply(compute_state_after, axis=1, result_type="expand")
    X_after = after_states[FEATURE_COLS].copy()
    # Restore categorical dtype for venue in after states
    if "venue" in X_after.columns:
        X_after["venue"] = X_after["venue"].astype("category")
    wp_after = model.predict_proba(X_after)[:, 1]

    # Delta (from batting team's perspective)
    df = df.copy()
    df["wp_before"] = wp_before
    df["wp_after"] = wp_after
    df["delta_wp"] = wp_after - wp_before

    print(f"  Mean |delta_wp|: {df['delta_wp'].abs().mean():.4f}")
    print(f"  Max delta_wp: {df['delta_wp'].max():.4f}")
    print(f"  Min delta_wp: {df['delta_wp'].min():.4f}")

    return df


# %% Aggregate per player
def aggregate_player_tilt(deltas_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate ball-level deltas into per-player TILT scores."""
    print("\nAggregating per-player TILT scores...")

    # Batting TILT: delta_wp credited to batter (grouped by player ID)
    batting = (
        deltas_df.groupby("batter_id")
        .agg(
            player=("batter", "last"),
            batting_total_tilt=("delta_wp", "sum"),
            batting_balls=("delta_wp", "count"),
            batting_matches=("match_id", "nunique"),
            batting_avg_delta=("delta_wp", "mean"),
        )
        .reset_index()
    )
    batting["batting_tilt_per_match"] = batting["batting_total_tilt"] / batting["batting_matches"]

    # Bowling TILT: -delta_wp credited to bowler (positive = good for bowler)
    deltas_df = deltas_df.copy()
    deltas_df["bowling_delta"] = -deltas_df["delta_wp"]

    bowling = (
        deltas_df.groupby("bowler_id")
        .agg(
            player=("bowler", "last"),
            bowling_total_tilt=("bowling_delta", "sum"),
            bowling_balls=("bowling_delta", "count"),
            bowling_matches=("match_id", "nunique"),
            bowling_avg_delta=("bowling_delta", "mean"),
        )
        .reset_index()
    )
    bowling["bowling_tilt_per_match"] = bowling["bowling_total_tilt"] / bowling["bowling_matches"]

    # Merge batting and bowling on player ID
    batting_merge = batting.rename(columns={"batter_id": "player_id"})
    bowling_merge = bowling.rename(columns={"bowler_id": "player_id"}).drop(columns=["player"])
    combined = pd.merge(batting_merge, bowling_merge, on="player_id", how="outer").fillna(0)

    # For players who only bowled, fill in their name from bowling data
    bowl_only = bowling.rename(columns={"bowler_id": "player_id"})
    combined["player"] = combined["player"].replace(0, np.nan)
    combined = combined.set_index("player_id")
    bowl_only_names = bowl_only.set_index("player_id")["player"]
    combined["player"] = combined["player"].fillna(bowl_only_names)
    combined = combined.reset_index()

    # Total matches (union of batting and bowling appearances)
    batting_matches = deltas_df.groupby("batter_id")["match_id"].apply(set).reset_index()
    batting_matches.columns = ["player_id", "bat_match_set"]
    bowling_matches = deltas_df.groupby("bowler_id")["match_id"].apply(set).reset_index()
    bowling_matches.columns = ["player_id", "bowl_match_set"]

    match_sets = pd.merge(batting_matches, bowling_matches, on="player_id", how="outer")
    match_sets["bat_match_set"] = match_sets["bat_match_set"].apply(lambda x: x if isinstance(x, set) else set())
    match_sets["bowl_match_set"] = match_sets["bowl_match_set"].apply(lambda x: x if isinstance(x, set) else set())
    match_sets["total_matches"] = match_sets.apply(lambda r: len(r["bat_match_set"] | r["bowl_match_set"]), axis=1)

    combined = combined.merge(match_sets[["player_id", "total_matches"]], on="player_id", how="left")

    # Total TILT per match
    combined["total_tilt"] = combined["batting_total_tilt"] + combined["bowling_total_tilt"]
    combined["total_tilt_per_match"] = combined["total_tilt"] / combined["total_matches"]

    # Most recent team
    last_team = (
        deltas_df.sort_values("date")
        .groupby("batter_id")["batting_team"]
        .last()
        .reset_index()
        .rename(columns={"batter_id": "player_id", "batting_team": "team"})
    )
    combined = combined.merge(last_team, on="player_id", how="left")
    combined["team"] = combined["team"].fillna("Unknown")

    # Sort by total TILT per match
    combined = combined.sort_values("total_tilt_per_match", ascending=False).reset_index(drop=True)

    print(f"  Total players: {len(combined)}")
    print(f"\n  Top 10 by Total TILT/Match:")
    top10 = combined.head(10)[["player", "team", "total_tilt_per_match", "batting_tilt_per_match", "bowling_tilt_per_match", "total_matches"]]
    print(top10.to_string(index=False))

    return combined


# %% Slug helper
def make_slug(name: str, player_id: Optional[str] = None) -> str:
    """Convert player name to URL-friendly slug, with optional ID suffix for uniqueness."""
    slug = name.lower().strip()
    slug = re.sub(r"[^a-z0-9\s-]", "", slug)
    slug = re.sub(r"[\s]+", "-", slug)
    if player_id:
        slug = f"{slug}-{player_id}"
    return slug


# %% Main
def compute_tilt(
    featured_balls_path: Optional[str] = None,
    model_path: Optional[str] = None,
) -> tuple:
    """Load model and data, compute TILT for all players."""
    config = load_config()
    processed_dir = Path(config["data"]["processed_dir"])

    featured_path = featured_balls_path or str(processed_dir / config["data"]["featured_balls_file"])
    model_path = model_path or config["model"]["save_path"]

    print(f"Loading model from {model_path}...")
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    print(f"Loading featured balls from {featured_path}...")
    df = pd.read_parquet(featured_path)
    print(f"  Loaded {len(df):,} balls")

    deltas_df = compute_ball_deltas(model, df)
    player_tilt = aggregate_player_tilt(deltas_df)

    return deltas_df, player_tilt


if __name__ == "__main__":
    deltas_df, player_tilt = compute_tilt()
