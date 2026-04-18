# %% Imports
import json
import pickle
import re
from pathlib import Path
from typing import Dict, Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml

from pipeline.build_features import shrunk_run_rate


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
    "over",
    "recent_wickets",
    "venue",
    "batting_team_chose_to_bat",
    "season_numeric",
    "opponent_bowler_economy",
    "batting_team_nrr",
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

    balls_bowled_after = row["ball_number"]  # ball_number is 1-indexed, equals balls bowled after this delivery
    after["run_rate"] = shrunk_run_rate(after["runs_scored"], balls_bowled_after)

    after["over"] = row["over"]

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

    after["recent_wickets"] = row["recent_wickets"] + (1 if row["is_wicket"] else 0)

    # Carry forward context features (unchanged within a ball)
    after["venue"] = row["venue"]
    after["batting_team_chose_to_bat"] = row["batting_team_chose_to_bat"]
    after["season_numeric"] = row["season_numeric"]
    after["opponent_bowler_economy"] = row["opponent_bowler_economy"]
    after["batting_team_nrr"] = row["batting_team_nrr"]

    return after


# %% Compute win probability deltas
def compute_ball_deltas(
    model: lgb.LGBMClassifier,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute win probability delta for each ball."""
    print("Computing win probability for each ball...")

    # Ensure categorical features for LightGBM
    if "venue" in df.columns:
        df["venue"] = df["venue"].astype("category")

    # Win prob BEFORE each ball
    X_before = df[FEATURE_COLS].copy()
    wp_before = model.predict_proba(X_before)[:, 1]

    # Win prob AFTER each ball
    print("Computing post-delivery states...")
    after_states = df.apply(compute_state_after, axis=1, result_type="expand")
    X_after = after_states[FEATURE_COLS].copy()
    # Restore categorical dtypes for after states
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

    # Legal-delivery flags: batting excludes wides; bowling excludes wides + no-balls.
    deltas_df = deltas_df.copy()
    deltas_df["legal_bat"] = (~deltas_df["is_wide"]).astype(int)
    deltas_df["legal_bowl"] = (~deltas_df["is_wide"] & ~deltas_df["is_noball"]).astype(int)

    # Batting TILT: delta_wp credited to batter (grouped by player ID)
    batting = (
        deltas_df.groupby("batter_id")
        .agg(
            player=("batter", "last"),
            batting_total_tilt=("delta_wp", "sum"),
            batting_balls=("legal_bat", "sum"),
            batting_matches=("match_id", "nunique"),
            batting_avg_delta=("delta_wp", "mean"),
        )
        .reset_index()
    )
    batting["batting_tilt_per_match"] = batting["batting_total_tilt"] / batting["batting_matches"]

    # Bowling TILT: -delta_wp credited to bowler (positive = good for bowler)
    deltas_df["bowling_delta"] = -deltas_df["delta_wp"]

    bowling = (
        deltas_df.groupby("bowler_id")
        .agg(
            player=("bowler", "last"),
            bowling_total_tilt=("bowling_delta", "sum"),
            bowling_balls=("legal_bowl", "sum"),
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

    # Collect all teams per player (from both batting and bowling sides)
    sorted_df = deltas_df.sort_values("date")
    bat_teams = (
        sorted_df
        .groupby("batter_id")
        .agg(
            bat_last_team=("batting_team", "last"),
            bat_last_date=("date", "last"),
            all_teams_bat=("batting_team", lambda x: list(x.unique())),
        )
        .reset_index()
        .rename(columns={"batter_id": "player_id"})
    )
    bowl_teams = (
        sorted_df
        .groupby("bowler_id")
        .agg(
            bowl_last_team=("bowling_team", "last"),
            bowl_last_date=("date", "last"),
            all_teams_bowl=("bowling_team", lambda x: list(x.unique())),
        )
        .reset_index()
        .rename(columns={"bowler_id": "player_id"})
    )
    # Merge: union of teams from batting and bowling
    all_teams = pd.merge(bat_teams, bowl_teams, on="player_id", how="outer")
    all_teams["all_teams_bat"] = all_teams["all_teams_bat"].apply(lambda x: x if isinstance(x, list) else [])
    all_teams["all_teams_bowl"] = all_teams["all_teams_bowl"].apply(lambda x: x if isinstance(x, list) else [])
    all_teams["teams"] = all_teams.apply(lambda r: sorted(set(r["all_teams_bat"]) | set(r["all_teams_bowl"])), axis=1)
    # Current team = whichever appearance (batting or bowling) is most recent
    def _pick_current_team(r):
        bat_date = r["bat_last_date"] if pd.notna(r.get("bat_last_date")) else ""
        bowl_date = r["bowl_last_date"] if pd.notna(r.get("bowl_last_date")) else ""
        bat_team = r.get("bat_last_team")
        bowl_team = r.get("bowl_last_team")
        if pd.notna(bat_team) and (str(bat_date) >= str(bowl_date) or pd.isna(bowl_team)):
            return bat_team
        if pd.notna(bowl_team):
            return bowl_team
        return "Unknown"
    all_teams["team"] = all_teams.apply(_pick_current_team, axis=1)

    combined = combined.merge(all_teams[["player_id", "team", "teams"]], on="player_id", how="left")
    combined["team"] = combined["team"].fillna("Unknown")
    combined["teams"] = combined["teams"].apply(lambda x: x if isinstance(x, list) else [])

    # Sort by total TILT per match (before shrinkage)
    combined = combined.sort_values("total_tilt_per_match", ascending=False).reset_index(drop=True)

    print(f"  Total players: {len(combined)}")
    print(f"\n  Top 10 by Raw TILT/Match:")
    top10 = combined.head(10)[["player", "team", "total_tilt_per_match", "batting_tilt_per_match", "bowling_tilt_per_match", "total_matches"]]
    print(top10.to_string(index=False))

    return combined


# %% Bayesian shrinkage
def apply_shrinkage(combined: pd.DataFrame, deltas_df: pd.DataFrame) -> pd.DataFrame:
    """Apply empirical Bayes shrinkage to stabilize small-sample rankings.

    Uses James-Stein / empirical Bayes formula:
        shrunk = (n / (n + k)) * observed + (k / (n + k)) * population_mean
    where k = within_variance / between_variance
    """
    print("\nApplying Bayesian shrinkage...")

    # Compute per-match TILT for each player (need match-level variance)
    # Batting: sum delta_wp per player per match
    bat_match = (
        deltas_df.groupby(["batter_id", "match_id"])["delta_wp"]
        .sum()
        .reset_index()
        .rename(columns={"batter_id": "player_id", "delta_wp": "match_tilt"})
    )
    # Bowling: sum -delta_wp per player per match
    bowl_match = (
        deltas_df.groupby(["bowler_id", "match_id"])
        .agg(match_tilt=("delta_wp", lambda x: -x.sum()))
        .reset_index()
        .rename(columns={"bowler_id": "player_id"})
    )
    # Total: merge batting + bowling per match
    total_match = pd.concat([
        bat_match.assign(role="bat"),
        bowl_match.assign(role="bowl"),
    ]).groupby(["player_id", "match_id"])["match_tilt"].sum().reset_index()

    def _compute_shrinkage(combined_df, col, match_df, match_col="match_tilt"):
        """Compute shrinkage for a single TILT column."""
        # Population mean (across all players)
        pop_mean = combined_df[col].mean()

        # Between-player variance: variance of player-level averages
        between_var = combined_df[col].var()

        # Within-player variance: average of each player's match-to-match variance
        player_var = match_df.groupby("player_id")[match_col].var().dropna()
        within_var = player_var.mean()

        if between_var <= 0 or within_var <= 0:
            print(f"  Warning: cannot compute shrinkage for {col} (between={between_var:.6f}, within={within_var:.6f})")
            combined_df[f"shrunk_{col}"] = combined_df[col]
            return combined_df, 0

        k = within_var / between_var
        print(f"  {col}: k={k:.1f} (within_var={within_var:.6f}, between_var={between_var:.6f}, pop_mean={pop_mean:.6f})")

        # Apply shrinkage
        n = combined_df["total_matches"]
        combined_df[f"shrunk_{col}"] = (n / (n + k)) * combined_df[col] + (k / (n + k)) * pop_mean

        return combined_df, k

    # Apply to each TILT column
    combined, k_total = _compute_shrinkage(combined, "total_tilt_per_match", total_match)

    # For batting/bowling shrinkage, use batting/bowling specific match data
    bat_player_match = bat_match.copy()
    combined, k_bat = _compute_shrinkage(combined, "batting_tilt_per_match", bat_player_match)
    bowl_player_match = bowl_match.copy()
    combined, k_bowl = _compute_shrinkage(combined, "bowling_tilt_per_match", bowl_player_match)

    # Bayesian posterior confidence intervals
    # The shrinkage formula is: shrunk = (n/(n+k)) * observed + (k/(n+k)) * pop_mean
    # The posterior variance is: var_posterior = within_var / (n + k)
    # This is narrower than the frequentist SE^2 = within_var / n because
    # shrinkage itself reduces uncertainty (we're borrowing strength from the population)
    player_std = total_match.groupby("player_id")["match_tilt"].std().fillna(0)
    combined = combined.merge(
        player_std.rename("match_tilt_std").reset_index(),
        on="player_id",
        how="left",
    )
    combined["match_tilt_std"] = combined["match_tilt_std"].fillna(0)
    n = combined["total_matches"]
    # Use within_var (average match-to-match variance) for posterior, not per-player std
    # This is more stable and consistent with the shrinkage model
    player_var = total_match.groupby("player_id")["match_tilt"].var().fillna(0)
    combined = combined.merge(
        player_var.rename("match_tilt_var").reset_index(),
        on="player_id",
        how="left",
    )
    combined["match_tilt_var"] = combined["match_tilt_var"].fillna(0)
    # Posterior SE: sqrt(within_var / (n + k)) — accounts for shrinkage reducing uncertainty
    posterior_se = np.sqrt(combined["match_tilt_var"] / (n + k_total).clip(lower=1))
    combined["tilt_ci_lower"] = combined["shrunk_total_tilt_per_match"] - 1.96 * posterior_se
    combined["tilt_ci_upper"] = combined["shrunk_total_tilt_per_match"] + 1.96 * posterior_se
    combined["tilt_ci_lower_90"] = combined["shrunk_total_tilt_per_match"] - 1.645 * posterior_se
    combined["tilt_ci_upper_90"] = combined["shrunk_total_tilt_per_match"] + 1.645 * posterior_se

    # Confidence level
    combined["confidence"] = "low"
    combined.loc[combined["total_matches"] >= 30, "confidence"] = "medium"
    combined.loc[combined["total_matches"] >= 100, "confidence"] = "high"

    # Re-sort by 90% CI lower bound (penalizes small samples naturally)
    combined = combined.sort_values("tilt_ci_lower_90", ascending=False).reset_index(drop=True)

    print(f"\n  Top 10 by TILT Floor (90% CI Lower Bound):")
    top10 = combined.head(10)[["player", "team", "shrunk_total_tilt_per_match", "tilt_ci_lower_90", "total_matches", "confidence"]]
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
    player_tilt = apply_shrinkage(player_tilt, deltas_df)

    # Merge full names from cached Wikidata resolution
    full_names_path = processed_dir / "full_names.json"
    if full_names_path.exists():
        with open(full_names_path, "r") as f:
            full_names: Dict[str, str] = json.load(f)
        player_tilt["full_name"] = player_tilt["player_id"].map(full_names)
        player_tilt["full_name"] = player_tilt["full_name"].fillna(player_tilt["player"])
        resolved = player_tilt["full_name"].ne(player_tilt["player"]).sum()
        print(f"\n  Full names resolved: {resolved} / {len(player_tilt)}")
    else:
        player_tilt["full_name"] = player_tilt["player"]
        print("\n  No full_names.json found, using display names")

    return deltas_df, player_tilt


if __name__ == "__main__":
    deltas_df, player_tilt = compute_tilt()
