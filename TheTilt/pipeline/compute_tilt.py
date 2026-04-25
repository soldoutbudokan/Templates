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


# %% Bayesian shrinkage helper (module-level so other aggregators can call it)
def _compute_shrinkage(
    combined_df: pd.DataFrame,
    col: str,
    match_df: pd.DataFrame,
    *,
    match_col: str = "match_tilt",
    n_col: str = "total_matches",
    group_keys: Optional[list] = None,
) -> tuple:
    """Compute James-Stein shrinkage for a single TILT column.

    `group_keys` defaults to ["player_id"]; player-season uses
    ["player_id", "season"]. Variance is computed within each row's natural
    group (its match-by-match TILT for that group).
    """
    if group_keys is None:
        group_keys = ["player_id"]

    pop_mean = combined_df[col].mean()
    between_var = combined_df[col].var()
    player_var = match_df.groupby(group_keys)[match_col].var().dropna()
    within_var = player_var.mean()

    if between_var is None or pd.isna(between_var) or between_var <= 0 or pd.isna(within_var) or within_var <= 0:
        print(f"  Warning: cannot compute shrinkage for {col} (between={between_var}, within={within_var})")
        combined_df[f"shrunk_{col}"] = combined_df[col]
        return combined_df, 0

    k = within_var / between_var
    print(f"  {col}: k={k:.1f} (within_var={within_var:.6f}, between_var={between_var:.6f}, pop_mean={pop_mean:.6f})")

    n = combined_df[n_col]
    combined_df[f"shrunk_{col}"] = (n / (n + k)) * combined_df[col] + (k / (n + k)) * pop_mean
    return combined_df, k


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


# %% Slug helpers
def make_slug(name: str, player_id: Optional[str] = None) -> str:
    """Convert player name to URL-friendly slug, with optional ID suffix for uniqueness."""
    slug = name.lower().strip()
    slug = re.sub(r"[^a-z0-9\s-]", "", slug)
    slug = re.sub(r"[\s]+", "-", slug)
    if player_id:
        slug = f"{slug}-{player_id}"
    return slug


def make_team_slug(canonical: str) -> str:
    """Look up the canonical team's slug from the YAML alias map."""
    from pipeline.parse_matches import TEAM_SLUG
    return TEAM_SLUG.get(canonical, re.sub(r"[^a-z0-9-]", "-", canonical.lower()))


# %% Helper: legal-delivery flags (mirrors export_json._add_legal_flags but local)
def _ensure_legal_flags(df: pd.DataFrame) -> pd.DataFrame:
    if "legal_bat" not in df.columns:
        df = df.copy()
        df["legal_bat"] = (~df["is_wide"]).astype(int)
    if "legal_bowl" not in df.columns:
        df["legal_bowl"] = (~df["is_wide"] & ~df["is_noball"]).astype(int)
    return df


def _season_year(season: str) -> int:
    s = str(season)
    return int(s.split("/")[0]) if "/" in s else int(s)


# %% Aggregate per (player, season)
def aggregate_player_season_tilt(deltas_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate ball-level deltas into per-player-per-season TILT.

    Mirrors `aggregate_player_tilt` but adds `season` to the groupby keys.
    Applies shrinkage at the player-season level (small-sample seasons get
    pulled toward the global player-season mean).
    """
    print("\nAggregating per-player-per-season TILT scores...")
    deltas_df = _ensure_legal_flags(deltas_df)
    deltas_df = deltas_df.copy()
    deltas_df["bowling_delta"] = -deltas_df["delta_wp"]

    # Batting per (player, season)
    bat = (
        deltas_df.groupby(["batter_id", "season"])
        .agg(
            player=("batter", "last"),
            batting_total_tilt=("delta_wp", "sum"),
            batting_balls=("legal_bat", "sum"),
            batting_matches=("match_id", "nunique"),
            batting_team=("batting_team", lambda s: s.mode().iloc[0] if len(s) else None),
        )
        .reset_index()
        .rename(columns={"batter_id": "player_id"})
    )
    bat["batting_tilt_per_match"] = bat["batting_total_tilt"] / bat["batting_matches"].replace(0, 1)

    # Bowling per (player, season)
    bowl = (
        deltas_df.groupby(["bowler_id", "season"])
        .agg(
            player=("bowler", "last"),
            bowling_total_tilt=("bowling_delta", "sum"),
            bowling_balls=("legal_bowl", "sum"),
            bowling_matches=("match_id", "nunique"),
            bowling_team=("bowling_team", lambda s: s.mode().iloc[0] if len(s) else None),
        )
        .reset_index()
        .rename(columns={"bowler_id": "player_id"})
    )
    bowl["bowling_tilt_per_match"] = bowl["bowling_total_tilt"] / bowl["bowling_matches"].replace(0, 1)

    # Outer-merge on (player_id, season)
    bowl_no_player = bowl.drop(columns=["player"])
    combined = pd.merge(bat, bowl_no_player, on=["player_id", "season"], how="outer")
    # Fill name from bowling-only rows
    bowl_only_names = bowl.set_index(["player_id", "season"])["player"]
    combined = combined.set_index(["player_id", "season"])
    combined["player"] = combined["player"].fillna(bowl_only_names)
    combined = combined.reset_index()

    numeric_fill = [
        "batting_total_tilt", "batting_balls", "batting_matches", "batting_tilt_per_match",
        "bowling_total_tilt", "bowling_balls", "bowling_matches", "bowling_tilt_per_match",
    ]
    for col in numeric_fill:
        if col in combined.columns:
            combined[col] = combined[col].fillna(0)

    # Total matches: union of batting and bowling appearances per (player, season)
    bat_match_sets = deltas_df.groupby(["batter_id", "season"])["match_id"].apply(set).reset_index()
    bat_match_sets.columns = ["player_id", "season", "bat_match_set"]
    bowl_match_sets = deltas_df.groupby(["bowler_id", "season"])["match_id"].apply(set).reset_index()
    bowl_match_sets.columns = ["player_id", "season", "bowl_match_set"]
    match_sets = pd.merge(bat_match_sets, bowl_match_sets, on=["player_id", "season"], how="outer")
    match_sets["bat_match_set"] = match_sets["bat_match_set"].apply(lambda x: x if isinstance(x, set) else set())
    match_sets["bowl_match_set"] = match_sets["bowl_match_set"].apply(lambda x: x if isinstance(x, set) else set())
    match_sets["total_matches"] = match_sets.apply(lambda r: len(r["bat_match_set"] | r["bowl_match_set"]), axis=1)
    combined = combined.merge(match_sets[["player_id", "season", "total_matches"]], on=["player_id", "season"], how="left")
    combined["total_matches"] = combined["total_matches"].fillna(0).astype(int)

    combined["total_tilt"] = combined["batting_total_tilt"].fillna(0) + combined["bowling_total_tilt"].fillna(0)
    combined["total_tilt_per_match"] = combined["total_tilt"] / combined["total_matches"].replace(0, 1)

    # Season-team: prefer batting_team (more reliable signal); fall back to bowling_team
    combined["team"] = combined["batting_team"].fillna(combined["bowling_team"])
    # Season teams list — collect any teams the player appeared for that season
    season_team_lookup = (
        pd.concat([
            deltas_df[["batter_id", "season", "batting_team"]].rename(columns={"batter_id": "player_id", "batting_team": "team"}),
            deltas_df[["bowler_id", "season", "bowling_team"]].rename(columns={"bowler_id": "player_id", "bowling_team": "team"}),
        ])
        .dropna(subset=["player_id"])
        .groupby(["player_id", "season"])["team"]
        .apply(lambda s: sorted(set(s)))
        .reset_index()
        .rename(columns={"team": "season_teams"})
    )
    combined = combined.merge(season_team_lookup, on=["player_id", "season"], how="left")

    # Apply shrinkage (per-season aggregate; group_keys = ["player_id", "season"])
    print("  Applying player-season shrinkage...")
    bat_match_df = (
        deltas_df.groupby(["batter_id", "season", "match_id"])["delta_wp"]
        .sum()
        .reset_index()
        .rename(columns={"batter_id": "player_id", "delta_wp": "match_tilt"})
    )
    bowl_match_df = (
        deltas_df.groupby(["bowler_id", "season", "match_id"])
        .agg(match_tilt=("delta_wp", lambda x: -x.sum()))
        .reset_index()
        .rename(columns={"bowler_id": "player_id"})
    )
    total_match_df = (
        pd.concat([bat_match_df.assign(role="bat"), bowl_match_df.assign(role="bowl")])
        .groupby(["player_id", "season", "match_id"])["match_tilt"]
        .sum()
        .reset_index()
    )
    combined, _ = _compute_shrinkage(combined, "total_tilt_per_match", total_match_df, group_keys=["player_id", "season"])
    combined, _ = _compute_shrinkage(combined, "batting_tilt_per_match", bat_match_df, group_keys=["player_id", "season"])
    combined, _ = _compute_shrinkage(combined, "bowling_tilt_per_match", bowl_match_df, group_keys=["player_id", "season"])

    combined = combined.sort_values(["season", "shrunk_total_tilt_per_match"], ascending=[True, False]).reset_index(drop=True)
    print(f"  Total player-seasons: {len(combined)}")
    return combined


# %% Aggregate per team (career)
def aggregate_team_tilt(deltas_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per canonical team (across all seasons).

    `team_total_tilt = sum(delta_wp where batting_team == T) + sum(-delta_wp where bowling_team == T)`.
    No shrinkage — teams accumulate hundreds of matches.
    """
    print("\nAggregating per-team TILT scores...")
    deltas_df = _ensure_legal_flags(deltas_df)

    bat = (
        deltas_df.groupby("batting_team")
        .agg(batting_total_tilt=("delta_wp", "sum"), batting_balls=("legal_bat", "sum"))
        .reset_index()
        .rename(columns={"batting_team": "team"})
    )
    bowl = (
        deltas_df.groupby("bowling_team")
        .agg(bowling_total_tilt=("delta_wp", lambda x: -x.sum()), bowling_balls=("legal_bowl", "sum"))
        .reset_index()
        .rename(columns={"bowling_team": "team"})
    )

    # Match-level appearances: a team appeared if it batted OR bowled
    teams_per_match = deltas_df.groupby("match_id")[["batting_team", "bowling_team", "winner"]].first().reset_index()
    appearances = pd.concat([
        teams_per_match[["match_id", "batting_team"]].rename(columns={"batting_team": "team"}),
        teams_per_match[["match_id", "bowling_team"]].rename(columns={"bowling_team": "team"}),
    ]).dropna().drop_duplicates()
    matches = appearances.groupby("team").size().reset_index(name="matches")

    wins = (
        teams_per_match.dropna(subset=["winner"]).groupby("winner").size().reset_index(name="wins")
        .rename(columns={"winner": "team"})
    )

    out = matches.merge(bat, on="team", how="left").merge(bowl, on="team", how="left").merge(wins, on="team", how="left")
    for col in ["batting_total_tilt", "batting_balls", "bowling_total_tilt", "bowling_balls", "wins"]:
        if col in out.columns:
            out[col] = out[col].fillna(0)
    out["wins"] = out["wins"].astype(int)
    out["losses"] = (out["matches"] - out["wins"]).astype(int)
    out["win_pct"] = (out["wins"] / out["matches"].replace(0, 1)).round(4)
    out["team_total_tilt"] = out["batting_total_tilt"] + out["bowling_total_tilt"]
    out["team_tilt_per_match"] = out["team_total_tilt"] / out["matches"].replace(0, 1)
    out["batting_tilt_per_match"] = out["batting_total_tilt"] / out["matches"].replace(0, 1)
    out["bowling_tilt_per_match"] = out["bowling_total_tilt"] / out["matches"].replace(0, 1)

    # First/last active season (in calendar-year terms)
    season_year = deltas_df["season"].apply(_season_year)
    bat_seasons = (
        deltas_df.assign(season_year=season_year)
        .groupby("batting_team")["season_year"].agg(["min", "max"]).reset_index()
        .rename(columns={"batting_team": "team", "min": "bat_first", "max": "bat_last"})
    )
    bowl_seasons = (
        deltas_df.assign(season_year=season_year)
        .groupby("bowling_team")["season_year"].agg(["min", "max"]).reset_index()
        .rename(columns={"bowling_team": "team", "min": "bowl_first", "max": "bowl_last"})
    )
    out = out.merge(bat_seasons, on="team", how="left").merge(bowl_seasons, on="team", how="left")
    out["first_season"] = out[["bat_first", "bowl_first"]].min(axis=1).fillna(0).astype(int)
    out["last_season"] = out[["bat_last", "bowl_last"]].max(axis=1).fillna(0).astype(int)
    out = out.drop(columns=["bat_first", "bat_last", "bowl_first", "bowl_last"])

    out = out.sort_values("team_tilt_per_match", ascending=False).reset_index(drop=True)
    print(f"  Total teams: {len(out)}")
    print(out[["team", "matches", "wins", "win_pct", "team_tilt_per_match", "first_season", "last_season"]].to_string(index=False))
    return out


# %% Aggregate per (team, season)
def aggregate_team_season_tilt(deltas_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per (team, season). Same metrics as team aggregate plus
    a position estimate (rank by win_pct in the season)."""
    print("\nAggregating per-team-per-season TILT scores...")
    deltas_df = _ensure_legal_flags(deltas_df)

    # Match-level info per (team, season): for wins/losses, season per match
    match_first = deltas_df.groupby("match_id").first()[["season", "winner", "batting_team", "bowling_team"]].reset_index()
    appearances = pd.concat([
        match_first[["match_id", "season", "batting_team"]].rename(columns={"batting_team": "team"}),
        match_first[["match_id", "season", "bowling_team"]].rename(columns={"bowling_team": "team"}),
    ]).dropna().drop_duplicates()
    matches = appearances.groupby(["team", "season"]).size().reset_index(name="matches")

    wins = (
        match_first.dropna(subset=["winner"]).groupby(["winner", "season"]).size().reset_index(name="wins")
        .rename(columns={"winner": "team"})
    )

    bat = (
        deltas_df.groupby(["batting_team", "season"])
        .agg(batting_total_tilt=("delta_wp", "sum"), batting_balls=("legal_bat", "sum"))
        .reset_index()
        .rename(columns={"batting_team": "team"})
    )
    bowl = (
        deltas_df.groupby(["bowling_team", "season"])
        .agg(bowling_total_tilt=("delta_wp", lambda x: -x.sum()), bowling_balls=("legal_bowl", "sum"))
        .reset_index()
        .rename(columns={"bowling_team": "team"})
    )

    out = (
        matches
        .merge(wins, on=["team", "season"], how="left")
        .merge(bat, on=["team", "season"], how="left")
        .merge(bowl, on=["team", "season"], how="left")
    )
    for col in ["wins", "batting_total_tilt", "batting_balls", "bowling_total_tilt", "bowling_balls"]:
        if col in out.columns:
            out[col] = out[col].fillna(0)
    out["wins"] = out["wins"].astype(int)
    out["losses"] = (out["matches"] - out["wins"]).astype(int)
    out["win_pct"] = (out["wins"] / out["matches"].replace(0, 1)).round(4)
    out["team_total_tilt"] = out["batting_total_tilt"] + out["bowling_total_tilt"]
    out["team_tilt_per_match"] = out["team_total_tilt"] / out["matches"].replace(0, 1)

    # Position estimate: rank within season by win_pct (1 = best)
    out["position_est"] = (
        out.groupby("season")["win_pct"]
        .rank(method="min", ascending=False)
        .astype(int)
    )

    out = out.sort_values(["season", "win_pct"], ascending=[True, False]).reset_index(drop=True)
    print(f"  Total team-seasons: {len(out)}")
    return out


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
