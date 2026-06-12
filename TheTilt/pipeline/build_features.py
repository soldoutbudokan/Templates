# %% Imports
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml


# %% Configuration
# Shrinkage prior for `run_rate`. The raw formula runs/overs explodes at tiny
# denominators (e.g. a 4-run ball 1 reads as 24 rpo), which shows up in the
# trained model as implausible early-innings WP swings. Shrinking toward the
# league-average 1st-innings rpo with ~3 overs of prior weight keeps the
# feature in-distribution early and drowns in real data after a few overs.
PRIOR_BALLS = 18
PRIOR_RPO = 8.4
PRIOR_RUNS = PRIOR_BALLS * PRIOR_RPO / 6  # 25.2

# Shrinkage prior for the as-of-date opponent_bowler_economy (issue #199):
# a debut bowler starts at the league-average rpo and earns their own
# economy as prior legal balls accumulate — ~10 overs of prior weight.
ECON_PRIOR_BALLS = 60


def shrunk_run_rate(runs_scored: float, balls_bowled: float) -> float:
    """Empirical-Bayes shrunk run rate. Mirrors compute_tilt.compute_state_after."""
    return (runs_scored + PRIOR_RUNS) / ((balls_bowled + PRIOR_BALLS) / 6)


def load_config(config_path: str = "config/pipeline_config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# %% Build features for a single match-innings
def build_innings_features(innings_df: pd.DataFrame, target: Optional[int] = None) -> pd.DataFrame:
    """Add match state features to each ball in an innings.

    Features represent the state BEFORE the delivery is bowled.
    """
    df = innings_df.copy().reset_index(drop=True)
    n = len(df)

    # Cumulative runs and wickets (BEFORE each ball). Every dismissal on a
    # delivery counts — cricsheet can record two on one ball (issue #194);
    # is_wicket alone would understate wickets_fallen for the innings tail.
    df["runs_scored"] = df["runs_total"].cumsum().shift(1, fill_value=0)
    wicket_counts = df["wicket_count"] if "wicket_count" in df.columns else df["is_wicket"].astype(int)
    df["wickets_fallen"] = wicket_counts.cumsum().shift(1, fill_value=0)

    # Ball number in innings (1-indexed, counts every delivery incl. wides/
    # no-balls) — the within-innings ordering key.
    df["ball_number"] = range(1, n + 1)

    # Balls remaining. Default T20 = 120 deliveries; DLS-revised innings carry
    # their reduced allocation through (a 12-over chase is 72 balls, not 120),
    # so revised chases aren't scored as if phantom deliveries remained
    # (issue #192 — re-landed from the reverted #111 bundle; DLS matches are
    # excluded from training, so this only affects scoring).
    total_balls = 120
    if "innings_allocation" in df.columns:
        alloc = df["innings_allocation"].iloc[0]
        if pd.notna(alloc) and alloc > 0:
            total_balls = int(round(float(alloc) * 6))
    # A wide/no-ball doesn't consume one of the innings' deliveries, so
    # balls_bowled counts only LEGAL deliveries before this ball (the other
    # half of issue #192 — the old ball_number-based count drifted
    # balls_remaining low by the innings' running extras count). Mirrored in
    # compute_tilt.compute_state_after.
    legal = (~df["is_wide"].astype(bool) & ~df["is_noball"].astype(bool)).astype(int)
    df["balls_bowled"] = legal.cumsum().shift(1, fill_value=0)
    df["balls_remaining"] = total_balls - df["balls_bowled"]
    df["balls_remaining"] = df["balls_remaining"].clip(lower=1)

    # Wickets in hand
    df["wickets_in_hand"] = 10 - df["wickets_fallen"]

    # Run rate (shrunk toward league-average prior — see PRIOR_* constants).
    df["run_rate"] = shrunk_run_rate(df["runs_scored"], df["balls_bowled"])

    # Phase flags (over is 0-indexed: 0 = first over, 19 = last over)
    df["is_powerplay"] = (df["over"] <= 5).astype(int)  # Overs 0-5 (cricket overs 1-6)
    df["is_middle"] = ((df["over"] >= 6) & (df["over"] <= 14)).astype(int)  # Overs 6-14 (cricket overs 7-15)
    df["is_death"] = (df["over"] >= 15).astype(int)  # Overs 15-19 (cricket overs 16-20)

    # Target and chase features (innings 2 only)
    if target is not None and target > 0:
        df["target"] = target
        df["runs_needed"] = target - df["runs_scored"]
        df["required_run_rate"] = (df["runs_needed"] / (df["balls_remaining"] / 6)).clip(upper=36)
    else:
        df["target"] = 0
        df["runs_needed"] = 0
        df["required_run_rate"] = 0.0

    # Recent form: wickets in last 18 balls (3 overs)
    window = 18
    df["recent_wickets"] = wicket_counts.rolling(window=window, min_periods=1).sum().shift(1, fill_value=0)

    return df


# %% Build features for all matches
def build_all_features(ball_events_path: Optional[str] = None) -> pd.DataFrame:
    """Load ball events and add match state features."""
    config = load_config()
    processed_dir = Path(config["data"]["processed_dir"])

    if ball_events_path is None:
        ball_events_path = processed_dir / config["data"]["ball_events_file"]

    print(f"Loading ball events from {ball_events_path}...")
    df = pd.read_parquet(ball_events_path)
    print(f"  Loaded {len(df):,} balls from {df['match_id'].nunique()} matches")

    # Compute first innings totals for each match (needed for innings 2 target)
    first_innings_totals = (
        df[df["innings"] == 1]
        .groupby("match_id")["runs_total"]
        .sum()
        .to_dict()
    )

    # Merge target into dataframe for vectorized processing
    df["_target"] = df.apply(
        lambda r: first_innings_totals.get(r["match_id"], 0) + 1 if r["innings"] == 2 else 0,
        axis=1,
    )

    # Process all match-innings in one vectorized groupby
    def _build_group(group):
        target = group["_target"].iloc[0] if group["_target"].iloc[0] > 0 else None
        return build_innings_features(group, target=target)

    print("  Building features per match-innings...")
    result = df.groupby(["match_id", "innings"], group_keys=False).apply(_build_group)
    result = result.reset_index(drop=True)
    result = result.drop(columns=["_target"], errors="ignore")

    # Add binary target: did the batting team win this match?
    result["batting_team_won"] = (result["batting_team"] == result["winner"]).astype(int)

    # Deduplicate venue names (same ground, different strings)
    venue_map = {
        "Wankhede Stadium, Mumbai": "Wankhede Stadium",
        "M Chinnaswamy Stadium, Bengaluru": "M Chinnaswamy Stadium",
        "M.Chinnaswamy Stadium": "M Chinnaswamy Stadium",
        "Feroz Shah Kotla": "Arun Jaitley Stadium",
        "Arun Jaitley Stadium, Delhi": "Arun Jaitley Stadium",
        "MA Chidambaram Stadium, Chepauk, Chennai": "MA Chidambaram Stadium, Chepauk",
        "MA Chidambaram Stadium": "MA Chidambaram Stadium, Chepauk",
        "Eden Gardens, Kolkata": "Eden Gardens",
        "Rajiv Gandhi International Stadium, Uppal, Hyderabad": "Rajiv Gandhi International Stadium, Uppal",
        "Rajiv Gandhi International Stadium": "Rajiv Gandhi International Stadium, Uppal",
        "Sawai Mansingh Stadium, Jaipur": "Sawai Mansingh Stadium",
        "Dr DY Patil Sports Academy, Mumbai": "Dr DY Patil Sports Academy",
        "Maharashtra Cricket Association Stadium, Pune": "Maharashtra Cricket Association Stadium",
        "Brabourne Stadium, Mumbai": "Brabourne Stadium",
        "Punjab Cricket Association Stadium, Mohali": "Punjab Cricket Association IS Bindra Stadium, Mohali",
        "Punjab Cricket Association IS Bindra Stadium": "Punjab Cricket Association IS Bindra Stadium, Mohali",
        "Punjab Cricket Association IS Bindra Stadium, Mohali, Chandigarh": "Punjab Cricket Association IS Bindra Stadium, Mohali",
        "Sardar Patel Stadium, Motera": "Narendra Modi Stadium, Ahmedabad",
        "Himachal Pradesh Cricket Association Stadium, Dharamsala": "Himachal Pradesh Cricket Association Stadium",
        "Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium, Visakhapatnam": "Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium",
        "Maharaja Yadavindra Singh International Cricket Stadium, New Chandigarh": "Maharaja Yadavindra Singh International Cricket Stadium, Mullanpur",
    }
    result["venue"] = result["venue"].replace(venue_map)
    print(f"  Venues after deduplication: {result['venue'].nunique()}")

    # Categorical features (LightGBM handles natively)
    result["venue"] = result["venue"].astype("category")
    # Over as continuous integer (0-19) — gives LightGBM more granular split points
    # than the 3 coarse phase dummies
    result["over"] = result["over"].astype(int)

    # Toss-derived feature: did the batting team choose to bat?
    if "toss_winner" in result.columns and "toss_decision" in result.columns:
        result["batting_team_chose_to_bat"] = (
            (result["batting_team"] == result["toss_winner"]) &
            (result["toss_decision"] == "bat")
        ).astype(int)
    else:
        result["batting_team_chose_to_bat"] = 0

    # Era feature: numeric season year
    result["season_numeric"] = result["season"].apply(
        lambda s: int(s.split("/")[0]) if "/" in str(s) else int(s) if str(s).isdigit() else 2008
    )

    # Opponent quality proxy: bowler economy as an expanding as-of-date mean
    # (issue #199 — replaces the full-career lookup whose temporal lookahead
    # let a 2008 delivery see the bowler's whole career). Each delivery sees
    # the bowler's runs conceded per 6 legal balls across PRIOR matches only,
    # shrunk toward the PRIOR_RPO league prior with ECON_PRIOR_BALLS of
    # weight so debutants and short histories stay in-distribution. Constant
    # within a (bowler, match), so post-ball states carry it unchanged.
    print("  Computing as-of-date opponent bowler economy...")
    _legal_ball = (~result["is_wide"].astype(bool) & ~result["is_noball"].astype(bool)).astype(int)
    bowler_match = (
        result.assign(_legal=_legal_ball)
        .groupby(["bowler_id", "match_id", "date"])
        .agg(runs=("runs_total", "sum"), balls=("_legal", "sum"))
        .reset_index()
        .sort_values(["bowler_id", "date", "match_id"])
    )
    _g = bowler_match.groupby("bowler_id")
    prior_runs = _g["runs"].cumsum() - bowler_match["runs"]
    prior_balls = _g["balls"].cumsum() - bowler_match["balls"]
    bowler_match["econ_asof"] = (prior_runs + ECON_PRIOR_BALLS * PRIOR_RPO / 6) / (
        (prior_balls + ECON_PRIOR_BALLS) / 6
    )
    result = result.merge(
        bowler_match[["bowler_id", "match_id", "econ_asof"]],
        on=["bowler_id", "match_id"],
        how="left",
    )
    result["opponent_bowler_economy"] = result.pop("econ_asof")
    print(f"  Opponent bowler economy: mean={result['opponent_bowler_economy'].mean():.2f}, std={result['opponent_bowler_economy'].std():.2f}")

    # Team strength proxy: rolling season NRR (net run rate) from prior matches
    # NRR = (runs scored per over) - (runs conceded per over) across season so far
    # Computed per team per season using only matches BEFORE the current one (no leakage)
    print("  Computing team season NRR...")
    match_summary = (
        result.groupby(["match_id", "date", "season", "innings", "batting_team", "bowling_team"])
        .agg(runs=("runs_total", "sum"), balls=("runs_total", "count"))
        .reset_index()
    )

    # Each match has two innings: team bats in one, bowls in the other
    # For batting: runs scored and overs faced
    # For bowling: runs conceded and overs bowled
    bat_side = match_summary.rename(columns={
        "batting_team": "team", "runs": "runs_scored", "balls": "balls_faced"
    })[["match_id", "date", "season", "team", "runs_scored", "balls_faced"]]

    bowl_side = match_summary.rename(columns={
        "bowling_team": "team", "runs": "runs_conceded", "balls": "balls_bowled"
    })[["match_id", "date", "season", "team", "runs_conceded", "balls_bowled"]]

    team_match = bat_side.merge(bowl_side, on=["match_id", "season", "team"], suffixes=("", "_bowl"))
    team_match = team_match.sort_values(["team", "date_bowl"]).rename(columns={"date": "date"})

    # Rolling cumulative NRR from prior matches (exclude current match)
    team_match = team_match.sort_values(["season", "team", "date"])
    team_nrr = {}
    for (season, team), group in team_match.groupby(["season", "team"]):
        cum_scored = group["runs_scored"].cumsum().shift(1, fill_value=0)
        cum_faced = group["balls_faced"].cumsum().shift(1, fill_value=0)
        cum_conceded = group["runs_conceded"].cumsum().shift(1, fill_value=0)
        cum_bowled = group["balls_bowled"].cumsum().shift(1, fill_value=0)
        overs_faced = (cum_faced / 6).clip(lower=1)
        overs_bowled = (cum_bowled / 6).clip(lower=1)
        nrr = (cum_scored / overs_faced) - (cum_conceded / overs_bowled)
        for match_id, val in zip(group["match_id"], nrr):
            team_nrr[(match_id, team)] = val

    result["batting_team_nrr"] = result.apply(
        lambda r: team_nrr.get((r["match_id"], r["batting_team"]), 0.0), axis=1
    )
    print(f"  Team NRR: mean={result['batting_team_nrr'].mean():.2f}, std={result['batting_team_nrr'].std():.2f}")

    # Select feature columns + identifiers
    feature_cols = [
        # Identifiers
        "match_id", "date", "venue", "season",
        "batting_team", "bowling_team", "winner",
        "innings", "over", "ball",
        "batter", "bowler", "non_striker",
        "batter_id", "bowler_id", "non_striker_id",
        # Raw event data
        "runs_batter", "runs_extras", "runs_total",
        "is_wide", "is_noball",
        "is_wicket", "wicket_count", "wicket_kind", "player_dismissed", "player_dismissed_id", "wicket_fielders",
        # Features (state before delivery)
        "ball_number", "balls_bowled", "balls_remaining", "wickets_in_hand",
        "runs_scored", "wickets_fallen", "run_rate",
        "is_powerplay", "is_middle", "is_death",
        "target", "runs_needed", "required_run_rate",
        "recent_wickets",
        # New features (venue, toss, era, opponent quality, team strength)
        "batting_team_chose_to_bat", "season_numeric",
        "opponent_bowler_economy", "batting_team_nrr",
        # Target
        "batting_team_won",
    ]

    # Include DLS/impact sub metadata if available (for filtering, not as model features)
    for optional_col in [
        "dls_method", "is_impact_sub_match", "toss_winner", "toss_decision",
        "event_stage", "event_match_number",
        # Per-innings overs allocation, used by NRR (#107). Default 20 for
        # IPL T20; DLS-revised on truncated chases / rain-cut innings.
        "innings_allocation",
    ]:
        if optional_col in result.columns:
            feature_cols.append(optional_col)

    result = result[feature_cols]

    # Save
    output_path = processed_dir / config["data"]["featured_balls_file"]
    result.to_parquet(output_path, index=False)
    print(f"\n  Saved {len(result):,} featured balls to {output_path}")
    print(f"  Win rate (batting team): {result['batting_team_won'].mean():.3f}")

    return result


# %% Main
if __name__ == "__main__":
    df = build_all_features()
    print(f"\nShape: {df.shape}")
    print(f"\nFeature stats:")
    print(df[["balls_remaining", "wickets_in_hand", "runs_scored", "run_rate",
              "required_run_rate", "batting_team_nrr"]].describe())
