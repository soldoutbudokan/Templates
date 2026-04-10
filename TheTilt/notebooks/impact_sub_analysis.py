"""
Impact Substitute Analysis for IPL Cricket

Analyzes the effect of the Impact Sub rule (introduced IPL 2023) on match
volatility and team usage patterns using ball-by-ball win probability data.
"""

# %% Imports and setup
import pandas as pd
import numpy as np
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "deltas.parquet"

df = pd.read_parquet(DATA_PATH)
print(f"Loaded {len(df):,} balls across {df['match_id'].nunique()} matches")
print(f"Seasons: {sorted(df['season'].unique())}")
print(f"Columns: {list(df.columns)}")

# %% 1. Count matches with impact subs by season
print("\n" + "=" * 60)
print("1. MATCHES WITH IMPACT SUBS BY SEASON")
print("=" * 60)

match_info = df.groupby("match_id").agg(
    season=("season", "first"),
    is_impact_sub=("is_impact_sub_match", "first"),
).reset_index()

season_counts = (
    match_info
    .groupby(["season", "is_impact_sub"])
    .size()
    .unstack(fill_value=0)
    .rename(columns={True: "Impact Sub", False: "No Impact Sub"})
)

# Add total and percentage columns
season_counts["Total"] = season_counts.sum(axis=1)
if "Impact Sub" in season_counts.columns:
    season_counts["% Impact Sub"] = (
        season_counts["Impact Sub"] / season_counts["Total"] * 100
    ).round(1)

print(season_counts.to_string())
print(f"\nNote: Impact subs were introduced in IPL 2023.")

# %% 2. Average |delta_wp| per ball: impact sub vs non-impact sub matches
print("\n" + "=" * 60)
print("2. MATCH VOLATILITY: AVG |delta_wp| PER BALL")
print("=" * 60)

df["abs_delta_wp"] = df["delta_wp"].abs()

volatility = (
    df.groupby("is_impact_sub_match")["abs_delta_wp"]
    .agg(["mean", "median", "std", "count"])
    .rename(index={True: "Impact Sub", False: "No Impact Sub"})
)
volatility.columns = ["Mean |delta_wp|", "Median |delta_wp|", "Std |delta_wp|", "Ball Count"]
print(volatility.to_string())

# Also break down by season for context
volatility_by_season = (
    df.groupby(["season", "is_impact_sub_match"])["abs_delta_wp"]
    .mean()
    .unstack()
    .rename(columns={True: "Impact Sub", False: "No Impact Sub"})
)
print("\nBy season:")
print(volatility_by_season.round(4).to_string())

# %% 3. Average total TILT per match (batting team perspective)
print("\n" + "=" * 60)
print("3. TOTAL TILT PER MATCH (SUM OF |delta_wp| PER MATCH)")
print("=" * 60)

match_tilt = (
    df.groupby(["match_id", "is_impact_sub_match"])["abs_delta_wp"]
    .sum()
    .reset_index()
    .rename(columns={"abs_delta_wp": "total_tilt"})
)

tilt_comparison = (
    match_tilt
    .groupby("is_impact_sub_match")["total_tilt"]
    .agg(["mean", "median", "std", "min", "max", "count"])
    .rename(index={True: "Impact Sub", False: "No Impact Sub"})
)
tilt_comparison.columns = ["Mean TILT", "Median TILT", "Std TILT", "Min TILT", "Max TILT", "Match Count"]
print(tilt_comparison.round(2).to_string())

# Merge season info for per-season breakdown
match_tilt_season = match_tilt.merge(
    match_info[["match_id", "season"]], on="match_id"
)
tilt_by_season = (
    match_tilt_season
    .groupby(["season", "is_impact_sub_match"])["total_tilt"]
    .mean()
    .unstack()
    .rename(columns={True: "Impact Sub", False: "No Impact Sub"})
)
print("\nMean total TILT per match by season:")
print(tilt_by_season.round(2).to_string())

# %% 4. Which teams use impact subs most (per team per season)
print("\n" + "=" * 60)
print("4. IMPACT SUB USAGE BY TEAM AND SEASON")
print("=" * 60)

# Get one row per match-team combination for impact sub matches
impact_matches = df[df["is_impact_sub_match"]].copy()

if len(impact_matches) == 0:
    print("No impact sub matches found in the dataset.")
else:
    team_matches = (
        impact_matches
        .groupby(["season", "batting_team", "match_id"])
        .size()
        .reset_index()
        .groupby(["season", "batting_team"])
        .size()
        .reset_index(name="impact_sub_matches")
    )

    # Pivot for readability
    team_pivot = team_matches.pivot_table(
        index="batting_team",
        columns="season",
        values="impact_sub_matches",
        fill_value=0,
        aggfunc="sum",
    )
    team_pivot["Total"] = team_pivot.sum(axis=1)
    team_pivot = team_pivot.sort_values("Total", ascending=False)

    print("Impact sub matches per team (from batting_team field):")
    print(team_pivot.to_string())
    print(f"\nNote: Each match appears twice (once per batting team), so these")
    print(f"counts reflect appearances in impact sub matches, not unique matches.")

# %% 5. Summary of findings
print("\n" + "=" * 60)
print("5. SUMMARY OF FINDINGS")
print("=" * 60)

# Gather key stats
total_matches = match_info.shape[0]
impact_matches_count = match_info["is_impact_sub"].sum()
non_impact_matches_count = total_matches - impact_matches_count

print(f"\nDataset: {total_matches} total matches, {impact_matches_count} with impact subs")

if impact_matches_count > 0:
    mean_vol_impact = df.loc[df["is_impact_sub_match"], "abs_delta_wp"].mean()
    mean_vol_no_impact = df.loc[~df["is_impact_sub_match"], "abs_delta_wp"].mean()
    vol_diff_pct = (mean_vol_impact - mean_vol_no_impact) / mean_vol_no_impact * 100

    mean_tilt_impact = match_tilt.loc[
        match_tilt["is_impact_sub_match"], "total_tilt"
    ].mean()
    mean_tilt_no_impact = match_tilt.loc[
        ~match_tilt["is_impact_sub_match"], "total_tilt"
    ].mean()
    tilt_diff_pct = (mean_tilt_impact - mean_tilt_no_impact) / mean_tilt_no_impact * 100

    print(f"\nVolatility (avg |delta_wp| per ball):")
    print(f"  Impact sub matches:     {mean_vol_impact:.4f}")
    print(f"  Non-impact sub matches: {mean_vol_no_impact:.4f}")
    print(f"  Difference:             {vol_diff_pct:+.1f}%")

    print(f"\nTotal TILT per match:")
    print(f"  Impact sub matches:     {mean_tilt_impact:.2f}")
    print(f"  Non-impact sub matches: {mean_tilt_no_impact:.2f}")
    print(f"  Difference:             {tilt_diff_pct:+.1f}%")

    direction = "more" if vol_diff_pct > 0 else "less"
    print(f"\nConclusion: Impact sub matches are {abs(vol_diff_pct):.1f}% {direction} volatile")
    print(f"on a per-ball basis compared to non-impact sub matches.")
else:
    print("\nNo impact sub matches found. The dataset may predate IPL 2023.")
