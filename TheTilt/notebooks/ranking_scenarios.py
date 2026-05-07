# %% Ranking scenarios — explore alternatives to the current TILT calibration.
"""
Builds a ball-level deltas dataframe from the shipped per-match JSONs in
`public/data/matches/*.json`, then runs five scenarios:

  A. Baseline (shipped)
  B. Telescoping construction — wp_before(k) := wp_after(k-1) chained
  C. Per-role mean centering — subtract role-population mean
  D. Exclude DLS-affected matches (truncated innings <110 balls in inn 1)
  E. Cap |delta_wp| at 0.10 (winsorize big wicket-credits)

Each scenario reports:
  - Top 10 by TILT floor (90% CI lower bound, shrunk)
  - Top 10 by total career TILT
  - Top 10 single-game batting TILT
  - Top 10 single-game bowling TILT

Note: scenario B uses chained wp_after; data we have only includes calibrated
boundary values, so the result is a structural-fix simulation rather than
a pure model rerun.
"""

# %% Imports
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd


# %% Config
PROJECT_ROOT = Path("/home/user/Templates/TheTilt")
MATCHES_DIR = PROJECT_ROOT / "public/data/matches"
RANKINGS_PATH = PROJECT_ROOT / "public/data/tilt_rankings.json"
OUTPUT_DIR = Path("/tmp/ranking_scenarios")
OUTPUT_DIR.mkdir(exist_ok=True)

MIN_MATCHES = 10  # Same threshold as shipped pipeline


# %% Load all matches into a deltas dataframe
def load_deltas() -> pd.DataFrame:
    rows = []
    match_meta = []
    files = sorted(MATCHES_DIR.glob("*.json"))
    print(f"Loading {len(files)} match JSONs...")

    for fp in files:
        with open(fp) as f:
            d = json.load(f)
        match_id = d["match_id"]
        date = d["date"]
        venue = d["venue"]
        season = d["season"]
        winner = d.get("winner")

        # Map innings -> batting/bowling team
        team_for_innings = {}
        for inn in d.get("innings", []):
            team_for_innings[inn["innings"]] = (inn["batting_team"], inn["bowling_team"])

        inn1_balls = sum(1 for b in d.get("balls", []) if b["inn"] == 1)
        is_truncated = inn1_balls > 0 and inn1_balls < 110  # DLS proxy

        match_meta.append(
            {"match_id": match_id, "date": date, "venue": venue, "season": season,
             "winner": winner, "is_truncated": is_truncated}
        )

        for i, b in enumerate(d.get("balls", [])):
            inn = b["inn"]
            bat_team, bowl_team = team_for_innings.get(inn, (None, None))
            rows.append({
                "match_id": match_id,
                "date": date,
                "venue": venue,
                "season": season,
                "innings": inn,
                "ball_idx": i,  # global ball index in match
                "over": b["over"],
                "ball": b["ball"],
                "batter": b["batter"],
                "bowler": b["bowler"],
                "batting_team": bat_team,
                "bowling_team": bowl_team,
                "runs": b.get("runs", 0),
                "wp_before": b["wp"],
                "wp_after": b["wp_after"],
                "delta_wp": b["delta"],
                "is_wicket": bool(b.get("wicket", False)),
                "is_truncated": is_truncated,
            })

    deltas = pd.DataFrame(rows)
    meta = pd.DataFrame(match_meta)
    print(f"  {len(deltas):,} ball events, {len(meta)} matches "
          f"(truncated: {meta['is_truncated'].sum()})")
    return deltas, meta


# %% Aggregation helpers
def aggregate_player(deltas: pd.DataFrame, label: str) -> pd.DataFrame:
    """Aggregate ball-level deltas to per-player career TILT.

    Returns DataFrame with columns: player, total_matches, batting_balls,
    bowling_balls, batting_total_tilt, bowling_total_tilt, total_tilt,
    total_tilt_per_match (raw), shrunk_total_tilt_per_match,
    tilt_ci_lower_90.
    """
    bat = (
        deltas.groupby("batter")
        .agg(
            batting_total_tilt=("delta_wp", "sum"),
            batting_balls=("delta_wp", "count"),
            bat_match_set=("match_id", set),
        )
        .reset_index()
        .rename(columns={"batter": "player"})
    )
    bowl = (
        deltas.assign(bowling_delta=-deltas["delta_wp"])
        .groupby("bowler")
        .agg(
            bowling_total_tilt=("bowling_delta", "sum"),
            bowling_balls=("bowling_delta", "count"),
            bowl_match_set=("match_id", set),
        )
        .reset_index()
        .rename(columns={"bowler": "player"})
    )
    combined = pd.merge(bat, bowl, on="player", how="outer")
    for col in ["batting_total_tilt", "batting_balls", "bowling_total_tilt", "bowling_balls"]:
        combined[col] = combined[col].fillna(0)
    combined["bat_match_set"] = combined["bat_match_set"].apply(lambda x: x if isinstance(x, set) else set())
    combined["bowl_match_set"] = combined["bowl_match_set"].apply(lambda x: x if isinstance(x, set) else set())
    combined["total_matches"] = combined.apply(
        lambda r: len(r["bat_match_set"] | r["bowl_match_set"]), axis=1
    )
    combined["total_tilt"] = combined["batting_total_tilt"] + combined["bowling_total_tilt"]
    combined["total_tilt_per_match"] = combined["total_tilt"] / combined["total_matches"].replace(0, 1)
    combined = combined.drop(columns=["bat_match_set", "bowl_match_set"])

    # Filter to ranked players (>=10 matches)
    combined = combined[combined["total_matches"] >= MIN_MATCHES].copy()

    # Build per-player per-match TILT for shrinkage
    bat_match = (
        deltas.groupby(["batter", "match_id"])["delta_wp"].sum().reset_index()
        .rename(columns={"batter": "player", "delta_wp": "match_tilt"})
    )
    bowl_match = (
        deltas.assign(bd=-deltas["delta_wp"])
        .groupby(["bowler", "match_id"])["bd"].sum().reset_index()
        .rename(columns={"bowler": "player", "bd": "match_tilt"})
    )
    total_match = (
        pd.concat([bat_match, bowl_match])
        .groupby(["player", "match_id"])["match_tilt"].sum().reset_index()
    )

    pop_mean = combined["total_tilt_per_match"].mean()
    between_var = combined["total_tilt_per_match"].var()
    within_var = total_match.groupby("player")["match_tilt"].var().dropna().mean()
    if pd.isna(within_var) or within_var <= 0 or between_var <= 0:
        k = 5.0
    else:
        k = within_var / between_var
    print(f"  [{label}] k={k:.2f}, pop_mean={pop_mean:+.4f}, between_var={between_var:.5f}, within_var={within_var:.5f}")

    # Merge per-player match-tilt variance BEFORE computing shrinkage (so the
    # index reset from the merge doesn't misalign downstream operations).
    player_var = total_match.groupby("player")["match_tilt"].var().fillna(within_var)
    combined = combined.merge(player_var.rename("match_tilt_var").reset_index(), on="player", how="left")
    combined["match_tilt_var"] = combined["match_tilt_var"].fillna(within_var)

    n = combined["total_matches"]
    combined["shrunk_total_tilt_per_match"] = (
        (n / (n + k)) * combined["total_tilt_per_match"] + (k / (n + k)) * pop_mean
    )
    posterior_se = np.sqrt(combined["match_tilt_var"] / (n + k).clip(lower=1))
    combined["tilt_ci_lower_90"] = combined["shrunk_total_tilt_per_match"] - 1.645 * posterior_se

    combined = combined.sort_values("tilt_ci_lower_90", ascending=False).reset_index(drop=True)
    return combined


def aggregate_match_perf(deltas: pd.DataFrame) -> tuple:
    """Aggregate ball-level deltas to per-(player, match) batting + bowling tilt."""
    bat = (
        deltas.groupby(["batter", "match_id", "date", "venue", "batting_team", "innings"])
        .agg(tilt=("delta_wp", "sum"), runs=("runs", "sum"), balls=("delta_wp", "count"))
        .reset_index()
        .rename(columns={"batter": "player", "batting_team": "team"})
    )
    bowl = (
        deltas.assign(bd=-deltas["delta_wp"])
        .groupby(["bowler", "match_id", "date", "venue", "bowling_team", "innings"])
        .agg(tilt=("bd", "sum"), runs_conceded=("runs", "sum"),
             balls=("bd", "count"), wickets=("is_wicket", "sum"))
        .reset_index()
        .rename(columns={"bowler": "player", "bowling_team": "team"})
    )
    return bat, bowl


# %% Scenario transforms
def scenario_baseline(deltas: pd.DataFrame, _meta: pd.DataFrame) -> pd.DataFrame:
    return deltas.copy()


def scenario_telescoping(deltas: pd.DataFrame, _meta: pd.DataFrame) -> pd.DataFrame:
    """Issue #110 fix: wp_before(k) := wp_after(k-1) chained WITHIN each innings.

    Chains over-boundary feature reshuffles inside an innings (where the
    batting team POV is consistent), but resets at the inn1->inn2 boundary
    (different batting teams = different POVs; existing boundary calibration
    handles the discontinuity). New delta_wp = wp_after(k) - wp_after(k-1)
    for k>0; first ball of each innings keeps wp_after(0) - wp_before(0).
    """
    df = deltas.copy().sort_values(["match_id", "innings", "ball_idx"]).reset_index(drop=True)
    df["_grp"] = df["match_id"].astype(str) + "_" + df["innings"].astype(str)
    df["_prev_wp_after"] = df.groupby("_grp")["wp_after"].shift(1)
    df["delta_wp"] = df["wp_after"] - df["_prev_wp_after"]
    # First ball of each innings: fall back to wp_after - wp_before (original)
    first_ball_mask = df["_prev_wp_after"].isna()
    df.loc[first_ball_mask, "delta_wp"] = (
        df.loc[first_ball_mask, "wp_after"] - df.loc[first_ball_mask, "wp_before"]
    )
    return df.drop(columns=["_grp", "_prev_wp_after"])


def scenario_per_role_centering(deltas: pd.DataFrame, _meta: pd.DataFrame) -> pd.DataFrame:
    """Subtract per-role population mean from each ball's delta_wp.

    Two adjustments: batting deltas centered on the population batting mean,
    bowling deltas (which are -delta_wp) centered separately.

    Implementation: compute mean delta_wp across all balls (which is the
    batter-side mean). For aggregation we'll handle the bowling side in
    aggregate_player by adjusting the bowling sum separately.

    To keep the math clean we instead store *adjusted* delta in delta_wp.
    Note: this means batter sum != -bowler sum anymore. The accounting
    becomes asymmetric — that's the point.
    """
    df = deltas.copy()
    bat_mean = df["delta_wp"].mean()  # average per-ball batter credit
    # Adjusted: subtract bat_mean from each ball's batter credit
    # Bowler credit becomes -(delta_wp - bat_mean) = -delta_wp + bat_mean
    # which means bowler is credited the negation, plus a constant.
    df["delta_wp"] = df["delta_wp"] - bat_mean
    # The bowler will pick up -delta_wp from this same column, so we ALSO
    # need to neutralize bowler population mean. Computed: original bowler
    # mean = -bat_mean. After subtracting bat_mean from delta_wp, bowler's
    # delta = -(delta - bat_mean) = -delta + bat_mean. New mean = -bat_mean
    # + bat_mean = 0. So both sides end up centered at 0.
    return df


def scenario_exclude_dls(deltas: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    truncated_ids = set(meta[meta["is_truncated"]]["match_id"])
    print(f"  Excluding {len(truncated_ids)} truncated/DLS matches")
    return deltas[~deltas["match_id"].isin(truncated_ids)].copy()


def scenario_winsorize(deltas: pd.DataFrame, _meta: pd.DataFrame) -> pd.DataFrame:
    """Cap |delta_wp| at 0.10."""
    df = deltas.copy()
    df["delta_wp"] = df["delta_wp"].clip(lower=-0.10, upper=0.10)
    return df


# %% Print top 10 helpers
def print_top10(label: str, df: pd.DataFrame, sort_col: str, value_cols: list) -> None:
    sub = df.sort_values(sort_col, ascending=False).head(10)
    cols = ["player"] + value_cols
    print(f"\n  {label}")
    print(sub[cols].to_string(index=False))


def run_scenario(name: str, deltas: pd.DataFrame, meta: pd.DataFrame, transform) -> dict:
    print(f"\n{'=' * 80}")
    print(f"  SCENARIO {name}")
    print(f"{'=' * 80}")
    transformed = transform(deltas, meta)
    player = aggregate_player(transformed, name)
    bat_match, bowl_match = aggregate_match_perf(transformed)

    # Top 10 floor
    print(f"\n  TOP 10 — TILT FLOOR (90% CI lower)")
    sub = player.head(10)
    print(f"  {'#':<3} {'player':<24} {'floor':>8} {'TPM':>8} {'mat':>5}")
    for i, (_, r) in enumerate(sub.iterrows(), 1):
        print(f"  {i:<3} {r['player']:<24} {r['tilt_ci_lower_90']:>+8.4f} {r['shrunk_total_tilt_per_match']:>+8.4f} {r['total_matches']:>5}")

    # Top 10 total career
    print(f"\n  TOP 10 — TOTAL CAREER TILT")
    sub_total = player.sort_values("total_tilt", ascending=False).head(10)
    print(f"  {'#':<3} {'player':<24} {'total':>8} {'TPM':>8} {'mat':>5}")
    for i, (_, r) in enumerate(sub_total.iterrows(), 1):
        print(f"  {i:<3} {r['player']:<24} {r['total_tilt']:>+8.3f} {r['shrunk_total_tilt_per_match']:>+8.4f} {r['total_matches']:>5}")

    # Top 10 single-game batting
    print(f"\n  TOP 10 — SINGLE-GAME BATTING TILT")
    top_bat = bat_match.nlargest(10, "tilt")
    print(f"  {'#':<3} {'player':<22} {'tilt':>7} {'R':>4} {'B':>4} {'date':<12} {'venue':<28}")
    for i, (_, r) in enumerate(top_bat.iterrows(), 1):
        print(f"  {i:<3} {r['player']:<22} {r['tilt']:>+7.3f} {r['runs']:>4} {r['balls']:>4} {r['date']:<12} {r['venue'][:28]:<28}")

    # Top 10 single-game bowling
    print(f"\n  TOP 10 — SINGLE-GAME BOWLING TILT")
    top_bowl = bowl_match.nlargest(10, "tilt")
    print(f"  {'#':<3} {'player':<22} {'tilt':>7} {'W':>3} {'R':>4} {'B':>4} {'date':<12} {'venue':<28}")
    for i, (_, r) in enumerate(top_bowl.iterrows(), 1):
        print(f"  {i:<3} {r['player']:<22} {r['tilt']:>+7.3f} {r['wickets']:>3} {r['runs_conceded']:>4} {r['balls']:>4} {r['date']:<12} {r['venue'][:28]:<28}")

    return {"player": player, "bat_match": bat_match, "bowl_match": bowl_match}


# %% Main
def main():
    deltas, meta = load_deltas()

    scenarios = [
        ("A — Baseline (shipped)", scenario_baseline),
        ("B — Telescoping construction (issue #110)", scenario_telescoping),
        ("C — Per-role mean centering", scenario_per_role_centering),
        ("D — Exclude DLS / truncated matches", scenario_exclude_dls),
        ("E — Winsorize |delta_wp| at 0.10", scenario_winsorize),
    ]

    results = {}
    for name, fn in scenarios:
        results[name] = run_scenario(name, deltas, meta, fn)

    return results


if __name__ == "__main__":
    main()
