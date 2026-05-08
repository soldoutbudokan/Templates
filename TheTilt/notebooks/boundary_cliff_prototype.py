# %% Boundary-cliff calibration prototype (issues #71 / #110)
"""
Compare candidate calibrations for innings/over boundaries.

Originally written for issue #71 (cliff migrated from ball 1 to ball 2 of
inn2 after the per-match midpoint fix from #62). Extended for issue #110
with V5 — chained endpoints across ALL balls — to evaluate whether
collapsing every per-ball wp_before(k) := wp_after(k-1) closes the
~0.038pp telescoping residual without disrupting rankings.

Each variant rewrites `wp_before` / `wp_after` over different scopes.
Per-player TILT is recomputed and the top-10 batting / bowling / overall
lists are printed side-by-side. We also print median |delta_wp| at each
of the first 6 balls of inn2 (the original diagnostic) and a per-innings
telescoping residual diagnostic for all variants.

Variants:
    V0 — current production. Per-side isotonic + 1-ball midpoint at the
         inn1↔inn2 seam. (Issue #62 fix.)
    V1 — linear decay over first 6 balls of inn2.
    V2 — cosine decay over first 6 balls.
    V3 — 2-ball midpoint extension.
    V4 — score-anchored prior over first 6 balls.
    V5 — V0 seam fix, then wp_before(k) := wp_after(k-1) for every
         ball k > 1 within each innings. Targets issue #110.

Run from TheTilt/:
    ./venv/bin/python notebooks/boundary_cliff_prototype.py
"""

# %% Imports
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.compute_tilt import FEATURE_COLS, apply_shrinkage, compute_state_after  # noqa: E402

# %% Paths
ROOT = Path(__file__).resolve().parent.parent
FEATURED = ROOT / "data" / "processed" / "featured_balls.parquet"
MODEL = ROOT / "models" / "win_prob_lgbm.pkl"


# %% Load model + features once
def load_inputs():
    print("Loading model + features...")
    with open(MODEL, "rb") as f:
        model = pickle.load(f)
    df = pd.read_parquet(FEATURED)
    print(f"  {len(df):,} balls across {df['match_id'].nunique():,} matches")
    return model, df


# %% Compute raw wp_before / wp_after / delta_wp for every ball
def compute_raw_deltas(model, df: pd.DataFrame) -> pd.DataFrame:
    print("Computing raw WP for every ball...")
    df = df.copy()
    if "venue" in df.columns:
        df["venue"] = df["venue"].astype("category")

    X_before = df[FEATURE_COLS].copy()
    wp_before = model.predict_proba(X_before)[:, 1]

    after_states = df.apply(compute_state_after, axis=1, result_type="expand")
    X_after = after_states[FEATURE_COLS].copy()
    if "venue" in X_after.columns:
        X_after["venue"] = X_after["venue"].astype("category")
    wp_after = model.predict_proba(X_after)[:, 1]

    df["wp_before"] = wp_before
    df["wp_after"] = wp_after
    df["delta_wp"] = wp_after - wp_before
    return df


# %% Helpers shared by every variant
def boundary_indices(df: pd.DataFrame) -> tuple:
    inn1_last_idx = (
        df[df["innings"] == 1]
        .sort_values(["match_id", "ball_number"])
        .groupby("match_id").tail(1).index
    )
    inn2_first_idx = (
        df[df["innings"] == 2]
        .sort_values(["match_id", "ball_number"])
        .groupby("match_id").head(1).index
    )
    return inn1_last_idx, inn2_first_idx


def fit_isotonic(df: pd.DataFrame, inn1_last_idx, inn2_first_idx):
    """Fit per-side isotonic on non-DLS matches."""
    inn1_pairs = df.loc[inn1_last_idx, ["match_id", "wp_after", "batting_team_won"]].rename(
        columns={"wp_after": "wp_inn1_end_bf"}
    )
    inn2_pairs = df.loc[inn2_first_idx, ["match_id", "wp_before"]].rename(
        columns={"wp_before": "wp_inn2_start_chase"}
    )
    pairs = inn1_pairs.merge(inn2_pairs, on="match_id")
    pairs["wp_inn2_start_bf"] = 1.0 - pairs["wp_inn2_start_chase"]
    pairs["bf_won"] = pairs["batting_team_won"]

    dls = set(df.loc[df["dls_method"].notna(), "match_id"].unique())
    fit = pairs[~pairs["match_id"].isin(dls)]

    iso_inn1 = IsotonicRegression(out_of_bounds="clip", y_min=0.001, y_max=0.999)
    iso_inn1.fit(fit["wp_inn1_end_bf"].values, fit["bf_won"].values)
    iso_inn2 = IsotonicRegression(out_of_bounds="clip", y_min=0.001, y_max=0.999)
    iso_inn2.fit(fit["wp_inn2_start_bf"].values, fit["bf_won"].values)
    return iso_inn1, iso_inn2, pairs


def midpoints_per_match(df: pd.DataFrame, iso_inn1, iso_inn2, inn1_last_idx, inn2_first_idx) -> dict:
    inn1_cal = iso_inn1.transform(df.loc[inn1_last_idx, "wp_after"].values)
    chase_raw = df.loc[inn2_first_idx, "wp_before"].values
    inn2_cal_bf = iso_inn2.transform(1.0 - chase_raw)
    midpoint_bf = 0.5 * (inn1_cal + inn2_cal_bf)
    mid_by_match = dict(
        zip(df.loc[inn1_last_idx, "match_id"].values, midpoint_bf)
    )
    return mid_by_match


# %% Variant calibrators
def apply_v0_current(df: pd.DataFrame) -> pd.DataFrame:
    """Production logic: per-side isotonic + 1-ball midpoint."""
    df = df.copy()
    inn1_last_idx, inn2_first_idx = boundary_indices(df)
    iso1, iso2, _ = fit_isotonic(df, inn1_last_idx, inn2_first_idx)
    mid_by_match = midpoints_per_match(df, iso1, iso2, inn1_last_idx, inn2_first_idx)

    inn1_mids = np.array([mid_by_match[m] for m in df.loc[inn1_last_idx, "match_id"].values])
    inn2_mids = np.array([mid_by_match[m] for m in df.loc[inn2_first_idx, "match_id"].values])
    df.loc[inn1_last_idx, "wp_after"] = inn1_mids
    df.loc[inn2_first_idx, "wp_before"] = 1.0 - inn2_mids

    df.loc[inn1_last_idx, "delta_wp"] = df.loc[inn1_last_idx, "wp_after"] - df.loc[inn1_last_idx, "wp_before"]
    df.loc[inn2_first_idx, "delta_wp"] = df.loc[inn2_first_idx, "wp_after"] - df.loc[inn2_first_idx, "wp_before"]
    return df


def _apply_decay(df: pd.DataFrame, weights: np.ndarray) -> pd.DataFrame:
    """Apply a decay-weight calibration to inn1.last and the first 6 balls of inn2.

    `weights` must be length 7: index 0 = inn1.last, indices 1..6 = inn2 ball 1..6.
    weights[0] should be 1.0 (full midpoint at boundary). weights[6] should be 0.0
    (no adjustment by end of first over). Intermediate weights interpolate.

    Smearing rule (BF perspective):
        wp_after_bf[k] = w_k * midpoint_bf + (1 - w_k) * raw_wp_after_bf[k]
        wp_before_bf[k] = wp_after_bf[k-1]   (continuity)
    """
    assert len(weights) == 7
    df = df.copy()
    inn1_last_idx, inn2_first_idx = boundary_indices(df)
    iso1, iso2, _ = fit_isotonic(df, inn1_last_idx, inn2_first_idx)
    mid_by_match = midpoints_per_match(df, iso1, iso2, inn1_last_idx, inn2_first_idx)

    # ---- inn1.last: w_0 * midpoint + (1-w_0) * raw_wp_after ----
    inn1_mids = np.array([mid_by_match[m] for m in df.loc[inn1_last_idx, "match_id"].values])
    inn1_raw = df.loc[inn1_last_idx, "wp_after"].values
    inn1_adj = weights[0] * inn1_mids + (1 - weights[0]) * inn1_raw
    df.loc[inn1_last_idx, "wp_after"] = inn1_adj
    df.loc[inn1_last_idx, "delta_wp"] = (
        df.loc[inn1_last_idx, "wp_after"] - df.loc[inn1_last_idx, "wp_before"]
    )

    # ---- inn2 first 6 balls: smear toward midpoint with decaying weight ----
    inn2_df = df[df["innings"] == 2].sort_values(["match_id", "ball_number"]).copy()
    inn2_first6 = inn2_df.groupby("match_id").head(6).copy()
    inn2_first6["ball_in_inn"] = inn2_first6.groupby("match_id").cumcount()  # 0..5
    inn2_first6["midpoint_bf"] = inn2_first6["match_id"].map(mid_by_match)
    inn2_first6["weight"] = inn2_first6["ball_in_inn"].map(lambda k: weights[k + 1])

    # raw wp_after_bf = 1 - wp_after (since inn2 is chasing)
    inn2_first6["raw_wp_after_bf"] = 1.0 - inn2_first6["wp_after"]
    inn2_first6["adj_wp_after_bf"] = (
        inn2_first6["weight"] * inn2_first6["midpoint_bf"]
        + (1 - inn2_first6["weight"]) * inn2_first6["raw_wp_after_bf"]
    )
    inn2_first6["adj_wp_after_chase"] = 1.0 - inn2_first6["adj_wp_after_bf"]

    # Continuity: wp_before[k] = wp_after[k-1]; for k=0, wp_before = 1 - midpoint (matches inn1.last fix)
    def _wp_before_chase(grp):
        midpoint = grp["midpoint_bf"].iloc[0]
        # If weights[0] == 1, inn1.last.wp_after_bf = midpoint exactly, so ball-0 wp_before_chase = 1 - midpoint.
        # But we apply the inn1 weight too — replicate what we did above.
        # Rather than recompute, use the now-modified inn1.last.wp_after:
        match_id = grp["match_id"].iloc[0]
        inn1_after_bf = df.loc[(df["match_id"] == match_id) & (df["innings"] == 1)].sort_values("ball_number").iloc[-1]["wp_after"]
        prev_after_chase = 1.0 - inn1_after_bf
        out = []
        for _, row in grp.iterrows():
            out.append(prev_after_chase)
            prev_after_chase = row["adj_wp_after_chase"]
        return pd.Series(out, index=grp.index)

    inn2_first6["adj_wp_before_chase"] = (
        inn2_first6.groupby("match_id", group_keys=False).apply(_wp_before_chase)
    )

    # Patch back into df at original indices
    df.loc[inn2_first6.index, "wp_after"] = inn2_first6["adj_wp_after_chase"].values
    df.loc[inn2_first6.index, "wp_before"] = inn2_first6["adj_wp_before_chase"].values
    df.loc[inn2_first6.index, "delta_wp"] = (
        df.loc[inn2_first6.index, "wp_after"] - df.loc[inn2_first6.index, "wp_before"]
    )
    return df


def apply_v1_linear(df: pd.DataFrame) -> pd.DataFrame:
    """Linear decay: weights = [1.0, 5/6, 4/6, 3/6, 2/6, 1/6, 0]."""
    weights = np.array([1.0] + [(6 - k) / 6 for k in range(1, 7)])
    return _apply_decay(df, weights)


def apply_v2_cosine(df: pd.DataFrame) -> pd.DataFrame:
    """Cosine decay: smoother S-curve with same boundary conditions."""
    # 0..6 mapped to weights via half-cosine: w_k = 0.5 * (1 + cos(pi * k/6))
    weights = np.array([0.5 * (1 + np.cos(np.pi * k / 6)) for k in range(7)])
    return _apply_decay(df, weights)


def apply_v3_two_ball_midpoint(df: pd.DataFrame) -> pd.DataFrame:
    """Force inn1.last and the first TWO balls of inn2 to the midpoint."""
    # weight 1.0 at inn1.last + ball1 + ball2 of inn2, then 0
    weights = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    return _apply_decay(df, weights)


def apply_v4_score_prior(df: pd.DataFrame) -> pd.DataFrame:
    """Blend raw inn2 WP with a score-only empirical prior over the first over.

    Score-only prior at (target, ball_k, score, wickets): empirical P(BF wins)
    in non-DLS matches conditional on those four state variables, smoothed
    via a small bin-pooling. We then blend toward the midpoint with a linear
    decay — same envelope as V1, but the "raw" baseline used is the prior
    instead of the model's prediction. This isolates how much of the cliff
    is the model picking up state-dependent noise vs. genuine signal.
    """
    df = df.copy()
    inn1_last_idx, inn2_first_idx = boundary_indices(df)
    iso1, iso2, _ = fit_isotonic(df, inn1_last_idx, inn2_first_idx)
    mid_by_match = midpoints_per_match(df, iso1, iso2, inn1_last_idx, inn2_first_idx)

    # Apply V0-style fix to inn1.last
    inn1_mids = np.array([mid_by_match[m] for m in df.loc[inn1_last_idx, "match_id"].values])
    df.loc[inn1_last_idx, "wp_after"] = inn1_mids
    df.loc[inn1_last_idx, "delta_wp"] = (
        df.loc[inn1_last_idx, "wp_after"] - df.loc[inn1_last_idx, "wp_before"]
    )

    # Build the score prior: bin (target_bucket, ball_in_over, runs_scored_bucket, wickets_fallen)
    inn2 = df[df["innings"] == 2].sort_values(["match_id", "ball_number"]).copy()
    inn2["ball_in_inn"] = inn2.groupby("match_id").cumcount()
    inn2_first6 = inn2[inn2["ball_in_inn"] < 6].copy()
    # Use match-level target bucketed to nearest 20 runs to keep bins populous
    inn2_first6["target_bucket"] = (inn2_first6["target"] // 20).astype(int)
    inn2_first6["runs_bucket"] = (inn2_first6["runs_scored"] // 5).astype(int)

    dls = set(df.loc[df["dls_method"].notna(), "match_id"].unique())
    fit = inn2_first6[~inn2_first6["match_id"].isin(dls)]
    fit["bf_won"] = fit["batting_team_won"]

    prior = (
        fit.groupby(["target_bucket", "ball_in_inn", "runs_bucket", "wickets_fallen"])["bf_won"]
        .agg(["mean", "count"])
        .reset_index()
    )
    prior_lookup = {
        (r["target_bucket"], r["ball_in_inn"], r["runs_bucket"], r["wickets_fallen"]): r["mean"]
        for _, r in prior[prior["count"] >= 30].iterrows()
    }
    overall_mean = float(fit["bf_won"].mean())

    inn2_first6["prior_bf"] = inn2_first6.apply(
        lambda r: prior_lookup.get(
            (r["target_bucket"], r["ball_in_inn"], r["runs_bucket"], r["wickets_fallen"]),
            overall_mean,
        ),
        axis=1,
    )
    weights = np.array([(6 - k) / 6 for k in range(6)])  # 1, 5/6, ..., 1/6
    inn2_first6["weight"] = inn2_first6["ball_in_inn"].map(lambda k: weights[k])
    inn2_first6["midpoint_bf"] = inn2_first6["match_id"].map(mid_by_match)
    inn2_first6["adj_wp_after_bf"] = (
        inn2_first6["weight"] * inn2_first6["midpoint_bf"]
        + (1 - inn2_first6["weight"]) * inn2_first6["prior_bf"]
    )
    inn2_first6["adj_wp_after_chase"] = 1.0 - inn2_first6["adj_wp_after_bf"]

    # Continuity for wp_before
    def _wp_before_chase(grp):
        match_id = grp["match_id"].iloc[0]
        inn1_after_bf = df.loc[(df["match_id"] == match_id) & (df["innings"] == 1)].sort_values("ball_number").iloc[-1]["wp_after"]
        prev_after_chase = 1.0 - inn1_after_bf
        out = []
        for _, row in grp.iterrows():
            out.append(prev_after_chase)
            prev_after_chase = row["adj_wp_after_chase"]
        return pd.Series(out, index=grp.index)

    inn2_first6["adj_wp_before_chase"] = (
        inn2_first6.groupby("match_id", group_keys=False).apply(_wp_before_chase)
    )
    df.loc[inn2_first6.index, "wp_after"] = inn2_first6["adj_wp_after_chase"].values
    df.loc[inn2_first6.index, "wp_before"] = inn2_first6["adj_wp_before_chase"].values
    df.loc[inn2_first6.index, "delta_wp"] = (
        df.loc[inn2_first6.index, "wp_after"] - df.loc[inn2_first6.index, "wp_before"]
    )
    return df


def apply_v5_chained_all_balls(df: pd.DataFrame) -> pd.DataFrame:
    """Issue #110: chained endpoints across ALL balls.

    Apply V0 (per-side isotonic + per-match midpoint at the inn1↔inn2 seam)
    first, then enforce `wp_before(k) := wp_after(k-1)` for every ball k > 1
    within each innings. Eliminates over-boundary feature-reshuffle gaps —
    the per-match telescoping residual collapses to ~0 by construction.
    """
    df = apply_v0_current(df)
    sorter = df.sort_values(["match_id", "innings", "ball_number"])
    chained = sorter.groupby(["match_id", "innings"])["wp_after"].shift(1)
    valid = chained.dropna()
    df.loc[valid.index, "wp_before"] = valid.values
    df["delta_wp"] = df["wp_after"] - df["wp_before"]
    return df


# %% Diagnostic: per-innings telescoping residual
def telescoping_diagnostic(df: pd.DataFrame) -> dict:
    """Per-match: |Σ(deltas) − (wp_after_last − wp_before_first)| in pp.

    Under chained endpoints across all balls, this collapses to ~0 by
    construction. Under V0/V1/.../V4 the residual accumulates from the
    over-boundary feature-reshuffle gaps the model produces.
    """
    out = {}
    for innings in (1, 2):
        sub = df[df["innings"] == innings].sort_values(["match_id", "ball_number"])
        per_match = sub.groupby("match_id").agg(
            sum_delta=("delta_wp", "sum"),
            wp_first_before=("wp_before", "first"),
            wp_last_after=("wp_after", "last"),
        )
        per_match["expected"] = per_match["wp_last_after"] - per_match["wp_first_before"]
        per_match["residual_pp"] = (per_match["sum_delta"] - per_match["expected"]) * 100
        out[f"inn{innings}_mean_abs"] = per_match["residual_pp"].abs().mean()
        out[f"inn{innings}_p95_abs"] = per_match["residual_pp"].abs().quantile(0.95)
        out[f"inn{innings}_max_abs"] = per_match["residual_pp"].abs().max()
    return out


# %% Aggregation: per-player TILT (raw, no shrinkage — same comparison basis across variants)
def aggregate_tilt(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["legal_bat"] = (~df["is_wide"]).astype(int)
    df["legal_bowl"] = (~df["is_wide"] & ~df["is_noball"]).astype(int)

    bat = (
        df.groupby("batter_id")
        .agg(
            player=("batter", "last"),
            batting_total_tilt=("delta_wp", "sum"),
            batting_balls=("legal_bat", "sum"),
            batting_matches=("match_id", "nunique"),
        )
        .reset_index()
        .rename(columns={"batter_id": "player_id"})
    )
    bat["batting_tilt_per_match"] = bat["batting_total_tilt"] / bat["batting_matches"]

    df["bowling_delta"] = -df["delta_wp"]
    bowl = (
        df.groupby("bowler_id")
        .agg(
            bowling_total_tilt=("bowling_delta", "sum"),
            bowling_balls=("legal_bowl", "sum"),
            bowling_matches=("match_id", "nunique"),
        )
        .reset_index()
        .rename(columns={"bowler_id": "player_id"})
    )
    bowl["bowling_tilt_per_match"] = bowl["bowling_total_tilt"] / bowl["bowling_matches"]

    combined = pd.merge(bat, bowl, on="player_id", how="outer").fillna(0)

    bat_matches = df.groupby("batter_id")["match_id"].apply(set).reset_index()
    bat_matches.columns = ["player_id", "bat_set"]
    bowl_matches = df.groupby("bowler_id")["match_id"].apply(set).reset_index()
    bowl_matches.columns = ["player_id", "bowl_set"]
    sets = pd.merge(bat_matches, bowl_matches, on="player_id", how="outer")
    sets["bat_set"] = sets["bat_set"].apply(lambda x: x if isinstance(x, set) else set())
    sets["bowl_set"] = sets["bowl_set"].apply(lambda x: x if isinstance(x, set) else set())
    sets["total_matches"] = sets.apply(lambda r: len(r["bat_set"] | r["bowl_set"]), axis=1)
    combined = combined.merge(sets[["player_id", "total_matches"]], on="player_id", how="left")

    combined["total_tilt"] = combined["batting_total_tilt"] + combined["bowling_total_tilt"]
    combined["total_tilt_per_match"] = combined["total_tilt"] / combined["total_matches"]

    # Stub `team` column so production apply_shrinkage's tail print doesn't KeyError.
    last_team = (
        df.sort_values("date").groupby("batter_id")["batting_team"].last().reset_index()
        .rename(columns={"batter_id": "player_id", "batting_team": "team"})
    )
    combined = combined.merge(last_team, on="player_id", how="left")
    combined["team"] = combined["team"].fillna("Unknown")

    return combined


# %% Diagnostic: median |delta_wp| at each of the first N balls of inn2
def jump_diagnostic(df: pd.DataFrame, n_balls: int = 8) -> pd.Series:
    inn2 = df[df["innings"] == 2].sort_values(["match_id", "ball_number"]).copy()
    inn2["ball_in_inn"] = inn2.groupby("match_id").cumcount()
    rows = inn2[inn2["ball_in_inn"] < n_balls]
    return rows.groupby("ball_in_inn")["delta_wp"].apply(lambda x: x.abs().median() * 100)


# %% Top-10 printer — VOLUME ranking (total TILT) with floor (90% CI lower bound)
def print_top10(combined: pd.DataFrame, label: str, min_matches: int = 10):
    qual = combined[combined["total_matches"] >= min_matches].copy()
    print(f"\n{'='*70}\n{label}\n{'='*70}")

    print("\nTOP 10 BATTING (by batting_total_tilt, batting_balls >= 100):")
    bat_q = qual[qual["batting_balls"] >= 100].sort_values("batting_total_tilt", ascending=False).head(10)
    print(bat_q[["player", "batting_total_tilt", "batting_tilt_per_match", "batting_matches", "batting_balls"]].to_string(index=False))

    print("\nTOP 10 BOWLING (by bowling_total_tilt, bowling_balls >= 100):")
    bowl_q = qual[qual["bowling_balls"] >= 100].sort_values("bowling_total_tilt", ascending=False).head(10)
    print(bowl_q[["player", "bowling_total_tilt", "bowling_tilt_per_match", "bowling_matches", "bowling_balls"]].to_string(index=False))

    print("\nTOP 10 OVERALL (by total_tilt — career volume):")
    cols_overall = ["player", "total_tilt", "total_tilt_per_match"]
    if "tilt_ci_lower_90" in qual.columns:
        cols_overall.append("tilt_ci_lower_90")
    cols_overall.append("total_matches")
    if "confidence" in qual.columns:
        cols_overall.append("confidence")
    ov = qual.sort_values("total_tilt", ascending=False).head(10)
    print(ov[cols_overall].to_string(index=False))

    if "tilt_ci_lower_90" in qual.columns:
        print("\nTOP 10 BY TILT FLOOR (shrunk per-match minus 1.645·SE — production sort):")
        floor = qual.sort_values("tilt_ci_lower_90", ascending=False).head(10)
        print(floor[["player", "tilt_ci_lower_90", "shrunk_total_tilt_per_match", "total_tilt", "total_matches", "confidence"]].to_string(index=False))


# %% Driver
def main():
    model, df_raw = load_inputs()
    df_raw = compute_raw_deltas(model, df_raw)

    variants = {
        "V0 — current production (1-ball midpoint)": apply_v0_current,
        "V1 — linear decay over first over": apply_v1_linear,
        "V2 — cosine decay over first over": apply_v2_cosine,
        "V3 — 2-ball midpoint extension": apply_v3_two_ball_midpoint,
        "V4 — score-prior blend over first over": apply_v4_score_prior,
        "V5 — chained endpoints (all balls, #110)": apply_v5_chained_all_balls,
    }

    diagnostics = {}
    telescoping = {}
    rankings = {}
    for label, fn in variants.items():
        print(f"\n>>> Applying {label}")
        d = fn(df_raw)
        diagnostics[label] = jump_diagnostic(d)
        telescoping[label] = telescoping_diagnostic(d)
        agg = aggregate_tilt(d)
        # Run production shrinkage so the top-10 by floor is comparable to the live site.
        try:
            agg = apply_shrinkage(agg, d)
        except Exception as e:
            print(f"  Warning: shrinkage failed for {label}: {e}")
        rankings[label] = agg

    # ---- Diagnostic table ----
    print("\n\n" + "=" * 70)
    print("MEDIAN |delta_wp| AT EACH BALL OF INNINGS 2 (percentage points)")
    print("=" * 70)
    diag_table = pd.DataFrame(diagnostics)
    diag_table.index.name = "ball_in_inn2"
    print(diag_table.round(2).to_string())

    print("\n\n" + "=" * 70)
    print("PER-INNINGS TELESCOPING RESIDUAL: |Σ deltas − (wp_after_last − wp_before_first)| (pp)")
    print("=" * 70)
    tele_table = pd.DataFrame(telescoping)
    print(tele_table.round(4).to_string())

    # ---- Top-10s ----
    for label, combined in rankings.items():
        print_top10(combined, label)

    # ---- Side-by-side delta vs V0 ----
    print("\n\n" + "=" * 70)
    print("PLAYERS WHOSE OVERALL RANK CHANGES UNDER EACH VARIANT (vs V0)")
    print("=" * 70)
    base = rankings["V0 — current production (1-ball midpoint)"].copy()
    base = base[base["total_matches"] >= 10].copy()
    base["rank_v0"] = base["total_tilt_per_match"].rank(ascending=False, method="min")
    base = base.set_index("player_id")[["player", "total_tilt_per_match", "rank_v0"]]
    for label, c in rankings.items():
        if label.startswith("V0"):
            continue
        c = c[c["total_matches"] >= 10].copy()
        c["rank_v"] = c["total_tilt_per_match"].rank(ascending=False, method="min")
        c = c.set_index("player_id")[["total_tilt_per_match", "rank_v"]].rename(
            columns={"total_tilt_per_match": "tilt_v"}
        )
        merged = base.join(c, how="inner")
        merged["rank_change"] = merged["rank_v0"] - merged["rank_v"]
        merged["tilt_delta_pp"] = (merged["tilt_v"] - merged["total_tilt_per_match"]) * 100
        movers = merged.reindex(merged["rank_change"].abs().sort_values(ascending=False).index)
        movers = movers[movers["rank_v0"].le(50) | movers["rank_v"].le(50)]
        print(f"\n{label} — top movers in/out of top-50:")
        print(
            movers.head(10)[
                ["player", "rank_v0", "rank_v", "rank_change", "tilt_delta_pp"]
            ].to_string()
        )


if __name__ == "__main__":
    main()
