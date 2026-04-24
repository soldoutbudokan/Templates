"""
Innings-Boundary Win-Probability Jump Diagnostic (Issue #56)

Quantifies the discontinuity in the LightGBM win-probability model at the
innings-1 -> innings-2 boundary. For every match, computes a signed
"boundary jump" from the batting-first team's perspective:

    boundary_jump = (1 - wp_before[first ball of innings 2])
                  -      wp_after[last ball of innings 1]

A positive jump means the batting-first team's model-estimated win prob
*rose* instantaneously at the innings switch with zero balls of play. A
symmetrically-calibrated model should produce jumps centered near zero
with small magnitude.

Reference: pipeline/build_features.py initializes innings-2 features
(target, runs_needed, required_run_rate, balls_remaining, wickets_in_hand)
when innings 2 starts, restructuring the feature vector in one step. The
LightGBM model trained in pipeline/train_win_prob.py has learned separate
patterns per innings, and the discontinuity shows up as a visual cliff on
match.html.

Ground-truth example: match 1527690 (2026, PBKS vs SRH) — batting-first
team's wp goes 76.5% -> 93.7% (+17.2 pp) across the boundary.
"""

# %% Imports and setup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

sns.set_theme(style="white", palette="muted")

PLOTS_DIR = Path(__file__).resolve().parent / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
deltas = pd.read_parquet(DATA_DIR / "deltas.parquet")

print(f"Loaded {len(deltas):,} balls across {deltas['match_id'].nunique()} matches")
print(f"Innings distribution: {deltas['innings'].value_counts().to_dict()}")


# %% Build per-match boundary-jump table
# For each match, find last ball of innings 1 and first ball of innings 2,
# then compute the batting-first team's wp jump across the boundary.

deltas_sorted = deltas.sort_values(["match_id", "innings", "ball_number"])

inn1_last = (
    deltas_sorted[deltas_sorted["innings"] == 1]
    .groupby("match_id")
    .tail(1)
    .set_index("match_id")
)
inn2_first = (
    deltas_sorted[deltas_sorted["innings"] == 2]
    .groupby("match_id")
    .head(1)
    .set_index("match_id")
)

# Inner-join so we only keep matches with both innings present.
boundary = inn1_last.join(
    inn2_first[["wp_before", "balls_remaining", "wickets_in_hand", "target"]],
    how="inner",
    rsuffix="_inn2",
)

# wp convention: wp / wp_after is the *currently batting* team's wp.
# End of inn 1: wp_after is already batting-first team's perspective.
# Start of inn 2: wp_before is the chasing team — flip to batting-first view.
boundary["bf_wp_end_inn1"] = boundary["wp_after"]
boundary["bf_wp_start_inn2"] = 1.0 - boundary["wp_before_inn2"]
boundary["boundary_jump"] = boundary["bf_wp_start_inn2"] - boundary["bf_wp_end_inn1"]
boundary["abs_jump"] = boundary["boundary_jump"].abs()

# Context: innings-1 total, wickets lost at end of inn 1
boundary["inn1_total"] = boundary["runs_scored"]
boundary["inn1_wickets_lost"] = 10 - boundary["wickets_in_hand"]

print(f"\nMatches with both innings: {len(boundary):,}")
print(f"Mean signed jump:   {boundary['boundary_jump'].mean():+.4f} ({boundary['boundary_jump'].mean()*100:+.2f} pp)")
print(f"Median signed jump: {boundary['boundary_jump'].median():+.4f} ({boundary['boundary_jump'].median()*100:+.2f} pp)")
print(f"Mean |jump|:        {boundary['abs_jump'].mean():.4f} ({boundary['abs_jump'].mean()*100:.2f} pp)")
print(f"Median |jump|:      {boundary['abs_jump'].median():.4f} ({boundary['abs_jump'].median()*100:.2f} pp)")


# %% 1. Magnitude distribution
print("\n" + "=" * 60)
print("1. MAGNITUDE DISTRIBUTION OF |boundary_jump|")
print("=" * 60)

pct_levels = [0.50, 0.75, 0.90, 0.95, 0.99]
summary = boundary["abs_jump"].describe(percentiles=pct_levels)
print(summary.to_string())

# Share of matches with meaningfully large boundary jumps
for thresh_pp in [1, 3, 5, 10, 15, 20]:
    share = (boundary["abs_jump"] >= thresh_pp / 100).mean() * 100
    print(f"  |jump| >= {thresh_pp:>2d} pp: {share:5.1f}% of matches")

# Histogram of signed jump
fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(boundary["boundary_jump"] * 100, bins=80, color="steelblue", edgecolor="white")
ax.axvline(0, color="black", linestyle="-", linewidth=0.8)
ax.axvline(boundary["boundary_jump"].mean() * 100, color="red", linestyle="--",
           label=f"mean = {boundary['boundary_jump'].mean()*100:+.2f} pp")
ax.axvline(boundary["boundary_jump"].median() * 100, color="darkorange", linestyle="--",
           label=f"median = {boundary['boundary_jump'].median()*100:+.2f} pp")
ax.set_xlabel("Signed boundary jump (batting-first team's view, percentage points)")
ax.set_ylabel("Matches")
ax.set_title("Innings-Boundary WP Jump — Signed Distribution (all matches)")
ax.legend()
fig.tight_layout()
fig.savefig(PLOTS_DIR / "innings_boundary_hist.png", dpi=150)
print(f"\nSaved: {PLOTS_DIR / 'innings_boundary_hist.png'}")


# %% 2. Directional bias
print("\n" + "=" * 60)
print("2. DIRECTIONAL BIAS")
print("=" * 60)

pos = (boundary["boundary_jump"] > 0).sum()
neg = (boundary["boundary_jump"] < 0).sum()
zero = (boundary["boundary_jump"] == 0).sum()
total = len(boundary)
print(f"Jump > 0 (favors batting-first): {pos:>4d} ({pos/total*100:5.1f}%)")
print(f"Jump < 0 (favors chasing):       {neg:>4d} ({neg/total*100:5.1f}%)")
print(f"Jump = 0:                        {zero:>4d} ({zero/total*100:5.1f}%)")

t_stat, t_p = stats.ttest_1samp(boundary["boundary_jump"], 0.0)
print(f"\nOne-sample t-test (H0: mean jump = 0): t={t_stat:+.3f}, p={t_p:.3e}")
print("  Interpretation: significant non-zero mean => systematic directional bias")


# %% 3. Covariates: does the jump depend on innings-1 state?
print("\n" + "=" * 60)
print("3. COVARIATES OF |boundary_jump|")
print("=" * 60)

# Innings-1 total
corr_total, p_total = stats.spearmanr(boundary["inn1_total"], boundary["abs_jump"])
print(f"Spearman |jump| vs inn1 total:      rho = {corr_total:+.3f}, p = {p_total:.2e}")

# Innings-1 wickets lost
corr_w, p_w = stats.spearmanr(boundary["inn1_wickets_lost"], boundary["abs_jump"])
print(f"Spearman |jump| vs inn1 wkts lost:  rho = {corr_w:+.3f}, p = {p_w:.2e}")

# By season
season_summary = (
    boundary.groupby("season")
    .agg(n=("boundary_jump", "size"),
         mean_signed=("boundary_jump", "mean"),
         median_abs=("abs_jump", "median"))
    .sort_index()
)
print("\nBy season:")
print(season_summary.to_string(float_format="{:.4f}".format))

# Scatter: inn1 total vs signed jump
fig, ax = plt.subplots(figsize=(9, 5))
ax.scatter(boundary["inn1_total"], boundary["boundary_jump"] * 100,
           alpha=0.3, s=12, color="steelblue")
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xlabel("Innings-1 total runs")
ax.set_ylabel("Boundary jump (pp)")
ax.set_title("Boundary Jump vs Innings-1 Total")
fig.tight_layout()
fig.savefig(PLOTS_DIR / "innings_boundary_by_total.png", dpi=150)
print(f"\nSaved: {PLOTS_DIR / 'innings_boundary_by_total.png'}")


# %% 4. Match 1527690 ground-truth sanity check
print("\n" + "=" * 60)
print("4. MATCH 1527690 SANITY CHECK (the user's example)")
print("=" * 60)

m = deltas_sorted[deltas_sorted["match_id"] == "1527690"]
if len(m) == 0:
    print("WARNING: match 1527690 not in deltas.parquet")
else:
    m1 = m[m["innings"] == 1].tail(3)
    m2 = m[m["innings"] == 2].head(3)
    cols = ["innings", "over", "ball", "runs_scored", "wickets_in_hand",
            "wp_before", "wp_after", "delta_wp"]
    print("Last 3 balls of innings 1:")
    print(m1[cols].to_string(index=False))
    print("\nFirst 3 balls of innings 2:")
    print(m2[cols].to_string(index=False))

    end1 = m1["wp_after"].iloc[-1]
    start2_bf = 1 - m2["wp_before"].iloc[0]
    jump = start2_bf - end1
    print(f"\nBatting-first team's wp at end of inn 1: {end1*100:.2f}%")
    print(f"Batting-first team's wp at start of inn 2: {start2_bf*100:.2f}%")
    print(f"Boundary jump: {jump*100:+.2f} pp")
    print("(expected approx +17.2 pp from match JSON ground truth)")


# %% 5. Headline recommendation
print("\n" + "=" * 60)
print("5. HEADLINE RECOMMENDATION")
print("=" * 60)

median_abs_pp = boundary["abs_jump"].median() * 100
mean_signed_pp = boundary["boundary_jump"].mean() * 100
bias_significant = t_p < 0.01

print(f"Median |jump|:       {median_abs_pp:.2f} pp")
print(f"Mean signed jump:    {mean_signed_pp:+.2f} pp  (t-test p = {t_p:.2e})")
print(f"Directional bias:    {'YES' if bias_significant else 'no'}")
print()

if median_abs_pp < 3.0 and not bias_significant:
    print("=> Marker-only fix is sufficient:")
    print("   boundary jumps are small & symmetric; add a dashed vertical")
    print("   'innings break' line on match.html and break the chart")
    print("   segments. No model changes.")
elif median_abs_pp >= 3.0 or bias_significant:
    print("=> Structural fix justified:")
    print("   boundary jumps are large or systematically biased. Redesign")
    print("   features in pipeline/build_features.py to remove the innings")
    print("   discontinuity (e.g., continuous score-diff feature, projected")
    print("   total in innings 1), then retrain the model.")
else:
    print("=> Continuity-offset hack is the pragmatic middle path:")
    print("   apply a per-match offset to innings-2 wp values so the chart")
    print("   line joins at the boundary. TILT deltas unchanged.")
