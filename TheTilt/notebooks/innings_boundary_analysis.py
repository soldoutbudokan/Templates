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

Plots embedded in the blog: public/notes/innings-boundary.md.
"""

# %% Imports and setup
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

sns.set_theme(style="white", palette="muted")

# Site palette — matches public/notes/kohli-2016-paradox.md plots.
# Keep in sync across notebooks: venue_importance_analysis.py,
# innings_bias_analysis.py, kohli_2016_analysis.py.
COLOR_POS = "#4ade80"     # green-400 — positive TILT / favours batting-first
COLOR_NEG = "#f87171"     # red-400   — negative TILT / favours chasing side
COLOR_BLUE = "#60a5fa"    # blue-400  — primary categorical
COLOR_AMBER = "#fbbf24"   # amber-400 — secondary categorical

REPO_ROOT = Path(__file__).resolve().parents[1]
PLOTS_DIR = REPO_ROOT / "public" / "notes" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = REPO_ROOT / "data" / "processed"
MATCHES_DIR = REPO_ROOT / "public" / "data" / "matches"

deltas = pd.read_parquet(DATA_DIR / "deltas.parquet")

print(f"Loaded {len(deltas):,} balls across {deltas['match_id'].nunique()} matches")
print(f"Innings distribution: {deltas['innings'].value_counts().to_dict()}")


# %% Build per-match boundary-jump table
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

boundary = inn1_last.join(
    inn2_first[["wp_before", "balls_remaining", "wickets_in_hand", "target"]],
    how="inner",
    rsuffix="_inn2",
)

# wp / wp_after = batting-team-at-that-ball's perspective.
# End inn 1: wp_after = batting-first team's perspective.
# Start inn 2: wp_before is the chasing team — flip to batting-first view.
boundary["bf_wp_end_inn1"] = boundary["wp_after"]
boundary["bf_wp_start_inn2"] = 1.0 - boundary["wp_before_inn2"]
boundary["boundary_jump"] = boundary["bf_wp_start_inn2"] - boundary["bf_wp_end_inn1"]
boundary["abs_jump"] = boundary["boundary_jump"].abs()
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
print(boundary["abs_jump"].describe(percentiles=pct_levels).to_string())

for thresh_pp in [1, 3, 5, 10, 15, 20]:
    share = (boundary["abs_jump"] >= thresh_pp / 100).mean() * 100
    print(f"  |jump| >= {thresh_pp:>2d} pp: {share:5.1f}% of matches")

# Histogram: colour bars by sign (favours batting-first vs chasing).
fig, ax = plt.subplots(figsize=(9, 5))
pos_mask = boundary["boundary_jump"] >= 0
ax.hist(boundary.loc[~pos_mask, "boundary_jump"] * 100, bins=60,
        color=COLOR_NEG, edgecolor="white", alpha=0.85,
        label=f"favours chasing ({(~pos_mask).sum()} matches)")
ax.hist(boundary.loc[pos_mask, "boundary_jump"] * 100, bins=60,
        color=COLOR_POS, edgecolor="white", alpha=0.85,
        label=f"favours batting-first ({pos_mask.sum()} matches)")
ax.axvline(0, color="black", linewidth=0.8)
ax.axvline(boundary["boundary_jump"].mean() * 100, color=COLOR_BLUE,
           linestyle="--", linewidth=1.5,
           label=f"mean = {boundary['boundary_jump'].mean()*100:+.2f} pp")
ax.axvline(boundary["boundary_jump"].median() * 100, color=COLOR_AMBER,
           linestyle="--", linewidth=1.5,
           label=f"median = {boundary['boundary_jump'].median()*100:+.2f} pp")
ax.set_xlabel("Signed boundary jump (batting-first team's view, percentage points)")
ax.set_ylabel("Matches")
ax.set_title("Innings-boundary WP jump — signed distribution (all matches)")
ax.grid(False)
ax.legend()
fig.tight_layout()
fig.savefig(PLOTS_DIR / "innings_boundary_hist.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved: {PLOTS_DIR / 'innings_boundary_hist.png'}")


# %% 2. Directional bias
print("\n" + "=" * 60)
print("2. DIRECTIONAL BIAS")
print("=" * 60)

pos = (boundary["boundary_jump"] > 0).sum()
neg = (boundary["boundary_jump"] < 0).sum()
total = len(boundary)
print(f"Jump > 0 (favours batting-first): {pos:>4d} ({pos/total*100:5.1f}%)")
print(f"Jump < 0 (favours chasing):       {neg:>4d} ({neg/total*100:5.1f}%)")

t_stat, t_p = stats.ttest_1samp(boundary["boundary_jump"], 0.0)
print(f"\nOne-sample t-test (H0: mean jump = 0): t={t_stat:+.3f}, p={t_p:.3e}")


# %% 3. Covariate: |jump| vs innings-1 wickets lost
print("\n" + "=" * 60)
print("3. COVARIATE: |jump| vs innings-1 wickets lost")
print("=" * 60)

corr_w, p_w = stats.spearmanr(boundary["inn1_wickets_lost"], boundary["abs_jump"])
print(f"Spearman |jump| vs inn1 wkts lost:  rho = {corr_w:+.3f}, p = {p_w:.2e}")

# Median |jump| by wickets lost
by_wkts = boundary.groupby("inn1_wickets_lost").agg(
    n=("abs_jump", "size"),
    median_abs=("abs_jump", "median"),
).reset_index()
by_wkts = by_wkts[by_wkts["n"] >= 20]  # drop sparse buckets
print("\nMedian |jump| by innings-1 wickets lost (buckets with >=20 matches):")
print(by_wkts.to_string(index=False, float_format="{:.4f}".format))

fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(by_wkts["inn1_wickets_lost"], by_wkts["median_abs"] * 100,
       color=COLOR_BLUE, edgecolor="white", alpha=0.9)
for x, y, n in zip(by_wkts["inn1_wickets_lost"],
                   by_wkts["median_abs"] * 100, by_wkts["n"]):
    ax.text(x, y + 0.15, f"n={n}", ha="center", fontsize=8, color="#444")
ax.set_xlabel("Innings-1 wickets lost")
ax.set_ylabel("Median |boundary jump| (pp)")
ax.set_title("Boundary jump shrinks as innings 1 ends with more wickets lost")
ax.set_xticks(by_wkts["inn1_wickets_lost"])
ax.grid(False)
fig.tight_layout()
fig.savefig(PLOTS_DIR / "innings_boundary_by_wickets.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved: {PLOTS_DIR / 'innings_boundary_by_wickets.png'}")


# %% 4. Season trend
print("\n" + "=" * 60)
print("4. SEASON TREND")
print("=" * 60)

season_summary = boundary.groupby("season").agg(
    n=("boundary_jump", "size"),
    mean_signed=("boundary_jump", "mean"),
    median_abs=("abs_jump", "median"),
).reset_index()
print(season_summary.to_string(index=False, float_format="{:.4f}".format))

fig, ax = plt.subplots(figsize=(10, 5))
mean_pp = season_summary["mean_signed"] * 100
colors = [COLOR_POS if v > 0 else COLOR_NEG for v in mean_pp]
ax.bar(np.arange(len(season_summary)), mean_pp, color=colors, edgecolor="white")
ax.axhline(0, color="black", linewidth=0.6)
ax.set_xticks(np.arange(len(season_summary)))
ax.set_xticklabels(season_summary["season"], rotation=45, ha="right", fontsize=9)
ax.set_ylabel("Mean signed boundary jump (pp, batting-first POV)")
ax.set_title("The chasing-side bias has shrunk toward zero over time")
ax.grid(False)
fig.tight_layout()
fig.savefig(PLOTS_DIR / "innings_boundary_by_season.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved: {PLOTS_DIR / 'innings_boundary_by_season.png'}")


# %% 5. Ground-truth example: match 1527690
print("\n" + "=" * 60)
print("5. MATCH 1527690 — THE USER'S EXAMPLE")
print("=" * 60)

mj_path = MATCHES_DIR / "1527690.json"
mj = json.loads(mj_path.read_text())
balls = mj["balls"]
inn1 = [b for b in balls if b["inn"] == 1]
inn2 = [b for b in balls if b["inn"] == 2]

# Build a batting-first-POV wp series across the full match.
# Inn 1: wp is already batting-first POV.
# Inn 2: flip (1 - wp) because the JSON's wp is the currently-batting (chasing) team.
series = []
for b in inn1:
    series.append((1, b["over"], b["ball"], b["wp"] * 100, b["wp_after"] * 100))
for b in inn2:
    series.append((2, b["over"], b["ball"],
                   (1 - b["wp"]) * 100, (1 - b["wp_after"]) * 100))
df_m = pd.DataFrame(series, columns=["inn", "over", "ball", "bf_wp", "bf_wp_after"])

print(f"Teams: {mj['teams']}, winner: {mj['winner']}, season: {mj['season']}")
print(f"Batting-first team wp end of inn 1:   {df_m[df_m.inn==1].bf_wp_after.iloc[-1]:.2f}%")
print(f"Batting-first team wp start of inn 2: {df_m[df_m.inn==2].bf_wp.iloc[0]:.2f}%")
print(f"Boundary jump:                         "
      f"{df_m[df_m.inn==2].bf_wp.iloc[0] - df_m[df_m.inn==1].bf_wp_after.iloc[-1]:+.2f} pp")

# Plot with a dashed vertical line + two-segment colour.
fig, ax = plt.subplots(figsize=(10, 5))
idx = np.arange(len(df_m))
inn1_len = (df_m["inn"] == 1).sum()
ax.plot(idx[:inn1_len], df_m.loc[:inn1_len - 1, "bf_wp_after"],
        color=COLOR_POS, linewidth=1.8, label="Innings 1 (batting first)")
ax.plot(idx[inn1_len:], df_m.loc[inn1_len:, "bf_wp_after"],
        color=COLOR_NEG, linewidth=1.8, label="Innings 2 (defending)")
ax.axvline(inn1_len - 0.5, color="#666", linestyle="--", linewidth=1,
           label="innings break")
ax.set_ylim(0, 100)
ax.set_xlim(0, len(df_m) - 1)
ax.set_xlabel(f"Ball number (batting first: {mj['teams'][0]})")
ax.set_ylabel(f"Win probability (%) — {mj['teams'][0]}'s POV")
ax.set_title(f"Match 1527690: {mj['teams'][0]} vs {mj['teams'][1]} "
             f"— the 17 pp boundary jump")
ax.grid(False)
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig(PLOTS_DIR / "innings_boundary_match1527690.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved: {PLOTS_DIR / 'innings_boundary_match1527690.png'}")


# %% 6. Headline summary
print("\n" + "=" * 60)
print("6. HEADLINE NUMBERS (for blog prose)")
print("=" * 60)

print(f"Matches analysed:     {len(boundary):,}")
print(f"Median |jump|:        {boundary['abs_jump'].median()*100:.2f} pp")
print(f"Mean signed jump:     {boundary['boundary_jump'].mean()*100:+.2f} pp")
print(f"Share favouring chasing side: {(boundary['boundary_jump']<0).mean()*100:.1f}%")
print(f"Share with |jump|>=10pp:      {(boundary['abs_jump']>=0.10).mean()*100:.1f}%")
print(f"Share with |jump|>=15pp:      {(boundary['abs_jump']>=0.15).mean()*100:.1f}%")
print(f"t-test p-value:       {t_p:.3e}")
print(f"Spearman |jump| vs wickets lost: rho={corr_w:+.3f}")
print()
print(f"2025 mean signed jump: "
      f"{season_summary.loc[season_summary.season=='2025','mean_signed'].iloc[0]*100:+.2f} pp")
print(f"2026 mean signed jump: "
      f"{season_summary.loc[season_summary.season=='2026','mean_signed'].iloc[0]*100:+.2f} pp")
