"""
Second Innings TILT Bias Diagnostic (Issue #45)

Investigates whether TILT scores are systematically inflated for 2nd innings
performances due to asymmetric win probability dynamics. The model uses chase-only
features (target, runs_needed, required_run_rate) that are zero in innings 1,
and win probability resolves to 0/1 by end of innings 2, creating larger swings.
"""

# %% Imports and setup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

sns.set_theme(style="white", palette="muted")

# Site palette — matches public/notes/kohli-2016-paradox.md plots.
# Keep in sync across notebooks: venue_importance_analysis.py,
# innings_boundary_analysis.py, kohli_2016_analysis.py.
COLOR_POS = "#4ade80"     # green-400 — positive TILT
COLOR_NEG = "#f87171"     # red-400   — negative TILT
COLOR_BLUE = "#60a5fa"    # blue-400  — primary categorical (e.g. Innings 1)
COLOR_AMBER = "#fbbf24"   # amber-400 — secondary categorical (e.g. Innings 2)

PLOTS_DIR = Path(__file__).resolve().parents[1] / "public" / "notes" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
deltas = pd.read_parquet(DATA_DIR / "deltas.parquet")
player_tilt = pd.read_parquet(DATA_DIR / "player_tilt.parquet")

deltas["abs_delta_wp"] = deltas["delta_wp"].abs()
deltas["phase"] = pd.cut(
    deltas["over"],
    bins=[-1, 5, 14, 19],
    labels=["Powerplay (1-6)", "Middle (7-15)", "Death (16-20)"],
)

print(f"Loaded {len(deltas):,} balls across {deltas['match_id'].nunique()} matches")
print(f"Innings distribution: {deltas['innings'].value_counts().to_dict()}")


# %% 1. Headline Asymmetry
print("\n" + "=" * 60)
print("1. HEADLINE INNINGS ASYMMETRY")
print("=" * 60)

inn_stats = deltas.groupby("innings")["abs_delta_wp"].agg(
    ["count", "mean", "median", "std"]
)
for pct in [0.75, 0.90, 0.95, 0.99]:
    inn_stats[f"p{int(pct*100)}"] = deltas.groupby("innings")["abs_delta_wp"].quantile(pct)

print("\nSummary of |delta_wp| by innings:")
print(inn_stats.to_string())

ratio = inn_stats.loc[2, "mean"] / inn_stats.loc[1, "mean"]
print(f"\nMean |delta_wp| ratio (Inn 2 / Inn 1): {ratio:.2f}x")

# Statistical tests
inn1 = deltas.loc[deltas["innings"] == 1, "abs_delta_wp"]
inn2 = deltas.loc[deltas["innings"] == 2, "abs_delta_wp"]

t_stat, t_p = stats.ttest_ind(inn2, inn1, equal_var=False)
u_stat, u_p = stats.mannwhitneyu(inn2, inn1, alternative="greater")
pooled_sd = np.sqrt((inn1.var() + inn2.var()) / 2)
cohens_d = (inn2.mean() - inn1.mean()) / pooled_sd

print(f"\nWelch t-test: t={t_stat:.2f}, p={t_p:.2e}")
print(f"Wilcoxon rank-sum: U={u_stat:.0f}, p={u_p:.2e}")
print(f"Cohen's d: {cohens_d:.3f}")

# Plot: density of |delta_wp| by innings
fig, ax = plt.subplots(figsize=(8, 5))
inn_colors = {1: COLOR_BLUE, 2: COLOR_AMBER}
for inn in [1, 2]:
    data = deltas.loc[deltas["innings"] == inn, "abs_delta_wp"]
    data_clipped = data[data > 0.001]  # avoid log(0) issues
    ax.hist(data_clipped, bins=100, alpha=0.5, density=True,
            color=inn_colors[inn], label=f"Innings {inn}")
    ax.axvline(data.mean(), linestyle="--", alpha=0.8,
               color=inn_colors[inn], label=f"Inn {inn} mean={data.mean():.4f}")
ax.set_xscale("log")
ax.set_xlabel("|delta_wp|")
ax.set_ylabel("Density")
ax.set_title("Distribution of |delta_wp| by Innings")
ax.grid(False)
ax.legend()
fig.tight_layout()
fig.savefig(PLOTS_DIR / "innings_delta_density.png", dpi=150)
print(f"\nSaved: {PLOTS_DIR / 'innings_delta_density.png'}")


# %% 2. Distribution Deep-Dive
print("\n" + "=" * 60)
print("2. DISTRIBUTION DEEP-DIVE")
print("=" * 60)

percentiles = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 0.999]
pct_table = pd.DataFrame(index=[f"p{p*100:.1f}" for p in percentiles])
for inn in [1, 2]:
    data = deltas.loc[deltas["innings"] == inn, "abs_delta_wp"]
    pct_table[f"Inn {inn}"] = [data.quantile(p) for p in percentiles]
pct_table["Ratio (2/1)"] = pct_table["Inn 2"] / pct_table["Inn 1"].replace(0, np.nan)

print("\nPercentile comparison of |delta_wp|:")
print(pct_table.to_string(float_format="{:.5f}".format))

# QQ plot
fig, ax = plt.subplots(figsize=(6, 6))
q1 = np.quantile(inn1, np.linspace(0, 1, 200))
q2 = np.quantile(inn2, np.linspace(0, 1, 200))
ax.scatter(q1, q2, s=10, alpha=0.6, color=COLOR_BLUE)
max_val = max(q1.max(), q2.max())
ax.plot([0, max_val], [0, max_val], "--", color=COLOR_NEG, alpha=0.7, label="y = x (no bias)")
ax.set_xlabel("Innings 1 quantiles |delta_wp|")
ax.set_ylabel("Innings 2 quantiles |delta_wp|")
ax.set_title("QQ Plot: Innings 2 vs Innings 1 |delta_wp|")
ax.grid(False)
ax.legend()
fig.tight_layout()
# QQ plot is a diagnostic; not embedded in the blog. Skipping save.


# %% 3. Top-N Performance Innings Breakdown
print("\n" + "=" * 60)
print("3. TOP-N PERFORMANCE INNINGS BREAKDOWN")
print("=" * 60)

# Batting: aggregate per (batter, match), determine innings
bat_match = (
    deltas.groupby(["batter_id", "match_id"])
    .agg(tilt=("delta_wp", "sum"), innings=("innings", "first"), batter=("batter", "first"))
    .reset_index()
    .sort_values("tilt", ascending=False)
)

# Bowling: aggregate per (bowler, match)
bowl_match = (
    deltas.groupby(["bowler_id", "match_id"])
    .agg(tilt=("delta_wp", lambda x: -x.sum()), innings=("innings", "first"), bowler=("bowler", "first"))
    .reset_index()
    .sort_values("tilt", ascending=False)
)

ns = [10, 25, 50, 100, 200, 500]
results = []
for n in ns:
    bat_pct = (bat_match.head(n)["innings"] == 2).mean() * 100
    bowl_pct = (bowl_match.head(n)["innings"] == 2).mean() * 100
    results.append({"N": n, "Batting % Inn 2": bat_pct, "Bowling % Inn 2": bowl_pct})
    print(f"Top {n:>3d}: Batting {bat_pct:.0f}% Inn 2 | Bowling {bowl_pct:.0f}% Inn 2")

results_df = pd.DataFrame(results)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(results_df["N"], results_df["Batting % Inn 2"], "o-",
        color=COLOR_BLUE, label="Batting")
ax.plot(results_df["N"], results_df["Bowling % Inn 2"], "s-",
        color=COLOR_AMBER, label="Bowling")
ax.axhline(50, color="gray", linestyle="--", alpha=0.5, label="50% (no bias)")
ax.set_xlabel("Top N performances")
ax.set_ylabel("% from Innings 2")
ax.set_title("Innings 2 Dominance in Top-N Performances")
ax.set_ylim(0, 105)
ax.grid(False)
ax.legend()
fig.tight_layout()
fig.savefig(PLOTS_DIR / "innings_topn_share.png", dpi=150)
print(f"\nSaved: {PLOTS_DIR / 'innings_topn_share.png'}")


# %% 4. Win Probability Trajectory by Over
print("\n" + "=" * 60)
print("4. WIN PROBABILITY TRAJECTORY BY OVER")
print("=" * 60)

over_stats = deltas.groupby(["innings", "over"]).agg(
    wp_mean=("wp_before", "mean"),
    wp_std=("wp_before", "std"),
    abs_delta_mean=("abs_delta_wp", "mean"),
    n=("delta_wp", "count"),
).reset_index()

# Plot 1: WP trajectory with variance ribbon
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
for i, inn in enumerate([1, 2]):
    d = over_stats[over_stats["innings"] == inn]
    ax = axes[i]
    c = COLOR_BLUE if inn == 1 else COLOR_AMBER
    ax.plot(d["over"] + 1, d["wp_mean"], "-", color=c, linewidth=2)
    ax.fill_between(
        d["over"] + 1,
        (d["wp_mean"] - d["wp_std"]).clip(0, 1),
        (d["wp_mean"] + d["wp_std"]).clip(0, 1),
        color=c,
        alpha=0.3,
    )
    ax.set_xlabel("Over")
    ax.set_ylabel("Win Probability (batting team)")
    ax.set_title(f"Innings {inn}: WP Trajectory (mean +/- 1 SD)")
    ax.set_xlim(1, 20)
    ax.set_ylim(0, 1)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.3)
    ax.grid(False)
fig.tight_layout()
# WP trajectory is a diagnostic; not embedded in the blog. Skipping save.

# Plot 2: Mean |delta_wp| by over
fig, ax = plt.subplots(figsize=(8, 5))
for inn in [1, 2]:
    d = over_stats[over_stats["innings"] == inn]
    ax.plot(d["over"] + 1, d["abs_delta_mean"], "o-",
            color=inn_colors[inn], label=f"Innings {inn}", markersize=5)
ax.set_xlabel("Over")
ax.set_ylabel("Mean |delta_wp|")
ax.set_title("Mean Win Probability Swing per Ball by Over")
ax.set_xlim(1, 20)
ax.grid(False)
ax.legend()
fig.tight_layout()
fig.savefig(PLOTS_DIR / "innings_volatility_by_over.png", dpi=150)
print(f"Saved: {PLOTS_DIR / 'innings_volatility_by_over.png'}")


# %% 5. Phase-Level Bias
print("\n" + "=" * 60)
print("5. PHASE-LEVEL BIAS")
print("=" * 60)

phase_stats = deltas.groupby(["innings", "phase"])["abs_delta_wp"].agg(["mean", "median", "count"]).reset_index()
phase_pivot = phase_stats.pivot(index="phase", columns="innings", values="mean")
phase_pivot["Ratio (2/1)"] = phase_pivot[2] / phase_pivot[1]

print("\nMean |delta_wp| by phase and innings:")
print(phase_pivot.to_string(float_format="{:.5f}".format))

fig, ax = plt.subplots(figsize=(8, 5))
phase_plot = phase_stats.copy()
phase_plot["innings"] = phase_plot["innings"].map({1: "Innings 1", 2: "Innings 2"})
sns.barplot(data=phase_plot, x="phase", y="mean", hue="innings", ax=ax,
            palette={"Innings 1": COLOR_BLUE, "Innings 2": COLOR_AMBER})
ax.set_xlabel("Match Phase")
ax.set_ylabel("Mean |delta_wp|")
ax.set_title("Mean Win Probability Swing by Phase and Innings")
ax.grid(False)
# Annotate ratios
for i, phase in enumerate(phase_pivot.index):
    r = phase_pivot.loc[phase, "Ratio (2/1)"]
    ax.text(i, phase_pivot.loc[phase, 2] + 0.0005, f"{r:.1f}x", ha="center", fontsize=10, fontweight="bold")
fig.tight_layout()
fig.savefig(PLOTS_DIR / "innings_phase_ratio.png", dpi=150)
print(f"Saved: {PLOTS_DIR / 'innings_phase_ratio.png'}")


# %% 6. Career-Level Impact
print("\n" + "=" * 60)
print("6. CAREER-LEVEL IMPACT")
print("=" * 60)

# Batting: % of balls faced in innings 2 per batter
bat_inn_pct = (
    deltas.groupby("batter_id")
    .agg(
        total_balls=("innings", "count"),
        inn2_balls=("innings", lambda x: (x == 2).sum()),
    )
    .reset_index()
)
bat_inn_pct["pct_inn2"] = bat_inn_pct["inn2_balls"] / bat_inn_pct["total_balls"]

# Merge with player_tilt
career = player_tilt.rename(columns={"player_id": "batter_id"}).merge(
    bat_inn_pct, on="batter_id", how="inner"
)
# Filter to 30+ matches
career_30 = career[career["total_matches"] >= 30].copy()

r_pearson, p_pearson = stats.pearsonr(career_30["pct_inn2"], career_30["batting_tilt_per_match"])
r_spearman, p_spearman = stats.spearmanr(career_30["pct_inn2"], career_30["batting_tilt_per_match"])

print(f"\nBatting TILT vs % innings-2 balls (players with 30+ matches, n={len(career_30)}):")
print(f"  Pearson:  r={r_pearson:.3f}, p={p_pearson:.3e}")
print(f"  Spearman: r={r_spearman:.3f}, p={p_spearman:.3e}")

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(career_30["pct_inn2"], career_30["batting_tilt_per_match"],
           alpha=0.5, s=20, color=COLOR_BLUE)
# Regression line
z = np.polyfit(career_30["pct_inn2"], career_30["batting_tilt_per_match"], 1)
x_line = np.linspace(career_30["pct_inn2"].min(), career_30["pct_inn2"].max(), 100)
ax.plot(x_line, np.polyval(z, x_line), "-", color=COLOR_NEG, alpha=0.8, label=f"r={r_pearson:.3f}")
ax.set_xlabel("% of career balls in Innings 2")
ax.set_ylabel("Batting TILT per match")
ax.set_title("Career Batting TILT vs Innings 2 Exposure (30+ matches)")
ax.grid(False)
ax.legend()
fig.tight_layout()
fig.savefig(PLOTS_DIR / "innings_career_correlation.png", dpi=150)
print(f"Saved: {PLOTS_DIR / 'innings_career_correlation.png'}")


# %% 7. Simulated Normalization
print("\n" + "=" * 60)
print("7. SIMULATED NORMALIZATION")
print("=" * 60)

overall_mean = deltas["abs_delta_wp"].mean()
inn_means = deltas.groupby("innings")["abs_delta_wp"].mean()
scale_factors = overall_mean / inn_means
print(f"\nScale factors: Innings 1 = {scale_factors[1]:.4f}, Innings 2 = {scale_factors[2]:.4f}")

deltas["normalized_delta_wp"] = deltas["delta_wp"] * deltas["innings"].map(scale_factors)

# Re-aggregate top 50 batting with normalized deltas
bat_match_norm = (
    deltas.groupby(["batter_id", "match_id"])
    .agg(
        tilt_raw=("delta_wp", "sum"),
        tilt_norm=("normalized_delta_wp", "sum"),
        innings=("innings", "first"),
        batter=("batter", "first"),
    )
    .reset_index()
)

top50_raw = set(bat_match_norm.nlargest(50, "tilt_raw")[["batter_id", "match_id"]].apply(tuple, axis=1))
top50_norm = set(bat_match_norm.nlargest(50, "tilt_norm")[["batter_id", "match_id"]].apply(tuple, axis=1))
overlap = len(top50_raw & top50_norm)
print(f"\nTop 50 batting: {overlap}/50 overlap between raw and normalized rankings")
print(f"  {50 - overlap} performances would leave the top 50 with normalization")

raw_inn2 = bat_match_norm.nlargest(50, "tilt_raw")["innings"].eq(2).mean() * 100
norm_inn2 = bat_match_norm.nlargest(50, "tilt_norm")["innings"].eq(2).mean() * 100
print(f"  Raw top 50: {raw_inn2:.0f}% innings 2")
print(f"  Normalized top 50: {norm_inn2:.0f}% innings 2")

# Career-level rank correlation before/after normalization
career_norm_bat = (
    deltas.groupby("batter_id")
    .agg(
        raw_tilt=("delta_wp", "sum"),
        norm_tilt=("normalized_delta_wp", "sum"),
        matches=("match_id", "nunique"),
    )
    .reset_index()
)
career_norm_bat["raw_per_match"] = career_norm_bat["raw_tilt"] / career_norm_bat["matches"]
career_norm_bat["norm_per_match"] = career_norm_bat["norm_tilt"] / career_norm_bat["matches"]
career_norm_30 = career_norm_bat[career_norm_bat["matches"] >= 30]

rho, p_rho = stats.spearmanr(career_norm_30["raw_per_match"], career_norm_30["norm_per_match"])
print(f"\nCareer TILT rank correlation (raw vs normalized, 30+ matches): rho={rho:.4f}, p={p_rho:.2e}")

# Biggest movers
career_norm_30 = career_norm_30.copy()
career_norm_30["raw_rank"] = career_norm_30["raw_per_match"].rank(ascending=False)
career_norm_30["norm_rank"] = career_norm_30["norm_per_match"].rank(ascending=False)
career_norm_30["rank_change"] = career_norm_30["raw_rank"] - career_norm_30["norm_rank"]
career_norm_30 = career_norm_30.merge(
    player_tilt.rename(columns={"player_id": "batter_id"})[["batter_id", "player"]],
    on="batter_id", how="left",
)
print("\nBiggest risers with normalization (moved UP in rankings):")
risers = career_norm_30.nlargest(10, "rank_change")[["player", "raw_rank", "norm_rank", "rank_change"]]
print(risers.to_string(index=False))

print("\nBiggest fallers with normalization (moved DOWN in rankings):")
fallers = career_norm_30.nsmallest(10, "rank_change")[["player", "raw_rank", "norm_rank", "rank_change"]]
print(fallers.to_string(index=False))


# %% 8. Summary
print("\n" + "=" * 60)
print("8. SUMMARY")
print("=" * 60)

print(f"""
SECOND INNINGS TILT BIAS DIAGNOSTIC
====================================

HEADLINE:
  Mean |delta_wp|: Innings 1 = {inn_stats.loc[1, 'mean']:.5f}, Innings 2 = {inn_stats.loc[2, 'mean']:.5f}
  Ratio (Inn 2 / Inn 1): {ratio:.2f}x
  Cohen's d: {cohens_d:.3f}

TOP PERFORMANCES:
  Top 50 batting:  {(bat_match.head(50)['innings'] == 2).mean()*100:.0f}% from innings 2
  Top 50 bowling:  {(bowl_match.head(50)['innings'] == 2).mean()*100:.0f}% from innings 2

PHASE BREAKDOWN:
  Powerplay ratio (Inn 2/Inn 1): {phase_pivot.loc['Powerplay (1-6)', 'Ratio (2/1)']:.2f}x
  Middle ratio (Inn 2/Inn 1):    {phase_pivot.loc['Middle (7-15)', 'Ratio (2/1)']:.2f}x
  Death ratio (Inn 2/Inn 1):     {phase_pivot.loc['Death (16-20)', 'Ratio (2/1)']:.2f}x

CAREER CORRELATION (batting TILT vs % inn-2 balls):
  Pearson r = {r_pearson:.3f} (p = {p_pearson:.3e})

NORMALIZATION IMPACT:
  Top 50 batting innings-2 share: {raw_inn2:.0f}% (raw) -> {norm_inn2:.0f}% (normalized)
  Career rank correlation (raw vs normalized): rho = {rho:.4f}

Plots saved to: {PLOTS_DIR}/
""")

plt.close("all")
