"""
All-Rounders Underrating Diagnostic (Issue #96)

Investigates whether TILT systematically underrates all-rounders compared to
specialist batters and bowlers. Hypothesis: dual-role players have their
match-by-match impact split across two phases, dampening the per-match
variance and reducing their representation at the top of career rankings.

Mirrors the structure of `notebooks/innings_bias_analysis.py`.
"""

# %% Imports and setup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set_theme(style="white", palette="muted")

COLOR_POS = "#4ade80"     # green-400 — positive TILT
COLOR_NEG = "#f87171"     # red-400   — negative TILT
COLOR_BLUE = "#60a5fa"    # blue-400  — batter cohort
COLOR_AMBER = "#fbbf24"   # amber-400 — all-rounder cohort
COLOR_PURPLE = "#a78bfa"  # violet-400 — bowler cohort

PLOTS_DIR = Path(__file__).resolve().parents[1] / "public" / "notes" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
deltas = pd.read_parquet(DATA_DIR / "deltas.parquet")
player_tilt = pd.read_parquet(DATA_DIR / "player_tilt.parquet")

# Filter to ranked-eligible players (mirrors export_json: total_matches >= 10).
MIN_MATCHES = 10
qualified = player_tilt[player_tilt["total_matches"] >= MIN_MATCHES].copy()
print(f"Loaded {len(qualified)} qualified players (>= {MIN_MATCHES} matches)")


# %% 1. Role classification (mirrors pipeline/export_json.py)
qualified["role"] = "batter"
ball_ratio = qualified["bowling_balls"] / qualified["batting_balls"].replace(0, 1)
qualified.loc[
    (qualified["bowling_balls"] >= 50)
    & (qualified["batting_balls"] >= 50)
    & ball_ratio.between(0.3, 3.0),
    "role",
] = "all-rounder"
qualified.loc[
    (qualified["bowling_balls"] >= 100)
    & (qualified["bowling_balls"] > qualified["batting_balls"] * 1.5),
    "role",
] = "bowler"

print("\n" + "=" * 60)
print("1. ROLE COHORT SIZES")
print("=" * 60)
role_counts = qualified["role"].value_counts()
print(role_counts.to_string())
print(f"\nAll-rounders are {role_counts.get('all-rounder', 0) / len(qualified) * 100:.1f}% of the qualified pool")


# %% 2. Top-N representation by role
print("\n" + "=" * 60)
print("2. TOP-N CAREER TILT BY ROLE")
print("=" * 60)

ranked = qualified.sort_values("total_tilt_per_match", ascending=False).reset_index(drop=True)

ns = [10, 25, 50, 100, 200]
topn_records = []
for n in ns:
    head = ranked.head(n)
    role_share = head["role"].value_counts(normalize=True) * 100
    topn_records.append({
        "N": n,
        "batter_pct": role_share.get("batter", 0),
        "all_rounder_pct": role_share.get("all-rounder", 0),
        "bowler_pct": role_share.get("bowler", 0),
        "ar_count": int((head["role"] == "all-rounder").sum()),
    })
    print(
        f"Top {n:>3d}: "
        f"batters {role_share.get('batter', 0):.0f}% | "
        f"all-rounders {role_share.get('all-rounder', 0):.0f}% "
        f"({(head['role'] == 'all-rounder').sum()} of {n}) | "
        f"bowlers {role_share.get('bowler', 0):.0f}%"
    )

pool_share = role_counts / len(qualified) * 100
print(f"\nFor reference, pool share — batters {pool_share.get('batter', 0):.0f}%, "
      f"all-rounders {pool_share.get('all-rounder', 0):.0f}%, "
      f"bowlers {pool_share.get('bowler', 0):.0f}%")

# Plot: top-N representation
topn_df = pd.DataFrame(topn_records)
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(topn_df["N"], topn_df["batter_pct"], "o-", color=COLOR_BLUE, label="Batters")
ax.plot(topn_df["N"], topn_df["all_rounder_pct"], "s-", color=COLOR_AMBER, label="All-rounders")
ax.plot(topn_df["N"], topn_df["bowler_pct"], "^-", color=COLOR_PURPLE, label="Bowlers")
ax.axhline(pool_share.get("all-rounder", 0), color=COLOR_AMBER, linestyle="--", alpha=0.5, label="all-rounder pool share")
ax.set_xlabel("Top N by career TILT/match")
ax.set_ylabel("% of cohort")
ax.set_title("Role mix in the top-N TILT leaders")
ax.set_ylim(0, 100)
ax.grid(False)
ax.legend()
fig.tight_layout()
fig.savefig(PLOTS_DIR / "all_rounders_topn_share.png", dpi=150)
print(f"\nSaved: {PLOTS_DIR / 'all_rounders_topn_share.png'}")


# %% 3. Career TILT distribution by role
print("\n" + "=" * 60)
print("3. CAREER TILT DISTRIBUTION BY ROLE")
print("=" * 60)

dist_stats = qualified.groupby("role")["total_tilt_per_match"].agg(["count", "mean", "median", "std"])
for pct in [0.50, 0.75, 0.90, 0.95]:
    dist_stats[f"p{int(pct*100)}"] = qualified.groupby("role")["total_tilt_per_match"].quantile(pct)
print(dist_stats.to_string(float_format="{:.5f}".format))

# Compare top-quintile means
top_quintile = {}
for role in ["batter", "all-rounder", "bowler"]:
    sub = qualified[qualified["role"] == role]["total_tilt_per_match"]
    threshold = sub.quantile(0.80)
    top_quintile[role] = sub[sub >= threshold].mean()
print("\nMean TILT/match in the top quintile of each role:")
for role, val in top_quintile.items():
    print(f"  {role:<12} {val:.5f}")

# Plot: violin/box of distributions
fig, ax = plt.subplots(figsize=(8, 5))
order = ["batter", "all-rounder", "bowler"]
palette = {"batter": COLOR_BLUE, "all-rounder": COLOR_AMBER, "bowler": COLOR_PURPLE}
sns.boxplot(data=qualified, x="role", y="total_tilt_per_match", order=order,
            palette=palette, ax=ax, showfliers=False, linewidth=1.2)
ax.axhline(0, color="gray", linestyle="--", alpha=0.4)
ax.set_xlabel("Role")
ax.set_ylabel("Career TILT per match")
ax.set_title("Career TILT/match distribution by role")
ax.grid(False)
fig.tight_layout()
fig.savefig(PLOTS_DIR / "all_rounders_distribution.png", dpi=150)
print(f"\nSaved: {PLOTS_DIR / 'all_rounders_distribution.png'}")


# %% 4. Match-level TILT variance (the 'cancellation' question)
print("\n" + "=" * 60)
print("4. MATCH-LEVEL VARIANCE BY ROLE")
print("=" * 60)

# Compute per-(player, match) total TILT = batting delta_wp - bowling delta_wp.
bat_match = (
    deltas.groupby(["batter_id", "match_id"])["delta_wp"].sum()
    .reset_index().rename(columns={"batter_id": "player_id", "delta_wp": "bat_tilt"})
)
bowl_match = (
    deltas.groupby(["bowler_id", "match_id"])["delta_wp"].sum()
    .reset_index().rename(columns={"bowler_id": "player_id", "delta_wp": "bowl_tilt_raw"})
)
bowl_match["bowl_tilt"] = -bowl_match["bowl_tilt_raw"]
bowl_match = bowl_match.drop(columns=["bowl_tilt_raw"])

match_tilt = pd.merge(bat_match, bowl_match, on=["player_id", "match_id"], how="outer").fillna(0)
match_tilt["total_tilt"] = match_tilt["bat_tilt"] + match_tilt["bowl_tilt"]

role_lookup = qualified.set_index("player_id")["role"].to_dict()
match_tilt["role"] = match_tilt["player_id"].map(role_lookup)
match_tilt = match_tilt[match_tilt["role"].notna()]

variance_stats = match_tilt.groupby("role")["total_tilt"].agg(["count", "mean", "std"])
for pct in [0.95, 0.99, 0.995]:
    variance_stats[f"p{pct*100:g}"] = match_tilt.groupby("role")["total_tilt"].quantile(pct)
print(variance_stats.to_string(float_format="{:.4f}".format))

# Top single-match performances (>= 0.30 TILT)
hi_threshold = 0.30
hi_share = (match_tilt.assign(big=lambda d: d["total_tilt"] >= hi_threshold)
            .groupby("role")["big"].mean() * 100)
print(f"\n% of matches per cohort with TILT >= {hi_threshold} (a 'GOAT-tier' performance):")
print(hi_share.to_string(float_format="{:.3f}".format))

# Plot: density of match-tilt by role (positive tail)
fig, ax = plt.subplots(figsize=(8, 5))
for role, color in [("batter", COLOR_BLUE), ("all-rounder", COLOR_AMBER), ("bowler", COLOR_PURPLE)]:
    data = match_tilt.loc[match_tilt["role"] == role, "total_tilt"]
    data = data[(data > 0.01) & (data < 0.6)]
    ax.hist(data, bins=80, alpha=0.45, density=True, color=color, label=role.title())
ax.axvline(hi_threshold, color="gray", linestyle="--", alpha=0.7, label=f"GOAT-tier ≥ {hi_threshold}")
ax.set_xlabel("Per-match total TILT")
ax.set_ylabel("Density")
ax.set_title("Match-TILT distribution by role (positive tail)")
ax.grid(False)
ax.legend()
fig.tight_layout()
fig.savefig(PLOTS_DIR / "all_rounders_match_distribution.png", dpi=150)
print(f"\nSaved: {PLOTS_DIR / 'all_rounders_match_distribution.png'}")


# %% 5. The cancellation effect: bat vs bowl per match for all-rounders
print("\n" + "=" * 60)
print("5. CANCELLATION: BAT vs BOWL TILT PER MATCH (ALL-ROUNDERS)")
print("=" * 60)

ar_matches = match_tilt[(match_tilt["role"] == "all-rounder")
                        & ((match_tilt["bat_tilt"] != 0) | (match_tilt["bowl_tilt"] != 0))].copy()

# What fraction of all-rounder matches have one role positive and the other negative?
ar_matches["bat_pos"] = ar_matches["bat_tilt"] > 0
ar_matches["bowl_pos"] = ar_matches["bowl_tilt"] > 0
both_pos = ((ar_matches["bat_pos"]) & (ar_matches["bowl_pos"])).mean() * 100
both_neg = ((~ar_matches["bat_pos"]) & (~ar_matches["bowl_pos"])).mean() * 100
mixed = 100 - both_pos - both_neg
print(f"All-rounder match outcomes: both-positive {both_pos:.1f}% | both-negative {both_neg:.1f}% | mixed {mixed:.1f}%")
print("(In 'mixed' matches the two contributions partially cancel.)")

# Compare summed |bat| + |bowl| (specialist counterfactual) vs combined total
ar_matches["abs_sum"] = ar_matches["bat_tilt"].abs() + ar_matches["bowl_tilt"].abs()
ar_matches["combined"] = ar_matches["total_tilt"].abs()
mean_abs = ar_matches["abs_sum"].mean()
mean_comb = ar_matches["combined"].mean()
ratio = mean_abs / max(mean_comb, 1e-9)
print(f"\nMean |bat| + |bowl| per match (specialist counterfactual): {mean_abs:.4f}")
print(f"Mean combined |bat + bowl|: {mean_comb:.4f}")
print(f"Cancellation drag: combined is {(1 - mean_comb / mean_abs) * 100:.1f}% smaller than the specialist counterfactual")

# Plot: scatter of bat vs bowl TILT per match, with cancellation diagonal
fig, ax = plt.subplots(figsize=(7, 6))
sample = ar_matches.sample(min(len(ar_matches), 4000), random_state=42)
ax.scatter(sample["bat_tilt"], sample["bowl_tilt"], s=8, alpha=0.35, color=COLOR_AMBER)
lim = max(sample["bat_tilt"].abs().max(), sample["bowl_tilt"].abs().max()) * 0.85
ax.plot([-lim, lim], [lim, -lim], "--", color=COLOR_NEG, alpha=0.6, label="Cancellation line (bat + bowl = 0)")
ax.axhline(0, color="gray", linewidth=0.5)
ax.axvline(0, color="gray", linewidth=0.5)
ax.set_xlabel("Bat TILT in match")
ax.set_ylabel("Bowl TILT in match")
ax.set_title("All-rounder match contributions: bat vs bowl")
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.legend()
ax.grid(False)
fig.tight_layout()
fig.savefig(PLOTS_DIR / "all_rounders_cancellation.png", dpi=150)
print(f"\nSaved: {PLOTS_DIR / 'all_rounders_cancellation.png'}")


# %% 6. Headline summary block (numbers to embed in prose)
print("\n" + "=" * 60)
print("6. SUMMARY (numbers for prose)")
print("=" * 60)
n_pool = len(qualified)
n_ar = int(role_counts.get("all-rounder", 0))
print(f"Pool: {n_pool} ranked players. All-rounders: {n_ar} ({n_ar / n_pool * 100:.1f}%).")

top50 = ranked.head(50)
top50_ar = int((top50["role"] == "all-rounder").sum())
expected = n_ar / n_pool * 50
print(f"Top-50 all-rounders: {top50_ar} (expected at pool share: {expected:.1f}). "
      f"Ratio observed/expected = {top50_ar / max(expected, 0.01):.2f}.")

print(f"All-rounder mean career TILT/match: {dist_stats.loc['all-rounder', 'mean']:.5f}")
print(f"Batter mean career TILT/match:       {dist_stats.loc['batter', 'mean']:.5f}")
print(f"Bowler mean career TILT/match:       {dist_stats.loc['bowler', 'mean']:.5f}")

print(f"All-rounder mixed-sign matches: {mixed:.1f}% (one role positive, other negative)")
print(f"Cancellation drag: {(1 - mean_comb / mean_abs) * 100:.1f}% reduction vs specialist counterfactual")
print(f"GOAT-tier (>= {hi_threshold}) match share — batters: {hi_share.get('batter', 0):.3f}%, "
      f"all-rounders: {hi_share.get('all-rounder', 0):.3f}%, "
      f"bowlers: {hi_share.get('bowler', 0):.3f}%")
