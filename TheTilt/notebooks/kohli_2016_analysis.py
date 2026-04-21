"""
Kohli 2016 paradox — analysis notebook for the blog post.

Reproduces all numbers and plots embedded in
public/notes/kohli-2016-paradox.md so they can be refreshed whenever the
win-probability model is retrained or the underlying data changes.

Plots are written to public/notes/plots/ with descriptive filenames matching
the markdown's ![...](plots/...) references. No gridlines on any chart
(per blog editorial direction).
"""

# %% Imports and setup
import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="white", palette="muted")

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "processed"
PUBLIC_DATA_DIR = REPO_ROOT / "public" / "data"
PLOTS_DIR = REPO_ROOT / "public" / "notes" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

KOHLI_ID = "ba607b88"  # Cricsheet people-registry ID; matches slug v-kohli-ba607b88

deltas = pd.read_parquet(DATA_DIR / "deltas.parquet")
player_tilt = pd.read_parquet(DATA_DIR / "player_tilt.parquet")

deltas["season"] = deltas["season"].astype(str)
deltas["season_year"] = deltas["season"].apply(
    lambda s: int(s.split("/")[0]) if "/" in s else int(s) if s.isdigit() else None
)

print(f"Loaded {len(deltas):,} balls; Kohli batter_id check: "
      f"{(deltas['batter_id'] == KOHLI_ID).sum()} balls")


# %% 1. Kohli per-match table (full career)
def per_match_for_player(player_id: str) -> pd.DataFrame:
    """Per-match batting summary for one player across full dataset."""
    bat = deltas[deltas["batter_id"] == player_id].copy()
    bat["legal"] = (~bat["is_wide"]).astype(int)
    grp = (
        bat.groupby(["match_id", "date", "season_year", "innings",
                     "batting_team", "bowling_team"], observed=True)
        .agg(
            runs=("runs_batter", "sum"),
            balls=("legal", "sum"),
            tilt=("delta_wp", "sum"),
        )
        .reset_index()
        .sort_values("date")
    )
    grp["sr"] = (grp["runs"] / grp["balls"].clip(lower=1)) * 100
    return grp


# Pre-compute a single batter_id -> display-name lookup so we don't rescan deltas
# per comparison row in the scatter plot.
BATTER_NAMES = (
    deltas.groupby("batter_id", observed=True)["batter"]
    .agg(lambda x: x.mode().iloc[0] if len(x.mode()) else x.iloc[0])
    .to_dict()
)


kohli_matches = per_match_for_player(KOHLI_ID)
print(f"\nKohli total matches: {len(kohli_matches)}")
print(f"Kohli total runs: {kohli_matches['runs'].sum():,}")


# %% 2. Season summaries (2016 vs 2019, plus career)
def season_summary(matches: pd.DataFrame) -> pd.DataFrame:
    s = (
        matches.groupby("season_year")
        .agg(
            matches=("match_id", "nunique"),
            runs=("runs", "sum"),
            balls=("balls", "sum"),
            total_tilt=("tilt", "sum"),
        )
        .reset_index()
    )
    s["sr"] = (s["runs"] / s["balls"].clip(lower=1)) * 100
    s["tilt_per_match"] = s["total_tilt"] / s["matches"]
    s["tilt_per_ball"] = s["total_tilt"] / s["balls"].clip(lower=1)
    return s


kohli_seasons = season_summary(kohli_matches)
print("\nKohli season summary:")
print(kohli_seasons.to_string(index=False, float_format="{:.2f}".format))

s2016 = kohli_seasons[kohli_seasons["season_year"] == 2016].iloc[0]
s2019 = kohli_seasons[kohli_seasons["season_year"] == 2019].iloc[0]
print(f"\n2016 vs 2019 headline:")
print(f"  2016: {s2016['runs']} runs / {s2016['balls']} balls (SR {s2016['sr']:.1f}), "
      f"{s2016['matches']} matches, TILT/match {s2016['tilt_per_match']*100:+.2f}%")
print(f"  2019: {s2019['runs']} runs / {s2019['balls']} balls (SR {s2019['sr']:.1f}), "
      f"{s2019['matches']} matches, TILT/match {s2019['tilt_per_match']*100:+.2f}%")


# %% 3. 2016 match log with match_ids (for blog table links)
kohli_2016 = kohli_matches[kohli_matches["season_year"] == 2016].copy()
print("\n2016 match log (date, innings, runs(balls), TILT%, match_id):")
for _, r in kohli_2016.iterrows():
    print(f"  {r['date']:>10s} | inn {int(r['innings'])} | "
          f"{int(r['runs']):>3d}({int(r['balls']):>2d}) | "
          f"{r['tilt']*100:+7.2f}% | match_id={r['match_id']}")


# %% 4. Plot: kohli_match_by_match.png
fig, ax = plt.subplots(figsize=(11, 5))
xs = np.arange(len(kohli_2016))
tilts_pct = kohli_2016["tilt"].values * 100
colors = ["#4ade80" if t >= 0 else "#f87171" for t in tilts_pct]
bars = ax.bar(xs, tilts_pct, color=colors, edgecolor="none")
for x, t, runs, balls in zip(xs, tilts_pct, kohli_2016["runs"], kohli_2016["balls"]):
    label = f"{int(runs)}({int(balls)})"
    voff = 1.5 if t >= 0 else -1.5
    va = "bottom" if t >= 0 else "top"
    ax.text(x, t + voff, label, ha="center", va=va, fontsize=8, color="#444")
ax.axhline(0, color="black", linewidth=0.6)
ax.set_xticks(xs)
ax.set_xticklabels(
    [pd.to_datetime(d).strftime("%b %d") for d in kohli_2016["date"]],
    rotation=45, ha="right", fontsize=8,
)
ax.set_ylabel("Batting TILT (%)")
ax.set_title("Kohli IPL 2016: per-match batting TILT (bars labelled with runs(balls))")
ax.grid(False)
ax.set_ylim(min(tilts_pct.min() * 1.2, -10), max(tilts_pct.max() * 1.15, 10))
fig.tight_layout()
fig.savefig(PLOTS_DIR / "kohli_match_by_match.png", dpi=150, bbox_inches="tight")
print(f"\nSaved: {PLOTS_DIR / 'kohli_match_by_match.png'}")


# %% 5. Phase breakdown 2016 vs 2019
def phase_of_over(over: int) -> str:
    if over <= 5:
        return "Powerplay"
    if over <= 14:
        return "Middle"
    return "Death"


kohli_balls = deltas[deltas["batter_id"] == KOHLI_ID].copy()
kohli_balls["phase"] = kohli_balls["over"].apply(phase_of_over)
kohli_balls["legal"] = (~kohli_balls["is_wide"]).astype(int)

phase_seasons = (
    kohli_balls[kohli_balls["season_year"].isin([2016, 2019])]
    .groupby(["season_year", "phase"], observed=True)
    .agg(
        runs=("runs_batter", "sum"),
        balls=("legal", "sum"),
        tilt=("delta_wp", "sum"),
    )
    .reset_index()
)
phase_seasons["sr"] = (phase_seasons["runs"] / phase_seasons["balls"].clip(lower=1)) * 100
phase_seasons["tilt_pct"] = phase_seasons["tilt"] * 100
print("\nPhase breakdown (Kohli):")
print(phase_seasons.to_string(index=False, float_format="{:.2f}".format))

# %% 6. Plot: kohli_phase_breakdown.png
phase_order = ["Powerplay", "Middle", "Death"]
phase_seasons["phase"] = pd.Categorical(phase_seasons["phase"], categories=phase_order, ordered=True)
phase_seasons = phase_seasons.sort_values(["season_year", "phase"])

fig, ax = plt.subplots(figsize=(9, 5))
width = 0.36
x = np.arange(len(phase_order))
y_2016 = phase_seasons[phase_seasons["season_year"] == 2016].set_index("phase")["tilt_pct"].reindex(phase_order).values
y_2019 = phase_seasons[phase_seasons["season_year"] == 2019].set_index("phase")["tilt_pct"].reindex(phase_order).values

b1 = ax.bar(x - width / 2, y_2016, width, label="2016", color="#60a5fa", edgecolor="none")
b2 = ax.bar(x + width / 2, y_2019, width, label="2019", color="#fbbf24", edgecolor="none")

ax.bar_label(b1, fmt="%+.1f%%", padding=4, fontsize=9)
ax.bar_label(b2, fmt="%+.1f%%", padding=4, fontsize=9)
ax.axhline(0, color="black", linewidth=0.6)
ax.set_xticks(x)
ax.set_xticklabels(phase_order)
ax.set_ylabel("Batting TILT in phase (%)")
ax.set_title("Kohli phase-level TILT: 2016 vs 2019")
ax.grid(False)
ax.legend(frameon=False)
ymin = min(y_2016.min(), y_2019.min())
ymax = max(y_2016.max(), y_2019.max())
pad = max(abs(ymin), abs(ymax)) * 0.25
ax.set_ylim(ymin - pad, ymax + pad)
fig.tight_layout()
fig.savefig(PLOTS_DIR / "kohli_phase_breakdown.png", dpi=150, bbox_inches="tight")
print(f"Saved: {PLOTS_DIR / 'kohli_phase_breakdown.png'}")


# %% 7. Dot-ball tax
def dot_breakdown(season_year: int) -> dict:
    s = kohli_balls[(kohli_balls["season_year"] == season_year) & (kohli_balls["legal"] == 1)]
    n_total = len(s)
    is_dot = s["runs_batter"] == 0
    n_dots = is_dot.sum()
    dot_tilt = s.loc[is_dot, "delta_wp"].sum()
    score_tilt = s.loc[~is_dot, "delta_wp"].sum()
    return dict(
        season=season_year,
        n_balls=n_total,
        n_dots=int(n_dots),
        dot_pct=n_dots / max(n_total, 1) * 100,
        dot_tilt_pct=dot_tilt * 100,
        score_tilt_pct=score_tilt * 100,
    )


dot_2016 = dot_breakdown(2016)
dot_2019 = dot_breakdown(2019)
print(f"\nDot-ball breakdown:")
print(f"  2016: {dot_2016}")
print(f"  2019: {dot_2019}")

# %% 8. Plot: kohli_dot_ball_tax.png  (label-overlap fix: bar_label with padding + extended ylim)
categories = ["Dot ball TILT", "Scoring ball TILT"]
y_2016 = [dot_2016["dot_tilt_pct"], dot_2016["score_tilt_pct"]]
y_2019 = [dot_2019["dot_tilt_pct"], dot_2019["score_tilt_pct"]]

fig, ax = plt.subplots(figsize=(8.5, 5))
x = np.arange(len(categories))
width = 0.36
b1 = ax.bar(x - width / 2, y_2016, width, label="2016", color="#60a5fa", edgecolor="none")
b2 = ax.bar(x + width / 2, y_2019, width, label="2019", color="#fbbf24", edgecolor="none")
ax.bar_label(b1, fmt="%+.1f%%", padding=5, fontsize=9)
ax.bar_label(b2, fmt="%+.1f%%", padding=5, fontsize=9)
ax.axhline(0, color="black", linewidth=0.6)
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.set_ylabel("Total TILT contribution (%)")
ax.set_title("Kohli dot-ball tax: total TILT from dot vs scoring balls")
ax.grid(False)
ax.legend(frameon=False)
ymin = min(min(y_2016), min(y_2019))
ymax = max(max(y_2016), max(y_2019))
pad = max(abs(ymin), abs(ymax)) * 0.30
ax.set_ylim(ymin - pad, ymax + pad)
fig.tight_layout()
fig.savefig(PLOTS_DIR / "kohli_dot_ball_tax.png", dpi=150, bbox_inches="tight")
print(f"Saved: {PLOTS_DIR / 'kohli_dot_ball_tax.png'}")


# %% 9. Career timeline plot
fig, ax1 = plt.subplots(figsize=(11, 5))
seasons = kohli_seasons.sort_values("season_year")
xs = seasons["season_year"]
tilt_per_match_pct = seasons["tilt_per_match"] * 100

bar_colors = ["#fbbf24" if y == 2016 else "#60a5fa" for y in xs]
bars = ax1.bar(xs, tilt_per_match_pct, color=bar_colors, edgecolor="none", label="Batting TILT/match")
ax1.axhline(0, color="black", linewidth=0.6)
ax1.set_xlabel("IPL season")
ax1.set_ylabel("Batting TILT/match (%)", color="#1e3a8a")
ax1.tick_params(axis="y", labelcolor="#1e3a8a")
ax1.grid(False)

ax2 = ax1.twinx()
ax2.plot(xs, seasons["runs"], "o-", color="#7c3aed", label="Runs", linewidth=2)
ax2.set_ylabel("Runs in season", color="#7c3aed")
ax2.tick_params(axis="y", labelcolor="#7c3aed")
ax2.grid(False)

ax1.set_title("Kohli IPL career arc: runs (line) vs batting TILT/match (bars, 2016 highlighted)")
fig.tight_layout()
fig.savefig(PLOTS_DIR / "kohli_career_timeline.png", dpi=150, bbox_inches="tight")
print(f"Saved: {PLOTS_DIR / 'kohli_career_timeline.png'}")


# %% 10. Comparison player-seasons table — recompute current numbers
COMPARISON = [
    ("V Kohli", KOHLI_ID, 2016),
    ("V Kohli", KOHLI_ID, 2019),
    ("DA Warner", None, 2016),
    ("KL Rahul", None, 2018),
    ("F du Plessis", None, 2021),
    ("AM Rahane", None, 2013),
    ("JH Kallis", None, 2012),
    ("Ishan Kishan", None, 2022),
]


def find_player_id(display_name: str) -> Optional[str]:
    """Find batter_id by display name (uses the cached BATTER_NAMES lookup)."""
    candidates = pd.Series(BATTER_NAMES)
    matches = candidates[candidates == display_name]
    if len(matches) == 0:
        last = display_name.split()[-1]
        matches = candidates[candidates.str.endswith(" " + last)]
    if len(matches) == 0:
        return None
    if len(matches) == 1:
        return matches.index[0]
    # Pick the one with most balls
    counts = deltas[deltas["batter_id"].isin(matches.index)].groupby("batter_id", observed=True).size()
    return counts.idxmax()


comparison_rows = []
for display, pid, year in COMPARISON:
    if pid is None:
        pid = find_player_id(display)
        if pid is None:
            print(f"  WARN: could not find player_id for {display}")
            continue
    matches = per_match_for_player(pid)
    s = matches[matches["season_year"] == year]
    if len(s) == 0:
        print(f"  WARN: no {year} matches for {display}")
        continue
    runs = s["runs"].sum()
    balls = s["balls"].sum()
    n_match = s["match_id"].nunique()
    sr = runs / max(balls, 1) * 100
    tilt_pm_pct = s["tilt"].sum() / n_match * 100

    # Dot % and inn-2 ball %
    pb = deltas[(deltas["batter_id"] == pid) & (deltas["season_year"] == year) & (~deltas["is_wide"])]
    n_legal = len(pb)
    dot_pct = (pb["runs_batter"] == 0).sum() / max(n_legal, 1) * 100
    inn2_pct = (pb["innings"] == 2).sum() / max(n_legal, 1) * 100

    comparison_rows.append({
        "Player": display, "Season": year, "Runs": int(runs), "SR": round(sr, 1),
        "Matches": int(n_match), "TILT/match": round(tilt_pm_pct, 2),
        "Dot %": round(dot_pct, 1), "Inn 2 %": round(inn2_pct, 1),
    })

comparison_df = pd.DataFrame(comparison_rows)
print("\nComparison player-seasons (current model):")
print(comparison_df.to_string(index=False))


# %% 11. Plot: runs_vs_tilt_scatter.png — every IPL season with 400+ runs
legal_only = deltas[~deltas["is_wide"]]
season_player = (
    legal_only.groupby(["batter_id", "season_year"], observed=True)
    .agg(
        runs=("runs_batter", "sum"),
        balls=("runs_batter", "count"),
        tilt=("delta_wp", "sum"),
        n_match=("match_id", "nunique"),
    )
    .reset_index()
)
season_player["batter"] = season_player["batter_id"].map(BATTER_NAMES)
season_player["tilt_pm_pct"] = (season_player["tilt"] / season_player["n_match"]) * 100
qualifying = season_player[season_player["runs"] >= 400].copy()
print(f"\nQualifying season-players (400+ runs): {len(qualifying)}")
corr_r = qualifying["runs"].corr(qualifying["tilt_pm_pct"])
print(f"Pearson r (runs vs TILT/match): {corr_r:.3f}")

# Build a (player_name, season) -> (runs, tilt_pm_pct) lookup for the highlight loop
qual_idx = qualifying.set_index(["batter", "season_year"])

fig, ax = plt.subplots(figsize=(9, 6))
ax.scatter(qualifying["runs"], qualifying["tilt_pm_pct"], s=18, alpha=0.35, color="#94a3b8", edgecolor="none")
for _, row in comparison_df.iterrows():
    key = (row["Player"], row["Season"])
    if key not in qual_idx.index:
        continue
    sub = qual_idx.loc[key]
    if isinstance(sub, pd.DataFrame):
        sub = sub.iloc[0]
    x_val = sub["runs"]
    y_val = sub["tilt_pm_pct"]
    ax.scatter([x_val], [y_val], s=70, color="#ef4444", edgecolor="white", linewidth=1.2, zorder=3)
    ax.annotate(f"{row['Player']} {row['Season']}", (x_val, y_val), xytext=(7, 4),
                textcoords="offset points", fontsize=8.5, color="#1f2937")
ax.axhline(0, color="black", linewidth=0.6)
ax.set_xlabel("Runs in season")
ax.set_ylabel("Batting TILT/match (%)")
ax.set_title(f"IPL season-players (400+ runs): runs vs TILT/match (r = {corr_r:.2f})")
ax.grid(False)
fig.tight_layout()
fig.savefig(PLOTS_DIR / "runs_vs_tilt_scatter.png", dpi=150, bbox_inches="tight")
print(f"Saved: {PLOTS_DIR / 'runs_vs_tilt_scatter.png'}")


# %% 12. Credit-sharing plot for Kohli's 2016 centuries
# Find the centuries (>= 100 runs) in Kohli's 2016 matches
kohli_centuries_2016 = kohli_2016[kohli_2016["runs"] >= 100].copy()
print(f"\nKohli 2016 centuries: {len(kohli_centuries_2016)}")
print(kohli_centuries_2016[["date", "match_id", "runs", "balls", "tilt"]].to_string(index=False))


def innings_credit_breakdown(match_id, innings: int) -> pd.DataFrame:
    """Per-batter total delta_wp in a given match-innings."""
    inn = deltas[(deltas["match_id"] == match_id) & (deltas["innings"] == innings)]
    by_bat = (
        inn.groupby(["batter_id", "batter"], observed=True)
        .agg(tilt=("delta_wp", "sum"), balls=("runs_batter", "count"), runs=("runs_batter", "sum"))
        .reset_index()
        .sort_values("tilt", ascending=False)
    )
    return by_bat


# Build a panel of credit-sharing per century innings
fig, axes = plt.subplots(1, len(kohli_centuries_2016), figsize=(4 * len(kohli_centuries_2016), 4.5), sharey=True)
if len(kohli_centuries_2016) == 1:
    axes = [axes]

for ax, (_, century) in zip(axes, kohli_centuries_2016.iterrows()):
    breakdown = innings_credit_breakdown(century["match_id"], int(century["innings"]))
    breakdown["tilt_pct"] = breakdown["tilt"] * 100
    breakdown["label"] = breakdown["batter"] + breakdown.apply(
        lambda r: f"\n{int(r['runs'])}({int(r['balls'])})", axis=1
    )
    is_kohli = breakdown["batter_id"] == KOHLI_ID
    colors = ["#fbbf24" if k else "#60a5fa" for k in is_kohli]
    ax.bar(np.arange(len(breakdown)), breakdown["tilt_pct"], color=colors, edgecolor="none")
    ax.set_xticks(np.arange(len(breakdown)))
    ax.set_xticklabels(breakdown["label"], rotation=45, ha="right", fontsize=7)
    ax.axhline(0, color="black", linewidth=0.6)
    opp = century["bowling_team"]
    date_str = pd.to_datetime(century["date"]).strftime("%b %d")
    ax.set_title(f"{date_str} vs {opp}\nKohli {int(century['runs'])}({int(century['balls'])}) "
                 f"— TILT {century['tilt']*100:+.2f}%", fontsize=9)
    ax.set_ylabel("Batter TILT (%)")
    ax.grid(False)

fig.suptitle("Where the credit went in each Kohli 2016 century (Kohli highlighted)")
fig.tight_layout()
fig.savefig(PLOTS_DIR / "kohli_credit_sharing.png", dpi=150, bbox_inches="tight")
print(f"Saved: {PLOTS_DIR / 'kohli_credit_sharing.png'}")


# %% 13. Gujarat Lions ABD analysis (May 14 2016: Kohli 109 + ABD 129 vs GL)
GL_MATCH_ID = "980987"  # RCB vs GL, May 14 2016 (Kohli 109(55), ABD 129(53)). match_id is string-typed.
gl_inn = 1            # RCB batted first

inn = (
    deltas[(deltas["match_id"] == GL_MATCH_ID) & (deltas["innings"] == gl_inn)]
    .sort_values(["over", "ball"])
    .reset_index(drop=True)
)
print(f"\nGujarat Lions May 14 match (id={GL_MATCH_ID}, innings 1, RCB batting):")
print(f"  Total balls in innings: {len(inn)}")
print(f"  Batters: {inn['batter'].unique().tolist()}")

abd_balls = inn[inn["batter"].str.contains("Villiers", na=False, regex=False)].copy()
print(f"  ABD balls: {len(abd_balls)}, runs: {int(abd_balls['runs_batter'].sum())}")

if len(abd_balls):
    abd_balls = abd_balls.copy()
    abd_balls["cum_runs"] = abd_balls["runs_batter"].cumsum()
    abd_100 = abd_balls[abd_balls["cum_runs"] >= 100].head(1)
    if len(abd_100):
        abd_100_idx = abd_100.index[0]
        wp_at_abd_100 = inn.loc[abd_100_idx, "wp_after"]
        before = inn.loc[:abd_100_idx]
        after = inn.loc[abd_100_idx + 1:]
        kohli_runs_so_far = before.loc[before["batter_id"] == KOHLI_ID, "runs_batter"].sum()
        kohli_balls_so_far = (before["batter_id"] == KOHLI_ID).sum()
        kohli_tilt_so_far = before.loc[before["batter_id"] == KOHLI_ID, "delta_wp"].sum()
        kohli_tilt_after = after.loc[after["batter_id"] == KOHLI_ID, "delta_wp"].sum()
        print(f"  When ABD reached 100:")
        print(f"    RCB win prob: {wp_at_abd_100:.3f}")
        print(f"    Kohli to that point: {int(kohli_runs_so_far)} runs off {int(kohli_balls_so_far)} balls, "
              f"TILT {kohli_tilt_so_far*100:+.2f}%")
        print(f"    Kohli TILT from that ball forward: {kohli_tilt_after*100:+.2f}%")


# %% 14. Summary
print("\n" + "=" * 60)
print("SUMMARY (paste-ready numbers for the blog)")
print("=" * 60)
print(f"\n2016 vs 2019 headline:")
print(f"  Runs: {int(s2016['runs'])} vs {int(s2019['runs'])}")
print(f"  Balls: {int(s2016['balls'])} vs {int(s2019['balls'])}")
print(f"  SR: {s2016['sr']:.1f} vs {s2019['sr']:.1f}")
print(f"  Matches: {int(s2016['matches'])} vs {int(s2019['matches'])}")
print(f"  Batting TILT/match: {s2016['tilt_per_match']*100:+.2f}% vs {s2019['tilt_per_match']*100:+.2f}%")
print(f"  Total WPA: {s2016['total_tilt']*100:+.1f}% vs {s2019['total_tilt']*100:+.1f}%")
print(f"  TILT per ball: {s2016['tilt_per_ball']*100:+.3f}% vs {s2019['tilt_per_ball']*100:+.3f}%")
print(f"\nDot-ball summary:")
print(f"  2016: {dot_2016['n_dots']} dots ({dot_2016['dot_pct']:.1f}%), "
      f"dot-TILT {dot_2016['dot_tilt_pct']:+.1f}%, score-TILT {dot_2016['score_tilt_pct']:+.1f}%")
print(f"  2019: {dot_2019['n_dots']} dots ({dot_2019['dot_pct']:.1f}%), "
      f"dot-TILT {dot_2019['dot_tilt_pct']:+.1f}%, score-TILT {dot_2019['score_tilt_pct']:+.1f}%")
print(f"\nRuns vs TILT scatter Pearson r: {corr_r:.3f}")

plt.close("all")
