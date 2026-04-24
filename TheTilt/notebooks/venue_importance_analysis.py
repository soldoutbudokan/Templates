"""
Venue importance — counterfactual TILT under venue swaps.

For each featured cohort, relocates their home matches to alternate venues by
overriding `venue` in the featured-balls frame and re-running
`compute_ball_deltas` with the committed model. Plots are written to
public/notes/plots/; every number that appears in
public/notes/venue-importance.md is printed by the summary block at the end.

Cohorts:
  1. ABD, RCB 2016, M Chinnaswamy -> 9 whitelisted targets
  2. Kohli, RCB 2016, M Chinnaswamy (companion to #1 — same match set)
  3. Gayle, RCB career, M Chinnaswamy
  4. Dhoni, CSK career, Chepauk  (note: 2014 not viable — CSK played in UAE)
  5. Rohit, MI career, Wankhede
  6. Data-driven under-appreciated pick (resolved by a home/away gap filter)
"""

# %% Imports and setup
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from pipeline.compute_tilt import compute_ball_deltas

sns.set_theme(style="white", palette="muted")

# Site palette — matches public/notes/kohli-2016-paradox.md plots.
# Keep in sync across notebooks: innings_bias_analysis.py,
# innings_boundary_analysis.py, kohli_2016_analysis.py.
COLOR_POS = "#4ade80"     # green-400 — positive TILT
COLOR_NEG = "#f87171"     # red-400   — negative TILT
COLOR_BLUE = "#60a5fa"    # blue-400  — primary categorical
COLOR_AMBER = "#fbbf24"   # amber-400 — secondary categorical

DATA_DIR = REPO_ROOT / "data" / "processed"
MODELS_DIR = REPO_ROOT / "models"
PLOTS_DIR = REPO_ROOT / "public" / "notes" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Whitelisted target venues — 9 grounds with substantial pre-2021 data.
# Names MUST exactly match the model's learned categorical values.
CHINNASWAMY = "M Chinnaswamy Stadium"
WANKHEDE = "Wankhede Stadium"
EDEN = "Eden Gardens"
CHEPAUK = "MA Chidambaram Stadium, Chepauk"
UPPAL = "Rajiv Gandhi International Stadium, Uppal"
MOHALI = "Punjab Cricket Association IS Bindra Stadium, Mohali"
KOTLA = "Arun Jaitley Stadium"
SAWAI = "Sawai Mansingh Stadium"
MOTERA = "Narendra Modi Stadium, Ahmedabad"
TARGET_VENUES = [CHINNASWAMY, WANKHEDE, EDEN, CHEPAUK, UPPAL, MOHALI,
                 KOTLA, SAWAI, MOTERA]

PLAYER_IDS = {
    "ABD": "c4487b84",
    "Kohli": "ba607b88",
    "Gayle": "db584dad",
    "Dhoni": "4a8a2e3b",
    "Rohit": "740742ef",
}


# %% Load model and data
print("Loading model...")
with open(MODELS_DIR / "win_prob_lgbm.pkl", "rb") as f:
    model = pickle.load(f)

model_venue_cats = list(model.booster_.pandas_categorical[0])
missing = set(TARGET_VENUES) - set(model_venue_cats)
assert not missing, f"Whitelist venues missing from model: {missing}"
print(f"  model has {len(model_venue_cats)} venue categories; "
      f"all {len(TARGET_VENUES)} whitelist targets present")

print("Loading featured_balls.parquet...")
fb_raw = pd.read_parquet(DATA_DIR / "featured_balls.parquet")
fb = fb_raw[fb_raw["dls_method"].isna()].copy()
fb["season"] = fb["season"].astype(str)
fb["season_year"] = fb["season"].apply(
    lambda s: int(s.split("/")[0]) if "/" in s else int(s) if s.isdigit() else None
)
print(f"  {len(fb):,} rows after DLS filter "
      f"(dropped {len(fb_raw) - len(fb):,} DLS-affected rows)")

committed_deltas = pd.read_parquet(DATA_DIR / "deltas.parquet")


# %% Helpers
def select_cohort_matches(fb, player_id, team=None, year=None, home_venue=None):
    """Return match_ids where `player_id` batted for `team` in `year` at `home_venue`."""
    q = fb[fb["batter_id"] == player_id]
    if team is not None:
        q = q[q["batting_team"].isin(team if isinstance(team, (list, tuple)) else [team])]
    if year is not None:
        q = q[q["season_year"] == year]
    if home_venue is not None:
        q = q[q["venue"] == home_venue]
    return sorted(q["match_id"].unique().tolist())


def run_counterfactual(cohort_df, target_venue):
    """Override venue to `target_venue` on the whole cohort and re-run the model.

    Uses pd.Categorical with the model's full category list so LightGBM
    takes the learned categorical split path instead of the unknown-category
    default.
    """
    cf = cohort_df.copy()
    cf["venue"] = pd.Categorical([target_venue] * len(cf),
                                  categories=model_venue_cats)
    return compute_ball_deltas(model, cf)


def player_total_tilt_per_match(deltas_df, player_id):
    """Combined bat minus bowl TILT per match, using union-of-matches as denom."""
    bat = deltas_df[deltas_df["batter_id"] == player_id]
    bowl = deltas_df[deltas_df["bowler_id"] == player_id]
    match_union = set(bat["match_id"]) | set(bowl["match_id"])
    n = len(match_union)
    if n == 0:
        return 0.0, 0
    total = bat["delta_wp"].sum() - bowl["delta_wp"].sum()
    return total / n, n


def player_batting_stats(deltas_df, player_id):
    """Batting-only stats: TILT/match, runs, balls, matches."""
    bat = deltas_df[deltas_df["batter_id"] == player_id]
    n = bat["match_id"].nunique()
    if n == 0:
        return {"tilt_per_match": 0.0, "runs": 0, "balls": 0, "matches": 0}
    legal = bat[~bat["is_wide"]]
    return {
        "tilt_per_match": bat["delta_wp"].sum() / n,
        "runs": int(bat["runs_batter"].sum()),
        "balls": int(len(legal)),
        "matches": int(n),
    }


# %% Sanity replay
print("\n--- Sanity replay (ABD 2016 RCB Chinnaswamy home) ---")
abd_pid = PLAYER_IDS["ABD"]
abd16_home = select_cohort_matches(
    fb, abd_pid, team="Royal Challengers Bangalore",
    year=2016, home_venue=CHINNASWAMY,
)
abd16_df = fb[fb["match_id"].isin(abd16_home)].copy()
print(f"Cohort: {len(abd16_home)} matches, {len(abd16_df):,} balls")

replay = compute_ball_deltas(model, abd16_df.copy())
replay_sum = replay[replay["batter_id"] == abd_pid]["delta_wp"].sum()

committed_sum = committed_deltas[
    (committed_deltas["batter_id"] == abd_pid)
    & (committed_deltas["match_id"].isin(abd16_home))
]["delta_wp"].sum()

diff = abs(replay_sum - committed_sum)
print(f"  Replay sum:    {replay_sum:.10f}")
print(f"  Committed sum: {committed_sum:.10f}")
print(f"  |diff|:        {diff:.2e}")
assert diff < 1e-9, f"Replay sanity check FAILED (diff={diff})"
print("  OK — replay matches committed deltas to <1e-9")


# %% Cohort definitions
COHORTS = [
    {
        "key": "abd_2016",
        "label": "ABD, RCB 2016",
        "player_id": PLAYER_IDS["ABD"],
        "team": "Royal Challengers Bangalore",
        "year": 2016,
        "home_venue": CHINNASWAMY,
        "headline_target": CHEPAUK,
    },
    {
        "key": "kohli_2016",
        "label": "Kohli, RCB 2016",
        "player_id": PLAYER_IDS["Kohli"],
        "team": "Royal Challengers Bangalore",
        "year": 2016,
        "home_venue": CHINNASWAMY,
        "headline_target": CHEPAUK,
    },
    {
        "key": "gayle_rcb",
        "label": "Gayle, RCB career",
        "player_id": PLAYER_IDS["Gayle"],
        "team": "Royal Challengers Bangalore",
        "year": None,
        "home_venue": CHINNASWAMY,
        "headline_target": WANKHEDE,
    },
    {
        "key": "dhoni_csk",
        "label": "Dhoni, CSK career",
        "player_id": PLAYER_IDS["Dhoni"],
        "team": "Chennai Super Kings",
        "year": None,
        "home_venue": CHEPAUK,
        "headline_target": EDEN,
    },
    {
        "key": "rohit_mi",
        "label": "Rohit, MI career",
        "player_id": PLAYER_IDS["Rohit"],
        "team": "Mumbai Indians",
        "year": None,
        "home_venue": WANKHEDE,
        "headline_target": CHEPAUK,
    },
]


# %% Run full 9-venue sweep for each cohort
def sweep(cohort):
    match_ids = select_cohort_matches(
        fb, cohort["player_id"], cohort["team"],
        cohort["year"], cohort["home_venue"],
    )
    cohort_df = fb[fb["match_id"].isin(match_ids)].copy()
    print(f"\n[{cohort['key']}] {cohort['label']}: "
          f"{len(match_ids)} matches at {cohort['home_venue']}")

    results = []
    for tv in TARGET_VENUES:
        swapped = run_counterfactual(cohort_df, tv)
        tilt_pm, n = player_total_tilt_per_match(swapped, cohort["player_id"])
        bat = player_batting_stats(swapped, cohort["player_id"])
        results.append({
            "target_venue": tv,
            "is_home": tv == cohort["home_venue"],
            "tilt_per_match": tilt_pm,
            "bat_tilt_per_match": bat["tilt_per_match"],
            "bat_runs": bat["runs"],
            "bat_balls": bat["balls"],
            "matches": n,
        })

    df = pd.DataFrame(results)
    home_row = df[df["is_home"]].iloc[0]
    target_row = df[df["target_venue"] == cohort["headline_target"]].iloc[0]
    delta = target_row["tilt_per_match"] - home_row["tilt_per_match"]
    print(f"  Home ({cohort['home_venue']}): TILT/match = {home_row['tilt_per_match']*100:+.2f}%")
    print(f"  Swap ({cohort['headline_target']}): TILT/match = {target_row['tilt_per_match']*100:+.2f}%")
    print(f"  Delta: {delta*100:+.2f}pp")
    print(f"  Full sweep:")
    for _, r in df.iterrows():
        flag = "  (home)" if r["is_home"] else ""
        print(f"    {r['target_venue'][:55]:<55s}  "
              f"{r['tilt_per_match']*100:+7.2f}%  (bat {r['bat_tilt_per_match']*100:+7.2f}%){flag}")

    return df, match_ids


cohort_sweeps = {}
cohort_matches = {}
for c in COHORTS:
    df, mids = sweep(c)
    cohort_sweeps[c["key"]] = df
    cohort_matches[c["key"]] = mids


# %% Per-match paired TILT for ABD 2016 (used by match-by-match plot and ABD-vs-Kohli)
def per_match_paired(match_ids, player_id, home_venue, target_venue):
    """Return a DataFrame with one row per match: real TILT, swapped TILT, runs, balls."""
    rows = []
    for mid in match_ids:
        m_df = fb[fb["match_id"] == mid].copy()
        # real: re-score with actual venue (for clean sums in the same precision as swap)
        real = compute_ball_deltas(model, m_df.copy())
        # swap
        swapped = run_counterfactual(m_df, target_venue)

        def batter_stats(d):
            b = d[d["batter_id"] == player_id]
            legal = b[~b["is_wide"]]
            return (b["delta_wp"].sum(), int(b["runs_batter"].sum()), int(len(legal)))

        real_tilt, runs, balls = batter_stats(real)
        swap_tilt, _, _ = batter_stats(swapped)
        date = m_df["date"].iloc[0]
        opp = m_df[m_df["batting_team"] == m_df[m_df["batter_id"] == player_id]["batting_team"].iloc[0]]["bowling_team"].iloc[0] if (m_df["batter_id"] == player_id).any() else ""
        rows.append({
            "match_id": mid,
            "date": date,
            "opponent": opp,
            "runs": runs,
            "balls": balls,
            "real_bat_tilt": real_tilt,
            "swap_bat_tilt": swap_tilt,
        })
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


print("\n--- Per-match paired TILT, ABD 2016 (home -> Chepauk) ---")
abd16_paired = per_match_paired(cohort_matches["abd_2016"], abd_pid,
                                 CHINNASWAMY, CHEPAUK)
print(abd16_paired.to_string(index=False))

print("\n--- Per-match paired TILT, Kohli 2016 (home -> Chepauk) ---")
kohli16_paired = per_match_paired(cohort_matches["kohli_2016"],
                                    PLAYER_IDS["Kohli"], CHINNASWAMY, CHEPAUK)
print(kohli16_paired.to_string(index=False))


# %% Plot 1 — ABD 2016 match-by-match paired bars
def plot_match_by_match(paired_df, title, outfile):
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(paired_df))
    width = 0.38

    real_pct = paired_df["real_bat_tilt"].values * 100
    swap_pct = paired_df["swap_bat_tilt"].values * 100

    b1 = ax.bar(x - width / 2, real_pct, width, label="Real (Chinnaswamy)",
                color=COLOR_BLUE, edgecolor="none")
    b2 = ax.bar(x + width / 2, swap_pct, width, label="Swapped to Chepauk",
                color=COLOR_AMBER, edgecolor="none")

    for xi, r, s, runs, balls in zip(x, real_pct, swap_pct,
                                       paired_df["runs"], paired_df["balls"]):
        label = f"{int(runs)}({int(balls)})"
        top = max(r, s)
        ax.text(xi, top + 1.5, label, ha="center", va="bottom",
                fontsize=8, color="#444")

    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [pd.to_datetime(d).strftime("%b %d") for d in paired_df["date"]],
        rotation=45, ha="right", fontsize=8,
    )
    ax.set_ylabel("Batting TILT (%)")
    ax.set_title(title)
    ax.grid(False)
    ax.legend(frameon=False, loc="best")

    ymin = min(real_pct.min(), swap_pct.min())
    ymax = max(real_pct.max(), swap_pct.max())
    pad = max(abs(ymin), abs(ymax)) * 0.25
    ax.set_ylim(ymin - pad, ymax + pad)

    fig.tight_layout()
    fig.savefig(outfile, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {outfile}")


plot_match_by_match(
    abd16_paired,
    "ABD, RCB 2016 home matches: actual vs venue-swapped to Chepauk",
    PLOTS_DIR / "venue_abd_2016_match_by_match.png",
)


# %% Plot 2 — ABD vs Kohli 2016 paired comparison
def plot_abd_vs_kohli(abd_df, kohli_df, outfile):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for ax, (df, label, color_real, color_swap) in zip(
        axes,
        [(abd_df, "AB de Villiers", COLOR_BLUE, COLOR_AMBER),
         (kohli_df, "Virat Kohli", COLOR_BLUE, COLOR_AMBER)],
    ):
        x = np.arange(len(df))
        width = 0.38
        real_pct = df["real_bat_tilt"].values * 100
        swap_pct = df["swap_bat_tilt"].values * 100
        ax.bar(x - width / 2, real_pct, width, label="Real (Chinnaswamy)",
               color=color_real, edgecolor="none")
        ax.bar(x + width / 2, swap_pct, width, label="Swapped to Chepauk",
               color=color_swap, edgecolor="none")
        for xi, r, s, runs in zip(x, real_pct, swap_pct, df["runs"]):
            top = max(r, s)
            ax.text(xi, top + 1.5, f"{int(runs)}", ha="center", va="bottom",
                    fontsize=8, color="#444")
        ax.axhline(0, color="black", linewidth=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels([pd.to_datetime(d).strftime("%b %d") for d in df["date"]],
                            rotation=45, ha="right", fontsize=8)
        mean_real = real_pct.mean()
        mean_swap = swap_pct.mean()
        ax.set_title(f"{label}\navg real {mean_real:+.2f}%, swap {mean_swap:+.2f}%")
        ax.set_ylabel("Batting TILT per match (%)")
        ax.grid(False)

    axes[0].legend(frameon=False, loc="best")
    fig.suptitle("RCB's 2016 home matches: venue lever for ABD vs Kohli "
                 "(same matches, same swap)")
    fig.tight_layout()
    fig.savefig(outfile, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {outfile}")


plot_abd_vs_kohli(abd16_paired, kohli16_paired,
                   PLOTS_DIR / "venue_abd_vs_kohli_2016.png")


# %% Data-driven under-appreciated pick
# Find players with a moderate sample size whose observed home-vs-away batting
# TILT gap is largest. Then run the full counterfactual on the chosen pick.
print("\n--- Data-driven under-appreciated pick ---")

cd = committed_deltas
already = set(PLAYER_IDS.values())

# Per-player match count (from committed deltas as batter)
bat_match_n = (
    cd.groupby("batter_id")["match_id"].nunique().rename("n_matches")
)

# Per-player home venue = most-played venue as batter (DLS-filtered)
bat_fb = fb[["batter_id", "venue", "batting_team", "match_id"]]
bat_home = (
    bat_fb.groupby(["batter_id", "venue"])["match_id"].nunique()
    .reset_index(name="venue_matches")
    .sort_values(["batter_id", "venue_matches"], ascending=[True, False])
    .drop_duplicates("batter_id", keep="first")
    .set_index("batter_id")
)
# Per-player most-common team
bat_team = (
    bat_fb.groupby(["batter_id", "batting_team"])["match_id"].nunique()
    .reset_index(name="team_matches")
    .sort_values(["batter_id", "team_matches"], ascending=[True, False])
    .drop_duplicates("batter_id", keep="first")
    .set_index("batter_id")["batting_team"]
)

cand = pd.DataFrame(bat_match_n).join(bat_home).join(bat_team.rename("team"))
# Sample-size filter (wider than plan's 30-80 — most multi-team stars don't clear
# 60% home share, so we allow 30-150 matches to keep a reasonable pool)
cand = cand[(cand["n_matches"] >= 30) & (cand["n_matches"] <= 150)]
# Only candidates whose primary home venue is in our 9-venue swap whitelist
cand = cand[cand["venue"].isin(TARGET_VENUES)]
# Exclude already-featured
cand = cand[~cand.index.isin(already)]

# Map-based home-ball-share (avoids DataFrame/Series merge alignment issues)
home_map = cand["venue"].to_dict()
legal = fb[~fb["is_wide"]].copy()
legal["home_venue"] = legal["batter_id"].map(home_map)
legal_by = legal.groupby("batter_id").size().rename("total_balls")
legal_home = (
    legal.dropna(subset=["home_venue"])
    .loc[lambda d: d["venue"] == d["home_venue"]]
    .groupby("batter_id").size().rename("home_balls")
)
cand = cand.join(legal_by).join(legal_home).fillna({"home_balls": 0})
cand["home_ball_share"] = cand["home_balls"] / cand["total_balls"].clip(lower=1)

# Home vs away per-ball batting TILT — from committed deltas merged with venue tags
cd_venue = cd[["match_id", "batter_id", "delta_wp", "is_wide"]].merge(
    fb[["match_id", "venue"]].drop_duplicates(), on="match_id", how="left"
)
cd_venue = cd_venue[~cd_venue["is_wide"]]
cd_venue["home_venue"] = cd_venue["batter_id"].map(home_map)
cd_venue = cd_venue.dropna(subset=["home_venue"])
cd_venue["at_home"] = cd_venue["venue"] == cd_venue["home_venue"]

grouped = (
    cd_venue.groupby(["batter_id", "at_home"])["delta_wp"].mean().unstack()
)
grouped.columns = ["away_tilt_per_ball", "home_tilt_per_ball"]
cand = cand.join(grouped)
cand["home_away_gap"] = (
    cand["home_tilt_per_ball"] - cand["away_tilt_per_ball"]
)

# Minimum share so we're not picking someone with 5 home balls
cand = cand[cand["home_ball_share"] >= 0.25]
cand["name"] = cand.index.map(
    lambda pid: fb[fb["batter_id"] == pid]["batter"].mode().iloc[0]
    if (fb["batter_id"] == pid).any() else pid
)

print(f"\nCandidates after filters: {len(cand)}")
print("\nTop 15 by |home_away_gap|:")
top = cand.reindex(cand["home_away_gap"].abs().sort_values(ascending=False).index).head(15)
print(top[["name", "team", "venue", "n_matches", "home_ball_share",
          "home_tilt_per_ball", "away_tilt_per_ball", "home_away_gap"]]
      .to_string(float_format="{:.5f}".format))


# Pick the top candidate and run a full counterfactual sweep
pick_pid = top.index[0]
pick_row = top.loc[pick_pid]
pick_name = pick_row["name"]
pick_team = pick_row["team"]
pick_home = pick_row["venue"]

# Pick a target that's meaningfully different — use the venue most opposite in character.
# Simplest heuristic: if home is typically batter-friendly, swap to typically bowler-friendly.
# We'll just use Chepauk as a default target unless home=Chepauk, in which case use Wankhede.
pick_target = CHEPAUK if pick_home != CHEPAUK else WANKHEDE

pick_match_ids = select_cohort_matches(
    fb, pick_pid, team=pick_team, year=None, home_venue=pick_home,
)
print(f"\n--- Full swap sweep for pick: {pick_name} ({pick_team}), home={pick_home}, "
      f"{len(pick_match_ids)} home matches ---")
pick_df = fb[fb["match_id"].isin(pick_match_ids)].copy()
pick_results = []
for tv in TARGET_VENUES:
    sw = run_counterfactual(pick_df, tv)
    t, n = player_total_tilt_per_match(sw, pick_pid)
    pick_results.append({"target_venue": tv, "tilt_per_match": t, "matches": n,
                          "is_home": tv == pick_home})
pick_df_results = pd.DataFrame(pick_results)
print(pick_df_results.to_string(index=False))

# Store pick info for summary and adding to the dumbbell
PICK_COHORT = {
    "key": "pick",
    "label": f"{pick_name}, {pick_team[:20]} career",
    "player_id": pick_pid,
    "team": pick_team,
    "year": None,
    "home_venue": pick_home,
    "headline_target": pick_target,
}
cohort_sweeps["pick"] = pick_df_results
cohort_matches["pick"] = pick_match_ids


# %% Plot 3 — dumbbell across all 6 cohorts
def plot_dumbbell(cohorts_with_pick, outfile):
    fig, ax = plt.subplots(figsize=(11, 6))
    # Order top-to-bottom, largest |delta| first
    cdata = []
    for c in cohorts_with_pick:
        sweep = cohort_sweeps[c["key"]]
        home = sweep[sweep["target_venue"] == c["home_venue"]].iloc[0]["tilt_per_match"]
        swap = sweep[sweep["target_venue"] == c["headline_target"]].iloc[0]["tilt_per_match"]
        cdata.append({
            "label": f"{c['label']}\n{c['home_venue'].split(',')[0]} -> {c['headline_target'].split(',')[0]}",
            "home_tilt": home * 100,
            "swap_tilt": swap * 100,
            "delta": (swap - home) * 100,
        })
    cdata = sorted(cdata, key=lambda r: abs(r["delta"]), reverse=True)

    labels = [c["label"] for c in cdata]
    y = np.arange(len(cdata))
    home_vals = np.array([c["home_tilt"] for c in cdata])
    swap_vals = np.array([c["swap_tilt"] for c in cdata])

    for yi, h, s in zip(y, home_vals, swap_vals):
        color = COLOR_NEG if s < h else COLOR_POS
        ax.plot([h, s], [yi, yi], color=color, linewidth=2.2, zorder=1)
    ax.scatter(home_vals, y, s=90, color=COLOR_BLUE, zorder=2, label="Home (real)",
               edgecolor="white", linewidth=1.0)
    ax.scatter(swap_vals, y, s=90, color=COLOR_AMBER, zorder=2, label="Swapped venue",
               edgecolor="white", linewidth=1.0)

    for yi, h, s in zip(y, home_vals, swap_vals):
        ax.text(h, yi - 0.24, f"{h:+.2f}%", ha="center", fontsize=8, color="#1e40af")
        ax.text(s, yi + 0.30, f"{s:+.2f}%", ha="center", fontsize=8, color="#b45309")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=0.6)
    ax.set_xlabel("Total TILT per match (%)")
    ax.set_title("Venue lever: total TILT/match when home matches are relocated "
                 "(the trained model is simply asked to re-score the same ball states)")
    ax.grid(False)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(outfile, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {outfile}")


plot_dumbbell(COHORTS + [PICK_COHORT], PLOTS_DIR / "venue_dumbbell_six_cohorts.png")


# %% Summary block (paste-ready numbers for the blog)
print("\n" + "=" * 70)
print("SUMMARY — paste into public/notes/venue-importance.md")
print("=" * 70)

print("\n# 1. ABD 2016 RCB Chinnaswamy")
s = cohort_sweeps["abd_2016"]
h = s[s["target_venue"] == CHINNASWAMY].iloc[0]
t = s[s["target_venue"] == CHEPAUK].iloc[0]
print(f"  Matches: {int(h['matches'])}, cohort balls: {len(fb[fb['match_id'].isin(cohort_matches['abd_2016'])]):,}")
print(f"  Real (Chinnaswamy)   TILT/match: {h['tilt_per_match']*100:+.2f}% "
      f"(bat {h['bat_tilt_per_match']*100:+.2f}%, runs {int(h['bat_runs'])} off {int(h['bat_balls'])})")
print(f"  Swap (Chepauk)       TILT/match: {t['tilt_per_match']*100:+.2f}% "
      f"(bat {t['bat_tilt_per_match']*100:+.2f}%, runs {int(t['bat_runs'])} off {int(t['bat_balls'])})")
print(f"  Delta: {(t['tilt_per_match']-h['tilt_per_match'])*100:+.2f}pp")

print("\n# 2. Kohli 2016 RCB Chinnaswamy")
s = cohort_sweeps["kohli_2016"]
h = s[s["target_venue"] == CHINNASWAMY].iloc[0]
t = s[s["target_venue"] == CHEPAUK].iloc[0]
print(f"  Matches: {int(h['matches'])}")
print(f"  Real   TILT/match: {h['tilt_per_match']*100:+.2f}% (bat {h['bat_tilt_per_match']*100:+.2f}%)")
print(f"  Swap   TILT/match: {t['tilt_per_match']*100:+.2f}% (bat {t['bat_tilt_per_match']*100:+.2f}%)")
print(f"  Delta: {(t['tilt_per_match']-h['tilt_per_match'])*100:+.2f}pp")

print("\n# 3. Gayle RCB career Chinnaswamy")
s = cohort_sweeps["gayle_rcb"]
h = s[s["target_venue"] == CHINNASWAMY].iloc[0]
t = s[s["target_venue"] == WANKHEDE].iloc[0]
print(f"  Matches: {int(h['matches'])}")
print(f"  Real (Chinnaswamy) TILT/match: {h['tilt_per_match']*100:+.2f}%")
print(f"  Swap (Wankhede)    TILT/match: {t['tilt_per_match']*100:+.2f}%")
print(f"  Delta: {(t['tilt_per_match']-h['tilt_per_match'])*100:+.2f}pp")

print("\n# 4. Dhoni CSK career Chepauk")
s = cohort_sweeps["dhoni_csk"]
h = s[s["target_venue"] == CHEPAUK].iloc[0]
t = s[s["target_venue"] == EDEN].iloc[0]
print(f"  Matches: {int(h['matches'])}")
print(f"  Real (Chepauk) TILT/match: {h['tilt_per_match']*100:+.2f}%")
print(f"  Swap (Eden)    TILT/match: {t['tilt_per_match']*100:+.2f}%")
print(f"  Delta: {(t['tilt_per_match']-h['tilt_per_match'])*100:+.2f}pp")

print("\n# 5. Rohit MI career Wankhede")
s = cohort_sweeps["rohit_mi"]
h = s[s["target_venue"] == WANKHEDE].iloc[0]
t = s[s["target_venue"] == CHEPAUK].iloc[0]
print(f"  Matches: {int(h['matches'])}")
print(f"  Real (Wankhede) TILT/match: {h['tilt_per_match']*100:+.2f}%")
print(f"  Swap (Chepauk)  TILT/match: {t['tilt_per_match']*100:+.2f}%")
print(f"  Delta: {(t['tilt_per_match']-h['tilt_per_match'])*100:+.2f}pp")

print("\n# 6. Under-appreciated pick")
s = cohort_sweeps["pick"]
h = s[s["target_venue"] == PICK_COHORT["home_venue"]].iloc[0]
t = s[s["target_venue"] == PICK_COHORT["headline_target"]].iloc[0]
print(f"  {pick_name} ({pick_team}), home = {pick_home}")
print(f"  Matches: {int(h['matches'])}")
print(f"  Real  TILT/match: {h['tilt_per_match']*100:+.2f}%")
print(f"  Swap ({PICK_COHORT['headline_target']}) TILT/match: {t['tilt_per_match']*100:+.2f}%")
print(f"  Delta: {(t['tilt_per_match']-h['tilt_per_match'])*100:+.2f}pp")

print("\n# Per-match ABD 2016 (for the match-by-match blog table)")
for _, r in abd16_paired.iterrows():
    d = pd.to_datetime(r["date"]).strftime("%b %d")
    print(f"  {d} vs {r['opponent']:<28s}  "
          f"{int(r['runs']):>3d}({int(r['balls']):>2d})  "
          f"real {r['real_bat_tilt']*100:+7.2f}%  swap {r['swap_bat_tilt']*100:+7.2f}%  "
          f"Δ {(r['swap_bat_tilt']-r['real_bat_tilt'])*100:+6.2f}pp  id={r['match_id']}")

print("\n# Per-match Kohli 2016 (same matches as ABD — for comparison narrative)")
for _, r in kohli16_paired.iterrows():
    d = pd.to_datetime(r["date"]).strftime("%b %d")
    print(f"  {d}  {int(r['runs']):>3d}({int(r['balls']):>2d})  "
          f"real {r['real_bat_tilt']*100:+7.2f}%  swap {r['swap_bat_tilt']*100:+7.2f}%  "
          f"Δ {(r['swap_bat_tilt']-r['real_bat_tilt'])*100:+6.2f}pp")

print("\n# Full sweep tables (per-cohort, TILT/match % at each target venue)")
for c in COHORTS + [PICK_COHORT]:
    print(f"\n  [{c['key']}] {c['label']}")
    for _, r in cohort_sweeps[c["key"]].iterrows():
        marker = "  (home)" if r["is_home"] else ""
        print(f"    {r['target_venue'][:55]:<55s}  {r['tilt_per_match']*100:+7.2f}%{marker}")

print("\n=== DONE ===")
