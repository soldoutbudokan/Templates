# %% Compact summary of how key players move across scenarios.
"""
Runs the same five scenarios as `ranking_scenarios.py` but produces a tracking
table — for ABD, Ashwin, Kohli, Rohit, Bumrah, Narine, etc., how does
their rank/floor/total change across scenarios?

Also reports: how many pure batters appear in each scenario's Top 10 floor.
"""

from notebooks.ranking_scenarios import (
    load_deltas, aggregate_player, aggregate_match_perf,
    scenario_baseline, scenario_telescoping, scenario_per_role_centering,
    scenario_exclude_dls, scenario_winsorize,
)

import pandas as pd

TRACK_PLAYERS = [
    "AB de Villiers", "R Ashwin", "V Kohli", "RG Sharma", "MS Dhoni",
    "CH Gayle", "DA Warner", "JC Buttler", "SK Raina",
    "JJ Bumrah", "SP Narine", "SL Malinga", "Rashid Khan", "B Kumar",
    "YS Chahal", "Harbhajan Singh",
    "PD Salt", "Shubman Gill", "Suryakumar Yadav",
]

SCENARIOS = [
    ("A: Baseline", scenario_baseline),
    ("B: Telescoping", scenario_telescoping),
    ("C: Per-role", scenario_per_role_centering),
    ("D: No DLS", scenario_exclude_dls),
    ("E: Winsorize", scenario_winsorize),
]


def main():
    deltas, meta = load_deltas()

    rows_by_player = {p: {} for p in TRACK_PLAYERS}
    pure_batter_count = {}
    role_label = {}

    for name, fn in SCENARIOS:
        transformed = fn(deltas, meta)
        player = aggregate_player(transformed, name)
        # Sort by floor for ranking
        player = player.sort_values("tilt_ci_lower_90", ascending=False).reset_index(drop=True)
        player["floor_rank"] = player.index + 1
        # Sort by total
        player_total = player.sort_values("total_tilt", ascending=False).reset_index(drop=True)
        player_total["total_rank"] = player_total.index + 1

        for p in TRACK_PLAYERS:
            sub = player[player["player"] == p]
            if len(sub) == 0:
                rows_by_player[p][name] = None
                continue
            r = sub.iloc[0]
            tot_rank = player_total[player_total["player"] == p].iloc[0]["total_rank"]
            rows_by_player[p][name] = {
                "floor_rank": int(r["floor_rank"]),
                "floor": r["tilt_ci_lower_90"],
                "tpm": r["shrunk_total_tilt_per_match"],
                "total": r["total_tilt"],
                "total_rank": int(tot_rank),
            }

        # Count pure batters in top 10 floor
        top10 = player.head(10)
        is_pure_batter = (top10["bowling_balls"] < 50) & (top10["batting_balls"] >= 50)
        is_pure_bowler = (top10["batting_balls"] < 50) & (top10["bowling_balls"] >= 50)
        pure_batter_count[name] = {
            "pure_batter": int(is_pure_batter.sum()),
            "pure_bowler": int(is_pure_bowler.sum()),
            "all_round": int(((top10["batting_balls"] >= 50) & (top10["bowling_balls"] >= 50)).sum()),
        }

    print()
    print("=" * 100)
    print("  COMPOSITION OF TOP 10 BY TILT FLOOR (pure-batters / pure-bowlers / all-rounders)")
    print("=" * 100)
    print(f"  {'Scenario':<22} {'Pure batters':>14} {'Pure bowlers':>14} {'All-rounders':>14}")
    for name in [s[0] for s in SCENARIOS]:
        c = pure_batter_count[name]
        print(f"  {name:<22} {c['pure_batter']:>14} {c['pure_bowler']:>14} {c['all_round']:>14}")

    print()
    print("=" * 100)
    print("  TRACKED PLAYERS — FLOOR RANK across scenarios (lower = better)")
    print("=" * 100)
    header = f"  {'Player':<22}" + "".join(f"{n:>14}" for n in [s[0] for s in SCENARIOS])
    print(header)
    for p in TRACK_PLAYERS:
        cells = []
        for n in [s[0] for s in SCENARIOS]:
            v = rows_by_player[p].get(n)
            if v is None:
                cells.append("    --")
            else:
                cells.append(f"#{v['floor_rank']} ({v['floor']:+.3f})".rjust(14))
        print(f"  {p:<22}" + "".join(cells))

    print()
    print("=" * 100)
    print("  TRACKED PLAYERS — TOTAL TILT RANK across scenarios")
    print("=" * 100)
    print(header)
    for p in TRACK_PLAYERS:
        cells = []
        for n in [s[0] for s in SCENARIOS]:
            v = rows_by_player[p].get(n)
            if v is None:
                cells.append("    --")
            else:
                cells.append(f"#{v['total_rank']} ({v['total']:+.2f})".rjust(14))
        print(f"  {p:<22}" + "".join(cells))


if __name__ == "__main__":
    main()
