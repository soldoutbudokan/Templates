#!/usr/bin/env python3
"""
Cricket Test Match Win Probability Predictor
=============================================

Simple CLI to predict test match outcomes using Monte Carlo simulation.

Usage:
    python predict.py <team1> <team2> <venue> <date>

Examples:
    python predict.py England "New Zealand" "Lord's" 2026-06-04
    python predict.py India Australia MCG 2026-12-26
    python predict.py "South Africa" Pakistan Centurion 2026-01-15

Arguments:
    team1   - Name of first team (use quotes for spaces)
    team2   - Name of second team
    venue   - Name of venue (use quotes for spaces)
    date    - Match start date (YYYY-MM-DD format)

To see available teams and venues:
    python predict.py --list
"""

import sys
import argparse
from datetime import datetime

from config_loader import (
    load_match_setup,
    list_available_teams,
    list_available_venues,
    print_team_summary,
    print_venue_summary
)
from cricket_monte_carlo import MonteCarloSimulator


def parse_date(date_str: str) -> datetime:
    """Parse date string in various formats"""
    formats = [
        "%Y-%m-%d",
        "%d-%m-%Y",
        "%Y/%m/%d",
        "%d/%m/%Y",
    ]

    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue

    raise ValueError(f"Could not parse date '{date_str}'. Use YYYY-MM-DD format.")


def run_prediction(
    team1_name: str,
    team2_name: str,
    venue_name: str,
    match_date: datetime,
    n_simulations: int = 2000,
    verbose: bool = False,
    show_details: bool = True
):
    """Run the Monte Carlo prediction"""

    print("\n" + "=" * 60)
    print("CRICKET TEST MATCH WIN PROBABILITY PREDICTOR")
    print("=" * 60)

    # Load match setup from config files
    print(f"\nLoading configuration...")
    try:
        team1, team2, venue, pitch = load_match_setup(
            team1_name, team2_name, venue_name, match_date
        )
    except ValueError as e:
        print(f"\nError: {e}")
        print("\nUse 'python predict.py --list' to see available teams and venues")
        sys.exit(1)

    # Print match info
    print(f"\n{'Match Details':^60}")
    print("-" * 60)
    print(f"  {team1.name} vs {team2.name}")
    print(f"  Venue: {venue.name}, {venue.country}")
    print(f"  Date: {match_date.strftime('%B %d, %Y')}")

    if show_details:
        print(f"\n{'Conditions':^60}")
        print("-" * 60)
        print(f"  Pitch Type: {pitch.pitch_type.value}")
        print(f"  Pace Assistance: {pitch.initial_pace_assistance:.0%}")
        print(f"  Spin Assistance: {pitch.initial_spin_assistance:.0%}")
        print(f"  Rain Probability: {venue.rain_probability:.0%} per session")
        print(f"  Humidity: {venue.humidity:.0%}")

        # Key players
        def get_best_batsman(team):
            return max(team.players, key=lambda p: p.batting_stats.skill_rating)

        def get_best_bowler(team):
            bowlers = [p for p in team.players if p.bowling_stats]
            if bowlers:
                return max(bowlers, key=lambda p: p.bowling_stats.skill_rating)
            return None

        print(f"\n{'Key Players':^60}")
        print("-" * 60)
        for team in [team1, team2]:
            best_bat = get_best_batsman(team)
            best_bowl = get_best_bowler(team)
            print(f"  {team.name}:")
            print(f"    Best Batsman: {best_bat.name} (skill: {best_bat.batting_stats.skill_rating})")
            if best_bowl:
                print(f"    Best Bowler: {best_bowl.name} (skill: {best_bowl.bowling_stats.skill_rating})")

    # Run simulation
    print(f"\n{'Running Simulation':^60}")
    print("-" * 60)
    print(f"  Simulating {n_simulations} matches...")

    mc = MonteCarloSimulator(seed=42)
    results = mc.run_simulations(
        team1, team2, venue, pitch,
        n_simulations=n_simulations,
        verbose=verbose
    )

    # Print results
    probs = results['probabilities']

    print(f"\n" + "=" * 60)
    print(f"{'PREDICTION RESULTS':^60}")
    print("=" * 60)

    print(f"\n{'Outcome':<25} {'Probability':>15} {'Simulations':>15}")
    print("-" * 60)
    print(f"{team1.name + ' Win':<25} {probs['team1_win']*100:>14.1f}% {results['results_count']['team1_wins']:>15}")
    print(f"{team2.name + ' Win':<25} {probs['team2_win']*100:>14.1f}% {results['results_count']['team2_wins']:>15}")
    print(f"{'Draw':<25} {probs['draw']*100:>14.1f}% {results['results_count']['draws']:>15}")

    # Determine prediction
    if probs['team1_win'] > probs['team2_win'] and probs['team1_win'] > probs['draw']:
        prediction = team1.name
        pred_prob = probs['team1_win']
    elif probs['team2_win'] > probs['team1_win'] and probs['team2_win'] > probs['draw']:
        prediction = team2.name
        pred_prob = probs['team2_win']
    else:
        prediction = "Draw"
        pred_prob = probs['draw']

    print(f"\n{'PREDICTION':^60}")
    print("-" * 60)
    print(f"  {prediction} ({pred_prob*100:.1f}%)")

    if show_details:
        # Score statistics
        print(f"\n{'Score Statistics':^60}")
        print("-" * 60)
        for team, stats_key in [(team1, 'team1'), (team2, 'team2')]:
            stats = results['score_stats'][stats_key]
            print(f"  {team.name}:")
            print(f"    Average Match Total: {stats['mean']:.0f} runs")
            print(f"    Range: {stats['min']:.0f} - {stats['max']:.0f}")

    print("\n" + "=" * 60)

    return results


def list_options():
    """Print available teams and venues"""
    print("\n" + "=" * 60)
    print("AVAILABLE OPTIONS")
    print("=" * 60)

    print("\nTeams:")
    print("-" * 40)
    for team in sorted(list_available_teams()):
        print(f"  - {team}")

    print("\nVenues:")
    print("-" * 40)
    for venue in sorted(list_available_venues()):
        print(f"  - {venue}")

    print("\n" + "=" * 60)
    print("\nExample usage:")
    print('  python predict.py England "New Zealand" "Lord\'s" 2026-06-04')
    print('  python predict.py India Australia MCG 2026-12-26')
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Cricket Test Match Win Probability Predictor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python predict.py England "New Zealand" "Lord's" 2026-06-04
  python predict.py India Australia MCG 2026-12-26
  python predict.py --list
        """
    )

    parser.add_argument('--list', action='store_true',
                        help='List available teams and venues')
    parser.add_argument('--simulations', '-n', type=int, default=2000,
                        help='Number of simulations (default: 2000)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show simulation progress')
    parser.add_argument('--brief', '-b', action='store_true',
                        help='Show brief output (less detail)')

    parser.add_argument('team1', nargs='?', help='First team name')
    parser.add_argument('team2', nargs='?', help='Second team name')
    parser.add_argument('venue', nargs='?', help='Venue name')
    parser.add_argument('date', nargs='?', help='Match date (YYYY-MM-DD)')

    args = parser.parse_args()

    if args.list:
        list_options()
        return

    # Check required arguments
    if not all([args.team1, args.team2, args.venue, args.date]):
        parser.print_help()
        print("\nError: Missing required arguments.")
        print("Use --list to see available teams and venues.")
        sys.exit(1)

    # Parse date
    try:
        match_date = parse_date(args.date)
    except ValueError as e:
        print(f"\nError: {e}")
        sys.exit(1)

    # Run prediction
    run_prediction(
        args.team1,
        args.team2,
        args.venue,
        match_date,
        n_simulations=args.simulations,
        verbose=args.verbose,
        show_details=not args.brief
    )


if __name__ == "__main__":
    main()
