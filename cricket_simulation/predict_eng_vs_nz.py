"""
WTC Prediction: England vs New Zealand at Lord's
1st Test, June 4-8, 2026

Uses Monte Carlo simulation to predict match outcome.
"""

from cricket_monte_carlo import (
    Team, Player, BattingStats, BowlingStats, BattingStyle, BowlingStyle,
    VenueProfile, PitchConditions, PitchType,
    MonteCarloSimulator, create_batsman, create_bowler, create_all_rounder
)


def create_england_squad() -> Team:
    """
    Create England Test squad (projected for June 2026)
    Based on current/recent form and likely selections
    """
    players = [
        # Openers
        create_batsman("Zak Crawley", 68, 32.0, 62, BattingStyle.RIGHT_HAND,
                       vs_pace=0.55, vs_spin=0.45),
        create_batsman("Ben Duckett", 72, 38.0, 70, BattingStyle.LEFT_HAND,
                       vs_pace=0.5, vs_spin=0.55),

        # Middle order
        create_batsman("Ollie Pope", 75, 40.0, 58, BattingStyle.RIGHT_HAND,
                       vs_pace=0.55, vs_spin=0.5),
        create_batsman("Joe Root", 92, 51.0, 56, BattingStyle.RIGHT_HAND,
                       vs_pace=0.7, vs_spin=0.75),
        create_batsman("Harry Brook", 82, 55.0, 72, BattingStyle.RIGHT_HAND,
                       vs_pace=0.65, vs_spin=0.55),

        # Wicketkeeper
        Player(
            name="Jamie Smith",
            batting_stats=BattingStats(
                style=BattingStyle.RIGHT_HAND,
                skill_rating=70,
                average=38.0,
                strike_rate=68,
                against_pace=0.55,
                against_spin=0.5,
                defensive_ability=0.45,
                attacking_ability=0.68,
                temperament=0.5,
                pressure_handling=0.5
            ),
            is_wicketkeeper=True
        ),

        # All-rounders
        create_all_rounder("Ben Stokes", 78, 36.0, 65, BowlingStyle.FAST_MEDIUM,
                           72, 3.4, 0.018),
        create_all_rounder("Chris Woakes", 55, 22.0, 45, BowlingStyle.FAST_MEDIUM,
                           78, 2.8, 0.020),

        # Bowlers
        create_bowler("Gus Atkinson", 35, 12.0, BowlingStyle.FAST, 80, 2.9, 0.022, 90),
        create_bowler("Mark Wood", 25, 8.0, BowlingStyle.FAST, 78, 3.2, 0.021, 70),
        create_bowler("Shoaib Bashir", 30, 10.0, BowlingStyle.OFF_SPIN, 72, 3.0, 0.018, 120),
    ]

    return Team(
        name="England",
        players=players,
        captain_index=6,  # Ben Stokes
        declaration_aggression=0.7  # "Bazball" aggressive approach
    )


def create_new_zealand_squad() -> Team:
    """
    Create New Zealand Test squad (projected for June 2026)
    Based on current/recent form and likely selections
    """
    players = [
        # Openers
        create_batsman("Tom Latham", 80, 42.0, 48, BattingStyle.LEFT_HAND,
                       vs_pace=0.6, vs_spin=0.55),
        create_batsman("Devon Conway", 78, 45.0, 52, BattingStyle.LEFT_HAND,
                       vs_pace=0.6, vs_spin=0.6),

        # Middle order
        create_batsman("Kane Williamson", 90, 54.0, 50, BattingStyle.RIGHT_HAND,
                       vs_pace=0.7, vs_spin=0.7),
        create_batsman("Rachin Ravindra", 75, 40.0, 58, BattingStyle.LEFT_HAND,
                       vs_pace=0.55, vs_spin=0.5),
        create_batsman("Daryl Mitchell", 76, 42.0, 60, BattingStyle.RIGHT_HAND,
                       vs_pace=0.6, vs_spin=0.55),

        # Wicketkeeper
        Player(
            name="Tom Blundell",
            batting_stats=BattingStats(
                style=BattingStyle.RIGHT_HAND,
                skill_rating=72,
                average=40.0,
                strike_rate=55,
                against_pace=0.55,
                against_spin=0.55,
                defensive_ability=0.5,
                attacking_ability=0.55,
                temperament=0.55,
                pressure_handling=0.55
            ),
            is_wicketkeeper=True
        ),

        # All-rounders
        create_all_rounder("Glenn Phillips", 65, 32.0, 62, BowlingStyle.OFF_SPIN,
                           60, 3.5, 0.014),

        # Bowlers
        create_bowler("Tim Southee", 40, 18.0, BowlingStyle.FAST_MEDIUM, 82, 3.0, 0.022, 100),
        create_bowler("Matt Henry", 30, 12.0, BowlingStyle.FAST_MEDIUM, 80, 2.8, 0.021, 95),
        create_bowler("William O'Rourke", 25, 8.0, BowlingStyle.FAST, 76, 3.1, 0.020, 85),
        create_bowler("Mitchell Santner", 45, 20.0, BowlingStyle.LEFT_ARM_SPIN, 74, 2.6, 0.018, 130),
    ]

    return Team(
        name="New Zealand",
        players=players,
        captain_index=0,  # Tom Latham
        declaration_aggression=0.5  # More conservative approach
    )


def create_lords_venue() -> VenueProfile:
    """
    Create Lord's Cricket Ground venue profile
    'The Home of Cricket' - traditionally good for batting with
    assistance for seam bowlers, especially under overcast skies
    """
    return VenueProfile(
        name="Lord's",
        country="England",
        pace_friendly=0.55,  # Traditionally assists seamers
        spin_friendly=0.3,   # Less spin-friendly
        typical_first_innings_score=380,
        altitude=40,  # London elevation
        humidity=0.65,  # English summer humidity
        home_advantage_factor=1.1,  # Home advantage at Lord's
        rain_probability=0.15,  # June weather in England
        bad_light_probability=0.05
    )


def create_lords_pitch() -> PitchConditions:
    """
    Create pitch conditions for Lord's in early June
    Typically starts with some grass, good for batting days 2-3,
    can offer variable bounce later
    """
    return PitchConditions(
        pitch_type=PitchType.BALANCED,
        initial_pace_assistance=0.55,
        initial_spin_assistance=0.2,
        initial_bounce=0.8,
        deterioration_rate=0.1
    )


def run_prediction():
    """Run the Monte Carlo simulation for England vs New Zealand at Lord's"""

    print("=" * 60)
    print("WTC PREDICTION: ENGLAND vs NEW ZEALAND")
    print("1st Test, Lord's, June 4-8, 2026")
    print("=" * 60)

    # Create teams
    print("\nCreating squads...")
    england = create_england_squad()
    new_zealand = create_new_zealand_squad()

    # Create venue and pitch
    lords = create_lords_venue()
    pitch = create_lords_pitch()

    print(f"\nVenue: {lords.name}, {lords.country}")
    print(f"Pitch type: {pitch.pitch_type.value}")
    print(f"Conditions: Pace assistance {pitch.initial_pace_assistance:.0%}, "
          f"Spin assistance {pitch.initial_spin_assistance:.0%}")
    print(f"Rain probability: {lords.rain_probability:.0%}")

    # Print key players
    print(f"\n{england.name} Key Players:")
    print(f"  Captain: {england.captain.name}")
    print(f"  Star batsman: Joe Root (avg 51.0, skill 92)")
    print(f"  Key bowler: Gus Atkinson (skill 80)")

    print(f"\n{new_zealand.name} Key Players:")
    print(f"  Captain: {new_zealand.captain.name}")
    print(f"  Star batsman: Kane Williamson (avg 54.0, skill 90)")
    print(f"  Key bowler: Tim Southee (skill 82)")

    # Run Monte Carlo simulation
    print("\n" + "-" * 60)
    print("Running Monte Carlo simulation (2000 iterations)...")
    print("-" * 60)

    mc = MonteCarloSimulator(seed=2026)
    results = mc.run_simulations(
        england, new_zealand, lords, pitch,
        n_simulations=2000,
        verbose=True
    )

    # Print results
    probs = results['probabilities']

    print("\n" + "=" * 60)
    print("PREDICTION RESULTS")
    print("=" * 60)

    print(f"\n{'Team':<20} {'Win Probability':>20}")
    print("-" * 42)
    print(f"{'England':<20} {probs['team1_win']*100:>19.1f}%")
    print(f"{'New Zealand':<20} {probs['team2_win']*100:>19.1f}%")
    print(f"{'Draw':<20} {probs['draw']*100:>19.1f}%")

    # Determine favorite
    if probs['team1_win'] > probs['team2_win'] and probs['team1_win'] > probs['draw']:
        favorite = "England"
        fav_prob = probs['team1_win']
    elif probs['team2_win'] > probs['team1_win'] and probs['team2_win'] > probs['draw']:
        favorite = "New Zealand"
        fav_prob = probs['team2_win']
    else:
        favorite = "Draw"
        fav_prob = probs['draw']

    print(f"\n🏏 PREDICTION: {favorite} ({fav_prob*100:.1f}%)")

    # Score statistics
    print(f"\n{'Score Statistics':^42}")
    print("-" * 42)

    print(f"\nEngland:")
    eng_stats = results['score_stats']['team1']
    print(f"  Average match total: {eng_stats['mean']:.0f} runs")
    print(f"  Range: {eng_stats['min']:.0f} - {eng_stats['max']:.0f}")

    print(f"\nNew Zealand:")
    nz_stats = results['score_stats']['team2']
    print(f"  Average match total: {nz_stats['mean']:.0f} runs")
    print(f"  Range: {nz_stats['min']:.0f} - {nz_stats['max']:.0f}")

    # Result breakdown
    print(f"\n{'Result Breakdown (out of 2000 simulations)':^42}")
    print("-" * 42)
    print(f"  England wins: {results['results_count']['team1_wins']}")
    print(f"  New Zealand wins: {results['results_count']['team2_wins']}")
    print(f"  Draws: {results['results_count']['draws']}")

    # Key factors analysis
    print("\n" + "=" * 60)
    print("KEY FACTORS")
    print("=" * 60)
    print("""
• Home advantage: England benefit from familiar conditions at Lord's
• English conditions: Seam-friendly pitch suits England's pace attack
• Weather factor: 15% rain probability per session at Lord's
• Star quality: Root vs Williamson - two of the best batsmen
• Bazball factor: England's aggressive approach (0.7 aggression)
• NZ experience: Williamson's team has history of success in England
""")

    return results


if __name__ == "__main__":
    run_prediction()
