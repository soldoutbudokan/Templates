"""
Configuration Loader for Cricket Simulation
============================================

Loads team, venue, and weather configurations from YAML files.
"""

import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from cricket_monte_carlo import (
    Team, Player, BattingStats, BowlingStats,
    BattingStyle, BowlingStyle,
    VenueProfile, PitchConditions, PitchType
)


# Path to config directory
CONFIG_DIR = Path(__file__).parent / "config"


def load_yaml(filename: str) -> dict:
    """Load a YAML configuration file"""
    filepath = CONFIG_DIR / filename
    with open(filepath, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


# =============================================================================
# PLAYER/TEAM LOADING
# =============================================================================

BATTING_STYLE_MAP = {
    "right_hand": BattingStyle.RIGHT_HAND,
    "left_hand": BattingStyle.LEFT_HAND,
}

BOWLING_STYLE_MAP = {
    "fast": BowlingStyle.FAST,
    "fast_medium": BowlingStyle.FAST_MEDIUM,
    "medium": BowlingStyle.MEDIUM,
    "off_spin": BowlingStyle.OFF_SPIN,
    "leg_spin": BowlingStyle.LEG_SPIN,
    "left_arm_spin": BowlingStyle.LEFT_ARM_SPIN,
    "left_arm_fast": BowlingStyle.LEFT_ARM_FAST,
}


def load_player(player_data: dict) -> Player:
    """Create a Player object from config data"""
    bat = player_data['batting']

    batting_stats = BattingStats(
        style=BATTING_STYLE_MAP.get(bat.get('style', 'right_hand'), BattingStyle.RIGHT_HAND),
        skill_rating=bat['skill_rating'],
        average=bat['average'],
        strike_rate=bat.get('strike_rate', 50.0),
        against_pace=bat.get('vs_pace', 0.5),
        against_spin=bat.get('vs_spin', 0.5),
        defensive_ability=0.3 + (bat['skill_rating'] / 200),
        attacking_ability=bat.get('strike_rate', 50.0) / 100,
        temperament=0.5,
        pressure_handling=0.5
    )

    bowling_stats = None
    if 'bowling' in player_data:
        bowl = player_data['bowling']
        bowling_stats = BowlingStats(
            style=BOWLING_STYLE_MAP.get(bowl['style'], BowlingStyle.MEDIUM),
            skill_rating=bowl['skill_rating'],
            economy_base=bowl.get('economy', 3.0),
            wicket_probability_base=bowl.get('wicket_prob', 0.018),
            stamina=bowl.get('stamina', 100.0),
            swing_ability=0.6 if bowl['style'] in ['fast', 'fast_medium', 'left_arm_fast'] else 0.2,
            seam_ability=0.6 if bowl['style'] in ['fast', 'fast_medium', 'left_arm_fast'] else 0.2,
            spin_amount=0.7 if bowl['style'] in ['off_spin', 'leg_spin', 'left_arm_spin'] else 0.1,
            bounce_variation=0.5
        )

    is_keeper = player_data.get('role') == 'wicketkeeper'

    return Player(
        name=player_data['name'],
        batting_stats=batting_stats,
        bowling_stats=bowling_stats,
        is_wicketkeeper=is_keeper
    )


def load_team(team_name: str, config: dict = None) -> Team:
    """Load a team from configuration"""
    if config is None:
        config = load_yaml('players.yaml')

    teams = config['teams']

    if team_name not in teams:
        available = list(teams.keys())
        raise ValueError(f"Team '{team_name}' not found. Available teams: {available}")

    team_data = teams[team_name]
    players = [load_player(p) for p in team_data['players']]

    return Team(
        name=team_name,
        players=players,
        captain_index=team_data.get('captain_index', 0),
        declaration_aggression=team_data.get('declaration_aggression', 0.5)
    )


def list_available_teams() -> List[str]:
    """List all available teams"""
    config = load_yaml('players.yaml')
    return list(config['teams'].keys())


# =============================================================================
# VENUE LOADING
# =============================================================================

def load_venue(venue_name: str, config: dict = None) -> Tuple[VenueProfile, str]:
    """
    Load a venue from configuration.

    Returns:
        Tuple of (VenueProfile, city_name) for weather lookup
    """
    if config is None:
        config = load_yaml('venues.yaml')

    venues = config['venues']

    if venue_name not in venues:
        available = list(venues.keys())
        raise ValueError(f"Venue '{venue_name}' not found. Available venues: {available}")

    v = venues[venue_name]

    profile = VenueProfile(
        name=venue_name,
        country=v['country'],
        pace_friendly=v.get('pace_friendly', 0.5),
        spin_friendly=v.get('spin_friendly', 0.4),
        typical_first_innings_score=v.get('typical_first_innings', 350),
        altitude=v.get('altitude', 0),
        humidity=0.5,  # Will be set from weather
        home_advantage_factor=v.get('home_advantage', 1.1),
        rain_probability=0.1,  # Will be set from weather
        bad_light_probability=0.05  # Will be set from weather
    )

    return profile, v.get('city', venue_name)


def create_pitch_from_venue(venue: VenueProfile) -> PitchConditions:
    """Create pitch conditions based on venue characteristics"""

    # Determine pitch type from venue characteristics
    if venue.pace_friendly >= 0.65:
        pitch_type = PitchType.PACE_FRIENDLY
    elif venue.spin_friendly >= 0.65:
        pitch_type = PitchType.SPIN_FRIENDLY
    elif venue.pace_friendly <= 0.35 and venue.spin_friendly <= 0.35:
        pitch_type = PitchType.FLAT
    else:
        pitch_type = PitchType.BALANCED

    return PitchConditions(
        pitch_type=pitch_type,
        initial_pace_assistance=venue.pace_friendly,
        initial_spin_assistance=venue.spin_friendly,
        initial_bounce=0.75,
        deterioration_rate=0.09  # Default, could be venue-specific
    )


def list_available_venues() -> List[str]:
    """List all available venues"""
    config = load_yaml('venues.yaml')
    return list(config['venues'].keys())


# =============================================================================
# WEATHER LOADING
# =============================================================================

def load_weather_for_date(city: str, date: datetime, config: dict = None) -> dict:
    """
    Load historical weather averages for a city and month.

    Args:
        city: City name
        date: Date of the match
        config: Optional pre-loaded config

    Returns:
        Dict with rain_prob, bad_light_prob, overcast_prob, humidity
    """
    if config is None:
        config = load_yaml('weather.yaml')

    weather_data = config['weather_by_city']
    month = date.month

    # Try exact city match
    if city in weather_data:
        city_weather = weather_data[city]
    else:
        # Try partial match
        matching_cities = [c for c in weather_data.keys() if city.lower() in c.lower()]
        if matching_cities:
            city_weather = weather_data[matching_cities[0]]
            print(f"Note: Using weather data for '{matching_cities[0]}' (matched '{city}')")
        else:
            print(f"Warning: No weather data for '{city}', using defaults")
            city_weather = weather_data.get('_default', {})

    month_weather = city_weather.get(month, weather_data.get('_default', {}).get(month, {}))

    return {
        'rain_probability': month_weather.get('rain_prob', 0.10),
        'bad_light_probability': month_weather.get('bad_light_prob', 0.05),
        'overcast_probability': month_weather.get('overcast_prob', 0.30),
        'humidity': month_weather.get('humidity', 0.55)
    }


def apply_weather_to_venue(venue: VenueProfile, weather: dict) -> VenueProfile:
    """Apply weather data to a venue profile"""
    venue.rain_probability = weather['rain_probability']
    venue.bad_light_probability = weather['bad_light_probability']
    venue.humidity = weather['humidity']
    return venue


# =============================================================================
# HIGH-LEVEL LOADING
# =============================================================================

def load_match_setup(
    team1_name: str,
    team2_name: str,
    venue_name: str,
    match_date: datetime
) -> Tuple[Team, Team, VenueProfile, PitchConditions]:
    """
    Load complete match setup from configuration files.

    Args:
        team1_name: Name of first team
        team2_name: Name of second team
        venue_name: Name of venue
        match_date: Date of match (for weather)

    Returns:
        Tuple of (team1, team2, venue, pitch)
    """
    # Load teams
    team1 = load_team(team1_name)
    team2 = load_team(team2_name)

    # Load venue
    venue, city = load_venue(venue_name)

    # Load weather for the month
    weather = load_weather_for_date(city, match_date)

    # Apply weather to venue
    venue = apply_weather_to_venue(venue, weather)

    # Create pitch from venue
    pitch = create_pitch_from_venue(venue)

    return team1, team2, venue, pitch


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def print_team_summary(team: Team):
    """Print a summary of a team's players"""
    print(f"\n{team.name}")
    print("=" * 50)
    print(f"Captain: {team.captain.name}")
    print(f"Declaration Aggression: {team.declaration_aggression}")
    print("\nPlayers:")
    for i, player in enumerate(team.players):
        bat = player.batting_stats
        role = "WK" if player.is_wicketkeeper else ("AR" if player.bowling_stats else "BAT")
        if player.bowling_stats and not player.is_wicketkeeper:
            if player.bowling_stats.skill_rating > player.batting_stats.skill_rating:
                role = "BOWL"

        bowl_str = ""
        if player.bowling_stats:
            bowl_str = f" | Bowl: {player.bowling_stats.skill_rating}"

        print(f"  {i+1}. {player.name:<20} ({role:4}) Bat: {bat.skill_rating:2}{bowl_str}")


def print_venue_summary(venue: VenueProfile, pitch: PitchConditions):
    """Print a summary of venue and conditions"""
    print(f"\nVenue: {venue.name}, {venue.country}")
    print("=" * 50)
    print(f"Pitch Type: {pitch.pitch_type.value}")
    print(f"Pace Friendly: {venue.pace_friendly:.0%}")
    print(f"Spin Friendly: {venue.spin_friendly:.0%}")
    print(f"Rain Probability: {venue.rain_probability:.0%} per session")
    print(f"Humidity: {venue.humidity:.0%}")
    print(f"Home Advantage: {venue.home_advantage_factor:.0%}")


if __name__ == "__main__":
    # Test loading
    print("Available Teams:", list_available_teams())
    print("\nAvailable Venues:", list_available_venues())

    # Load a sample match
    team1, team2, venue, pitch = load_match_setup(
        "England", "New Zealand", "Lord's",
        datetime(2026, 6, 4)
    )

    print_team_summary(team1)
    print_team_summary(team2)
    print_venue_summary(venue, pitch)
