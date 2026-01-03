"""
Cricket Test Match Monte Carlo Simulation
==========================================

A comprehensive Monte Carlo simulation for cricket test matches that models:
- Ball-by-ball simulation
- Individual player batting and bowling skills
- Batsman settling-in mechanics
- Player fatigue
- Pitch deterioration over 5 days
- Venue-specific adjustments
- Weather effects (rain, bad light)
- Captain declaration decisions

Author: Cricket Analytics
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
from abc import ABC, abstractmethod
import random
from collections import defaultdict


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================

class BowlingStyle(Enum):
    FAST = "fast"
    FAST_MEDIUM = "fast_medium"
    MEDIUM = "medium"
    OFF_SPIN = "off_spin"
    LEG_SPIN = "leg_spin"
    LEFT_ARM_SPIN = "left_arm_spin"
    LEFT_ARM_FAST = "left_arm_fast"


class BattingStyle(Enum):
    RIGHT_HAND = "right_hand"
    LEFT_HAND = "left_hand"


class DismissalType(Enum):
    BOWLED = "bowled"
    CAUGHT = "caught"
    LBW = "lbw"
    RUN_OUT = "run_out"
    STUMPED = "stumped"
    HIT_WICKET = "hit_wicket"
    NOT_OUT = "not_out"


class PitchType(Enum):
    PACE_FRIENDLY = "pace_friendly"
    BALANCED = "balanced"
    SPIN_FRIENDLY = "spin_friendly"
    FLAT = "flat"
    GREEN_TOP = "green_top"


class WeatherCondition(Enum):
    SUNNY = "sunny"
    OVERCAST = "overcast"
    LIGHT_RAIN = "light_rain"
    HEAVY_RAIN = "heavy_rain"
    BAD_LIGHT = "bad_light"


class MatchResult(Enum):
    TEAM1_WIN = "team1_win"
    TEAM2_WIN = "team2_win"
    DRAW = "draw"


# Test match constants
OVERS_PER_DAY = 90
BALLS_PER_OVER = 6
MAX_DAYS = 5
INNINGS_PER_TEAM = 2


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class BowlingStats:
    """Bowling statistics and skills for a player"""
    style: BowlingStyle
    skill_rating: float  # 0-100 scale
    economy_base: float  # Expected runs per over
    wicket_probability_base: float  # Base probability of taking wicket per ball
    stamina: float = 100.0  # Maximum overs before significant fatigue
    current_fatigue: float = 0.0  # Current fatigue level (0-100)

    # Style-specific modifiers
    swing_ability: float = 0.5  # 0-1, how much they can swing the ball
    seam_ability: float = 0.5  # 0-1, seam movement
    spin_amount: float = 0.5  # 0-1, spin turn
    bounce_variation: float = 0.5  # 0-1, variation in bounce

    def reset_fatigue(self):
        """Reset fatigue for new innings/match"""
        self.current_fatigue = 0.0


@dataclass
class BattingStats:
    """Batting statistics and skills for a player"""
    style: BattingStyle
    skill_rating: float  # 0-100 scale
    average: float  # Career/recent batting average
    strike_rate: float  # Career/recent strike rate

    # Skill breakdowns
    against_pace: float = 0.5  # 0-1 modifier vs pace
    against_spin: float = 0.5  # 0-1 modifier vs spin
    defensive_ability: float = 0.5  # 0-1, ability to survive
    attacking_ability: float = 0.5  # 0-1, ability to score quickly

    # Mental attributes
    temperament: float = 0.5  # 0-1, ability to concentrate for long periods
    pressure_handling: float = 0.5  # 0-1, performance under pressure


@dataclass
class Player:
    """Represents a cricket player"""
    name: str
    batting_stats: BattingStats
    bowling_stats: Optional[BowlingStats] = None
    is_wicketkeeper: bool = False

    # In-match state
    balls_faced: int = 0
    runs_scored: int = 0
    is_out: bool = False
    dismissal_type: DismissalType = DismissalType.NOT_OUT
    overs_bowled: float = 0.0
    wickets_taken: int = 0
    runs_conceded: int = 0

    def reset_batting_innings(self):
        """Reset batting stats for new innings"""
        self.balls_faced = 0
        self.runs_scored = 0
        self.is_out = False
        self.dismissal_type = DismissalType.NOT_OUT

    def reset_match(self):
        """Reset all match stats"""
        self.reset_batting_innings()
        self.overs_bowled = 0.0
        self.wickets_taken = 0
        self.runs_conceded = 0
        if self.bowling_stats:
            self.bowling_stats.reset_fatigue()

    @property
    def is_bowler(self) -> bool:
        return self.bowling_stats is not None

    def get_settling_factor(self) -> float:
        """
        Returns a factor (0-1) representing how settled the batsman is.
        Batsmen are most vulnerable early in their innings.
        """
        if self.balls_faced <= 10:
            # Very vulnerable first 10 balls
            return 0.6 + (self.balls_faced * 0.02)
        elif self.balls_faced <= 30:
            # Gradually settling in
            return 0.8 + ((self.balls_faced - 10) * 0.01)
        elif self.balls_faced <= 100:
            # Well set
            return 1.0 + ((self.balls_faced - 30) * 0.002)
        else:
            # Very well set but fatigue starts
            base = 1.14
            fatigue = min(0.15, (self.balls_faced - 100) * 0.0005)
            return base - fatigue


@dataclass
class Team:
    """Represents a cricket team"""
    name: str
    players: List[Player]
    batting_order: List[int] = field(default_factory=list)  # Indices into players
    captain_index: int = 0  # Index of captain in players list

    # Declaration strategy parameters
    declaration_aggression: float = 0.5  # 0-1, how aggressive in declarations

    def __post_init__(self):
        if not self.batting_order:
            self.batting_order = list(range(len(self.players)))

    @property
    def captain(self) -> Player:
        return self.players[self.captain_index]

    def get_bowlers(self) -> List[Player]:
        """Get all players who can bowl"""
        return [p for p in self.players if p.is_bowler]

    def reset_for_innings(self):
        """Reset all players for new innings"""
        for player in self.players:
            player.reset_batting_innings()

    def reset_for_match(self):
        """Reset all players for new match"""
        for player in self.players:
            player.reset_match()


@dataclass
class BallOutcome:
    """Result of a single ball"""
    runs_scored: int = 0
    is_wicket: bool = False
    dismissal_type: Optional[DismissalType] = None
    is_wide: bool = False
    is_no_ball: bool = False
    is_bye: bool = False
    is_leg_bye: bool = False
    extras: int = 0

    @property
    def total_runs(self) -> int:
        return self.runs_scored + self.extras

    @property
    def is_legal_delivery(self) -> bool:
        return not self.is_wide and not self.is_no_ball


@dataclass
class OverState:
    """State of a single over"""
    over_number: int
    bowler: Player
    balls: List[BallOutcome] = field(default_factory=list)

    @property
    def legal_deliveries(self) -> int:
        return sum(1 for b in self.balls if b.is_legal_delivery)

    @property
    def runs_conceded(self) -> int:
        return sum(b.total_runs for b in self.balls)

    @property
    def wickets_taken(self) -> int:
        return sum(1 for b in self.balls if b.is_wicket)

    @property
    def is_complete(self) -> bool:
        return self.legal_deliveries >= BALLS_PER_OVER


@dataclass
class InningsState:
    """State of an innings"""
    batting_team: Team
    bowling_team: Team
    innings_number: int  # 1-4

    runs: int = 0
    wickets: int = 0
    overs_completed: float = 0.0
    balls_in_current_over: int = 0

    current_striker_idx: int = 0  # Index in batting order
    current_non_striker_idx: int = 1

    overs: List[OverState] = field(default_factory=list)
    is_declared: bool = False
    is_follow_on: bool = False

    @property
    def is_all_out(self) -> bool:
        return self.wickets >= 10

    @property
    def is_complete(self) -> bool:
        return self.is_all_out or self.is_declared

    @property
    def current_striker(self) -> Player:
        return self.batting_team.players[self.batting_team.batting_order[self.current_striker_idx]]

    @property
    def current_non_striker(self) -> Player:
        return self.batting_team.players[self.batting_team.batting_order[self.current_non_striker_idx]]

    def swap_strike(self):
        """Swap striker and non-striker"""
        self.current_striker_idx, self.current_non_striker_idx = (
            self.current_non_striker_idx, self.current_striker_idx
        )

    def wicket_fallen(self):
        """Handle wicket falling"""
        self.wickets += 1
        if self.wickets < 10:
            # Next batsman comes in
            self.current_striker_idx = self.wickets + 1

    def add_runs(self, runs: int):
        """Add runs to innings total"""
        self.runs += runs


@dataclass
class PitchConditions:
    """Models the pitch conditions throughout the match"""
    pitch_type: PitchType
    initial_pace_assistance: float  # 0-1
    initial_spin_assistance: float  # 0-1
    initial_bounce: float  # 0-1, consistency of bounce
    deterioration_rate: float = 0.1  # How quickly pitch deteriorates

    # Current state
    current_day: int = 1
    total_overs_bowled: int = 0

    def get_pace_assistance(self) -> float:
        """Get current pace assistance (decreases over time)"""
        decay = self.deterioration_rate * (self.current_day - 1)
        return max(0.1, self.initial_pace_assistance - decay)

    def get_spin_assistance(self) -> float:
        """Get current spin assistance (increases over time)"""
        increase = self.deterioration_rate * (self.current_day - 1) * 1.5
        return min(1.0, self.initial_spin_assistance + increase)

    def get_bounce_consistency(self) -> float:
        """Get bounce consistency (decreases over time, variable bounce)"""
        decay = self.deterioration_rate * 0.5 * (self.current_day - 1)
        return max(0.3, self.initial_bounce - decay)

    def get_batting_difficulty(self) -> float:
        """
        Overall batting difficulty factor.
        Higher = more difficult to bat.
        """
        # Day 1: Pace helps bowlers
        # Day 2-3: Best for batting
        # Day 4-5: Deterioration helps spinners
        if self.current_day <= 1:
            base = 0.55
        elif self.current_day <= 3:
            base = 0.45
        else:
            base = 0.5 + (self.current_day - 3) * 0.08

        # Add effect of variable bounce
        bounce_var = 1 - self.get_bounce_consistency()
        return base + bounce_var * 0.15

    def advance_overs(self, overs: int = 1):
        """Advance the pitch state by number of overs"""
        self.total_overs_bowled += overs

    def advance_day(self):
        """Move to next day"""
        self.current_day = min(MAX_DAYS, self.current_day + 1)


@dataclass
class VenueProfile:
    """Venue-specific characteristics"""
    name: str
    country: str

    # Pitch tendencies
    pace_friendly: float = 0.5  # 0-1
    spin_friendly: float = 0.5  # 0-1
    typical_first_innings_score: float = 350.0

    # Environmental factors
    altitude: float = 0.0  # meters above sea level
    humidity: float = 0.5  # average humidity 0-1

    # Home advantage
    home_advantage_factor: float = 1.1  # Multiplier for home team

    # Weather patterns
    rain_probability: float = 0.1  # Base probability of rain per session
    bad_light_probability: float = 0.05

    def get_pace_modifier(self) -> float:
        """Get pace bowling modifier for this venue"""
        # Higher altitude = more carry
        altitude_bonus = min(0.1, self.altitude / 15000)
        return self.pace_friendly + altitude_bonus

    def get_spin_modifier(self) -> float:
        """Get spin bowling modifier for this venue"""
        return self.spin_friendly

    def get_swing_modifier(self, weather: WeatherCondition) -> float:
        """Get swing modifier based on conditions"""
        base = self.humidity * 0.5
        if weather == WeatherCondition.OVERCAST:
            base += 0.3
        return min(1.0, base)


@dataclass
class WeatherState:
    """Current weather conditions"""
    condition: WeatherCondition = WeatherCondition.SUNNY
    overs_lost_today: int = 0

    def is_playable(self) -> bool:
        """Check if play is possible"""
        return self.condition not in [
            WeatherCondition.HEAVY_RAIN,
            WeatherCondition.BAD_LIGHT
        ]

    def get_swing_bonus(self) -> float:
        """Get swing bonus from weather"""
        if self.condition == WeatherCondition.OVERCAST:
            return 0.2
        return 0.0


@dataclass
class MatchState:
    """Complete state of a test match"""
    team1: Team
    team2: Team
    venue: VenueProfile
    pitch: PitchConditions
    weather: WeatherState

    current_innings: int = 1  # 1-4
    innings_states: List[InningsState] = field(default_factory=list)

    current_day: int = 1
    overs_remaining_today: int = OVERS_PER_DAY

    is_complete: bool = False
    result: Optional[MatchResult] = None
    winning_margin: Optional[str] = None

    toss_winner: Optional[Team] = None
    toss_decision: Optional[str] = None  # "bat" or "bowl"

    def get_batting_team(self) -> Team:
        """Get currently batting team"""
        if self.current_innings in [1, 4]:
            return self.team1 if (self.current_innings == 1) == (self.toss_decision == "bat" and self.toss_winner == self.team1 or self.toss_decision == "bowl" and self.toss_winner == self.team2) else self.team2
        else:
            return self.team2 if self.innings_states[0].batting_team == self.team1 else self.team1

    def get_bowling_team(self) -> Team:
        """Get currently bowling team"""
        batting = self.get_batting_team()
        return self.team2 if batting == self.team1 else self.team1

    def get_target(self) -> Optional[int]:
        """Get target for 4th innings, if applicable"""
        if self.current_innings == 4 and len(self.innings_states) >= 3:
            team1_total = sum(
                inn.runs for inn in self.innings_states
                if inn.batting_team == self.team1
            )
            team2_total = sum(
                inn.runs for inn in self.innings_states
                if inn.batting_team == self.team2
            )
            if self.innings_states[-1].batting_team == self.team1:
                return team2_total - team1_total + 1
            else:
                return team1_total - team2_total + 1
        return None

    def get_lead(self) -> int:
        """Get current lead/deficit for batting team"""
        if not self.innings_states:
            return 0

        batting_team = self.innings_states[-1].batting_team
        batting_total = sum(
            inn.runs for inn in self.innings_states
            if inn.batting_team == batting_team
        )
        bowling_total = sum(
            inn.runs for inn in self.innings_states
            if inn.batting_team != batting_team
        )
        return batting_total - bowling_total


# =============================================================================
# SIMULATION ENGINE
# =============================================================================

class BallSimulator:
    """Simulates individual ball outcomes"""

    def __init__(self, rng: np.random.Generator = None):
        self.rng = rng or np.random.default_rng()

    def simulate_ball(
        self,
        batsman: Player,
        bowler: Player,
        pitch: PitchConditions,
        venue: VenueProfile,
        weather: WeatherState,
        match_situation: Dict
    ) -> BallOutcome:
        """
        Simulate a single ball delivery.

        Args:
            batsman: Current striker
            bowler: Current bowler
            pitch: Current pitch conditions
            venue: Venue characteristics
            weather: Current weather
            match_situation: Dict with keys like 'innings', 'runs_required', 'overs_remaining'

        Returns:
            BallOutcome with the result of the ball
        """
        outcome = BallOutcome()

        # Check for extras first
        if self._check_wide(bowler, pitch):
            outcome.is_wide = True
            outcome.extras = 1
            return outcome

        if self._check_no_ball(bowler):
            outcome.is_no_ball = True
            outcome.extras = 1
            # Can still be hit or take wicket (except bowled/lbw)

        # Calculate wicket probability
        wicket_prob = self._calculate_wicket_probability(
            batsman, bowler, pitch, venue, weather, match_situation
        )

        # Check for wicket
        if self.rng.random() < wicket_prob and not outcome.is_no_ball:
            outcome.is_wicket = True
            outcome.dismissal_type = self._determine_dismissal_type(bowler)
            return outcome

        # Calculate runs scored
        outcome.runs_scored = self._calculate_runs(
            batsman, bowler, pitch, venue, match_situation
        )

        # Check for byes/leg byes (when no runs scored off bat)
        if outcome.runs_scored == 0 and self.rng.random() < 0.03:
            if self.rng.random() < 0.5:
                outcome.is_bye = True
            else:
                outcome.is_leg_bye = True
            outcome.extras = self.rng.choice([1, 2], p=[0.8, 0.2])

        return outcome

    def _check_wide(self, bowler: Player, pitch: PitchConditions) -> bool:
        """Check if delivery is a wide"""
        base_prob = 0.015  # ~1.5% of deliveries

        # Spinners have slightly more wides
        if bowler.bowling_stats.style in [
            BowlingStyle.OFF_SPIN, BowlingStyle.LEG_SPIN, BowlingStyle.LEFT_ARM_SPIN
        ]:
            base_prob *= 1.2

        # Fatigue increases wides
        fatigue_factor = 1 + (bowler.bowling_stats.current_fatigue / 200)

        return self.rng.random() < base_prob * fatigue_factor

    def _check_no_ball(self, bowler: Player) -> bool:
        """Check if delivery is a no ball"""
        base_prob = 0.008  # ~0.8% of deliveries

        # Faster bowlers have slightly more no balls
        if bowler.bowling_stats.style in [BowlingStyle.FAST, BowlingStyle.LEFT_ARM_FAST]:
            base_prob *= 1.3

        # Fatigue increases no balls
        fatigue_factor = 1 + (bowler.bowling_stats.current_fatigue / 150)

        return self.rng.random() < base_prob * fatigue_factor

    def _calculate_wicket_probability(
        self,
        batsman: Player,
        bowler: Player,
        pitch: PitchConditions,
        venue: VenueProfile,
        weather: WeatherState,
        match_situation: Dict
    ) -> float:
        """Calculate probability of wicket on this ball"""

        # Base probability from bowler skill (reduced for more realistic test match scores)
        # In real tests, ~1 wicket per 55-60 balls on average (wicket every ~10 overs)
        base_prob = bowler.bowling_stats.wicket_probability_base * 0.45

        # Bowler skill modifier (0.7 to 1.3 - narrower range)
        skill_mod = 0.7 + (bowler.bowling_stats.skill_rating / 166)

        # Batsman defense modifier (0.7 to 1.3, inverted - better defense = lower prob)
        bat_skill = batsman.batting_stats.skill_rating
        defense_mod = 1.3 - (bat_skill / 166)

        # Settling factor (batsmen more vulnerable early)
        settling = batsman.get_settling_factor()
        settling_mod = 1 / settling  # Inverted so lower settling = higher wicket prob

        # Pitch conditions
        pitch_mod = 1.0
        is_spinner = bowler.bowling_stats.style in [
            BowlingStyle.OFF_SPIN, BowlingStyle.LEG_SPIN, BowlingStyle.LEFT_ARM_SPIN
        ]
        is_pacer = bowler.bowling_stats.style in [
            BowlingStyle.FAST, BowlingStyle.FAST_MEDIUM, BowlingStyle.LEFT_ARM_FAST
        ]

        if is_pacer:
            pitch_mod *= (1 + pitch.get_pace_assistance() * 0.2)
            pitch_mod *= (1 + venue.get_pace_modifier() * 0.15)
            # Swing in overcast conditions
            pitch_mod *= (1 + weather.get_swing_bonus() * bowler.bowling_stats.swing_ability * 0.2)
        elif is_spinner:
            pitch_mod *= (1 + pitch.get_spin_assistance() * 0.3)
            pitch_mod *= (1 + venue.get_spin_modifier() * 0.2)

        # Variable bounce increases wicket chances
        bounce_var = 1 - pitch.get_bounce_consistency()
        pitch_mod *= (1 + bounce_var * 0.15)

        # Bowler fatigue (reduces effectiveness)
        fatigue_mod = 1 - (bowler.bowling_stats.current_fatigue / 250)

        # Match situation modifiers
        situation_mod = 1.0
        if match_situation.get('innings', 1) >= 3:
            # Late innings pitch deterioration effect
            situation_mod *= 1.03

        # Batsman vs bowling type matchup
        matchup_mod = 1.0
        if is_pacer:
            matchup_mod = 1.3 - batsman.batting_stats.against_pace * 0.6
        elif is_spinner:
            matchup_mod = 1.3 - batsman.batting_stats.against_spin * 0.6

        # Calculate final probability
        prob = (
            base_prob
            * skill_mod
            * defense_mod
            * settling_mod
            * pitch_mod
            * fatigue_mod
            * situation_mod
            * matchup_mod
        )

        # Clamp to reasonable range (lower ceiling for more realistic test batting)
        return np.clip(prob, 0.001, 0.08)

    def _determine_dismissal_type(self, bowler: Player) -> DismissalType:
        """Determine how the batsman was dismissed"""
        is_spinner = bowler.bowling_stats.style in [
            BowlingStyle.OFF_SPIN, BowlingStyle.LEG_SPIN, BowlingStyle.LEFT_ARM_SPIN
        ]

        if is_spinner:
            # Spinners: more caught, stumped, lbw
            probs = [0.1, 0.45, 0.3, 0.05, 0.08, 0.02]
        else:
            # Pacers: more caught, bowled, lbw
            probs = [0.2, 0.5, 0.25, 0.03, 0.01, 0.01]

        types = [
            DismissalType.BOWLED,
            DismissalType.CAUGHT,
            DismissalType.LBW,
            DismissalType.RUN_OUT,
            DismissalType.STUMPED,
            DismissalType.HIT_WICKET
        ]

        return self.rng.choice(types, p=probs)

    def _calculate_runs(
        self,
        batsman: Player,
        bowler: Player,
        pitch: PitchConditions,
        venue: VenueProfile,
        match_situation: Dict
    ) -> int:
        """Calculate runs scored off the bat"""

        # Base run distribution for test cricket
        # Realistic distribution: ~50% dots, ~28% singles, boundaries ~12%
        base_probs = np.array([0.50, 0.28, 0.07, 0.02, 0.11, 0.005, 0.015])
        # Corresponds to: 0, 1, 2, 3, 4, 5, 6

        # Modify based on batsman skill and strike rate
        sr_factor = batsman.batting_stats.strike_rate / 50  # Normalize around 50 SR for tests

        # Shift probability towards runs
        if sr_factor > 1:
            # Higher strike rate = more boundaries
            base_probs[0] *= (1 / (sr_factor ** 0.5))
            base_probs[4] *= (sr_factor ** 0.6)
            base_probs[6] *= (sr_factor ** 0.6)
        else:
            # Lower strike rate = more dots
            base_probs[0] *= (1 / (sr_factor ** 0.5))

        # Settling affects run scoring
        settling = batsman.get_settling_factor()
        base_probs[1] *= (0.9 + settling * 0.15)  # More singles when settled
        base_probs[4] *= (0.85 + settling * 0.2)  # More boundaries when settled

        # Attacking ability
        attack = batsman.batting_stats.attacking_ability
        base_probs[4] *= (0.8 + attack * 0.4)
        base_probs[6] *= (0.7 + attack * 0.6)

        # Bowler quality reduces scoring
        bowler_skill = bowler.bowling_stats.skill_rating
        bowler_mod = (120 - bowler_skill) / 100
        base_probs[1:] *= (0.7 + bowler_mod * 0.35)

        # Pitch conditions affect scoring
        difficulty = pitch.get_batting_difficulty()
        base_probs[0] *= (0.9 + difficulty * 0.2)
        base_probs[4:] *= (1.1 - difficulty * 0.2)

        # Normalize
        base_probs = np.clip(base_probs, 0.001, None)
        base_probs /= base_probs.sum()

        return self.rng.choice([0, 1, 2, 3, 4, 5, 6], p=base_probs)


class OverSimulator:
    """Simulates complete overs"""

    def __init__(self, ball_simulator: BallSimulator):
        self.ball_sim = ball_simulator

    def simulate_over(
        self,
        innings: InningsState,
        bowler: Player,
        pitch: PitchConditions,
        venue: VenueProfile,
        weather: WeatherState,
        match_situation: Dict
    ) -> OverState:
        """Simulate a complete over"""
        over = OverState(
            over_number=int(innings.overs_completed),
            bowler=bowler
        )

        legal_deliveries = 0

        while legal_deliveries < BALLS_PER_OVER and not innings.is_all_out:
            batsman = innings.current_striker

            outcome = self.ball_sim.simulate_ball(
                batsman, bowler, pitch, venue, weather, match_situation
            )
            over.balls.append(outcome)

            # Update innings state
            innings.add_runs(outcome.total_runs)
            batsman.runs_scored += outcome.runs_scored
            bowler.runs_conceded += outcome.total_runs

            if outcome.is_legal_delivery:
                legal_deliveries += 1
                batsman.balls_faced += 1

            if outcome.is_wicket:
                batsman.is_out = True
                batsman.dismissal_type = outcome.dismissal_type
                bowler.wickets_taken += 1
                innings.wicket_fallen()
            elif outcome.runs_scored % 2 == 1:
                # Odd runs = swap strike
                innings.swap_strike()

        # End of over: swap strike
        if not innings.is_all_out:
            innings.swap_strike()

        # Update bowler fatigue
        bowler.overs_bowled += 1
        self._update_bowler_fatigue(bowler)

        # Update innings overs
        innings.overs_completed += 1

        return over

    def _update_bowler_fatigue(self, bowler: Player):
        """Update bowler fatigue after bowling an over"""
        if not bowler.bowling_stats:
            return

        # Fatigue accumulation based on bowling style
        base_fatigue = 1.0
        if bowler.bowling_stats.style in [BowlingStyle.FAST, BowlingStyle.LEFT_ARM_FAST]:
            base_fatigue = 2.0  # Fast bowlers tire quicker
        elif bowler.bowling_stats.style == BowlingStyle.FAST_MEDIUM:
            base_fatigue = 1.5

        # Stamina affects fatigue rate
        stamina_mod = 100 / bowler.bowling_stats.stamina

        bowler.bowling_stats.current_fatigue += base_fatigue * stamina_mod

        # Cap fatigue
        bowler.bowling_stats.current_fatigue = min(
            100, bowler.bowling_stats.current_fatigue
        )


class DeclarationEngine:
    """Handles captain declaration decisions"""

    def __init__(self, rng: np.random.Generator = None):
        self.rng = rng or np.random.default_rng()

    def should_declare(
        self,
        match: MatchState,
        innings: InningsState
    ) -> bool:
        """
        Determine if the batting team should declare.

        Only applicable for 1st, 2nd, or 3rd innings.
        """
        if match.current_innings == 4:
            # Never declare in 4th innings when chasing
            return False

        lead = match.get_lead()
        overs_remaining = self._estimate_overs_remaining(match)
        wickets_in_hand = 10 - innings.wickets

        captain_aggression = innings.batting_team.declaration_aggression

        # First innings: rarely declare unless huge score
        if match.current_innings == 1:
            if innings.runs >= 600 and wickets_in_hand <= 4:
                return True
            if innings.runs >= 500 and match.overs_remaining_today <= 20:
                return self.rng.random() < captain_aggression
            return False

        # Second innings: declare to set target or avoid follow-on
        if match.current_innings == 2:
            if lead > 0:
                # Already ahead, consider declaring to bowl
                if lead >= 150 and overs_remaining >= 120:
                    return self.rng.random() < captain_aggression
            return False

        # Third innings: set target for opponents
        if match.current_innings == 3:
            target = lead  # Opponent needs to chase this + 1

            # Consider: overs remaining, lead, and risk
            if overs_remaining <= 30:
                # Not much time left, may draw anyway
                if target >= 100:
                    return True
            elif overs_remaining <= 60:
                if target >= 200:
                    return True
                elif target >= 150:
                    return self.rng.random() < captain_aggression
            elif overs_remaining <= 90:
                if target >= 300:
                    return True
                elif target >= 250:
                    return self.rng.random() < captain_aggression
            else:
                if target >= 400:
                    return True
                elif target >= 350:
                    return self.rng.random() < captain_aggression

        return False

    def _estimate_overs_remaining(self, match: MatchState) -> int:
        """Estimate total overs remaining in match"""
        days_left = MAX_DAYS - match.current_day + 1
        return match.overs_remaining_today + (days_left - 1) * OVERS_PER_DAY


class BowlerSelector:
    """Selects which bowler should bowl the next over"""

    def __init__(self, rng: np.random.Generator = None):
        self.rng = rng or np.random.default_rng()

    def select_bowler(
        self,
        bowling_team: Team,
        last_bowler: Optional[Player],
        pitch: PitchConditions,
        match_situation: Dict
    ) -> Player:
        """
        Select the best bowler for the next over.

        Cannot bowl same bowler as last over.
        """
        bowlers = bowling_team.get_bowlers()
        if not bowlers:
            raise ValueError("No bowlers in team!")

        # Filter out last bowler
        available = [b for b in bowlers if b != last_bowler]
        if not available:
            available = bowlers  # Fallback

        # Score each bowler
        scores = []
        for bowler in available:
            score = self._score_bowler(bowler, pitch, match_situation)
            scores.append(score)

        # Weight selection by score
        scores = np.array(scores)
        probs = scores / scores.sum()

        return self.rng.choice(available, p=probs)

    def _score_bowler(
        self,
        bowler: Player,
        pitch: PitchConditions,
        match_situation: Dict
    ) -> float:
        """Score a bowler's suitability for current situation"""
        if not bowler.bowling_stats:
            return 0.1

        stats = bowler.bowling_stats

        # Base score from skill
        score = stats.skill_rating

        # Pitch suitability
        is_spinner = stats.style in [
            BowlingStyle.OFF_SPIN, BowlingStyle.LEG_SPIN, BowlingStyle.LEFT_ARM_SPIN
        ]
        if is_spinner:
            score *= (1 + pitch.get_spin_assistance())
        else:
            score *= (1 + pitch.get_pace_assistance())

        # Fatigue penalty
        score *= (1 - stats.current_fatigue / 150)

        # Recent overs bowled
        if bowler.overs_bowled > 15:
            score *= 0.8
        if bowler.overs_bowled > 25:
            score *= 0.7

        return max(0.1, score)


class WeatherSimulator:
    """Simulates weather changes during the match"""

    def __init__(self, venue: VenueProfile, rng: np.random.Generator = None):
        self.venue = venue
        self.rng = rng or np.random.default_rng()

    def simulate_session_weather(
        self,
        current_weather: WeatherState
    ) -> Tuple[WeatherState, int]:
        """
        Simulate weather for a session.

        Returns:
            Tuple of (new weather state, overs lost to weather)
        """
        overs_lost = 0
        new_condition = current_weather.condition

        # Check for rain
        if self.rng.random() < self.venue.rain_probability:
            if self.rng.random() < 0.3:
                new_condition = WeatherCondition.HEAVY_RAIN
                overs_lost = self.rng.integers(10, 30)
            else:
                new_condition = WeatherCondition.LIGHT_RAIN
                overs_lost = self.rng.integers(3, 10)

        # Check for bad light (more likely in overcast)
        elif self.rng.random() < self.venue.bad_light_probability:
            new_condition = WeatherCondition.BAD_LIGHT
            overs_lost = self.rng.integers(5, 15)

        # Otherwise, sunny or overcast
        elif new_condition in [WeatherCondition.HEAVY_RAIN, WeatherCondition.BAD_LIGHT]:
            # Recovery from bad weather
            new_condition = WeatherCondition.OVERCAST
        else:
            # Random fluctuation
            if self.rng.random() < 0.3:
                new_condition = WeatherCondition.OVERCAST
            else:
                new_condition = WeatherCondition.SUNNY

        return WeatherState(condition=new_condition, overs_lost_today=overs_lost), overs_lost


# =============================================================================
# MAIN MATCH SIMULATOR
# =============================================================================

class TestMatchSimulator:
    """Main simulator for complete test matches"""

    def __init__(self, seed: int = None):
        self.rng = np.random.default_rng(seed)
        self.ball_sim = BallSimulator(self.rng)
        self.over_sim = OverSimulator(self.ball_sim)
        self.declaration_engine = DeclarationEngine(self.rng)
        self.bowler_selector = BowlerSelector(self.rng)

    def simulate_match(
        self,
        team1: Team,
        team2: Team,
        venue: VenueProfile,
        pitch: PitchConditions,
        verbose: bool = False
    ) -> MatchState:
        """
        Simulate a complete test match.

        Args:
            team1: First team
            team2: Second team
            venue: Venue for the match
            pitch: Pitch conditions
            verbose: Whether to print progress

        Returns:
            MatchState with complete match result
        """
        # Reset teams
        team1.reset_for_match()
        team2.reset_for_match()

        # Initialize weather simulator
        weather_sim = WeatherSimulator(venue, self.rng)

        # Toss
        toss_winner = self.rng.choice([team1, team2])
        # Decision based on pitch and conditions
        if pitch.pitch_type in [PitchType.GREEN_TOP, PitchType.PACE_FRIENDLY]:
            toss_decision = "bowl"
        else:
            toss_decision = "bat"

        # Initialize match state
        match = MatchState(
            team1=team1,
            team2=team2,
            venue=venue,
            pitch=pitch,
            weather=WeatherState(),
            toss_winner=toss_winner,
            toss_decision=toss_decision
        )

        if verbose:
            print(f"{toss_winner.name} won the toss and chose to {toss_decision}")

        # Determine batting order based on toss
        if toss_decision == "bat":
            first_batting = toss_winner
            second_batting = team2 if toss_winner == team1 else team1
        else:
            first_batting = team2 if toss_winner == team1 else team1
            second_batting = toss_winner

        # Track total overs available in match
        total_overs_available = MAX_DAYS * OVERS_PER_DAY

        # Simulate match day by day
        while not match.is_complete and match.current_day <= MAX_DAYS:
            self._simulate_day(
                match, first_batting, second_batting, weather_sim, verbose
            )

            if not match.is_complete:
                match.current_day += 1
                match.overs_remaining_today = OVERS_PER_DAY
                pitch.advance_day()

        # Determine result if match not already complete (time ran out)
        if not match.is_complete:
            match.is_complete = True

            # Check if 4th innings was in progress
            if match.innings_states and match.current_innings == 4:
                fourth_innings = match.innings_states[-1]
                target = match.get_target()

                if fourth_innings.runs >= target:
                    # Team chasing won
                    wickets_left = 10 - fourth_innings.wickets
                    match.result = (
                        MatchResult.TEAM1_WIN
                        if fourth_innings.batting_team == match.team1
                        else MatchResult.TEAM2_WIN
                    )
                    match.winning_margin = f"{wickets_left} wickets"
                else:
                    # Time ran out, it's a draw
                    match.result = MatchResult.DRAW
                    match.winning_margin = "Match drawn (time)"
            else:
                # Match didn't reach 4th innings in time
                match.result = MatchResult.DRAW
                match.winning_margin = "Match drawn (time)"

            if verbose:
                print(f"\nMatch result: {match.result.value} - {match.winning_margin}")

        return match

    def _simulate_day(
        self,
        match: MatchState,
        first_batting: Team,
        second_batting: Team,
        weather_sim: WeatherSimulator,
        verbose: bool
    ):
        """Simulate a single day of play"""
        sessions = 3  # Morning, afternoon, evening
        overs_per_session = OVERS_PER_DAY // sessions

        for session in range(sessions):
            if match.is_complete:
                break

            # Weather check
            new_weather, overs_lost = weather_sim.simulate_session_weather(match.weather)
            match.weather = new_weather

            if overs_lost > 0:
                match.overs_remaining_today -= overs_lost
                if verbose:
                    print(f"  Day {match.current_day} Session {session+1}: "
                          f"{overs_lost} overs lost to {new_weather.condition.value}")

            if match.overs_remaining_today <= 0:
                continue

            session_overs = min(overs_per_session, match.overs_remaining_today)

            # Simulate session
            for _ in range(session_overs):
                if match.is_complete:
                    break

                self._simulate_single_over(
                    match, first_batting, second_batting, verbose
                )

                match.overs_remaining_today -= 1

    def _simulate_single_over(
        self,
        match: MatchState,
        first_batting: Team,
        second_batting: Team,
        verbose: bool
    ):
        """Simulate a single over of play"""
        # Get or create current innings
        if not match.innings_states or match.innings_states[-1].is_complete:
            # Start new innings
            if len(match.innings_states) >= 4:
                match.is_complete = True
                self._determine_result(match)
                return

            match.current_innings = len(match.innings_states) + 1

            # Determine batting team for this innings
            if match.current_innings in [1, 3]:
                batting = first_batting
                bowling = second_batting
            else:
                batting = second_batting
                bowling = first_batting

            # Handle 3rd/4th innings order based on follow-on
            if match.current_innings == 3:
                # Check for follow-on
                if len(match.innings_states) >= 2:
                    first_total = match.innings_states[0].runs
                    second_total = match.innings_states[1].runs
                    follow_on_margin = first_total - second_total

                    if follow_on_margin >= 200:
                        # Can enforce follow-on
                        if self.rng.random() < 0.7:  # Usually enforce
                            batting = match.innings_states[1].batting_team
                            bowling = match.innings_states[0].batting_team
                            if verbose:
                                print(f"  Follow-on enforced!")

            batting.reset_for_innings()

            innings = InningsState(
                batting_team=batting,
                bowling_team=bowling,
                innings_number=match.current_innings
            )
            match.innings_states.append(innings)

            if verbose:
                print(f"\n--- Innings {match.current_innings}: {batting.name} batting ---")

        innings = match.innings_states[-1]

        # Check for declaration
        if self.declaration_engine.should_declare(match, innings):
            innings.is_declared = True
            if verbose:
                print(f"  {innings.batting_team.name} declared at {innings.runs}/{innings.wickets}")
            return

        # Get last bowler (can't bowl consecutive overs)
        last_bowler = None
        if innings.overs:
            last_bowler = innings.overs[-1].bowler

        # Select bowler
        bowler = self.bowler_selector.select_bowler(
            innings.bowling_team,
            last_bowler,
            match.pitch,
            {'innings': match.current_innings, 'lead': match.get_lead()}
        )

        # Simulate the over
        match_situation = {
            'innings': match.current_innings,
            'lead': match.get_lead(),
            'overs_remaining': match.overs_remaining_today,
            'target': match.get_target()
        }

        over = self.over_sim.simulate_over(
            innings, bowler, match.pitch, match.venue, match.weather, match_situation
        )
        innings.overs.append(over)

        # Check for match end conditions
        if match.current_innings == 4:
            target = match.get_target()
            if target and innings.runs >= target:
                match.is_complete = True
                wickets_left = 10 - innings.wickets
                match.result = (
                    MatchResult.TEAM1_WIN
                    if innings.batting_team == match.team1
                    else MatchResult.TEAM2_WIN
                )
                match.winning_margin = f"{wickets_left} wickets"
                if verbose:
                    print(f"\n{innings.batting_team.name} wins by {match.winning_margin}!")

        if innings.is_all_out:
            if verbose:
                print(f"  {innings.batting_team.name} all out for {innings.runs}")
            self._check_innings_defeat(match, verbose)

    def _check_innings_defeat(self, match: MatchState, verbose: bool):
        """Check if match ended by innings defeat"""
        if len(match.innings_states) >= 3 and match.current_innings == 3:
            # Check if team batting second lost by innings
            first = match.innings_states[0]
            second = match.innings_states[1]
            third = match.innings_states[2]

            if third.batting_team == second.batting_team:
                # Team batting second batted again (follow-on or regular 3rd)
                combined = second.runs + third.runs
                if combined < first.runs:
                    match.is_complete = True
                    match.result = (
                        MatchResult.TEAM1_WIN
                        if first.batting_team == match.team1
                        else MatchResult.TEAM2_WIN
                    )
                    margin = first.runs - combined
                    match.winning_margin = f"innings and {margin} runs"
                    if verbose:
                        print(f"\n{first.batting_team.name} wins by {match.winning_margin}!")

    def _determine_result(self, match: MatchState):
        """Determine match result when all innings complete"""
        if len(match.innings_states) < 4:
            match.result = MatchResult.DRAW
            match.winning_margin = "Match drawn"
            return

        # Calculate totals
        team1_total = sum(
            inn.runs for inn in match.innings_states
            if inn.batting_team == match.team1
        )
        team2_total = sum(
            inn.runs for inn in match.innings_states
            if inn.batting_team == match.team2
        )

        if team1_total > team2_total:
            match.result = MatchResult.TEAM1_WIN
            match.winning_margin = f"{team1_total - team2_total} runs"
        elif team2_total > team1_total:
            match.result = MatchResult.TEAM2_WIN
            match.winning_margin = f"{team2_total - team1_total} runs"
        else:
            match.result = MatchResult.DRAW
            match.winning_margin = "Match tied"


# =============================================================================
# MONTE CARLO SIMULATION
# =============================================================================

class MonteCarloSimulator:
    """Run multiple match simulations to estimate win probabilities"""

    def __init__(self, seed: int = None):
        self.base_seed = seed or 42

    def run_simulations(
        self,
        team1: Team,
        team2: Team,
        venue: VenueProfile,
        pitch: PitchConditions,
        n_simulations: int = 1000,
        verbose: bool = False
    ) -> Dict:
        """
        Run multiple match simulations.

        Args:
            team1: First team
            team2: Second team
            venue: Venue for match
            pitch: Pitch conditions
            n_simulations: Number of simulations to run
            verbose: Print progress

        Returns:
            Dictionary with win probabilities and statistics
        """
        results = {
            MatchResult.TEAM1_WIN: 0,
            MatchResult.TEAM2_WIN: 0,
            MatchResult.DRAW: 0
        }

        detailed_results = []

        for i in range(n_simulations):
            if verbose and (i + 1) % 100 == 0:
                print(f"Simulation {i + 1}/{n_simulations}")

            # Create fresh copies of teams and pitch
            t1_copy = self._copy_team(team1)
            t2_copy = self._copy_team(team2)
            pitch_copy = PitchConditions(
                pitch_type=pitch.pitch_type,
                initial_pace_assistance=pitch.initial_pace_assistance,
                initial_spin_assistance=pitch.initial_spin_assistance,
                initial_bounce=pitch.initial_bounce,
                deterioration_rate=pitch.deterioration_rate
            )

            # Run simulation with unique seed
            sim = TestMatchSimulator(seed=self.base_seed + i)
            match = sim.simulate_match(t1_copy, t2_copy, venue, pitch_copy, verbose=False)

            results[match.result] += 1
            detailed_results.append({
                'result': match.result,
                'margin': match.winning_margin,
                'innings': [
                    {'team': inn.batting_team.name, 'runs': inn.runs, 'wickets': inn.wickets}
                    for inn in match.innings_states
                ]
            })

        # Calculate probabilities
        probs = {
            'team1_win': results[MatchResult.TEAM1_WIN] / n_simulations,
            'team2_win': results[MatchResult.TEAM2_WIN] / n_simulations,
            'draw': results[MatchResult.DRAW] / n_simulations,
            'team1_name': team1.name,
            'team2_name': team2.name
        }

        # Calculate score distributions
        team1_scores = []
        team2_scores = []
        for dr in detailed_results:
            t1_total = sum(inn['runs'] for inn in dr['innings'] if inn['team'] == team1.name)
            t2_total = sum(inn['runs'] for inn in dr['innings'] if inn['team'] == team2.name)
            team1_scores.append(t1_total)
            team2_scores.append(t2_total)

        stats = {
            'probabilities': probs,
            'total_simulations': n_simulations,
            'results_count': {
                'team1_wins': results[MatchResult.TEAM1_WIN],
                'team2_wins': results[MatchResult.TEAM2_WIN],
                'draws': results[MatchResult.DRAW]
            },
            'score_stats': {
                'team1': {
                    'mean': np.mean(team1_scores),
                    'std': np.std(team1_scores),
                    'min': np.min(team1_scores),
                    'max': np.max(team1_scores)
                },
                'team2': {
                    'mean': np.mean(team2_scores),
                    'std': np.std(team2_scores),
                    'min': np.min(team2_scores),
                    'max': np.max(team2_scores)
                }
            },
            'detailed_results': detailed_results
        }

        return stats

    def _copy_team(self, team: Team) -> Team:
        """Create a deep copy of a team for simulation"""
        players = []
        for p in team.players:
            batting = BattingStats(
                style=p.batting_stats.style,
                skill_rating=p.batting_stats.skill_rating,
                average=p.batting_stats.average,
                strike_rate=p.batting_stats.strike_rate,
                against_pace=p.batting_stats.against_pace,
                against_spin=p.batting_stats.against_spin,
                defensive_ability=p.batting_stats.defensive_ability,
                attacking_ability=p.batting_stats.attacking_ability,
                temperament=p.batting_stats.temperament,
                pressure_handling=p.batting_stats.pressure_handling
            )

            bowling = None
            if p.bowling_stats:
                bowling = BowlingStats(
                    style=p.bowling_stats.style,
                    skill_rating=p.bowling_stats.skill_rating,
                    economy_base=p.bowling_stats.economy_base,
                    wicket_probability_base=p.bowling_stats.wicket_probability_base,
                    stamina=p.bowling_stats.stamina,
                    swing_ability=p.bowling_stats.swing_ability,
                    seam_ability=p.bowling_stats.seam_ability,
                    spin_amount=p.bowling_stats.spin_amount,
                    bounce_variation=p.bowling_stats.bounce_variation
                )

            player = Player(
                name=p.name,
                batting_stats=batting,
                bowling_stats=bowling,
                is_wicketkeeper=p.is_wicketkeeper
            )
            players.append(player)

        return Team(
            name=team.name,
            players=players,
            batting_order=team.batting_order.copy(),
            captain_index=team.captain_index,
            declaration_aggression=team.declaration_aggression
        )


# =============================================================================
# HELPER FUNCTIONS FOR CREATING TEAMS
# =============================================================================

def create_batsman(
    name: str,
    skill: float,
    average: float,
    strike_rate: float = 50.0,
    style: BattingStyle = BattingStyle.RIGHT_HAND,
    vs_pace: float = 0.5,
    vs_spin: float = 0.5,
    is_keeper: bool = False
) -> Player:
    """Helper to create a batsman"""
    return Player(
        name=name,
        batting_stats=BattingStats(
            style=style,
            skill_rating=skill,
            average=average,
            strike_rate=strike_rate,
            against_pace=vs_pace,
            against_spin=vs_spin,
            defensive_ability=0.4 + (skill / 200),
            attacking_ability=strike_rate / 100,
            temperament=0.5,
            pressure_handling=0.5
        ),
        is_wicketkeeper=is_keeper
    )


def create_bowler(
    name: str,
    bat_skill: float,
    bat_avg: float,
    bowl_style: BowlingStyle,
    bowl_skill: float,
    economy: float = 3.0,
    wicket_prob: float = 0.02,
    stamina: float = 100.0
) -> Player:
    """Helper to create a bowling all-rounder or specialist bowler"""
    return Player(
        name=name,
        batting_stats=BattingStats(
            style=BattingStyle.RIGHT_HAND,
            skill_rating=bat_skill,
            average=bat_avg,
            strike_rate=45.0,
            against_pace=0.4,
            against_spin=0.4,
            defensive_ability=0.3,
            attacking_ability=0.3,
            temperament=0.4,
            pressure_handling=0.4
        ),
        bowling_stats=BowlingStats(
            style=bowl_style,
            skill_rating=bowl_skill,
            economy_base=economy,
            wicket_probability_base=wicket_prob,
            stamina=stamina,
            swing_ability=0.6 if bowl_style in [BowlingStyle.FAST, BowlingStyle.FAST_MEDIUM] else 0.2,
            seam_ability=0.6 if bowl_style in [BowlingStyle.FAST, BowlingStyle.FAST_MEDIUM] else 0.2,
            spin_amount=0.7 if bowl_style in [BowlingStyle.OFF_SPIN, BowlingStyle.LEG_SPIN, BowlingStyle.LEFT_ARM_SPIN] else 0.1,
            bounce_variation=0.5
        )
    )


def create_all_rounder(
    name: str,
    bat_skill: float,
    bat_avg: float,
    bat_sr: float,
    bowl_style: BowlingStyle,
    bowl_skill: float,
    economy: float = 3.2,
    wicket_prob: float = 0.018
) -> Player:
    """Helper to create a genuine all-rounder"""
    return Player(
        name=name,
        batting_stats=BattingStats(
            style=BattingStyle.RIGHT_HAND,
            skill_rating=bat_skill,
            average=bat_avg,
            strike_rate=bat_sr,
            against_pace=0.5,
            against_spin=0.5,
            defensive_ability=0.45,
            attacking_ability=bat_sr / 100,
            temperament=0.5,
            pressure_handling=0.5
        ),
        bowling_stats=BowlingStats(
            style=bowl_style,
            skill_rating=bowl_skill,
            economy_base=economy,
            wicket_probability_base=wicket_prob,
            stamina=120.0,
            swing_ability=0.5,
            seam_ability=0.5,
            spin_amount=0.5 if bowl_style in [BowlingStyle.OFF_SPIN, BowlingStyle.LEG_SPIN] else 0.2,
            bounce_variation=0.4
        )
    )


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

def create_sample_teams() -> Tuple[Team, Team]:
    """Create sample teams for testing"""

    # Team 1: India-like team
    india = Team(
        name="India",
        players=[
            create_batsman("R Sharma", 85, 45.5, 55, vs_pace=0.6, vs_spin=0.7),
            create_batsman("S Gill", 75, 42.0, 58, vs_pace=0.55, vs_spin=0.6),
            create_batsman("C Pujara", 80, 43.5, 40, vs_pace=0.65, vs_spin=0.7),
            create_batsman("V Kohli", 90, 50.0, 55, vs_pace=0.7, vs_spin=0.65),
            create_batsman("S Iyer", 70, 38.0, 55, vs_pace=0.5, vs_spin=0.45),
            create_batsman("R Pant", 75, 43.0, 70, is_keeper=True, vs_pace=0.55, vs_spin=0.5),
            create_all_rounder("R Jadeja", 70, 36.0, 60, BowlingStyle.LEFT_ARM_SPIN, 80, 2.4, 0.022),
            create_all_rounder("R Ashwin", 55, 28.0, 50, BowlingStyle.OFF_SPIN, 88, 2.6, 0.025),
            create_bowler("M Shami", 40, 12.0, BowlingStyle.FAST_MEDIUM, 82, 3.0, 0.022),
            create_bowler("J Bumrah", 25, 8.0, BowlingStyle.FAST, 92, 2.5, 0.028),
            create_bowler("M Siraj", 30, 10.0, BowlingStyle.FAST, 78, 3.2, 0.020),
        ],
        captain_index=3,  # Kohli
        declaration_aggression=0.6
    )

    # Team 2: Australia-like team
    australia = Team(
        name="Australia",
        players=[
            create_batsman("U Khawaja", 78, 44.0, 48, vs_pace=0.6, vs_spin=0.55),
            create_batsman("D Warner", 80, 46.0, 70, vs_pace=0.65, vs_spin=0.5),
            create_batsman("M Labuschagne", 85, 48.5, 52, vs_pace=0.65, vs_spin=0.6),
            create_batsman("S Smith", 92, 58.0, 50, vs_pace=0.75, vs_spin=0.7),
            create_batsman("T Head", 72, 40.0, 65, vs_pace=0.55, vs_spin=0.45),
            create_batsman("A Carey", 65, 32.0, 55, is_keeper=True, vs_pace=0.5, vs_spin=0.5),
            create_all_rounder("C Green", 68, 35.0, 52, BowlingStyle.FAST_MEDIUM, 70, 3.2, 0.016),
            create_bowler("P Cummins", 45, 18.0, BowlingStyle.FAST, 90, 2.6, 0.026),
            create_bowler("M Starc", 40, 20.0, BowlingStyle.LEFT_ARM_FAST, 85, 3.0, 0.024),
            create_bowler("J Hazlewood", 30, 12.0, BowlingStyle.FAST_MEDIUM, 84, 2.5, 0.022),
            create_bowler("N Lyon", 35, 14.0, BowlingStyle.OFF_SPIN, 82, 2.8, 0.020),
        ],
        captain_index=7,  # Cummins
        declaration_aggression=0.55
    )

    return india, australia


def create_sample_venue() -> VenueProfile:
    """Create a sample venue"""
    return VenueProfile(
        name="MCG",
        country="Australia",
        pace_friendly=0.6,
        spin_friendly=0.35,
        typical_first_innings_score=380,
        altitude=30,
        humidity=0.45,
        home_advantage_factor=1.1,
        rain_probability=0.15,  # Increased for more draws
        bad_light_probability=0.05
    )


def create_sample_pitch() -> PitchConditions:
    """Create sample pitch conditions"""
    return PitchConditions(
        pitch_type=PitchType.BALANCED,
        initial_pace_assistance=0.6,
        initial_spin_assistance=0.25,
        initial_bounce=0.85,
        deterioration_rate=0.08
    )


def run_example_simulation():
    """Run an example Monte Carlo simulation"""
    print("Creating teams...")
    india, australia = create_sample_teams()
    venue = create_sample_venue()
    pitch = create_sample_pitch()

    print(f"\nMatch: {india.name} vs {australia.name}")
    print(f"Venue: {venue.name}, {venue.country}")
    print(f"Pitch: {pitch.pitch_type.value}")
    print("-" * 50)

    # Run Monte Carlo simulation
    print("\nRunning Monte Carlo simulation (1000 iterations)...")
    mc = MonteCarloSimulator(seed=42)
    results = mc.run_simulations(
        india, australia, venue, pitch,
        n_simulations=1000,
        verbose=True
    )

    # Print results
    probs = results['probabilities']
    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    print(f"\n{india.name} Win Probability: {probs['team1_win']*100:.1f}%")
    print(f"{australia.name} Win Probability: {probs['team2_win']*100:.1f}%")
    print(f"Draw Probability: {probs['draw']*100:.1f}%")

    print(f"\nScore Statistics:")
    print(f"  {india.name}:")
    print(f"    Average Total: {results['score_stats']['team1']['mean']:.0f}")
    print(f"    Std Dev: {results['score_stats']['team1']['std']:.0f}")
    print(f"    Range: {results['score_stats']['team1']['min']:.0f} - {results['score_stats']['team1']['max']:.0f}")

    print(f"  {australia.name}:")
    print(f"    Average Total: {results['score_stats']['team2']['mean']:.0f}")
    print(f"    Std Dev: {results['score_stats']['team2']['std']:.0f}")
    print(f"    Range: {results['score_stats']['team2']['min']:.0f} - {results['score_stats']['team2']['max']:.0f}")

    return results


if __name__ == "__main__":
    run_example_simulation()
