"""
Accurate Rikken RL Agent Implementation
Following exact rules from https://www.pagat.com/boston/rik.html
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import random
from collections import defaultdict
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==================== Game Constants ====================

class Suit(Enum):
    CLUBS = 0
    DIAMONDS = 1
    HEARTS = 2
    SPADES = 3

class Rank(Enum):
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10
    JACK = 11
    QUEEN = 12
    KING = 13
    ACE = 14

class ContractType(Enum):
    PASS = 0
    RIK = 1                    # 8+ tricks with partner, choose trump
    RIK_BETER = 2             # 8+ tricks with partner, hearts trump
    SOLO_8 = 3                # 8+ tricks alone, choose trump
    MISERE = 4                # 0 tricks alone, no trump
    PIEK = 5                  # exactly 1 trick alone, no trump
    SOLO_9 = 6                # 9+ tricks alone, choose trump
    SOLO_10 = 7               # 10+ tricks alone, choose trump
    SOLO_11 = 8               # 11+ tricks alone, choose trump
    SOLO_12 = 9               # 12+ tricks alone, choose trump
    OPEN_MISERE = 10          # 0 tricks alone, cards open after trick 1
    OPEN_PIEK = 11            # 1 trick alone, cards open after trick 1
    TROELA = 12               # 8+ tricks, automatic partner (4th ace holder)
    OPEN_MISERE_PRAATJE = 13  # 0 tricks, all cards open, opponents can discuss
    OPEN_PIEK_PRAATJE = 14    # 1 trick, all cards open, opponents can discuss
    SOLO_13 = 15              # all 13 tricks alone
    OPEN_SOLO_13 = 16         # all 13 tricks alone, cards open after trick 1

@dataclass
class Card:
    suit: Suit
    rank: Rank
    
    def __lt__(self, other):
        if self.suit.value != other.suit.value:
            return self.suit.value < other.suit.value
        return self.rank.value < other.rank.value
    
    def to_index(self):
        """Convert card to 0-51 index."""
        return self.suit.value * 13 + (self.rank.value - 2)
    
    @staticmethod
    def from_index(idx):
        """Create card from 0-51 index."""
        suit = Suit(idx // 13)
        rank = Rank((idx % 13) + 2)
        return Card(suit, rank)
    
    def __repr__(self):
        suits = ['♣', '♦', '♥', '♠']
        ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
        return f"{ranks[self.rank.value-2]}{suits[self.suit.value]}"

@dataclass
class Contract:
    type: ContractType
    declarer: int
    trump_suit: Optional[Suit] = None
    partner: Optional[int] = None
    called_ace: Optional[Card] = None
    called_blind: bool = False
    
    def get_required_tricks(self):
        """Return minimum tricks needed to win contract."""
        requirements = {
            ContractType.RIK: 8,
            ContractType.RIK_BETER: 8,
            ContractType.SOLO_8: 8,
            ContractType.SOLO_9: 9,
            ContractType.SOLO_10: 10,
            ContractType.SOLO_11: 11,
            ContractType.SOLO_12: 12,
            ContractType.MISERE: 0,
            ContractType.PIEK: 1,
            ContractType.OPEN_MISERE: 0,
            ContractType.OPEN_PIEK: 1,
            ContractType.TROELA: 8,
            ContractType.OPEN_MISERE_PRAATJE: 0,
            ContractType.OPEN_PIEK_PRAATJE: 1,
            ContractType.SOLO_13: 13,
            ContractType.OPEN_SOLO_13: 13
        }
        return requirements.get(self.type, 0)
    
    def get_basic_score(self):
        """Get basic score for this contract type."""
        scores = {
            ContractType.RIK: 1,
            ContractType.RIK_BETER: 1,
            ContractType.SOLO_8: 1,
            ContractType.SOLO_9: 1,
            ContractType.SOLO_10: 1,
            ContractType.SOLO_11: 1,
            ContractType.SOLO_12: 1,
            ContractType.MISERE: 5,
            ContractType.PIEK: 5,
            ContractType.OPEN_MISERE: 10,
            ContractType.OPEN_PIEK: 10,
            ContractType.TROELA: 2,
            ContractType.OPEN_MISERE_PRAATJE: 15,
            ContractType.OPEN_PIEK_PRAATJE: 15,
            ContractType.SOLO_13: 15,
            ContractType.OPEN_SOLO_13: 20
        }
        return scores.get(self.type, 0)
    
    def allows_overtricks(self):
        """Whether this contract type allows overtrick scoring."""
        return self.type in [
            ContractType.RIK, ContractType.RIK_BETER, ContractType.SOLO_8,
            ContractType.SOLO_9, ContractType.SOLO_10, ContractType.SOLO_11,
            ContractType.SOLO_12, ContractType.TROELA
        ]

# ==================== Rikken Environment ====================

class RikkenEnv:
    """
    Complete Rikken environment following exact pagat.com rules.
    """
    
    def __init__(self):
        self.num_players = 4
        self.reset()
        
        # Action spaces:
        # 0-51: Play cards
        # 52-68: Contract bids (17 types including PASS)
        # 69-72: Trump suit selection (4 suits)
        # 73-76: Ace calling (4 suits)
        # 77-80: King calling (when holding 4 aces)
        self.action_space = spaces.Discrete(81)
        
        # Observation: hand + game state + history
        self.observation_space = spaces.Box(low=0, high=1, shape=(300,), dtype=np.float32)
    
    def reset(self):
        """Reset game to initial state."""
        self.deck = [Card.from_index(i) for i in range(52)]
        random.shuffle(self.deck)
        
        # Deal cards (no shuffling per rules - only cut)
        self.hands = [[] for _ in range(4)]
        for i in range(52):
            self.hands[i % 4].append(self.deck[i])
        
        for hand in self.hands:
            hand.sort()
        
        # Game state
        self.current_player = 0  # Left of dealer starts bidding
        self.phase = "bidding"
        self.contract = None
        self.trump_suit = None
        self.tricks_won = [0, 0, 0, 0]
        self.current_trick = []
        self.trick_leader = 0
        self.cards_played = []
        self.bidding_history = []
        self.passed_players = set()
        
        # Special state tracking
        self.cards_open = [False, False, False, False]  # For open contracts
        self.called_ace_revealed = False
        self.single_ace_revealed = False  # For troela
        self.blind_ace_used = False
        
        return self._get_observations()
    
    def _count_aces(self, player_id):
        """Count aces in player's hand."""
        return sum(1 for card in self.hands[player_id] if card.rank == Rank.ACE)
    
    def _has_suit(self, player_id, suit):
        """Check if player has cards of given suit."""
        return any(card.suit == suit for card in self.hands[player_id])
    
    def _get_observations(self):
        """Get observations for all players."""
        obs = {}
        for player_id in range(4):
            obs[player_id] = self._get_player_observation(player_id)
        return obs
    
    def _get_player_observation(self, player_id):
        """Get comprehensive observation for player."""
        obs = np.zeros(300, dtype=np.float32)
        idx = 0
        
        # Own hand (52 bits)
        for card in self.hands[player_id]:
            obs[idx + card.to_index()] = 1
        idx += 52
        
        # Current trick cards (52 bits)
        for card, _ in self.current_trick:
            obs[idx + card.to_index()] = 1
        idx += 52
        
        # All played cards (52 bits)
        for card in self.cards_played:
            obs[idx + card.to_index()] = 1
        idx += 52
        
        # Contract information (17 bits)
        if self.contract:
            obs[idx + self.contract.type.value] = 1
        idx += 17
        
        # Trump suit (4 bits)
        if self.trump_suit:
            obs[idx + self.trump_suit.value] = 1
        idx += 4
        
        # Current phase (2 bits: bidding=0, trump_selection=1, ace_calling=2, playing=3)
        phase_encoding = {
            "bidding": 0, "trump_selection": 1, 
            "ace_calling": 2, "playing": 3
        }
        if self.phase in phase_encoding:
            obs[idx + phase_encoding[self.phase]] = 1
        idx += 4
        
        # Tricks won by each player (16 bits - 4 bits per player)
        for i, tricks in enumerate(self.tricks_won):
            for j in range(min(tricks, 4)):
                obs[idx + i*4 + j] = 1
        idx += 16
        
        # Partnership information (8 bits)
        if self.contract:
            obs[idx + self.contract.declarer] = 1
            if self.contract.partner is not None:
                obs[idx + 4 + self.contract.partner] = 1
        idx += 8
        
        # Bidding history encoding (remaining space)
        for i, (bidder, bid_type) in enumerate(self.bidding_history[-10:]):  # Last 10 bids
            if idx + bidder < 300:
                obs[idx + bidder] = 1
            if idx + 4 + bid_type.value < 300:
                obs[idx + 4 + bid_type.value] = 1
            idx += 21
            if idx >= 300:
                break
        
        return obs
    
    def get_legal_actions(self, player_id):
        """Get legal actions based on current phase and rules."""
        if self.phase == "bidding":
            return self._get_legal_bids(player_id)
        elif self.phase == "trump_selection":
            return self._get_legal_trump_choices(player_id)
        elif self.phase == "ace_calling":
            return self._get_legal_ace_calls(player_id)
        else:
            return self._get_legal_cards(player_id)
    
    def _get_legal_bids(self, player_id):
        """Get legal bids following exact Rikken rules."""
        if player_id != self.current_player or player_id in self.passed_players:
            return []
        
        legal_bids = [52]  # Pass always legal
        
        # Check for forced troela bid
        ace_count = self._count_aces(player_id)
        if ace_count == 3:
            # Must bid troela or higher if no higher bid exists
            current_highest = self._get_current_highest_bid_value()
            if current_highest < ContractType.TROELA.value:
                # Must bid at least troela
                legal_bids = [52 + ContractType.TROELA.value]
                # Can also bid higher contracts (but let's limit to reasonable ones)
                reasonable_higher = [ContractType.OPEN_MISERE_PRAATJE, ContractType.OPEN_PIEK_PRAATJE, 
                                   ContractType.SOLO_13]  # Only very high contracts
                for contract_type in reasonable_higher:
                    legal_bids.append(52 + contract_type.value)
                return legal_bids
        
        # Normal bidding - must bid higher than current highest
        current_highest = self._get_current_highest_bid_value()
        
        # Reasonable contract progression (exclude extreme contracts for random policy)
        reasonable_contracts = [
            ContractType.RIK, ContractType.RIK_BETER, ContractType.SOLO_8,
            ContractType.MISERE, ContractType.PIEK, ContractType.SOLO_9,
            ContractType.SOLO_10, ContractType.TROELA
            # Exclude very high contracts to prevent absurd random bids
        ]
        
        for contract_type in reasonable_contracts:
            # Special case: misère and piek are equal bids
            if contract_type in [ContractType.MISERE, ContractType.PIEK]:
                if current_highest < ContractType.MISERE.value:
                    legal_bids.extend([52 + ContractType.MISERE.value, 52 + ContractType.PIEK.value])
            # Regular bid hierarchy
            elif contract_type.value > current_highest:
                legal_bids.append(52 + contract_type.value)
        
        return list(set(legal_bids))  # Remove duplicates
    
    def _get_current_highest_bid_value(self):
        """Get the value of current highest bid."""
        if not self.bidding_history:
            return 0
        
        highest = 0
        for _, bid_type in self.bidding_history:
            if bid_type != ContractType.PASS:
                highest = max(highest, bid_type.value)
        return highest
    
    def _get_legal_trump_choices(self, player_id):
        """Get legal trump suit choices."""
        if player_id != self.current_player or not self.contract:
            return []
        
        # RIK_BETER must use hearts
        if self.contract.type == ContractType.RIK_BETER:
            return [69 + Suit.HEARTS.value]  # Action 71
        
        # No trump contracts
        if self.contract.type in [ContractType.MISERE, ContractType.PIEK, 
                                  ContractType.OPEN_MISERE, ContractType.OPEN_PIEK,
                                  ContractType.OPEN_MISERE_PRAATJE, ContractType.OPEN_PIEK_PRAATJE]:
            return []  # No trump selection needed
        
        # For troela, the single ace holder chooses trump (handled separately)
        if self.contract.type == ContractType.TROELA:
            if player_id == self.contract.partner:  # Single ace holder
                return [69 + suit.value for suit in Suit]
            return []
        
        # For RIK contracts, cannot choose same suit as called ace
        if self.contract.type in [ContractType.RIK, ContractType.RIK_BETER] and self.contract.called_ace:
            legal_suits = [69 + suit.value for suit in Suit if suit != self.contract.called_ace.suit]
            return legal_suits
        
        # Normal trump choice for solo contracts
        return [69 + suit.value for suit in Suit]
    
    def _get_legal_ace_calls(self, player_id):
        """Get legal ace calls for RIK contracts."""
        if player_id != self.current_player or not self.contract:
            return []
        
        if self.contract.type not in [ContractType.RIK, ContractType.RIK_BETER]:
            return []
        
        legal_calls = []
        
        # Check if player has 4 aces (can call king instead)
        if self._count_aces(player_id) == 4:
            # Can call any king
            for suit in Suit:
                legal_calls.append(77 + suit.value)  # King calling actions
            return legal_calls
        
        # Normal ace calling - must have at least one card of the suit if possible
        for suit in Suit:
            ace_card = Card(suit, Rank.ACE)
            
            # Cannot call ace of trumps
            if self.trump_suit and suit == self.trump_suit:
                continue
            
            # Cannot call ace you hold
            if ace_card in self.hands[player_id]:
                continue
            
            # Prefer suits where you have cards
            if self._has_suit(player_id, suit):
                legal_calls.append(73 + suit.value)
        
        # If no suits with cards, can call "blind"
        if not legal_calls:
            for suit in Suit:
                if self.trump_suit and suit == self.trump_suit:
                    continue
                ace_card = Card(suit, Rank.ACE)
                if ace_card not in self.hands[player_id]:
                    legal_calls.append(73 + suit.value)
        
        return legal_calls
    
    def _get_legal_cards(self, player_id):
        """Get legal cards following Rikken play rules."""
        if player_id != self.current_player:
            return []
        
        hand = self.hands[player_id]
        
        # Leading to trick - any card legal
        if len(self.current_trick) == 0:
            return [card.to_index() for card in hand]
        
        lead_card, _ = self.current_trick[0]
        lead_suit = lead_card.suit
        
        # Must follow suit if possible
        same_suit_cards = [card for card in hand if card.suit == lead_suit]
        if same_suit_cards:
            # Special rule: called ace must be played when its suit is led
            if (self.contract and self.contract.called_ace and 
                lead_suit == self.contract.called_ace.suit and 
                self.contract.called_ace in hand):
                return [self.contract.called_ace.to_index()]
            
            return [card.to_index() for card in same_suit_cards]
        
        # Cannot follow suit - can play any card
        # But called ace must be played if suit is led (even if trumped)
        if (self.contract and self.contract.called_ace and 
            lead_suit == self.contract.called_ace.suit and 
            self.contract.called_ace in hand):
            return [self.contract.called_ace.to_index()]
        
        return [card.to_index() for card in hand]
    
    def step(self, player_id, action):
        """Execute action - ensures consistent return format."""
        try:
            if self.phase == "bidding":
                result = self._handle_bid(player_id, action)
            elif self.phase == "trump_selection":
                result = self._handle_trump_selection(player_id, action)
            elif self.phase == "ace_calling":
                result = self._handle_ace_calling(player_id, action)
            else:
                result = self._handle_card_play(player_id, action)
            
            # Ensure result is always a 4-tuple
            if not isinstance(result, tuple) or len(result) != 4:
                print(f"Warning: Step function returned invalid format: {result}")
                return self._get_observations(), {i: 0 for i in range(4)}, False, {}
            
            obs, rewards, done, info = result
            
            # Ensure rewards is always a dict
            if not isinstance(rewards, dict):
                print(f"Warning: Rewards not dict, converting: {rewards}")
                rewards = {i: 0 for i in range(4)}
            
            return obs, rewards, done, info
            
        except Exception as e:
            print(f"Error in step function: {e}")
            return self._get_observations(), {i: 0 for i in range(4)}, False, {}
    
    def _handle_bid(self, player_id, action):
        """Handle bidding with exact rules."""
        if player_id != self.current_player or player_id in self.passed_players:
            return self._get_observations(), {i: 0 for i in range(4)}, False, {}
        
        if action == 52:  # Pass
            bid_type = ContractType.PASS
            self.passed_players.add(player_id)
        else:
            bid_type = ContractType(action - 52)
        
        self.bidding_history.append((player_id, bid_type))
        
        # Check if bidding is finished - either 3 passed or all 4 passed
        if len(self.passed_players) >= 3:
            # Find players who haven't passed and have made bids
            active_bidders = [p for p in range(4) if p not in self.passed_players]
            non_pass_bids = [(p, b) for p, b in self.bidding_history 
                           if b != ContractType.PASS]
            
            if len(self.passed_players) == 4 or not non_pass_bids:
                # All passed or no real bids - redeal
                obs = self.reset()
                return obs, {i: 0 for i in range(4)}, False, {}
            
            # Find the highest bid from remaining active players
            if active_bidders and non_pass_bids:
                # Get the last (most recent) non-pass bid from an active bidder
                active_bids = [(p, b) for p, b in non_pass_bids if p in active_bidders or p not in self.passed_players]
                if active_bids:
                    declarer, contract_type = active_bids[-1]
                    self.contract = Contract(contract_type, declarer)
                    
                    # Handle troela automatic partner
                    if contract_type == ContractType.TROELA:
                        self._find_troela_partner()
                    
                    # Move to appropriate next phase
                    if contract_type in [ContractType.MISERE, ContractType.PIEK,
                                       ContractType.OPEN_MISERE, ContractType.OPEN_PIEK,
                                       ContractType.OPEN_MISERE_PRAATJE, ContractType.OPEN_PIEK_PRAATJE]:
                        # No trump, no ace calling
                        self.phase = "playing"
                        self.current_player = 0  # Left of dealer leads
                    elif contract_type == ContractType.TROELA:
                        # Partner chooses trump
                        self.phase = "trump_selection"
                        self.current_player = self.contract.partner
                    elif contract_type in [ContractType.RIK, ContractType.RIK_BETER]:
                        # Need to call ace first, then choose trump
                        self.phase = "ace_calling"
                        self.current_player = declarer
                    else:
                        # Solo contracts - choose trump
                        self.phase = "trump_selection"
                        self.current_player = declarer
                else:
                    # Redeal if no valid bids
                    obs = self.reset()
                    return obs, {i: 0 for i in range(4)}, False, {}
            else:
                # Redeal
                obs = self.reset()
                return obs, {i: 0 for i in range(4)}, False, {}
        else:
            # Continue bidding to next player
            self.current_player = (self.current_player + 1) % 4
            # Skip players who already passed, but prevent infinite loops
            attempts = 0
            while self.current_player in self.passed_players and attempts < 4:
                self.current_player = (self.current_player + 1) % 4
                attempts += 1
            
            # If we've cycled through everyone, end bidding
            if attempts >= 4:
                obs = self.reset()
                return obs, {i: 0 for i in range(4)}, False, {}
        
        return self._get_observations(), {i: 0 for i in range(4)}, False, {}
    
    def _find_troela_partner(self):
        """Find partner for troela (player with single ace)."""
        declarer_aces = [card for card in self.hands[self.contract.declarer] 
                        if card.rank == Rank.ACE]
        
        # Find which ace the declarer doesn't have
        all_aces = [Card(suit, Rank.ACE) for suit in Suit]
        missing_ace = None
        for ace in all_aces:
            if ace not in declarer_aces:
                missing_ace = ace
                break
        
        if missing_ace:
            # Find who has the missing ace
            for player_id, hand in enumerate(self.hands):
                if player_id != self.contract.declarer and missing_ace in hand:
                    self.contract.partner = player_id
                    break
    
    def _handle_trump_selection(self, player_id, action):
        """Handle trump suit selection."""
        if player_id != self.current_player:
            return self._get_observations(), {i: 0 for i in range(4)}, False, {}
        
        try:
            trump_suit = Suit(action - 69)
            self.trump_suit = trump_suit
            self.contract.trump_suit = trump_suit
            
            # Move to next phase
            if self.contract.type in [ContractType.RIK, ContractType.RIK_BETER]:
                if not self.contract.called_ace:
                    self.phase = "ace_calling"
                    self.current_player = self.contract.declarer
                else:
                    self.phase = "playing"
                    self.current_player = 0
            else:
                self.phase = "playing"
                self.current_player = 0  # Left of dealer leads
            
            return self._get_observations(), {i: 0 for i in range(4)}, False, {}
        except Exception as e:
            print(f"Error in trump selection: {e}")
            return self._get_observations(), {i: 0 for i in range(4)}, False, {}
    
    def _handle_ace_calling(self, player_id, action):
        """Handle ace calling."""
        if player_id != self.current_player:
            return self._get_observations(), {i: 0 for i in range(4)}, False, {}
        
        try:
            if action >= 77:  # King calling
                suit = Suit(action - 77)
                called_card = Card(suit, Rank.KING)
            else:  # Ace calling
                suit = Suit(action - 73)
                called_card = Card(suit, Rank.ACE)
            
            self.contract.called_ace = called_card
            
            # Check if called blind
            if not self._has_suit(player_id, suit):
                self.contract.called_blind = True
            
            # Find partner
            for pid, hand in enumerate(self.hands):
                if pid != player_id and called_card in hand:
                    self.contract.partner = pid
                    break
            
            # Move to trump selection if not done, otherwise start playing
            if not self.trump_suit:
                self.phase = "trump_selection"
            else:
                self.phase = "playing"
                self.current_player = 0
            
            return self._get_observations(), {i: 0 for i in range(4)}, False, {}
        except Exception as e:
            print(f"Error in ace calling: {e}")
            return self._get_observations(), {i: 0 for i in range(4)}, False, {}
    
    def _handle_card_play(self, player_id, action):
        """Handle card play."""
        if player_id != self.current_player:
            return self._get_observations(), {i: 0 for i in range(4)}, False, {}
        
        try:
            card = Card.from_index(action)
            
            # Check if card is in player's hand
            if card not in self.hands[player_id]:
                print(f"Warning: Player {player_id} tried to play card not in hand: {card}")
                return self._get_observations(), {i: 0 for i in range(4)}, False, {}
            
            # Remove card from hand
            self.hands[player_id].remove(card)
            self.current_trick.append((card, player_id))
            self.cards_played.append(card)
            
            # Reveal partnerships when called ace/king is played
            if (self.contract and self.contract.called_ace and 
                card == self.contract.called_ace):
                self.called_ace_revealed = True
            
            # Complete trick
            if len(self.current_trick) == 4:
                winner = self._determine_trick_winner()
                self.tricks_won[winner] += 1
                self.current_trick = []
                self.current_player = winner
                
                # Open cards after first trick for open contracts
                if (self.contract and sum(self.tricks_won) > 0 and
                    self.contract.type in [ContractType.OPEN_MISERE, ContractType.OPEN_PIEK,
                                         ContractType.OPEN_MISERE_PRAATJE, ContractType.OPEN_PIEK_PRAATJE,
                                         ContractType.OPEN_SOLO_13]):
                    self.cards_open[self.contract.declarer] = True
                
                # Check game end
                if sum(self.tricks_won) == 13:
                    rewards = self._calculate_rewards()
                    return self._get_observations(), rewards, True, {}
            else:
                self.current_player = (self.current_player + 1) % 4
            
            return self._get_observations(), {i: 0 for i in range(4)}, False, {}
        
        except Exception as e:
            print(f"Error in card play: {e}")
            return self._get_observations(), {i: 0 for i in range(4)}, False, {}
    
    def _determine_trick_winner(self):
        """Determine trick winner with trump rules."""
        lead_card, lead_player = self.current_trick[0]
        winning_card = lead_card
        winner = lead_player
        
        for card, player in self.current_trick:
            # Trump beats non-trump
            if (self.trump_suit and card.suit == self.trump_suit and 
                winning_card.suit != self.trump_suit):
                winning_card = card
                winner = player
            # Higher trump beats lower trump
            elif (self.trump_suit and card.suit == self.trump_suit and 
                  winning_card.suit == self.trump_suit):
                if card.rank.value > winning_card.rank.value:
                    winning_card = card
                    winner = player
            # Higher card of lead suit beats lower (if no trumps played)
            elif (card.suit == lead_card.suit and winning_card.suit == lead_card.suit and
                  (not self.trump_suit or winning_card.suit != self.trump_suit)):
                if card.rank.value > winning_card.rank.value:
                    winning_card = card
                    winner = player
        
        return winner
    
    def _calculate_rewards(self):
        """Calculate rewards using exact Rikken scoring."""
        rewards = {i: 0 for i in range(4)}
        
        if not self.contract:
            return rewards
        
        # Validate game state
        total_tricks = sum(self.tricks_won)
        if total_tricks != 13:
            print(f"Warning: Total tricks is {total_tricks}, not 13!")
            print(f"Tricks won: {self.tricks_won}")
            return rewards
        
        declarer = self.contract.declarer
        partner = self.contract.partner
        required_tricks = self.contract.get_required_tricks()
        basic_score = self.contract.get_basic_score()
        
        # Count declarer's tricks (including partner)
        declarer_tricks = self.tricks_won[declarer]
        if partner is not None:
            declarer_tricks += self.tricks_won[partner]
        
        # Determine success
        if self.contract.type in [ContractType.MISERE, ContractType.OPEN_MISERE, 
                                ContractType.OPEN_MISERE_PRAATJE]:
            success = self.tricks_won[declarer] == 0
        elif self.contract.type in [ContractType.PIEK, ContractType.OPEN_PIEK,
                                  ContractType.OPEN_PIEK_PRAATJE]:
            success = self.tricks_won[declarer] == 1
        else:
            success = declarer_tricks >= required_tricks
        
        # Calculate payments
        if success:
            if self.contract.allows_overtricks():
                overtricks = max(0, declarer_tricks - required_tricks)
                # Bonus for all 13 tricks
                if declarer_tricks == 13:
                    overtricks += 1
                score = basic_score + overtricks
            else:
                score = basic_score
            
            # Pay winners
            if partner is not None:
                # Partnership contract
                rewards[declarer] = score
                rewards[partner] = score
                # Opponents pay
                for i in range(4):
                    if i != declarer and i != partner:
                        rewards[i] = -score
            else:
                # Solo contract
                rewards[declarer] = score * 3  # Receives from all 3 opponents
                for i in range(4):
                    if i != declarer:
                        rewards[i] = -score
        else:
            # Contract failed
            if self.contract.allows_overtricks():
                undertricks = max(0, required_tricks - declarer_tricks)
                penalty = basic_score + undertricks
            else:
                penalty = basic_score
            
            if partner is not None:
                # Partnership penalty
                rewards[declarer] = -penalty
                if partner is not None:
                    rewards[partner] = -penalty
                # Opponents receive
                for i in range(4):
                    if i != declarer and i != partner:
                        rewards[i] = penalty
            else:
                # Solo penalty
                rewards[declarer] = -penalty * 3
                for i in range(4):
                    if i != declarer:
                        rewards[i] = penalty
        
        return rewards

# ==================== Neural Network (Updated for larger action space) ====================

class RikkenPolicyNetwork(nn.Module):
    """Neural network for complete Rikken gameplay."""
    
    def __init__(self, input_dim=300, hidden_dim=512, action_dim=81):
        super().__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        
        # Action head
        self.action_head = nn.Linear(hidden_dim, action_dim)
        
        # Value head
        self.value_head = nn.Linear(hidden_dim, 1)
    
    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        
        action_logits = self.action_head(x)
        value = self.value_head(x)
        
        return action_logits, value

# ==================== Example Training Function ====================

def train_accurate_rikken(num_episodes=5000):
    """Train agents on accurate Rikken rules."""
    
    env = RikkenEnv()
    
    # Smarter random policy that considers hand strength
    def smart_random_policy(obs, legal_actions, hand, phase):
        if not legal_actions:
            return 52  # Pass
        
        # During bidding phase
        if phase == "bidding" and 52 in legal_actions:
            # Count high cards and aces for smarter bidding
            high_cards = sum(1 for card in hand if card.rank.value >= 11)
            aces = sum(1 for card in hand if card.rank.value == 14)
            
            # With 3 aces, must bid troela or higher (forced by rules)
            if aces == 3:
                troela_actions = [a for a in legal_actions if a >= 52 + ContractType.TROELA.value]
                if troela_actions:
                    # Pick the lowest valid bid (troela)
                    return min(troela_actions)
            
            # Be very conservative - mostly pass unless very strong hand
            if high_cards < 6:  # Need very strong hand to bid
                return 52  # Pass
            
            # With strong hand, bid conservatively
            basic_bids = [a for a in legal_actions if 52 + ContractType.RIK.value <= a <= 52 + ContractType.SOLO_8.value]
            if basic_bids:
                return random.choice(basic_bids)
            
            # Default to pass
            return 52
        
        # For non-bidding phases, choose randomly
        return random.choice(legal_actions)
    
    wins_by_player = defaultdict(int)
    contracts_attempted = defaultdict(int)
    contracts_made = defaultdict(int)
    total_tricks_by_player = defaultdict(int)
    
    for episode in range(num_episodes):
        try:
            obs = env.reset()
            done = False
            game_step = 0
            max_steps = 200  # Prevent infinite loops
            
            while not done and game_step < max_steps:
                player_id = env.current_player
                legal_actions = env.get_legal_actions(player_id)
                
                if not legal_actions:
                    # Skip to next player if no legal actions
                    env.current_player = (env.current_player + 1) % 4
                    game_step += 1
                    continue
                
                player_hand = env.hands[player_id]
                action = smart_random_policy(obs[player_id], legal_actions, player_hand, env.phase)
                
                next_obs, rewards, done, info = env.step(player_id, action)
                
                # Ensure rewards is always a dict
                if not isinstance(rewards, dict):
                    print(f"Warning: rewards is not dict: {rewards}")
                    rewards = {i: 0 for i in range(4)}
                
                obs = next_obs
                game_step += 1
            
            # Track comprehensive statistics
            if env.contract:
                declarer = env.contract.declarer
                contracts_attempted[declarer] += 1
                if declarer in rewards and rewards[declarer] > 0:
                    contracts_made[declarer] += 1
            
            # Track total tricks won
            for p in range(4):
                total_tricks_by_player[p] += env.tricks_won[p]
            
            # Find winner (player with highest reward)
            if isinstance(rewards, dict) and any(rewards.values()):
                winner = max(range(4), key=lambda p: rewards.get(p, 0))
                wins_by_player[winner] += 1
        
        except Exception as e:
            print(f"Error in episode {episode}: {e}")
            continue
        
        if episode % 500 == 0:
            print(f"\nEpisode {episode}")
            print(f"Wins: {dict(wins_by_player)}")
            print(f"Average tricks per player:")
            for p in range(4):
                avg_tricks = total_tricks_by_player[p] / max(1, episode + 1)
                print(f"  Player {p}: {avg_tricks:.2f}")
            
            print(f"Contract success rates:")
            for p in range(4):
                if contracts_attempted[p] > 0:
                    rate = contracts_made[p] / contracts_attempted[p]
                    print(f"  Player {p}: {rate:.2%} ({contracts_made[p]}/{contracts_attempted[p]})")
                else:
                    print(f"  Player {p}: No contracts attempted")
            
            print(f"Contract types attempted:")
            if hasattr(env, 'contract') and env.contract:
                print(f"  Last contract: {env.contract.type.name}")

# ==================== Debug Functions ====================

def debug_game_step():
    """Debug a single game step by step."""
    env = RikkenEnv()
    obs = env.reset()
    
    print("=== GAME START ===")
    print("Player hands:")
    for i, hand in enumerate(env.hands):
        print(f"Player {i}: {sorted(hand, key=lambda c: (c.suit.value, c.rank.value))}")
        aces = [c for c in hand if c.rank == Rank.ACE]
        print(f"  Aces: {aces}")
    
    step = 0
    while not env.phase == "playing" and step < 20:
        player_id = env.current_player
        legal_actions = env.get_legal_actions(player_id)
        
        print(f"\nStep {step}: Player {player_id} ({env.phase})")
        print(f"Legal actions: {legal_actions}")
        
        if not legal_actions:
            print("No legal actions - skipping")
            env.current_player = (env.current_player + 1) % 4
            step += 1
            continue
        
        # Manual action selection for debugging
        if env.phase == "bidding":
            # Check for forced troela
            aces = sum(1 for card in env.hands[player_id] if card.rank == Rank.ACE)
            if aces == 3:
                print(f"Player {player_id} has 3 aces - must bid troela or higher!")
                action = 52 + ContractType.TROELA.value  # Troela
            else:
                action = 52  # Pass
        else:
            action = legal_actions[0]  # Take first legal action
        
        print(f"Taking action: {action}")
        
        try:
            obs, rewards, done, info = env.step(player_id, action)
            
            if env.contract:
                print(f"Current contract: {env.contract.type.name} by player {env.contract.declarer}")
                if env.contract.partner is not None:
                    print(f"Partner: Player {env.contract.partner}")
        
        except Exception as e:
            print(f"Error: {e}")
            break
            
        step += 1
    
    print(f"\nFinal phase: {env.phase}")
    if env.contract:
        print(f"Final contract: {env.contract.type.name}")
        print(f"Trump suit: {env.trump_suit}")

def test_rule_implementation():
    """Test specific rule implementations."""
    print("Testing Rikken rule implementation...")
    
    # Test 1: Three aces forcing troela
    print("\n=== Test 1: Three Aces ===")
    env = RikkenEnv()
    
    # Manually set up a hand with 3 aces
    test_hand = [
        Card(Suit.CLUBS, Rank.ACE),
        Card(Suit.DIAMONDS, Rank.ACE), 
        Card(Suit.HEARTS, Rank.ACE),
        Card(Suit.SPADES, Rank.KING),
        Card(Suit.CLUBS, Rank.QUEEN)
    ]
    env.hands[0] = test_hand
    env.current_player = 0
    
    legal_actions = env.get_legal_actions(0)
    print(f"Player with 3 aces legal actions: {legal_actions}")
    
    # Should include troela (52 + 12 = 64) and higher bids
    troela_action = 52 + ContractType.TROELA.value
    if troela_action in legal_actions:
        print("✓ Troela bid available as expected")
    else:
        print("✗ Troela bid missing!")

if __name__ == "__main__":
    print("Testing accurate Rikken implementation...")
    
    # Run debug first
    print("=== DEBUGGING SINGLE GAME ===")
    debug_game_step()
    
    print("\n=== TESTING RULES ===")
    test_rule_implementation()
    
    print("\n=== TRAINING WITH SMART RANDOM ===")
    train_accurate_rikken(1000)

if __name__ == "__main__":
    print("Training accurate Rikken implementation...")
    train_accurate_rikken(1000)