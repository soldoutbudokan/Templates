# Minimal, self-contained Rikken-like environment with the API expected by the trainer/analyzer.
# This is a lightweight mock (NOT full game rules) so you can train/run the pipeline end-to-end.
# Replace this with your real environment later, but keep the same attributes/methods.

from __future__ import annotations
import random
from dataclasses import dataclass
from enum import IntEnum, Enum
from typing import List, Dict, Optional, Tuple

# ---------------- Cards ----------------

class Suit(Enum):
    CLUBS = 0
    DIAMONDS = 1
    HEARTS = 2
    SPADES = 3

class Rank(IntEnum):
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

@dataclass(frozen=True)
class Card:
    suit: Suit
    rank: Rank

# ---------------- Contracts ----------------

class ContractType(IntEnum):
    RIK = 0
    RIK_BETER = 1
    TROELA = 2
    SOLO_8 = 3
    SOLO_9 = 4
    SOLO_10 = 5
    SOLO_11 = 6
    SOLO_12 = 7
    SOLO_13 = 8
    MISERE = 9
    PIEK = 10
    OPEN_MISERE = 11
    OPEN_PIEK = 12
    # Add more as needed

@dataclass
class Contract:
    type: ContractType
    declarer: int
    partner: int  # for simplicity we set partner = (declarer+2) % 4 in this mock

# ---------------- Environment ----------------

class RikkenEnv:
    """
    Mock environment exposing:
      - attributes: current_player, hands, phase, trump_suit, contract, tricks_won, bidding_history
      - methods: reset(), get_legal_actions(player_id), step(player_id, action)
    Action ids:
      - 0..51  -> play that card index (suit*13 + (rank-2))
      - 52     -> PASS (during bidding)
      - 52 + k -> bid ContractType(k)
    This mock keeps rules simple to allow training to run; replace with your full logic as needed.
    """

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)
        self.current_player: int = 0
        self.hands: List[List[Card]] = [[] for _ in range(4)]
        self.phase: str = "bidding"  # "bidding" -> "play"
        self.trump_suit: Optional[Suit] = None
        self.contract: Optional[Contract] = None
        self.bidding_history: List[Tuple[int, str]] = []  # (player, "PASS" or "BID:<name>")
        self.tricks_won: List[int] = [0, 0, 0, 0]
        self._passes_in_row = 0
        self._last_bid: Optional[Tuple[int, ContractType]] = None  # (player, type)
        self._cards_played_count = 0
        self._max_play_steps = 32  # short episodes for speed

    # ------------ Helpers ------------

    @staticmethod
    def _card_index(card: Card) -> int:
        return card.suit.value * 13 + (int(card.rank) - 2)

    @staticmethod
    def _index_to_card(idx: int) -> Card:
        suit = Suit(idx // 13)
        rank = Rank((idx % 13) + 2)
        return Card(suit, rank)

    def _deal(self):
        deck = [Card(s, Rank(r)) for s in Suit for r in range(2, 15)]
        self.rng.shuffle(deck)
        self.hands = [deck[i*13:(i+1)*13] for i in range(4)]

    def _evaluate_hand_strength(self, hand: List[Card]) -> int:
        # Simple high-card point count
        p = 0
        for c in hand:
            if c.rank >= Rank.TEN:
                p += int(c.rank) - 9  # 10->1, J->2, Q->3, K->4, A->5
        return p

    # ------------ Public API ------------

    def reset(self):
        self._deal()
        self.phase = "bidding"
        self.current_player = 0
        self.trump_suit = None
        self.contract = None
        self.tricks_won = [0, 0, 0, 0]
        self.bidding_history = []
        self._passes_in_row = 0
        self._last_bid = None
        self._cards_played_count = 0
        # Observation per-seat is built in the trainer; here we return a dict for compatibility
        return {i: None for i in range(4)}

    def get_legal_actions(self, player_id: int):
        if self.phase == "bidding":
            base = [52]  # PASS
            # Expose a small set of bids for training
            allowed = [
                ContractType.RIK, ContractType.RIK_BETER, ContractType.TROELA,
                ContractType.SOLO_8, ContractType.SOLO_9, ContractType.SOLO_10,
                ContractType.MISERE
            ]
            bids = [52 + int(ct) for ct in allowed]
            return base + bids
        else:
            # Play phase: any card in hand may be played (no enforcement in mock)
            hand = self.hands[player_id]
            return [self._card_index(c) for c in hand]

    def step(self, player_id: int, action: int):
        assert player_id == self.current_player, "Only current_player may act"

        info = {}
        rewards: Dict[int, int] = {0: 0, 1: 0, 2: 0, 3: 0}
        done = False

        if self.phase == "bidding":
            if action == 52:
                self.bidding_history.append((player_id, "PASS"))
                self._passes_in_row += 1
            elif action >= 52:
                ct_val = action - 52
                try:
                    ctype = ContractType(ct_val)
                except ValueError:
                    ctype = ContractType.RIK
                self._last_bid = (player_id, ctype)
                self._passes_in_row = 0
                self.bidding_history.append((player_id, f"BID:{ctype.name}"))
            else:
                # Invalid during bidding -> treat as PASS
                self.bidding_history.append((player_id, "PASS"))
                self._passes_in_row += 1

            # End bidding: if 3 passes after a last bid, accept last bid
            if self._last_bid and self._passes_in_row >= 3:
                dec, ctype = self._last_bid
                partner = (dec + 2) % 4
                self.contract = Contract(ctype, dec, partner)
                # pick random trump for simplicity (none for MISERE)
                self.trump_suit = None if ctype in (ContractType.MISERE, ContractType.OPEN_MISERE) \
                    else Suit(self.rng.randrange(4))
                self.phase = "play"
                self.current_player = dec
            else:
                self.current_player = (self.current_player + 1) % 4

        else:
            # Play phase
            hand = self.hands[player_id]
            # If illegal, choose first legal
            idxs = self.get_legal_actions(player_id)
            if action not in idxs:
                action = idxs[0]
            # remove card from hand
            # find the card by index
            for i, c in enumerate(hand):
                if self._card_index(c) == action:
                    hand.pop(i)
                    break

            # crude "trick" accounting: every 4 plays equals a trick, winner random-biased
            self._cards_played_count += 1
            if self._cards_played_count % 4 == 0:
                # Bias winner toward declarer team if they have stronger average hands
                winner = self.rng.randrange(4)
                if self.contract:
                    dec, part = self.contract.declarer, self.contract.partner
                    team = {dec, part}
                    opp = {0, 1, 2, 3} - team
                    dec_strength = self._evaluate_hand_strength(self.hands[dec])
                    part_strength = self._evaluate_hand_strength(self.hands[part])
                    opp_strength = sum(self._evaluate_hand_strength(self.hands[x]) for x in opp)
                    if dec_strength + part_strength + self.rng.randint(0, 5) > opp_strength:
                        winner = dec if self.rng.random() < 0.5 else part
                self.tricks_won[winner] += 1
                self.current_player = winner
            else:
                self.current_player = (self.current_player + 1) % 4

            # End after fixed number of plays or when all hands empty
            if all(len(h) == 0 for h in self.hands) or self._cards_played_count >= 32:
                done = True
                # Determine success by simple threshold on tricks for declarer team
                if self.contract:
                    dec, part = self.contract.declarer, self.contract.partner
                    team_tricks = self.tricks_won[dec] + self.tricks_won[part]
                    opp_tricks = sum(self.tricks_won) - team_tricks
                    # crude target: 7+ tricks to succeed (mock)
                    success = team_tricks >= 7
                    if success:
                        rewards[dec] = 1
                        rewards[part] = 1
                        for o in {0, 1, 2, 3} - {dec, part}:
                            rewards[o] = -1
                    else:
                        rewards[dec] = -1
                        rewards[part] = -1
                        for o in {0, 1, 2, 3} - {dec, part}:
                            rewards[o] = 1
                else:
                    # no contract => zero-sum draw
                    pass

        # Return obs as dict for compatibility (trainer builds per-seat features itself)
        obs = {i: None for i in range(4)}
        return obs, rewards, done, info
