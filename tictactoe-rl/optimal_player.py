"""
Optimal Tic Tac Toe player using Minimax algorithm.
Used as baseline for measuring RL agent performance.
"""
from typing import Tuple, Optional, Dict, List
from game import TicTacToe
import random


class OptimalPlayer:
    """
    Optimal player using Minimax algorithm.
    Guaranteed to never lose in Tic Tac Toe.
    """

    def __init__(self):
        # Cache for minimax values only (not actions)
        self._value_cache: Dict[Tuple[int, ...], float] = {}

    def get_action(self, game: TicTacToe) -> int:
        """Get the optimal action for the current game state."""
        valid_actions = game.get_valid_actions()

        if not valid_actions:
            raise ValueError("No valid actions")

        # Evaluate all actions and pick the best one(s)
        action_values = []
        for action in valid_actions:
            child = game.clone()
            child.step(action)
            # Negate because we want value from current player's perspective
            value = -self._minimax(child)
            action_values.append((action, value))

        # Find best value
        best_value = max(v for _, v in action_values)
        best_actions = [a for a, v in action_values if v == best_value]

        return random.choice(best_actions)

    def _minimax(self, game: TicTacToe) -> float:
        """
        Return the value of the game state from current player's perspective.

        +1 = current player wins
        -1 = current player loses
         0 = draw
        """
        state = game.get_state()

        # Check cache
        cache_key = (state, game.current_player)
        if cache_key in self._value_cache:
            return self._value_cache[cache_key]

        # Terminal state
        if game.done:
            if game.winner == 0:
                return 0.0
            # Winner is from perspective of the player who just moved (opponent)
            # So if there's a winner and game is done, current player lost
            return -1.0

        # Recursive case: maximize over all actions
        valid_actions = game.get_valid_actions()
        best_value = -float('inf')

        for action in valid_actions:
            child = game.clone()
            child.step(action)
            # Negate because opponent's loss is our gain
            value = -self._minimax(child)
            best_value = max(best_value, value)

        self._value_cache[cache_key] = best_value
        return best_value

    def get_optimal_value(self, game: TicTacToe) -> float:
        """Get the optimal value (expected outcome) from current state."""
        if game.done:
            if game.winner == game.current_player:
                return 1.0
            elif game.winner == -game.current_player:
                return -1.0
            else:
                return 0.0
        return self._minimax(game)

    def get_all_optimal_actions(self, game: TicTacToe) -> List[int]:
        """Return all actions that lead to optimal play."""
        valid_actions = game.get_valid_actions()
        if not valid_actions:
            return []

        action_values = []
        for action in valid_actions:
            child = game.clone()
            child.step(action)
            value = -self._minimax(child)
            action_values.append((action, value))

        best_value = max(v for _, v in action_values)
        return [a for a, v in action_values if v == best_value]


class RandomPlayer:
    """Random player for comparison."""

    def get_action(self, game: TicTacToe) -> int:
        """Get a random valid action."""
        return random.choice(game.get_valid_actions())


def verify_optimal_player():
    """Verify that optimal player never loses."""
    optimal = OptimalPlayer()
    random_player = RandomPlayer()

    # Test: optimal vs optimal should always draw
    print("Testing optimal vs optimal (100 games)...")
    results = {'X': 0, 'O': 0, 'draw': 0}
    for _ in range(100):
        game = TicTacToe()
        while not game.done:
            action = optimal.get_action(game)
            game.step(action)
        if game.winner == 1:
            results['X'] += 1
        elif game.winner == -1:
            results['O'] += 1
        else:
            results['draw'] += 1

    print(f"  Results: X={results['X']}, O={results['O']}, Draw={results['draw']}")
    assert results['draw'] == 100, "Optimal vs optimal should always draw!"

    # Test: optimal vs random - optimal should never lose
    print("Testing optimal vs random (100 games as X, 100 as O)...")
    losses = 0
    for i in range(200):
        game = TicTacToe()
        optimal_is_x = (i < 100)
        optimal_sign = 1 if optimal_is_x else -1

        while not game.done:
            if game.current_player == optimal_sign:
                action = optimal.get_action(game)
            else:
                action = random_player.get_action(game)
            game.step(action)

        if game.winner == -optimal_sign:
            losses += 1
            print(f"  ERROR: Optimal lost game {i}!")

    print(f"  Optimal losses: {losses} (should be 0)")
    assert losses == 0, "Optimal should never lose!"

    print("All tests passed!")


if __name__ == "__main__":
    verify_optimal_player()
