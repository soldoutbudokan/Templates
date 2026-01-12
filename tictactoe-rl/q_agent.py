"""
Tabula Rasa Q-Learning Agent for Tic Tac Toe.
Starts with no knowledge and learns purely from experience.
"""
import numpy as np
import random
from typing import Dict, Tuple, List, Optional
from collections import defaultdict
from game import TicTacToe


class QLearningAgent:
    """
    Q-Learning agent that learns from scratch (tabula rasa).

    The agent maintains a Q-table mapping (state, action) pairs to expected values.
    It uses epsilon-greedy exploration and learns via temporal difference updates.
    """

    def __init__(
        self,
        learning_rate: float = 0.1,
        discount_factor: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.9995,
        epsilon_min: float = 0.01
    ):
        """
        Initialize Q-Learning agent.

        Args:
            learning_rate: Alpha - how much to update Q-values
            discount_factor: Gamma - importance of future rewards
            epsilon: Initial exploration rate
            epsilon_decay: Decay rate for epsilon after each episode
            epsilon_min: Minimum exploration rate
        """
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Q-table: maps (state_tuple, action) -> Q-value
        # Starts empty (tabula rasa)
        self.q_table: Dict[Tuple[Tuple[int, ...], int], float] = defaultdict(float)

        # Track which player we're playing as for this episode
        self.player_sign = 1

    def get_q_value(self, state: Tuple[int, ...], action: int) -> float:
        """Get Q-value for state-action pair."""
        return self.q_table[(state, action)]

    def get_max_q_value(self, game: TicTacToe) -> float:
        """Get maximum Q-value for current state."""
        state = game.get_state()
        valid_actions = game.get_valid_actions()

        if not valid_actions:
            return 0.0

        return max(self.get_q_value(state, a) for a in valid_actions)

    def get_action(self, game: TicTacToe, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.

        Args:
            game: Current game state
            training: If True, use exploration. If False, exploit only.

        Returns:
            Selected action
        """
        valid_actions = game.get_valid_actions()

        if not valid_actions:
            raise ValueError("No valid actions available")

        # Exploration vs exploitation
        if training and random.random() < self.epsilon:
            return random.choice(valid_actions)

        # Greedy selection
        state = game.get_state()
        q_values = [(a, self.get_q_value(state, a)) for a in valid_actions]

        # Find all actions with max Q-value (for tie-breaking)
        max_q = max(q for _, q in q_values)
        best_actions = [a for a, q in q_values if q == max_q]

        return random.choice(best_actions)

    def update(
        self,
        state: Tuple[int, ...],
        action: int,
        reward: float,
        next_state: Tuple[int, ...],
        next_valid_actions: List[int],
        done: bool
    ):
        """
        Update Q-value using TD learning.

        Q(s,a) <- Q(s,a) + lr * [r + gamma * max_a' Q(s',a') - Q(s,a)]
        """
        current_q = self.get_q_value(state, action)

        if done:
            target = reward
        else:
            # Max Q-value for next state
            if next_valid_actions:
                next_max_q = max(
                    self.get_q_value(next_state, a) for a in next_valid_actions
                )
            else:
                next_max_q = 0.0
            target = reward + self.gamma * next_max_q

        # TD update
        new_q = current_q + self.lr * (target - current_q)
        self.q_table[(state, action)] = new_q

    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_stats(self) -> Dict:
        """Get agent statistics."""
        return {
            "epsilon": self.epsilon,
            "q_table_size": len(self.q_table),
            "avg_q_value": np.mean(list(self.q_table.values())) if self.q_table else 0.0,
            "max_q_value": max(self.q_table.values()) if self.q_table else 0.0,
            "min_q_value": min(self.q_table.values()) if self.q_table else 0.0,
        }

    def save(self, filepath: str):
        """Save Q-table to file."""
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(dict(self.q_table), f)

    def load(self, filepath: str):
        """Load Q-table from file."""
        import pickle
        with open(filepath, 'rb') as f:
            self.q_table = defaultdict(float, pickle.load(f))


if __name__ == "__main__":
    # Quick test
    agent = QLearningAgent()
    game = TicTacToe()

    print("Testing Q-learning agent:")
    print(f"Initial stats: {agent.get_stats()}")

    # Make some moves
    state = game.get_state()
    action = agent.get_action(game)
    print(f"Selected action: {action}")
    print(f"Q-value before update: {agent.get_q_value(state, action)}")

    game.step(action)
    agent.update(state, action, 0.0, game.get_state(), game.get_valid_actions(), False)
    print(f"Q-value after update: {agent.get_q_value(state, action)}")
