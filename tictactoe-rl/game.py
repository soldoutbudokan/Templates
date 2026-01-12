"""
Tic Tac Toe Game Environment
"""
import numpy as np
from typing import Tuple, List, Optional


class TicTacToe:
    """
    Tic Tac Toe game environment.

    Board representation:
    - 0: Empty
    - 1: Player X
    - -1: Player O
    """

    def __init__(self):
        self.reset()

    def reset(self) -> np.ndarray:
        """Reset the game and return initial state."""
        self.board = np.zeros(9, dtype=np.int8)
        self.current_player = 1  # X starts
        self.done = False
        self.winner = None
        return self.board.copy()

    def get_state(self) -> Tuple[int, ...]:
        """Return current state as a hashable tuple."""
        return tuple(self.board)

    def get_valid_actions(self) -> List[int]:
        """Return list of valid actions (empty positions)."""
        return [i for i in range(9) if self.board[i] == 0]

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Take an action (place a piece).

        Returns:
            state: New board state
            reward: Reward for this action
            done: Whether game is over
            info: Additional info dict
        """
        if self.done:
            raise ValueError("Game is already over")

        if self.board[action] != 0:
            raise ValueError(f"Invalid action: position {action} is occupied")

        # Place piece
        self.board[action] = self.current_player

        # Check for winner
        winner = self._check_winner()

        if winner is not None:
            self.done = True
            self.winner = winner
            reward = 1.0 if winner == self.current_player else -1.0
        elif len(self.get_valid_actions()) == 0:
            # Draw
            self.done = True
            self.winner = 0
            reward = 0.0
        else:
            reward = 0.0

        # Switch player
        self.current_player *= -1

        return self.board.copy(), reward, self.done, {"winner": self.winner}

    def _check_winner(self) -> Optional[int]:
        """Check if there's a winner. Returns winner (1, -1) or None."""
        # Winning lines: rows, columns, diagonals
        lines = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # rows
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # columns
            [0, 4, 8], [2, 4, 6]               # diagonals
        ]

        for line in lines:
            total = sum(self.board[i] for i in line)
            if total == 3:
                return 1
            elif total == -3:
                return -1

        return None

    def render(self) -> str:
        """Return string representation of the board."""
        symbols = {0: '.', 1: 'X', -1: 'O'}
        rows = []
        for i in range(3):
            row = ' '.join(symbols[self.board[i*3 + j]] for j in range(3))
            rows.append(row)
        return '\n'.join(rows)

    def clone(self) -> 'TicTacToe':
        """Create a copy of the current game state."""
        new_game = TicTacToe()
        new_game.board = self.board.copy()
        new_game.current_player = self.current_player
        new_game.done = self.done
        new_game.winner = self.winner
        return new_game


if __name__ == "__main__":
    # Quick test
    game = TicTacToe()
    print("Initial board:")
    print(game.render())
    print(f"\nValid actions: {game.get_valid_actions()}")
