"""
Training loop for the Q-Learning agent with evaluation against optimal player.
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from game import TicTacToe
from q_agent import QLearningAgent
from optimal_player import OptimalPlayer, RandomPlayer


@dataclass
class TrainingMetrics:
    """Container for training metrics over time."""
    episodes: List[int] = field(default_factory=list)

    # Win rates against different opponents
    win_rate_vs_optimal: List[float] = field(default_factory=list)
    draw_rate_vs_optimal: List[float] = field(default_factory=list)
    loss_rate_vs_optimal: List[float] = field(default_factory=list)

    win_rate_vs_random: List[float] = field(default_factory=list)
    draw_rate_vs_random: List[float] = field(default_factory=list)

    # Distance from optimal metrics
    optimality_gap: List[float] = field(default_factory=list)
    action_agreement_rate: List[float] = field(default_factory=list)

    # Agent stats
    epsilon_values: List[float] = field(default_factory=list)
    q_table_sizes: List[int] = field(default_factory=list)

    # Training rewards
    avg_rewards: List[float] = field(default_factory=list)


def play_episode(
    agent: QLearningAgent,
    opponent,
    agent_plays_first: bool = True,
    training: bool = True
) -> Tuple[int, List[Tuple]]:
    """
    Play a single episode.

    Args:
        agent: The Q-learning agent
        opponent: The opponent player
        agent_plays_first: Whether agent plays X (first)
        training: Whether to update agent's Q-values

    Returns:
        (result, experience) where result is 1 (win), 0 (draw), -1 (loss)
        and experience is list of (state, action, reward, next_state, done)
    """
    game = TicTacToe()
    agent_sign = 1 if agent_plays_first else -1
    experience = []
    total_reward = 0

    while not game.done:
        is_agent_turn = (game.current_player == agent_sign)

        if is_agent_turn:
            state = game.get_state()
            action = agent.get_action(game, training=training)
            _, reward, done, info = game.step(action)

            # Adjust reward from agent's perspective
            if done:
                if info['winner'] == agent_sign:
                    reward = 1.0
                elif info['winner'] == -agent_sign:
                    reward = -1.0
                else:
                    reward = 0.0

            next_state = game.get_state()
            next_valid = game.get_valid_actions()

            experience.append((state, action, reward, next_state, next_valid, done))
            total_reward += reward

        else:
            # Opponent's turn
            action = opponent.get_action(game)
            _, _, done, info = game.step(action)

            # If opponent just won, update last experience with negative reward
            if done and experience and info['winner'] == -agent_sign:
                last = experience[-1]
                experience[-1] = (last[0], last[1], -1.0, game.get_state(), [], True)
                total_reward -= 1.0

    # Determine result
    if game.winner == agent_sign:
        result = 1
    elif game.winner == -agent_sign:
        result = -1
    else:
        result = 0

    return result, experience, total_reward


def train_agent(
    agent: QLearningAgent,
    num_episodes: int = 50000,
    eval_interval: int = 1000,
    eval_games: int = 100,
    verbose: bool = True
) -> TrainingMetrics:
    """
    Train the Q-learning agent.

    Args:
        agent: The agent to train
        num_episodes: Total episodes to train
        eval_interval: How often to evaluate
        eval_games: Number of games per evaluation
        verbose: Print progress

    Returns:
        TrainingMetrics with learning curves
    """
    metrics = TrainingMetrics()
    optimal_player = OptimalPlayer()
    random_player = RandomPlayer()

    episode_rewards = []

    for episode in range(num_episodes):
        # Alternate between playing as X and O
        agent_plays_first = (episode % 2 == 0)

        # Self-play or vs random for exploration during training
        if episode % 3 == 0:
            opponent = random_player
        else:
            # Self-play using a copy of current policy
            opponent = agent

        result, experience, total_reward = play_episode(
            agent, opponent, agent_plays_first, training=True
        )

        # Update Q-values from experience
        for state, action, reward, next_state, next_valid, done in experience:
            agent.update(state, action, reward, next_state, next_valid, done)

        episode_rewards.append(total_reward)
        agent.decay_epsilon()

        # Evaluation
        if (episode + 1) % eval_interval == 0:
            eval_results = evaluate_agent(agent, optimal_player, random_player, eval_games)

            metrics.episodes.append(episode + 1)
            metrics.win_rate_vs_optimal.append(eval_results['vs_optimal']['win'])
            metrics.draw_rate_vs_optimal.append(eval_results['vs_optimal']['draw'])
            metrics.loss_rate_vs_optimal.append(eval_results['vs_optimal']['loss'])
            metrics.win_rate_vs_random.append(eval_results['vs_random']['win'])
            metrics.draw_rate_vs_random.append(eval_results['vs_random']['draw'])
            metrics.optimality_gap.append(eval_results['optimality_gap'])
            metrics.action_agreement_rate.append(eval_results['action_agreement'])
            metrics.epsilon_values.append(agent.epsilon)
            metrics.q_table_sizes.append(len(agent.q_table))
            metrics.avg_rewards.append(np.mean(episode_rewards[-eval_interval:]))

            if verbose:
                print(f"Episode {episode + 1:6d} | "
                      f"vs Optimal: W{eval_results['vs_optimal']['win']:.1%} "
                      f"D{eval_results['vs_optimal']['draw']:.1%} "
                      f"L{eval_results['vs_optimal']['loss']:.1%} | "
                      f"Gap: {eval_results['optimality_gap']:.3f} | "
                      f"Agreement: {eval_results['action_agreement']:.1%} | "
                      f"eps: {agent.epsilon:.3f}")

    return metrics


def evaluate_agent(
    agent: QLearningAgent,
    optimal_player: OptimalPlayer,
    random_player: RandomPlayer,
    num_games: int = 100
) -> Dict:
    """
    Evaluate agent performance.

    Returns dict with:
    - vs_optimal: win/draw/loss rates against optimal
    - vs_random: win/draw/loss rates against random
    - optimality_gap: how far agent's decisions are from optimal
    - action_agreement: how often agent agrees with optimal
    """
    results = {
        'vs_optimal': {'win': 0, 'draw': 0, 'loss': 0},
        'vs_random': {'win': 0, 'draw': 0, 'loss': 0},
    }

    # Play against optimal
    for i in range(num_games):
        result, _, _ = play_episode(agent, optimal_player, i % 2 == 0, training=False)
        if result == 1:
            results['vs_optimal']['win'] += 1
        elif result == 0:
            results['vs_optimal']['draw'] += 1
        else:
            results['vs_optimal']['loss'] += 1

    # Play against random
    for i in range(num_games):
        result, _, _ = play_episode(agent, random_player, i % 2 == 0, training=False)
        if result == 1:
            results['vs_random']['win'] += 1
        elif result == 0:
            results['vs_random']['draw'] += 1
        else:
            results['vs_random']['loss'] += 1

    # Normalize
    for key in results:
        for outcome in results[key]:
            results[key][outcome] /= num_games

    # Measure optimality gap and action agreement
    optimality_gap, action_agreement = measure_optimality(agent, optimal_player)
    results['optimality_gap'] = optimality_gap
    results['action_agreement'] = action_agreement

    return results


def measure_optimality(
    agent: QLearningAgent,
    optimal_player: OptimalPlayer,
    num_samples: int = 200
) -> Tuple[float, float]:
    """
    Measure how far agent's policy is from optimal.

    Returns:
        (optimality_gap, action_agreement_rate)
        - optimality_gap: average difference between agent's action value and optimal
        - action_agreement: fraction of states where agent picks an optimal action
    """
    game = TicTacToe()
    gaps = []
    agreements = 0
    total_states = 0

    # Sample random game states
    for _ in range(num_samples):
        game.reset()

        # Play some random moves to get varied states
        num_moves = np.random.randint(0, 6)
        for _ in range(num_moves):
            if game.done:
                break
            action = np.random.choice(game.get_valid_actions())
            game.step(action)

        if game.done:
            continue

        # Compare agent's choice to optimal
        agent_action = agent.get_action(game, training=False)

        # Get optimal value for comparison
        optimal_value = optimal_player.get_optimal_value(game)

        # Get agent's Q-value for its chosen action
        state = game.get_state()
        agent_q = agent.get_q_value(state, agent_action)

        # Gap is difference between optimal and agent's estimate
        gap = abs(optimal_value - agent_q)
        gaps.append(gap)

        # Check if agent's action is among optimal actions
        optimal_actions = optimal_player.get_all_optimal_actions(game)
        if agent_action in optimal_actions:
            agreements += 1

        total_states += 1

    if total_states == 0:
        return 0.0, 0.0

    return np.mean(gaps), agreements / total_states


if __name__ == "__main__":
    print("Quick training test...")
    agent = QLearningAgent()
    metrics = train_agent(agent, num_episodes=5000, eval_interval=1000)
    print("\nTraining complete!")
    print(f"Final stats: {agent.get_stats()}")
