#!/usr/bin/env python3
"""
Tabula Rasa Reinforcement Learning Agent for Tic Tac Toe

This script trains a Q-learning agent from scratch (no prior knowledge)
and visualizes how far it is from optimal play at each stage of training.

Usage:
    python main.py [--episodes N] [--output DIR]
"""
import argparse
import os
from datetime import datetime

from game import TicTacToe
from q_agent import QLearningAgent
from optimal_player import OptimalPlayer, RandomPlayer
from trainer import train_agent, evaluate_agent, play_episode
from visualize import plot_all_metrics, plot_optimality_comparison, plot_learning_summary


def main():
    parser = argparse.ArgumentParser(
        description='Train a tabula rasa RL agent to play Tic Tac Toe'
    )
    parser.add_argument(
        '--episodes', type=int, default=50000,
        help='Number of training episodes (default: 50000)'
    )
    parser.add_argument(
        '--eval-interval', type=int, default=1000,
        help='Evaluate every N episodes (default: 1000)'
    )
    parser.add_argument(
        '--output', type=str, default='results',
        help='Output directory for charts (default: results)'
    )
    parser.add_argument(
        '--lr', type=float, default=0.1,
        help='Learning rate (default: 0.1)'
    )
    parser.add_argument(
        '--gamma', type=float, default=0.99,
        help='Discount factor (default: 0.99)'
    )
    parser.add_argument(
        '--epsilon-decay', type=float, default=0.9995,
        help='Epsilon decay rate (default: 0.9995)'
    )
    parser.add_argument(
        '--demo', action='store_true',
        help='Run a demo game after training'
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    print("=" * 60)
    print("TABULA RASA RL AGENT FOR TIC TAC TOE")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Episodes:       {args.episodes:,}")
    print(f"  Eval interval:  {args.eval_interval:,}")
    print(f"  Learning rate:  {args.lr}")
    print(f"  Discount:       {args.gamma}")
    print(f"  Epsilon decay:  {args.epsilon_decay}")
    print(f"  Output dir:     {args.output}")
    print()

    # Create agent (starts with empty Q-table - tabula rasa)
    agent = QLearningAgent(
        learning_rate=args.lr,
        discount_factor=args.gamma,
        epsilon=1.0,
        epsilon_decay=args.epsilon_decay,
        epsilon_min=0.01
    )

    print("Initial agent stats (tabula rasa - knows nothing!):")
    print(f"  Q-table size: {len(agent.q_table)} entries")
    print()

    # Evaluate initial (random) performance
    print("Initial evaluation (untrained agent)...")
    optimal_player = OptimalPlayer()
    random_player = RandomPlayer()
    initial_eval = evaluate_agent(agent, optimal_player, random_player, num_games=100)
    print(f"  vs Optimal: Win {initial_eval['vs_optimal']['win']:.1%}, "
          f"Draw {initial_eval['vs_optimal']['draw']:.1%}, "
          f"Loss {initial_eval['vs_optimal']['loss']:.1%}")
    print(f"  vs Random:  Win {initial_eval['vs_random']['win']:.1%}, "
          f"Draw {initial_eval['vs_random']['draw']:.1%}")
    print(f"  Optimality gap: {initial_eval['optimality_gap']:.3f}")
    print(f"  Action agreement: {initial_eval['action_agreement']:.1%}")
    print()

    # Train the agent
    print("=" * 60)
    print("TRAINING")
    print("=" * 60)
    print()

    metrics = train_agent(
        agent,
        num_episodes=args.episodes,
        eval_interval=args.eval_interval,
        eval_games=100,
        verbose=True
    )

    print()
    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    # Final evaluation
    print("\nFinal agent stats:")
    stats = agent.get_stats()
    print(f"  Q-table size:    {stats['q_table_size']:,} entries")
    print(f"  Epsilon:         {stats['epsilon']:.4f}")
    print(f"  Avg Q-value:     {stats['avg_q_value']:.4f}")
    print(f"  Max Q-value:     {stats['max_q_value']:.4f}")
    print(f"  Min Q-value:     {stats['min_q_value']:.4f}")

    print("\nFinal performance:")
    final_eval = evaluate_agent(agent, optimal_player, random_player, num_games=500)
    print(f"  vs Optimal (500 games):")
    print(f"    Win:  {final_eval['vs_optimal']['win']:.1%}")
    print(f"    Draw: {final_eval['vs_optimal']['draw']:.1%}")
    print(f"    Loss: {final_eval['vs_optimal']['loss']:.1%}")
    print(f"  vs Random (500 games):")
    print(f"    Win:  {final_eval['vs_random']['win']:.1%}")
    print(f"    Draw: {final_eval['vs_random']['draw']:.1%}")
    print(f"  Optimality gap: {final_eval['optimality_gap']:.3f}")
    print(f"  Action agreement: {final_eval['action_agreement']:.1%}")

    # Note about optimal play
    print()
    print("Note: Against an optimal player, the best possible outcome is 100% draws.")
    print("Any wins against optimal indicate opponent mistakes (shouldn't happen).")
    print("Losses indicate the agent is not yet playing optimally.")

    # Generate visualizations
    print()
    print("=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)
    print()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # All metrics
    all_metrics_path = os.path.join(args.output, f"all_metrics_{timestamp}.png")
    plot_all_metrics(metrics, all_metrics_path)

    # Optimality comparison
    optimality_path = os.path.join(args.output, f"optimality_{timestamp}.png")
    plot_optimality_comparison(metrics, optimality_path)

    # Learning summary
    summary_path = os.path.join(args.output, f"summary_{timestamp}.png")
    plot_learning_summary(metrics, summary_path)

    # Save agent
    agent_path = os.path.join(args.output, f"agent_{timestamp}.pkl")
    agent.save(agent_path)
    print(f"Agent saved to {agent_path}")

    print()
    print(f"All outputs saved to: {args.output}/")

    # Demo game
    if args.demo:
        print()
        print("=" * 60)
        print("DEMO: TRAINED AGENT vs OPTIMAL PLAYER")
        print("=" * 60)
        demo_game(agent, optimal_player)

    return agent, metrics


def demo_game(agent: QLearningAgent, optimal_player: OptimalPlayer):
    """Play a demo game between trained agent and optimal player."""
    game = TicTacToe()
    agent_is_x = True  # Agent plays X

    print(f"\nAgent plays: X")
    print("Optimal plays: O")
    print()

    move_num = 0
    while not game.done:
        move_num += 1
        is_agent_turn = (game.current_player == 1) == agent_is_x

        if is_agent_turn:
            action = agent.get_action(game, training=False)
            player_name = "Agent (X)"
        else:
            action = optimal_player.get_action(game)
            player_name = "Optimal (O)"

        row, col = action // 3, action % 3
        print(f"Move {move_num}: {player_name} plays position {action} (row {row}, col {col})")

        game.step(action)
        print(game.render())
        print()

    if game.winner == 1:
        result = "Agent (X) WINS!"
    elif game.winner == -1:
        result = "Optimal (O) WINS!"
    else:
        result = "DRAW!"

    print(f"Result: {result}")


if __name__ == "__main__":
    agent, metrics = main()
