"""
Visualization module for RL training metrics.
Creates charts showing agent performance and distance from optimal play.
"""
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional
from trainer import TrainingMetrics


def plot_all_metrics(metrics: TrainingMetrics, save_path: Optional[str] = None):
    """
    Create a comprehensive visualization of all training metrics.

    Args:
        metrics: TrainingMetrics object from training
        save_path: If provided, save figure to this path
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Tic Tac Toe Q-Learning Agent: Training Progress', fontsize=14, fontweight='bold')

    # 1. Performance vs Optimal Player
    ax1 = axes[0, 0]
    ax1.stackplot(
        metrics.episodes,
        [np.array(metrics.win_rate_vs_optimal) * 100,
         np.array(metrics.draw_rate_vs_optimal) * 100,
         np.array(metrics.loss_rate_vs_optimal) * 100],
        labels=['Win', 'Draw', 'Loss'],
        colors=['#2ecc71', '#f39c12', '#e74c3c'],
        alpha=0.8
    )
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Percentage')
    ax1.set_title('Performance vs Optimal Player')
    ax1.legend(loc='upper right')
    ax1.set_ylim(0, 100)
    ax1.grid(True, alpha=0.3)

    # Add annotation for optimal benchmark
    ax1.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='Draw baseline')
    ax1.text(metrics.episodes[-1] * 0.02, 52, 'Optimal: 100% draws',
             fontsize=8, color='gray', style='italic')

    # 2. Performance vs Random Player
    ax2 = axes[0, 1]
    ax2.plot(metrics.episodes, np.array(metrics.win_rate_vs_random) * 100,
             label='Win Rate', color='#2ecc71', linewidth=2)
    ax2.plot(metrics.episodes, np.array(metrics.draw_rate_vs_random) * 100,
             label='Draw Rate', color='#f39c12', linewidth=2)
    loss_vs_random = 1 - np.array(metrics.win_rate_vs_random) - np.array(metrics.draw_rate_vs_random)
    ax2.plot(metrics.episodes, loss_vs_random * 100,
             label='Loss Rate', color='#e74c3c', linewidth=2)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Percentage')
    ax2.set_title('Performance vs Random Player')
    ax2.legend(loc='center right')
    ax2.set_ylim(0, 100)
    ax2.grid(True, alpha=0.3)

    # 3. Optimality Gap (Distance from Optimal)
    ax3 = axes[0, 2]
    ax3.plot(metrics.episodes, metrics.optimality_gap,
             color='#9b59b6', linewidth=2, marker='o', markersize=4)
    ax3.fill_between(metrics.episodes, 0, metrics.optimality_gap,
                     color='#9b59b6', alpha=0.3)
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Optimality Gap')
    ax3.set_title('Distance from Optimal Policy')
    ax3.axhline(y=0, color='green', linestyle='--', alpha=0.7, label='Optimal (0)')
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # Add annotation
    final_gap = metrics.optimality_gap[-1]
    ax3.annotate(f'Final: {final_gap:.3f}',
                xy=(metrics.episodes[-1], final_gap),
                xytext=(metrics.episodes[-1] * 0.7, final_gap + 0.1),
                arrowprops=dict(arrowstyle='->', color='gray'),
                fontsize=9)

    # 4. Action Agreement with Optimal
    ax4 = axes[1, 0]
    ax4.plot(metrics.episodes, np.array(metrics.action_agreement_rate) * 100,
             color='#3498db', linewidth=2, marker='s', markersize=4)
    ax4.fill_between(metrics.episodes, 0, np.array(metrics.action_agreement_rate) * 100,
                     color='#3498db', alpha=0.3)
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Agreement Rate (%)')
    ax4.set_title('Action Agreement with Optimal Player')
    ax4.axhline(y=100, color='green', linestyle='--', alpha=0.7, label='Perfect (100%)')
    ax4.set_ylim(0, 105)
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    # 5. Exploration Rate (Epsilon)
    ax5 = axes[1, 1]
    ax5.plot(metrics.episodes, metrics.epsilon_values,
             color='#e67e22', linewidth=2)
    ax5.set_xlabel('Episode')
    ax5.set_ylabel('Epsilon')
    ax5.set_title('Exploration Rate Decay')
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim(0, 1.05)

    # 6. Q-Table Size (States Learned)
    ax6 = axes[1, 2]
    ax6.plot(metrics.episodes, metrics.q_table_sizes,
             color='#1abc9c', linewidth=2, marker='^', markersize=4)
    ax6.set_xlabel('Episode')
    ax6.set_ylabel('Q-Table Entries')
    ax6.set_title('States Explored (Q-Table Size)')
    ax6.grid(True, alpha=0.3)

    # Add theoretical max annotation
    # Tic-tac-toe has ~5478 legal game states
    ax6.axhline(y=5478, color='gray', linestyle='--', alpha=0.5)
    ax6.text(metrics.episodes[0], 5600, 'Theoretical max states',
             fontsize=8, color='gray', style='italic')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    return fig


def plot_optimality_comparison(metrics: TrainingMetrics, save_path: Optional[str] = None):
    """
    Create a focused visualization showing distance from optimal play.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('How Far is the Agent from Optimal Play?', fontsize=14, fontweight='bold')

    # Left: Combined optimality metrics
    ax1 = axes[0]

    # Normalize metrics to 0-1 scale where 1 is optimal
    optimality_score = 1 - np.array(metrics.optimality_gap)
    optimality_score = np.clip(optimality_score, 0, 1)  # Clip to valid range

    ax1.plot(metrics.episodes, optimality_score * 100,
             label='Value Accuracy', color='#9b59b6', linewidth=2)
    ax1.plot(metrics.episodes, np.array(metrics.action_agreement_rate) * 100,
             label='Action Agreement', color='#3498db', linewidth=2)
    ax1.plot(metrics.episodes, np.array(metrics.draw_rate_vs_optimal) * 100,
             label='Draw Rate vs Optimal', color='#2ecc71', linewidth=2)

    ax1.axhline(y=100, color='green', linestyle='--', alpha=0.5, linewidth=1)
    ax1.fill_between(metrics.episodes, 100, np.array(metrics.draw_rate_vs_optimal) * 100,
                     color='red', alpha=0.1, label='Gap from optimal')

    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Percentage (%)')
    ax1.set_title('Convergence to Optimal Policy')
    ax1.legend(loc='lower right')
    ax1.set_ylim(0, 105)
    ax1.grid(True, alpha=0.3)

    # Right: Loss rate decay (should go to 0 against optimal)
    ax2 = axes[1]
    loss_rate = np.array(metrics.loss_rate_vs_optimal) * 100

    ax2.plot(metrics.episodes, loss_rate, color='#e74c3c', linewidth=2, marker='o', markersize=4)
    ax2.fill_between(metrics.episodes, 0, loss_rate, color='#e74c3c', alpha=0.3)

    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Loss Rate (%)')
    ax2.set_title('Loss Rate vs Optimal (Should Approach 0%)')
    ax2.axhline(y=0, color='green', linestyle='--', alpha=0.7, label='Optimal (0% losses)')
    ax2.set_ylim(0, max(loss_rate) * 1.1 + 5)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    return fig


def plot_learning_summary(metrics: TrainingMetrics, save_path: Optional[str] = None):
    """
    Create a single summary chart with key metrics.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create composite "distance from optimal" score
    # 0 = random/bad, 100 = optimal
    loss_penalty = np.array(metrics.loss_rate_vs_optimal) * 100
    draw_score = np.array(metrics.draw_rate_vs_optimal) * 100
    agreement_score = np.array(metrics.action_agreement_rate) * 100

    # Weighted composite score
    composite = (draw_score * 0.4 + agreement_score * 0.4 + (100 - loss_penalty) * 0.2)

    ax.plot(metrics.episodes, composite, color='#2c3e50', linewidth=3, label='Overall Optimality Score')
    ax.fill_between(metrics.episodes, 0, composite, color='#3498db', alpha=0.3)

    # Add benchmark lines
    ax.axhline(y=100, color='green', linestyle='--', alpha=0.7, linewidth=2)
    ax.text(metrics.episodes[-1] * 0.02, 102, 'Optimal Play', fontsize=10, color='green')

    ax.axhline(y=33, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax.text(metrics.episodes[-1] * 0.02, 35, 'Random Play', fontsize=10, color='red')

    # Annotations
    initial_score = composite[0]
    final_score = composite[-1]

    ax.annotate(f'Start: {initial_score:.1f}%',
               xy=(metrics.episodes[0], initial_score),
               xytext=(metrics.episodes[0] + metrics.episodes[-1] * 0.1, initial_score - 10),
               arrowprops=dict(arrowstyle='->', color='gray'),
               fontsize=10, fontweight='bold')

    ax.annotate(f'Final: {final_score:.1f}%',
               xy=(metrics.episodes[-1], final_score),
               xytext=(metrics.episodes[-1] * 0.75, final_score + 8),
               arrowprops=dict(arrowstyle='->', color='gray'),
               fontsize=10, fontweight='bold')

    ax.set_xlabel('Training Episode', fontsize=12)
    ax.set_ylabel('Optimality Score (%)', fontsize=12)
    ax.set_title('Tabula Rasa RL Agent: Journey to Optimal Play', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 110)
    ax.set_xlim(0, metrics.episodes[-1] * 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    return fig


if __name__ == "__main__":
    # Test with dummy data
    metrics = TrainingMetrics()
    for i in range(1, 11):
        metrics.episodes.append(i * 1000)
        metrics.win_rate_vs_optimal.append(0.05 + i * 0.02)
        metrics.draw_rate_vs_optimal.append(0.1 + i * 0.08)
        metrics.loss_rate_vs_optimal.append(0.85 - i * 0.1)
        metrics.win_rate_vs_random.append(0.3 + i * 0.06)
        metrics.draw_rate_vs_random.append(0.2 + i * 0.03)
        metrics.optimality_gap.append(0.8 - i * 0.07)
        metrics.action_agreement_rate.append(0.2 + i * 0.07)
        metrics.epsilon_values.append(1.0 * (0.9 ** i))
        metrics.q_table_sizes.append(100 + i * 400)
        metrics.avg_rewards.append(-0.3 + i * 0.05)

    print("Creating test visualizations...")
    plot_all_metrics(metrics, "test_all_metrics.png")
    plot_optimality_comparison(metrics, "test_optimality.png")
    plot_learning_summary(metrics, "test_summary.png")
    print("Done!")
