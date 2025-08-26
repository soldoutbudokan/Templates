"""
Enhanced Rikken RL Analysis with Strategy Extraction and Visualization
Creates comprehensive analysis of training results and playable strategies.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import json
import os
from datetime import datetime
import pandas as pd

# ==================== Enhanced Training with Analytics ====================

class RikkenAnalyzer:
    """Comprehensive analysis of Rikken training results."""
    
    def __init__(self, output_dir=None):
        if output_dir:
            self.output_dir = output_dir
            print(f"Using provided output directory: {self.output_dir}")
        else:
            try:
                script_dir = os.path.dirname(os.path.abspath(__file__))
                print(f"Using script directory: {script_dir}")
            except (NameError, TypeError):
                script_dir = os.getcwd()
                print(f"__file__ not available, using current directory: {script_dir}")
            self.output_dir = os.path.join(script_dir, "output")
            print(f"Output directory will be: {self.output_dir}")

        # final safeguard if something went wrong above
        if not self.output_dir:
            self.output_dir = os.path.join(os.getcwd(), "output")
            print(f"Emergency fallback to: {self.output_dir}")

        self.ensure_output_dir()
        
        # Training statistics
        self.game_history = []
        self.bidding_patterns = defaultdict(lambda: defaultdict(int))
        self.contract_outcomes = defaultdict(lambda: defaultdict(list))
        self.hand_strength_bids = defaultdict(list)
        self.partnership_success = defaultdict(list)
        self.trump_choices = defaultdict(int)
        
    def ensure_output_dir(self):
        """Create output directory structure."""
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(f"{self.output_dir}/graphs", exist_ok=True)
        os.makedirs(f"{self.output_dir}/strategies", exist_ok=True)
        os.makedirs(f"{self.output_dir}/logs", exist_ok=True)
    
    def record_game(self, env, rewards, episode_num):
        """Record comprehensive game data."""
        game_data = {
            'episode': episode_num,
            'contract': env.contract.type.name if env.contract else None,
            'declarer': env.contract.declarer if env.contract else None,
            'partner': env.contract.partner if env.contract else None,
            'trump': env.trump_suit.name if env.trump_suit else None,
            'tricks_won': env.tricks_won.copy(),
            'rewards': rewards.copy(),
            'success': rewards[env.contract.declarer] > 0 if env.contract else False,
            'total_tricks': sum(env.tricks_won),
            'bidding_rounds': len(env.bidding_history)
        }
        
        self.game_history.append(game_data)
        
        # Update detailed statistics
        if env.contract:
            declarer = env.contract.declarer
            contract_type = env.contract.type.name
            
            # Bidding patterns by player
            self.bidding_patterns[declarer][contract_type] += 1
            
            # Contract outcomes
            self.contract_outcomes[contract_type]['success'].append(game_data['success'])
            self.contract_outcomes[contract_type]['tricks'].append(env.tricks_won[declarer])
            
            # Trump choices
            if env.trump_suit:
                self.trump_choices[env.trump_suit.name] += 1
    
    def generate_comprehensive_analysis(self):
        """Generate complete analysis with visualizations."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print(f"Generating comprehensive analysis...")
        print(f"Output directory: {self.output_dir}")
        
        # 1. Basic Statistics
        self.create_basic_stats_report(timestamp)
        
        # 2. Visualizations
        self.create_performance_graphs(timestamp)
        self.create_contract_analysis_graphs(timestamp)
        self.create_strategy_heatmaps(timestamp)
        
        # 3. Strategy Extraction
        self.extract_player_strategies(timestamp)
        
        # 4. Playable Strategy Guide
        self.create_strategy_guide(timestamp)
        
        print(f"Analysis complete! Check {self.output_dir} for results.")
    
    def create_basic_stats_report(self, timestamp):
        """Create basic statistics report."""
        if not self.game_history:
            return
            
        report = {
            'summary': {
                'total_games': len(self.game_history),
                'games_with_contracts': sum(1 for g in self.game_history if g['contract']),
                'overall_success_rate': np.mean([g['success'] for g in self.game_history if g['contract']]),
            },
            'player_performance': {},
            'contract_analysis': {},
            'trump_preferences': dict(self.trump_choices)
        }
        
        # Player-specific analysis
        for player_id in range(4):
            player_games = [g for g in self.game_history if g['declarer'] == player_id]
            if player_games:
                report['player_performance'][f'player_{player_id}'] = {
                    'contracts_attempted': len(player_games),
                    'success_rate': np.mean([g['success'] for g in player_games]),
                    'avg_tricks': np.mean([g['tricks_won'][player_id] for g in self.game_history]),
                    'favorite_contracts': dict(Counter([g['contract'] for g in player_games]).most_common(3)),
                    'avg_reward': np.mean([g['rewards'][player_id] for g in self.game_history])
                }
        
        # Contract analysis
        for contract_type, outcomes in self.contract_outcomes.items():
            if outcomes['success']:
                report['contract_analysis'][contract_type] = {
                    'attempts': len(outcomes['success']),
                    'success_rate': np.mean(outcomes['success']),
                    'avg_tricks_when_attempted': np.mean(outcomes['tricks'])
                }
        
        # Save report
        with open(f"{self.output_dir}/analysis_report_{timestamp}.json", 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Create readable summary
        self.create_readable_summary(report, timestamp)
    
    def create_readable_summary(self, report, timestamp):
        """Create human-readable summary."""
        summary = f"""
RIKKEN AI TRAINING ANALYSIS REPORT
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
{'='*50}

OVERVIEW
--------
Total Games Played: {report['summary']['total_games']}
Games with Contracts: {report['summary']['games_with_contracts']}
Overall Success Rate: {report['summary']['overall_success_rate']:.1%}

PLAYER PERFORMANCE
------------------
"""
        
        for player, stats in report['player_performance'].items():
            summary += f"""
{player.upper()}:
  Contracts Attempted: {stats['contracts_attempted']}
  Success Rate: {stats['success_rate']:.1%}
  Average Tricks per Game: {stats['avg_tricks']:.1f}
  Average Reward: {stats['avg_reward']:.1f}
  Favorite Contracts: {list(stats['favorite_contracts'].keys())[:2]}
"""
        
        summary += f"""
CONTRACT ANALYSIS
-----------------
"""
        for contract, stats in report['contract_analysis'].items():
            summary += f"""
{contract}:
  Attempts: {stats['attempts']}
  Success Rate: {stats['success_rate']:.1%}
  Avg Tricks: {stats['avg_tricks_when_attempted']:.1f}
"""
        
        summary += f"""
TRUMP PREFERENCES
-----------------
"""
        total_trump_games = sum(report['trump_preferences'].values())
        for trump, count in sorted(report['trump_preferences'].items(), key=lambda x: x[1], reverse=True):
            percentage = count / max(total_trump_games, 1) * 100
            summary += f"{trump}: {count} games ({percentage:.1f}%)\n"
        
        with open(f"{self.output_dir}/summary_{timestamp}.txt", 'w') as f:
            f.write(summary)
        
        print(summary)
    
    def create_performance_graphs(self, timestamp):
        """Create performance visualization graphs."""
        if not self.game_history:
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Win rate over time (rolling average)
        episodes = [g['episode'] for g in self.game_history]
        window_size = min(100, len(episodes) // 10)
        
        for player_id in range(4):
            wins = [(1 if g['rewards'][player_id] > 0 else 0) for g in self.game_history]
            if len(wins) >= window_size:
                rolling_wins = pd.Series(wins).rolling(window_size).mean()
                axes[0,0].plot(episodes, rolling_wins, label=f'Player {player_id}', alpha=0.8)
        
        axes[0,0].set_title('Win Rate Over Time (Rolling Average)')
        axes[0,0].set_xlabel('Episode')
        axes[0,0].set_ylabel('Win Rate')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. Contract success rates by player
        players = []
        success_rates = []
        contract_counts = []
        
        for player_id in range(4):
            player_games = [g for g in self.game_history if g['declarer'] == player_id]
            if player_games:
                players.append(f'Player {player_id}')
                success_rates.append(np.mean([g['success'] for g in player_games]))
                contract_counts.append(len(player_games))
        
        bars = axes[0,1].bar(players, success_rates, alpha=0.7)
        axes[0,1].set_title('Contract Success Rate by Player')
        axes[0,1].set_ylabel('Success Rate')
        axes[0,1].set_ylim(0, 1)
        
        # Add count labels on bars
        for bar, count in zip(bars, contract_counts):
            height = bar.get_height()
            axes[0,1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                          f'n={count}', ha='center', va='bottom', fontsize=8)
        
        # 3. Tricks distribution
        all_tricks = [g['tricks_won'] for g in self.game_history]
        tricks_by_player = list(zip(*all_tricks))
        
        axes[1,0].boxplot(tricks_by_player, labels=[f'P{i}' for i in range(4)])
        axes[1,0].set_title('Tricks Won Distribution by Player')
        axes[1,0].set_ylabel('Tricks Won')
        axes[1,0].grid(True, alpha=0.3)
        
        # 4. Contract type frequency
        contracts = [g['contract'] for g in self.game_history if g['contract']]
        contract_counts = Counter(contracts)
        
        if contract_counts:
            contracts, counts = zip(*contract_counts.most_common(10))
            axes[1,1].bar(range(len(contracts)), counts, alpha=0.7)
            axes[1,1].set_title('Most Common Contract Types')
            axes[1,1].set_ylabel('Frequency')
            axes[1,1].set_xticks(range(len(contracts)))
            axes[1,1].set_xticklabels(contracts, rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/graphs/performance_analysis_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_contract_analysis_graphs(self, timestamp):
        """Create detailed contract analysis graphs."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Contract success rate heatmap by player
        contract_success_matrix = []
        contract_names = []
        
        all_contracts = set()
        for player_id in range(4):
            player_contracts = set(g['contract'] for g in self.game_history if g['declarer'] == player_id and g['contract'])
            all_contracts.update(player_contracts)
        
        contract_names = sorted(list(all_contracts))
        
        for player_id in range(4):
            player_row = []
            for contract in contract_names:
                player_contract_games = [g for g in self.game_history 
                                       if g['declarer'] == player_id and g['contract'] == contract]
                if player_contract_games:
                    success_rate = np.mean([g['success'] for g in player_contract_games])
                    player_row.append(success_rate)
                else:
                    player_row.append(np.nan)
            contract_success_matrix.append(player_row)
        
        if contract_success_matrix and contract_names:
            sns.heatmap(contract_success_matrix, 
                       xticklabels=contract_names,
                       yticklabels=[f'Player {i}' for i in range(4)],
                       annot=True, fmt='.2f', cmap='RdYlGn', 
                       ax=axes[0,0], cbar_kws={'label': 'Success Rate'})
            axes[0,0].set_title('Contract Success Rate by Player')
            axes[0,0].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/graphs/contract_analysis_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_strategy_heatmaps(self, timestamp):
        """Create strategy heatmaps showing player preferences."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Trump preference by player
        trump_matrix = []
        trumps = ['CLUBS', 'DIAMONDS', 'HEARTS', 'SPADES']
        
        for player_id in range(4):
            player_trumps = []
            player_trump_games = [g for g in self.game_history if g['declarer'] == player_id and g['trump']]
            total_trump_games = len(player_trump_games)
            
            for trump in trumps:
                trump_count = sum(1 for g in player_trump_games if g['trump'] == trump)
                percentage = trump_count / max(total_trump_games, 1)
                player_trumps.append(percentage)
            trump_matrix.append(player_trumps)
        
        sns.heatmap(trump_matrix,
                   xticklabels=trumps,
                   yticklabels=[f'Player {i}' for i in range(4)],
                   annot=True, fmt='.2f', cmap='Blues',
                   ax=axes[0], cbar_kws={'label': 'Preference'})
        axes[0].set_title('Trump Suit Preferences by Player')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/graphs/strategy_heatmaps_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def extract_player_strategies(self, timestamp):
        """Extract interpretable strategies for each player."""
        strategies = {}
        
        for player_id in range(4):
            player_games = [g for g in self.game_history if g['declarer'] == player_id]
            
            if not player_games:
                continue
            
            strategy = {
                'bidding_style': self._analyze_bidding_style(player_id),
                'risk_tolerance': self._analyze_risk_tolerance(player_id),
                'trump_preferences': self._analyze_trump_preferences(player_id),
                'performance_metrics': self._analyze_performance_metrics(player_id)
            }
            
            strategies[f'player_{player_id}'] = strategy
        
        # Save strategies
        with open(f"{self.output_dir}/strategies/extracted_strategies_{timestamp}.json", 'w') as f:
            json.dump(strategies, f, indent=2, default=str)
        
        return strategies
    
    def _analyze_bidding_style(self, player_id):
        """Analyze player's bidding style."""
        player_games = [g for g in self.game_history if g['declarer'] == player_id]
        contracts = [g['contract'] for g in player_games if g['contract']]
        
        if not contracts:
            return "No contracts attempted"
        
        contract_counts = Counter(contracts)
        most_common = contract_counts.most_common(3)
        
        # Classify bidding style
        aggressive_contracts = ['SOLO_9', 'SOLO_10', 'SOLO_11', 'SOLO_12', 'SOLO_13']
        conservative_contracts = ['RIK', 'RIK_BETER', 'SOLO_8']
        risky_contracts = ['MISERE', 'PIEK', 'OPEN_MISERE', 'OPEN_PIEK']
        
        aggressive_count = sum(contract_counts[c] for c in aggressive_contracts)
        conservative_count = sum(contract_counts[c] for c in conservative_contracts)
        risky_count = sum(contract_counts[c] for c in risky_contracts)
        
        total = len(contracts)
        if conservative_count / total > 0.6:
            style = "Conservative"
        elif aggressive_count / total > 0.3:
            style = "Aggressive"
        elif risky_count / total > 0.4:
            style = "Risk-seeking"
        else:
            style = "Balanced"
        
        return {
            'style': style,
            'favorite_contracts': [c[0] for c in most_common],
            'contract_distribution': dict(contract_counts)
        }
    
    def _analyze_risk_tolerance(self, player_id):
        """Analyze player's risk tolerance based on contract choices and success."""
        player_games = [g for g in self.game_history if g['declarer'] == player_id]
        
        if not player_games:
            return "Unknown"
        
        success_rate = np.mean([g['success'] for g in player_games])
        avg_reward = np.mean([g['rewards'][player_id] for g in self.game_history])
        
        # Risk score based on contract difficulty and frequency
        high_risk_contracts = ['SOLO_10', 'SOLO_11', 'SOLO_12', 'MISERE', 'OPEN_MISERE', 'SOLO_13']
        high_risk_attempts = sum(1 for g in player_games if g['contract'] in high_risk_contracts)
        risk_ratio = high_risk_attempts / len(player_games)
        
        if risk_ratio > 0.3:
            tolerance = "High"
        elif risk_ratio > 0.1:
            tolerance = "Medium" 
        else:
            tolerance = "Low"
        
        return {
            'tolerance_level': tolerance,
            'success_rate': success_rate,
            'risk_ratio': risk_ratio,
            'avg_reward': avg_reward
        }
    
    def _analyze_trump_preferences(self, player_id):
        """Analyze trump suit preferences."""
        trump_games = [g for g in self.game_history if g['declarer'] == player_id and g['trump']]
        
        if not trump_games:
            return "No trump data"
        
        trump_counts = Counter([g['trump'] for g in trump_games])
        trump_success = defaultdict(list)
        
        for game in trump_games:
            trump_success[game['trump']].append(game['success'])
        
        preferences = {}
        for trump, count in trump_counts.items():
            success_rate = np.mean(trump_success[trump]) if trump_success[trump] else 0
            preferences[trump] = {
                'usage_count': count,
                'success_rate': success_rate,
                'preference_score': count * success_rate
            }
        
        return preferences
    
    def _analyze_performance_metrics(self, player_id):
        """Analyze detailed performance metrics."""
        all_games = self.game_history
        player_games = [g for g in all_games if g['declarer'] == player_id]
        
        metrics = {
            'games_as_declarer': len(player_games),
            'overall_win_rate': np.mean([1 if g['rewards'][player_id] > 0 else 0 for g in all_games]),
            'declarer_success_rate': np.mean([g['success'] for g in player_games]) if player_games else 0,
            'avg_tricks_per_game': np.mean([g['tricks_won'][player_id] for g in all_games]),
            'avg_reward_per_game': np.mean([g['rewards'][player_id] for g in all_games]),
            'consistency_score': 1 - np.std([g['rewards'][player_id] for g in all_games]) / (np.mean([abs(g['rewards'][player_id]) for g in all_games]) + 1e-6)
        }
        
        return metrics
    
    def create_strategy_guide(self, timestamp):
        """Create human-readable strategy guide."""
        strategies = self.extract_player_strategies(timestamp)
        
        guide = f"""
RIKKEN AI STRATEGY GUIDE
Generated from {len(self.game_history)} games
{'='*50}

"""
        
        for player_name, strategy in strategies.items():
            guide += f"""
{player_name.upper()} STRATEGY PROFILE
{'-'*30}

BIDDING STYLE: {strategy['bidding_style']['style']}
Favorite Contracts: {', '.join(strategy['bidding_style']['favorite_contracts'][:3])}

RISK PROFILE: {strategy['risk_tolerance']['tolerance_level']} Risk Tolerance
Success Rate as Declarer: {strategy['risk_tolerance']['success_rate']:.1%}
Average Reward per Game: {strategy['risk_tolerance']['avg_reward']:.1f}

PERFORMANCE METRICS:
- Overall Win Rate: {strategy['performance_metrics']['overall_win_rate']:.1%}
- Average Tricks per Game: {strategy['performance_metrics']['avg_tricks_per_game']:.1f}
- Declarer Success Rate: {strategy['performance_metrics']['declarer_success_rate']:.1%}
- Consistency Score: {strategy['performance_metrics']['consistency_score']:.2f}

TRUMP PREFERENCES:
"""
            
            if isinstance(strategy['trump_preferences'], dict):
                for trump, data in sorted(strategy['trump_preferences'].items(), 
                                        key=lambda x: x[1]['preference_score'], reverse=True):
                    guide += f"  {trump}: {data['usage_count']} games, {data['success_rate']:.1%} success\n"
            
            guide += "\n"
        
        # Add general insights
        guide += """
GENERAL INSIGHTS
----------------
"""
        
        if self.game_history:
            overall_success = np.mean([g['success'] for g in self.game_history if g['contract']])
            guide += f"Overall Contract Success Rate: {overall_success:.1%}\n"
            
            most_successful_contract = max(self.contract_outcomes.items(), 
                                         key=lambda x: np.mean(x[1]['success']) if x[1]['success'] else 0)
            guide += f"Most Successful Contract Type: {most_successful_contract[0]}\n"
        
        guide += """
RECOMMENDED PLAYING STRATEGIES
------------------------------
Based on the AI analysis, here are key strategic insights:

1. CONTRACT SELECTION: Focus on contracts with >40% success rate
2. TRUMP MANAGEMENT: Choose trump suits where you have length and strength
3. RISK ASSESSMENT: Balance high-reward contracts with sustainable success rates
4. PARTNERSHIP PLAY: In Rik/Troela, coordinate with partner for maximum tricks
"""
        
        with open(f"{self.output_dir}/strategies/strategy_guide_{timestamp}.txt", 'w') as f:
            f.write(guide)
        
        print("Strategy Guide Preview:")
        print(guide[:1000] + "..." if len(guide) > 1000 else guide)

# ==================== Enhanced Training Function ====================

def train_with_comprehensive_analysis(num_episodes=1000):
    """Train Rikken agents with full analysis and visualization."""
    
    # Import the fixed environment (assuming it's in the same file)
    from rikclaude import RikkenEnv, ContractType
    
    env = RikkenEnv()
    analyzer = RikkenAnalyzer()
    
    def smart_random_policy(obs, legal_actions, hand, phase):
        if not legal_actions:
            return 52
        
        if phase == "bidding" and 52 in legal_actions:
            high_cards = sum(1 for card in hand if card.rank.value >= 11)
            aces = sum(1 for card in hand if card.rank.value == 14)
            
            if aces == 3:
                troela_actions = [a for a in legal_actions if a >= 52 + ContractType.TROELA.value]
                if troela_actions:
                    return min(troela_actions)
            
            if high_cards < 6:
                return 52
            
            basic_bids = [a for a in legal_actions if 52 + ContractType.RIK.value <= a <= 52 + ContractType.SOLO_8.value]
            if basic_bids:
                return np.random.choice(basic_bids)
            
            return 52
        
        return np.random.choice(legal_actions)
    
    print(f"Starting enhanced Rikken training with analysis...")
    print(f"Output directory: {analyzer.output_dir}")
    
    for episode in range(num_episodes):
        try:
            obs = env.reset()
            done = False
            game_step = 0
            max_steps = 200
            
            while not done and game_step < max_steps:
                player_id = env.current_player
                legal_actions = env.get_legal_actions(player_id)
                
                if not legal_actions:
                    env.current_player = (env.current_player + 1) % 4
                    game_step += 1
                    continue
                
                player_hand = env.hands[player_id]
                action = smart_random_policy(obs[player_id], legal_actions, player_hand, env.phase)
                
                next_obs, rewards, done, info = env.step(player_id, action)
                
                if not isinstance(rewards, dict):
                    rewards = {i: 0 for i in range(4)}
                
                obs = next_obs
                game_step += 1
            
            # Record game for analysis
            if done and env.contract:
                analyzer.record_game(env, rewards, episode)
            
        except Exception as e:
            print(f"Error in episode {episode}: {e}")
            continue
        
        # Progress updates
        if episode > 0 and episode % 200 == 0:
            print(f"Episode {episode}/{num_episodes} complete")
            
            # Intermediate analysis
            if episode % 500 == 0:
                print("Generating intermediate analysis...")
                analyzer.generate_comprehensive_analysis()
    
    # Final comprehensive analysis
    print("\nGenerating final comprehensive analysis...")
    analyzer.generate_comprehensive_analysis()
    
    return analyzer

if __name__ == "__main__":
    print("Starting Enhanced Rikken Analysis Training...")
    analyzer = train_with_comprehensive_analysis(1000)
    print(f"\nTraining complete! Check '{analyzer.output_dir}' for comprehensive results.")