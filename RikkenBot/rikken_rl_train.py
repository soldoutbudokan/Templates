"""
Rikken RL: Maskable/Recurrent PPO + Analysis
- Self-play with a shared policy that controls all seats
- Action masking for legality
- Optional recurrent policy (card counting via LSTM)
- Integrated analysis reports and plots

Install:
  python -m pip install "stable-baselines3>=2.3.0" "sb3-contrib>=2.3.0" gymnasium torch numpy pandas matplotlib seaborn
"""

import os
import json
import argparse
from collections import defaultdict, Counter
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import gymnasium as gym
from gymnasium import spaces

from sb3_contrib import MaskablePPO, RecurrentPPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import set_random_seed

# Your env API (mock provided in rikclaude.py)
from rikclaude import RikkenEnv, ContractType, Suit, Card

# ==================== Analyzer ====================

class RikkenAnalyzer:
    """Comprehensive analysis of Rikken training results."""

    def __init__(self, output_dir=None):
        if output_dir:
            self.output_dir = output_dir
        else:
            try:
                script_dir = os.path.dirname(os.path.abspath(__file__))
            except (NameError, TypeError):
                script_dir = os.getcwd()
            self.output_dir = os.path.join(script_dir, "output")

        if not self.output_dir:
            self.output_dir = os.path.join(os.getcwd(), "output")

        self.ensure_output_dir()

        self.game_history = []
        self.bidding_patterns = defaultdict(lambda: defaultdict(int))
        self.contract_outcomes = defaultdict(lambda: defaultdict(list))
        self.trump_choices = defaultdict(int)

    def ensure_output_dir(self):
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(f"{self.output_dir}/graphs", exist_ok=True)
        os.makedirs(f"{self.output_dir}/strategies", exist_ok=True)
        os.makedirs(f"{self.output_dir}/logs", exist_ok=True)

    def record_game(self, env, rewards, episode_num):
        game_data = {
            'episode': episode_num,
            'contract': env.contract.type.name if getattr(env, "contract", None) else None,
            'declarer': env.contract.declarer if getattr(env, "contract", None) else None,
            'partner': env.contract.partner if getattr(env, "contract", None) else None,
            'trump': env.trump_suit.name if getattr(env, "trump_suit", None) else None,
            'tricks_won': getattr(env, "tricks_won", [0, 0, 0, 0]).copy(),
            'rewards': rewards.copy() if isinstance(rewards, dict) else {i: 0 for i in range(4)},
            'success': (rewards.get(env.contract.declarer, 0) > 0) if getattr(env, "contract", None) else False,
            'total_tricks': sum(getattr(env, "tricks_won", [0, 0, 0, 0])),
            'bidding_rounds': len(getattr(env, "bidding_history", [])),
        }
        self.game_history.append(game_data)

        if getattr(env, "contract", None):
            declarer = env.contract.declarer
            contract_type = env.contract.type.name
            self.bidding_patterns[declarer][contract_type] += 1
            self.contract_outcomes[contract_type]['success'].append(game_data['success'])
            self.contract_outcomes[contract_type]['tricks'].append(env.tricks_won[declarer])
            if getattr(env, "trump_suit", None):
                self.trump_choices[env.trump_suit.name] += 1

    def generate_comprehensive_analysis(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.create_basic_stats_report(timestamp)
        self.create_performance_graphs(timestamp)
        self.create_contract_analysis_graphs(timestamp)
        self.create_strategy_heatmaps(timestamp)
        self.create_strategy_guide(timestamp)

    def create_basic_stats_report(self, timestamp):
        if not self.game_history:
            return
        report = {
            'summary': {
                'total_games': len(self.game_history),
                'games_with_contracts': sum(1 for g in self.game_history if g['contract']),
                'overall_success_rate': float(np.mean([g['success'] for g in self.game_history if g['contract']])) if any(g['contract'] for g in self.game_history) else 0.0,
            },
            'player_performance': {},
            'contract_analysis': {},
            'trump_preferences': dict(self.trump_choices),
        }
        for player_id in range(4):
            player_games = [g for g in self.game_history if g['declarer'] == player_id]
            if player_games:
                report['player_performance'][f'player_{player_id}'] = {
                    'contracts_attempted': len(player_games),
                    'success_rate': float(np.mean([g['success'] for g in player_games])),
                    'avg_tricks': float(np.mean([g['tricks_won'][player_id] for g in self.game_history])),
                    'favorite_contracts': dict(Counter([g['contract'] for g in player_games]).most_common(3)),
                    'avg_reward': float(np.mean([g['rewards'][player_id] for g in self.game_history])),
                }
        for contract_type, outcomes in self.contract_outcomes.items():
            if outcomes['success']:
                report['contract_analysis'][contract_type] = {
                    'attempts': len(outcomes['success']),
                    'success_rate': float(np.mean(outcomes['success'])),
                    'avg_tricks_when_attempted': float(np.mean(outcomes['tricks'])),
                }
        with open(f"{self.output_dir}/analysis_report_{timestamp}.json", 'w') as f:
            json.dump(report, f, indent=2, default=str)
        self.create_readable_summary(report, timestamp)

    def create_readable_summary(self, report, timestamp):
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
        total_trump_games = sum(report['trump_preferences'].values()) or 1
        for trump, count in sorted(report['trump_preferences'].items(), key=lambda x: x[1], reverse=True):
            percentage = 100.0 * count / total_trump_games
            summary += f"{trump}: {count} games ({percentage:.1f}%)\n"
        with open(f"{self.output_dir}/summary_{timestamp}.txt", 'w') as f:
            f.write(summary)

    def create_performance_graphs(self, timestamp):
        if not self.game_history:
            return
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        episodes = [g['episode'] for g in self.game_history]
        window_size = max(1, min(100, max(1, len(episodes) // 10)))

        for player_id in range(4):
            wins = [(1 if g['rewards'][player_id] > 0 else 0) for g in self.game_history]
            if len(wins) >= window_size:
                rolling_wins = pd.Series(wins).rolling(window_size).mean()
                axes[0, 0].plot(episodes, rolling_wins, label=f'Player {player_id}', alpha=0.8)

        axes[0, 0].set_title('Win Rate Over Time (Rolling Average)')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Win Rate')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        players, success_rates, contract_counts = [], [], []
        for player_id in range(4):
            player_games = [g for g in self.game_history if g['declarer'] == player_id]
            if player_games:
                players.append(f'Player {player_id}')
                success_rates.append(np.mean([g['success'] for g in player_games]))
                contract_counts.append(len(player_games))

        if players:
            bars = axes[0, 1].bar(players, success_rates, alpha=0.7)
            axes[0, 1].set_title('Contract Success Rate by Player')
            axes[0, 1].set_ylabel('Success Rate')
            axes[0, 1].set_ylim(0, 1)
            for bar, count in zip(bars, contract_counts):
                height = bar.get_height()
                axes[0, 1].text(bar.get_x() + bar.get_width() / 2., height + 0.01, f'n={count}', ha='center', va='bottom', fontsize=8)

        all_tricks = [g['tricks_won'] for g in self.game_history]
        if all_tricks and len(all_tricks[0]) == 4:
            tricks_by_player = list(zip(*all_tricks))
            axes[1, 0].boxplot(tricks_by_player, labels=[f'P{i}' for i in range(4)])
            axes[1, 0].set_title('Tricks Won Distribution by Player')
            axes[1, 0].set_ylabel('Tricks Won')
            axes[1, 0].grid(True, alpha=0.3)

        contracts = [g['contract'] for g in self.game_history if g['contract']]
        contract_counts = Counter(contracts)
        if contract_counts:
            top = contract_counts.most_common(10)
            c_names, counts = zip(*top)
            axes[1, 1].bar(range(len(c_names)), counts, alpha=0.7)
            axes[1, 1].set_title('Most Common Contract Types')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].set_xticks(range(len(c_names)))
            axes[1, 1].set_xticklabels(c_names, rotation=45, ha='right')

        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/graphs/performance_analysis_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.close()

    def create_contract_analysis_graphs(self, timestamp):
        if not self.game_history:
            return
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        all_contracts = set()
        for player_id in range(4):
            player_contracts = set(g['contract'] for g in self.game_history if g['declarer'] == player_id and g['contract'])
            all_contracts.update(player_contracts)
        contract_names = sorted(list(all_contracts))
        contract_success_matrix = []
        for player_id in range(4):
            row = []
            for contract in contract_names:
                player_contract_games = [g for g in self.game_history if g['declarer'] == player_id and g['contract'] == contract]
                row.append(np.mean([g['success'] for g in player_contract_games]) if player_contract_games else np.nan)
            contract_success_matrix.append(row)
        if contract_success_matrix and contract_names:
            sns.heatmap(contract_success_matrix,
                        xticklabels=contract_names,
                        yticklabels=[f'Player {i}' for i in range(4)],
                        annot=True, fmt='.2f', cmap='RdYlGn',
                        ax=axes[0, 0], cbar_kws={'label': 'Success Rate'})
            axes[0, 0].set_title('Contract Success Rate by Player')
            axes[0, 0].tick_params(axis='x', rotation=45)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/graphs/contract_analysis_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.close()

    def create_strategy_heatmaps(self, timestamp):
        if not self.game_history:
            return
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        trump_matrix = []
        trumps = ['CLUBS', 'DIAMONDS', 'HEARTS', 'SPADES']
        for player_id in range(4):
            player_trumps = []
            player_trump_games = [g for g in self.game_history if g['declarer'] == player_id and g['trump']]
            total_trump_games = len(player_trump_games) or 1
            for trump in trumps:
                trump_count = sum(1 for g in player_trump_games if g['trump'] == trump)
                percentage = trump_count / total_trump_games
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
        strategies = {}
        for player_id in range(4):
            player_games = [g for g in self.game_history if g['declarer'] == player_id]
            if not player_games:
                continue
            strategy = {
                'bidding_style': self._analyze_bidding_style(player_id),
                'risk_tolerance': self._analyze_risk_tolerance(player_id),
                'trump_preferences': self._analyze_trump_preferences(player_id),
                'performance_metrics': self._analyze_performance_metrics(player_id),
            }
            strategies[f'player_{player_id}'] = strategy
        with open(f"{self.output_dir}/strategies/extracted_strategies_{timestamp}.json", 'w') as f:
            json.dump(strategies, f, indent=2, default=str)
        return strategies

    def _analyze_bidding_style(self, player_id):
        player_games = [g for g in self.game_history if g['declarer'] == player_id]
        contracts = [g['contract'] for g in player_games if g['contract']]
        if not contracts:
            return {"style": "None", "favorite_contracts": [], "contract_distribution": {}}
        contract_counts = Counter(contracts)
        most_common = contract_counts.most_common(3)
        aggressive_contracts = ['SOLO_9', 'SOLO_10', 'SOLO_11', 'SOLO_12', 'SOLO_13']
        conservative_contracts = ['RIK', 'RIK_BETER', 'SOLO_8']
        risky_contracts = ['MISERE', 'PIEK', 'OPEN_MISERE', 'OPEN_PIEK']
        aggressive_count = sum(contract_counts.get(c, 0) for c in aggressive_contracts)
        conservative_count = sum(contract_counts.get(c, 0) for c in conservative_contracts)
        risky_count = sum(contract_counts.get(c, 0) for c in risky_contracts)
        total = len(contracts)
        if total == 0:
            style = "None"
        elif conservative_count / total > 0.6:
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
            'contract_distribution': dict(contract_counts),
        }

    def _analyze_risk_tolerance(self, player_id):
        player_games = [g for g in self.game_history if g['declarer'] == player_id]
        if not player_games:
            return {"tolerance_level": "Unknown", "success_rate": 0.0, "risk_ratio": 0.0, "avg_reward": 0.0}
        success_rate = float(np.mean([g['success'] for g in player_games]))
        avg_reward = float(np.mean([g['rewards'][player_id] for g in self.game_history]))
        high_risk_contracts = ['SOLO_10', 'SOLO_11', 'SOLO_12', 'MISERE', 'OPEN_MISERE', 'SOLO_13']
        high_risk_attempts = sum(1 for g in player_games if g['contract'] in high_risk_contracts)
        risk_ratio = high_risk_attempts / max(1, len(player_games))
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
            'avg_reward': avg_reward,
        }

    def _analyze_trump_preferences(self, player_id):
        trump_games = [g for g in self.game_history if g['declarer'] == player_id and g['trump']]
        if not trump_games:
            return {}
        trump_counts = Counter([g['trump'] for g in trump_games])
        trump_success = defaultdict(list)
        for game in trump_games:
            trump_success[game['trump']].append(game['success'])
        preferences = {}
        for trump, count in trump_counts.items():
            sr = float(np.mean(trump_success[trump])) if trump_success[trump] else 0.0
            preferences[trump] = {'usage_count': count, 'success_rate': sr, 'preference_score': count * sr}
        return preferences

    def _analyze_performance_metrics(self, player_id):
        all_games = self.game_history
        player_games = [g for g in all_games if g['declarer'] == player_id]
        rewards = [g['rewards'][player_id] for g in all_games] if all_games else [0]
        metrics = {
            'games_as_declarer': len(player_games),
            'overall_win_rate': float(np.mean([1 if g['rewards'][player_id] > 0 else 0 for g in all_games])) if all_games else 0.0,
            'declarer_success_rate': float(np.mean([g['success'] for g in player_games])) if player_games else 0.0,
            'avg_tricks_per_game': float(np.mean([g['tricks_won'][player_id] for g in all_games])) if all_games else 0.0,
            'avg_reward_per_game': float(np.mean(rewards)) if rewards else 0.0,
        }
        denom = float(np.mean([abs(r) for r in rewards]) + 1e-6)
        metrics['consistency_score'] = float(1.0 - (np.std(rewards) / denom)) if denom > 0 else 0.0
        return metrics

    def create_strategy_guide(self, timestamp):
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
Favorite Contracts: {', '.join(strategy['bidding_style'].get('favorite_contracts', [])[:3])}

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
            tp = strategy.get('trump_preferences', {})
            for trump, data in sorted(tp.items(), key=lambda x: x[1]['preference_score'], reverse=True):
                guide += f"  {trump}: {data['usage_count']} games, {data['success_rate']:.1%} success\n"
            guide += "\n"

        guide += """
GENERAL INSIGHTS
----------------
"""
        if self.game_history:
            overall_success = float(np.mean([g['success'] for g in self.game_history if g['contract']])) if any(g['contract'] for g in self.game_history) else 0.0
            guide += f"Overall Contract Success Rate: {overall_success:.1%}\n"
            if self.contract_outcomes:
                most_successful_contract = max(
                    self.contract_outcomes.items(),
                    key=lambda x: (np.mean(x[1]['success']) if x[1]['success'] else 0),
                )
                guide += f"Most Successful Contract Type: {most_successful_contract[0]}\n"

        guide += """
RECOMMENDED PLAYING STRATEGIES
------------------------------
1. Choose contracts where success rate exceeds 40%.
2. Prefer trump suits with length and strength.
3. Balance high-reward contracts with steady success.
4. Coordinate in partnership contracts; optimize trick planning.
"""
        with open(f"{self.output_dir}/strategies/strategy_guide_{timestamp}.txt", 'w') as f:
            f.write(guide)

# ==================== Obs & masking helpers ====================

SUITS = ['CLUBS', 'DIAMONDS', 'HEARTS', 'SPADES']

def card_to_index(card) -> int:
    try:
        s = SUITS.index(card.suit.name)
        r = int(card.rank)
        return s * 13 + (r - 2)
    except Exception:
        return -1

def one_hot(n, idx):
    v = np.zeros(n, dtype=np.float32)
    if 0 <= idx < n:
        v[idx] = 1.0
    return v

def build_observation(env: RikkenEnv, seat: int):
    hand_vec = np.zeros(52, dtype=np.float32)
    try:
        for c in env.hands[seat]:
            k = card_to_index(c)
            if 0 <= k < 52:
                hand_vec[k] = 1.0
    except Exception:
        pass

    trump_idx = 0  # 0 = none
    try:
        trump = env.trump_suit
        if trump is not None and hasattr(trump, "name"):
            trump_idx = 1 + SUITS.index(trump.name)  # 1..4
    except Exception:
        pass
    trump_vec = one_hot(5, trump_idx)
    seat_vec = one_hot(4, seat)
    phase_flag = np.array([1.0 if getattr(env, "phase", "") == "bidding" else 0.0], dtype=np.float32)
    obs = np.concatenate([hand_vec, trump_vec, seat_vec, phase_flag], dtype=np.float32)
    return obs

OBS_DIM = 52 + 5 + 4 + 1
ACTION_SPACE_N = 128  # 0..51 cards, 52 PASS, 52+ContractType

# ==================== Wrapper with action masking ====================

class RikkenEnvSB3Wrapper(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, seed: int = 0):
        super().__init__()
        self.env = RikkenEnv(seed=seed)
        self.action_space = spaces.Discrete(ACTION_SPACE_N)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            _ = RikkenEnv(seed=seed)  # no-op; mock doesn't use external seeding here
        self.env.reset()
        seat = getattr(self.env, "current_player", 0)
        obs = build_observation(self.env, seat)
        return obs, {}

    def compute_action_mask(self):
        mask = np.zeros(self.action_space.n, dtype=bool)
        try:
            seat = getattr(self.env, "current_player", 0)
            legal = self.env.get_legal_actions(seat) or []
            legal = [a for a in legal if 0 <= a < self.action_space.n]
            mask[legal] = True
        except Exception:
            mask[52] = True  # PASS fallback
        if not mask.any():
            mask[52 if 52 < self.action_space.n else 0] = True
        return mask

    def step(self, action):
        seat = getattr(self.env, "current_player", 0)
        legal = self.env.get_legal_actions(seat) or []
        if action not in legal:
            action = 52 if 52 in legal else (legal[0] if legal else 52)
        next_obs_all, rewards, done, info = self.env.step(seat, int(action))
        rew = team_reward(self.env, rewards, seat)
        next_seat = getattr(self.env, "current_player", seat)
        obs = build_observation(self.env, next_seat)
        terminated = bool(done)
        truncated = False
        return obs, float(rew), terminated, truncated, (info or {})

def team_reward(env: RikkenEnv, rewards_dict, seat):
    try:
        if getattr(env, "contract", None):
            dec = env.contract.declarer
            partner = env.contract.partner
            team = {dec, partner}
            if seat in team:
                return sum(rewards_dict.get(p, 0) for p in team)
            else:
                others = {0, 1, 2, 3} - team
                return sum(rewards_dict.get(p, 0) for p in others)
        else:
            return 0.0
    except Exception:
        return 0.0

# ==================== Training / Evaluation ====================

def make_env(seed: int):
    def _thunk():
        e = RikkenEnvSB3Wrapper(seed=seed)
        # ActionMasker needs a function with signature: fn(env) -> mask
        e = ActionMasker(e, lambda env: env.compute_action_mask())
        return e
    return _thunk

def train_maskable_ppo(timesteps=500_000, seed=0, recurrent=False, save_path="rikken_ppo.zip"):
    set_random_seed(seed)
    env = DummyVecEnv([make_env(seed)])
    if recurrent:
        model = RecurrentPPO("MlpLstmPolicy", env, verbose=1, n_steps=512, batch_size=256, learning_rate=3e-4, seed=seed)
    else:
        model = MaskablePPO("MlpPolicy", env, verbose=1, n_steps=2048, batch_size=256, learning_rate=3e-4, ent_coef=0.01, seed=seed)
    model.learn(total_timesteps=int(timesteps))
    model.save(save_path)
    return save_path

def _select_action(model, obs, env_raw: RikkenEnv, lstm_state=None, episode_start=True, recurrent=False):
    seat = getattr(env_raw, "current_player", 0)
    legal = env_raw.get_legal_actions(seat) or []
    mask = np.zeros(ACTION_SPACE_N, dtype=bool)
    for a in legal:
        if 0 <= a < ACTION_SPACE_N:
            mask[a] = True
    if not mask.any():
        mask[52 if 52 < ACTION_SPACE_N else 0] = True
    if recurrent:
        action, lstm_state = model.predict(obs, state=lstm_state, episode_start=np.array([episode_start]),
                                           deterministic=False, action_masks=mask)
        return int(action), lstm_state
    else:
        action, _ = model.predict(obs, deterministic=False, action_masks=mask)
        return int(action), None

def evaluate_model(model_path, episodes=300, recurrent=False, output_dir=None, seed=0):
    model = RecurrentPPO.load(model_path) if recurrent else MaskablePPO.load(model_path)
    analyzer = RikkenAnalyzer(output_dir=output_dir)

    for ep in range(episodes):
        env = RikkenEnv(seed + ep)
        env.reset()
        done = False
        lstm_state = None
        episode_start = True

        while not done:
            seat = getattr(env, "current_player", 0)
            obs = build_observation(env, seat)
            action, lstm_state = _select_action(model, obs, env, lstm_state, episode_start, recurrent)
            episode_start = False
            _, rewards, done, _ = env.step(seat, int(action))

        if getattr(env, "contract", None):
            analyzer.record_game(env, rewards if isinstance(rewards, dict) else {i: 0 for i in range(4)}, ep)

    analyzer.generate_comprehensive_analysis()
    return analyzer

# ==================== CLI ====================

def main():
    parser = argparse.ArgumentParser(description="Train and evaluate Rikken PPO")
    parser.add_argument("--timesteps", type=int, default=300_000, help="Total training timesteps")
    parser.add_argument("--episodes", type=int, default=500, help="Evaluation episodes")
    parser.add_argument("--recurrent", action="store_true", help="Use RecurrentPPO with LSTM")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save", type=str, default="rikken_ppo.zip")
    parser.add_argument("--out", type=str, default=None, help="Output directory for analysis")
    args = parser.parse_args()

    print("=== Training ===")
    model_path = train_maskable_ppo(
        timesteps=args.timesteps,
        seed=args.seed,
        recurrent=args.recurrent,
        save_path=args.save
    )

    print("\n=== Evaluation & Analysis ===")
    analyzer = evaluate_model(
        model_path=model_path,
        episodes=args.episodes,
        recurrent=args.recurrent,
        output_dir=args.out,
        seed=args.seed
    )
    print(f"\nDone. See '{analyzer.output_dir}' for results.")

if __name__ == "__main__":
    main()
