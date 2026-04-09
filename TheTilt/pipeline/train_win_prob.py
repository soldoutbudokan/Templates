# %% Imports
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit


# %% Configuration
@dataclass
class WinProbModelConfig:
    features: List[str] = field(default_factory=lambda: [
        "innings",
        "balls_remaining",
        "wickets_in_hand",
        "runs_scored",
        "run_rate",
        "required_run_rate",
        "target",
        "runs_needed",
        "is_powerplay",
        "is_middle",
        "is_death",
        "recent_run_rate",
        "recent_wickets",
    ])
    target: str = "batting_team_won"
    n_estimators: int = 500
    learning_rate: float = 0.05
    max_depth: int = 6
    min_child_samples: int = 100
    test_size: float = 0.2
    random_state: int = 42


def load_config(config_path: str = "config/pipeline_config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# %% Train/test split by match
def split_by_match(
    df: pd.DataFrame,
    config: WinProbModelConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split data by match (not by ball) to avoid leakage."""
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=config.test_size,
        random_state=config.random_state,
    )
    train_idx, test_idx = next(splitter.split(df, groups=df["match_id"]))
    return df.iloc[train_idx], df.iloc[test_idx]


# %% Train model
def train_win_prob_model(
    df: pd.DataFrame,
    config: WinProbModelConfig,
) -> Tuple[lgb.LGBMClassifier, pd.DataFrame, pd.DataFrame]:
    """Train LightGBM win probability model."""
    train_df, test_df = split_by_match(df, config)

    print(f"Training set: {len(train_df):,} balls from {train_df['match_id'].nunique()} matches")
    print(f"Test set: {len(test_df):,} balls from {test_df['match_id'].nunique()} matches")

    X_train = train_df[config.features]
    y_train = train_df[config.target]
    X_test = test_df[config.features]
    y_test = test_df[config.target]

    model = lgb.LGBMClassifier(
        n_estimators=config.n_estimators,
        learning_rate=config.learning_rate,
        max_depth=config.max_depth,
        min_child_samples=config.min_child_samples,
        random_state=config.random_state,
        objective="binary",
        metric="binary_logloss",
        verbose=-1,
    )

    print("Training LightGBM model...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.log_evaluation(100)],
    )

    return model, train_df, test_df


# %% Evaluate model
def evaluate_model(
    model: lgb.LGBMClassifier,
    test_df: pd.DataFrame,
    config: WinProbModelConfig,
) -> Dict[str, float]:
    """Evaluate model with Brier score, log loss, and AUC."""
    X_test = test_df[config.features]
    y_test = test_df[config.target]

    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "brier_score": brier_score_loss(y_test, y_prob),
        "log_loss": log_loss(y_test, y_prob),
        "auc": roc_auc_score(y_test, y_prob),
    }

    print("\n--- Model Evaluation ---")
    print(f"  Brier Score: {metrics['brier_score']:.4f} (target < 0.20)")
    print(f"  Log Loss:    {metrics['log_loss']:.4f}")
    print(f"  AUC:         {metrics['auc']:.4f}")

    # Calibration check
    prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)
    print("\n  Calibration (predicted → actual):")
    for pred, true in zip(prob_pred, prob_true):
        bar = "#" * int(true * 40)
        print(f"    {pred:.2f} → {true:.2f}  {bar}")

    # Feature importances
    importances = pd.Series(
        model.feature_importances_,
        index=config.features,
    ).sort_values(ascending=False)

    print("\n  Feature Importances:")
    for feat, imp in importances.items():
        bar = "#" * int(imp / importances.max() * 30)
        print(f"    {feat:25s} {imp:6.0f}  {bar}")

    return metrics


# %% Sanity checks
def sanity_check(model: lgb.LGBMClassifier, config: WinProbModelConfig) -> None:
    """Run sanity checks on the model with known scenarios."""
    print("\n--- Sanity Checks ---")

    scenarios = [
        {
            "name": "Innings 2, need 2 off 6 balls, 8 wickets in hand",
            "features": {
                "innings": 2, "balls_remaining": 6, "wickets_in_hand": 8,
                "runs_scored": 160, "run_rate": 8.42, "required_run_rate": 2.0,
                "target": 162, "runs_needed": 2, "is_powerplay": 0,
                "is_middle": 0, "is_death": 1, "recent_run_rate": 10.0,
                "recent_wickets": 0,
            },
            "expected": "high (>0.90)",
        },
        {
            "name": "Innings 2, need 60 off 6 balls, 2 wickets in hand",
            "features": {
                "innings": 2, "balls_remaining": 6, "wickets_in_hand": 2,
                "runs_scored": 100, "run_rate": 5.26, "required_run_rate": 60.0,
                "target": 160, "runs_needed": 60, "is_powerplay": 0,
                "is_middle": 0, "is_death": 1, "recent_run_rate": 4.0,
                "recent_wickets": 3,
            },
            "expected": "very low (<0.05)",
        },
        {
            "name": "Innings 1, start of match, 0/0",
            "features": {
                "innings": 1, "balls_remaining": 120, "wickets_in_hand": 10,
                "runs_scored": 0, "run_rate": 0.0, "required_run_rate": 0.0,
                "target": 0, "runs_needed": 0, "is_powerplay": 1,
                "is_middle": 0, "is_death": 0, "recent_run_rate": 0.0,
                "recent_wickets": 0,
            },
            "expected": "near 0.50",
        },
        {
            "name": "Innings 1, 200/2 off 18 overs (dominant)",
            "features": {
                "innings": 1, "balls_remaining": 12, "wickets_in_hand": 8,
                "runs_scored": 200, "run_rate": 11.11, "required_run_rate": 0.0,
                "target": 0, "runs_needed": 0, "is_powerplay": 0,
                "is_middle": 0, "is_death": 1, "recent_run_rate": 14.0,
                "recent_wickets": 0,
            },
            "expected": "high (>0.70)",
        },
    ]

    for scenario in scenarios:
        X = pd.DataFrame([scenario["features"]])[config.features]
        prob = model.predict_proba(X)[0, 1]
        print(f"  {scenario['name']}")
        print(f"    Win prob: {prob:.3f} (expected: {scenario['expected']})")


# %% Save model
def save_model(model: lgb.LGBMClassifier, path: Optional[str] = None) -> Path:
    config = load_config()
    path = Path(path or config["model"]["save_path"])
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "wb") as f:
        pickle.dump(model, f)

    print(f"\nModel saved to {path}")
    return path


# %% Main
def train_and_evaluate(featured_balls_path: Optional[str] = None) -> lgb.LGBMClassifier:
    """Full training pipeline: load data, train, evaluate, save."""
    config_dict = load_config()
    model_config = WinProbModelConfig(
        n_estimators=config_dict["model"]["n_estimators"],
        learning_rate=config_dict["model"]["learning_rate"],
        max_depth=config_dict["model"]["max_depth"],
        min_child_samples=config_dict["model"]["min_child_samples"],
        test_size=config_dict["model"]["test_size"],
        random_state=config_dict["model"]["random_state"],
    )

    processed_dir = Path(config_dict["data"]["processed_dir"])
    path = featured_balls_path or str(processed_dir / config_dict["data"]["featured_balls_file"])

    print(f"Loading featured balls from {path}...")
    df = pd.read_parquet(path)
    print(f"  Loaded {len(df):,} balls")

    model, train_df, test_df = train_win_prob_model(df, model_config)
    metrics = evaluate_model(model, test_df, model_config)
    sanity_check(model, model_config)
    save_model(model)

    return model


# %% Script entry
if __name__ == "__main__":
    train_and_evaluate()
