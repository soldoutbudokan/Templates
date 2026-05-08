# %% Imports
import os
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


# %% Ensemble wrapper (issue #111)
class EnsembleModel:
    """Average predict_proba across K identically-configured LGBM members.

    Members differ only in `random_state` (42, 43, ..., 42+K-1) and are all
    trained on the same fixed 10% holdout split (also seed=42). Differences
    between members are pure gradient-boosting trajectory variance —
    averaging cancels it. See issue #111: a single retrain on a +2-match
    delta moved AB de Villiers from rank #3 to rank #9 because the random
    train/holdout reshuffle produced a substantively different model.

    Exposes the same `predict_proba` / `feature_importances_` interface as
    a single LGBMClassifier, so all callers (compute_tilt, sanity_check)
    work unchanged.
    """

    def __init__(
        self,
        models: List[lgb.LGBMClassifier],
        features: List[str],
        categorical_features: List[str],
    ) -> None:
        self.models = list(models)
        self.features = list(features)
        self.categorical_features = list(categorical_features)

    def __len__(self) -> int:
        return len(self.models)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.models:
            raise RuntimeError("EnsembleModel has no members")
        acc = np.zeros((len(X), 2), dtype=np.float64)
        for m in self.models:
            acc += m.predict_proba(X)
        return acc / len(self.models)

    def predict_proba_per_member(self, X: pd.DataFrame) -> np.ndarray:
        """Stack of per-member predictions, shape (K, n, 2). Diagnostic only."""
        return np.stack([m.predict_proba(X) for m in self.models])

    @property
    def feature_importances_(self) -> np.ndarray:
        return np.mean(
            np.stack([m.feature_importances_ for m in self.models]), axis=0
        )


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
        "over",
        "recent_wickets",
        "venue",
        "batting_team_chose_to_bat",
        "season_numeric",
        "opponent_bowler_economy",
        "batting_team_nrr",
    ])
    categorical_features: List[str] = field(default_factory=lambda: [
        "venue",
    ])
    target: str = "batting_team_won"
    n_estimators: int = 500
    learning_rate: float = 0.05
    max_depth: int = 6
    min_child_samples: int = 100
    num_leaves: int = 31
    reg_lambda: float = 0.0
    test_size: float = 0.2
    random_state: int = 42
    ensemble_size: int = 1


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


# %% Fit a single ensemble member
def _fit_member(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: WinProbModelConfig,
    seed: int,
) -> lgb.LGBMClassifier:
    """Train one LGBMClassifier on a fixed split with the given seed."""
    X_train = train_df[config.features].copy()
    y_train = train_df[config.target]
    X_test = test_df[config.features].copy()
    y_test = test_df[config.target]

    # Monotone constraints guard against pathological splits. Applied only to
    # features whose direction is unambiguous from the batting team's point of
    # view: +1 = more is better, -1 = more is worse, 0 = context-dependent.
    direction = {
        "wickets_in_hand": 1,
        "runs_needed": -1,
        "required_run_rate": -1,
    }
    monotone_constraints = [direction.get(f, 0) for f in config.features]

    model = lgb.LGBMClassifier(
        n_estimators=config.n_estimators,
        learning_rate=config.learning_rate,
        max_depth=config.max_depth,
        min_child_samples=config.min_child_samples,
        num_leaves=config.num_leaves,
        reg_lambda=config.reg_lambda,
        random_state=seed,
        objective="binary",
        metric="binary_logloss",
        monotone_constraints=monotone_constraints,
        verbose=-1,
    )

    for col in config.categorical_features:
        if col in X_train.columns:
            X_train[col] = X_train[col].astype("category")
            X_test[col] = X_test[col].astype("category")

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        categorical_feature=config.categorical_features,
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
        ],
    )

    return model


# %% Train K-model ensemble (single-member behavior recovered when K=1)
def train_ensemble(
    df: pd.DataFrame,
    config: WinProbModelConfig,
) -> Tuple[EnsembleModel, pd.DataFrame, pd.DataFrame]:
    """Train a K-member ensemble on a fixed holdout split."""
    train_df, test_df = split_by_match(df, config)

    print(
        f"Training set: {len(train_df):,} balls from {train_df['match_id'].nunique()} matches"
    )
    print(
        f"Holdout set:  {len(test_df):,} balls from {test_df['match_id'].nunique()} matches "
        f"(seed={config.random_state}, test_size={config.test_size})"
    )

    K = max(1, int(config.ensemble_size))
    print(f"\nTraining K={K}-model ensemble (seeds {config.random_state}..{config.random_state + K - 1})")

    members: List[lgb.LGBMClassifier] = []
    for i, seed in enumerate(range(config.random_state, config.random_state + K)):
        print(f"  [{i + 1:3d}/{K}] random_state={seed} ...", end=" ", flush=True)
        m = _fit_member(train_df, test_df, config, seed)
        members.append(m)
        best_iter = getattr(m, "best_iteration_", None)
        if best_iter is not None:
            print(f"best_iter={best_iter}")
        else:
            print("done")

    ensemble = EnsembleModel(members, config.features, config.categorical_features)
    return ensemble, train_df, test_df


# %% Evaluate model
def evaluate_model(
    model: EnsembleModel,
    test_df: pd.DataFrame,
    config: WinProbModelConfig,
) -> Dict[str, float]:
    """Evaluate model with Brier score, log loss, and AUC."""
    X_test = test_df[config.features].copy()
    y_test = test_df[config.target]

    for col in config.categorical_features:
        if col in X_test.columns:
            X_test[col] = X_test[col].astype("category")

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

    if hasattr(model, "feature_importances_"):
        importances = pd.Series(
            model.feature_importances_,
            index=config.features,
        ).sort_values(ascending=False)

        print("\n  Feature Importances (mean across ensemble members):")
        for feat, imp in importances.items():
            bar = "#" * int(imp / importances.max() * 30)
            print(f"    {feat:25s} {imp:7.1f}  {bar}")

    # K-model disagreement diagnostic (issue #111)
    if isinstance(model, EnsembleModel) and len(model) > 1:
        per_member = model.predict_proba_per_member(X_test)[:, :, 1]
        std_per_ball = per_member.std(axis=0)
        median_pp = float(np.median(std_per_ball)) * 100
        p95_pp = float(np.percentile(std_per_ball, 95)) * 100
        max_pp = float(std_per_ball.max()) * 100
        print(
            f"\n  K-model disagreement on holdout (std across {len(model)} members): "
            f"median={median_pp:.2f}pp, p95={p95_pp:.2f}pp, max={max_pp:.2f}pp"
        )

    return metrics


# %% Sanity checks
def sanity_check(model: EnsembleModel, config: WinProbModelConfig) -> None:
    """Run sanity checks on the model with known scenarios."""
    print("\n--- Sanity Checks ---")

    scenarios = [
        {
            "name": "Innings 2, need 2 off 6 balls, 8 wickets in hand",
            "features": {
                "innings": 2, "balls_remaining": 6, "wickets_in_hand": 8,
                "runs_scored": 160, "run_rate": 8.42, "required_run_rate": 2.0,
                "target": 162, "runs_needed": 2, "over": 19,
                "recent_wickets": 0, "venue": "Wankhede Stadium",
                "batting_team_chose_to_bat": 0, "season_numeric": 2024,
                "opponent_bowler_economy": 8.0, "batting_team_nrr": 0.5,
            },
            "expected": "high (>0.90)",
        },
        {
            "name": "Innings 2, need 60 off 6 balls, 2 wickets in hand",
            "features": {
                "innings": 2, "balls_remaining": 6, "wickets_in_hand": 2,
                "runs_scored": 100, "run_rate": 5.26, "required_run_rate": 60.0,
                "target": 160, "runs_needed": 60, "over": 19,
                "recent_wickets": 3, "venue": "Wankhede Stadium",
                "batting_team_chose_to_bat": 0, "season_numeric": 2024,
                "opponent_bowler_economy": 8.0, "batting_team_nrr": -0.3,
            },
            "expected": "very low (<0.05)",
        },
        {
            "name": "Innings 1, start of match, 0/0",
            "features": {
                "innings": 1, "balls_remaining": 120, "wickets_in_hand": 10,
                "runs_scored": 0, "run_rate": 0.0, "required_run_rate": 0.0,
                "target": 0, "runs_needed": 0, "over": 0,
                "recent_wickets": 0, "venue": "Wankhede Stadium",
                "batting_team_chose_to_bat": 1, "season_numeric": 2024,
                "opponent_bowler_economy": 8.0, "batting_team_nrr": 0.0,
            },
            "expected": "near 0.50",
        },
        {
            "name": "Innings 1, 200/2 off 18 overs (dominant)",
            "features": {
                "innings": 1, "balls_remaining": 12, "wickets_in_hand": 8,
                "runs_scored": 200, "run_rate": 11.11, "required_run_rate": 0.0,
                "target": 0, "runs_needed": 0, "over": 19,
                "recent_wickets": 0, "venue": "M Chinnaswamy Stadium",
                "batting_team_chose_to_bat": 1, "season_numeric": 2024,
                "opponent_bowler_economy": 8.0, "batting_team_nrr": 1.0,
            },
            "expected": "high (>0.70)",
        },
    ]

    for scenario in scenarios:
        X = pd.DataFrame([scenario["features"]])[config.features]
        for col in config.categorical_features:
            if col in X.columns:
                X[col] = X[col].astype("category")
        prob = model.predict_proba(X)[0, 1]
        print(f"  {scenario['name']}")
        print(f"    Win prob: {prob:.3f} (expected: {scenario['expected']})")


# %% Save model
def save_model(model: EnsembleModel, path: Optional[str] = None) -> Path:
    config = load_config()
    path = Path(path or config["model"]["save_path"])
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "wb") as f:
        pickle.dump(model, f)

    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"\nModel saved to {path}  ({size_mb:.1f} MB)")
    return path


# %% Main
def train_and_evaluate(featured_balls_path: Optional[str] = None) -> EnsembleModel:
    """Full training pipeline: load data, train K members, evaluate, save."""
    config_dict = load_config()
    save_path = config_dict["model"]["save_path"]

    # Guardrail: refuse to overwrite an existing pickle unless RETRAIN=1 is set.
    # Two interactive runs of pipeline/run_pipeline.py silently retrained the
    # model in May 2026, swapping a stable pickle for a noisier one — that's
    # how Bhuvneshwar Kumar ended up above ABD in the career top 10. The cron
    # data-refresh workflow uses refresh_tilt_data_only(), which doesn't call
    # this function, so it's unaffected. The retrain-tilt-model workflow sets
    # RETRAIN=1 explicitly. (Issue #111.)
    if not os.environ.get("RETRAIN") and Path(save_path).exists():
        raise RuntimeError(
            f"Refusing to overwrite existing pickle at {save_path}. "
            f"Set RETRAIN=1 to confirm. See issue #111 for context."
        )

    model_config = WinProbModelConfig(
        n_estimators=config_dict["model"]["n_estimators"],
        learning_rate=config_dict["model"]["learning_rate"],
        max_depth=config_dict["model"]["max_depth"],
        min_child_samples=config_dict["model"]["min_child_samples"],
        num_leaves=config_dict["model"].get("num_leaves", 31),
        reg_lambda=config_dict["model"].get("reg_lambda", 0.0),
        test_size=config_dict["model"]["test_size"],
        random_state=config_dict["model"]["random_state"],
        ensemble_size=int(config_dict["model"].get("ensemble_size", 1)),
    )

    processed_dir = Path(config_dict["data"]["processed_dir"])
    path = featured_balls_path or str(processed_dir / config_dict["data"]["featured_balls_file"])

    print(f"Loading featured balls from {path}...")
    df = pd.read_parquet(path)
    print(f"  Loaded {len(df):,} balls")

    # Filter out DLS-affected matches from training (unreliable targets)
    if "dls_method" in df.columns:
        dls_matches = df[df["dls_method"].notna()]["match_id"].unique()
        n_dls = len(dls_matches)
        if n_dls > 0:
            df = df[~df["match_id"].isin(dls_matches)]
            print(f"  Excluded {n_dls} DLS-affected matches ({len(df):,} balls remaining)")

    ensemble, _, test_df = train_ensemble(df, model_config)

    evaluate_model(ensemble, test_df, model_config)
    sanity_check(ensemble, model_config)
    save_model(ensemble)

    return ensemble


# %% Script entry
if __name__ == "__main__":
    train_and_evaluate()
