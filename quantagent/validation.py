"""Expanding-window walk-forward evaluation for time-ordered market data."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

from .backtest import evaluate_backtest
from .features import FEATURE_COLUMNS


def walk_forward_evaluate(
    features: pd.DataFrame,
    initial_train_fraction: float = 0.5,
    n_splits: int = 5,
    transaction_cost_bps: float = 5.0,
) -> dict:
    """Evaluate expanding training windows with non-overlapping future test windows.

    Hyperparameters remain fixed; no test-fold fitting or tuning is performed. This is a
    transparent baseline, not a claim that walk-forward results guarantee future performance.
    """
    if not 0.3 <= initial_train_fraction <= 0.8:
        raise ValueError("initial_train_fraction must be between 0.3 and 0.8")
    if not isinstance(n_splits, int) or not 2 <= n_splits <= 10:
        raise ValueError("n_splits must be an integer between 2 and 10")
    if not np.isfinite(transaction_cost_bps) or transaction_cost_bps < 0:
        raise ValueError("transaction_cost_bps must be finite and non-negative")
    data = features.sort_values("timestamp").reset_index(drop=True)
    if data["timestamp"].duplicated().any() or not data["timestamp"].is_monotonic_increasing:
        raise ValueError("Timestamps must be unique and chronological")
    required = set(FEATURE_COLUMNS + ["target", "next_return", "timestamp", "close"])
    if not required.issubset(data.columns):
        raise ValueError(f"Features must include {sorted(required)}")
    if data[list(FEATURE_COLUMNS)].isna().any().any() or data[["target", "next_return"]].isna().any().any():
        raise ValueError("Features and labels must not contain missing values")
    initial = int(len(data) * initial_train_fraction)
    remaining = len(data) - initial
    fold_size = remaining // n_splits
    if initial < 30 or fold_size < 20:
        raise ValueError("Not enough rows; require >=30 initial training rows and >=20 rows per fold")

    fold_reports = []
    all_predictions = []
    for fold in range(n_splits):
        train_end = initial + fold * fold_size
        test_end = len(data) if fold == n_splits - 1 else initial + (fold + 1) * fold_size
        train = data.iloc[:train_end]
        test = data.iloc[train_end:test_end]
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, random_state=7),
        )
        model.fit(train[FEATURE_COLUMNS], train["target"])
        probability = model.predict_proba(test[FEATURE_COLUMNS])[:, 1]
        predicted = (probability >= 0.5).astype(int)
        output = test[["timestamp", "close", "next_return", "target"]].copy()
        output["probability_up"] = probability
        output["prediction"] = predicted
        output["position"] = predicted.astype(float)
        output["fold"] = fold + 1
        backtest = evaluate_backtest(output, transaction_cost_bps)
        fold_reports.append({
            "fold": fold + 1,
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "train_end": str(train["timestamp"].iloc[-1]),
            "test_start": str(test["timestamp"].iloc[0]),
            "test_end": str(test["timestamp"].iloc[-1]),
            "accuracy": float(accuracy_score(test["target"], predicted)),
            "f1": float(f1_score(test["target"], predicted, zero_division=0)),
            "backtest": backtest,
        })
        all_predictions.append(output)

    predictions = pd.concat(all_predictions, ignore_index=True)
    if predictions["timestamp"].duplicated().any():
        raise ValueError("Walk-forward test windows overlap")
    return {
        "method": "expanding_window",
        "initial_train_fraction": initial_train_fraction,
        "n_splits": n_splits,
        "transaction_cost_bps": float(transaction_cost_bps),
        "folds": fold_reports,
        "out_of_sample_rows": int(len(predictions)),
        "aggregate_backtest": evaluate_backtest(predictions, transaction_cost_bps),
        "disclaimer": "Historical simulation only; no guarantee of future performance.",
    }
