"""Expanding-window walk-forward evaluation for time-ordered market data."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .backtest import evaluate_backtest
from .features import FEATURE_COLUMNS


def walk_forward_evaluate(
    features: pd.DataFrame,
    initial_train_fraction: float = 0.5,
    n_splits: int = 5,
    transaction_cost_bps: float = 5.0,
) -> dict:
    """Evaluate expanding training windows with non-overlapping future test windows.

    The final training observation before each test window is purged because its next-period
    target uses the following close, which is the first observation in the test window.
    Hyperparameters remain fixed; no test-fold fitting or tuning is performed.
    """
    if not np.isfinite(initial_train_fraction) or not 0.3 <= initial_train_fraction <= 0.8:
        raise ValueError("initial_train_fraction must be between 0.3 and 0.8")
    if not isinstance(n_splits, int) or isinstance(n_splits, bool) or not 2 <= n_splits <= 10:
        raise ValueError("n_splits must be an integer between 2 and 10")
    if (
        not isinstance(transaction_cost_bps, (int, float))
        or not np.isfinite(transaction_cost_bps)
        or transaction_cost_bps < 0
    ):
        raise ValueError("transaction_cost_bps must be finite and non-negative")

    required = set(FEATURE_COLUMNS + ["target", "next_return", "timestamp", "close"])
    if not required.issubset(features.columns):
        raise ValueError(f"Features must include {sorted(required)}")
    data = features.sort_values("timestamp").reset_index(drop=True)
    if data["timestamp"].duplicated().any() or not data["timestamp"].is_monotonic_increasing:
        raise ValueError("Timestamps must be unique and chronological")
    numeric_columns = FEATURE_COLUMNS + ["target", "next_return", "close"]
    numeric_values = data[numeric_columns].to_numpy(dtype=float)
    if not np.isfinite(numeric_values).all():
        raise ValueError("Features, labels, returns, and prices must be finite and non-missing")
    if not data["target"].isin([0, 1]).all():
        raise ValueError("Targets must be binary values 0 or 1")

    initial = int(len(data) * initial_train_fraction)
    remaining = len(data) - initial
    fold_size = remaining // n_splits
    if initial < 31 or fold_size < 20:
        raise ValueError("Not enough rows; require >=31 initial rows and >=20 rows per fold")

    fold_reports = []
    all_predictions = []
    for fold in range(n_splits):
        test_start = initial + fold * fold_size
        test_end = len(data) if fold == n_splits - 1 else initial + (fold + 1) * fold_size

        # Purge the observation immediately before test_start: its target uses the
        # first test-period close, so it must not be available during model fitting.
        train = data.iloc[: test_start - 1]
        test = data.iloc[test_start:test_end]
        if train["target"].nunique() < 2:
            raise ValueError(f"Fold {fold + 1} training data must contain both target classes")

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
        fold_reports.append(
            {
                "fold": fold + 1,
                "train_rows": len(train),
                "purged_rows": 1,
                "test_rows": len(test),
                "train_end": str(train["timestamp"].iloc[-1]),
                "test_start": str(test["timestamp"].iloc[0]),
                "test_end": str(test["timestamp"].iloc[-1]),
                "accuracy": float(accuracy_score(test["target"], predicted)),
                "f1": float(f1_score(test["target"], predicted, zero_division=0)),
                "backtest": backtest,
            }
        )
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
        "out_of_sample_rows": len(predictions),
        "aggregate_backtest": evaluate_backtest(predictions, transaction_cost_bps),
        "disclaimer": "Historical simulation only; no guarantee of future performance.",
    }
