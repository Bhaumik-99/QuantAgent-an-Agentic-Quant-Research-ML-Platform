"""Leakage-aware baseline model training."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_COLUMNS


@dataclass
class ModelResult:
    predictions: pd.DataFrame
    metrics: dict[str, float]
    train_rows: int
    test_rows: int
    model_name: str = "logistic_regression"


def train_baseline(features: pd.DataFrame, train_fraction: float = 0.7) -> ModelResult:
    if not 0.5 <= train_fraction <= 0.9:
        raise ValueError("train_fraction must be between 0.5 and 0.9")
    split = int(len(features) * train_fraction)
    if split < 30 or len(features) - split < 20:
        raise ValueError("Not enough rows for chronological train/test evaluation")
    train, test = features.iloc[:split], features.iloc[split:]
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=7),
    )
    model.fit(train[FEATURE_COLUMNS], train["target"])
    probabilities = model.predict_proba(test[FEATURE_COLUMNS])[:, 1]
    predicted = (probabilities >= 0.5).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(test["target"], predicted)),
        "precision": float(precision_score(test["target"], predicted, zero_division=0)),
        "recall": float(recall_score(test["target"], predicted, zero_division=0)),
        "f1": float(f1_score(test["target"], predicted, zero_division=0)),
    }
    if test["target"].nunique() > 1:
        metrics["roc_auc"] = float(roc_auc_score(test["target"], probabilities))
    output = test[["timestamp", "close", "next_return", "target"]].copy()
    output["probability_up"] = probabilities
    output["prediction"] = predicted
    output["position"] = predicted.astype(float)
    return ModelResult(output.reset_index(drop=True), metrics, len(train), len(test))
