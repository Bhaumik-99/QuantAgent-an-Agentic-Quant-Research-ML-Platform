"""Simple long/flat backtest with transaction costs."""
from __future__ import annotations

import numpy as np
import pandas as pd


def evaluate_backtest(predictions: pd.DataFrame, transaction_cost_bps: float = 5.0) -> dict[str, float]:
    if transaction_cost_bps < 0:
        raise ValueError("transaction_cost_bps must be non-negative")
    required = {"next_return", "position"}
    if not required.issubset(predictions.columns):
        raise ValueError(f"Predictions must include {sorted(required)}")
    # A signal generated at t is applied to the following return, avoiding same-bar look-ahead.
    positions = predictions["position"].astype(float).clip(0, 1)
    held = positions.shift(1).fillna(0.0)
    turnover = held.diff().abs().fillna(held.abs())
    costs = turnover * (transaction_cost_bps / 10_000.0)
    strategy_returns = held * predictions["next_return"].astype(float) - costs
    strategy_returns = strategy_returns.replace([np.inf, -np.inf], np.nan).dropna()
    if strategy_returns.empty:
        raise ValueError("No valid returns to evaluate")
    equity = (1 + strategy_returns).cumprod()
    peak = equity.cummax()
    drawdown = equity / peak - 1
    std = float(strategy_returns.std(ddof=1)) if len(strategy_returns) > 1 else 0.0
    sharpe = float(np.sqrt(252) * strategy_returns.mean() / std) if std > 0 else 0.0
    return {
        "total_return": float(equity.iloc[-1] - 1),
        "annualized_sharpe": sharpe,
        "max_drawdown": float(drawdown.min()),
        "turnover": float(turnover.mean()),
        "observations": int(len(strategy_returns)),
        "transaction_cost_bps": float(transaction_cost_bps),
    }
