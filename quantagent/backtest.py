"""Execution-aligned long/flat backtest with transaction costs and benchmarks."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _returns_metrics(returns: pd.Series) -> dict[str, float]:
    values = pd.to_numeric(returns, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        raise ValueError("No valid returns to evaluate")
    if (values < -1).any():
        raise ValueError("Returns cannot be less than -100%")
    equity = (1.0 + values).cumprod()
    # Include starting capital (equity=1) when calculating drawdown.
    peak = np.maximum.accumulate(np.r_[1.0, equity.to_numpy()])[1:]
    drawdown = equity.to_numpy() / peak - 1.0
    std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    sharpe = float(np.sqrt(252) * values.mean() / std) if std > 0 else 0.0
    return {
        "total_return": float(equity.iloc[-1] - 1.0),
        "annualized_sharpe": sharpe,
        "max_drawdown": float(drawdown.min()),
        "observations": len(values),
    }


def evaluate_backtest(predictions: pd.DataFrame, transaction_cost_bps: float = 5.0) -> dict[str, float]:
    """Signals formed at t apply to next_return on the same row; costs use position changes.

    Features are known at close t and next_return is close(t+1)/close(t)-1. Shifting positions
    again would add an unintended extra-period delay. Entry costs are charged from cash.
    """
    if not np.isfinite(transaction_cost_bps) or transaction_cost_bps < 0:
        raise ValueError("transaction_cost_bps must be finite and non-negative")
    required = {"next_return", "position"}
    if not required.issubset(predictions.columns):
        raise ValueError(f"Predictions must include {sorted(required)}")
    positions = pd.to_numeric(predictions["position"], errors="coerce")
    returns = pd.to_numeric(predictions["next_return"], errors="coerce")
    if positions.isna().any() or returns.isna().any():
        raise ValueError("Positions and returns must be finite numeric values")
    if not np.isfinite(positions.to_numpy()).all() or not np.isfinite(returns.to_numpy()).all():
        raise ValueError("Positions and returns must be finite numeric values")
    if ((positions < 0) | (positions > 1)).any():
        raise ValueError("Positions must be between 0 (flat) and 1 (long)")
    if (returns < -1).any():
        raise ValueError("Returns cannot be less than -100%")
    turnover = positions.diff().abs()
    turnover.iloc[0] = abs(float(positions.iloc[0]))
    costs = turnover * (transaction_cost_bps / 10_000.0)
    result = _returns_metrics(positions * returns - costs)
    benchmark = _returns_metrics(returns)
    result.update({
        "turnover": float(turnover.mean()),
        "transaction_cost_bps": float(transaction_cost_bps),
        "buy_hold_total_return": benchmark["total_return"],
        "buy_hold_annualized_sharpe": benchmark["annualized_sharpe"],
        "buy_hold_max_drawdown": benchmark["max_drawdown"],
        "excess_return_vs_buy_hold": result["total_return"] - benchmark["total_return"],
    })
    return result
