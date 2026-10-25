"""Deterministic orchestration and experiment report generation."""
from __future__ import annotations

import hashlib
import math
from datetime import UTC, datetime

from .agents.data_agent import DataAgent
from .agents.ml_agent import MLAgent
from .agents.research import ResearchAgent
from .agents.risk_agent import RiskEvaluationAgent
from .data import synthetic_prices, yahoo_prices
from .features import make_features
from .validation import walk_forward_evaluate


def run_research(
    symbol: str = "DEMO",
    periods: int = 800,
    seed: int = 7,
    transaction_cost_bps: float = 5.0,
    source: str = "synthetic",
) -> dict:
    if source not in {"synthetic", "yfinance"}:
        raise ValueError("source must be 'synthetic' or 'yfinance'")
    if not isinstance(periods, int) or isinstance(periods, bool) or periods < 150 or periods > 10000:
        raise ValueError("periods must be an integer between 150 and 10000")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0 or seed > 2**32 - 1:
        raise ValueError("seed must be an integer between 0 and 2**32 - 1")
    if (
        not isinstance(transaction_cost_bps, (int, float))
        or not math.isfinite(transaction_cost_bps)
        or transaction_cost_bps < 0
    ):
        raise ValueError("transaction_cost_bps must be finite and non-negative")
    trace = []
    research_note = ResearchAgent().run(symbol)
    trace.append(research_note.__dict__)
    raw = synthetic_prices(periods=periods, seed=seed) if source == "synthetic" else yahoo_prices(symbol)
    prices, data_note = DataAgent().run(raw)
    trace.append(data_note.__dict__)
    model_result, ml_note = MLAgent().run(prices)
    trace.append(ml_note.__dict__)
    backtest, checks, risk_note = RiskEvaluationAgent().run(model_result, transaction_cost_bps)
    trace.append(risk_note.__dict__)
    walk_forward = walk_forward_evaluate(
        make_features(prices),
        initial_train_fraction=0.5,
        n_splits=3,
        transaction_cost_bps=transaction_cost_bps,
    )
    fingerprint = hashlib.sha256(pd_to_csv_bytes(prices)).hexdigest()
    run_material = f"{symbol}|{source}|{seed}|{periods}|{transaction_cost_bps}|{fingerprint}"
    run_id = hashlib.sha256(run_material.encode()).hexdigest()[:16]
    return {
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "symbol": symbol,
        "source": source,
        "dataset_fingerprint_sha256": fingerprint,
        "assumptions": {
            "strategy": "long/flat direction classifier",
            "transaction_cost_bps": transaction_cost_bps,
            "train_fraction": 0.7,
            "seed": seed,
            "live_trading": False,
        },
        "model": {
            "name": model_result.model_name,
            "train_rows": model_result.train_rows,
            "test_rows": model_result.test_rows,
            "metrics": model_result.metrics,
        },
        "backtest": backtest,
        "walk_forward": walk_forward,
        "integrity_checks": checks,
        "agent_trace": trace,
        "disclaimer": "Research prototype only. Backtests are not predictive guarantees or investment advice.",
    }


def pd_to_csv_bytes(frame) -> bytes:
    return frame.to_csv(index=False, date_format="%Y-%m-%dT%H:%M:%S").encode("utf-8")
