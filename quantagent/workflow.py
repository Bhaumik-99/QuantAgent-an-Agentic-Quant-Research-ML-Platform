"""Deterministic orchestration and experiment report generation."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib

from .data import synthetic_prices, yahoo_prices
from .agents.research import ResearchAgent
from .agents.data_agent import DataAgent
from .agents.ml_agent import MLAgent
from .agents.risk_agent import RiskEvaluationAgent


def run_research(
    symbol: str = "DEMO",
    periods: int = 800,
    seed: int = 7,
    transaction_cost_bps: float = 5.0,
    source: str = "synthetic",
) -> dict:
    if source not in {"synthetic", "yfinance"}:
        raise ValueError("source must be 'synthetic' or 'yfinance'")
    if transaction_cost_bps < 0:
        raise ValueError("transaction_cost_bps must be non-negative")
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
    fingerprint = hashlib.sha256(pd_to_csv_bytes(prices)).hexdigest()
    run_material = f"{symbol}|{source}|{seed}|{periods}|{transaction_cost_bps}|{fingerprint}"
    run_id = hashlib.sha256(run_material.encode()).hexdigest()[:16]
    return {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
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
        "integrity_checks": checks,
        "agent_trace": trace,
        "disclaimer": "Research prototype only. Backtests are not predictive guarantees or investment advice.",
    }


def pd_to_csv_bytes(frame) -> bytes:
    return frame.to_csv(index=False, date_format="%Y-%m-%dT%H:%M:%S").encode("utf-8")
