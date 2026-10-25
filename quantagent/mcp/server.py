"""Optional MCP tool server. Install with: pip install -e '.[mcp]'"""
from __future__ import annotations

from ..data import synthetic_prices, yahoo_prices
from ..workflow import run_research

try:
    from mcp.server.fastmcp import FastMCP
except ImportError as exc:  # pragma: no cover
    raise SystemExit("MCP SDK missing. Install the optional extra: pip install -e '.[mcp]'") from exc

mcp = FastMCP("quantagent")


@mcp.tool()
def get_market_data(
    symbol: str = "DEMO", source: str = "synthetic", periods: int = 800, seed: int = 7
) -> dict:
    """Return metadata and a small sample of validated historical or synthetic prices."""
    if source not in {"synthetic", "yfinance"}:
        raise ValueError("source must be synthetic or yfinance")
    frame = synthetic_prices(periods, seed) if source == "synthetic" else yahoo_prices(symbol)
    sample = frame.tail(10).assign(timestamp=lambda x: x.timestamp.astype(str))
    return {"symbol": symbol, "rows": len(frame), "sample": sample.to_dict("records")}


@mcp.tool()
def train_model(symbol: str = "DEMO", periods: int = 800, seed: int = 7) -> dict:
    """Run the baseline model evaluation and return predictive metrics."""
    report = run_research(symbol=symbol, periods=periods, seed=seed)
    return {"run_id": report["run_id"], "model": report["model"]}


@mcp.tool()
def run_backtest(
    symbol: str = "DEMO", periods: int = 800, seed: int = 7, transaction_cost_bps: float = 5.0
) -> dict:
    """Run a cost-aware long/flat backtest on a chronological test period."""
    report = run_research(
        symbol=symbol, periods=periods, seed=seed, transaction_cost_bps=transaction_cost_bps
    )
    return report["backtest"]


@mcp.tool()
def evaluate_model(symbol: str = "DEMO", periods: int = 800, seed: int = 7) -> dict:
    """Return model metrics and basic evaluation-integrity checks."""
    report = run_research(symbol=symbol, periods=periods, seed=seed)
    return {"model": report["model"], "integrity_checks": report["integrity_checks"]}


@mcp.tool()
def compare_experiments(
    seeds: list[int], periods: int = 800, transaction_cost_bps: float = 5.0
) -> dict:
    """Compare runs over explicit seeds; sensitivity analysis, not proof of model superiority."""
    if not seeds or len(seeds) > 20:
        raise ValueError("Provide between 1 and 20 seeds")
    results = [
        run_research(periods=periods, seed=seed, transaction_cost_bps=transaction_cost_bps)
        for seed in seeds
    ]
    return {
        "runs": [
            {
                "run_id": report["run_id"],
                "seed": report["assumptions"]["seed"],
                "model_metrics": report["model"]["metrics"],
                "backtest": report["backtest"],
            }
            for report in results
        ]
    }


if __name__ == "__main__":
    mcp.run()
