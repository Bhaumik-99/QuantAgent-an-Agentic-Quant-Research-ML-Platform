# QuantAgent

Quantitative research and machine-learning evaluation prototype with a deterministic Python workflow, typed MCP tools, FastAPI, and a Streamlit dashboard.

## What it does

- Validates and fingerprints market-price datasets.
- Trains a logistic-regression direction baseline using a chronological holdout.
- Runs an expanding-window walk-forward evaluation with three future test windows.
- Simulates a long/flat strategy with transaction costs, turnover, drawdown, annualized Sharpe, and buy-and-hold comparison.
- Exposes a reproducibility/contract benchmark for the current Python orchestration layer.
- Includes API tests and continuous integration on Python 3.11 and 3.12.

## Quickstart

Python 3.11 or 3.12 is recommended.

```bash
python -m venv .venv
# Activate the environment, then:
python -m pip install -e ".[dev,mcp]"
pytest -q
python scripts/benchmark_workflow.py
uvicorn quantagent.api:app --reload
```

In another terminal, run `streamlit run quantagent/dashboard.py`.

API docs: http://127.0.0.1:8000/docs

## Evaluation design and limitations

- **Signal alignment:** features are formed using information available at time t; the signal applies to `next_return = close(t+1)/close(t)-1` on the same row. The backtester does not shift that signal a second time.
- **Costs:** transaction costs are charged on entry and changes in position. The default is 5 basis points per unit of turnover.
- **Walk-forward:** each fold fits a fresh scaler and classifier on all prior observations, then evaluates on the next non-overlapping window. No test fold is used for fitting.
- **Baseline:** buy-and-hold metrics are reported alongside the strategy. Results are historical simulations and depend on the chosen data, assumptions, and transaction-cost model.
- **Agent scope:** current agents are deterministic Python components, not autonomous LLM agents. The benchmark measures workflow contract success, reproducibility, and latency; it does not claim to measure LLM reasoning or prove multi-agent superiority.
- **Not production trading:** no brokerage connection, live orders, or guarantee of profitability. This is an educational research prototype, not investment advice.

## Optional integrations

Install `.[tracking]` to experiment with MLflow tracking, or `.[boosting]` to experiment with XGBoost. These optional dependencies are not required by the baseline workflow.
