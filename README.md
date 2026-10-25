# QuantAgent

Agentic Quant Research & ML Platform: a reproducible research workflow with specialized agents, typed MCP tools, machine-learning experiments, and simulated backtesting.

## Features

- Research, data validation, model training, and risk-evaluation agents
- Deterministic workflow orchestration
- FastAPI service and Streamlit dashboard
- MCP tools for data, training, evaluation, backtesting, and experiment comparison
- Synthetic data for offline demonstrations
- Unit tests and Docker packaging

## Quickstart

Python 3.11 is recommended.

```bash
python -m venv .venv
# Activate the environment, then:
python -m pip install -e ".[dev,mcp]"
pytest
uvicorn quantagent.api:app --reload
```

In another terminal, run `streamlit run quantagent/dashboard.py`.

API docs: http://127.0.0.1:8000/docs

## Project status

This is a research prototype, not a production trading system. It uses simulated portfolios and does not connect to a brokerage or execute live trades. Backtest results are experimental and are not investment advice.
