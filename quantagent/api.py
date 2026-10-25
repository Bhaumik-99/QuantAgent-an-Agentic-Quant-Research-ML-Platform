"""FastAPI service."""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .data import synthetic_prices, yahoo_prices
from .workflow import run_research

app = FastAPI(
    title="QuantAgent API",
    version="0.1.0",
    description="Research-only agentic quantitative ML platform",
)


class ResearchRequest(BaseModel):
    symbol: str = Field(default="DEMO", min_length=1, max_length=20)
    periods: int = Field(default=800, ge=80, le=10000)
    seed: int = Field(default=7, ge=0, le=2**32 - 1)
    transaction_cost_bps: float = Field(default=5.0, ge=0, le=1000)
    source: str = Field(default="synthetic", pattern="^(synthetic|yfinance)$")


@app.get("/health")
def health():
    return {"status": "ok", "service": "quantagent"}


@app.get("/market-data/{symbol}")
def market_data(symbol: str, source: str = "synthetic", periods: int = 800, seed: int = 7):
    try:
        frame = (
            synthetic_prices(periods=periods, seed=seed)
            if source == "synthetic"
            else yahoo_prices(symbol)
        )
        sample = frame.tail(20).assign(timestamp=lambda x: x.timestamp.astype(str))
        return {"symbol": symbol, "rows": len(frame), "data": sample.to_dict("records")}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/research/run")
def research_run(request: ResearchRequest):
    try:
        return run_research(**request.model_dump())
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def run():
    import uvicorn

    uvicorn.run("quantagent.api:app", host="127.0.0.1", port=8000, reload=True)
