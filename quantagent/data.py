"""Market data adapters and schema validation."""
from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {"timestamp", "close"}


def synthetic_prices(periods: int = 800, seed: int = 7, start: str = "2020-01-01") -> pd.DataFrame:
    """Create deterministic synthetic daily prices for offline development and tests."""
    if periods < 80:
        raise ValueError("periods must be at least 80")
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, periods=periods)
    # Mildly persistent returns make this useful for testing pipelines, not claiming alpha.
    noise = rng.normal(0.0002, 0.012, periods)
    returns = np.empty(periods)
    returns[0] = noise[0]
    for i in range(1, periods):
        returns[i] = 0.12 * returns[i - 1] + noise[i]
    close = 100 * np.exp(np.cumsum(returns))
    return pd.DataFrame({"timestamp": dates, "close": close})


def yahoo_prices(symbol: str, period: str = "5y") -> pd.DataFrame:
    """Download daily adjusted close prices using yfinance."""
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install yfinance to use the Yahoo Finance adapter") from exc
    frame = yf.download(symbol, period=period, auto_adjust=True, progress=False)
    if frame.empty:
        raise ValueError(f"No market data returned for symbol {symbol!r}")
    close = frame["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    result = pd.DataFrame({"timestamp": pd.to_datetime(frame.index), "close": close.to_numpy()})
    return validate_prices(result)


def validate_prices(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize timestamp/close observations."""
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    result = frame[["timestamp", "close"]].copy()
    result["timestamp"] = (
        pd.to_datetime(result["timestamp"], errors="coerce", utc=True).dt.tz_localize(None)
    )
    result["close"] = pd.to_numeric(result["close"], errors="coerce")
    if result.isna().any().any():
        raise ValueError("Data contains invalid or missing timestamps/prices")
    if (result["close"] <= 0).any():
        raise ValueError("Close prices must be positive")
    if result["timestamp"].duplicated().any():
        raise ValueError("Timestamps must be unique")
    result = result.sort_values("timestamp").reset_index(drop=True)
    if len(result) < 80:
        raise ValueError("At least 80 observations are required")
    return result
