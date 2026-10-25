"""Past-only feature construction for next-period direction prediction."""
from __future__ import annotations

import pandas as pd

FEATURE_COLUMNS = ["return_1", "return_5", "volatility_10", "momentum_10"]


def make_features(prices: pd.DataFrame) -> pd.DataFrame:
    data = prices.copy().sort_values("timestamp").reset_index(drop=True)
    data["return_1"] = data["close"].pct_change(1)
    data["return_5"] = data["close"].pct_change(5)
    data["volatility_10"] = data["return_1"].rolling(10).std()
    data["momentum_10"] = data["close"].pct_change(10)
    data["next_return"] = data["close"].shift(-1) / data["close"] - 1
    data["target"] = (data["next_return"] > 0).astype(int)
    return data.dropna(subset=FEATURE_COLUMNS + ["next_return"]).reset_index(drop=True)
