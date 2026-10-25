import pandas as pd
import pytest

from quantagent.backtest import evaluate_backtest
from quantagent.data import synthetic_prices
from quantagent.features import make_features
from quantagent.validation import walk_forward_evaluate


def test_signal_is_applied_to_same_row_next_return():
    frame = pd.DataFrame({"position": [1.0, 1.0], "next_return": [0.10, 0.10]})
    result = evaluate_backtest(frame, transaction_cost_bps=0)
    assert result["total_return"] == pytest.approx(0.21)
    assert result["observations"] == 2


def test_backtest_charges_entry_cost_and_reports_benchmark():
    frame = pd.DataFrame({"position": [1.0, 0.0], "next_return": [0.01, -0.02]})
    result = evaluate_backtest(frame, transaction_cost_bps=100)
    assert result["total_return"] < 0.01
    assert "buy_hold_total_return" in result
    assert "excess_return_vs_buy_hold" in result


def test_backtest_rejects_invalid_positions_and_costs():
    with pytest.raises(ValueError, match="between 0"):
        evaluate_backtest(pd.DataFrame({"position": [2.0], "next_return": [0.1]}))
    with pytest.raises(ValueError, match="finite and non-negative"):
        evaluate_backtest(pd.DataFrame({"position": [1.0], "next_return": [0.1]}), float("nan"))


def test_walk_forward_has_non_overlapping_future_folds():
    features = make_features(synthetic_prices(400, seed=3))
    result = walk_forward_evaluate(features, initial_train_fraction=0.5, n_splits=4)
    assert result["out_of_sample_rows"] == sum(f["test_rows"] for f in result["folds"])
    assert all(f["train_end"] < f["test_start"] for f in result["folds"])
    assert result["aggregate_backtest"]["observations"] == result["out_of_sample_rows"]


def test_walk_forward_rejects_small_dataset():
    with pytest.raises(ValueError, match="Not enough rows"):
        walk_forward_evaluate(make_features(synthetic_prices(80)), n_splits=5)
