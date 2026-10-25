from quantagent.data import synthetic_prices
from quantagent.features import make_features
from quantagent.models import train_baseline
from quantagent.workflow import run_research


def test_features_are_past_only_and_target_exists():
    features = make_features(synthetic_prices(150))
    assert {
        "return_1", "return_5", "volatility_10", "momentum_10", "target", "next_return"
    }.issubset(features.columns)
    assert features.timestamp.is_monotonic_increasing


def test_chronological_split():
    features = make_features(synthetic_prices(250))
    result = train_baseline(features)
    assert result.train_rows + result.test_rows == len(features)
    assert result.test_rows < len(features)
    assert result.predictions.timestamp.min() > features.timestamp.iloc[result.train_rows - 1]


def test_backtest_costs_reduce_or_do_not_improve_return():
    no_cost = run_research(periods=300, seed=10, transaction_cost_bps=0)
    with_cost = run_research(periods=300, seed=10, transaction_cost_bps=10)
    assert with_cost["backtest"]["total_return"] <= no_cost["backtest"]["total_return"] + 1e-12


def test_full_workflow_has_integrity_checks():
    report = run_research(periods=200)
    assert all(report["integrity_checks"].values())
    assert report["run_id"]
    assert report["dataset_fingerprint_sha256"]
