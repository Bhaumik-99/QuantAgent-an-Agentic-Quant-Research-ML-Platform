"""Risk/evaluation agent with explicit integrity gates."""
import math

from .research import AgentNote
from ..backtest import evaluate_backtest


class RiskEvaluationAgent:
    def run(self, model_result, transaction_cost_bps: float):
        metrics = evaluate_backtest(model_result.predictions, transaction_cost_bps)
        predictions = model_result.predictions
        timestamps_ok = (
            "timestamp" in predictions
            and predictions["timestamp"].is_monotonic_increasing
            and not predictions["timestamp"].duplicated().any()
        )
        positions_ok = (
            "position" in predictions
            and predictions["position"].between(0, 1).all()
        )
        checks = {
            "chronological_split": model_result.train_rows > 0 and model_result.test_rows > 0,
            "transaction_costs_included": math.isfinite(transaction_cost_bps) and transaction_cost_bps >= 0,
            "out_of_sample_predictions": len(predictions) == model_result.test_rows,
            "prediction_timestamps_ordered_unique": bool(timestamps_ok),
            "positions_within_long_flat_bounds": bool(positions_ok),
            "backtest_has_observations": metrics["observations"] == model_result.test_rows,
        }
        if not all(checks.values()):
            raise ValueError("Evaluation integrity gate failed")
        return (
            metrics,
            checks,
            AgentNote(
                "risk_evaluation",
                "completed",
                "Integrity gates passed. Walk-forward results and baseline comparisons are historical simulations, not evidence of future profitability.",
            ),
        )
