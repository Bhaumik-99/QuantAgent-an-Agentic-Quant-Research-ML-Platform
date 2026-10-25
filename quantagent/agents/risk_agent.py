from .research import AgentNote
from ..backtest import evaluate_backtest


class RiskEvaluationAgent:
    def run(self, model_result, transaction_cost_bps: float):
        metrics = evaluate_backtest(model_result.predictions, transaction_cost_bps)
        checks = {
            "chronological_split": model_result.train_rows > 0 and model_result.test_rows > 0,
            "transaction_costs_included": transaction_cost_bps >= 0,
            "out_of_sample_predictions": len(model_result.predictions) == model_result.test_rows,
        }
        if not all(checks.values()):
            raise ValueError("Evaluation integrity gate failed")
        return (
            metrics,
            checks,
            AgentNote(
                "risk_evaluation",
                "completed",
                "Basic evaluation-integrity checks passed; results are not evidence of future profitability.",
            ),
        )
