from .research import AgentNote
from ..features import make_features
from ..models import train_baseline


class MLAgent:
    def run(self, prices):
        features = make_features(prices)
        result = train_baseline(features)
        note = AgentNote(
            "ml",
            "completed",
            f"Trained chronological baseline on {result.train_rows} rows and evaluated on {result.test_rows} rows.",
        )
        return result, note
