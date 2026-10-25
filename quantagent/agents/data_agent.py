from .research import AgentNote
from ..data import validate_prices


class DataAgent:
    def run(self, frame):
        checked = validate_prices(frame)
        note = AgentNote(
            "data",
            "completed",
            f"Validated {len(checked)} observations; timestamps unique and prices positive.",
        )
        return checked, note
