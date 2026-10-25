"""Market data validation agent."""
from __future__ import annotations

from ..data import validate_prices
from .research import AgentNote


class DataAgent:
    def run(self, frame):
        checked = validate_prices(frame)
        note = AgentNote(
            "data",
            "completed",
            f"Validated {len(checked)} observations; timestamps unique and prices positive.",
        )
        return checked, note
