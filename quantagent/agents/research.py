from dataclasses import dataclass


@dataclass
class AgentNote:
    agent: str
    status: str
    message: str


class ResearchAgent:
    def run(self, symbol: str) -> AgentNote:
        return AgentNote(
            "research",
            "completed",
            f"Hypothesis: lagged returns may contain weak next-period direction signal for {symbol}; evaluate out of sample.",
        )
