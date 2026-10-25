"""Small reproducibility and contract benchmark for the current orchestration layer.

This measures deterministic workflow reliability, not LLM reasoning quality: the current agents
are Python components and this script intentionally does not claim an LLM-agent comparison.
"""
from __future__ import annotations

import json
import time
from statistics import mean

from quantagent.workflow import run_research


def run_benchmark(repeats: int = 5) -> dict:
    if not 1 <= repeats <= 100:
        raise ValueError("repeats must be between 1 and 100")
    latencies = []
    successes = 0
    run_ids = []
    for _ in range(repeats):
        start = time.perf_counter()
        try:
            report = run_research(periods=300, seed=123, source="synthetic")
            elapsed = time.perf_counter() - start
            valid = (
                bool(report.get("run_id"))
                and bool(report.get("dataset_fingerprint_sha256"))
                and bool(report.get("integrity_checks"))
                and all(report["integrity_checks"].values())
                and report.get("model", {}).get("test_rows", 0) > 0
                and report.get("backtest", {}).get("observations", 0) > 0
            )
            if valid:
                successes += 1
                run_ids.append(report["run_id"])
            latencies.append(elapsed)
        except Exception:
            latencies.append(time.perf_counter() - start)
    deterministic = len(set(run_ids)) <= 1
    return {
        "benchmark": "workflow_contract_and_reproducibility",
        "scope": "Python orchestration; not an LLM-agent quality benchmark",
        "repeats": repeats,
        "successful_runs": successes,
        "success_rate": successes / repeats,
        "mean_latency_seconds": mean(latencies),
        "max_latency_seconds": max(latencies),
        "deterministic_run_id": deterministic,
        "pass": successes == repeats and deterministic,
    }


if __name__ == "__main__":
    result = run_benchmark()
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["pass"] else 1)
