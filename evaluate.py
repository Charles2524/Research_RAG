"""Metrics harness: loads results/eval_set.jsonl, runs a config, appends a row to results/runs.csv.

Built in Phase 5 (retrieval metrics) and extended in Phase 6 (generation metrics).
"""

from __future__ import annotations

from config import Config


def load_eval_set(cfg: Config) -> list[dict]:
    raise NotImplementedError("Phase 5")


def run_eval(cfg: Config, run_name: str, generate_answers: bool = False) -> dict:
    """Evaluate one configuration and append the metrics row to cfg.runs_csv."""
    raise NotImplementedError("Phase 5")
