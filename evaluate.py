"""Metrics harness: loads results/eval_set.jsonl, runs a config, appends a row to results/runs.csv.

Phase 3 provides the CSV logger (metrics are logged from Phase 3 onward, SPEC 11.3).
Phase 5 adds the retrieval metrics; Phase 6 the generation metrics.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

from config import Config, peak_rss_mb

# Column order for runs.csv. New columns are appended at the end; missing values are blank.
BASE_COLUMNS = ["timestamp", "phase", "run_name", "chunk_size_tokens", "chunk_overlap_tokens", "retrieval_mode",
                "rerank", "context_chunks", "llm_model", "embedding_model", "vector_backend",
                "n_papers", "n_chunks", "index_size_mb", "index_build_s", "embed_s", "n_embedded",
                "recall_at_5", "recall_at_10", "mrr", "p50_latency_s", "p95_latency_s",
                "citation_integrity", "answers_with_citation", "tokens_per_s", "peak_rss_mb", "notes"]


def log_run(cfg: Config, row: dict, path: Path | None = None) -> Path:
    """Append one metrics row to results/runs.csv (created with a header on first use).

    Any key not in the existing header is added as a new column; earlier rows keep blanks.
    """
    path = path or cfg.runs_csv
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "peak_rss_mb": round(peak_rss_mb(), 1), **row}
    existing: list[dict] = []
    columns = list(BASE_COLUMNS)
    if path.exists():
        with path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            existing = list(reader)
            columns = list(reader.fieldnames or columns)
    for k in row:
        if k not in columns:
            columns.append(k)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=columns)
        w.writeheader()
        for r in existing:
            w.writerow({c: r.get(c, "") for c in columns})
        w.writerow({c: row.get(c, "") for c in columns})
    return path


def read_runs(cfg: Config) -> list[dict]:
    if not cfg.runs_csv.exists():
        return []
    with cfg.runs_csv.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def load_eval_set(cfg: Config) -> list[dict]:
    raise NotImplementedError("Phase 5")


def run_eval(cfg: Config, run_name: str, generate_answers: bool = False) -> dict:
    """Evaluate one configuration and append the metrics row to cfg.runs_csv."""
    raise NotImplementedError("Phase 5")
