"""Phase 7 gate: sweep runs every config without crashing, table populated, offline run completes (C3)."""

from __future__ import annotations

import dataclasses

import pytest

import ablate
import evaluate
import fetch
from config import load_config
from sources import SourceUnavailable

AXIS_FIELDS = {"chunk_size": {"chunk_size_tokens", "chunk_overlap_tokens"}, "retrieval_mode": {"retrieval_mode"},
               "rerank": {"rerank"}, "context_chunks": {"context_chunks"}, "model": {"llm_model"},
               "quantization": {"llm_model"}, "baseline": set()}


@pytest.fixture(scope="module")
def cfg():
    c = load_config()
    if not c.eval_set_path.exists():
        pytest.skip("eval set missing")
    return c


def test_sweep_varies_exactly_one_axis_from_baseline(cfg):
    base = cfg
    for name, axis, c in ablate.sweep(base):
        diff = {f.name for f in dataclasses.fields(base) if getattr(base, f.name) != getattr(c, f.name)}
        assert diff == AXIS_FIELDS[axis], f"{name}: differs in {diff}, expected {AXIS_FIELDS[axis]}"
    names = [n for n, _, _ in ablate.AXES]
    assert len(names) == len(set(names)) and 10 <= len(names) <= 14


def test_only_filter_and_dry_run(cfg, capsys):
    assert [n for n, _, _ in ablate.sweep(cfg, {"chunk_256", "rerank_on"})] == ["chunk_256", "rerank_on"]
    assert ablate.main(["--dry-run", "--only", "baseline"]) == 0
    assert "baseline" in capsys.readouterr().out


def test_fast_mini_sweep_completes_and_logs(cfg, tmp_path, monkeypatch):
    logged = []
    monkeypatch.setattr(evaluate, "log_run", lambda c, row, path=None: logged.append(row))
    rows = ablate.run_sweep(cfg, fast=True, limit=3, only={"baseline", "mode_bm25", "rerank_on"})
    assert [r["run_name"] for r in rows] == ["ablate_baseline", "ablate_mode_bm25", "ablate_rerank_on"]
    assert not any(str(r["notes"]).startswith("CRASHED") for r in rows)
    assert all(0 <= r["chunk_recall_at_10"] <= 1 for r in rows) and len(logged) == 3


def test_crashed_config_is_recorded_not_raised(cfg, monkeypatch):
    monkeypatch.setattr(evaluate, "log_run", lambda c, row, path=None: None)
    monkeypatch.setattr(ablate, "AXES", [("boom", "retrieval_mode", {"retrieval_mode": "dense"})])
    monkeypatch.setattr(evaluate, "run_eval", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("kaboom")))
    rows = ablate.run_sweep(cfg, fast=True, limit=1)
    assert len(rows) == 1 and rows[0]["notes"].startswith("CRASHED: RuntimeError: kaboom")


def test_offline_mode_pipeline_runs_and_network_refuses(cfg, tmp_path, monkeypatch):
    """C3: with OFFLINE=1 the whole RAG pipeline works unchanged; discovery/fetch fail fast."""
    monkeypatch.setenv("OFFLINE", "1")
    off = load_config()
    assert off.offline
    with pytest.raises(SourceUnavailable, match="offline"):
        fetch.discover(off, "anything", 1)
    with pytest.raises(SourceUnavailable, match="offline"):
        fetch.download_models(off)
    monkeypatch.setattr(evaluate, "log_run", lambda c, row, path=None: None)
    # any real network attempt would hit here (the pipeline must not make one)
    import requests
    monkeypatch.setattr(requests.Session, "request", lambda *a, **k: (_ for _ in ()).throw(AssertionError("network!")))
    rows = ablate.run_sweep(off, fast=True, limit=3, only={"baseline", "mode_dense"})
    assert not any(str(r["notes"]).startswith("CRASHED") for r in rows)
    import generate
    try:
        generate.ensure_model(off, off.llm_fallback_model)
    except generate.GenerationError as e:
        pytest.skip(f"UNVERIFIED offline generation: {e}")
    row = evaluate.run_eval(off.with_overrides(llm_model=off.llm_fallback_model), "offline_gen", mode="hybrid",
                            rerank=False, generate_answers=True, limit=1, log=False, phase=7)
    assert row["gen_failures"] == 0 and row["answers_with_citation"] == 1.0


def test_results_table_from_runs_csv(cfg, tmp_path):
    t = ablate.results_table(cfg, write=False)
    assert t.startswith("|") or t.startswith("(no phase 7")
