"""Phase 4 gate: every mode returns k; 5-query smoke set hits the expected paper in top 10; latency logged."""

from __future__ import annotations

import statistics

import pytest

import retrieve
from config import RETRIEVAL_MODES, load_config
from evaluate import log_run
from models import Retrieved

from retrieve import SMOKE_SET

RAG_ID = "10.48550/arxiv.2005.11401"
MODES = [(m, False) for m in RETRIEVAL_MODES] + [(m, True) for m in RETRIEVAL_MODES]


@pytest.fixture(scope="module")
def cfg():
    c = load_config()
    if retrieve.get_conn(c).execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 0:
        pytest.skip("no chunks; run Phase 2 first")
    assert retrieve.get_conn(c).execute("SELECT COUNT(*) FROM vectors").fetchone()[0] > 0   # ensure_index ran
    try:
        retrieve.get_reranker(c)
    except RuntimeError as e:
        pytest.skip(str(e))
    return c


def test_rrf_fusion_math():
    a = [("x", 9.0), ("y", 8.0), ("z", 7.0)]
    b = [("y", 0.9), ("q", 0.8)]
    fused = retrieve.rrf([a, b], [0.5, 0.5], 60)
    assert fused[0][0] == "y"                               # appears in both lists
    assert fused[0][1] == pytest.approx(0.5 / 62 + 0.5 / 61)
    assert [c for c, _ in fused] == ["y", "x", "q", "z"]
    only_dense = retrieve.rrf([a, b], [0.0, 1.0], 60)
    assert [c for c, _ in only_dense][:2] == ["y", "q"] and only_dense[2][1] == 0.0


def test_invalid_mode_rejected(cfg):
    with pytest.raises(ValueError):
        retrieve.retrieve(cfg, "x", mode="sparse")


@pytest.mark.parametrize("mode,rr", MODES)
def test_every_mode_returns_k(cfg, mode, rr):
    for k in (1, 5, 10):
        res = retrieve.retrieve(cfg, "large language model hallucination", mode=mode, k=k, rerank_=rr)
        assert len(res) == k
        assert all(isinstance(r, Retrieved) for r in res)
        assert [r.rank for r in res] == list(range(1, k + 1))
        assert all(r.mode == (f"{mode}+rerank" if rr else mode) for r in res)
        assert len({r.chunk.chunk_id for r in res}) == k              # no duplicates
        assert [r.score for r in res] == sorted((r.score for r in res), reverse=True)


@pytest.mark.parametrize("mode,rr", MODES)
def test_smoke_set_expected_paper_in_top10(cfg, mode, rr):
    misses = []
    for q, expected in SMOKE_SET:
        res = retrieve.retrieve(cfg, q, mode=mode, k=10, rerank_=rr)
        if expected not in {r.chunk.paper_id for r in res}:
            misses.append((q[:50], expected))
    assert not misses, f"{mode} rerank={rr} missed: {misses}"


def test_hybrid_weight_extremes_match_single_modes(cfg):
    q = SMOKE_SET[0][0]
    b = [r.chunk.chunk_id for r in retrieve.retrieve(cfg, q, mode="bm25", k=5, rerank_=False)]
    d = [r.chunk.chunk_id for r in retrieve.retrieve(cfg, q, mode="dense", k=5, rerank_=False)]
    h0 = [r.chunk.chunk_id for r in retrieve.retrieve(cfg, q, mode="hybrid", k=5, rerank_=False, rrf_weight=0.0)]
    h1 = [r.chunk.chunk_id for r in retrieve.retrieve(cfg, q, mode="hybrid", k=5, rerank_=False, rrf_weight=1.0)]
    assert h0 == b and h1 == d


def test_latency_benchmark_per_mode(cfg, tmp_path):
    rows = retrieve.benchmark(cfg, k=10, log=False)          # the CLI (--bench) logs the real rows
    for row in rows:
        log_run(cfg, row, path=tmp_path / "runs.csv")
        print(f"\nLATENCY {row['run_name']}: p50={row['p50_latency_s']}s max={row['p95_latency_s']}s {row['notes']}")
    assert len(rows) == 6 and all(r["p50_latency_s"] > 0 for r in rows)
    assert all(r["notes"].startswith("smoke set 5/5") for r in rows)
    assert (tmp_path / "runs.csv").exists()
