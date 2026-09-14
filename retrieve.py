"""BM25 / dense / hybrid retrieval with optional cross-encoder rerank, behind one interface (SPEC 7.1).

Hybrid fusion is Reciprocal Rank Fusion; ``rrf_weight`` is the weight of the dense list
(1 - weight goes to BM25) and is a swept parameter. The reranker is loaded lazily from the
local HF cache and can be released with ``unload_reranker()`` to protect the memory ceiling.
No network client may be imported here (C1).
"""

from __future__ import annotations

import logging
import os
import sqlite3
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")   # pipeline modules never touch the hub (C1)

import index  # noqa: E402
from config import RETRIEVAL_MODES, Config, hf_hub_cache  # noqa: E402
from models import Retrieved  # noqa: E402

log = logging.getLogger(__name__)
CANDIDATE_MULTIPLIER = 3        # each single-mode list feeding fusion/rerank is k * this
_RERANKER: dict[str, object] = {}
_CONN: dict[str, sqlite3.Connection] = {}


def get_conn(cfg: Config) -> sqlite3.Connection:
    key = str(cfg.index_path)
    if key not in _CONN:
        conn = index.connect(cfg.index_path)
        index.ensure_index(cfg, conn)          # re-chunking cascades vectors away; rebuild from cache
        _CONN[key] = conn
    return _CONN[key]


def close_conn(cfg: Config) -> None:
    c = _CONN.pop(str(cfg.index_path), None)
    if c:
        c.close()


# ----- single modes -----

def bm25(cfg: Config, query: str, k: int, conn: sqlite3.Connection | None = None) -> list[tuple[str, float]]:
    return index.query_fts(conn or get_conn(cfg), query, k)


def dense(cfg: Config, query: str, k: int, conn: sqlite3.Connection | None = None) -> list[tuple[str, float]]:
    conn = conn or get_conn(cfg)
    return index.query_vector(cfg, conn, index.embed_query(cfg, conn, query), k)


def rrf(lists: list[list[tuple[str, float]]], weights: list[float], k_const: int) -> list[tuple[str, float]]:
    """Weighted Reciprocal Rank Fusion: score(d) = sum_i w_i / (k_const + rank_i(d))."""
    scores: dict[str, float] = {}
    for ranked, w in zip(lists, weights):
        for rank, (cid, _) in enumerate(ranked, start=1):
            scores[cid] = scores.get(cid, 0.0) + w / (k_const + rank)
    return sorted(scores.items(), key=lambda x: -x[1])


def hybrid(cfg: Config, query: str, k: int, conn: sqlite3.Connection | None = None,
           rrf_weight: float | None = None) -> list[tuple[str, float]]:
    conn = conn or get_conn(cfg)
    w = cfg.rrf_weight if rrf_weight is None else rrf_weight
    n = k * CANDIDATE_MULTIPLIER
    fused = rrf([bm25(cfg, query, n, conn), dense(cfg, query, n, conn)], [1.0 - w, w], cfg.rrf_k)
    return fused[:k]


# ----- rerank -----

def get_reranker(cfg: Config):
    name = cfg.reranker_model
    if name not in _RERANKER:
        from sentence_transformers import CrossEncoder
        try:
            _RERANKER[name] = CrossEncoder(name, cache_folder=str(hf_hub_cache()), device="cpu",
                                           local_files_only=True)
        except Exception as e:
            raise RuntimeError(f"reranker {name} not in local HF cache ({hf_hub_cache()}); "
                               f"run `python -m fetch --models` first ({type(e).__name__}: {e})") from e
    return _RERANKER[name]


def unload_reranker() -> None:
    """Release the cross-encoder (sequential load/unload keeps peak RSS under the ceiling, SPEC 12)."""
    _RERANKER.clear()


def rerank(cfg: Config, query: str, candidates: list[tuple[str, float]], k: int,
           conn: sqlite3.Connection | None = None) -> list[tuple[str, float]]:
    if not candidates:
        return []
    conn = conn or get_conn(cfg)
    chunks = index.get_chunks(conn, [cid for cid, _ in candidates])
    pairs = [(query, chunks[cid].text) for cid, _ in candidates if cid in chunks]
    ids = [cid for cid, _ in candidates if cid in chunks]
    scores = get_reranker(cfg).predict(pairs, batch_size=16, show_progress_bar=False)
    ranked = sorted(zip(ids, (float(s) for s in scores)), key=lambda x: -x[1])
    return ranked[:k]


# ----- the one interface -----

def retrieve(cfg: Config, query: str, mode: str | None = None, k: int | None = None,
             rerank_: bool | None = None, rrf_weight: float | None = None) -> list[Retrieved]:
    """Top-k chunks for a query. mode in bm25|dense|hybrid; rerank applies a cross-encoder on k*3 candidates."""
    mode = mode or cfg.retrieval_mode
    k = k or cfg.k
    use_rerank = cfg.rerank if rerank_ is None else rerank_
    if mode not in RETRIEVAL_MODES:
        raise ValueError(f"mode must be one of {RETRIEVAL_MODES}, got {mode!r}")
    conn = get_conn(cfg)
    n = k * CANDIDATE_MULTIPLIER if use_rerank else k
    t0 = time.monotonic()
    if mode == "bm25":
        ranked = bm25(cfg, query, n, conn)
    elif mode == "dense":
        ranked = dense(cfg, query, n, conn)
    else:
        ranked = hybrid(cfg, query, n, conn, rrf_weight)
    label = mode
    if use_rerank:
        ranked = rerank(cfg, query, ranked, k, conn)
        label = f"{mode}+rerank"
    ranked = ranked[:k]
    chunks = index.get_chunks(conn, [cid for cid, _ in ranked])
    out = [Retrieved(chunk=chunks[cid], score=score, rank=i + 1, mode=label)
           for i, (cid, score) in enumerate(ranked) if cid in chunks]
    log.debug("%s k=%d %.3fs %r", label, k, time.monotonic() - t0, query[:60])
    return out


def timed_retrieve(cfg: Config, query: str, **kw) -> tuple[list[Retrieved], float]:
    t0 = time.perf_counter()          # monotonic() has ~16 ms resolution on Windows
    out = retrieve(cfg, query, **kw)
    return out, time.perf_counter() - t0


# 5-query smoke set (Phase 4 gate): expected paper must appear in the top 10 for every mode.
SMOKE_SET = [
    ("retrieval-augmented generation for knowledge-intensive NLP tasks with a seq2seq generator",
     "10.48550/arxiv.2005.11401"),
    ("survey of retrieval-augmented generation for large language models naive advanced modular",
     "10.48550/arxiv.2312.10997"),
    ("corrective retrieval augmented generation with a retrieval evaluator", "10.2139/ssrn.5267341"),
    ("hippocampal and neocortical systems as compressive retrieval", "10.1038/s41467-026-74357-6"),
    ("retrieval-augmented generation for AI-generated content survey", "10.1007/s41019-025-00335-5"),
]


def benchmark(cfg: Config, k: int = 10, log: bool = True) -> list[dict]:
    """Per-mode latency and smoke-set hit rate; logs one row per mode to results/runs.csv."""
    import statistics
    from evaluate import log_run
    rows = []
    for mode in RETRIEVAL_MODES:
        for rr in (False, True):
            lat, hits = [], 0
            for q, expected in SMOKE_SET:
                res, dt = timed_retrieve(cfg, q, mode=mode, k=k, rerank_=rr)
                lat.append(dt)
                hits += expected in {r.chunk.paper_id for r in res}
            row = {"phase": 4, "run_name": f"latency_{mode}{'+rerank' if rr else ''}", "retrieval_mode": mode,
                   "rerank": rr, "p50_latency_s": round(statistics.median(lat), 4),
                   "p95_latency_s": round(max(lat), 4), "notes": f"smoke set {hits}/{len(SMOKE_SET)} in top {k}"}
            if log:
                log_run(cfg, row)
            rows.append(row)
    unload_reranker()
    return rows


if __name__ == "__main__":
    import sys
    from config import load_config
    logging.basicConfig(level=logging.WARNING)
    c = load_config()
    if "--bench" in sys.argv:
        for row in benchmark(c):
            print(f"{row['run_name']:22} p50={row['p50_latency_s']:.4f}s max={row['p95_latency_s']:.4f}s  {row['notes']}")
        print(f"logged to {c.runs_csv}")
        sys.exit(0)
    q = " ".join(sys.argv[1:]) or "how does retrieval-augmented generation reduce hallucination"
    for m in RETRIEVAL_MODES:
        res, dt = timed_retrieve(c, q, mode=m, k=5)
        print(f"--- {m} ({dt:.3f}s)")
        for r in res:
            print(f"  {r.rank}. {r.score:.4f} {r.chunk.paper_id} §{r.chunk.section[:40]} p{r.chunk.page_start}")
