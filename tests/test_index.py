"""Phase 3 gate: both indexes queryable on the real corpus, backend reported, metrics logged."""

from __future__ import annotations

import numpy as np
import pytest

import index
from config import load_config
from evaluate import log_run, read_runs

RAG_ID = "10.48550/arxiv.2005.11401"
KNOWN_ANSWER_QUERIES = [                # (keyword query, expected paper at rank 1)
    ("hippocampo-neocortical interaction compressive", "10.1038/s41467-026-74357-6"),
    ("knowledge-intensive NLP tasks parametric non-parametric memory", RAG_ID),
    ("corrective retrieval augmented generation evaluator", "10.2139/ssrn.5267341"),
]


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def conn(cfg):
    c = index.connect(cfg.index_path)
    if index.count_chunks(c) == 0:
        pytest.skip("index has no chunks; run Phase 2 first")
    try:
        index.get_embedder(cfg)
    except RuntimeError as e:
        pytest.skip(str(e))
    yield c
    c.close()


@pytest.fixture(scope="module")
def built(cfg, conn):
    return index.build_index(cfg, conn)


def test_backend_probe_reports_explicitly(conn):
    backend = index.probe_vector_backend(conn)
    assert backend in (index.BACKEND_SQLITE_VEC, index.BACKEND_NUMPY)
    assert index.get_meta(conn, "vector_backend") == backend
    assert index.get_meta(conn, "vector_backend_reason")
    print(f"\nVECTOR BACKEND: {backend} ({index.get_meta(conn, 'vector_backend_reason')})")


def test_build_index_vectorizes_every_chunk(cfg, conn, built):
    assert built["n_vectors"] == built["n_chunks"] == index.count_chunks(conn)
    assert built["n_papers"] >= 20 and built["index_size_mb"] > 1
    assert built["index_build_s"] > 0 and built["peak_rss_mb"] < 6 * 1024
    print(f"\nINDEX: {built}")


def test_rebuild_uses_embedding_cache(cfg, conn, built):
    again = index.build_index(cfg, conn)
    assert again["n_embedded"] == 0 and again["n_vectors"] == built["n_vectors"]
    assert again["embed_s"] < 5


@pytest.mark.parametrize("query,expected", KNOWN_ANSWER_QUERIES)
def test_keyword_query_rank1(conn, built, query, expected):
    hits = index.query_fts(conn, query, k=5)
    assert hits, query
    top = index.get_chunk(conn, hits[0][0])
    assert top.paper_id == expected, f"{query!r} -> {top.paper_id} (score {hits[0][1]:.2f})"


def test_vector_query_returns_k_and_finds_paper(cfg, conn, built):
    q = index.embed_query(cfg, conn, "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks")
    assert q.shape == (cfg.embedding_dim,) and abs(np.linalg.norm(q) - 1) < 1e-3
    hits = index.query_vector(cfg, conn, q, k=10)
    assert len(hits) == 10
    assert all(-1.0 <= s <= 1.0001 for _, s in hits)
    assert hits == sorted(hits, key=lambda h: -h[1])
    papers = [index.get_chunk(conn, cid).paper_id for cid, _ in hits]
    assert RAG_ID in papers


def test_vector_backends_agree(cfg, conn, built):
    """Whichever backend is live, the NumPy flat index must rank the same top hit."""
    q = index.embed_query(cfg, conn, "graph based retrieval augmented generation survey")
    live = index.query_vector(cfg, conn, q, k=5)
    ids, mat = index._numpy_index(cfg, conn)
    sims = mat @ q
    top = ids[int(np.argmax(sims))]
    assert live[0][0] == top
    assert abs(live[0][1] - float(sims.max())) < 1e-3


def test_embedding_cache_keyed_by_text(cfg, conn, built):
    import uuid
    text = f"a brand new sentence about sparse retrieval {uuid.uuid4()}"
    vecs, n_new = index.embed_texts(cfg, conn, [text])
    assert n_new == 1
    vecs2, n_new2 = index.embed_texts(cfg, conn, [text])
    assert n_new2 == 0 and np.allclose(vecs, vecs2)


def test_metrics_row_logged(cfg, built, tmp_path):
    path = tmp_path / "runs.csv"
    log_run(cfg, {"phase": 3, "run_name": "t", **{k: built[k] for k in ("vector_backend", "n_chunks", "index_size_mb",
                                                                        "index_build_s")}}, path=path)
    log_run(cfg, {"phase": 3, "run_name": "t2", "new_col": 1}, path=path)
    rows = list(__import__("csv").DictReader(path.open(encoding="utf-8")))
    assert len(rows) == 2 and rows[0]["vector_backend"] == built["vector_backend"]
    assert rows[1]["new_col"] == "1" and rows[0]["new_col"] == "" and float(rows[0]["peak_rss_mb"]) > 0
    assert any(r.get("run_name") == "index_build" for r in read_runs(cfg)), "python index.py must log a row"
