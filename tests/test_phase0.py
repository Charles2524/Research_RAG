"""Phase 0 gate: imports clean, config validates, schema round-trips."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

import config as config_mod
import index
from config import Config, ConfigError, load_config, peak_rss_mb
from models import STATUS_FETCHED, STATUS_NO_OA_PDF, Answer, Chunk, Paper, Retrieved

ROOT = Path(__file__).resolve().parent.parent
MODULES = ["config", "models", "sources", "fetch", "parse", "chunk", "index",
           "retrieve", "generate", "evaluate", "ablate", "app"]


# ----- imports -----

@pytest.mark.parametrize("name", MODULES)
def test_module_imports_clean(name):
    mod = importlib.import_module(name)
    assert mod is not None


def test_setup_scripts_track_the_code():
    setup = (ROOT / "setup.bat").read_text(encoding="utf-8")
    run = (ROOT / "run.bat").read_text(encoding="utf-8")
    assert "requirements.txt" in setup and "-m fetch --models" in setup and "ollama pull qwen3:1.7b" in setup
    assert "tests\\test_phase0.py" in setup
    assert "server.py %PORT% --open" in run and "11434" in run


def test_every_spec_module_file_exists():
    for name in MODULES:
        p = ROOT / f"{name}.py" if name != "sources" else ROOT / "sources" / "__init__.py"
        assert p.exists(), p


# ----- config -----

def test_default_config_loads_and_validates():
    cfg = load_config()
    assert isinstance(cfg, Config)
    assert cfg.retrieval_mode in config_mod.RETRIEVAL_MODES
    assert cfg.index_path == cfg.data_dir / "index.db"
    assert cfg.pdf_dir.name == "pdfs"


def test_env_example_is_current():
    text = (ROOT / ".env.example").read_text(encoding="utf-8")
    for var in ("CONTACT_EMAIL", *config_mod.KEYED_SOURCES.values()):
        assert var in text, f"{var} missing from .env.example"


def test_empty_env_yields_no_keys_and_no_email(tmp_path):
    env = tmp_path / ".env"
    env.write_text("", encoding="utf-8")
    cfg = load_config(env_path=env)
    assert cfg.contact_email == ""
    assert all(v is None for v in cfg.api_keys.values())
    assert not any(cfg.has_key(s) for s in config_mod.KEYED_SOURCES)


def test_env_provides_email_and_keys(tmp_path):
    env = tmp_path / ".env"
    env.write_text("CONTACT_EMAIL=a@b.c\nCORE_API_KEY=xyz\n", encoding="utf-8")
    cfg = load_config(env_path=env)
    assert cfg.contact_email == "a@b.c"
    assert cfg.has_key("core") and not cfg.has_key("semanticscholar")


def test_offline_flag_from_env_overrides_yaml(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("OFFLINE=1\n", encoding="utf-8")
    assert load_config(env_path=env).offline is True
    monkeypatch.setenv("OFFLINE", "0")
    assert load_config(env_path=env).offline is False


@pytest.mark.parametrize("bad", [
    {"retrieval_mode": "sparse"},
    {"chunk_overlap_tokens": 512, "chunk_size_tokens": 512},
    {"k": 0},
    {"rrf_weight": 1.5},
    {"ollama_host": "http://api.example.com:11434"},   # C1: non-loopback host
    {"ollama_host": "https://127.0.0.1:11434"},
    {"parser": "docling"},
    {"context_chunks": 0},
])
def test_invalid_overrides_rejected(bad):
    cfg = load_config()
    with pytest.raises(ConfigError):
        cfg.with_overrides(**bad)


def test_with_overrides_returns_new_validated_config():
    cfg = load_config()
    v = cfg.with_overrides(retrieval_mode="bm25", chunk_size_tokens=256, chunk_overlap_tokens=32)
    assert v.retrieval_mode == "bm25" and v.chunk_size_tokens == 256
    assert cfg.retrieval_mode == "hybrid"          # original untouched (frozen)
    with pytest.raises(ConfigError):
        cfg.with_overrides(not_a_field=1)


def test_invalid_yaml_value_rejected(tmp_path):
    bad = tmp_path / "config.yaml"
    bad.write_text("retrieval:\n  mode: nope\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(path=bad, env_path=tmp_path / ".env")


def test_peak_rss_is_measurable_and_under_ceiling():
    mb = peak_rss_mb()
    assert 0 < mb < 6 * 1024


# ----- models -----

def test_paper_json_roundtrip():
    p = Paper(paper_id="10.1000/x", title="T", authors=["Patrick Lewis", "Ethan Perez"], year=2024, doi="10.1000/x",
              source="openalex", status=STATUS_FETCHED, references=["10.1/y"])
    assert Paper.from_json(p.to_json()) == p
    assert p.citation_label() == "Lewis 2024"
    assert Paper("x", "t", authors=["Lewis P"], year=2020).citation_label() == "Lewis 2020"   # surname-first form
    assert p.has_full_text


def test_paper_rejects_unknown_status():
    with pytest.raises(ValueError):
        Paper(paper_id="x", title="t", status="downloaded")


def test_answer_citation_integrity():
    a = Answer(text="claim [p#0] other [p#9]", cited_ids=["p#0", "p#9"], retrieved_ids=["p#0", "p#1"],
               model="qwen3:4b", completion_tokens=50, latency_s=2.0)
    assert a.invalid_citations == ["p#9"]
    assert a.citation_integrity == 0.5
    assert a.tokens_per_s == 25.0
    assert Answer(text="", cited_ids=[], retrieved_ids=[], model="m").citation_integrity == 1.0


# ----- schema -----

def _sample_paper(pid="10.48550/arxiv.2005.11401"):
    return Paper(paper_id=pid, title="Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
                 authors=["Patrick Lewis", "Ethan Perez"], year=2020, doi=pid, arxiv_id="2005.11401",
                 source="arxiv", status=STATUS_FETCHED, abstract="RAG combines...", references=["10.1/a"])


def _sample_chunks(pid):
    texts = [
        "Introduction\n\nLarge pre-trained language models store factual knowledge in their parameters.",
        "Methods\n\nWe combine a dense passage retriever with a seq2seq generator.",
        "Results\n\nRAG sets a new state of the art on open-domain question answering benchmarks.",
    ]
    return [Chunk(chunk_id=Chunk.make_id(pid, i), paper_id=pid, ordinal=i,
                  section=t.split("\n")[0], page_start=i + 1, page_end=i + 1, text=t, n_tokens=len(t.split()))
            for i, t in enumerate(texts)]


def test_schema_roundtrip_papers_and_chunks(tmp_path):
    conn = index.connect(tmp_path / "index.db")
    p = _sample_paper()
    index.upsert_paper(conn, p)
    assert index.get_paper(conn, p.paper_id) == p
    assert index.count_papers(conn) == 1

    chunks = _sample_chunks(p.paper_id)
    assert index.replace_chunks(conn, p.paper_id, chunks) == 3
    assert list(index.iter_chunks(conn, p.paper_id)) == chunks
    assert index.get_chunk(conn, chunks[1].chunk_id) == chunks[1]
    assert set(index.get_chunks(conn, [c.chunk_id for c in chunks])) == {c.chunk_id for c in chunks}

    # upsert updates in place, replace_chunks is idempotent
    p2 = Paper.from_dict({**p.to_dict(), "status": STATUS_NO_OA_PDF})
    index.upsert_paper(conn, p2)
    assert index.get_paper(conn, p.paper_id).status == STATUS_NO_OA_PDF
    index.replace_chunks(conn, p.paper_id, chunks[:2])
    assert index.count_chunks(conn) == 2

    index.set_meta(conn, "vector_backend", "numpy")
    assert index.get_meta(conn, "vector_backend") == "numpy"
    conn.close()

    # persists across reconnect
    conn = index.connect(tmp_path / "index.db")
    assert index.count_papers(conn) == 1 and index.count_chunks(conn) == 2
    conn.close()


def test_fts_keyword_query_hits_right_chunk(tmp_path):
    conn = index.connect(tmp_path / "index.db")
    p = _sample_paper()
    index.upsert_paper(conn, p)
    chunks = _sample_chunks(p.paper_id)
    index.replace_chunks(conn, p.paper_id, chunks)
    hits = index.query_fts(conn, "dense passage retriever", k=5)
    assert hits and hits[0][0] == chunks[1].chunk_id
    assert index.query_fts(conn, "", k=5) == []
    assert index.query_fts(conn, 'quote"injection OR x', k=5) == []   # escaped, not a syntax error
    # deleting chunks keeps FTS in sync via triggers
    index.replace_chunks(conn, p.paper_id, [])
    assert index.query_fts(conn, "retriever", k=5) == []
    conn.close()


def test_chunks_cascade_on_paper_delete(tmp_path):
    conn = index.connect(tmp_path / "index.db")
    p = _sample_paper()
    index.upsert_paper(conn, p)
    index.replace_chunks(conn, p.paper_id, _sample_chunks(p.paper_id))
    conn.execute("DELETE FROM papers WHERE paper_id = ?", (p.paper_id,))
    conn.commit()
    assert index.count_chunks(conn) == 0
    conn.close()


def test_extension_probe_returns_bool():
    assert isinstance(index.extension_loading_supported(), bool)


def test_retrieved_wraps_chunk():
    c = _sample_chunks("x")[0]
    r = Retrieved(chunk=c, score=1.0, rank=1, mode="bm25")
    assert json.loads(json.dumps(r.chunk.to_dict()))["chunk_id"] == "x#0"
