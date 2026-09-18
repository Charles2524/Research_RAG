"""Web UI server: local endpoints, streaming ask, offline refusal of the network action."""

from __future__ import annotations

import json

import pytest
from starlette.testclient import TestClient

import generate
import server
from config import load_config


@pytest.fixture(scope="module")
def client():
    with TestClient(server.app) as c:      # runs the startup warm-up thread too
        yield c


def _events(resp):
    out = []
    for block in resp.text.split("\n\n"):
        line = next((l for l in block.split("\n") if l.startswith("data: ")), None)
        if line:
            out.append(json.loads(line[6:]))
    return out


def test_home_and_static(client):
    r = client.get("/")
    assert r.status_code == 200 and "Corpus" in r.text and "/ui/app.js" in r.text
    assert client.get("/ui/app.css").status_code == 200
    f = client.get("/ui/fonts/fonts.css")
    assert f.status_code == 200 and "IBM Plex Sans" in f.text
    assert "fonts.googleapis" not in r.text and "fonts.gstatic" not in f.text      # offline-first: no font CDN


def test_status_and_papers(client):
    s = client.get("/api/status").json()
    for key in ("offline", "ollama", "models", "n_papers", "n_chunks", "vector_backend", "modes"):
        assert key in s
    assert s["n_papers"] >= 20 and s["n_fulltext"] >= 20
    p = client.get("/api/papers").json()["papers"]
    assert len(p) == s["n_papers"]
    full = [x for x in p if x["has_full_text"]]
    assert full and all(x["n_chunks"] > 0 for x in full)
    assert all("label" in x and x["title"] for x in p)


def test_runs_log(client):
    r = client.get("/api/runs?limit=5").json()
    assert "runs" in r and r["total"] >= len(r["runs"]) and len(r["runs"]) <= 5
    assert all("timestamp" in row for row in r["runs"])
    assert client.get("/api/runs?limit=0").json()["runs"] == client.get("/api/runs?limit=1").json()["runs"]


def test_chunk_lookup(client):
    p = next(x for x in client.get("/api/papers").json()["papers"] if x["has_full_text"])
    from urllib.parse import quote
    r = client.get(f"/api/chunk/{quote(p['paper_id'] + '#0', safe='')}")     # '#' must be percent-encoded
    assert r.status_code == 200 and r.json()["paper_id"] == p["paper_id"]
    assert client.get("/api/chunk/nope%2399").status_code == 404


def test_ask_validation(client):
    assert client.post("/api/ask", json={"question": ""}).status_code == 400
    assert client.post("/api/ask", json={"question": "x", "mode": "sparse"}).status_code == 400


def test_discover_refuses_when_offline(monkeypatch):
    monkeypatch.setenv("OFFLINE", "1")
    server._CFG = None
    try:
        with TestClient(server.app) as c:
            r = c.post("/api/discover", json={"query": "x", "limit": 1})
            assert r.status_code == 409 and "Offline" in r.json()["error"]
            assert c.get("/api/status").json()["offline"] is True
    finally:
        server._CFG = None


def test_ask_streams_sources_tokens_and_done(client):
    cfg = load_config()
    try:
        generate.ensure_model(cfg, cfg.llm_fallback_model)
    except generate.GenerationError as e:
        pytest.skip(f"UNVERIFIED: {e}")
    with client.stream("POST", "/api/ask", json={"question": "What does the RAG model combine a seq2seq model with?",
                                                 "mode": "hybrid", "k": 6, "model": cfg.llm_fallback_model}) as r:
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
        r.read()
    evs = _events(r)
    kinds = [e["type"] for e in evs]
    assert kinds[0] == "sources" and "done" in kinds and "token" in kinds
    src = evs[0]["sources"]
    assert len(src) == 6 and [s["id"] for s in src if s["in_context"]] == [f"S{i}" for i in range(1, 1 + sum(s["in_context"] for s in src))]
    done = next(e for e in evs if e["type"] == "done")
    assert done["answer"] and done["integrity"] == 1.0 and done["invalid"] == []
    assert "".join(e["delta"] for e in evs if e["type"] == "token").strip() == done["answer"]
    assert done["tokens_per_s"] > 0 and done["completion_tokens"] > 0
