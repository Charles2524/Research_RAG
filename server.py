"""Local API + static server for the web UI (ui/, "Corpus"). Run: ``python server.py`` -> http://127.0.0.1:8765

Same modules as the Streamlit page, same boundary: /api/ask and /api/papers are local-only;
/api/discover is the single explicit network action (C2) and refuses in offline mode (C3).
Streaming uses server-sent events over a POST body so the 1-2 minute CPU generation reads as
progress. No new dependencies: starlette + uvicorn ship with streamlit.
"""

from __future__ import annotations

import contextlib
import json
import logging
import mimetypes
import time
from pathlib import Path

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, StreamingResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

import chunk as chunk_mod
import fetch
import generate
import index
import parse
import retrieve
from config import RETRIEVAL_MODES, Config, load_config
from sources import SourceUnavailable

log = logging.getLogger(__name__)
mimetypes.add_type("font/woff2", ".woff2")      # Windows' registry lacks it; browsers warn on octet-stream fonts
ROOT = Path(__file__).resolve().parent
UI_DIR = ROOT / "ui"
_CFG: Config | None = None


CORPORA_ROOT = ROOT                              # sibling corpora live in <root>/data_<slug>/ (tests point this at tmp)
ACTIVE_FILE = ROOT / ".active_corpus"           # remembers the chosen corpus across restarts (gitignored)


def cfg() -> Config:
    global _CFG
    if _CFG is None:
        c = load_config()
        if ACTIVE_FILE.exists():
            d = CORPORA_ROOT / ACTIVE_FILE.read_text(encoding="utf-8").strip()
            if d.is_dir() and d.resolve() != c.data_dir.resolve():
                c = _corpus_cfg(c, d)
        _CFG = c
    return _CFG


# ----- corpora: one folder per research topic (pdfs, md, meta, cache, index.db) -----

def _slug(name: str) -> str:
    import re
    s = re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")
    return s[:40] or "corpus"


def _corpus_meta(d: Path, default_query: str) -> dict:
    f = d / "corpus.json"
    if f.exists():
        try:
            return json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
    return {"name": d.name if d.name != "data" else "Default corpus", "query": default_query}


def _corpus_cfg(base: Config, d: Path) -> Config:
    meta = _corpus_meta(d, base.fetch_query)
    c = base.with_overrides(data_dir=d, fetch_query=str(meta.get("query") or base.fetch_query))
    c.ensure_dirs()
    return c


def _corpus_dirs(base: Config) -> list[Path]:
    dirs = {base.data_dir.resolve()}
    for d in CORPORA_ROOT.glob("data_*"):
        if d.is_dir():
            dirs.add(d.resolve())
    return sorted(dirs, key=lambda p: (p.resolve() != base.data_dir.resolve(), p.name))


def _corpora_payload() -> dict:
    base = load_config()
    active = cfg().data_dir.resolve()
    out = []
    for d in _corpus_dirs(base):
        meta = _corpus_meta(d, base.fetch_query)
        n_papers = n_fulltext = n_chunks = 0
        meta_dir = d / "meta"
        if meta_dir.is_dir():
            for f in meta_dir.glob("*.json"):
                n_papers += 1
                try:
                    if json.loads(f.read_text(encoding="utf-8"))["paper"].get("status") == "fetched":
                        n_fulltext += 1
                except (OSError, ValueError, KeyError):
                    pass
        db = d / "index.db"
        if db.exists():
            conn = index.connect(db)
            try:
                n_chunks = index.count_chunks(conn)
            finally:
                conn.close()
        out.append({"id": d.name, "name": meta.get("name") or d.name, "query": meta.get("query") or "",
                    "created": meta.get("created"), "n_papers": n_papers, "n_fulltext": n_fulltext, "n_chunks": n_chunks,
                    "index_size_mb": round(db.stat().st_size / 1e6, 2) if db.exists() else 0.0,
                    "active": d.resolve() == active})
    return {"active": cfg().data_dir.name, "corpora": out}


def corpora(request: Request) -> JSONResponse:
    return JSONResponse(_corpora_payload())


async def switch_corpus(request: Request) -> JSONResponse:
    """Create (if new) and activate a corpus folder. Body: {"id": "data_x"} to switch, or {"name", "query"} to create."""
    global _CFG
    body = await request.json()
    base = load_config()
    if body.get("id"):
        d = CORPORA_ROOT / str(body["id"])
        if d.name == base.data_dir.name:
            d = base.data_dir
        if not d.is_dir() or (d.resolve() != base.data_dir.resolve() and not d.name.startswith("data_")):
            return JSONResponse({"error": f"no such corpus: {body['id']}"}, status_code=404)
    else:
        name = (body.get("name") or "").strip()
        query = (body.get("query") or "").strip()
        if not name or not query:
            return JSONResponse({"error": "name and query are required"}, status_code=400)
        d = CORPORA_ROOT / f"data_{_slug(name)}"
        if d.exists():
            return JSONResponse({"error": f"a corpus folder named {d.name} already exists"}, status_code=409)
        d.mkdir(parents=True)
        (d / "corpus.json").write_text(json.dumps({"name": name, "query": query, "created": time.strftime("%Y-%m-%d")},
                                                  indent=2), encoding="utf-8")
    _CFG = _corpus_cfg(base, d)
    ACTIVE_FILE.write_text(d.name, encoding="utf-8")
    log.info("corpus -> %s (%s)", d.name, _CFG.fetch_query)
    return JSONResponse(_corpora_payload())


def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


def _papers_by_id(c: Config) -> dict:
    return {p.paper_id: p for p in fetch.load_corpus(c)}


# ----- read-only, local -----

def status(request: Request) -> JSONResponse:      # sync: runs on the thread pool, never blocks the loop
    c = cfg()
    conn = index.connect(c.index_path)
    try:
        stats = index.index_stats(c, conn)
        per_paper = dict(conn.execute("SELECT paper_id, COUNT(*) FROM chunks GROUP BY paper_id").fetchall())
    finally:
        conn.close()
    papers = fetch.load_corpus(c)
    try:
        models = generate.available_models(c)
        ollama = "up"
    except generate.GenerationError as e:
        models, ollama = [], f"down: {e}"
    return JSONResponse({
        "offline": c.offline, "ollama": ollama, "models": models,
        "default_model": c.llm_model, "fallback_model": c.llm_fallback_model,
        "retrieval_mode": c.retrieval_mode, "rerank": c.rerank, "k": c.k, "context_chunks": c.context_chunks,
        "n_papers": len(papers), "n_fulltext": sum(1 for p in papers if p.has_full_text),
        "n_chunks": stats["n_chunks"], "vector_backend": stats["vector_backend"],
        "index_size_mb": stats["index_size_mb"], "embedding_model": c.embedding_model,
        "contact_email_set": bool(c.contact_email), "modes": list(RETRIEVAL_MODES),
        "chunks_per_paper": per_paper, "fetch_query": c.fetch_query,
        "corpus": c.data_dir.name, "corpus_name": _corpus_meta(c.data_dir, c.fetch_query).get("name"),
    })


def runs(request: Request) -> JSONResponse:
    """Latest rows of results/runs.csv (the metrics log every phase writes to), newest first."""
    import csv
    c = cfg()
    limit = max(1, min(int(request.query_params.get("limit") or 40), 500))
    rows: list[dict] = []
    if c.runs_csv.exists():
        with c.runs_csv.open(newline="", encoding="utf-8") as fh:
            rows = [{k: v for k, v in r.items() if v not in ("", None)} for r in csv.DictReader(fh)]
    return JSONResponse({"runs": rows[::-1][:limit], "total": len(rows), "path": str(c.runs_csv)})


def papers(request: Request) -> JSONResponse:
    c = cfg()
    conn = index.connect(c.index_path)
    try:
        per_paper = dict(conn.execute("SELECT paper_id, COUNT(*) FROM chunks GROUP BY paper_id").fetchall())
    finally:
        conn.close()
    out = []
    for p in sorted(fetch.load_corpus(c), key=lambda p: (-(p.year or 0), p.title.lower())):
        out.append({"paper_id": p.paper_id, "title": p.title, "authors": p.authors, "year": p.year,
                    "venue": p.venue, "doi": p.doi, "status": p.status, "has_full_text": p.has_full_text,
                    "metadata_resolved": p.metadata_resolved, "n_chunks": per_paper.get(p.paper_id, 0),
                    "abstract": (p.abstract or "")[:600], "label": p.citation_label()})
    return JSONResponse({"papers": out})


def chunk_by_id(request: Request) -> JSONResponse:
    c = cfg()
    conn = index.connect(c.index_path)
    try:
        ch = index.get_chunk(conn, request.path_params["chunk_id"])
    finally:
        conn.close()
    if not ch:
        return JSONResponse({"error": "no such chunk"}, status_code=404)
    return JSONResponse(ch.to_dict())


def _source_record(i: int, r, papers: dict, in_context: bool) -> dict:
    p = papers.get(r.chunk.paper_id)
    return {
        "id": f"S{i}" if in_context else None, "rank": r.rank, "score": round(r.score, 4),
        "chunk_id": r.chunk.chunk_id, "paper_id": r.chunk.paper_id,
        "title": p.title if p else r.chunk.paper_id, "label": p.citation_label() if p else r.chunk.paper_id,
        "year": p.year if p else None, "section": r.chunk.section,
        "pages": [r.chunk.page_start, r.chunk.page_end], "n_tokens": r.chunk.n_tokens,
        "text": r.chunk.text, "in_context": in_context,
    }


async def ask(request: Request) -> StreamingResponse:
    body = await request.json()
    question = (body.get("question") or "").strip()
    if not question:
        return JSONResponse({"error": "question is required"}, status_code=400)
    c = cfg()
    mode = body.get("mode") or c.retrieval_mode
    if mode not in RETRIEVAL_MODES:
        return JSONResponse({"error": f"mode must be one of {RETRIEVAL_MODES}"}, status_code=400)
    rerank = bool(body.get("rerank", c.rerank))
    k = int(body.get("k") or c.k)
    model = body.get("model") or c.llm_model
    n_ctx = int(body.get("context_chunks") or c.context_chunks)
    run_cfg = c.with_overrides(context_chunks=n_ctx)

    def events():
        t0 = time.perf_counter()
        try:
            hits = retrieve.retrieve(run_cfg, question, mode=mode, k=k, rerank_=rerank)
            t1 = time.perf_counter()
            papers = _papers_by_id(run_cfg)
            context_ids = {r.chunk.chunk_id for r in generate.assemble_context(run_cfg, hits)}
            log.info("ask: retrieve %.3fs, papers+context %.3fs (%s, k=%d, rerank=%s)",
                     t1 - t0, time.perf_counter() - t1, mode, k, rerank)
            i = 0
            sources = []
            for r in hits:
                in_ctx = r.chunk.chunk_id in context_ids
                if in_ctx:
                    i += 1
                sources.append(_source_record(i, r, papers, in_ctx))
            yield _sse({"type": "sources", "sources": sources, "retrieval_s": round(time.perf_counter() - t0, 3),
                        "mode": mode + ("+rerank" if rerank else "")})
            if rerank:
                retrieve.unload_reranker()
            for kind, payload in generate.generate_stream(run_cfg, question, hits, model=model, papers=papers):
                if kind == "answer":
                    a = payload
                    yield _sse({"type": "done", "answer": a.text, "cited": a.cited_ids, "invalid": a.invalid_citations,
                                "integrity": a.citation_integrity, "model": a.model, "latency_s": a.latency_s,
                                "eval_s": a.eval_s, "prompt_eval_s": a.prompt_eval_s, "load_s": a.load_s,
                                "prompt_tokens": a.prompt_tokens, "completion_tokens": a.completion_tokens,
                                "tokens_per_s": round(a.tokens_per_s, 1), "think": a.think,
                                "thinking_chars": a.thinking_chars})
                else:
                    yield _sse({"type": kind, "delta": payload})
        except generate.GenerationError as e:
            yield _sse({"type": "error", "message": str(e)})
        except Exception as e:                       # surface, never hide
            log.exception("ask failed")
            yield _sse({"type": "error", "message": f"{type(e).__name__}: {e}"})

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ----- the one network action (C2) -----

async def discover(request: Request):
    body = await request.json()
    c = cfg()
    if c.offline:
        return JSONResponse({"error": "Offline mode is on (OFFLINE=1). Discovery is disabled; asking works unchanged."},
                            status_code=409)
    if not c.contact_email:
        return JSONResponse({"error": "CONTACT_EMAIL is not set in .env; scholarly APIs require a contact address."},
                            status_code=409)
    query = (body.get("query") or c.fetch_query).strip()
    limit = max(1, min(int(body.get("limit") or 10), 100))

    def events():
        try:
            yield _sse({"type": "step", "message": f"Searching OpenAlex, arXiv, Europe PMC and PubMed for “{query}”…"})
            found = fetch.discover(c, query, limit)
            yield _sse({"type": "step", "message": f"{len(found)} papers after de-duplication. Resolving open-access PDFs…"})
            done = fetch.download(c, found)
            counts: dict[str, int] = {}
            for p in done:
                counts[p.status] = counts.get(p.status, 0) + 1
            yield _sse({"type": "step", "message": "Download: " + ", ".join(f"{v} {k.replace('_', ' ')}" for k, v in counts.items())})
            written = parse.parse_all(c)
            yield _sse({"type": "step", "message": f"Parsed {len(written)} new PDFs to markdown."})
            chunks = chunk_mod.chunk_all(c)
            yield _sse({"type": "step", "message": f"{len(chunks)} chunks in the index. Embedding new chunks…"})
            conn = index.connect(c.index_path)
            try:
                stats = index.build_index(c, conn)
            finally:
                conn.close()
            retrieve.close_conn(c)
            yield _sse({"type": "done", "message": f"Ready: {stats['n_vectors']} vectors on {stats['vector_backend']}, "
                                                    f"{stats['n_embedded']} newly embedded in {stats['embed_s']} s.",
                        "counts": counts})
        except SourceUnavailable as e:
            yield _sse({"type": "error", "message": str(e)})
        except Exception as e:
            log.exception("discover failed")
            yield _sse({"type": "error", "message": f"{type(e).__name__}: {e}"})

    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


def home(request: Request) -> FileResponse:
    return FileResponse(UI_DIR / "index.html")


def _warm_up() -> None:
    """Load the embedder (and the fast model) once at startup so the first question is not the slow one."""
    import threading

    def run():
        c = cfg()
        try:
            import torch
            torch.set_num_threads(2)        # a query embedding is ~40 ms; fewer threads avoids stalling behind Ollama
        except Exception:
            pass
        try:
            import uuid
            conn = retrieve.get_conn(c)
            if index.count_chunks(conn):
                t = time.perf_counter()
                index.embed_texts(c, conn, [f"warm-up {uuid.uuid4().hex}"])    # unique text: a real encode, not a cache hit
                log.info("embedder warm (%.1f s)", time.perf_counter() - t)
        except Exception as e:
            log.warning("embedder warm-up skipped: %s", e)
        try:
            generate.warm_up(c, c.llm_fallback_model)
            log.info("%s warm", c.llm_fallback_model)
        except generate.GenerationError as e:
            log.warning("model warm-up skipped: %s", e)

    threading.Thread(target=run, name="warm-up", daemon=True).start()


@contextlib.asynccontextmanager
async def _lifespan(app):
    _warm_up()
    yield


app = Starlette(lifespan=_lifespan, routes=[
    Route("/", home),
    Route("/api/status", status),
    Route("/api/papers", papers),
    Route("/api/runs", runs),
    Route("/api/corpora", corpora),
    Route("/api/corpus", switch_corpus, methods=["POST"]),
    Route("/api/chunk/{chunk_id:path}", chunk_by_id),
    Route("/api/ask", ask, methods=["POST"]),
    Route("/api/discover", discover, methods=["POST"]),
    Mount("/ui", StaticFiles(directory=str(UI_DIR)), name="ui"),
])


if __name__ == "__main__":
    import sys
    import uvicorn
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    print(f"Local RAG Research Companion UI -> http://127.0.0.1:{port}  (loopback only)")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
