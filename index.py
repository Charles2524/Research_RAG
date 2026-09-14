"""SQLite schema and index build/query: papers, chunks, FTS5 and vectors.

Phase 0: schema, connection helpers, paper/chunk round-trip, FTS keyword query.
Phase 3 adds embedding, the sqlite-vec / NumPy vector backend and build_index().
No network client may ever be imported here (C1).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import time
from collections.abc import Iterable, Iterator
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")   # pipeline modules never touch the hub (C1)

import numpy as np  # noqa: E402

from config import Config, hf_hub_cache, peak_rss_mb  # noqa: E402
from models import Chunk, Paper  # noqa: E402

log = logging.getLogger(__name__)
BACKEND_SQLITE_VEC = "sqlite-vec"
BACKEND_NUMPY = "numpy"
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "   # bge v1.5 query prefix
_EMBEDDER: dict[str, object] = {}
_NUMPY_INDEX: dict[str, tuple[list[str], np.ndarray]] = {}

SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    paper_id          TEXT PRIMARY KEY,
    title             TEXT NOT NULL,
    authors           TEXT NOT NULL DEFAULT '[]',   -- JSON list
    year              INTEGER,
    doi               TEXT,
    openalex_id       TEXT,
    arxiv_id          TEXT,
    s2_id             TEXT,
    venue             TEXT,
    abstract          TEXT,
    source            TEXT NOT NULL DEFAULT '',
    pdf_url           TEXT,
    status            TEXT NOT NULL DEFAULT 'pending',
    metadata_resolved INTEGER NOT NULL DEFAULT 1,
    references_json   TEXT NOT NULL DEFAULT '[]'
);
CREATE INDEX IF NOT EXISTS idx_papers_doi ON papers(doi);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id    TEXT PRIMARY KEY,
    paper_id    TEXT NOT NULL REFERENCES papers(paper_id) ON DELETE CASCADE,
    ordinal     INTEGER NOT NULL,
    section     TEXT NOT NULL DEFAULT '',
    page_start  INTEGER NOT NULL,
    page_end    INTEGER NOT NULL,
    n_tokens    INTEGER NOT NULL,
    text        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_paper ON chunks(paper_id, ordinal);

-- External-content FTS5 over chunks; triggers keep it in sync.
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text, section,
    content='chunks', content_rowid='rowid',
    tokenize='porter unicode61'
);
CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, text, section) VALUES (new.rowid, new.text, new.section);
END;
CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text, section)
        VALUES ('delete', old.rowid, old.text, old.section);
END;
CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text, section)
        VALUES ('delete', old.rowid, old.text, old.section);
    INSERT INTO chunks_fts(rowid, text, section) VALUES (new.rowid, new.text, new.section);
END;

-- Dense vectors as raw float32 bytes. sqlite-vec (if loadable) builds its own
-- vec0 table from these in Phase 3; the NumPy fallback reads them directly.
CREATE TABLE IF NOT EXISTS vectors (
    chunk_id  TEXT PRIMARY KEY REFERENCES chunks(chunk_id) ON DELETE CASCADE,
    model     TEXT NOT NULL,
    dim       INTEGER NOT NULL,
    embedding BLOB NOT NULL
);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- Embedding cache keyed by text hash so re-chunking only embeds new texts (SPEC 4.3).
CREATE TABLE IF NOT EXISTS embedding_cache (
    text_hash TEXT NOT NULL,
    model     TEXT NOT NULL,
    dim       INTEGER NOT NULL,
    embedding BLOB NOT NULL,
    PRIMARY KEY (text_hash, model)
);
"""


def connect(path: Path | str) -> sqlite3.Connection:
    """Open (creating if needed) the index database with the schema applied."""
    path = Path(path)
    if str(path) != ":memory:":
        path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    init_schema(conn)
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()


def extension_loading_supported(conn: sqlite3.Connection | None = None) -> bool:
    """Probe whether this CPython build can load SQLite extensions (SPEC 4.2)."""
    own = conn is None
    conn = conn or sqlite3.connect(":memory:")
    try:
        if not hasattr(conn, "enable_load_extension"):
            return False
        conn.enable_load_extension(True)
        conn.enable_load_extension(False)
        return True
    except (AttributeError, sqlite3.OperationalError, sqlite3.NotSupportedError):
        return False
    finally:
        if own:
            conn.close()


# ----- papers -----

def upsert_paper(conn: sqlite3.Connection, paper: Paper) -> None:
    conn.execute(
        """INSERT INTO papers (paper_id, title, authors, year, doi, openalex_id, arxiv_id, s2_id,
                               venue, abstract, source, pdf_url, status, metadata_resolved, references_json)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
           ON CONFLICT(paper_id) DO UPDATE SET
               title=excluded.title, authors=excluded.authors, year=excluded.year, doi=excluded.doi,
               openalex_id=excluded.openalex_id, arxiv_id=excluded.arxiv_id, s2_id=excluded.s2_id,
               venue=excluded.venue, abstract=excluded.abstract, source=excluded.source,
               pdf_url=excluded.pdf_url, status=excluded.status,
               metadata_resolved=excluded.metadata_resolved, references_json=excluded.references_json""",
        (
            paper.paper_id, paper.title, json.dumps(paper.authors, ensure_ascii=False), paper.year,
            paper.doi, paper.openalex_id, paper.arxiv_id, paper.s2_id, paper.venue, paper.abstract,
            paper.source, paper.pdf_url, paper.status, int(paper.metadata_resolved),
            json.dumps(paper.references, ensure_ascii=False),
        ),
    )
    conn.commit()


def _row_to_paper(row: sqlite3.Row) -> Paper:
    return Paper(
        paper_id=row["paper_id"], title=row["title"], authors=json.loads(row["authors"]),
        year=row["year"], doi=row["doi"], openalex_id=row["openalex_id"], arxiv_id=row["arxiv_id"],
        s2_id=row["s2_id"], venue=row["venue"], abstract=row["abstract"], source=row["source"],
        pdf_url=row["pdf_url"], status=row["status"], metadata_resolved=bool(row["metadata_resolved"]),
        references=json.loads(row["references_json"]),
    )


def get_paper(conn: sqlite3.Connection, paper_id: str) -> Paper | None:
    row = conn.execute("SELECT * FROM papers WHERE paper_id = ?", (paper_id,)).fetchone()
    return _row_to_paper(row) if row else None


def iter_papers(conn: sqlite3.Connection) -> Iterator[Paper]:
    for row in conn.execute("SELECT * FROM papers ORDER BY paper_id"):
        yield _row_to_paper(row)


def count_papers(conn: sqlite3.Connection) -> int:
    return conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]


# ----- chunks -----

def replace_chunks(conn: sqlite3.Connection, paper_id: str, chunks: Iterable[Chunk]) -> int:
    """Delete a paper's existing chunks (and vectors, via cascade) and insert the new set."""
    conn.execute("DELETE FROM chunks WHERE paper_id = ?", (paper_id,))
    n = 0
    for c in chunks:
        if c.paper_id != paper_id:
            raise ValueError(f"chunk {c.chunk_id} belongs to {c.paper_id}, not {paper_id}")
        conn.execute(
            """INSERT INTO chunks (chunk_id, paper_id, ordinal, section, page_start, page_end, n_tokens, text)
               VALUES (?,?,?,?,?,?,?,?)""",
            (c.chunk_id, c.paper_id, c.ordinal, c.section, c.page_start, c.page_end, c.n_tokens, c.text),
        )
        n += 1
    conn.commit()
    return n


def _row_to_chunk(row: sqlite3.Row) -> Chunk:
    return Chunk(
        chunk_id=row["chunk_id"], paper_id=row["paper_id"], ordinal=row["ordinal"],
        section=row["section"], page_start=row["page_start"], page_end=row["page_end"],
        text=row["text"], n_tokens=row["n_tokens"],
    )


def get_chunk(conn: sqlite3.Connection, chunk_id: str) -> Chunk | None:
    row = conn.execute("SELECT * FROM chunks WHERE chunk_id = ?", (chunk_id,)).fetchone()
    return _row_to_chunk(row) if row else None


def get_chunks(conn: sqlite3.Connection, chunk_ids: Iterable[str]) -> dict[str, Chunk]:
    ids = list(chunk_ids)
    if not ids:
        return {}
    out: dict[str, Chunk] = {}
    for i in range(0, len(ids), 500):
        batch = ids[i:i + 500]
        q = f"SELECT * FROM chunks WHERE chunk_id IN ({','.join('?' * len(batch))})"
        for row in conn.execute(q, batch):
            out[row["chunk_id"]] = _row_to_chunk(row)
    return out


def iter_chunks(conn: sqlite3.Connection, paper_id: str | None = None) -> Iterator[Chunk]:
    if paper_id is None:
        rows = conn.execute("SELECT * FROM chunks ORDER BY paper_id, ordinal")
    else:
        rows = conn.execute("SELECT * FROM chunks WHERE paper_id = ? ORDER BY ordinal", (paper_id,))
    for row in rows:
        yield _row_to_chunk(row)


def count_chunks(conn: sqlite3.Connection) -> int:
    return conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]


# ----- lexical query -----

def query_fts(conn: sqlite3.Connection, query: str, k: int = 10) -> list[tuple[str, float]]:
    """BM25 keyword search. Returns [(chunk_id, score)] best first; score is -bm25 (higher = better)."""
    match = _fts_escape(query)
    if not match:
        return []
    rows = conn.execute(
        """SELECT c.chunk_id AS chunk_id, -bm25(chunks_fts) AS score
           FROM chunks_fts JOIN chunks c ON c.rowid = chunks_fts.rowid
           WHERE chunks_fts MATCH ? ORDER BY bm25(chunks_fts) LIMIT ?""",
        (match, k),
    ).fetchall()
    return [(r["chunk_id"], float(r["score"])) for r in rows]


def _fts_escape(query: str) -> str:
    """Turn free text into a safe FTS5 query: each token double-quoted, implicit AND."""
    tokens = [t.replace('"', '""') for t in query.split() if t.strip()]
    return " ".join(f'"{t}"' for t in tokens)


# ----- meta -----

def set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute("INSERT INTO meta(key, value) VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                 (key, value))
    conn.commit()


def get_meta(conn: sqlite3.Connection, key: str, default: str | None = None) -> str | None:
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else default


# ===== Phase 3: embeddings and vector index =====

def probe_vector_backend(conn: sqlite3.Connection) -> str:
    """Try to load sqlite-vec; fall back to the NumPy flat index (SPEC 4.2). Result stored in meta."""
    backend = BACKEND_NUMPY
    reason = ""
    if extension_loading_supported(conn):
        try:
            import sqlite_vec
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.enable_load_extension(False)
            ver = conn.execute("SELECT vec_version()").fetchone()[0]
            backend, reason = BACKEND_SQLITE_VEC, f"vec_version {ver}"
        except Exception as e:                       # ImportError, OperationalError
            reason = f"sqlite-vec unavailable: {type(e).__name__}: {e}"
    else:
        reason = "sqlite3 extension loading not supported by this CPython build"
    set_meta(conn, "vector_backend", backend)
    set_meta(conn, "vector_backend_reason", reason)
    log.info("vector backend: %s (%s)", backend, reason)
    return backend


def _load_vec_extension(conn: sqlite3.Connection) -> None:
    import sqlite_vec
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)


def get_embedder(cfg: Config):
    """sentence-transformers model from the local HF cache only (never downloads)."""
    name = cfg.embedding_model
    if name not in _EMBEDDER:
        from sentence_transformers import SentenceTransformer
        try:
            model = SentenceTransformer(name, cache_folder=str(hf_hub_cache()), device="cpu",
                                        local_files_only=True)
        except Exception as e:
            raise RuntimeError(f"embedding model {name} not in local HF cache ({hf_hub_cache()}); "
                               f"run `python -m fetch --models` first ({type(e).__name__}: {e})") from e
        get_dim = getattr(model, "get_embedding_dimension", None) or model.get_sentence_embedding_dimension
        if get_dim() != cfg.embedding_dim:
            raise RuntimeError(f"embedding.dim={cfg.embedding_dim} but {name} produces {get_dim()}")
        _EMBEDDER[name] = model
    return _EMBEDDER[name]


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _to_blob(vec: np.ndarray) -> bytes:
    return np.asarray(vec, dtype=np.float32).tobytes()


def _from_blob(blob: bytes, dim: int) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32, count=dim)


def embed_texts(cfg: Config, conn: sqlite3.Connection, texts: list[str], batch_size: int = 32) -> tuple[np.ndarray, int]:
    """Embed texts (normalized), using and filling the embedding cache. Returns (matrix, n_newly_embedded)."""
    model_name = cfg.embedding_model
    hashes = [_text_hash(t) for t in texts]
    out = np.zeros((len(texts), cfg.embedding_dim), dtype=np.float32)
    missing: list[int] = []
    for i in range(0, len(hashes), 500):
        batch = hashes[i:i + 500]
        rows = conn.execute(
            f"SELECT text_hash, embedding FROM embedding_cache WHERE model = ? AND text_hash IN ({','.join('?' * len(batch))})",
            (model_name, *batch)).fetchall()
        found = {r["text_hash"]: r["embedding"] for r in rows}
        for j, h in enumerate(batch):
            if h in found:
                out[i + j] = _from_blob(found[h], cfg.embedding_dim)
            else:
                missing.append(i + j)
    if missing:
        model = get_embedder(cfg)
        vecs = model.encode([texts[i] for i in missing], batch_size=batch_size, normalize_embeddings=True,
                            convert_to_numpy=True, show_progress_bar=False)
        conn.executemany(
            "INSERT OR REPLACE INTO embedding_cache(text_hash, model, dim, embedding) VALUES (?,?,?,?)",
            [(hashes[i], model_name, cfg.embedding_dim, _to_blob(v)) for i, v in zip(missing, vecs)])
        conn.commit()
        for i, v in zip(missing, vecs):
            out[i] = v
    return out, len(missing)


def embed_query(cfg: Config, conn: sqlite3.Connection, query: str) -> np.ndarray:
    vec, _ = embed_texts(cfg, conn, [QUERY_INSTRUCTION + query])
    return vec[0]


def build_index(cfg: Config, conn: sqlite3.Connection) -> dict:
    """Embed every chunk lacking a vector and (re)build the vector index. FTS5 is maintained by triggers."""
    t0 = time.monotonic()
    backend = probe_vector_backend(conn)
    model_name = cfg.embedding_model
    rows = conn.execute(
        """SELECT c.chunk_id, c.text FROM chunks c
           LEFT JOIN vectors v ON v.chunk_id = c.chunk_id AND v.model = ?
           WHERE v.chunk_id IS NULL ORDER BY c.paper_id, c.ordinal""", (model_name,)).fetchall()
    n_new = 0
    t_embed = time.monotonic()
    if rows:
        ids = [r["chunk_id"] for r in rows]
        vecs, n_new = embed_texts(cfg, conn, [r["text"] for r in rows])
        conn.executemany("INSERT OR REPLACE INTO vectors(chunk_id, model, dim, embedding) VALUES (?,?,?,?)",
                         [(cid, model_name, cfg.embedding_dim, _to_blob(v)) for cid, v in zip(ids, vecs)])
        conn.commit()
    embed_s = time.monotonic() - t_embed
    # drop vectors for chunks that no longer exist (re-chunk) or other models
    conn.execute("DELETE FROM vectors WHERE model != ? OR chunk_id NOT IN (SELECT chunk_id FROM chunks)", (model_name,))
    conn.commit()
    if backend == BACKEND_SQLITE_VEC:
        _build_vec_table(cfg, conn)
    _NUMPY_INDEX.clear()
    conn.execute("VACUUM")
    stats = index_stats(cfg, conn)
    stats.update({"index_build_s": round(time.monotonic() - t0, 2), "embed_s": round(embed_s, 2),
                  "n_embedded": n_new, "n_vectorized": len(rows), "peak_rss_mb": round(peak_rss_mb(), 1)})
    set_meta(conn, "embedding_model", model_name)
    log.info("index built: %s", stats)
    return stats


def _build_vec_table(cfg: Config, conn: sqlite3.Connection) -> None:
    _load_vec_extension(conn)
    conn.execute("DROP TABLE IF EXISTS vec_chunks")
    conn.execute(f"CREATE VIRTUAL TABLE vec_chunks USING vec0(chunk_id TEXT PRIMARY KEY, "
                 f"embedding float[{cfg.embedding_dim}] distance_metric=cosine)")
    rows = conn.execute("SELECT chunk_id, embedding FROM vectors").fetchall()
    conn.executemany("INSERT INTO vec_chunks(chunk_id, embedding) VALUES (?, ?)",
                     [(r["chunk_id"], r["embedding"]) for r in rows])
    conn.commit()


def _numpy_index(cfg: Config, conn: sqlite3.Connection) -> tuple[list[str], np.ndarray]:
    key = str(cfg.index_path)
    if key not in _NUMPY_INDEX:
        rows = conn.execute("SELECT chunk_id, embedding FROM vectors WHERE model = ? ORDER BY chunk_id",
                            (cfg.embedding_model,)).fetchall()
        ids = [r["chunk_id"] for r in rows]
        mat = (np.vstack([_from_blob(r["embedding"], cfg.embedding_dim) for r in rows])
               if rows else np.zeros((0, cfg.embedding_dim), dtype=np.float32))
        _NUMPY_INDEX[key] = (ids, mat)
    return _NUMPY_INDEX[key]


def query_vector(cfg: Config, conn: sqlite3.Connection, query_vec: np.ndarray, k: int = 10) -> list[tuple[str, float]]:
    """Dense search. Returns [(chunk_id, cosine_similarity)] best first, via the probed backend."""
    backend = get_meta(conn, "vector_backend") or BACKEND_NUMPY
    q = np.asarray(query_vec, dtype=np.float32)
    if backend == BACKEND_SQLITE_VEC:
        try:
            _load_vec_extension(conn)
            rows = conn.execute(
                "SELECT chunk_id, distance FROM vec_chunks WHERE embedding MATCH ? AND k = ? ORDER BY distance",
                (_to_blob(q), k)).fetchall()
            return [(r["chunk_id"], 1.0 - float(r["distance"])) for r in rows]
        except Exception as e:
            log.warning("sqlite-vec query failed (%s); using numpy fallback", e)
    ids, mat = _numpy_index(cfg, conn)
    if not ids:
        return []
    sims = mat @ q
    top = np.argpartition(-sims, min(k, len(ids) - 1))[:k]
    top = top[np.argsort(-sims[top])]
    return [(ids[i], float(sims[i])) for i in top]


def index_stats(cfg: Config, conn: sqlite3.Connection) -> dict:
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    size = cfg.index_path.stat().st_size if cfg.index_path.exists() else 0
    return {
        "vector_backend": get_meta(conn, "vector_backend") or "unprobed",
        "vector_backend_reason": get_meta(conn, "vector_backend_reason") or "",
        "n_papers": count_papers(conn), "n_chunks": count_chunks(conn),
        "n_vectors": conn.execute("SELECT COUNT(*) FROM vectors").fetchone()[0],
        "n_cached_embeddings": conn.execute("SELECT COUNT(*) FROM embedding_cache").fetchone()[0],
        "index_size_mb": round(size / (1024 * 1024), 2),
        "embedding_model": cfg.embedding_model,
    }


if __name__ == "__main__":
    from config import load_config
    from evaluate import log_run
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = load_config()
    conn = connect(cfg.index_path)
    stats = build_index(cfg, conn)
    print(f"vector backend: {stats['vector_backend']} ({stats['vector_backend_reason']})")
    print(f"papers={stats['n_papers']} chunks={stats['n_chunks']} vectors={stats['n_vectors']} "
          f"index={stats['index_size_mb']} MB build={stats['index_build_s']} s embed={stats['embed_s']} s "
          f"new_embeddings={stats['n_embedded']} peak_rss={stats['peak_rss_mb']} MB")
    log_run(cfg, {"phase": 3, "run_name": "index_build", "chunk_size_tokens": get_meta(conn, "chunk_size_tokens"),
                  "chunk_overlap_tokens": get_meta(conn, "chunk_overlap_tokens"), **{k: v for k, v in stats.items()
                  if k in ("vector_backend", "n_papers", "n_chunks", "index_size_mb", "index_build_s", "embed_s",
                           "n_embedded", "embedding_model")}})
    print(f"logged to {cfg.runs_csv}")
    conn.close()
