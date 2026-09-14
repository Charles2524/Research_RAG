"""SQLite schema and index build/query: papers, chunks, FTS5 and vectors.

Phase 0: schema, connection helpers, paper/chunk round-trip, FTS keyword query.
Phase 3 adds embedding, the sqlite-vec / NumPy vector backend and build_index().
No network client may ever be imported here (C1).
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Iterator
from pathlib import Path

from models import Chunk, Paper

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
