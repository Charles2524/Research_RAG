"""Markdown -> chunks: split on headers, subdivide to target tokens with overlap (SPEC 6.4).

Token counts use the embedding model's own tokenizer, loaded from the local Hugging
Face cache only. The tokenizer files are fetched by ``python -m fetch --models``
(network layer); this module never downloads anything (C1).
"""

from __future__ import annotations

import logging
import os
import re
import statistics
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")   # pipeline modules never touch the hub

from huggingface_hub import try_to_load_from_cache  # noqa: E402  (local cache lookup only)
from tokenizers import Tokenizer  # noqa: E402

import index  # noqa: E402
from config import Config, hf_hub_cache  # noqa: E402
from models import Chunk, Paper  # noqa: E402

log = logging.getLogger(__name__)

PAGE_MARK = re.compile(r"^<!-- page:(\d+) -->\s*$", re.M)
HEADING = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
MIN_BODY_TOKENS = 24          # a trailing fragment smaller than this is merged into the previous chunk
_TOKENIZER: dict[str, Tokenizer] = {}


class TokenizerMissing(RuntimeError):
    pass


def get_tokenizer(cfg: Config) -> Tokenizer:
    """Embedding-model tokenizer from the local HF cache (never downloads)."""
    name = cfg.embedding_model
    if name not in _TOKENIZER:
        path = try_to_load_from_cache(name, "tokenizer.json", cache_dir=hf_hub_cache())
        if not isinstance(path, str):
            raise TokenizerMissing(
                f"tokenizer for {name} not in HF cache ({os.environ.get('HF_HOME', 'default')}); "
                "run `python -m fetch --models` first")
        _TOKENIZER[name] = Tokenizer.from_file(path)
    return _TOKENIZER[name]


def count_tokens(cfg: Config, text: str) -> int:
    return len(get_tokenizer(cfg).encode(text, add_special_tokens=False).ids)


# ----- markdown structure -----

@dataclass
class Section:
    title: str
    text: str            # body without the heading line
    page_start: int
    page_end: int
    page_offsets: list[tuple[int, int]]   # (char offset in text, page number) for page changes


def split_sections(md: str) -> list[Section]:
    """Split markdown on headings; track page numbers from ``<!-- page:N -->`` markers."""
    sections: list[Section] = []
    title, lines, page, start_page, offsets, pos = "", [], 1, 1, [], 0

    def flush():
        nonlocal lines, offsets
        text = "\n".join(lines).strip("\n")
        if text.strip():
            sections.append(Section(title=title, text=text, page_start=start_page,
                                    page_end=offsets[-1][1] if offsets else start_page, page_offsets=offsets))
        lines, offsets = [], []

    for raw in md.splitlines():
        m = PAGE_MARK.match(raw)
        if m:
            page = int(m.group(1))
            offsets.append((sum(len(l) + 1 for l in lines), page))
            continue
        h = HEADING.match(raw)
        if h:
            flush()
            title = " ".join(h.group(2).replace("*", "").split())
            start_page = page
            offsets = [(0, page)]
            continue
        if not lines and not offsets:
            offsets = [(0, page)]
            start_page = page
        lines.append(raw.rstrip())
    flush()
    return sections


def _page_at(offsets: list[tuple[int, int]], char: int, default: int) -> int:
    page = default
    for off, p in offsets:
        if off <= char:
            page = p
        else:
            break
    return page


# ----- chunking -----

def chunk_markdown(cfg: Config, paper_id: str, md: str, size_tokens: int | None = None,
                   overlap_tokens: int | None = None) -> list[Chunk]:
    """Header-first split, then token windows with overlap. Section title is prepended to every chunk."""
    size = size_tokens or cfg.chunk_size_tokens
    overlap = cfg.chunk_overlap_tokens if overlap_tokens is None else overlap_tokens
    if not 0 <= overlap < size:
        raise ValueError("overlap must be >= 0 and < size")
    tok = get_tokenizer(cfg)
    chunks: list[Chunk] = []
    ordinal = 0

    for sec in split_sections(md):
        prefix = f"{sec.title}\n\n" if sec.title else ""
        prefix_n = len(tok.encode(prefix, add_special_tokens=False).ids) if prefix else 0
        budget = size - prefix_n
        if budget < MIN_BODY_TOKENS:           # absurdly long heading: keep only the tail of it
            prefix = prefix[-200:]
            prefix_n = len(tok.encode(prefix, add_special_tokens=False).ids)
            budget = max(MIN_BODY_TOKENS, size - prefix_n)
        enc = tok.encode(sec.text, add_special_tokens=False)
        n = len(enc.ids)
        if n == 0:
            continue
        step = max(1, budget - overlap)
        windows: list[tuple[int, int]] = []       # (char_start, char_end)
        t0 = 0
        while t0 < n:
            t1 = min(n, t0 + budget)
            c0, c1 = enc.offsets[t0][0], enc.offsets[t1 - 1][1]
            if t1 < n:                            # snap the cut back to a sentence/paragraph boundary
                snapped = _snap(sec.text, c0, c1, min_frac=0.6)
                if snapped != c1:
                    c1 = snapped
                    t1 = _token_index_at(enc, c1, t0, t1)
            windows.append((c0, c1))
            if t1 >= n:
                break
            t0 = max(t0 + 1, t1 - overlap)
        # merge a tiny trailing fragment into the previous window when it fits
        if len(windows) >= 2:
            last_n = _token_index_at(enc, windows[-1][1], 0, n) - _token_index_at(enc, windows[-1][0], 0, n)
            if last_n < MIN_BODY_TOKENS:
                c0, _ = windows[-2]
                merged_n = n - _token_index_at(enc, c0, 0, n)
                if merged_n <= budget:
                    windows[-2] = (c0, windows[-1][1])
                    windows.pop()
        for c0, c1 in windows:
            body = sec.text[c0:c1].strip()
            if not body:
                continue
            text = prefix + body
            chunks.append(Chunk(
                chunk_id=Chunk.make_id(paper_id, ordinal), paper_id=paper_id, ordinal=ordinal,
                section=sec.title or "(untitled)",
                page_start=_page_at(sec.page_offsets, c0, sec.page_start),
                page_end=_page_at(sec.page_offsets, max(c0, c1 - 1), sec.page_start),
                text=text, n_tokens=len(tok.encode(text, add_special_tokens=False).ids),
            ))
            ordinal += 1
    return chunks


def _snap(text: str, c0: int, c1: int, min_frac: float) -> int:
    """Move a cut point back to the nearest paragraph / sentence end, if one lies in the last 1-min_frac."""
    floor = c0 + int((c1 - c0) * min_frac)
    for pat in (r"\n\n", r"(?<=[.!?])\s"):
        best = None
        for m in re.finditer(pat, text[floor:c1]):
            best = floor + m.end()
        if best and best > floor:
            return best
    return c1


def _token_index_at(enc, char: int, lo: int, hi: int) -> int:
    """Number of tokens (from 0) whose end offset is <= char, searched in [lo, hi]."""
    lo = max(0, lo)
    hi = min(len(enc.ids), hi)
    while lo < hi:
        mid = (lo + hi) // 2
        if enc.offsets[mid][1] <= char:
            lo = mid + 1
        else:
            hi = mid
    return lo


# ----- corpus -----

def describe(chunks: list[Chunk]) -> dict:
    """Count and token distribution, also printed by chunk_all (Phase 2 gate)."""
    if not chunks:
        return {"count": 0}
    ns = sorted(c.n_tokens for c in chunks)
    q = statistics.quantiles(ns, n=20) if len(ns) >= 20 else ns
    return {
        "count": len(ns), "papers": len({c.paper_id for c in chunks}),
        "min": ns[0], "p5": q[0], "p50": statistics.median(ns), "p95": q[-1], "max": ns[-1],
        "mean": round(statistics.fmean(ns), 1),
    }


def chunk_all(cfg: Config, size_tokens: int | None = None, overlap_tokens: int | None = None,
              papers: list[Paper] | None = None) -> list[Chunk]:
    """Chunk every cached .md in cfg.md_dir and store papers + chunks in the index database."""
    from fetch import load_corpus, safe_name   # metadata reader; no network is exercised
    papers = papers if papers is not None else load_corpus(cfg)
    by_stem = {safe_name(p.paper_id): p for p in papers}
    conn = index.connect(cfg.index_path)
    all_chunks: list[Chunk] = []
    missing = 0
    for md_path in sorted(cfg.md_dir.glob("*.md")):
        paper = by_stem.get(md_path.stem)
        if paper is None:
            missing += 1
            log.warning("no metadata for %s; skipping", md_path.name)
            continue
        chunks = chunk_markdown(cfg, paper.paper_id, md_path.read_text(encoding="utf-8"), size_tokens, overlap_tokens)
        index.upsert_paper(conn, paper)
        index.replace_chunks(conn, paper.paper_id, chunks)
        all_chunks.extend(chunks)
    index.set_meta(conn, "chunk_size_tokens", str(size_tokens or cfg.chunk_size_tokens))
    index.set_meta(conn, "chunk_overlap_tokens", str(cfg.chunk_overlap_tokens if overlap_tokens is None else overlap_tokens))
    conn.close()
    stats = describe(all_chunks)
    log.info("chunked %d papers -> %s (skipped %d without metadata)", stats.get("papers", 0), stats, missing)
    print(f"chunks: {stats}")
    return all_chunks


if __name__ == "__main__":
    import sys
    from config import load_config
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    c = load_config()
    size = int(sys.argv[1]) if len(sys.argv) > 1 else None
    chunk_all(c, size)
