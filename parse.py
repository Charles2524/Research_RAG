"""PDF -> markdown, deterministic cleanup, parse cache (SPEC 6.1-6.3).

Output: data/md/<safe_id>.md holding the chunkable body with ``<!-- page:N -->`` markers.
The References section is excised and stored in the paper's meta JSON together with
the DOIs found in it. Parse only when the .md is missing or older than the PDF.
No network client may be imported here (C1); metadata lookup for user-supplied PDFs
lives in fetch.py and runs before parsing.
"""

from __future__ import annotations

import json
import logging
import math
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from pathlib import Path
from typing import Protocol

from config import Config
from models import STATUS_FETCHED, Paper

log = logging.getLogger(__name__)

PAGE_MARK_FMT = "<!-- page:{n} -->"
MIN_CHARS_PER_PAGE = 200      # average below this => scanned / image-only PDF, excluded (SPEC 6.3)
MIN_PAGE_CHARS = 80           # pages with fewer word characters are dropped (figure-only pages)
EDGE_LINES = 3                # lines at top/bottom of a page considered for header/footer detection
DOI_RE = re.compile(r"\b10\.\d{4,9}/[^\s\"'<>\]\)\},;]+")
REF_HEADING = re.compile(
    r"^\s*(?:#{1,6}\s*)?(?:\*\*|__)?\s*(?:[0-9IVX]+\.?\s*)?"
    r"(references?|bibliography|works cited|reference list|literature cited)"
    r"\s*(?:\*\*|__)?\s*:?\s*$", re.I)
APPENDIX_HEADING = re.compile(r"^#{1,6}\s*(?:\*\*)?\s*(appendix|appendices|supplementary|supplemental)\b", re.I | re.M)
PAGE_NUMBER_LINE = re.compile(r"^\s*(?:page\s+)?\d{1,4}(?:\s*(?:of|/)\s*\d{1,4})?\s*$|^\s*[ivxlc]{1,6}\s*$", re.I)


class Parser(Protocol):
    name: str

    def parse(self, pdf_path: Path) -> list[tuple[int, str]]:
        """Return [(page_number, markdown)] for every page."""


class PyMuPDF4LLMParser:
    name = "pymupdf4llm"

    def parse(self, pdf_path: Path) -> list[tuple[int, str]]:
        import pymupdf4llm  # lazy: keeps import-time free of the native library

        pages = pymupdf4llm.to_markdown(str(pdf_path), page_chunks=True, show_progress=False)
        return [(i + 1, (p.get("text") or "")) for i, p in enumerate(pages)]


def get_parser(cfg: Config) -> Parser:
    if cfg.parser == "pymupdf4llm":
        return PyMuPDF4LLMParser()
    raise ValueError(f"unknown parser {cfg.parser!r}")


# ----- title extraction for user-supplied PDFs (SPEC 6.3 step 1) -----

def extract_title(pdf_path: Path) -> str | None:
    """Largest-font text on the first page above the abstract; None if nothing usable."""
    import pymupdf

    with pymupdf.open(str(pdf_path)) as doc:
        if doc.page_count == 0:
            return None
        page = doc[0]
        height = page.rect.height
        lines: list[tuple[float, float, str]] = []      # (size, y, text)
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                if abs(line.get("dir", (1, 0))[0]) < 0.9:       # rotated (arXiv watermark)
                    continue
                spans = [s for s in line.get("spans", []) if s.get("text", "").strip()]
                if not spans:
                    continue
                size = max(s["size"] for s in spans)
                y = line["bbox"][1]
                text = " ".join(s["text"].strip() for s in spans)
                if y < height * 0.6 and not re.match(r"^(abstract|arxiv:)", text, re.I):
                    lines.append((size, y, text))
    if not lines:
        return None
    top = max(s for s, _, _ in lines)
    title_lines = [t for s, _, t in sorted(lines, key=lambda x: x[1]) if s >= top - 0.5]
    title = " ".join(" ".join(title_lines).split())
    return title[:300] if len(title) >= 8 else None


# ----- cleanup (SPEC 6.2) -----

def _norm_edge(line: str) -> str:
    s = re.sub(r"[#*_`>\[\]|]", "", line).strip().lower()
    s = re.sub(r"\d+", "#", s)
    return " ".join(s.split())


def _edge_lines(page_lines: list[str]) -> tuple[list[int], list[int]]:
    """Indexes of the first and last EDGE_LINES non-empty lines."""
    non_empty = [i for i, l in enumerate(page_lines) if l.strip()]
    return non_empty[:EDGE_LINES], non_empty[-EDGE_LINES:]


def remove_headers_footers(pages: list[tuple[int, str]]) -> list[tuple[int, str]]:
    """Remove lines that recur at page edges across >= max(3, 30%) of pages, and bare page numbers."""
    split = [(n, t.splitlines()) for n, t in pages]
    counts: Counter[str] = Counter()
    for _, lines in split:
        head, tail = _edge_lines(lines)
        seen = {_norm_edge(lines[i]) for i in head + tail}
        counts.update(s for s in seen if s and len(s) < 120)
    threshold = max(3, math.ceil(0.3 * len(split)))
    recurring = {s for s, c in counts.items() if c >= threshold}
    out = []
    for n, lines in split:
        head, tail = _edge_lines(lines)
        drop = set()
        for i in head + tail:
            if _norm_edge(lines[i]) in recurring or PAGE_NUMBER_LINE.match(lines[i]):
                drop.add(i)
        out.append((n, "\n".join(l for i, l in enumerate(lines) if i not in drop)))
    return out


def _word_chars(text: str) -> int:
    return len(re.findall(r"\w", text))


def repair_text(text: str) -> str:
    """Join hyphenated line breaks, strip trailing spaces, collapse blank runs."""
    text = re.sub(r"(?<=\w)-\n(?=[a-z])", "", text)              # hyphen-\nated -> hyphenated
    text = re.sub(r"(?<=\w)- \n(?=[a-z])", "", text)
    text = "\n".join(l.rstrip() for l in text.splitlines())
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip("\n") + "\n"


def excise_references(md: str) -> tuple[str, str]:
    """Split off the References section (last matching heading past 40% of the text).

    An Appendix that follows the references is returned to the body.
    """
    lines = md.splitlines(keepends=True)
    total = len(md)
    pos, cut_line = 0, None
    for i, line in enumerate(lines):
        if pos >= total * 0.4 and REF_HEADING.match(line):
            cut_line = i
        pos += len(line)
    if cut_line is None:
        return md, ""
    body = "".join(lines[:cut_line])
    refs = "".join(lines[cut_line:])
    m = APPENDIX_HEADING.search(refs)
    if m:
        body = body + "\n" + refs[m.start():]
        refs = refs[:m.start()]
    return body, refs


def reference_dois(refs: str) -> list[str]:
    out: list[str] = []
    for d in DOI_RE.findall(refs):
        d = d.rstrip(".,;:").lower()
        if d not in out:
            out.append(d)
    return out


def clean_pages(pages: list[tuple[int, str]]) -> tuple[str, str, list[int]]:
    """Full deterministic cleanup. Returns (body_md, references_md, dropped_page_numbers)."""
    pages = remove_headers_footers(pages)
    kept, dropped = [], []
    for n, text in pages:
        if _word_chars(text) < MIN_PAGE_CHARS:
            dropped.append(n)
            continue
        kept.append(f"{PAGE_MARK_FMT.format(n=n)}\n{text.strip()}\n")
    md = repair_text("\n".join(kept))
    body, refs = excise_references(md)
    return repair_text(body), repair_text(refs) if refs.strip() else "", dropped


# ----- cache / corpus (SPEC 6.1) -----

def md_path(cfg: Config, paper: Paper) -> Path:
    from fetch import safe_name
    return cfg.md_dir / f"{safe_name(paper.paper_id)}.md"


def is_fresh(md: Path, pdf: Path) -> bool:
    return md.exists() and md.stat().st_mtime >= pdf.stat().st_mtime


def _update_meta(cfg: Config, paper: Paper, parse_info: dict, refs_text: str) -> None:
    from fetch import meta_path, write_meta
    p = meta_path(cfg, paper)
    rec = json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
    if rec is None:
        write_meta(cfg, paper, note="user-supplied pdf")
        rec = json.loads(p.read_text(encoding="utf-8"))
    rec["paper"] = paper.to_dict()
    rec["parse"] = parse_info
    rec["references_text"] = refs_text
    rec["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    p.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_one(cfg: Config, paper: Paper, pdf: Path, parser: Parser, force: bool = False) -> dict:
    """Parse one PDF if needed. Returns a parse-info dict with status: ok | cached | scanned | timeout | error."""
    out = md_path(cfg, paper)
    if not force and is_fresh(out, pdf):
        return {"status": "cached", "md": out.name}
    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(parser.parse, pdf)
        try:
            pages = fut.result(timeout=cfg.per_paper_timeout_s)
        except FutureTimeout:
            info = {"status": "timeout", "parser": parser.name, "seconds": cfg.per_paper_timeout_s}
            _update_meta(cfg, paper, info, "")
            log.warning("%s: parse timeout after %ss", paper.paper_id, cfg.per_paper_timeout_s)
            return info
        except Exception as e:
            info = {"status": "error", "parser": parser.name, "error": f"{type(e).__name__}: {e}"}
            _update_meta(cfg, paper, info, "")
            log.warning("%s: parse error %s", paper.paper_id, info["error"])
            return info
    chars = sum(_word_chars(t) for _, t in pages)
    if not pages or chars / len(pages) < MIN_CHARS_PER_PAGE:
        info = {"status": "scanned", "parser": parser.name, "pages": len(pages), "word_chars": chars}
        _update_meta(cfg, paper, info, "")
        log.warning("%s: near-zero extractable text (%d chars over %d pages); excluded, no OCR (SPEC 6.3)",
                    paper.paper_id, chars, len(pages))
        return info
    body, refs, dropped = clean_pages(pages)
    paper.references = reference_dois(refs) or paper.references
    cfg.md_dir.mkdir(parents=True, exist_ok=True)
    out.write_text(body, encoding="utf-8")
    info = {"status": "ok", "parser": parser.name, "pages": len(pages), "dropped_pages": dropped,
            "body_chars": len(body), "references_chars": len(refs), "reference_dois": len(paper.references),
            "seconds": round(time.monotonic() - t0, 2), "md": out.name}
    _update_meta(cfg, paper, info, refs)
    return info


def parse_all(cfg: Config, force: bool = False) -> list[Path]:
    """Parse every PDF in cfg.pdf_dir whose .md is missing or stale. Returns the .md files written."""
    from fetch import load_corpus, safe_name
    cfg.ensure_dirs()
    parser = get_parser(cfg)
    by_stem = {safe_name(p.paper_id): p for p in load_corpus(cfg)}
    written: list[Path] = []
    counts: Counter[str] = Counter()
    for pdf in sorted(cfg.pdf_dir.glob("*.pdf")):
        paper = by_stem.get(pdf.stem)
        if paper is None:                              # user-supplied PDF (SPEC 6.3 fallback)
            title = None
            try:
                title = extract_title(pdf)
            except Exception as e:
                log.info("%s: title extraction failed: %s", pdf.name, e)
            paper = Paper(paper_id=pdf.stem, title=title or pdf.stem, source="user",
                          status=STATUS_FETCHED, metadata_resolved=False)
            log.warning("%s: no metadata record; using filename id (metadata: unresolved). "
                        "Run `python -m fetch --resolve-dropped` to look it up.", pdf.name)
        info = parse_one(cfg, paper, pdf, parser, force=force)
        counts[info["status"]] += 1
        if info["status"] == "ok":
            written.append(md_path(cfg, paper))
    log.info("parse summary: %s", dict(counts))
    print(f"parse: {dict(counts)}")
    return written


if __name__ == "__main__":
    import sys
    from config import load_config
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    files = parse_all(load_config(), force="--force" in sys.argv)
    print(f"wrote {len(files)} markdown files")
