"""Phase 2 gate: parse cache, cleanup, real-corpus checks, chunk stats, user-supplied PDF metadata."""

from __future__ import annotations

import json
import re
import shutil

import pytest

import fetch
import index
import parse
from chunk import chunk_all, describe
from config import load_config
from models import Paper


def _pymupdf_available() -> bool:
    try:
        import pymupdf  # noqa: F401
        return True
    except Exception:
        return False


needs_pymupdf = pytest.mark.skipif(not _pymupdf_available(),
                                   reason="UNVERIFIED: pymupdf native library cannot load (VC++ runtime?)")


@pytest.fixture(scope="module")
def cfg():
    c = load_config()
    c.ensure_dirs()
    return c


# ----- cleanup (offline, synthetic) -----

def _page(n: int, body: str, header="Journal of RAG Studies", footer=True) -> tuple[int, str]:
    lines = [header, "", body, "", f"{n}"] if footer else [header, "", body]
    return n, "\n".join(lines)


def test_headers_footers_removed_by_recurrence():
    words = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf"]
    pages = [_page(i, f"Body text of page {i} discusses {words[i-1]} retrieval in {words[-i]} detail.")
             for i in range(1, 8)]
    pages.append((8, "Unique last line\n\nBody 8\n\nNot a header"))
    out = parse.remove_headers_footers(pages)
    for n, text in out[:7]:
        assert "Journal of RAG Studies" not in text
        assert not re.search(rf"^\s*{n}\s*$", text, re.M)          # bare page number gone
        assert f"Body text of page {n}" in text
    assert "Unique last line" in out[7][1] and "Not a header" in out[7][1]


def test_headers_not_removed_when_rare():
    pages = [_page(1, "A"), _page(2, "B", header="Different header"), (3, "Just body\nmore body")]
    out = parse.remove_headers_footers(pages)
    assert "Journal of RAG Studies" in out[0][1]           # appears once: below threshold


def test_repair_text_hyphens_and_whitespace():
    s = "retrie-\nval is aug-\nmented   \n\n\n\nnext para\nWord-\nBreak"
    out = parse.repair_text(s)
    assert "retrieval is augmented" in out
    assert "\n\n\n" not in out
    assert "Word-\nBreak" in out                           # capital after hyphen: left alone


def test_excise_references_and_keep_appendix():
    body = "# 1 Intro\n\n" + ("Text. " * 200) + "\n\n# 2 Method\n\n" + ("More. " * 200)
    refs = "\n\n## References\n\n[1] A. Author. 10.1000/abc123. 2020.\n[2] B. Author. https://doi.org/10.5555/xyz.\n"
    appx = "\n\n# Appendix A\n\nExtra material.\n"
    b, r = parse.excise_references(body + refs + appx)
    assert "References" not in b and "10.1000/abc123" not in b
    assert "Appendix A" in b and "Extra material" in b
    assert "[1] A. Author" in r and "Appendix" not in r
    assert parse.reference_dois(r) == ["10.1000/abc123", "10.5555/xyz"]


def test_excise_references_ignores_early_mention():
    md = "# Intro\n\nReferences\n\n" + ("Body. " * 300)
    b, r = parse.excise_references(md)
    assert r == "" and "Body." in b


def test_clean_pages_drops_figure_only_pages_and_marks_pages():
    pages = [_page(1, "Real content " * 20), (2, "Fig. 3"), _page(3, "More content " * 20)]
    body, refs, dropped = parse.clean_pages(pages)
    assert dropped == [2]
    assert "<!-- page:1 -->" in body and "<!-- page:3 -->" in body and "<!-- page:2 -->" not in body


# ----- real corpus (Phase 2 gate) -----

@needs_pymupdf
def test_every_fetched_pdf_has_markdown_and_rerun_parses_zero(cfg):
    parse.parse_all(cfg)
    pdfs = sorted(cfg.pdf_dir.glob("*.pdf"))
    assert pdfs, "no PDFs in data/pdfs; run Phase 1 first"
    missing = []
    for pdf in pdfs:
        meta = json.loads((cfg.meta_dir / f"{pdf.stem}.json").read_text(encoding="utf-8"))
        status = meta.get("parse", {}).get("status")
        md = cfg.md_dir / f"{pdf.stem}.md"
        if status == "ok":
            assert md.exists() and md.stat().st_size > 1000, pdf.name
        elif status not in ("scanned", "timeout", "error"):
            missing.append(pdf.name)
    assert not missing, f"PDFs without a parse outcome: {missing}"
    assert parse.parse_all(cfg) == []                       # second run: nothing re-parsed


@needs_pymupdf
def test_spot_check_three_files_clean(cfg):
    mds = sorted(cfg.md_dir.glob("*.md"))[:3]
    assert len(mds) == 3
    for md in mds:
        text = md.read_text(encoding="utf-8")
        # References excised: no references heading left in the body
        assert not re.search(r"^#{1,6}\s*(\*\*)?\s*references\s*(\*\*)?\s*$", text, re.I | re.M), md.name
        # No line recurring at >= 40% of page starts (header noise)
        pages = re.split(r"^<!-- page:\d+ -->\s*$", text, flags=re.M)[1:]
        firsts = [p.strip().splitlines()[0].strip() for p in pages if p.strip()]
        if len(firsts) >= 5:
            top = max(firsts.count(f) for f in set(firsts))
            assert top < 0.4 * len(firsts), f"{md.name}: recurring first line {top}/{len(firsts)}"
        # No bare page-number lines
        assert not re.search(r"^\s*\d{1,3}\s*$", text, re.M), md.name
        # No hyphenated line breaks
        assert not re.search(r"\w-\n[a-z]", text), md.name


@needs_pymupdf
def test_reference_dois_recorded_for_some_papers(cfg):
    papers = fetch.load_corpus(cfg)
    with_refs = [p for p in papers if p.references]
    assert with_refs, "no paper has reference DOIs after parsing"


@needs_pymupdf
def test_chunk_corpus_distribution_and_invariants(cfg, capsys):
    chunks = chunk_all(cfg)
    stats = describe(chunks)
    print(f"\nCHUNK DISTRIBUTION: {stats}")
    assert stats["count"] > 100 and stats["papers"] >= 20
    assert stats["max"] <= cfg.chunk_size_tokens and stats["min"] > 0
    for c in chunks:
        assert c.text.strip() and c.n_tokens > 0 and c.n_tokens <= cfg.chunk_size_tokens
        assert c.paper_id and c.section and c.page_start >= 1 and c.page_end >= c.page_start
        assert c.chunk_id == f"{c.paper_id}#{c.ordinal}"
    conn = index.connect(cfg.index_path)
    assert index.count_chunks(conn) == len(chunks)
    assert index.count_papers(conn) >= stats["papers"]
    hits = index.query_fts(conn, "retrieval augmented generation", k=5)
    assert hits
    conn.close()


@needs_pymupdf
@pytest.mark.parametrize("size", [256, 1024])
def test_chunk_sizes_sweep_cheaply_from_cache(cfg, size, tmp_path):
    """Re-chunking uses the .md cache only; a different size must respect its cap."""
    md = sorted(cfg.md_dir.glob("*.md"))[0]
    from chunk import chunk_markdown
    chunks = chunk_markdown(cfg, "x", md.read_text(encoding="utf-8"), size, 32)
    assert chunks and max(c.n_tokens for c in chunks) <= size


# ----- user-supplied PDF metadata (SPEC 6.3) -----

@needs_pymupdf
def test_user_supplied_pdf_resolves_metadata_by_title(cfg, tmp_path):
    src = cfg.pdf_dir / f"{fetch.safe_name('10.48550/arxiv.2005.11401')}.pdf"
    assert src.exists(), "RAG paper PDF missing; Phase 1 download test creates it"
    tcfg = cfg.with_overrides(data_dir=tmp_path / "data")
    tcfg.ensure_dirs()
    shutil.copy(src, tcfg.pdf_dir / "dropped_paper.pdf")
    # share the API response cache so lookups replay from disk after the first run
    if cfg.cache_dir.exists():
        shutil.copytree(cfg.cache_dir, tcfg.cache_dir, dirs_exist_ok=True)

    title = parse.extract_title(tcfg.pdf_dir / "dropped_paper.pdf")
    assert title and "retrieval-augmented generation" in title.lower()

    parse.parse_all(tcfg)
    unresolved = fetch.load_corpus(tcfg)
    assert len(unresolved) == 1 and not unresolved[0].metadata_resolved
    assert unresolved[0].paper_id == "dropped_paper" and (tcfg.md_dir / "dropped_paper.md").exists()

    resolved = fetch.resolve_dropped(tcfg)
    assert len(resolved) == 1
    p = resolved[0]
    assert p.metadata_resolved and (p.doi or p.openalex_id) and p.year == 2020 and p.authors
    assert p.paper_id != "dropped_paper" and p.paper_id == fetch.resolve_identity(p)
    stem = fetch.safe_name(p.paper_id)
    assert (tcfg.pdf_dir / f"{stem}.pdf").exists() and (tcfg.md_dir / f"{stem}.md").exists()
    assert not (tcfg.pdf_dir / "dropped_paper.pdf").exists()
    assert fetch.load_corpus(tcfg)[0].paper_id == p.paper_id
