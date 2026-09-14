"""Phase 1 gate: discovery, identity, dedup, download. Live tests hit real APIs once, then cache."""

from __future__ import annotations

import json

import pytest
import requests

import fetch
from config import load_config
from models import STATUS_FETCHED, STATUS_NO_OA_PDF, Paper
from sources import SourceUnavailable, available_adapters
from sources.arxiv import ArXiv
from sources.openalex import OpenAlex

RAG_ARXIV = "2005.11401"
RAG_DOI = f"10.48550/arxiv.{RAG_ARXIV}"
PAYWALLED_DOI = "10.1109/tkde.2021.3130191"     # IEEE TKDE; Unpaywall reports is_oa=false


@pytest.fixture(scope="module")
def cfg():
    c = load_config()
    if not c.contact_email:
        pytest.skip("CONTACT_EMAIL not set; live tests need it")
    c.ensure_dirs()
    return c


# ----- identity (offline) -----

@pytest.mark.parametrize("raw,expected", [
    ("https://doi.org/10.1000/ABC", "10.1000/abc"),
    ("http://dx.doi.org/10.1000/abc", "10.1000/abc"),
    ("doi:10.1000/abc", "10.1000/abc"),
    ("  10.1000/abc ", "10.1000/abc"),
    ("not a doi", None),
    ("", None),
    (None, None),
])
def test_normalize_doi(raw, expected):
    assert fetch.normalize_doi(raw) == expected


def test_arxiv_id_year():
    from sources.arxiv import id_year
    assert id_year("2005.11401") == 2020 and id_year("2312.10997") == 2023
    assert id_year("cs/0112017") is None


def test_normalize_arxiv_id():
    assert fetch.normalize_arxiv_id("2005.11401v3") == "2005.11401"
    assert fetch.normalize_arxiv_id("cs/0112017") == "cs/0112017"
    assert fetch.normalize_arxiv_id("W123") is None


def test_resolve_identity_precedence():
    assert fetch.resolve_identity(Paper("x", "T", doi="https://doi.org/10.1/A", openalex_id="W1")) == "10.1/a"
    assert fetch.resolve_identity(Paper("x", "T", openalex_id="W1", arxiv_id="2005.11401")) == "W1"
    assert fetch.resolve_identity(Paper("x", "T", arxiv_id="2005.11401v2", s2_id="abc")) == RAG_DOI
    assert fetch.resolve_identity(Paper("x", "T", s2_id="abc")) == "s2:abc"
    assert fetch.resolve_identity(Paper("x", "A Title: Here!", year=2021)) == "a title here:2021"


def test_safe_name():
    assert fetch.safe_name("10.1000/abc:def") == "10.1000_abc_def"
    assert "/" not in fetch.safe_name(RAG_DOI)


def test_dedup_merges_across_keys_and_prefers_real_doi():
    oa = Paper("", "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks", authors=["Patrick Lewis"],
               year=2020, doi=RAG_DOI, openalex_id="W1", source="openalex", venue="arXiv")
    ax = Paper("", "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks", authors=["P. Lewis"],
               year=2020, doi=RAG_DOI, arxiv_id=RAG_ARXIV, source="arxiv", pdf_url="https://arxiv.org/pdf/2005.11401")
    cr = Paper("", "Retrieval-augmented generation for knowledge-intensive NLP tasks", year=2020,
               doi="10.5555/3495724.3496517", source="crossref", references=["10.1/x"])
    other = Paper("", "Dense Passage Retrieval for Open-Domain Question Answering", year=2020,
                  arxiv_id="2004.04906", source="arxiv")
    out = fetch.dedup([oa, ax, cr, other])
    assert len(out) == 2
    m = next(p for p in out if p.arxiv_id == RAG_ARXIV)
    assert m.paper_id == "10.5555/3495724.3496517"         # real DOI beats synthesized
    assert m.openalex_id == "W1" and m.arxiv_id == RAG_ARXIV
    assert m.pdf_url == "https://arxiv.org/pdf/2005.11401"  # arXiv direct preferred
    assert m.authors == ["Patrick Lewis"]                    # openalex metadata preferred
    assert m.references == ["10.1/x"]
    assert set(m.source.split("+")) == {"openalex", "arxiv", "crossref"}


def test_dedup_does_not_merge_short_titles_only():
    a = Paper("", "Attention", year=2017, source="openalex", openalex_id="W1")
    b = Paper("", "Attention", year=2017, source="arxiv", arxiv_id="1706.03762")
    assert len(fetch.dedup([a, b])) == 2


# ----- C3 offline and C5 graceful degradation -----

def test_offline_mode_fails_fast(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("CONTACT_EMAIL=a@b.c\nOFFLINE=1\n", encoding="utf-8")
    off = load_config(env_path=env)
    assert off.offline
    with pytest.raises(SourceUnavailable, match="offline"):
        fetch.discover(off, "anything", 1)
    with pytest.raises(SourceUnavailable, match="offline"):
        fetch.download(off, [])
    with pytest.raises(SourceUnavailable):
        OpenAlex(off)


def test_keyed_sources_skip_cleanly_with_empty_env(tmp_path, caplog):
    env = tmp_path / ".env"
    env.write_text("", encoding="utf-8")
    empty = load_config(env_path=env)
    # No email at all: every adapter skips (logged), discover reports it clearly.
    with caplog.at_level("INFO", logger="sources"):
        assert available_adapters(empty, fetch.DISCOVERY_SOURCES) == []
    assert "skipping source" in caplog.text
    with pytest.raises(SourceUnavailable, match="CONTACT_EMAIL"):
        fetch.discover(empty, "x", 1)
    # Email but no keys: keyed sources skipped, key-free ones available.
    env.write_text("CONTACT_EMAIL=a@b.c\n", encoding="utf-8")
    keyless = load_config(env_path=env)
    names = {a.name for a in available_adapters(keyless, fetch.DISCOVERY_SOURCES)}
    assert "semanticscholar" not in names and "core" not in names
    assert {"openalex", "arxiv", "europepmc", "pubmed"} <= names


# ----- live: discovery -----

def test_live_openalex_returns_parsed_records(cfg):
    recs = list(OpenAlex(cfg).search("retrieval-augmented generation", 5))
    assert len(recs) >= 3
    for p in recs:
        assert p.title and p.openalex_id and p.source == "openalex"
    assert any(p.doi for p in recs) and any(p.year for p in recs)


def test_live_arxiv_search_returns_parsed_records(cfg):
    ax = ArXiv(cfg)
    recs = list(ax.search("retrieval-augmented generation", 5))
    if not recs and ax.last_error:
        pytest.skip(f"UNVERIFIED: arXiv search API unavailable from this IP: {ax.last_error}")
    assert len(recs) >= 3
    for p in recs:
        assert p.title and p.arxiv_id and p.doi and p.pdf_url.startswith("https://arxiv.org/pdf/")


def test_live_arxiv_oai_record(cfg):
    p = ArXiv(cfg).get_by_id(RAG_ARXIV)
    assert p and p.arxiv_id == RAG_ARXIV and p.year == 2020 and p.doi == RAG_DOI
    assert p.authors and p.authors[0].endswith("Lewis") and p.abstract


def test_live_dedup_merges_known_duplicate_across_two_sources(cfg):
    oa = OpenAlex(cfg).get_by_doi(RAG_DOI)
    ax = ArXiv(cfg).get_by_id(RAG_ARXIV)
    assert oa and ax
    merged = fetch.dedup([oa, ax])
    assert len(merged) == 1
    m = merged[0]
    assert m.arxiv_id == RAG_ARXIV and m.openalex_id and m.doi
    assert "openalex" in m.source and "arxiv" in m.source


# ----- live: download -----

def test_live_download_arxiv_pdf(cfg):
    paper = OpenAlex(cfg).get_by_doi(RAG_DOI)      # carries arxiv_id -> arXiv direct PDF
    assert paper and paper.arxiv_id == RAG_ARXIV
    out = fetch.download(cfg, [paper])
    assert out[0].status == STATUS_FETCHED
    path = fetch.pdf_path(cfg, out[0])
    assert path.exists() and path.read_bytes()[:4] == b"%PDF"
    meta = json.loads(fetch.meta_path(cfg, out[0]).read_text(encoding="utf-8"))
    assert meta["paper"]["status"] == STATUS_FETCHED and meta["pdf_file"] == path.name


def test_live_paywalled_doi_marked_no_oa_pdf(cfg):
    paper = Paper(paper_id="", title="A Review on Generative Adversarial Networks", doi=PAYWALLED_DOI, year=2021)
    out = fetch.download(cfg, [paper])
    assert out[0].status == STATUS_NO_OA_PDF
    assert not fetch.pdf_path(cfg, out[0]).exists()
    meta = fetch.read_meta(cfg, out[0].paper_id)
    assert meta["paper"]["status"] == STATUS_NO_OA_PDF and meta["attempts"]


def test_rerun_downloads_nothing_and_hits_no_network(cfg, monkeypatch):
    papers = [OpenAlex(cfg).get_by_doi(RAG_DOI),
              Paper(paper_id="", title="A Review on Generative Adversarial Networks", doi=PAYWALLED_DOI, year=2021)]
    before = {p.name: p.stat().st_mtime for p in cfg.pdf_dir.glob("*.pdf")}

    def no_network(*a, **k):
        raise AssertionError("live request attempted on re-run")
    monkeypatch.setattr(requests.Session, "request", no_network)
    out = fetch.download(cfg, papers)
    assert [p.status for p in out] == [STATUS_FETCHED, STATUS_NO_OA_PDF]
    after = {p.name: p.stat().st_mtime for p in cfg.pdf_dir.glob("*.pdf")}
    assert after == before
