"""Identity resolution, dedup, PDF resolution and download (network layer, SPEC 5.3-5.4).

Run ``python -m fetch --query "..." --limit N`` to build the corpus into data/pdfs/.
Every outcome is written to data/meta/<safe_id>.json. Re-runs are idempotent.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from collections.abc import Iterable
from pathlib import Path

import requests

from config import Config, hf_hub_cache, load_config
from models import (STATUS_FETCHED, STATUS_HTTP_ERROR, STATUS_NO_OA_PDF, STATUS_PENDING,
                    STATUS_TIMEOUT, Paper)
from sources import CachedClient, SourceAdapter, SourceUnavailable, available_adapters, norm_title, user_agent
from sources.arxiv import ArXiv, pdf_url as arxiv_pdf_url, synth_doi
from sources.core import Core
from sources.crossref import Crossref
from sources.datacite import DataCite
from sources.europepmc import EuropePMC
from sources.openalex import OpenAlex
from sources.pubmed import PubMed
from sources.semanticscholar import SemanticScholar
from sources.unpaywall import Unpaywall

log = logging.getLogger(__name__)

DISCOVERY_SOURCES: list[type[SourceAdapter]] = [OpenAlex, ArXiv, EuropePMC, SemanticScholar, Core, PubMed]
# Field preference when merging duplicate records (first wins for scalar fields).
MERGE_PRIORITY = ["openalex", "crossref", "semanticscholar", "europepmc", "pubmed", "datacite", "core", "arxiv"]
SYNTH_PREFIX = "10.48550/arxiv."
_ARXIV_ID_RE = re.compile(r"^(\d{4}\.\d{4,5}|[a-z\-]+/\d{7})(v\d+)?$", re.I)


# ----- identity (SPEC 5.3) -----

def normalize_doi(doi: str | None) -> str | None:
    """Strip URL prefixes / 'doi:' and lowercase. None if empty."""
    if not doi:
        return None
    d = doi.strip()
    d = re.sub(r"^https?://(dx\.)?doi\.org/", "", d, flags=re.I)
    d = re.sub(r"^doi:\s*", "", d, flags=re.I)
    d = d.strip().lower()
    return d if d.startswith("10.") else None


def normalize_arxiv_id(aid: str | None) -> str | None:
    if not aid:
        return None
    m = _ARXIV_ID_RE.match(aid.strip())
    return m.group(1).lower() if m else None


def is_synthetic_doi(doi: str | None) -> bool:
    return bool(doi) and doi.lower().startswith(SYNTH_PREFIX)


def resolve_identity(paper: Paper) -> str:
    """paper_id by precedence: DOI -> OpenAlex ID -> arXiv ID -> S2 ID -> normalized title + year."""
    doi = normalize_doi(paper.doi)
    if doi:
        return doi
    if paper.openalex_id:
        return paper.openalex_id
    if paper.arxiv_id:
        return synth_doi(normalize_arxiv_id(paper.arxiv_id) or paper.arxiv_id)
    if paper.s2_id:
        return f"s2:{paper.s2_id}"
    return f"{norm_title(paper.title)}:{paper.year or 'n.d.'}"


def safe_name(paper_id: str) -> str:
    """Filesystem-safe file stem for a paper_id (DOIs contain '/', ':' etc.)."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", paper_id).strip("_")[:150]


# ----- dedup (SPEC 5.3) -----

def _merge_keys(p: Paper) -> list[str]:
    keys = []
    doi = normalize_doi(p.doi)
    if doi:
        keys.append(f"doi:{doi}")
    if p.openalex_id:
        keys.append(f"oa:{p.openalex_id}")
    aid = normalize_arxiv_id(p.arxiv_id)
    if aid:
        keys.append(f"arxiv:{aid}")
        keys.append(f"doi:{synth_doi(aid)}")
    if p.s2_id:
        keys.append(f"s2:{p.s2_id}")
    t = norm_title(p.title)
    if len(t.split()) >= 4:          # short titles are too ambiguous to merge on
        keys.append(f"title:{t}")
    return keys


def _merge(group: list[Paper]) -> Paper:
    group = sorted(group, key=lambda p: MERGE_PRIORITY.index(p.source) if p.source in MERGE_PRIORITY else 99)

    def first(attr):
        for p in group:
            v = getattr(p, attr)
            if v:
                return v
        return None

    # Prefer a real publisher DOI over a synthesized arXiv DOI.
    dois = [normalize_doi(p.doi) for p in group]
    real = [d for d in dois if d and not is_synthetic_doi(d)]
    synth = [d for d in dois if d and is_synthetic_doi(d)]
    doi = real[0] if real else (synth[0] if synth else None)
    arxiv_id = next((normalize_arxiv_id(p.arxiv_id) for p in group if normalize_arxiv_id(p.arxiv_id)), None)
    if not arxiv_id and doi and is_synthetic_doi(doi):
        arxiv_id = doi[len(SYNTH_PREFIX):]
    # PDF: arXiv direct first, else whichever source offered one.
    pdf = arxiv_pdf_url(arxiv_id) if arxiv_id else first("pdf_url")
    refs: list[str] = []
    for p in group:
        refs.extend(r for r in p.references if r not in refs)
    merged = Paper(
        paper_id="", title=first("title") or "", authors=first("authors") or [], year=first("year"),
        doi=doi, openalex_id=first("openalex_id"), arxiv_id=arxiv_id, s2_id=first("s2_id"),
        venue=first("venue"), abstract=first("abstract"),
        source="+".join(dict.fromkeys(p.source for p in group)), pdf_url=pdf,
        status=STATUS_PENDING, metadata_resolved=all(p.metadata_resolved for p in group), references=refs,
    )
    merged.paper_id = resolve_identity(merged)
    return merged


def dedup(papers: Iterable[Paper]) -> list[Paper]:
    """Union-find over DOI / OpenAlex / arXiv / S2 / title keys; merge each group into one Paper."""
    papers = list(papers)
    parent = list(range(len(papers)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    seen: dict[str, int] = {}
    for i, p in enumerate(papers):
        for k in _merge_keys(p):
            if k in seen:
                a, b = find(i), find(seen[k])
                if a != b:
                    parent[max(a, b)] = min(a, b)
            else:
                seen[k] = i
    groups: dict[int, list[Paper]] = {}
    for i, p in enumerate(papers):
        groups.setdefault(find(i), []).append(p)
    out = [_merge(g) for _, g in sorted(groups.items())]
    log.info("dedup: %d records -> %d papers", len(papers), len(out))
    return out


# ----- discovery -----

def discover(cfg: Config, query: str | None = None, limit: int | None = None) -> list[Paper]:
    """Query every available discovery source, dedup, return up to `limit` records. No download."""
    if cfg.offline:
        raise SourceUnavailable("discovery disabled: offline mode is on (C3). Set OFFLINE=0 to fetch.")
    query = query or cfg.fetch_query
    limit = limit or cfg.fetch_max_papers
    if not query:
        raise ValueError("no query given and fetch.query is empty in config.yaml")
    adapters = available_adapters(cfg, DISCOVERY_SOURCES)
    if not adapters:
        raise SourceUnavailable("no discovery source available (is CONTACT_EMAIL set in .env?)")
    per_source: list[list[Paper]] = []
    for ad in adapters:
        found = list(ad.search(query, limit))
        log.info("%s: %d records for %r (live=%d cached=%d)", ad.name, len(found), query,
                 ad.client.live_requests, ad.client.cache_hits)
        per_source.append(found)
    # Interleave sources round-robin so truncation to `limit` keeps the corpus diverse.
    records: list[Paper] = []
    for i in range(max((len(f) for f in per_source), default=0)):
        records.extend(f[i] for f in per_source if i < len(f))
    papers = dedup(records)[:limit]
    return enrich_arxiv(cfg, papers)


ARXIV_AUTHORITATIVE = ("title", "authors", "abstract")


def enrich_arxiv(cfg: Config, papers: list[Paper]) -> list[Paper]:
    """For arXiv papers known only through aggregators, adopt arXiv's own title/authors/abstract.

    Aggregator records for arXiv DOIs can carry the wrong title (seen in OpenAlex for
    10.48550/arxiv.2005.11401). arXiv's OAI-PMH record is authoritative for its own papers.
    """
    try:
        ax = ArXiv(cfg)
    except SourceUnavailable as e:
        log.info("arxiv enrichment skipped: %s", e)
        return papers
    for p in papers:
        aid = normalize_arxiv_id(p.arxiv_id)
        if not aid or "arxiv" in p.source.split("+"):
            continue
        rec = ax.get_by_id(aid)
        if not rec:
            continue
        changed = []
        for field in ARXIV_AUTHORITATIVE:
            v = getattr(rec, field)
            if v and v != getattr(p, field):
                setattr(p, field, v)
                changed.append(field)
        if p.year is None:
            p.year = rec.year
        if changed:
            p.source = p.source + "+arxiv"
            log.info("%s: arXiv metadata adopted for %s", p.paper_id, ", ".join(changed))
    return papers


def refresh_arxiv_metadata(cfg: Config) -> list[Paper]:
    """Apply enrich_arxiv to every stored record and rewrite changed meta files."""
    changed = []
    for p in load_corpus(cfg):
        before = (p.title, tuple(p.authors), p.abstract)
        enrich_arxiv(cfg, [p])
        if (p.title, tuple(p.authors), p.abstract) != before:
            rec = read_meta(cfg, p.paper_id) or {}
            rec["paper"] = p.to_dict()
            rec["note"] = (rec.get("note", "") + " arXiv metadata adopted").strip()
            rec["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            meta_path(cfg, p).write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
            changed.append(p)
    return changed


# ----- download (SPEC 5.4) -----

class _Deadline:
    def __init__(self, seconds: float):
        self.end = time.monotonic() + seconds

    @property
    def remaining(self) -> float:
        return self.end - time.monotonic()

    def check(self) -> None:
        if self.remaining <= 0:
            raise TimeoutError("per-paper timeout exceeded")


def pdf_path(cfg: Config, paper: Paper) -> Path:
    return cfg.pdf_dir / f"{safe_name(paper.paper_id)}.pdf"


def meta_path(cfg: Config, paper: Paper) -> Path:
    return cfg.meta_dir / f"{safe_name(paper.paper_id)}.json"


def write_meta(cfg: Config, paper: Paper, attempts: list[dict] | None = None, note: str = "") -> None:
    cfg.meta_dir.mkdir(parents=True, exist_ok=True)
    rec = {"paper": paper.to_dict(), "pdf_file": pdf_path(cfg, paper).name if paper.status == STATUS_FETCHED else None,
           "attempts": attempts or [], "note": note, "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    meta_path(cfg, paper).write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")


def read_meta(cfg: Config, paper_id: str) -> dict | None:
    p = cfg.meta_dir / f"{safe_name(paper_id)}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _candidates(cfg: Config, paper: Paper, resolvers: dict[str, SourceAdapter]) -> Iterable[tuple[str, str]]:
    """Yield (source, url) in SPEC 5.4 order: arXiv -> CORE -> Europe PMC -> Unpaywall, then any
    OA URL the discovery record already carried."""
    aid = normalize_arxiv_id(paper.arxiv_id)
    doi = normalize_doi(paper.doi)
    if aid:
        yield "arxiv", arxiv_pdf_url(aid)
    if doi and not is_synthetic_doi(doi):
        if "core" in resolvers:
            u = resolvers["core"].full_text_url(doi)
            if u:
                yield "core", u
        if "europepmc" in resolvers:
            u = resolvers["europepmc"].oa_pdf_url(doi)
            if u:
                yield "europepmc", u
        if "unpaywall" in resolvers:
            for u in resolvers["unpaywall"].oa_pdf_urls(doi):
                yield "unpaywall", u
    if paper.pdf_url:
        yield "record", paper.pdf_url


def _looks_like_pdf(content_type: str, first_bytes: bytes = b"") -> bool:
    ct = (content_type or "").split(";")[0].strip().lower()
    return ct == "application/pdf" or first_bytes.startswith(b"%PDF")


def _download_one(cfg: Config, paper: Paper, resolvers: dict[str, SourceAdapter],
                  probe: CachedClient, session: requests.Session) -> tuple[Paper, list[dict]]:
    dest = pdf_path(cfg, paper)
    attempts: list[dict] = []
    deadline = _Deadline(cfg.per_paper_timeout_s)
    saw_http_error = False
    try:
        for source, url in _candidates(cfg, paper, resolvers):
            deadline.check()
            try:
                head = probe.probe(url, timeout=min(30, max(1, deadline.remaining)))
            except Exception as e:                           # SourceError incl. timeouts
                attempts.append({"source": source, "url": url, "result": f"probe error: {e}"})
                saw_http_error = True
                continue
            if head["status"] >= 500:
                attempts.append({"source": source, "url": url, "result": f"HTTP {head['status']}"})
                saw_http_error = True
                continue
            if head["status"] >= 400:
                attempts.append({"source": source, "url": url, "result": f"HTTP {head['status']} (unavailable)"})
                continue
            if not _looks_like_pdf(head["content_type"]):
                attempts.append({"source": source, "url": url,
                                 "result": f"not a PDF ({head['content_type'] or 'no content-type'})"})
                continue
            deadline.check()
            tmp = dest.with_suffix(".part")
            try:
                with session.get(head["url"], stream=True, timeout=min(60, max(1, deadline.remaining))) as r:
                    r.raise_for_status()
                    first = b""
                    with tmp.open("wb") as fh:
                        for chunk in r.iter_content(65536):
                            if not first:
                                first = chunk[:8]
                                if not _looks_like_pdf(r.headers.get("Content-Type", ""), first):
                                    raise ValueError("body is not a PDF")
                            fh.write(chunk)
                            deadline.check()
                tmp.replace(dest)
            except (requests.Timeout, TimeoutError):
                tmp.unlink(missing_ok=True)
                raise
            except Exception as e:
                tmp.unlink(missing_ok=True)
                attempts.append({"source": source, "url": url, "result": f"download failed: {e}"})
                saw_http_error = True
                continue
            attempts.append({"source": source, "url": url, "result": "fetched", "bytes": dest.stat().st_size})
            paper.status = STATUS_FETCHED
            paper.pdf_url = head["url"]
            return paper, attempts
    except (TimeoutError, requests.Timeout):
        paper.status = STATUS_TIMEOUT
        attempts.append({"result": f"per-paper timeout ({cfg.per_paper_timeout_s}s)"})
        return paper, attempts
    if not attempts:
        attempts.append({"result": "no OA PDF candidate from any resolver "
                                   f"({', '.join(sorted(resolvers)) or 'no resolvers'}) (unavailable)"})
    paper.status = STATUS_HTTP_ERROR if (saw_http_error and not attempts_all_unavailable(attempts)) else STATUS_NO_OA_PDF
    return paper, attempts


def attempts_all_unavailable(attempts: list[dict]) -> bool:
    """True when every attempt was a definitive 'no PDF here' (4xx / landing page), not a transient error."""
    return all(("unavailable" in a.get("result", "")) or ("not a PDF" in a.get("result", ""))
               for a in attempts) if attempts else True


def download(cfg: Config, papers: Iterable[Paper]) -> list[Paper]:
    """Resolve + download PDFs into cfg.pdf_dir; write status to cfg.meta_dir. Idempotent."""
    if cfg.offline:
        raise SourceUnavailable("download disabled: offline mode is on (C3).")
    cfg.ensure_dirs()
    resolvers = {a.name: a for a in available_adapters(cfg, [Core, EuropePMC, Unpaywall])}
    probe = CachedClient(cfg, "pdf", sleep_s=1.0)
    session = requests.Session()
    session.headers["User-Agent"] = user_agent(cfg)
    out: list[Paper] = []
    counts = {"fetched": 0, "skipped": 0, STATUS_NO_OA_PDF: 0, STATUS_HTTP_ERROR: 0, STATUS_TIMEOUT: 0}
    for paper in papers:
        paper.paper_id = resolve_identity(paper)
        if pdf_path(cfg, paper).exists():
            paper.status = STATUS_FETCHED
            if not meta_path(cfg, paper).exists():
                write_meta(cfg, paper, note="pdf already present")
            counts["skipped"] += 1
            out.append(paper)
            continue
        if not paper.pdf_url and not paper.arxiv_id and not normalize_doi(paper.doi):
            paper.status = STATUS_NO_OA_PDF
            attempts = [{"result": "no DOI, arXiv id or URL to resolve"}]
        else:
            paper, attempts = _download_one(cfg, paper, resolvers, probe, session)
        write_meta(cfg, paper, attempts)
        counts[paper.status] = counts.get(paper.status, 0) + 1
        log.info("%s: %s (%s)", paper.paper_id, paper.status, attempts[-1].get("result") if attempts else "")
        out.append(paper)
    log.info("download summary: %s (probe live=%d cached=%d)", counts, probe.live_requests, probe.cache_hits)
    return out


def load_corpus(cfg: Config) -> list[Paper]:
    """Every paper recorded in data/meta/, whatever its status."""
    papers = []
    for p in sorted(cfg.meta_dir.glob("*.json")) if cfg.meta_dir.exists() else []:
        papers.append(Paper.from_dict(json.loads(p.read_text(encoding="utf-8"))["paper"]))
    return papers


# ----- user-supplied PDFs: metadata by title lookup (SPEC 6.3) -----

TITLE_MATCH_MIN = 0.9


def _title_similarity(a: str, b: str) -> float:
    import difflib
    return difflib.SequenceMatcher(None, norm_title(a), norm_title(b)).ratio()


def resolve_dropped(cfg: Config) -> list[Paper]:
    """Look up every metadata-unresolved paper by title (Crossref, then OpenAlex).

    On a confident match the canonical record is adopted, paper_id is reassigned by the
    normal precedence chain and the pdf/md/meta files are renamed. Returns resolved papers.
    """
    if cfg.offline:
        raise SourceUnavailable("metadata lookup disabled: offline mode is on (C3).")
    lookups = [a for a in available_adapters(cfg, [Crossref, OpenAlex])]
    resolved: list[Paper] = []
    for paper in load_corpus(cfg):
        if paper.metadata_resolved or not paper.title or paper.title == paper.paper_id:
            continue
        match, score, via = None, 0.0, ""
        for ad in lookups:
            for cand in ad.lookup_title(paper.title, 3):
                s = _title_similarity(paper.title, cand.title)
                if s > score:
                    match, score, via = cand, s, ad.name
            if score >= TITLE_MATCH_MIN:
                break
        if not match or score < TITLE_MATCH_MIN:
            log.warning("%s: no confident title match (best %.2f); keeping filename citation", paper.paper_id, score)
            continue
        old = paper
        new = Paper(**{**match.to_dict(), "status": STATUS_FETCHED, "metadata_resolved": True,
                       "source": f"user+{via}", "references": old.references})
        new.paper_id = resolve_identity(new)
        _rename_paper_files(cfg, old, new)
        log.info("%s -> %s via %s (title similarity %.2f)", old.paper_id, new.paper_id, via, score)
        resolved.append(new)
    return resolved


def _rename_paper_files(cfg: Config, old: Paper, new: Paper) -> None:
    old_stem, new_stem = safe_name(old.paper_id), safe_name(new.paper_id)
    old_meta = cfg.meta_dir / f"{old_stem}.json"
    rec = json.loads(old_meta.read_text(encoding="utf-8")) if old_meta.exists() else {}
    for d, ext in ((cfg.pdf_dir, ".pdf"), (cfg.md_dir, ".md")):
        src = d / f"{old_stem}{ext}"
        if src.exists() and old_stem != new_stem:
            src.replace(d / f"{new_stem}{ext}")
    rec["paper"] = new.to_dict()
    rec["pdf_file"] = f"{new_stem}.pdf"
    rec.setdefault("note", "")
    rec["note"] = (rec["note"] + f" resolved from {old.paper_id}").strip()
    rec["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    if old_meta.exists() and old_stem != new_stem:
        old_meta.unlink()
    (cfg.meta_dir / f"{new_stem}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")


# ----- model assets (network layer; the pipeline loads them from the local HF cache only) -----

MODEL_FILES = ["config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
               "vocab.txt", "model.safetensors", "modules.json", "sentence_bert_config.json",
               "1_Pooling/*", "config_sentence_transformers.json"]


def download_models(cfg: Config, include_reranker: bool = True) -> list[str]:
    """Fetch embedding (and reranker) model files into the HF cache. Idempotent."""
    if cfg.offline:
        raise SourceUnavailable("model download disabled: offline mode is on (C3).")
    from huggingface_hub import snapshot_download
    names = [cfg.embedding_model] + ([cfg.reranker_model] if include_reranker else [])
    paths = []
    for name in names:
        p = snapshot_download(name, allow_patterns=MODEL_FILES, cache_dir=hf_hub_cache())
        log.info("model %s -> %s", name, p)
        paths.append(p)
    return paths


# ----- CLI -----

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Discover and download papers into data/pdfs/")
    ap.add_argument("--query", help="search query (default: fetch.query in config.yaml)")
    ap.add_argument("--limit", type=int, help="max papers per source and overall")
    ap.add_argument("--no-download", action="store_true", help="discovery only")
    ap.add_argument("--models", action="store_true", help="download embedding/reranker model files only")
    ap.add_argument("--resolve-dropped", action="store_true",
                    help="look up metadata for user-supplied PDFs in data/pdfs by title, then exit")
    ap.add_argument("--refresh-arxiv", action="store_true",
                    help="re-adopt arXiv's own metadata for stored arXiv papers, then exit")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)     # huggingface_hub's client is chatty
    cfg = load_config()
    try:
        if args.models:
            for p in download_models(cfg):
                print(f"model files: {p}")
            return 0
        if args.resolve_dropped:
            done = resolve_dropped(cfg)
            print(f"resolved {len(done)} user-supplied paper(s)")
            for p in done:
                print(f"  {p.paper_id}  {p.title[:80]}")
            return 0
        if args.refresh_arxiv:
            done = refresh_arxiv_metadata(cfg)
            print(f"updated {len(done)} paper(s) from arXiv metadata")
            for p in done:
                print(f"  {p.paper_id}  {p.title[:80]}")
            return 0
        papers = discover(cfg, args.query, args.limit)
        print(f"discovered {len(papers)} papers")
        if args.no_download:
            for p in papers:
                print(f"  {p.paper_id}  [{p.source}]  {p.title[:80]}")
            return 0
        papers = download(cfg, papers)
    except SourceUnavailable as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    by_status: dict[str, int] = {}
    for p in papers:
        by_status[p.status] = by_status.get(p.status, 0) + 1
    print("status counts:", by_status)
    return 0


if __name__ == "__main__":
    sys.exit(main())
