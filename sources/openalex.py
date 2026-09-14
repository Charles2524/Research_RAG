"""OpenAlex: primary discovery + metadata. No key; mailto for the polite pool."""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable

from models import Paper
from sources import SourceAdapter, SourceError, norm_title

log = logging.getLogger(__name__)
API = "https://api.openalex.org"
_ARXIV_RE = re.compile(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5}|[a-z\-]+/\d{7})(?:v\d+)?", re.I)


def _inverted_to_text(inv: dict | None) -> str | None:
    if not inv:
        return None
    positions: list[tuple[int, str]] = []
    for word, idxs in inv.items():
        positions.extend((i, word) for i in idxs)
    positions.sort()
    return " ".join(w for _, w in positions)


def _strip_doi(url: str | None) -> str | None:
    if not url:
        return None
    return re.sub(r"^https?://(dx\.)?doi\.org/", "", url, flags=re.I).lower()


class OpenAlex(SourceAdapter):
    name = "openalex"
    discovers = True

    def _params(self, **kw) -> dict:
        return {"mailto": self.cfg.contact_email, **kw}

    def parse_work(self, w: dict) -> Paper | None:
        title = w.get("title") or w.get("display_name")
        if not title:
            return None
        oa_id = (w.get("id") or "").rsplit("/", 1)[-1] or None
        doi = _strip_doi(w.get("doi"))
        arxiv_id = None
        for loc in (w.get("locations") or []):
            for url in (loc.get("landing_page_url"), loc.get("pdf_url")):
                m = _ARXIV_RE.search(url or "")
                if m:
                    arxiv_id = m.group(1)
                    break
            if arxiv_id:
                break
        if not arxiv_id and doi and doi.startswith("10.48550/arxiv."):
            arxiv_id = doi.split("10.48550/arxiv.", 1)[1]
        venue = ((w.get("primary_location") or {}).get("source") or {}).get("display_name")
        return Paper(
            paper_id=doi or oa_id or norm_title(title),
            title=title.strip(),
            authors=[a["author"]["display_name"] for a in w.get("authorships", [])
                     if a.get("author", {}).get("display_name")],
            year=w.get("publication_year"),
            doi=doi, openalex_id=oa_id, arxiv_id=arxiv_id, venue=venue,
            abstract=_inverted_to_text(w.get("abstract_inverted_index")),
            source=self.name,
            pdf_url=(w.get("open_access") or {}).get("oa_url"),
        )

    def search(self, query: str, limit: int) -> Iterable[Paper]:
        try:
            data = self.client.get_json(f"{API}/works", self._params(
                search=query, per_page=min(limit, 200), sort="relevance_score:desc",
                filter="type:article|preprint"))
        except SourceError as e:
            log.warning("%s search failed: %s", self.name, e)
            return []
        out = []
        for w in data.get("results", []):
            p = self.parse_work(w)
            if p:
                out.append(p)
        return out[:limit]

    def get_by_doi(self, doi: str) -> Paper | None:
        try:
            return self.parse_work(self.client.get_json(f"{API}/works/https://doi.org/{doi}", self._params()))
        except SourceError as e:
            log.info("%s lookup by DOI %s failed: %s", self.name, doi, e)
            return None

    def lookup_title(self, title: str, n: int = 3) -> list[Paper]:
        """Title search for user-supplied PDFs (SPEC 6.3). Caller decides confidence."""
        try:
            data = self.client.get_json(f"{API}/works", self._params(search=title, per_page=n))
        except SourceError as e:
            log.info("%s title lookup failed: %s", self.name, e)
            return []
        return [p for p in (self.parse_work(w) for w in data.get("results", [])) if p]
