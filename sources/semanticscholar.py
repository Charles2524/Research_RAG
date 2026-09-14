"""Semantic Scholar: discovery + abstracts. Free key from .env; 100 req / 5 min."""

from __future__ import annotations

import logging
from collections.abc import Iterable

from models import Paper
from sources import SourceAdapter, SourceError, norm_title

log = logging.getLogger(__name__)
API = "https://api.semanticscholar.org/graph/v1/paper/search"
FIELDS = "externalIds,title,authors,year,venue,abstract,openAccessPdf"


class SemanticScholar(SourceAdapter):
    name = "semanticscholar"
    requires_key = True
    discovers = True

    def parse_paper(self, r: dict) -> Paper | None:
        title = r.get("title")
        if not title:
            return None
        ext = r.get("externalIds") or {}
        doi = (ext.get("DOI") or "").lower() or None
        arxiv_id = ext.get("ArXiv")
        if not doi and arxiv_id:
            doi = f"10.48550/arxiv.{arxiv_id.lower()}"
        pdf = (r.get("openAccessPdf") or {}).get("url")
        return Paper(
            paper_id=doi or r.get("paperId") or norm_title(title), title=" ".join(title.split()),
            authors=[a.get("name") for a in r.get("authors") or [] if a.get("name")],
            year=r.get("year"), doi=doi, arxiv_id=arxiv_id, s2_id=r.get("paperId"),
            venue=r.get("venue") or None, abstract=r.get("abstract"), source=self.name, pdf_url=pdf,
        )

    def search(self, query: str, limit: int) -> Iterable[Paper]:
        try:
            data = self.client.get_json(API, {"query": query, "limit": min(limit, 100), "fields": FIELDS},
                                        headers={"x-api-key": self.api_key})
        except SourceError as e:
            log.warning("%s search failed: %s", self.name, e)
            return []
        return [p for p in (self.parse_paper(r) for r in data.get("data") or []) if p][:limit]
