"""CORE: OA full-text aggregation. Free key from .env; ~10k/day."""

from __future__ import annotations

import logging
from collections.abc import Iterable

from models import Paper
from sources import SourceAdapter, SourceError, norm_title

log = logging.getLogger(__name__)
API = "https://api.core.ac.uk/v3/search/works"


class Core(SourceAdapter):
    name = "core"
    requires_key = True
    discovers = True

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}"}

    def parse_work(self, r: dict) -> Paper | None:
        title = r.get("title")
        if not title:
            return None
        doi = (r.get("doi") or "").lower() or None
        arxiv_id = r.get("arxivId") or None
        if not doi and arxiv_id:
            doi = f"10.48550/arxiv.{arxiv_id.lower()}"
        return Paper(
            paper_id=doi or norm_title(title), title=" ".join(title.split()),
            authors=[a.get("name") for a in r.get("authors") or [] if a.get("name")],
            year=r.get("yearPublished"), doi=doi, arxiv_id=arxiv_id, venue=r.get("publisher"),
            abstract=r.get("abstract"), source=self.name, pdf_url=r.get("downloadUrl") or None,
        )

    def search(self, query: str, limit: int) -> Iterable[Paper]:
        try:
            data = self.client.get_json(API, {"q": query, "limit": min(limit, 100)}, headers=self._headers())
        except SourceError as e:
            log.warning("%s search failed: %s", self.name, e)
            return []
        return [p for p in (self.parse_work(r) for r in data.get("results") or []) if p][:limit]

    def full_text_url(self, doi: str) -> str | None:
        """CORE download URL for a DOI, if aggregated."""
        for p in self.search(f'doi:"{doi}"', 3):
            if p.doi == doi.lower() and p.pdf_url:
                return p.pdf_url
        return None
