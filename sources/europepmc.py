"""Europe PMC: biomedical discovery + OA full text. No key."""

from __future__ import annotations

import logging
from collections.abc import Iterable

from models import Paper
from sources import SourceAdapter, SourceError, norm_title

log = logging.getLogger(__name__)
API = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"


def pmc_pdf_url(pmcid: str) -> str:
    return f"https://europepmc.org/articles/{pmcid}?pdf=render"


class EuropePMC(SourceAdapter):
    name = "europepmc"
    discovers = True

    def parse_result(self, r: dict) -> Paper | None:
        title = r.get("title")
        if not title:
            return None
        doi = (r.get("doi") or "").lower() or None
        year = int(r["pubYear"]) if str(r.get("pubYear", "")).isdigit() else None
        authors = [a.strip() for a in (r.get("authorString") or "").rstrip(".").split(",") if a.strip()]
        pmcid = r.get("pmcid")
        pdf = pmc_pdf_url(pmcid) if pmcid and r.get("isOpenAccess") == "Y" else None
        return Paper(
            paper_id=doi or norm_title(title), title=" ".join(title.split()).rstrip("."),
            authors=authors, year=year, doi=doi, venue=r.get("journalTitle"),
            source=self.name, pdf_url=pdf,
        )

    def _search(self, query: str, limit: int) -> list[Paper]:
        try:
            data = self.client.get_json(API, {"query": query, "format": "json",
                                              "resultType": "lite", "pageSize": min(limit, 100)})
        except SourceError as e:
            log.warning("%s search failed: %s", self.name, e)
            return []
        results = (data.get("resultList") or {}).get("result") or []
        return [p for p in (self.parse_result(r) for r in results) if p][:limit]

    def search(self, query: str, limit: int) -> Iterable[Paper]:
        return self._search(query, limit)

    def oa_pdf_url(self, doi: str) -> str | None:
        """OA PDF for a DOI via its PMC record, if any."""
        for p in self._search(f'DOI:"{doi}"', 3):
            if p.doi == doi.lower() and p.pdf_url:
                return p.pdf_url
        return None
