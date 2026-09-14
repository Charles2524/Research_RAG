"""Crossref: DOI metadata, reference traversal, title lookup. No key; mailto for polite pool."""

from __future__ import annotations

import logging
import re

from models import Paper
from sources import SourceAdapter, SourceError, norm_title

log = logging.getLogger(__name__)
API = "https://api.crossref.org/works"


class Crossref(SourceAdapter):
    name = "crossref"

    def _params(self, **kw) -> dict:
        return {"mailto": self.cfg.contact_email, **kw}

    def parse_item(self, m: dict) -> Paper | None:
        titles = m.get("title") or []
        if not titles:
            return None
        doi = (m.get("DOI") or "").lower() or None
        issued = ((m.get("issued") or {}).get("date-parts") or [[None]])[0]
        year = issued[0] if issued and isinstance(issued[0], int) else None
        authors = []
        for a in m.get("author") or []:
            name = " ".join(x for x in (a.get("given"), a.get("family")) if x) or a.get("name")
            if name:
                authors.append(name)
        abstract = m.get("abstract")
        if abstract:
            abstract = " ".join(re.sub(r"<[^>]+>", " ", abstract).split())
        container = (m.get("container-title") or [None])[0]
        refs = [r["DOI"].lower() for r in (m.get("reference") or []) if r.get("DOI")]
        return Paper(
            paper_id=doi or norm_title(titles[0]),
            title=" ".join(titles[0].split()), authors=authors, year=year, doi=doi,
            venue=container, abstract=abstract, source=self.name, references=refs,
        )

    def get_by_doi(self, doi: str) -> Paper | None:
        try:
            data = self.client.get_json(f"{API}/{doi}", self._params())
        except SourceError as e:
            log.info("%s lookup by DOI %s failed: %s", self.name, doi, e)
            return None
        return self.parse_item(data.get("message") or {})

    def lookup_title(self, title: str, n: int = 3) -> list[Paper]:
        """Bibliographic title search (SPEC 6.3). Caller decides confidence."""
        try:
            data = self.client.get_json(API, self._params(**{"query.bibliographic": title, "rows": n}))
        except SourceError as e:
            log.info("%s title lookup failed: %s", self.name, e)
            return []
        items = (data.get("message") or {}).get("items") or []
        return [p for p in (self.parse_item(i) for i in items) if p]
