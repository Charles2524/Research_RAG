"""DataCite: metadata for arXiv-style DOIs (10.48550/arxiv.*). No key."""

from __future__ import annotations

import logging

from models import Paper
from sources import SourceAdapter, SourceError, norm_title

log = logging.getLogger(__name__)
API = "https://api.datacite.org/dois"


class DataCite(SourceAdapter):
    name = "datacite"

    def get_by_doi(self, doi: str) -> Paper | None:
        try:
            data = self.client.get_json(f"{API}/{doi}")
        except SourceError as e:
            if e.status != 404:
                log.info("%s lookup for %s failed: %s", self.name, doi, e)
            return None
        attrs = (data.get("data") or {}).get("attributes") or {}
        titles = attrs.get("titles") or []
        if not titles:
            return None
        doi_l = (attrs.get("doi") or doi).lower()
        arxiv_id = doi_l.split("10.48550/arxiv.", 1)[1] if doi_l.startswith("10.48550/arxiv.") else None
        descs = attrs.get("descriptions") or []
        return Paper(
            paper_id=doi_l, title=" ".join(titles[0].get("title", "").split()),
            authors=[c.get("name") for c in attrs.get("creators") or [] if c.get("name")],
            year=attrs.get("publicationYear"), doi=doi_l, arxiv_id=arxiv_id,
            venue=attrs.get("publisher"),
            abstract=descs[0].get("description") if descs else None,
            source=self.name,
        )
