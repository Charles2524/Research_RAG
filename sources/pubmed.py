"""PubMed (NCBI E-utilities): biomedical metadata. Key optional (raises rate limit)."""

from __future__ import annotations

import logging
from collections.abc import Iterable

from models import Paper
from sources import SourceAdapter, SourceError, norm_title

log = logging.getLogger(__name__)
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


class PubMed(SourceAdapter):
    name = "pubmed"
    requires_key = False      # SPEC 5.2: key optional
    discovers = True

    def _params(self, **kw) -> dict:
        p = {"db": "pubmed", "retmode": "json", "tool": "RAGResearchCompanion",
             "email": self.cfg.contact_email, **kw}
        if self.api_key:
            p["api_key"] = self.api_key
        return p

    def parse_summary(self, r: dict) -> Paper | None:
        title = r.get("title")
        if not title:
            return None
        doi = None
        for aid in r.get("articleids") or []:
            if aid.get("idtype") == "doi" and aid.get("value"):
                doi = aid["value"].lower()
        pubdate = str(r.get("pubdate") or "")
        year = int(pubdate[:4]) if pubdate[:4].isdigit() else None
        return Paper(
            paper_id=doi or norm_title(title), title=" ".join(title.split()).rstrip("."),
            authors=[a.get("name") for a in r.get("authors") or [] if a.get("name")],
            year=year, doi=doi, venue=r.get("fulljournalname") or r.get("source"), source=self.name,
        )

    def search(self, query: str, limit: int) -> Iterable[Paper]:
        try:
            ids = self.client.get_json(f"{EUTILS}/esearch.fcgi", self._params(term=query, retmax=min(limit, 100)))
            idlist = (ids.get("esearchresult") or {}).get("idlist") or []
            if not idlist:
                return []
            summ = self.client.get_json(f"{EUTILS}/esummary.fcgi", self._params(id=",".join(idlist)))
        except SourceError as e:
            log.warning("%s search failed: %s", self.name, e)
            return []
        result = summ.get("result") or {}
        out = []
        for uid in result.get("uids") or []:
            p = self.parse_summary(result.get(uid) or {})
            if p:
                out.append(p)
        return out[:limit]
