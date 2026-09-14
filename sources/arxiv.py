"""arXiv: preprint discovery + direct PDF. No key; ~3 s between calls.

Two endpoints, both official:
- Atom search API (export.arxiv.org) for free-text discovery. arXiv rate-limits this
  per IP with HTTP 429 "Rate exceeded."; we back off and report, never hammer.
- OAI-PMH (oaipmh.arxiv.org) for single-record metadata by id; separate service that
  is not subject to the search API's throttle. Used for get_by_id.
"""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from collections.abc import Iterable

from models import Paper
from sources import SourceAdapter, SourceError

log = logging.getLogger(__name__)
API = "https://export.arxiv.org/api/query"
OAI = "https://oaipmh.arxiv.org/oai"
NS = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom",
      "oai": "http://www.openarchives.org/OAI/2.0/", "ax": "http://arxiv.org/OAI/arXiv/"}
_ID_RE = re.compile(r"arxiv\.org/abs/(.+?)(v\d+)?$")


def synth_doi(arxiv_id: str) -> str:
    """DataCite-format DOI for an arXiv record (SPEC 5.3)."""
    return f"10.48550/arxiv.{arxiv_id.lower()}"


def pdf_url(arxiv_id: str) -> str:
    return f"https://arxiv.org/pdf/{arxiv_id}"


def id_year(arxiv_id: str) -> int | None:
    """First-submission year from a new-style id (YYMM.NNNNN); None for old-style ids."""
    m = re.match(r"^(\d{2})(\d{2})\.\d{4,5}$", arxiv_id)
    return 2000 + int(m.group(1)) if m else None


def _text(el, path: str) -> str | None:
    node = el.find(path, NS)
    return " ".join(node.text.split()) if node is not None and node.text else None


class ArXiv(SourceAdapter):
    name = "arxiv"
    discovers = True

    def _paper(self, arxiv_id: str, title: str, authors: list[str], year: int | None,
               real_doi: str | None, journal_ref: str | None, abstract: str | None) -> Paper:
        doi = (real_doi or synth_doi(arxiv_id)).lower()
        return Paper(paper_id=doi, title=title, authors=authors, year=year, doi=doi, arxiv_id=arxiv_id,
                     venue=journal_ref or "arXiv", abstract=abstract, source=self.name, pdf_url=pdf_url(arxiv_id))

    # ----- Atom search API -----

    def parse_entry(self, entry) -> Paper | None:
        m = _ID_RE.search(_text(entry, "a:id") or "")
        title = _text(entry, "a:title")
        if not m or not title:
            return None
        published = _text(entry, "a:published") or ""
        return self._paper(
            m.group(1), title,
            [n for n in (_text(a, "a:name") for a in entry.findall("a:author", NS)) if n],
            int(published[:4]) if published[:4].isdigit() else None,
            _text(entry, "arxiv:doi"), _text(entry, "arxiv:journal_ref"), _text(entry, "a:summary"),
        )

    def _query(self, params: dict) -> list[Paper]:
        try:
            root = ET.fromstring(self.client.get_text(API, params))
        except (SourceError, ET.ParseError) as e:
            log.warning("%s query failed: %s%s", self.name, e,
                        " [rate-limited by arXiv]" if self.client.saw_rate_limit else "")
            return []
        return [p for p in (self.parse_entry(e) for e in root.findall("a:entry", NS)) if p]

    def search(self, query: str, limit: int) -> Iterable[Paper]:
        terms = re.findall(r"[A-Za-z0-9\-]+", query)
        q = " AND ".join(f"all:{t}" for t in terms) if terms else f"all:{query}"
        return self._query({"search_query": q, "max_results": limit,
                            "sortBy": "relevance", "sortOrder": "descending"})[:limit]

    # ----- OAI-PMH single record -----

    def parse_oai(self, meta) -> Paper | None:
        arxiv_id = _text(meta, "ax:id")
        title = _text(meta, "ax:title")
        if not arxiv_id or not title:
            return None
        authors = []
        for a in meta.findall("ax:authors/ax:author", NS):
            name = " ".join(x for x in (_text(a, "ax:forenames"), _text(a, "ax:keyname")) if x)
            if name:
                authors.append(name)
        # OAI <created> can reflect the latest version; the id itself encodes first submission (YYMM.NNNNN).
        created = _text(meta, "ax:created") or ""
        year = id_year(arxiv_id) or (int(created[:4]) if created[:4].isdigit() else None)
        return self._paper(arxiv_id, title, authors, year,
                           _text(meta, "ax:doi"), _text(meta, "ax:journal-ref"), _text(meta, "ax:abstract"))

    def get_by_id(self, arxiv_id: str) -> Paper | None:
        arxiv_id = re.sub(r"v\d+$", "", arxiv_id.strip())
        try:
            root = ET.fromstring(self.client.get_text(
                OAI, {"verb": "GetRecord", "identifier": f"oai:arXiv.org:{arxiv_id}", "metadataPrefix": "arXiv"}))
        except (SourceError, ET.ParseError) as e:
            log.info("%s OAI lookup for %s failed: %s; trying search API", self.name, arxiv_id, e)
            res = self._query({"id_list": arxiv_id, "max_results": 1})
            return res[0] if res else None
        meta = root.find("oai:GetRecord/oai:record/oai:metadata/ax:arXiv", NS)
        return self.parse_oai(meta) if meta is not None else None
