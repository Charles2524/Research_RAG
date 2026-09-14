"""Unpaywall: OA PDF resolution by DOI. No key; email parameter required."""

from __future__ import annotations

import logging

from sources import SourceAdapter, SourceError

log = logging.getLogger(__name__)
API = "https://api.unpaywall.org/v2"


class Unpaywall(SourceAdapter):
    name = "unpaywall"

    def oa_pdf_urls(self, doi: str) -> list[str]:
        """Candidate PDF URLs for a DOI, best first. Empty when not OA or unknown."""
        try:
            data = self.client.get_json(f"{API}/{doi}", {"email": self.cfg.contact_email})
        except SourceError as e:
            if e.status != 404:
                log.info("%s lookup for %s failed: %s", self.name, doi, e)
            return []
        if not data.get("is_oa"):
            return []
        urls: list[str] = []
        best = data.get("best_oa_location") or {}
        for loc in [best, *(data.get("oa_locations") or [])]:
            for u in (loc.get("url_for_pdf"), loc.get("url")):
                if u and u not in urls:
                    urls.append(u)
        return urls
