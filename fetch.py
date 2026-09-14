"""Identity resolution, PDF resolution and download (network layer, SPEC 5.3-5.4).

Built in Phase 1. Phase 0 defines the public surface only.
"""

from __future__ import annotations

from collections.abc import Iterable

from config import Config
from models import Paper


def normalize_doi(doi: str | None) -> str | None:
    """Strip URL prefixes and case from a DOI. Implemented in Phase 1."""
    raise NotImplementedError("Phase 1")


def resolve_identity(paper: Paper) -> str:
    """paper_id by precedence DOI -> OpenAlex -> arXiv -> S2 -> normalized title+year."""
    raise NotImplementedError("Phase 1")


def dedup(papers: Iterable[Paper]) -> list[Paper]:
    """Merge records of the same paper across sources into one."""
    raise NotImplementedError("Phase 1")


def discover(cfg: Config, query: str | None = None, limit: int | None = None) -> list[Paper]:
    """Query every available source, dedup, return records (no download)."""
    raise NotImplementedError("Phase 1")


def download(cfg: Config, papers: Iterable[Paper]) -> list[Paper]:
    """Resolve and download PDFs into cfg.pdf_dir; write status to cfg.meta_dir. Idempotent."""
    raise NotImplementedError("Phase 1")
