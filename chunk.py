"""Markdown -> chunks: split on headers, subdivide to target tokens with overlap (SPEC 6.4).

Built in Phase 2. No network client may be imported here (C1).
"""

from __future__ import annotations

from config import Config
from models import Chunk


def chunk_markdown(paper_id: str, md: str, size_tokens: int, overlap_tokens: int) -> list[Chunk]:
    raise NotImplementedError("Phase 2")


def chunk_all(cfg: Config) -> list[Chunk]:
    """Chunk every cached .md under cfg.md_dir and store the result in the index."""
    raise NotImplementedError("Phase 2")
