"""Ollama client, context assembly, citation-forced prompt, integrity check (SPEC 7.2-7.3).

Built in Phase 6. The only LLM client in the codebase; it talks to the loopback
address validated by config.validate() and nothing else. No requests/httpx/urllib (C1).
"""

from __future__ import annotations

from collections.abc import Sequence

from config import Config
from models import Answer, Retrieved


def assemble_context(cfg: Config, retrieved: Sequence[Retrieved]) -> list[Retrieved]:
    """Keep chunks in rank order until the token budget is exceeded; drop from the tail."""
    raise NotImplementedError("Phase 6")


def generate(cfg: Config, query: str, retrieved: Sequence[Retrieved], model: str | None = None) -> Answer:
    """Produce a cited answer. Answer.invalid_citations reports IDs outside the retrieved set."""
    raise NotImplementedError("Phase 6")
