"""BM25 / dense / hybrid retrieval with optional rerank, behind one interface (SPEC 7.1).

Built in Phase 4. No network client may be imported here (C1).
"""

from __future__ import annotations

from config import Config
from models import Retrieved


def retrieve(
    cfg: Config,
    query: str,
    mode: str | None = None,
    k: int | None = None,
    rerank: bool | None = None,
) -> list[Retrieved]:
    """Return the top-k chunks for a query. Defaults come from cfg."""
    raise NotImplementedError("Phase 4")
