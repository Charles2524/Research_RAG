"""Discovery adapters (network layer, SPEC 5). One module per source, added in Phase 1.

This package is the *only* place besides fetch.py that may hold a network client.
Every adapter: descriptive user-agent with contact email, per-source sleep,
response cache in data/cache/, graceful skip when a required key is absent.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Iterable

from config import KEYED_SOURCES, Config
from models import Paper

log = logging.getLogger(__name__)


class SourceUnavailable(RuntimeError):
    """Raised when a source cannot be used (offline mode, missing key, missing email)."""


class SourceAdapter(ABC):
    name: str = ""
    requires_key: bool = False

    def __init__(self, cfg: Config) -> None:
        if cfg.offline:
            raise SourceUnavailable(f"{self.name}: offline mode is enabled (C3)")
        if not cfg.contact_email:
            raise SourceUnavailable(f"{self.name}: CONTACT_EMAIL missing from .env (SPEC 5.5)")
        if self.requires_key and not cfg.has_key(self.name):
            raise SourceUnavailable(
                f"{self.name}: no {KEYED_SOURCES[self.name]} in .env; source skipped")
        self.cfg = cfg
        self.sleep_s = cfg.sleep_s.get(self.name, 1.0)

    @abstractmethod
    def search(self, query: str, limit: int) -> Iterable[Paper]:
        """Yield Paper records for a free-text query."""


def available_adapters(cfg: Config, adapter_classes: Iterable[type[SourceAdapter]]) -> list[SourceAdapter]:
    """Instantiate every adapter that can run; log and skip the rest (SPEC 5.2)."""
    out: list[SourceAdapter] = []
    for cls in adapter_classes:
        try:
            out.append(cls(cfg))
        except SourceUnavailable as e:
            log.info("skipping source: %s", e)
    return out
