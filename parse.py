"""PDF -> markdown, deterministic cleanup, parse cache (SPEC 6.1-6.3).

Built in Phase 2. No network client may be imported here (C1).
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from config import Config


class Parser(Protocol):
    name: str

    def parse(self, pdf_path: Path) -> str:
        """Return raw markdown for a PDF."""


def get_parser(cfg: Config) -> Parser:
    raise NotImplementedError("Phase 2")


def clean_markdown(md: str) -> tuple[str, str]:
    """Return (body, references) after header/footer removal, hyphen repair, whitespace collapse."""
    raise NotImplementedError("Phase 2")


def parse_all(cfg: Config) -> list[Path]:
    """Parse every PDF whose .md is missing or stale. Returns the list of .md files written."""
    raise NotImplementedError("Phase 2")
