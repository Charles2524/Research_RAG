"""C1 enforcement: no module at or below the ingestion layer may import a network library.

Spec'd for Phase 5 but cheap to enforce from Phase 0 so every later phase is checked.
Inspects the AST, so it catches imports inside functions as well as at module top.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PIPELINE_MODULES = ["parse.py", "chunk.py", "index.py", "retrieve.py", "generate.py"]
NETWORK_LIBS = {
    "requests", "httpx", "urllib", "urllib3", "aiohttp", "http", "socket", "ssl",
    "ftplib", "smtplib", "websocket", "websockets", "pycurl",
}
# generate.py must reach the loopback Ollama endpoint; stdlib http.client is the one
# allowed transport there and config.validate() pins the host to loopback.
ALLOWED = {"generate.py": {"http"}}


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module.split(".")[0])
    return found


@pytest.mark.parametrize("module", PIPELINE_MODULES)
def test_pipeline_module_has_no_network_import(module):
    path = ROOT / module
    assert path.exists(), path
    bad = (_imports(path) & NETWORK_LIBS) - ALLOWED.get(module, set())
    assert not bad, f"{module} imports network library: {sorted(bad)} (C1 violation)"


def test_config_and_models_have_no_network_import():
    for name in ("config.py", "models.py"):
        bad = _imports(ROOT / name) & (NETWORK_LIBS - {"urllib"})   # urllib.parse for host validation only
        assert not bad, f"{name} imports {sorted(bad)}"
