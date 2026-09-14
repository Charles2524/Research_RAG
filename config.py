"""Config load, validation, offline flag and small process utilities.

Reads ``config.yaml`` and ``.env``. The resulting :class:`Config` is frozen;
ablation runs derive variants with :meth:`Config.with_overrides`.
"""

from __future__ import annotations

import ctypes
import dataclasses
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse  # stdlib URL parsing only; no network client

import yaml
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = ROOT / "config.yaml"
DEFAULT_ENV_PATH = ROOT / ".env"

RETRIEVAL_MODES = ("bm25", "dense", "hybrid")
PARSERS = ("pymupdf4llm",)
SOURCES = (
    "openalex", "arxiv", "crossref", "unpaywall", "europepmc", "datacite",
    "semanticscholar", "core", "pubmed",
)
# Keyed sources and the .env variable that unlocks each (SPEC 5.2).
KEYED_SOURCES = {
    "semanticscholar": "SEMANTIC_SCHOLAR_API_KEY",
    "core": "CORE_API_KEY",
    "pubmed": "NCBI_API_KEY",
}
LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "::1")


class ConfigError(ValueError):
    """Raised when config.yaml or .env contains an invalid value."""


@dataclass(frozen=True)
class Config:
    offline: bool
    contact_email: str
    data_dir: Path
    results_dir: Path
    parser: str
    chunk_size_tokens: int
    chunk_overlap_tokens: int
    embedding_model: str
    embedding_dim: int
    reranker_model: str
    rerank: bool
    retrieval_mode: str
    k: int
    rrf_weight: float
    rrf_k: int
    llm_model: str
    llm_fallback_model: str
    ollama_host: str
    context_chunks: int
    token_budget: int
    llm_timeout_s: int
    fetch_query: str
    fetch_max_papers: int
    per_paper_timeout_s: int
    sleep_s: dict[str, float] = field(default_factory=dict)
    api_keys: dict[str, str | None] = field(default_factory=dict)

    # ----- derived paths (SPEC 4.3) -----
    @property
    def pdf_dir(self) -> Path:
        return self.data_dir / "pdfs"

    @property
    def md_dir(self) -> Path:
        return self.data_dir / "md"

    @property
    def meta_dir(self) -> Path:
        return self.data_dir / "meta"

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

    @property
    def index_path(self) -> Path:
        return self.data_dir / "index.db"

    @property
    def runs_csv(self) -> Path:
        return self.results_dir / "runs.csv"

    @property
    def eval_set_path(self) -> Path:
        return self.results_dir / "eval_set.jsonl"

    def ensure_dirs(self) -> None:
        for d in (self.pdf_dir, self.md_dir, self.meta_dir, self.cache_dir, self.results_dir):
            d.mkdir(parents=True, exist_ok=True)

    def with_overrides(self, **overrides: Any) -> "Config":
        """Return a validated copy with the given fields replaced (used by ablate.py)."""
        unknown = set(overrides) - {f.name for f in dataclasses.fields(self)}
        if unknown:
            raise ConfigError(f"unknown config fields: {sorted(unknown)}")
        cfg = dataclasses.replace(self, **overrides)
        validate(cfg)
        return cfg

    def has_key(self, source: str) -> bool:
        return bool(self.api_keys.get(source))


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _get(section: dict, key: str, default: Any = None) -> Any:
    return section.get(key, default) if isinstance(section, dict) else default


def load_config(
    path: Path | None = None,
    env_path: Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> Config:
    """Load config.yaml + .env, apply overrides, validate, return a frozen Config."""
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    env_path = Path(env_path) if env_path else DEFAULT_ENV_PATH
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{path} must contain a YAML mapping")

    # .env values, then real environment variables on top (env wins, for CI / ablation).
    env: dict[str, str | None] = {}
    if env_path.exists():
        env.update(dotenv_values(env_path))
    for key in ("CONTACT_EMAIL", "OFFLINE", *KEYED_SOURCES.values()):
        if os.environ.get(key) is not None:
            env[key] = os.environ[key]

    offline = _as_bool(raw.get("offline", False))
    if env.get("OFFLINE") is not None:
        offline = _as_bool(env["OFFLINE"])

    chunk = raw.get("chunk", {}) or {}
    emb = raw.get("embedding", {}) or {}
    rer = raw.get("reranker", {}) or {}
    ret = raw.get("retrieval", {}) or {}
    gen = raw.get("generation", {}) or {}
    fetch = raw.get("fetch", {}) or {}
    sleeps = _get(fetch, "sleep_s", {}) or {}

    base = ROOT if not path.is_absolute() else path.parent

    cfg = Config(
        offline=offline,
        contact_email=(env.get("CONTACT_EMAIL") or "").strip(),
        data_dir=_resolve(base, raw.get("data_dir", "data")),
        results_dir=_resolve(base, raw.get("results_dir", "results")),
        parser=str(raw.get("parser", "pymupdf4llm")),
        chunk_size_tokens=int(_get(chunk, "size_tokens", 512)),
        chunk_overlap_tokens=int(_get(chunk, "overlap_tokens", 64)),
        embedding_model=str(_get(emb, "model", "BAAI/bge-small-en-v1.5")),
        embedding_dim=int(_get(emb, "dim", 384)),
        reranker_model=str(_get(rer, "model", "cross-encoder/ms-marco-MiniLM-L-6-v2")),
        rerank=_as_bool(_get(rer, "enabled", False)),
        retrieval_mode=str(_get(ret, "mode", "hybrid")),
        k=int(_get(ret, "k", 10)),
        rrf_weight=float(_get(ret, "rrf_weight", 0.5)),
        rrf_k=int(_get(ret, "rrf_k", 60)),
        llm_model=str(_get(gen, "model", "qwen3:4b")),
        llm_fallback_model=str(_get(gen, "fallback_model", "qwen3:1.7b")),
        ollama_host=str(_get(gen, "ollama_host", "http://127.0.0.1:11434")),
        context_chunks=int(_get(gen, "context_chunks", 5)),
        token_budget=int(_get(gen, "token_budget", 3000)),
        llm_timeout_s=int(_get(gen, "timeout_s", 300)),
        fetch_query=str(_get(fetch, "query", "")),
        fetch_max_papers=int(_get(fetch, "max_papers", 40)),
        per_paper_timeout_s=int(_get(fetch, "per_paper_timeout_s", 120)),
        sleep_s={str(k): float(v) for k, v in sleeps.items()},
        api_keys={src: (env.get(var) or None) for src, var in KEYED_SOURCES.items()},
    )
    if overrides:
        cfg = dataclasses.replace(cfg, **overrides)
    validate(cfg)
    return cfg


def _resolve(base: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (base / p)


def validate(cfg: Config) -> None:
    """Raise ConfigError on any invalid value. Called on every load and override."""
    errors: list[str] = []
    if cfg.parser not in PARSERS:
        errors.append(f"parser must be one of {PARSERS}, got {cfg.parser!r}")
    if cfg.chunk_size_tokens <= 0:
        errors.append("chunk.size_tokens must be > 0")
    if not 0 <= cfg.chunk_overlap_tokens < cfg.chunk_size_tokens:
        errors.append("chunk.overlap_tokens must be >= 0 and < chunk.size_tokens")
    if cfg.embedding_dim <= 0:
        errors.append("embedding.dim must be > 0")
    if cfg.retrieval_mode not in RETRIEVAL_MODES:
        errors.append(f"retrieval.mode must be one of {RETRIEVAL_MODES}, got {cfg.retrieval_mode!r}")
    if cfg.k <= 0:
        errors.append("retrieval.k must be > 0")
    if not 0.0 <= cfg.rrf_weight <= 1.0:
        errors.append("retrieval.rrf_weight must be in [0, 1]")
    if cfg.rrf_k <= 0:
        errors.append("retrieval.rrf_k must be > 0")
    if cfg.context_chunks <= 0:
        errors.append("generation.context_chunks must be > 0")
    if cfg.token_budget <= 0:
        errors.append("generation.token_budget must be > 0")
    if cfg.llm_timeout_s <= 0:
        errors.append("generation.timeout_s must be > 0")
    host = urlparse(cfg.ollama_host)
    if host.scheme != "http" or host.hostname not in LOOPBACK_HOSTS:
        errors.append(
            f"generation.ollama_host must be http://<loopback>[:port], got {cfg.ollama_host!r} (C1)"
        )
    if cfg.fetch_max_papers <= 0:
        errors.append("fetch.max_papers must be > 0")
    if cfg.per_paper_timeout_s <= 0:
        errors.append("fetch.per_paper_timeout_s must be > 0")
    if cfg.contact_email and "@" not in cfg.contact_email:
        errors.append("CONTACT_EMAIL in .env does not look like an email address")
    unknown_sources = set(cfg.sleep_s) - set(SOURCES)
    if unknown_sources:
        errors.append(f"fetch.sleep_s has unknown sources: {sorted(unknown_sources)}")
    if any(v < 0 for v in cfg.sleep_s.values()):
        errors.append("fetch.sleep_s values must be >= 0")
    if errors:
        raise ConfigError("invalid configuration:\n  - " + "\n  - ".join(errors))


# ----- process utilities -----

def peak_rss_mb() -> float:
    """Peak resident set size of this process in MB (C4 gate metric)."""
    if sys.platform == "win32":
        class _PMC(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_uint32),
                ("PageFaultCount", ctypes.c_uint32),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(_PMC), ctypes.c_uint32]
        psapi.GetProcessMemoryInfo.restype = ctypes.c_int

        pmc = _PMC()
        pmc.cb = ctypes.sizeof(_PMC)
        if not psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        return pmc.PeakWorkingSetSize / (1024 * 1024)
    import resource  # POSIX only

    kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return kb / 1024 if sys.platform != "darwin" else kb / (1024 * 1024)


if __name__ == "__main__":
    c = load_config()
    print(f"config OK: offline={c.offline} mode={c.retrieval_mode} model={c.llm_model}")
    print(f"data_dir={c.data_dir}")
    print(f"contact_email={'set' if c.contact_email else 'MISSING (discovery layer will refuse)'}")
    for src, key in c.api_keys.items():
        print(f"key {src}: {'present' if key else 'absent -> source skipped'}")
    print(f"peak RSS: {peak_rss_mb():.1f} MB")
