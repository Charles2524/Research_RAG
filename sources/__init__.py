"""Discovery adapters (network layer, SPEC 5). One module per source.

This package and fetch.py are the *only* places that may hold a network client (C1).
Every request: descriptive user-agent with contact email, per-source sleep, response
cached under data/cache/<source>/ so repeated runs never re-hit the network (SPEC 5.5).
Keyed sources skip with a log line when the key is absent (SPEC 5.2).
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import requests

from config import KEYED_SOURCES, Config
from models import Paper

log = logging.getLogger(__name__)

USER_AGENT_TEMPLATE = "RAGResearchCompanion/0.1 (local-first research tool; mailto:{email})"
DEFAULT_TIMEOUT_S = 30
RETRIES = 2                          # extra attempts on transient statuses
RETRY_BACKOFF_S = 10.0
TRANSIENT = {408, 429, 500, 502, 503, 504}
_last_call: dict[str, float] = {}   # per-source timestamp of the last live request


class SourceUnavailable(RuntimeError):
    """Raised when a source cannot be used (offline mode, missing key, missing email)."""


class SourceError(RuntimeError):
    """A live request failed (HTTP error, connection error, bad payload)."""

    def __init__(self, msg: str, status: int | None = None):
        super().__init__(msg)
        self.status = status


def user_agent(cfg: Config) -> str:
    return USER_AGENT_TEMPLATE.format(email=cfg.contact_email)


def norm_title(title: str | None) -> str:
    """Lowercase alphanumeric words joined by single spaces; used for dedup and lookup."""
    if not title:
        return ""
    return " ".join(re.findall(r"[a-z0-9]+", title.lower()))


def _cache_key(method: str, url: str, params: dict | None) -> str:
    payload = json.dumps([method, url, sorted((params or {}).items())], sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


class CachedClient:
    """HTTP client with per-source politeness delay and on-disk response cache.

    2xx and 4xx responses are cached (they are stable). 5xx, timeouts and connection
    errors are not, so a transient failure is retried on the next run.
    """

    def __init__(self, cfg: Config, source: str, sleep_s: float | None = None):
        if cfg.offline:
            raise SourceUnavailable(f"{source}: offline mode is enabled (C3); no network requests")
        if not cfg.contact_email:
            raise SourceUnavailable(f"{source}: CONTACT_EMAIL missing from .env (SPEC 5.5)")
        self.cfg = cfg
        self.source = source
        self.sleep_s = cfg.sleep_s.get(source, 1.0) if sleep_s is None else sleep_s
        self.cache_dir = cfg.cache_dir / source
        self.session = requests.Session()
        self.session.headers["User-Agent"] = user_agent(cfg)
        self.live_requests = 0
        self.cache_hits = 0
        self.saw_rate_limit = False   # set when any live request returned HTTP 429

    # ----- public -----

    def get_json(self, url: str, params: dict | None = None, headers: dict | None = None,
                 timeout: float = DEFAULT_TIMEOUT_S) -> Any:
        rec = self._request("GET", url, params, headers, timeout)
        if rec["status"] >= 400:
            raise SourceError(f"{self.source}: HTTP {rec['status']} for {rec['url']}", rec["status"])
        try:
            return json.loads(rec["text"])
        except json.JSONDecodeError as e:
            raise SourceError(f"{self.source}: non-JSON response from {rec['url']}: {e}") from e

    def get_text(self, url: str, params: dict | None = None, headers: dict | None = None,
                 timeout: float = DEFAULT_TIMEOUT_S) -> str:
        rec = self._request("GET", url, params, headers, timeout)
        if rec["status"] >= 400:
            raise SourceError(f"{self.source}: HTTP {rec['status']} for {rec['url']}", rec["status"])
        return rec["text"]

    def probe(self, url: str, timeout: float = DEFAULT_TIMEOUT_S) -> dict:
        """HEAD-probe a URL (falling back to a streamed GET when HEAD is refused).

        Returns {"status", "content_type", "url"(final, after redirects)}. Cached.
        """
        rec = self._request("HEAD", url, None, None, timeout)
        if rec["status"] in (405, 403, 501):
            rec = self._request("GET", url, None, None, timeout, stream_probe=True)
        return {"status": rec["status"], "content_type": rec["content_type"], "url": rec["url"]}

    # ----- internals -----

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def _throttle(self) -> None:
        last = _last_call.get(self.source)
        if last is not None:
            wait = self.sleep_s - (time.monotonic() - last)
            if wait > 0:
                time.sleep(wait)
        _last_call[self.source] = time.monotonic()

    def _request(self, method: str, url: str, params: dict | None, headers: dict | None,
                 timeout: float, stream_probe: bool = False) -> dict:
        key = _cache_key(method + ("-probe" if stream_probe else ""), url, params)
        path = self._cache_path(key)
        if path.exists():
            self.cache_hits += 1
            return json.loads(path.read_text(encoding="utf-8"))

        for attempt in range(RETRIES + 1):
            self._throttle()
            self.live_requests += 1
            log.debug("%s %s %s params=%s", self.source, method, url, params)
            try:
                resp = self.session.request(method, url, params=params, headers=headers,
                                            timeout=timeout, allow_redirects=True, stream=stream_probe)
                try:
                    # requests defaults text/* without a charset to ISO-8859-1; the
                    # scholarly APIs are all UTF-8 (arXiv OAI sends text/xml with no charset).
                    if "charset" not in (resp.headers.get("Content-Type") or "").lower():
                        resp.encoding = "utf-8"
                    text = "" if stream_probe else resp.text
                finally:
                    resp.close()
            except requests.Timeout as e:
                raise SourceError(f"{self.source}: timeout after {timeout}s for {url}") from e
            except requests.RequestException as e:
                raise SourceError(f"{self.source}: request failed for {url}: {e}") from e
            if resp.status_code == 429:
                self.saw_rate_limit = True
            if resp.status_code in TRANSIENT and attempt < RETRIES:
                wait = max(self.sleep_s, RETRY_BACKOFF_S * (attempt + 1))
                log.info("%s: HTTP %d, retrying in %.0fs", self.source, resp.status_code, wait)
                time.sleep(wait)
                continue
            break

        rec = {
            "method": method, "requested_url": url, "params": params, "url": resp.url,
            "status": resp.status_code, "content_type": resp.headers.get("Content-Type", ""),
            "text": text, "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        # Cache stable outcomes only; rate limits and server errors are retried next run.
        if resp.status_code < 500 and resp.status_code not in TRANSIENT:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
        return rec


class SourceAdapter:
    """Base for every source. Construction fails (SourceUnavailable) when the source can't run."""

    name: str = ""
    requires_key: bool = False
    discovers: bool = False      # True if search() yields records for a free-text query

    def __init__(self, cfg: Config) -> None:
        if cfg.offline:
            raise SourceUnavailable(f"{self.name}: offline mode is enabled (C3)")
        if not cfg.contact_email:
            raise SourceUnavailable(f"{self.name}: CONTACT_EMAIL missing from .env (SPEC 5.5)")
        if self.requires_key and not cfg.has_key(self.name):
            raise SourceUnavailable(
                f"{self.name}: no {KEYED_SOURCES[self.name]} in .env; source skipped")
        self.cfg = cfg
        self.api_key = cfg.api_keys.get(self.name)
        self.client = CachedClient(cfg, self.name)

    def search(self, query: str, limit: int) -> Iterable[Paper]:
        """Yield Paper records for a free-text query. Non-discovery sources yield nothing."""
        return []


def available_adapters(cfg: Config, adapter_classes: Iterable[type[SourceAdapter]]) -> list[SourceAdapter]:
    """Instantiate every adapter that can run; log and skip the rest (SPEC 5.2)."""
    out: list[SourceAdapter] = []
    for cls in adapter_classes:
        try:
            out.append(cls(cfg))
        except SourceUnavailable as e:
            log.info("skipping source: %s", e)
    return out
