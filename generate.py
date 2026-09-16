"""Ollama client, context assembly, citation-forced prompt, integrity check (SPEC 7.2-7.3).

The only LLM client in the codebase. It speaks to the loopback Ollama endpoint that
config.validate() pins, over the standard library's http.client and nothing else
(no requests / httpx / urllib here, C1). Chunk text never leaves the machine.

Citations: sources are presented to the model as [S1]..[Sn]; the model must end each
claim with one. Cited ids are mapped back to chunk ids. Any citation that does not
resolve to a retrieved chunk is kept and counted (Answer.invalid_citations), never
silently dropped.
"""

from __future__ import annotations

import http.client
import json
import logging
import re
import time
from collections.abc import Sequence

from config import Config
from models import Answer, Retrieved

log = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a careful research assistant. Answer the question using ONLY the numbered sources provided. "
    "Every sentence that states a fact must end with a citation in square brackets giving the source id "
    "exactly as shown, for example [S2] or [S1][S3]. The only valid citations are the S-prefixed ids of the "
    "sources given to you; never invent other ids and never output bare numbers in brackets. "
    "If the sources do not contain the answer, say so in one sentence and cite nothing. "
    "Be concise: two to four sentences. Do not use markdown."
)
_TOKENS_IN_BRACKETS = re.compile(r"\[([^\]]{1,40})\]")
# Small models sometimes write "(S2)" instead of "[S2]"; accept S-prefixed ids in parentheses too
# (never bare numbers there, which are ordinary prose).
_PAREN_S_IDS = re.compile(r"\((\s*S\d{1,3}(?:\s*[,;]\s*S\d{1,3})*\s*)\)", re.I)
# Papers' own in-text reference markers ("[12]", "[3, 7]", "[4-6]") are meaningless without the
# excised reference list and small models copy them as citations; drop them from source text.
_REF_MARKERS = re.compile(r"\s?\[\s*\d{1,3}(?:\s*[,;–-]\s*\d{1,3})*\s*\]")


class GenerationError(RuntimeError):
    """Ollama unreachable, model missing, or malformed response."""


# ----- Ollama transport (loopback only) -----

def _host_port(cfg: Config) -> tuple[str, int]:
    url = cfg.ollama_host
    rest = url.split("://", 1)[1] if "://" in url else url
    hostport = rest.split("/", 1)[0]
    if ":" in hostport:
        host, port = hostport.rsplit(":", 1)
        return host, int(port)
    return hostport, 11434


def ollama_call(cfg: Config, method: str, path: str, payload: dict | None = None,
                timeout: float | None = None) -> dict:
    host, port = _host_port(cfg)
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    conn = http.client.HTTPConnection(host, port, timeout=timeout or cfg.llm_timeout_s)
    try:
        conn.request(method, path, body=body, headers={"Content-Type": "application/json"} if body else {})
        resp = conn.getresponse()
        raw = resp.read()
    except (ConnectionError, OSError, http.client.HTTPException) as e:
        raise GenerationError(f"Ollama not reachable at {cfg.ollama_host} ({type(e).__name__}: {e}). "
                              "Start it with `ollama serve`.") from e
    finally:
        conn.close()
    if resp.status >= 400:
        raise GenerationError(f"Ollama {method} {path} -> HTTP {resp.status}: {raw[:300].decode('utf-8', 'replace')}")
    try:
        return json.loads(raw) if raw else {}
    except json.JSONDecodeError as e:
        raise GenerationError(f"Ollama returned non-JSON for {path}: {raw[:200]!r}") from e


def available_models(cfg: Config) -> list[str]:
    return [m.get("name", "") for m in ollama_call(cfg, "GET", "/api/tags", timeout=10).get("models", [])]


def ensure_model(cfg: Config, model: str) -> None:
    names = available_models(cfg)
    if model not in names and f"{model}:latest" not in names:
        raise GenerationError(f"model {model!r} not present in Ollama (have: {names}); run `ollama pull {model}`")


def loaded_models(cfg: Config) -> list[dict]:
    """Models currently resident in Ollama with their memory footprint (bytes) from /api/ps."""
    return ollama_call(cfg, "GET", "/api/ps", timeout=10).get("models", [])


def resident_mb(cfg: Config, model: str | None = None) -> float:
    total = 0
    for m in loaded_models(cfg):
        if model is None or m.get("name", "").startswith(model.split(":")[0]):
            total += int(m.get("size", 0))
    return round(total / (1024 * 1024), 1)


def warm_up(cfg: Config, model: str, keep_alive: str = "30m") -> float:
    """Load a model without generating so the (possibly minute-long) cold load is not inside a timed answer."""
    t0 = time.perf_counter()
    ollama_call(cfg, "POST", "/api/generate", {"model": model, "prompt": "", "keep_alive": keep_alive,
                                               "options": {"num_ctx": num_ctx(cfg)}},   # same ctx as generate(): no reload
                timeout=cfg.llm_timeout_s)
    return round(time.perf_counter() - t0, 1)


def unload_model(cfg: Config, model: str) -> None:
    """Release a model from RAM (sequential load/unload keeps the machine under the ceiling, SPEC 12)."""
    try:
        ollama_call(cfg, "POST", "/api/generate", {"model": model, "keep_alive": 0}, timeout=30)
    except GenerationError as e:
        log.info("unload %s: %s", model, e)


def num_ctx(cfg: Config) -> int:
    """Context window requested from Ollama; identical for warm-up and answers so the model is loaded once."""
    return max(2048, cfg.token_budget + 1024)


# ----- context assembly (SPEC 7.2) -----

def assemble_context(cfg: Config, retrieved: Sequence[Retrieved], budget: int | None = None,
                     max_chunks: int | None = None) -> list[Retrieved]:
    """Keep chunks in rank order while the cumulative token count fits the budget; drop from the tail."""
    budget = budget or cfg.token_budget
    max_chunks = max_chunks or cfg.context_chunks
    out: list[Retrieved] = []
    used = 0
    for r in retrieved[:max_chunks]:
        if used + r.chunk.n_tokens > budget:
            break
        out.append(r)
        used += r.chunk.n_tokens
    if not out and retrieved:                 # never send an empty context if something was retrieved
        out = [retrieved[0]]
    return out


def source_label(r: Retrieved, papers: dict | None = None) -> str:
    """Human citation form, e.g. 'Lewis 2020, §3.2 Method, p.4'."""
    c = r.chunk
    who = papers[c.paper_id].citation_label() if papers and c.paper_id in papers else c.paper_id
    pages = f"p.{c.page_start}" if c.page_start == c.page_end else f"pp.{c.page_start}-{c.page_end}"
    return f"{who}, §{c.section}, {pages}"


def clean_source_text(text: str) -> str:
    return _REF_MARKERS.sub("", text).strip()


def build_prompt(cfg: Config, query: str, context: Sequence[Retrieved], papers: dict | None = None) -> str:
    parts = ["Sources:"]
    for i, r in enumerate(context, start=1):
        parts.append(f"[S{i}] ({source_label(r, papers)})\n{clean_source_text(r.chunk.text)}")
    parts.append(f"Question: {query}\nAnswer with citations.")
    return "\n\n".join(parts)


# ----- citations (SPEC 7.3) -----

def parse_citations(text: str, n_sources: int) -> tuple[list[int], list[str]]:
    """Return (valid source indexes 1..n in order of first appearance, raw invalid citation tokens)."""
    valid: list[int] = []
    invalid: list[str] = []
    matches = [(m.start(), m.group(1)) for m in _TOKENS_IN_BRACKETS.finditer(text)]
    matches += [(m.start(), m.group(1)) for m in _PAREN_S_IDS.finditer(text)]
    for _, inner in sorted(matches):
        inner = inner.strip()
        nums = re.findall(r"S?(\d{1,3})", inner, re.I) if re.fullmatch(r"[\sS0-9,;]+", inner, re.I) else []
        if not nums:
            continue
        for n in nums:
            idx = int(n)
            if 1 <= idx <= n_sources:
                if idx not in valid:
                    valid.append(idx)
            else:
                invalid.append(f"S{idx}")
    return valid, invalid


# ----- generation -----

def _payload(cfg: Config, model: str, query: str, context: Sequence[Retrieved], papers: dict | None,
             think: bool, stream: bool) -> dict:
    return {
        "model": model, "stream": stream, "think": think,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                     {"role": "user", "content": build_prompt(cfg, query, context, papers)}],
        "options": {"temperature": 0.0, "num_ctx": num_ctx(cfg), "seed": 0, "num_predict": cfg.max_tokens},
    }


def _finish(cfg: Config, model: str, context: Sequence[Retrieved], text: str, thinking: str, resp: dict,
            latency: float, think: bool) -> Answer:
    text = text.strip()
    if not text:
        raise GenerationError(
            f"empty answer from {model} (done_reason={resp.get('done_reason')}, eval_count={resp.get('eval_count')}, "
            f"thinking_chars={len(thinking)}); raise generation.max_tokens or disable thinking")
    valid_idx, invalid = parse_citations(text, len(context))
    cited = [context[i - 1].chunk.chunk_id for i in valid_idx] + invalid
    return Answer(
        text=text, cited_ids=cited, retrieved_ids=[r.chunk.chunk_id for r in context], model=model,
        prompt_tokens=int(resp.get("prompt_eval_count", 0)), completion_tokens=int(resp.get("eval_count", 0)),
        latency_s=round(latency, 3),
        eval_s=round(int(resp.get("eval_duration", 0)) / 1e9, 3),
        prompt_eval_s=round(int(resp.get("prompt_eval_duration", 0)) / 1e9, 3),
        load_s=round(int(resp.get("load_duration", 0)) / 1e9, 3),
        thinking_chars=len(thinking.strip()), think=think,
    )


def generate(cfg: Config, query: str, retrieved: Sequence[Retrieved], model: str | None = None,
             papers: dict | None = None, think: bool | None = None) -> Answer:
    """Produce a cited answer over the retrieved chunks. Answer.invalid_citations lists out-of-set ids.

    Raises GenerationError when the model returns no answer text (e.g. the token cap was spent on
    reasoning); callers that aggregate metrics record that as a failed answer.
    """
    model = model or cfg.llm_model
    think = cfg.think_for(model) if think is None else think
    context = assemble_context(cfg, retrieved)
    t0 = time.perf_counter()
    resp = ollama_call(cfg, "POST", "/api/chat", _payload(cfg, model, query, context, papers, think, stream=False))
    latency = time.perf_counter() - t0
    msg = resp.get("message") or {}
    return _finish(cfg, model, context, msg.get("content") or "", msg.get("thinking") or "", resp, latency, think)


def generate_stream(cfg: Config, query: str, retrieved: Sequence[Retrieved], model: str | None = None,
                    papers: dict | None = None, think: bool | None = None):
    """Streaming twin of generate(): yields ("thinking", delta) / ("token", delta) events, then ("answer", Answer).

    Same loopback http.client transport; the NDJSON stream is read line by line. A final
    GenerationError (empty answer) is raised after the stream ends, as in generate().
    """
    model = model or cfg.llm_model
    think = cfg.think_for(model) if think is None else think
    context = assemble_context(cfg, retrieved)
    host, port = _host_port(cfg)
    body = json.dumps(_payload(cfg, model, query, context, papers, think, stream=True)).encode("utf-8")
    conn = http.client.HTTPConnection(host, port, timeout=cfg.llm_timeout_s)
    t0 = time.perf_counter()
    text, thinking, final = [], [], {}
    try:
        conn.request("POST", "/api/chat", body=body, headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        if resp.status >= 400:
            raise GenerationError(f"Ollama POST /api/chat -> HTTP {resp.status}: {resp.read()[:300].decode('utf-8', 'replace')}")
        while True:
            line = resp.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            msg = obj.get("message") or {}
            if msg.get("thinking"):
                thinking.append(msg["thinking"])
                yield ("thinking", msg["thinking"])
            if msg.get("content"):
                text.append(msg["content"])
                yield ("token", msg["content"])
            if obj.get("done"):
                final = obj
                break
    except (ConnectionError, OSError, http.client.HTTPException) as e:
        raise GenerationError(f"Ollama not reachable at {cfg.ollama_host} ({type(e).__name__}: {e}). "
                              "Start it with `ollama serve`.") from e
    finally:
        conn.close()
    latency = time.perf_counter() - t0
    yield ("answer", _finish(cfg, model, context, "".join(text), "".join(thinking), final, latency, think))


def answer_query(cfg: Config, query: str, mode: str | None = None, rerank: bool | None = None,
                 model: str | None = None) -> tuple[Answer, list[Retrieved]]:
    """Retrieve then generate. Convenience for the UI and the eval harness."""
    import fetch
    import retrieve
    hits = retrieve.retrieve(cfg, query, mode=mode, rerank_=rerank)
    papers = {p.paper_id: p for p in fetch.load_corpus(cfg)}
    return generate(cfg, query, hits, model=model, papers=papers), hits


def benchmark_models(cfg: Config, query: str, models: Sequence[str] | None = None, log: bool = True) -> list[dict]:
    """tokens/s, latency and resident memory for each model on one query (Phase 6 gate)."""
    from config import peak_rss_mb
    from evaluate import log_run
    models = list(models or [cfg.llm_model, cfg.llm_fallback_model])
    rows = []
    for m in models:
        ensure_model(cfg, m)
        answer, hits = answer_query(cfg, query, model=m)
        row = {"phase": 6, "run_name": f"gen_{m}", "llm_model": m, "retrieval_mode": cfg.retrieval_mode,
               "rerank": cfg.rerank, "context_chunks": cfg.context_chunks,
               "tokens_per_s": round(answer.tokens_per_s, 2), "prompt_tokens_per_s": round(answer.prompt_tokens_per_s, 2),
               "p50_latency_s": answer.latency_s, "load_s": answer.load_s,
               "citation_integrity": answer.citation_integrity, "answers_with_citation": int(bool(answer.cited_ids)),
               "llm_resident_mb": resident_mb(cfg, m), "peak_rss_mb": round(peak_rss_mb(), 1),
               "think": answer.think, "thinking_chars": answer.thinking_chars,
               "notes": f"1 query, {answer.prompt_tokens} prompt + {answer.completion_tokens} completion tokens "
                        f"(think={answer.think}, {answer.thinking_chars} reasoning chars), "
                        f"{len(answer.retrieved_ids)} context chunks"}
        row["machine_peak_mb"] = round(row["llm_resident_mb"] + row["peak_rss_mb"], 1)
        if log:
            log_run(cfg, row)
        rows.append(row | {"answer": answer.text})
        unload_model(cfg, m)
    return rows


if __name__ == "__main__":
    import sys
    from config import load_config
    logging.basicConfig(level=logging.WARNING)
    c = load_config()
    if "--bench" in sys.argv:
        q = "What are the two main components of retrieval-augmented generation and how do they interact?"
        for row in benchmark_models(c, q):
            print(f"{row['llm_model']:12} gen {row['tokens_per_s']:6.2f} tok/s  prompt {row['prompt_tokens_per_s']:6.2f} tok/s  "
                  f"wall {row['p50_latency_s']:6.1f}s (load {row['load_s']:.1f}s)  resident {row['llm_resident_mb']:7.1f} MB  "
                  f"our peak {row['peak_rss_mb']:6.1f} MB  integrity {row['citation_integrity']:.2f}\n  -> {row['answer'][:200]}")
        print(f"logged to {c.runs_csv}")
        sys.exit(0)
    q = " ".join(sys.argv[1:]) or "How does retrieval-augmented generation reduce hallucination?"
    ans, hits = answer_query(c, q)
    print(ans.text)
    print(f"\ncited: {ans.cited_ids}\ninvalid: {ans.invalid_citations}\n{ans.completion_tokens} tokens "
          f"in {ans.latency_s}s ({ans.tokens_per_s:.1f} tok/s) model={ans.model}")
