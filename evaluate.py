"""Metrics harness: loads results/eval_set.jsonl, runs a config, appends a row to results/runs.csv.

Phase 3: CSV logger (metrics logged from Phase 3 onward, SPEC 11.3).
Phase 5: retrieval metrics (recall@5, recall@10, MRR, latency) for any config.
Phase 6: generation metrics (citation integrity, tokens/s) when generate_answers=True.

Relevance is judged at the paper level: a retrieved chunk counts as a hit when its
paper_id is one of the question's ``paper_ids`` (SPEC 9.1).

Run ``python evaluate.py --mode all`` for one row per retrieval mode.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import statistics
import time
from pathlib import Path

from config import RETRIEVAL_MODES, Config, load_config, peak_rss_mb

log = logging.getLogger(__name__)

# Column order for runs.csv. New columns are appended at the end; missing values are blank.
BASE_COLUMNS = ["timestamp", "phase", "run_name", "chunk_size_tokens", "chunk_overlap_tokens", "retrieval_mode",
                "rerank", "context_chunks", "llm_model", "embedding_model", "vector_backend",
                "n_papers", "n_chunks", "index_size_mb", "index_build_s", "embed_s", "n_embedded",
                "recall_at_5", "recall_at_10", "mrr", "p50_latency_s", "p95_latency_s",
                "citation_integrity", "answers_with_citation", "tokens_per_s", "peak_rss_mb", "notes"]
EVAL_FIELDS = {"id", "question", "answer", "paper_ids"}


def log_run(cfg: Config, row: dict, path: Path | None = None) -> Path:
    """Append one metrics row to results/runs.csv (created with a header on first use).

    Any key not in the existing header is added as a new column; earlier rows keep blanks.
    """
    path = path or cfg.runs_csv
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "peak_rss_mb": round(peak_rss_mb(), 1), **row}
    existing: list[dict] = []
    columns = list(BASE_COLUMNS)
    if path.exists():
        with path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            existing = list(reader)
            columns = list(reader.fieldnames or columns)
    for k in row:
        if k not in columns:
            columns.append(k)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=columns)
        w.writeheader()
        for r in existing:
            w.writerow({c: r.get(c, "") for c in columns})
        w.writerow({c: row.get(c, "") for c in columns})
    return path


def read_runs(cfg: Config) -> list[dict]:
    if not cfg.runs_csv.exists():
        return []
    with cfg.runs_csv.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


# ----- eval set (SPEC 9.1) -----

def load_eval_set(cfg: Config, path: Path | None = None) -> list[dict]:
    """Load and validate results/eval_set.jsonl. Entries carry ``verified`` (user-checked) flags."""
    path = path or cfg.eval_set_path
    if not path.exists():
        raise FileNotFoundError(f"eval set not found: {path} (SPEC 9.1: user writes or verifies it)")
    items: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for n, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{n}: invalid JSON: {e}") from e
            missing = EVAL_FIELDS - set(obj)
            if missing:
                raise ValueError(f"{path}:{n}: missing fields {sorted(missing)}")
            if not isinstance(obj["paper_ids"], list) or not obj["paper_ids"]:
                raise ValueError(f"{path}:{n}: paper_ids must be a non-empty list")
            obj.setdefault("verified", False)
            items.append(obj)
    if not items:
        raise ValueError(f"{path} is empty")
    unverified = sum(1 for i in items if not i["verified"])
    if unverified:
        log.warning("eval set: %d/%d questions are UNVERIFIED (agent-drafted); metrics are provisional",
                    unverified, len(items))
    return items


def eval_set_note(items: list[dict]) -> str:
    unverified = sum(1 for i in items if not i.get("verified"))
    return f"eval_set n={len(items)}" + (f" UNVERIFIED {unverified}/{len(items)}" if unverified else " verified")


# ----- retrieval metrics -----

def retrieval_metrics(ranked_paper_ids: list[list[str]], relevant: list[set[str]], ks=(5, 10)) -> dict:
    """recall@k (fraction of questions with a relevant paper in the top k) and MRR over paper ranks.

    ``ranked_paper_ids[i]`` is the paper_id of each retrieved chunk in rank order for question i
    (duplicates allowed; the first occurrence of a paper defines its rank).
    """
    n = len(ranked_paper_ids)
    if n == 0:
        return {f"recall_at_{k}": 0.0 for k in ks} | {"mrr": 0.0}
    out: dict[str, float] = {}
    for k in ks:
        out[f"recall_at_{k}"] = sum(1 for r, rel in zip(ranked_paper_ids, relevant)
                                    if any(p in rel for p in _first_occurrences(r)[:k])) / n
    rr = []
    for r, rel in zip(ranked_paper_ids, relevant):
        papers = _first_occurrences(r)
        rank = next((i + 1 for i, p in enumerate(papers) if p in rel), None)
        rr.append(1.0 / rank if rank else 0.0)
    out["mrr"] = sum(rr) / n
    return {k: round(v, 4) for k, v in out.items()}


def _norm_ws(s: str) -> str:
    return " ".join(s.lower().split())


def chunk_contains_quote(chunk_text: str, quote: str) -> bool:
    """Whitespace-insensitive substring test; tolerates the chunk's section prefix and overlap."""
    q = _norm_ws(quote)
    return bool(q) and q in _norm_ws(chunk_text)


def _first_occurrences(seq: list[str]) -> list[str]:
    seen: set[str] = set()
    out = []
    for s in seq:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def run_eval(cfg: Config, run_name: str, mode: str | None = None, rerank: bool | None = None,
             k: int = 10, generate_answers: bool = False, log: bool = True, phase: int = 5,
             eval_path: Path | None = None, limit: int | None = None) -> dict:
    """Evaluate one configuration on the eval set and append a metrics row to cfg.runs_csv."""
    import retrieve
    items = load_eval_set(cfg, eval_path)
    if limit:
        items = items[:limit]
    mode = mode or cfg.retrieval_mode
    use_rerank = cfg.rerank if rerank is None else rerank
    ranked, relevant, lat = [], [], []
    q_ranked, q_relevant = [], []          # chunk-level: the retrieved chunk must contain the source quote
    per_question = []
    for it in items:
        res, dt = retrieve.timed_retrieve(cfg, it["question"], mode=mode, k=k, rerank_=use_rerank)
        ranked.append([r.chunk.paper_id for r in res])
        relevant.append(set(it["paper_ids"]))
        lat.append(dt)
        quote = it.get("source_quote") or ""
        quote_rank = None
        if quote:
            hits = ["hit" if chunk_contains_quote(r.chunk.text, quote) else f"miss{r.rank}" for r in res]
            q_ranked.append(hits)
            q_relevant.append({"hit"})
            quote_rank = next((r.rank for r, h in zip(res, hits) if h == "hit"), None)
        per_question.append({"id": it["id"], "hit_rank": next(
            (r.rank for r in res if r.chunk.paper_id in it["paper_ids"]), None),
            "quote_rank": quote_rank, "latency_s": round(dt, 4)})
    metrics = retrieval_metrics(ranked, relevant)
    if q_ranked:
        qm = retrieval_metrics(q_ranked, q_relevant)
        metrics.update({"chunk_recall_at_5": qm["recall_at_5"], "chunk_recall_at_10": qm["recall_at_10"],
                        "chunk_mrr": qm["mrr"]})
    row = {
        "phase": phase, "run_name": run_name, "chunk_size_tokens": cfg.chunk_size_tokens,
        "chunk_overlap_tokens": cfg.chunk_overlap_tokens, "retrieval_mode": mode, "rerank": use_rerank,
        "context_chunks": cfg.context_chunks, "embedding_model": cfg.embedding_model,
        **metrics,
        "p50_latency_s": round(statistics.median(lat), 4),
        "p95_latency_s": round(sorted(lat)[max(0, int(round(0.95 * len(lat))) - 1)], 4),
        "notes": eval_set_note(items) + (f" (first {limit})" if limit else ""),
    }
    if generate_answers:
        row["phase"] = max(phase, 6)
        row.update(_generation_metrics(cfg, items, mode, use_rerank, run_name))
    if log:
        log_run(cfg, row)
    row["per_question"] = per_question
    return row


def _generation_metrics(cfg: Config, items: list[dict], mode: str, use_rerank: bool,
                        run_name: str = "eval") -> dict:
    """Generate an answer per question; report citation integrity, citation coverage, tokens/s.

    Answers are written to results/answers_<run_name>.jsonl for the manual correctness grade (SPEC 9).
    """
    import fetch
    import generate
    import retrieve
    generate.ensure_model(cfg, cfg.llm_model)
    papers = {p.paper_id: p for p in fetch.load_corpus(cfg)}
    integrity, with_cit, tps, gen_lat, out, failures, thinking = [], 0, [], [], [], 0, []
    for it in items:
        hits = retrieve.retrieve(cfg, it["question"], mode=mode, rerank_=use_rerank)
        try:
            ans = generate.generate(cfg, it["question"], hits, papers=papers)
        except generate.GenerationError as e:          # counted, never hidden
            failures += 1
            log.warning("%s: generation failed: %s", it["id"], e)
            out.append({"id": it["id"], "question": it["question"], "expected": it["answer"], "answer": "",
                        "error": str(e), "cited": [], "invalid": [], "context": [r.chunk.chunk_id for r in hits],
                        "model": cfg.llm_model, "grade": None})
            continue
        integrity.append(ans.citation_integrity)
        with_cit += int(bool(ans.cited_ids))
        tps.append(ans.tokens_per_s)
        gen_lat.append(ans.latency_s)
        thinking.append(ans.thinking_chars)
        out.append({"id": it["id"], "question": it["question"], "expected": it["answer"],
                    "answer": ans.text, "cited": ans.cited_ids, "invalid": ans.invalid_citations,
                    "context": ans.retrieved_ids, "model": ans.model, "tokens_per_s": round(ans.tokens_per_s, 2),
                    "latency_s": ans.latency_s, "completion_tokens": ans.completion_tokens,
                    "thinking_chars": ans.thinking_chars, "grade": None})
    path = cfg.results_dir / f"answers_{run_name}.jsonl"
    path.write_text("".join(json.dumps(o, ensure_ascii=False) + "\n" for o in out), encoding="utf-8")
    llm_mb = generate.resident_mb(cfg, cfg.llm_model)
    generate.unload_model(cfg, cfg.llm_model)
    return {
        "llm_model": cfg.llm_model,
        "citation_integrity": round(sum(integrity) / len(integrity), 4) if integrity else 0.0,
        "answers_with_citation": round(with_cit / len(items), 4) if items else 0.0,
        "tokens_per_s": round(statistics.median(tps), 2) if tps else 0.0,
        "gen_p50_latency_s": round(statistics.median(gen_lat), 2) if gen_lat else 0.0,
        "gen_failures": failures, "think": cfg.think_for(cfg.llm_model),
        "thinking_chars_p50": round(statistics.median(thinking)) if thinking else 0,
        "llm_resident_mb": llm_mb, "answers_file": path.name,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run the eval harness for one or all retrieval modes")
    ap.add_argument("--mode", default="all", help="bm25 | dense | hybrid | all")
    ap.add_argument("--rerank", action="store_true")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--generate", action="store_true", help="also generate answers (Phase 6)")
    ap.add_argument("--model", help="LLM to use with --generate (default: generation.model)")
    ap.add_argument("--limit", type=int, help="only the first N questions")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    cfg = load_config()
    if args.model:
        cfg = cfg.with_overrides(llm_model=args.model)
    modes = list(RETRIEVAL_MODES) if args.mode == "all" else [args.mode]
    for m in modes:
        name = f"eval_{m}{'+rerank' if args.rerank else ''}" + (f"_{cfg.llm_model}" if args.generate else "")
        row = run_eval(cfg, name, mode=m, rerank=args.rerank, k=args.k, generate_answers=args.generate,
                       limit=args.limit)
        if args.generate:
            print(f"  generation: integrity={row['citation_integrity']:.3f} with_citation={row['answers_with_citation']:.3f} "
                  f"failures={row['gen_failures']} tok/s={row['tokens_per_s']} gen_p50={row['gen_p50_latency_s']}s "
                  f"think={row['think']} thinking_chars_p50={row['thinking_chars_p50']} "
                  f"resident={row['llm_resident_mb']} MB -> {row['answers_file']}")
        chunk = (f" | chunk: r@5={row['chunk_recall_at_5']:.3f} r@10={row['chunk_recall_at_10']:.3f} "
                 f"mrr={row['chunk_mrr']:.3f}") if "chunk_mrr" in row else ""
        print(f"{row['run_name']:20} paper: r@5={row['recall_at_5']:.3f} r@10={row['recall_at_10']:.3f} "
              f"mrr={row['mrr']:.3f}{chunk} | p50={row['p50_latency_s']:.3f}s p95={row['p95_latency_s']:.3f}s  "
              f"[{row['notes']}]")
    print(f"logged to {cfg.runs_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
