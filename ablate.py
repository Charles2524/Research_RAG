"""Config sweep runner: one dimension at a time from a fixed baseline -> results/runs.csv (SPEC 9).

    python -m ablate                # full sweep with generation (hours on CPU with the 4B model)
    python -m ablate --fast         # retrieval metrics only (minutes)
    python -m ablate --limit 5      # first 5 eval questions per config
    python -m ablate --only chunk_256,rerank_on
    python -m ablate --dry-run      # list the configs
    python -m ablate --table        # print / write the results table from runs.csv

The baseline is config.yaml. A config that raises is recorded as a CRASHED row and the sweep
continues; the exit code is non-zero if any config crashed (Phase 7 gate wants zero).
Runs work unchanged in offline mode (C3): nothing here touches the network.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import chunk as chunk_mod
import evaluate
import retrieve
from config import Config, load_config

log = logging.getLogger(__name__)

# (run_name, axis, overrides). Overlap scales with chunk size (1/8) so it is not a second variable.
AXES: list[tuple[str, str, dict]] = [
    ("baseline", "baseline", {}),
    ("chunk_256", "chunk_size", {"chunk_size_tokens": 256, "chunk_overlap_tokens": 32}),
    ("chunk_1024", "chunk_size", {"chunk_size_tokens": 1024, "chunk_overlap_tokens": 128}),
    ("mode_bm25", "retrieval_mode", {"retrieval_mode": "bm25"}),
    ("mode_dense", "retrieval_mode", {"retrieval_mode": "dense"}),
    ("rerank_on", "rerank", {"rerank": True}),
    ("context_3", "context_chunks", {"context_chunks": 3}),
    ("context_8", "context_chunks", {"context_chunks": 8}),
    ("model_other", "model", {"llm_model": "__fallback__"}),
    ("quant_q8", "quantization", {"llm_model": "qwen3:4b-q8_0"}),
]
TABLE_COLUMNS = ["run_name", "chunk_size_tokens", "retrieval_mode", "rerank", "context_chunks", "llm_model",
                 "chunk_recall_at_5", "chunk_recall_at_10", "chunk_mrr", "recall_at_5", "mrr",
                 "citation_integrity", "answers_with_citation", "gen_failures", "tokens_per_s",
                 "p50_latency_s", "p95_latency_s", "gen_p50_latency_s", "llm_resident_mb", "peak_rss_mb",
                 "index_build_s", "notes"]


def other_model(base: Config) -> str:
    """The model axis compares the baseline against the other configured model (4B <-> 1.7B)."""
    return base.llm_fallback_model if base.llm_model != base.llm_fallback_model else base.llm_model


def sweep(base: Config, only: set[str] | None = None) -> list[tuple[str, str, Config]]:
    """Materialise the sweep: each entry differs from the baseline in exactly one axis."""
    out = []
    for name, axis, ov in AXES:
        if only and name not in only:
            continue
        ov = {k: (other_model(base) if v == "__fallback__" else v) for k, v in ov.items()}
        out.append((name, axis, base.with_overrides(**ov)))
    return out


def _model_available(cfg: Config, model: str) -> bool:
    import generate
    try:
        names = generate.available_models(cfg)
    except generate.GenerationError:
        return False
    return model in names or f"{model}:latest" in names


def ensure_chunking(cfg: Config) -> dict | None:
    """Re-chunk + re-index if the stored chunk size differs from cfg (cheap when embeddings are cached)."""
    import index
    conn = index.connect(cfg.index_path)
    try:
        cur = (index.get_meta(conn, "chunk_size_tokens"), index.get_meta(conn, "chunk_overlap_tokens"))
    finally:
        conn.close()
    if cur == (str(cfg.chunk_size_tokens), str(cfg.chunk_overlap_tokens)):
        return None
    log.info("re-chunking corpus at %d/%d tokens", cfg.chunk_size_tokens, cfg.chunk_overlap_tokens)
    chunk_mod.chunk_all(cfg, cfg.chunk_size_tokens, cfg.chunk_overlap_tokens)
    conn = retrieve.get_conn(cfg)                     # cached connection; rebuild vectors from the embedding cache
    stats = index.ensure_index(cfg, conn) or index.build_index(cfg, conn)
    return stats


def run_sweep(base: Config, fast: bool = False, limit: int | None = None, only: set[str] | None = None,
              log_rows: bool = True) -> list[dict]:
    rows: list[dict] = []
    configs = sweep(base, only)
    t_all = time.monotonic()
    for i, (name, axis, cfg) in enumerate(configs, start=1):
        t0 = time.monotonic()
        print(f"[{i}/{len(configs)}] {name} ({axis}) ...", flush=True)
        try:
            if not fast and cfg.llm_model != base.llm_model and not _model_available(cfg, cfg.llm_model):
                # SPEC 9: the Q8 axis runs "RAM permitting"; an absent model is a documented skip, not a crash
                row = {"phase": 7, "run_name": f"ablate_{name}", "axis": axis, "llm_model": cfg.llm_model,
                       "notes": f"SKIPPED: model {cfg.llm_model} not present in Ollama (ollama pull {cfg.llm_model})"}
                row["run_s"] = 0.0
                if log_rows:
                    evaluate.log_run(cfg, row)
                rows.append(row)
                print(f"    SKIPPED: {row['notes']}", flush=True)
                continue
            idx = ensure_chunking(cfg)
            row = evaluate.run_eval(cfg, f"ablate_{name}", generate_answers=not fast, phase=7, limit=limit,
                                    log=False)
            row.pop("per_question", None)
            if idx:
                row["index_build_s"] = idx.get("index_build_s", "")
                row["index_size_mb"] = idx.get("index_size_mb", "")
            row["axis"] = axis
            row["notes"] = f"{row.get('notes', '')}; axis={axis}; {'retrieval only' if fast else 'with generation'}"
        except Exception as e:                        # crash is data, not an abort
            log.exception("%s crashed", name)
            row = {"phase": 7, "run_name": f"ablate_{name}", "axis": axis, "retrieval_mode": cfg.retrieval_mode,
                   "chunk_size_tokens": cfg.chunk_size_tokens, "rerank": cfg.rerank, "context_chunks": cfg.context_chunks,
                   "llm_model": cfg.llm_model, "notes": f"CRASHED: {type(e).__name__}: {str(e)[:200]}"}
        row["run_s"] = round(time.monotonic() - t0, 1)
        if log_rows:
            evaluate.log_run(cfg, row)
        rows.append(row)
        retrieve.unload_reranker()
        status = "CRASHED" if str(row.get("notes", "")).startswith("CRASHED") else "ok"
        print(f"    {status} in {row['run_s']}s  chunk_r@5={row.get('chunk_recall_at_5', '-')} "
              f"mrr={row.get('mrr', '-')} integrity={row.get('citation_integrity', '-')} "
              f"tok/s={row.get('tokens_per_s', '-')}", flush=True)
    # leave the corpus at the baseline chunking so later runs start from a known state
    ensure_chunking(base)
    crashed = [r["run_name"] for r in rows if str(r.get("notes", "")).startswith("CRASHED")]
    print(f"sweep: {len(rows)} configs in {(time.monotonic() - t_all) / 60:.1f} min, crashed: {crashed or 'none'}")
    return rows


def results_table(cfg: Config, write: bool = True) -> str:
    """Markdown table of the latest phase-7 row per run_name from runs.csv."""
    latest: dict[tuple[str, str], dict] = {}
    for r in evaluate.read_runs(cfg):
        if r.get("phase") == "7":
            latest[(r["run_name"], r.get("llm_model", ""))] = r      # latest row per (config, model)
    if not latest:
        return "(no phase 7 rows in runs.csv)"
    cols = [c for c in TABLE_COLUMNS if any(r.get(c) for r in latest.values())]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    order = [f"ablate_{n}" for n, _, _ in AXES]
    for key in sorted(latest, key=lambda k: (k[1], order.index(k[0]) if k[0] in order else 99, k[0])):
        r = latest[key]
        lines.append("| " + " | ".join(str(r.get(c, ""))[:60] for c in cols) + " |")
    table = "\n".join(lines)
    if write:
        (cfg.results_dir / "ablation_table.md").write_text(table + "\n", encoding="utf-8")
    return table


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="One-dimension-at-a-time ablation sweep")
    ap.add_argument("--fast", action="store_true", help="retrieval metrics only, no generation")
    ap.add_argument("--limit", type=int, help="first N eval questions per config")
    ap.add_argument("--only", help="comma-separated run names to include")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--table", action="store_true", help="print the results table from runs.csv and exit")
    ap.add_argument("--model", help="baseline LLM for this sweep (default: generation.model); the model axis "
                                    "then compares against the other configured model")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    base = load_config()
    if args.model:
        base = base.with_overrides(llm_model=args.model)
    if args.table:
        print(results_table(base))
        return 0
    only = set(args.only.split(",")) if args.only else None
    if args.dry_run:
        for name, axis, cfg in sweep(base, only):
            print(f"{name:12} {axis:15} chunk={cfg.chunk_size_tokens} mode={cfg.retrieval_mode} rerank={cfg.rerank} "
                  f"ctx={cfg.context_chunks} model={cfg.llm_model}")
        return 0
    print(f"offline={base.offline} fast={args.fast} limit={args.limit or 'all'} baseline model={base.llm_model}")
    rows = run_sweep(base, fast=args.fast, limit=args.limit, only=only)
    print(results_table(base))
    crashed = sum(1 for r in rows if str(r.get("notes", "")).startswith("CRASHED"))
    return 1 if crashed else 0


if __name__ == "__main__":
    sys.exit(main())
