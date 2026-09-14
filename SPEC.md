# Local RAG Research Companion — Build Specification

**Version:** 0.1 (draft)
**Target platform:** Windows 10/11
**Baseline hardware:** CPU only, 8 GB RAM, no GPU
**Build mode:** Single-shot build by a coding agent, verified at phase gates. No throwaway prototypes.

---

## 0. Parameters to fill before build starts

All build-time parameters are resolved. The build can start immediately.

| Parameter | Value |
|---|---|
| Python version | 3.11 or 3.12 (pin one at Phase 0) |
| Citation metadata for user-supplied PDFs | Title-lookup against Crossref/OpenAlex, filename fallback — see §6.3 |
| Corpus topic | **Runtime input, not a build blocker.** Supplied as a search query when the fetcher is first run. Recommended: RAG / LLM retrieval papers — the user can grade eval answers unaided and arXiv coverage is complete. |
| Corpus size | Whatever the fetcher pulls; top up from arXiv if the eval set needs more source material. |

---

## 1. Purpose and contribution

A local-first research companion that **discovers, downloads, ingests, retrieves and answers questions over scientific papers**, running entirely on low-end consumer hardware.

**Stated contribution (for the report):** cost and infrastructure engineering, not a novel retrieval algorithm. The deliverable result is a measured ablation showing which retrieval and generation choices actually matter when the model is small and the machine is constrained.

**Explicit non-goals:**
- No agentic loop, no multi-agent orchestration, no autonomous planning
- No model training or fine-tuning
- No GPU requirement
- No paywall circumvention; inaccessible full text is reported as unavailable
- No novel algorithm claim

---

## 2. Hard constraints

**C1 — Privacy boundary (load-bearing).** No module at or below the ingestion layer may hold a network client. Paper text, chunks, embeddings, query context and generated answers never leave the machine. Enforced by test, not convention (§8, Gate 5).

**C2 — Network use is confined to the discovery/fetch layer** and is always user-initiated as a discrete action, never triggered by a chat query. Accepted and documented leak: search terms sent to public scholarly APIs during discovery.

**C3 — Offline mode.** A single config flag disables every network client process-wide. With it set, discovery/fetch fail fast with a clear message and the full RAG pipeline works unchanged.

**C4 — Memory ceiling.** Peak RSS must stay under 6 GB on the baseline machine, leaving headroom for the OS. Measured and logged at every phase gate from Phase 2 onward.

**C5 — No paid services.** Every dependency free and open source; every API free-tier. Keyed APIs degrade gracefully when the key is absent (§5.2).

**C6 — Windows-native.** No WSL requirement, no Docker requirement, no POSIX-only dependencies. Paths via `pathlib` throughout.

---

## 3. Stack

| Layer | Choice | Rationale |
|---|---|---|
| Language | Python 3.11/3.12 | — |
| PDF → Markdown | `pymupdf4llm` | CPU-only, ~1 s/paper, emits structured markdown |
| Lexical index | SQLite FTS5 (stdlib) | No server, no extra process |
| Vector index | `sqlite-vec`, fallback NumPy flat index | Single file; fallback if extension loading is unavailable (§4.2) |
| Embeddings | `bge-small-en-v1.5` (384-dim) via `sentence-transformers` | ~130 MB, CPU-viable |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | ~90 MB; optional, ablated |
| LLM runtime | Ollama (Windows native) | OpenAI-compatible endpoint |
| Default model | Qwen3 4B, Q4_K_M | ~2.5–3 GB resident |
| Fallback model | Qwen3 1.7B, Q4_K_M | Demo responsiveness if 4B is too slow |
| UI | Streamlit | Single page |
| Tests | pytest | — |
| Config | YAML + `python-dotenv` | — |

**Generation quality is an open question, not an assumption.** The 4B default may prove inadequate; Phase 5 and Phase 7 measurements decide. Both models are spec'd so the comparison is part of the results rather than a late scramble.

---

## 4. Architecture

```
DISCOVERY (network)          │ PIPELINE (local only)
─────────────────────────────┼──────────────────────────────
search → resolve → download  │ parse → chunk → index
                             │   → retrieve → rerank → generate
        ↓                    │       ↑
   data/pdfs/  ══════════════╪═══════╝
        (handoff is a folder)
```

The fetcher is a corpus **producer**; the pipeline is a corpus **consumer**. They meet at `data/pdfs/`. The pipeline behaves identically whether a PDF was fetched or dropped in by hand. This keeps ablation runs reproducible against a frozen corpus.

### 4.1 Modules

```
config.py        Config load, validation, offline flag
models.py        Dataclasses: Paper, Chunk, Retrieved, Answer
sources/         Discovery adapters (network) — see §5
fetch.py         Identity resolution, PDF resolution, download
parse.py         PDF → markdown, cleanup, cache
chunk.py         Markdown → chunks
index.py         FTS5 + vector index build/query
retrieve.py      BM25 / dense / hybrid / rerank behind one interface
generate.py      Ollama client, context assembly, citation prompt
evaluate.py      Metrics harness → CSV
ablate.py        Config sweep runner
app.py           Streamlit UI
```

Flat layout. One module per concern. Do not create additional files without cause.

### 4.2 Vector index risk

Windows CPython may ship without `sqlite3` extension-loading support. Phase 2 must probe this at build time. If unavailable, fall back to a NumPy flat index persisted as `.npy` — brute-force cosine over <200k chunks is acceptable at this corpus scale and costs tens of milliseconds. The fallback is a documented supported path, not a failure.

### 4.3 Data layout

```
data/
  pdfs/{paper_id}.pdf       raw
  md/{paper_id}.md          parse cache
  meta/{paper_id}.json      metadata
  index.db                  SQLite: papers, chunks, FTS5, vectors
  cache/                    API responses, embeddings
results/
  runs.csv                  ablation output
  eval_set.jsonl            ground-truth QA pairs
```

`paper_id` derives from identity precedence (§5.3). The markdown cache is what makes the chunk-size sweep cheap: parse once, re-chunk freely.

---

## 5. Discovery and fetch layer

### 5.1 Sources — no key required

| Source | Role | Notes |
|---|---|---|
| OpenAlex | Primary discovery + metadata | ~250M works; send `mailto` for polite pool; stay under 100k calls/day |
| arXiv | Preprint discovery + **direct PDF** | Generous limits; ~3 s between calls |
| Crossref | DOI metadata, reference traversal | Contact email for polite pool |
| Unpaywall | OA PDF resolution by DOI | Email parameter required |
| Europe PMC | Biomedical discovery + OA full text | — |
| DataCite | arXiv-style DOI metadata | — |

### 5.2 Sources — free key, loaded from `.env`

| Source | Role | Limit |
|---|---|---|
| Semantic Scholar | Discovery, abstracts | 100 req / 5 min |
| CORE | OA **full text** aggregation | ~10k/day |
| PubMed (NCBI) | Biomedical metadata | Key optional |

**Graceful degradation is mandatory.** Missing key ⇒ source skipped with a log line, never an exception. The system must build, test and run to completion with an empty `.env`. Commit `.env.example`; gitignore `.env`.

### 5.3 Identity resolution

Precedence, frozen: **DOI → OpenAlex ID → arXiv ID → Semantic Scholar ID → normalized title + year.**

- arXiv papers are stored under synthesized DataCite-format DOIs (`10.48550/arxiv.*`) so an arXiv-sourced record and a DOI-sourced record of the same paper collapse to one entry.
- Strip URL prefixes from DOIs before comparison.
- Dedup is applied across all sources before any download.

### 5.4 PDF resolution and download

Order of attempts: arXiv direct → CORE full text → Europe PMC OA → Unpaywall OA link.

1. HEAD-probe the candidate URL; require a real `application/pdf` content type. Landing pages and 403s are recorded as unavailable, not retried indefinitely.
2. Per-paper timeout covering **both** download and parse.
3. Skip if the file already exists (idempotent re-runs).
4. Every outcome logged to `data/meta/` with status: `fetched` / `no_oa_pdf` / `http_error` / `timeout`.

Paywalled papers remain in the corpus **as metadata only** and are citable as references, clearly marked as full-text-unavailable.

### 5.5 API etiquette (required)

Descriptive user-agent with contact email on every request; sleep between calls per source; all responses cached to `data/cache/` so repeated test runs do not re-hit the network.

---

## 6. Ingestion

### 6.1 Parse

`pymupdf4llm` → markdown, behind a `Parser` interface with a config switch so an alternative (e.g. Docling) can be timed for the report without touching call sites. Parse only if the `.md` is missing or the PDF is newer.

### 6.2 Cleanup (deterministic, no LLM)

- Remove headers/footers/page numbers detected by recurrence across pages
- Excise the References section from the chunkable body; retain it separately for metadata and citation traversal
- Repair hyphenated line breaks; collapse whitespace
- Drop pages with negligible extractable text (figure-only pages)

### 6.3 Metadata

Fetched papers carry full metadata from their source.

For user-supplied PDFs, resolve metadata by lookup rather than extraction:

1. Pull the candidate title from the first page (largest text block above the abstract)
2. Query Crossref, then OpenAlex, by title
3. On a confident match, adopt the full canonical record — DOI, authors, year, venue — and assign `paper_id` by the normal precedence chain (§5.3)
4. On no match, fall back to filename-based citation with a `metadata: unresolved` flag

This reuses the discovery layer that already exists and yields real citations instead of guessed ones. The lookup is network-bound and therefore belongs to the discovery layer, not the pipeline (C1) — it runs at ingest time, never at query time.

**Scanned/image-only PDFs are out of scope.** Detect (near-zero extractable text) and exclude with a warning. No OCR.

### 6.4 Chunk

Split on markdown headers first, then subdivide oversized sections to the target token count with overlap. Every chunk carries `paper_id`, section title, and page range. The section title is prepended to chunk text — it improves both lexical and dense matching and yields citations of the form *Smith 2024, §3.2*.

---

## 7. Retrieval and generation

### 7.1 Retrieval

One interface, mode switch: `bm25` | `dense` | `hybrid` (+ optional rerank). Hybrid fusion via Reciprocal Rank Fusion, with the weight exposed as a swept parameter.

### 7.2 Generation

- Ollama client against the local endpoint; no other LLM client exists in the codebase
- Context assembled under an explicit token budget; chunks dropped from the tail when the budget is exceeded
- Citation-forced prompt: every claim carries a bracketed source ID
- Structured return: answer text + list of cited chunk IDs

### 7.3 Citation integrity

Post-generation check: every cited ID must exist in the retrieved set. Violations are counted and reported as a metric, never silently dropped. This is the primary hallucination guard and a headline number for the report.

---

## 8. Build phases and gates

The agent completes phases in order. Each gate must be executed and its **output shown** before advancing. A failing gate means fix and re-run, not proceed. Every phase adds pytest tests; the whole suite runs at every subsequent gate.

**Phase 0 — Skeleton**
Config, dataclasses, SQLite schema, module stubs, `.env.example`.
*Gate:* imports clean; config validates; schema round-trips; suite green.

**Phase 1 — Discovery and fetch**
Source adapters, identity resolution, PDF resolution, download.
*Gate:* live query to OpenAlex and arXiv returns parsed records; dedup across two sources merges a known duplicate into one record; ≥1 PDF downloads successfully; a known paywalled DOI is correctly marked `no_oa_pdf`; keyed sources skip cleanly with an empty `.env`; re-run downloads nothing new.

**Phase 2 — Parse and chunk**
Markdown conversion, cleanup, chunking.
*Gate:* every PDF has a `.md`; re-run parses zero files; references excised and header/footer noise absent on three spot-checked files; chunk count and token distribution printed; no empty or over-cap chunks; every chunk carries paper_id + section + page.

**Phase 3 — Index**
FTS5 + vector index, embedding cache, extension probe with fallback.
*Gate:* both indexes queryable; index size, build time and peak RSS logged; a known-answer keyword query returns the correct document at rank 1; vector backend in use is reported explicitly.

**Phase 4 — Retrieval**
All modes behind one interface.
*Gate:* each mode returns k results; on a 5-query smoke set the expected document appears in the top 10 for every mode; per-mode latency logged.

**Phase 5 — Eval harness (built before generation)**
Loads `eval_set.jsonl`, runs any config, emits metrics to CSV.
*Gate:* produces a results row for all three retrieval modes.
*Also:* an automated check asserts that no module under parse/chunk/index/retrieve/generate imports a network library. Fails the build if violated (C1).

**Phase 6 — Generation**
Ollama client, context assembly, citation prompt, integrity check.
*Gate:* end-to-end answer produced; every answer carries ≥1 citation; zero citations reference IDs outside the retrieved set; tokens/sec and peak RSS logged for both 4B and 1.7B.

**Phase 7 — Ablation runner**
Config sweep → single CSV.
*Gate:* full sweep completes with no crashed config; results table populated; offline-mode run (C3) completes successfully with the network disabled.

**Phase 8 — UI and packaging**
Streamlit app; README with Windows setup; `requirements.txt` pinned; LICENSE (MIT or Apache-2.0).
*Gate:* clean-clone install on Windows per README produces a working query.

---

## 9. Experiment design

Vary **one dimension at a time** from a fixed baseline. Roughly 12 runs total.

| Axis | Values |
|---|---|
| Chunk size | 256 / 512 / 1024 tokens |
| Retrieval mode | bm25 / dense / hybrid |
| Reranker | on / off |
| Context budget | 3 / 5 / 8 chunks |
| Model | Qwen3 1.7B / 4B (Q4_K_M) |
| Quantization | Q4_K_M / Q8 (4B only, RAM permitting) |

**Metrics, reported together:**
- *Retrieval:* recall@5, recall@10, MRR
- *Generation:* answer correctness (manual grade on the eval set), citation integrity rate
- *Systems:* p50/p95 latency, tokens/sec, peak RSS, index size, index build time

The systems column is what makes this an engineering contribution rather than a benchmark exercise. A negative result — reranking not worth its latency, or hybrid not beating BM25 on this corpus — is a valid and reportable finding.

### 9.1 Ground truth

25–30 question/answer pairs in `eval_set.jsonl`, each with the question, expected answer, and the paper ID(s) that contain it.

**The user writes or verifies these personally.** The agent may draft candidates from chunks, but auto-generated ground truth is trivially retrievable and will inflate every number in the report. This is the one task that is not delegated.

---

## 10. Deliverables

1. Public repository — MIT or Apache-2.0, README with Windows setup, `.env.example`, pinned requirements
2. Written report — architecture, ablation table, honest limitations
3. Live demo — Streamlit, must respond acceptably on the baseline machine (fallback model permitted)
4. `results/runs.csv` — the full sweep

---

## 11. Rules for the coding agent

1. Complete phases in order. Show gate output before advancing. Never advance past a red gate.
2. Test against **real data** from Phase 1 onward. Mocked retrieval tests prove nothing about retrieval.
3. Log metrics to CSV from Phase 3 onward so regressions are visible.
4. Report anything unverified as unverified. Do not assume a component works because it imports.
5. Do not add files, dependencies or abstraction layers beyond this spec without flagging the reason first.
6. Never place a network client below the ingestion layer (C1).
7. Cache every API response; respect per-source rate limits; never hammer an endpoint from a test loop.
8. Python only. Keep the module count flat and low.

---

## 12. Known risks

| Risk | Mitigation |
|---|---|
| 4B generation quality inadequate | Measured in Phase 6; 1.7B and larger-quant comparisons already in the sweep; report as a finding |
| `sqlite-vec` unloadable on Windows | NumPy flat-index fallback probed in Phase 3 |
| 8 GB ceiling breached with reranker + 4B loaded | Sequential load/unload; peak RSS gated at every phase |
| OA PDF availability lower than expected | HEAD-probe pass reports the ceiling before bulk download; metadata-only records remain citable |
| Eval set too small for significance | Report as a limitation; do not overclaim from ~30 questions |
| Discovery API schema drift | Cached responses + per-source adapter isolation |
