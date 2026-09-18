# Project context

Everything a new session (human or Claude) needs to know about this repo beyond `SPEC.md` (what to build)
and `CLAUDE.md` (how to work). Updated 2026-09-18. GitHub: https://github.com/Charles2524/Research_RAG (branch `main`).

## What it is

Local-first RAG research companion for Windows. Discovers and downloads open-access papers over the network
(the only network step), then parses, chunks, embeds, retrieves and answers **offline** with a local Ollama
model, with every citation checked against the retrieved passages. Phases 0–8 of `SPEC.md` are complete
and committed; the ablation sweep and the presentation deck are done. Current work is the demo UI, per-topic
corpora and the one-script installer.

## File map

### Pipeline (flat Python modules, in data-flow order)

| File | Role |
|---|---|
| `config.py` / `config.yaml` / `.env` | Frozen `Config` dataclass; YAML defaults; secrets and machine paths in `.env` (gitignored, `.env.example` is the template). `HF_HOME`, `OFFLINE`, `CONTACT_EMAIL`, API keys. `Config.with_overrides` is how ablation and the corpus switcher vary settings. |
| `models.py` | `Paper`, `Chunk`, `Retrieved`, `Answer` dataclasses (JSON round-trip, citation integrity math). |
| `sources/` | One adapter per scholarly API (OpenAlex, arXiv Atom + OAI-PMH, Crossref, Unpaywall, Europe PMC, DataCite, Semantic Scholar/CORE/PubMed keyed). Every response cached under `data/cache/`. |
| `fetch.py` | Discovery, DOI-first de-duplication, OA PDF download, per-paper metadata JSON in `data/meta/`. `python -m fetch --models` downloads the embedder and reranker into the HF cache (the app never downloads models itself). `--resolve-dropped` for user-supplied PDFs, `--refresh-arxiv` for title fixes. |
| `parse.py` | PDF → markdown (pymupdf4llm), header/footer removal, references excised, scanned PDFs skipped. `data/md/`. |
| `chunk.py` | Heading split + token windows (bge tokenizer from the local HF cache), tiny sections merged. |
| `index.py` | SQLite `data/index.db`: FTS5 (BM25), vectors (sqlite-vec, NumPy fallback), embedding cache by text hash. Dedicated embed thread. `ensure_index` keeps it in sync. |
| `retrieve.py` | bm25 / dense / hybrid (RRF), optional cross-encoder rerank, thread-local connections, `--bench`. |
| `generate.py` | Ollama client over `http.client` to loopback only; `generate`, `generate_stream` (NDJSON), `warm_up`, `ensure_model`. Citation parsing accepts `[S1]`, `(S2)`; invalid citations counted, never hidden. |
| `evaluate.py` | `results/runs.csv` logger, eval set loader, recall@k / MRR (paper and chunk level), generation metrics, answers files. |
| `ablate.py` | One-axis-at-a-time sweep (`AXES`), `--fast`, `--limit`, `--only`, `--model`, `--table` → `results/ablation_table.md`. |

Hard rule enforced by `tests/test_privacy.py`: `parse`, `chunk`, `index`, `retrieve`, `generate` import no HTTP client.

### User interfaces

| File | Role |
|---|---|
| `server.py` | Starlette app + static server for the **Corpus** web UI. `python server.py [port] [--open]`. Endpoints: `/api/status`, `/api/papers`, `/api/chunk/{id}`, `/api/runs`, `/api/corpora`, `POST /api/corpus`, `POST /api/ask` (SSE: sources → thinking/token deltas → done), `POST /api/discover` (SSE, the one network action, 409 when offline). Warm-up thread loads the embedder and the fast model at start. |
| `ui/index.html`, `ui/app.css`, `ui/app.js` | The Corpus UI, vanilla HTML/CSS/JS, no build step. Views: Ask & Synthesize, Library, Runs & Traces; modals: Add Papers, Research Corpora, Engine Settings; light and dark themes. Design and data contract in `ui/DESIGN.md`. Fonts self-hosted in `ui/fonts/` (OFL). |
| `app.py` + `.streamlit/config.toml` | The single-page Streamlit app from the SPEC (Ask / Discover tabs). Still works; the demo uses the Corpus UI. |
| `setup.bat` | One-time installer for a fresh Windows PC: Python 3.12 check, venv + requirements, VC++ runtime check, `.env` email prompt, Ollama check/start, `ollama pull qwen3:1.7b` (asks about 4b), `python -m fetch --models`, phase-0 tests, desktop shortcut. Idempotent. |
| `run.bat` | Daily launcher: starts Ollama if needed, runs `server.py --open`, which opens the browser once the port answers. |

### Corpora

A corpus is a folder: `data/` (default, from `config.yaml`) or `data_<slug>/` created from the UI, each with
`pdfs/`, `md/`, `meta/`, `cache/`, `index.db` and `corpus.json` (name, query, created). The active one is
remembered in `.active_corpus`. All of these are gitignored, so a fresh clone starts with an empty library.
The HF model cache defaults to `data/cache/hf` unless `.env` sets `HF_HOME`.

### Results and report

`results/eval_set.jsonl` (30 questions, all `verified: false` until the user checks them),
`results/runs.csv` (every metrics row, gitignored), `results/ablation_table.md`, `results/answers_*.jsonl`
(hand-gradable), `report/make_deck.py` → `report/RAG_Research_Presentation.pptx` / `.pdf`.

### Tests

`tests/test_<module>.py` per module plus `test_phase0.py` (imports, config, schema, RSS, scripts exist) and
`test_privacy.py` (AST check of the network boundary). ~150 tests; live tests replay from `data/cache`.
Full suite: `.venv\Scripts\python -m pytest -q`.

## Runtime facts about this machine

- Real Python: `C:\Users\charl\AppData\Local\Programs\Python\Python312\python.exe` (not on PATH); venv at `.venv`.
- Ollama: `F:\ollama\ollama.exe`, models in `F:\ollama\models` (`OLLAMA_MODELS`), start with `ollama serve`
  (the tray app does not open the port). HF cache at `F:\hf_cache` via `.env`. pip cache `F:\pip_cache`.
- GPU: GTX 1050, 3 GB. **After a reboot Ollama's GPU discovery can time out and it silently runs on CPU**
  (log says `offloaded 0/29 layers`). Restart `ollama serve` and check for `library=CUDA` / `29/29 layers`.
  On GPU: qwen3:1.7b ≈ 42 tok/s generation, 214 tok/s prompt; on CPU ≈ 9 and 15.
- qwen3:4b is a thinking-only build (ignores `think=false`); handled via `thinking_only_models`. It only partly
  fits in 3 GB VRAM, so the UI defaults to qwen3:1.7b.
- arXiv's search API rate-limits this IP; OAI-PMH and the PDF host work.
- Ports: Corpus UI 8765, Streamlit 8501, Ollama 11434, all loopback.

## Conventions

- Commits only at green gates (or user-approved feature commits) with test output shown; push to `main`.
- Metrics go to `results/runs.csv`, never stdout only. Unverified things are reported as unverified.
- The eval set and answer grades are the user's to verify.
- The design reference for the UI was `corpus-research-engine.zip` (AI Studio React export, untracked);
  it was recreated on the static stack rather than adopted, so no Node toolchain exists in the repo.
