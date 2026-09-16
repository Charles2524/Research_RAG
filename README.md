# Local RAG Research Companion

A local-first research companion that discovers and downloads scientific papers over the
network, then parses, indexes, retrieves and answers questions over them **entirely offline**
on a CPU-only Windows machine. The deliverable is a measured ablation of which retrieval and
generation choices matter when the model is small and the machine is constrained
(see `SPEC.md`; results in `results/runs.csv`).

Privacy boundary: no module below the ingestion layer holds a network client (enforced by
`tests/test_privacy.py`). Paper text, chunks, embeddings and answers never leave the machine.
The only network use is discovery/fetch, always an explicit user action. With `OFFLINE=1`
every network client is disabled and the full RAG pipeline works unchanged.

## Requirements (Windows 10/11)

| Component | Version used | Notes |
|---|---|---|
| Python | 3.12 (python.org installer, "Add to PATH") | 3.11 also fine |
| Git | any recent | |
| [Ollama for Windows](https://ollama.com) | 0.15.x | serves the local LLM |
| Microsoft Visual C++ 2015-2022 x64 runtime | [vc_redist.x64.exe](https://aka.ms/vs/17/release/vc_redist.x64.exe) | required by PyMuPDF |
| RAM | 8 GB baseline | peak measured ~3.9 GB with the 4B model |
| GPU | none | everything runs on CPU |

## Setup (Command Prompt)

```
git clone <this repo> RAG_Research
cd RAG_Research
python -m venv .venv
.venv\Scripts\activate.bat
pip install -r requirements.txt
copy .env.example .env
```

Edit `.env` and set `CONTACT_EMAIL` (sent to the scholarly APIs as the polite-pool contact;
required for discovery). Optional: `HF_HOME` to keep the ~220 MB of model files on another
drive; `OLLAMA_MODELS` (a user environment variable) does the same for Ollama.

Pull the language models and start Ollama (leave it running in its own window):

```
ollama pull qwen3:4b
ollama pull qwen3:1.7b
ollama serve
```

Fetch the embedding and reranker model files into the local cache (network, once):

```
python -m fetch --models
```

## Build a corpus and ask a question

```
python -m fetch --query "retrieval-augmented generation" --limit 40
python parse.py
python chunk.py
python index.py
python generate.py "How does retrieval-augmented generation reduce hallucination?"
```

- `fetch` queries OpenAlex, arXiv, Europe PMC and PubMed (Semantic Scholar and CORE too if their free
  keys are in `.env`), dedups across sources, downloads open-access PDFs into `data/pdfs/`, and
  records every outcome in `data/meta/`. Paywalled papers stay as metadata-only records.
  Re-runs download nothing new. Every API response is cached under `data/cache/`.
- `parse.py` converts PDFs to markdown (PyMuPDF4LLM), strips headers/footers/page numbers, excises
  the References section, and caches to `data/md/`. Scanned PDFs are excluded with a warning (no OCR).
- `chunk.py` splits on headings, then into token windows (bge-small tokenizer) with overlap; every chunk
  carries paper id, section title and page range.
- `index.py` embeds chunks with `BAAI/bge-small-en-v1.5` into `data/index.db` (SQLite FTS5 + sqlite-vec,
  NumPy fallback if the extension cannot load) and logs size/time/RSS to `results/runs.csv`.
- `generate.py` retrieves (hybrid by default), assembles a token-budgeted context and asks the local
  model for a citation-forced answer. Every citation is checked against the retrieved set; violations
  are counted, never hidden.

Drop your own PDFs into `data/pdfs/` and run `python -m fetch --resolve-dropped` to look up their
metadata by title (Crossref, then OpenAlex) before parsing; unresolved files get a filename citation.

## UI

```
streamlit run app.py
```

The **Ask** tab is fully local. The **Discover** tab is the only place that touches the network.

## Evaluation and ablation

```
python evaluate.py --mode all              # retrieval metrics per mode -> results/runs.csv
python evaluate.py --mode hybrid --generate --model qwen3:1.7b   # + citation integrity, tokens/s
python -m ablate --fast                    # retrieval-only sweep (minutes)
python -m ablate                           # full sweep with generation (hours on CPU)
python -m ablate --table                   # results table from runs.csv
pytest                                     # full suite; live tests replay from the API cache
```

`results/eval_set.jsonl` holds the question set (question, expected answer, paper ids, source quote,
page). Entries drafted automatically are flagged `verified: false`; the CSV `notes` column marks
every metric row computed from unverified questions. Answer correctness is graded by hand in the
`grade` slot of `results/answers_*.jsonl`.

Metrics: recall@5/@10 and MRR at paper level and at chunk level (the retrieved chunk must contain
the question's source quote), citation integrity, p50/p95 latency, tokens/s, peak RSS, index size
and build time.

## Configuration

`config.yaml` holds every knob (chunk size, retrieval mode, RRF weight, context budget, models,
thinking behaviour). `.env` holds secrets and machine-specific paths. `python config.py` validates.

Notes that matter on this stack:
- Ollama's `qwen3:4b` tag is a thinking-only build: it ignores `think=false`, so it is listed under
  `generation.thinking_only_models` and its reasoning stays in Ollama's separate field. Budget
  ~80 s per answer on a 4-thread CPU; `qwen3:1.7b` answers in ~6 s.
- The arXiv search API rate-limits some IPs with HTTP 429; the adapter backs off and the test skips
  with an explicit "UNVERIFIED" reason. arXiv OAI-PMH and PDF downloads are unaffected.
- First model load from an HDD takes about a minute; later loads come from the page cache.

## Layout

```
config.py models.py sources/ fetch.py parse.py chunk.py index.py retrieve.py generate.py
evaluate.py ablate.py app.py tests/ config.yaml .env.example requirements.txt SPEC.md CLAUDE.md
data/{pdfs,md,meta,cache,index.db}   results/{runs.csv,eval_set.jsonl,answers_*.jsonl}
```

## License

MIT, see `LICENSE`.
