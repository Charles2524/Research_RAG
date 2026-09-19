# Local RAG Research Companion

A local-first research companion that discovers and downloads scientific papers over the
network, then parses, indexes, retrieves and answers questions over them **entirely offline**
on an ordinary Windows machine (CPU, or a small GPU if one is present). The deliverable is a
measured ablation of which retrieval and generation choices matter when the model is small and the
machine is constrained (see `SPEC.md`; results in `results/runs.csv` and `results/ablation_table.md`),
plus a demo web UI ("Corpus"), per-topic corpora, a one-script installer and a presentation deck
(`report/`). A full file map and the machine-specific facts are in `context.md`.

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
| GPU | optional | Ollama uses an NVIDIA GPU if it finds one (qwen3:1.7b needs ~2 GB VRAM); everything else runs on CPU |
| Disk | ~5 GB | 1 GB Python packages, 1.9 GB `qwen3:1.7b` (+2.5 GB optional `qwen3:4b`), 220 MB embedding models, plus the corpus |

## Quick setup (one script)

Install [Python 3.12](https://www.python.org/downloads/) (tick "Add python.exe to PATH") and
[Ollama for Windows](https://ollama.com/download), then in a Command Prompt:

```
git clone https://github.com/Charles2524/Research_RAG.git
cd Research_RAG
setup.bat
```

`setup.bat` creates `.venv`, installs the Python packages (~1 GB, PyTorch), checks the VC++ runtime,
asks for the contact email the scholarly APIs require, starts Ollama, pulls `qwen3:1.7b` (1.9 GB;
optionally `qwen3:4b`, 2.5 GB), downloads the embedding and reranking models (~220 MB), runs the phase-0
checks and puts a **Corpus** shortcut on the desktop. Mostly download time: 10-25 minutes. It is safe
to run again; finished steps are skipped. If Python, Ollama or the runtime is missing it opens the
download page and waits.

Afterwards start the app with `run.bat` (or the desktop shortcut): it starts Ollama if needed, checks
that the model actually landed on the GPU (`python generate.py --gpu-check`; Ollama's start-up GPU probe
can time out after a reboot and silently fall back to CPU, so the launcher restarts it once if so), serves
the UI on http://127.0.0.1:8765 and opens your browser. The library starts empty; click **Add Papers**
to search, download and index open-access papers on any topic, or create a separate corpus per topic
from the breadcrumb menu.

## Manual setup (Command Prompt)

```
git clone https://github.com/Charles2524/Research_RAG.git RAG_Research
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
ollama pull qwen3:1.7b
ollama pull qwen3:4b
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

The demo interface is **Corpus**, a web UI served locally with no build step:

```
python server.py --open        # or run.bat; --open launches the browser once the port answers
```

Ask & Synthesize streams the answer with inline `[n]` citation badges next to a Retrieved Evidence
column: click a badge to jump to the exact passage. Every answer carries its metrics strip (citation
integrity, retrieval and generation time, tokens per second) and an execution trace. Library lists
every paper with a Paper Inspector; Runs & Traces keeps this session's questions and the pipeline log.
"Add Papers" is the one network action and says so. Fonts are self-hosted (`ui/fonts/`, OFL); no CDN
is contacted. The server binds to 127.0.0.1 only and needs no extra dependencies (starlette and uvicorn
ship with streamlit). Design decisions are recorded in `ui/DESIGN.md`.

Settings (model, retrieval mode, top-k, rerank, chunks sent to the model, light/dark theme) live behind
the gear icon and apply to the next question; defaults come from `config.yaml`. The 1.7B model is the
default for speed; `qwen3:4b` reasons before answering and is several times slower.

### Corpora: one folder per research topic

Every corpus is a folder with its own PDFs, markdown, metadata, cache and vector index: `data/` by
default, `data_<name>/` for topics created from the UI. Click the corpus name in the header breadcrumb
(or "switch" on the Corpus Index card) to list, switch or create one; creating a corpus opens Add Papers
with its search query pre-filled. Switching is instant, nothing is deleted or re-downloaded, and the
choice is remembered across restarts (`.active_corpus`). Corpus folders are gitignored.

The single-page Streamlit app from the SPEC is still available:

```
streamlit run app.py
```

Its **Ask** tab is fully local; its **Discover** tab is the only place that touches the network.

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
- With an NVIDIA GPU, Ollama's start-up GPU probe sometimes times out after a reboot and it silently
  serves the model from the CPU (`ollama ps` shows `100% CPU`; measured 9 tok/s vs 42 tok/s on a
  GTX 1050). `run.bat` detects this and restarts Ollama once; by hand, stop every Ollama process and
  run `ollama serve` again. A browser with hardware acceleration can hold enough VRAM to force a
  partial offload; close it before loading the model if you want the full speed.
- Two Ollama installs (the tray app and a manual `ollama serve`) cannot both serve on port 11434;
  the "bind: only one usage of each socket address" error just means one is already running.

## Presentation

`report/RAG_Research_Presentation.pptx` and `.pdf` (13 slides: literature survey, gap, method, results,
pending work, timeline) are generated from the project data by `python report/make_deck.py`.

## Layout

```
config.py models.py sources/ fetch.py parse.py chunk.py index.py retrieve.py generate.py   # pipeline
evaluate.py ablate.py                                                                       # metrics, sweep
server.py ui/{index.html,app.css,app.js,fonts/,DESIGN.md}   app.py .streamlit/              # Corpus UI, Streamlit UI
setup.bat run.bat                                                                            # installer, launcher
tests/ config.yaml .env.example requirements.txt SPEC.md CLAUDE.md context.md report/
data/{pdfs,md,meta,cache,index.db,corpus.json}  data_<name>/ (same layout)  .active_corpus  # gitignored
results/{runs.csv,eval_set.jsonl,ablation_table.md,answers_*.jsonl}
```

## License

MIT, see `LICENSE`.
