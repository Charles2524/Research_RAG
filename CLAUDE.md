# CLAUDE.md

Standing rules for this repo. `SPEC.md` defines what to build; this file defines how to work.

## Project

Local-first RAG research companion. Discovers and downloads papers over the network, then parses, indexes, retrieves and answers entirely offline. Read `SPEC.md` before starting any phase.

## Environment

- Windows, **Command Prompt** — not PowerShell, not bash. No `&&` chaining assumptions, no POSIX utilities.
- Python 3.12, **venv + pip**. Venv at `.venv`; activate with `.venv\Scripts\activate.bat`.
- All paths via `pathlib`. Never hardcode separators or absolute paths.
- Ollama runs natively; the local endpoint is the only LLM client in the codebase.

Phase 0 verifies Python, git and Ollama are present and reports versions. Do not assume — check.

## Hard rules

1. **No network client below the ingestion layer.** `parse.py`, `chunk.py`, `index.py`, `retrieve.py`, `generate.py` must not import requests, httpx, urllib or any HTTP library. This is enforced by a test.
2. **No new files, dependencies or abstraction layers** beyond `SPEC.md` without flagging the reason first and waiting for an answer.
3. **Python only.** Keep the module layout flat.
4. **No paid services.** Keyed APIs degrade gracefully when the key is missing — skip with a log line, never raise.
5. Secrets in `.env` (gitignored). `.env.example` stays committed and current.

## Phase discipline

Work through phases in `SPEC.md` in order.

- Run the gate check for a phase and **show its output** before moving on.
- A failing gate means fix and re-run. Never advance past a red gate.
- Every phase adds pytest tests. The full suite runs at every subsequent gate.
- Test against **real data** from Phase 1 onward. Mocked retrieval tests prove nothing about retrieval.

## Commits

Commit **only at a phase gate**, and only after:

1. The gate check has been run and its output shown
2. `pytest` passes with zero failures
3. Peak RSS has been logged and is under 6 GB

Message format: `phase N: <what was built>` with the gate result in the body. Do not commit mid-phase, do not commit to get to a checkpoint, do not push. If a gate cannot be made to pass, stop and report rather than committing partial work.

## Commands

```
.venv\Scripts\activate.bat
pip install -r requirements.txt
pytest                          # full suite
pytest tests\test_retrieve.py   # single module
python -m ablate                # full sweep -> results\runs.csv
streamlit run app.py
ollama list                     # confirm models present
```

## APIs

Every outbound request carries a descriptive user-agent with contact email. Sleep between calls per source. **Cache every response** to `data\cache\` — repeated test runs must not re-hit the network. arXiv needs ~3s between calls; OpenAlex and Crossref want the polite-pool email parameter.

## Reporting

- Report anything unverified as **unverified**. Do not assume a component works because it imports.
- Log metrics to `results\runs.csv` from Phase 3 onward, not to stdout alone.
- When something fails, show the actual error before proposing a fix.
- Prefer editing existing modules over adding new ones.
