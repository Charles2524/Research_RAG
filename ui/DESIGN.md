# Corpus — visual system for the web UI

The page in `ui/` recreates the "Corpus · Local-First Research Engine" design (an AI Studio React/Tailwind
mockup supplied as `corpus-research-engine.zip`) on the project's own stack: static HTML, CSS and vanilla JS
served by `server.py`. No Node, no build step, no CDN. Everything on screen comes from the local API;
nothing is hardcoded copy.

## Layout (from the reference)

- Fixed left rail, 256 px: brand, three views (Library · Ask & Synthesize · Runs & Traces), a Corpus Index
  card (papers, full text, chunks, index size, backend) and the "Local model · nothing leaves this machine" pill.
- Fixed top header, 56 px: Workspace / corpus-query breadcrumb, model badge, Add Papers, light/dark toggle,
  settings, avatar.
- Ask: scope row → centred query box with "Synthesize" → engine metadata line → 8/4 grid of the synthesis card
  (status eyebrow, serif headline, prose with `[n]` citation badges, metrics strip, copy actions) and the
  collapsible Execution Trace; sticky Retrieved Evidence column on the right.
- Library: filter chips, paper table (title & reference, venue & year, chunks, embeddings, actions), pager,
  sticky Paper Inspector with vector profile and BibTeX.
- Runs: this session's queries (click to reopen), then the pipeline log from `results/runs.csv`.
- Modals: Add Papers (the single network action, streaming steps) and Engine Settings (theme, model, chunks,
  retrieval mode, k, rerank, engine facts).

## Tokens (mirrored in ui/app.css)

Light scholarly: canvas `#f7f9fe`, cards `#ffffff`, containers `#f2f4f8 / #eceef2 / #e6e8ed`, ink `#191c1f`,
secondary ink `#43474e`, outline `#74777f / #c4c6cf`, primary navy `#022448` (hover `#1e3a5f`), cyan accent
`#0891b2` for icons, emerald `#306949` for verification, terra cotta `#431407 / #ffdbd1` for warnings.

Monastic dark: canvas `#0b0f17`, cards `#111722`, elevated `#17202e`, borders `#222f42`, primary cyan
`#0891b2 → #06b6d4`, accent ink `#67e8f9`, emerald `#34d399`. Chosen with the header toggle or the theme cards in
settings; remembered in `localStorage`, defaulting to the OS preference.

## Type

- Headlines and paper titles: **Source Serif 4** (600).
- UI and body: **IBM Plex Sans** in place of the reference's Inter (not in the repo; fonts are self-hosted).
- Citations, DOIs, metrics, trace: **IBM Plex Mono** in place of JetBrains Mono, tabular figures.
- Scale: 11 (labels/mono) · 12 · 13 · 14 · 15.5 (answer prose) · 17 · 24–26 (headlines).
- Icons: inline SVG symbols with 1.8 px strokes, drawn after the lucide set the reference uses.

## Data contract

`/api/status` fills the rail, header, scope row, settings and metadata line. `/api/papers` fills the Library.
`/api/ask` (SSE) drives the Ask view: `sources` → evidence cards, `thinking`/`token` deltas → live prose,
`done` → citation badges, metrics strip, trace, session run. `/api/discover` (SSE) drives Add Papers.
`/api/runs` fills the pipeline log.

Honesty rules kept from the first UI: the evidence "match" pill is the score relative to the top hit, labelled
as such (RRF and cross-encoder scores are not percentages); citation integrity, latency and token counts are
the real numbers from the answer; an answer with no citations is labelled as such rather than "grounded".

## Interaction

- Enter runs, Shift+Enter breaks a line. The button reads "Synthesizing…" while a run is live.
- Status eyebrow: Retrieving → Reading → Reasoning (thinking-only models) → Synthesizing → Grounded Synthesis.
- `[n]` badges are buttons: click highlights the badge and the matching evidence card and scrolls it into view;
  clicking a card highlights its badges. Cards expand to the full passage.
- Copy Markdown (answer + numbered sources + metrics) and Copy raw JSON (question, config, sources, answer).
- Runs view reopens any earlier answer from this session without re-querying the model.
- If Ollama is down the pill says so and the page re-polls every 10 s.
