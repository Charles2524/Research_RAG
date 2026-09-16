# Reading Room — visual system for the web UI

Recorded per the Impeccable new-work flow (world committed 2026-09-16). Mode: **Operate**.
Audience: researchers using the tool daily, on a laptop, close up. The tool disappears into the task:
ask, read, verify a claim against its passage, move on.

## World: the reading room

A warm paper ground with ink text. Not a chat app, not a SaaS dashboard: a desk with the corpus
on the left, the conversation in the middle, the evidence on the right. One accent, oxblood,
reserved for citations and the primary action. Emphasis comes from weight and size, never from
gradients, glows or coloured stripes.

## Tokens (mirrored in ui/app.css)

| Token | Value | Use |
|---|---|---|
| `--paper` | `oklch(97.5% 0.012 85)` | page ground |
| `--paper-2` | `oklch(94.8% 0.014 85)` | side panels, composer |
| `--paper-3` | `oklch(91.5% 0.016 85)` | hover, inset |
| `--ink` | `oklch(24% 0.02 60)` | text |
| `--ink-2` | `oklch(42% 0.02 60)` | secondary text (tinted, never gray) |
| `--ink-3` | `oklch(56% 0.018 60)` | meta, placeholders |
| `--rule` | `oklch(24% 0.02 60 / 0.14)` | hairlines |
| `--accent` | `oklch(44% 0.14 25)` | citations, primary action, focus |
| `--accent-soft` | `oklch(90% 0.045 30)` | citation chip fill |
| `--mark` | `oklch(93% 0.07 85)` | passage highlight (the marker pen) |
| `--ok` / `--warn` | `oklch(45% 0.12 150)` / `oklch(55% 0.15 60)` | integrity states |

Dark scheme: the same roles on a warm near-black lacquer (`oklch(15% 0.01 70)`), ink becomes
warm off-white; `prefers-color-scheme` only, no toggle. Both schemes keep body contrast ≥ 4.5:1.

## Type

- UI: **IBM Plex Sans** 14 px (13 px in dense lists), weights 400/500/600.
- Answer prose: **Source Serif 4** 17 px / 1.65, measure ≤ 68ch. Reading is the task.
- Identifiers (chunk ids, DOIs, numbers in the verification line): **IBM Plex Mono** 12 px, tabular figures.
- Scale: 12 · 13 · 14 · 15 · 17 · 20 · 24. Fixed rem, no fluid headings.
- Fonts self-hosted in `ui/fonts/` (OFL). The app is offline-first and never calls a font CDN.

## Layout

Three-column grid at ≥1180 px: library 272 px · conversation (max 72ch) · evidence 360 px.
Between 820 and 1180 px the evidence panel becomes a tab beside the conversation. Below 820 px one
column with a segmented Library / Ask / Evidence switch. Composer anchored at the bottom of the
conversation column; controls (mode, rerank, model, chunks) sit on one quiet row beneath the box.

## Interaction

- Enter sends, Shift+Enter breaks a line. The send control is disabled while a run is live.
- Status line during a run: "Retrieving…" → "Reading n passages…" → streamed tokens with a caret.
- Citation chips `[S3]` in the answer are buttons: click scrolls the passage into view and plays the
  single authored motion, a marker sweep across that passage (220 ms, exponential ease-out).
- Evidence entries: rank, id, label · §section · pages; cited ones carry the accent numeral and a
  "cited" tag; not-in-context ones are dimmed as "retrieved, not sent". Click expands the full text.
- Verification line under every answer: integrity · retrieval ms · generation s · tok/s · model.
- Reasoning (thinking-only models) is a collapsed disclosure with a live character count, muted.
- Discovery is a drawer opened by "Add papers"; its copy names the network use plainly.

## Refusals honoured (craft floor)

No kickers/eyebrows, no numbered section markers, no card grids, no gradient text, no glass, no
side-stripe borders above 1 px, no glyph/emoji icons (authored SVG at 1.5 px stroke), no page-load
choreography, no scattered fades. Skeleton rows for loading; empty state that teaches.
