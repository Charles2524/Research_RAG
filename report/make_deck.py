"""Build the review presentation as .pptx (python-pptx) and .pdf (reportlab) from one content model.

    python report/make_deck.py      -> report/RAG_Research_Presentation.pptx + .pdf

Content is drawn from the project itself: the corpus metadata in data/meta, the parsed papers,
results/runs.csv and results/ablation_table.md. Nothing here is a claim the build did not measure;
provisional numbers are labelled as such (the eval set is agent-drafted and unverified).
"""

from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt
from reportlab.lib import colors
from reportlab.lib.pagesizes import landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

OUT = Path(__file__).resolve().parent
TITLE = "Local-First RAG Research Companion"
SUBTITLE = "Discover, index, retrieve and answer over scientific papers on an 8 GB CPU-only Windows machine"
AUTHOR = "Charles Solomon"
DATE = "September 2026"
REPO = "github.com/Charles2524/Research_RAG"

# palette: ink on paper, one oxblood accent (mirrors the demo UI, ui/DESIGN.md)
INK, INK2, PAPER, PAPER2, ACCENT, ACCENT_SOFT, OK, DARK = (
    "2B2622", "6B625A", "FFFFFF", "F4F1EC", "8A2B24", "F1DED9", "2E6B3E", "1F1B18")

# ----------------------------------------------------------------- literature (16 papers, 2025-2026)
# (first author, year, venue, kind, short title, domain, retrieval, generator, key result)
PAPERS = [
    ("Yan et al.", 2025, "SSRN Electronic Journal", "journal", "Corrective Retrieval Augmented Generation (CRAG)",
     "Open-domain QA", "Any retriever + T5-large evaluator; web search fallback", "LLaMA-2 / others",
     "Retrieval evaluator (T5-large) grades retrieved docs and triggers correct / ambiguous / incorrect actions"),
    ("Ke et al.", 2025, "npj Digital Medicine", "journal", "RAG for 10 LLMs in assessing medical fitness",
     "Perioperative medicine", "LlamaIndex auto-merging, top-k 30, 58 guidelines", "GPT-3.5/4/4o, Gemini, Llama 2/3, Claude",
     "3,234 responses vs 448 human answers; GPT-4 + RAG answered in ~20 s at human-level accuracy"),
    ("Vishwanath et al.", 2025, "BMJ Digital Health & AI", "journal", "MedMobile: a 3.8B clinical model",
     "Medical QA on device", "RAG tried as an add-on", "phi-3-mini 3.8B (fine-tuned)",
     "75.7% on MedQA; RAG lowered accuracy by 12.6 points on this small model (negative result)"),
    ("Shen et al.", 2025, "BMJ Digital Health & AI", "journal", "LLM-enhanced framework for systematic reviews",
     "Evidence synthesis", "RAG + agent architecture (framework)", "Various",
     "21 studies reviewed; RAG and agents proposed against hallucination and bias; consultant vs assistant roles"),
    ("Klesel & Wittmann", 2025, "Business & Information Systems Eng.", "journal", "Retrieval-Augmented Generation (catchword)",
     "Enterprise search", "Chunk-level retrieval; hierarchical methods discussed", "Generic LLM",
     "Names the 'blinkered chunk effect': an isolated chunk cannot be understood without wider document context"),
    ("Yang et al.", 2025, "npj Health Systems", "journal", "RAG for generative AI in health care (perspective)",
     "Health care", "External-knowledge retrieval (conceptual)", "Generic LLM",
     "Frames RAG's contribution as equity, reliability and personalisation; lists implementation limits"),
    ("Zhao P. et al.", 2026, "Data Science and Engineering", "journal", "RAG for AI-Generated Content: a survey",
     "Cross-modal survey", "Taxonomy of RAG foundations and enhancements", "Any generator",
     "Surveys RAG foundations, enhancements, applications and benchmarks across modalities"),
    ("Spens & Burgess", 2026, "Nature Communications", "journal", "Hippocampo-neocortical interaction as compressive RAG",
     "Computational neuroscience", "Hippocampal episodic retrieval into working memory", "Neocortical generative net",
     "Models memory as compressive RAG: episodes compressed, consolidated, retrieved as a basis for generation"),
    ("Godoy et al.", 2026, "BMC Med. Informatics & Decision Making", "journal", "RAG extraction from mammography reports",
     "Radiology NLP (Spanish)", "all-MiniLM-L6-v2 embeddings; annotated reports as demonstrations", "GPT-based + local open-weight",
     "NER F1 0.89-0.94 vs fine-tuned BETO 0.97 with no task training; RE up to 0.78 F1; cents per report"),
    ("Liu et al.", 2026, "Digital Health", "journal", "Dynamic alignment for heart-failure decision support",
     "Cardiology", "Guideline RAG as the final alignment stage", "LLaMA-3.1 (CPT, SFT, GRPO, RAG)",
     "Clinical score 0.716 -> 0.864 and risk safety 0.820 -> 0.948 with RAG; BLEU-4 fell (alignment tax)"),
    ("Almohaimeed et al.", 2026, "Frontiers in Big Data", "journal", "Benchmarking RAG LLMs for Arabic noise robustness",
     "Arabic QA", "Retrieved documents with injected noise; Gemini-2.5 relevance labels", "Several LLMs",
     "Noise-robustness benchmark; relevance of ~300-word documents labelled automatically, 500 checked by hand"),
    ("Kang et al.", 2026, "J. Medical Internet Research", "journal", "Health-education agent for gastric cancer",
     "Patient education", "Sparse retrieval of 50-100 candidates, medical reranker picks top 3", "OpenMEDLab 2.0",
     "Action-research deployment at a cancer centre; accuracy, usability and experience assessed with patients"),
    ("Kuster et al.", 2026, "Frontiers in Neurorobotics", "journal", "VLM-guided RAG for robot action prediction",
     "Robotics", "Vision-language retrieval over demonstrations", "SigLIP2 / CLIP",
     "SigLIP2 reached 68.47% top-1 on cropped images vs 46.88% for CLIP-Large"),
    ("Fasolino", 2026, "Qeios (preprint)", "preprint", "In RAG We Trust? Robustness under document poisoning",
     "Fact checking (FEVER)", "MiniLM + FAISS, k = 3", "Llama 3.1 8B Q4_K_M on CPU (llama.cpp)",
     "Accuracy 77.9% clean -> 43.5% fully poisoned; entity swap flips most answers (23.5% fooled rate)"),
    ("Defilippo et al.", 2026, "Research Square (preprint)", "preprint", "RAGtio: evaluating hybrid RAG pipelines",
     "Biomedical corpora", "Sparse, dense, hybrid (RRF 0.7/0.3, k = 60), + cross-encoder rerank", "Pipeline-agnostic",
     "Synthetic queries generated from indexed chunks inflate retrieval scores; human queries (Mode B) are stricter"),
    ("Priyad et al.", 2026, "Research Square (preprint)", "preprint", "Agentic RAG for verifiable regulatory compliance",
     "Urban cyber-physical systems", "Retrieval agent + deterministic critic agent", "LLM after verification",
     "nDCG@5 0.75 vs 0.58 for CRAG; supersession detection 0.88; critic P = R = F1 = 1.00 on 78 probes"),
]

# ----------------------------------------------------------------- measured results (results/runs.csv)
RETRIEVAL_ROWS = [   # config, chunk recall@5, chunk MRR, retrieval p50
    ("Hybrid (baseline, 512-token chunks)", "0.867", "0.662", "17 ms"),
    ("BM25 only", "0.933", "0.588", "13 ms"),
    ("Dense only", "0.733", "0.458", "8 ms"),
    ("Hybrid + cross-encoder rerank", "0.867", "0.729", "4.1 s"),
    ("256-token chunks", "0.800", "0.598", "21 ms"),
    ("1024-token chunks", "0.767", "0.653", "14 ms"),
]
GENERATION_ROWS = [   # metric, qwen3:1.7b (8 configs), qwen3:4b (5 configs)
    ("Citation integrity", "1.00 in every config", "1.00 in every config"),
    ("Answers with a citation", "100%", "83-93% (1-4 lost to the 1,536-token reasoning cap)"),
    ("Generation p50 per answer", "50-195 s", "62-99 s (thinks first: ~2.5k chars of reasoning)"),
    ("Tokens per second", "8-12 (post-reboot) / 37 (day 1)", "8-10"),
    ("Resident memory (Ollama)", "1.8 GB", "3.4 GB; machine peak ~4.1 GB (< 6 GB ceiling)"),
    ("Correctness on 5 sampled questions", "3 of 5 (two wrong numbers)", "4 of 5"),
]

SLIDES = [
    {"kind": "title"},
    {"kind": "bullets", "title": "Literature survey (1 of 2): 2025-2026 journal publications",
     "items": [f"{p[0]} ({p[1]}), {p[2]}. {p[4]}." for p in PAPERS[:8]],
     "note": "16 works in total: 13 journal articles and 3 preprints, all published 2025 or 2026, all part of the indexed corpus."},
    {"kind": "bullets", "title": "Literature survey (2 of 2): 2026 journals and preprints",
     "items": [f"{p[0]} ({p[1]}), {p[2]}. {p[4]}." for p in PAPERS[8:]],
     "note": "Corpus discovered with the system itself: OpenAlex, arXiv, Europe PMC and PubMed, de-duplicated across sources."},
    {"kind": "table", "title": "Comparison of the surveyed papers (1 of 2)",
     "header": ["Paper", "Domain", "Retrieval approach", "Generator", "Key result"],
     "rows": [[f"{p[0]} {p[1]}", p[5], p[6], p[7], p[8]] for p in PAPERS[:8]],
     "widths": [1.35, 1.35, 2.55, 1.75, 4.0]},
    {"kind": "table", "title": "Comparison of the surveyed papers (2 of 2)",
     "header": ["Paper", "Domain", "Retrieval approach", "Generator", "Key result"],
     "rows": [[f"{p[0]} {p[1]}", p[5], p[6], p[7], p[8]] for p in PAPERS[8:]],
     "widths": [1.35, 1.35, 2.55, 1.75, 4.0]},
    {"kind": "two_col", "title": "Research gap from the literature",
     "left_title": "What the literature measures",
     "left": ["Answer quality on benchmarks, mostly with API-hosted models (GPT-4, Gemini, Claude) or 8B+ open models on GPUs.",
              "Component novelty: evaluators (CRAG), critics (Priyad), structure-aware retrieval (AeroSAR), graph communities (CGS-RAG).",
              "Robustness to noise and poisoning (Fasolino; Almohaimeed) for one fixed pipeline."],
     "right_title": "What it leaves open",
     "right": ["Which retrieval and generation choices matter when the model is small and the machine is an 8 GB CPU laptop: MedMobile shows RAG can even hurt a 3.8B model.",
               "Systems cost alongside quality: latency, tokens/s, resident memory and index size are rarely reported together (RAGtio is the exception, retrieval only).",
               "A hard privacy boundary: no paper separates a network discovery layer from an offline pipeline and enforces it by test.",
               "Citation integrity as a first-class metric: hallucinated references are measured post hoc, not checked against the retrieved set on every answer.",
               "Evaluation on synthetic questions inflates results (RAGtio); ground truth needs human verification."]},
    {"kind": "two_col", "title": "Novelty points of this work",
     "left_title": "Engineering contribution, not a new algorithm",
     "left": ["Local-first by construction: discovery and fetch are the only networked modules; parse, chunk, index, retrieve and generate must not import an HTTP client, enforced by an AST test.",
              "One-flag offline mode disables every network client; the full pipeline was run under it as a gate.",
              "Citation-forced prompt plus a post-generation integrity check: every cited id must exist in the retrieved set; violations are counted, never dropped."],
     "right_title": "Measured, one axis at a time",
     "right": ["Ablation over chunk size, retrieval mode, reranking, context budget and model, each recorded with quality and systems metrics in a single CSV.",
               "Chunk-level recall (the retrieved chunk must contain the source passage) because paper-level recall saturates on a small corpus.",
               "Runs on 8 GB CPU-only Windows: sqlite-vec with a NumPy fallback, bge-small embeddings, Qwen3 1.7B/4B via Ollama, peak ~4.1 GB.",
               "A demo UI that makes verification a click: citation chip -> exact passage, verification line under every answer."]},
    {"kind": "pipeline", "title": "Proposed methodology: producer-consumer architecture"},
    {"kind": "results", "title": "Results: retrieval ablation (30 questions, chunk-level, provisional)"},
    {"kind": "table", "title": "Results: generation and systems metrics",
     "header": ["Metric", "Qwen3 1.7B (8 configs)", "Qwen3 4B (5 configs)"],
     "rows": [list(r) for r in GENERATION_ROWS], "widths": [2.6, 3.7, 4.7],
     "note": "Eval set: 30 agent-drafted questions over 24 papers, flagged unverified; every metric row carries that flag until human verification. "
             "Throughput fell 3-4x after a system reset (CPU, memory, thermal, security and engine settings checked normal); reported as a finding."},
    {"kind": "two_col", "title": "Pending works",
     "left_title": "Evaluation",
     "left": ["Verify the 30-question eval set by hand against the quoted passages (SPEC 9.1) and rerun the sweep on verified ground truth.",
              "Grade answer correctness manually for the 13 generated configs (results/answers_*.jsonl).",
              "Run the 4B model over the full sweep once machine throughput is back to day-1 levels; add the Q8 quantisation axis (model not yet pulled)."],
     "right_title": "Corpus, report, demo",
     "right": ["Top up the corpus with retrieval-methods papers once arXiv search is reachable (currently rate-limited from this IP).",
               "Write the report: architecture, ablation table, limitations, throughput variance.",
               "Rehearse the Reading Room demo end to end; exercise the 'Add papers' flow live.",
               "Optional: user feedback from two or three researchers on the evidence panel."]},
    {"kind": "timeline", "title": "Timeline to complete the remaining works",
     "rows": [("Week 1  (17-23 Sep)", "Verify eval set; grade answers; rerun retrieval sweep on verified questions"),
              ("Week 2  (24-30 Sep)", "Full 4B sweep on a fast day; Q8 axis; arXiv top-up if reachable"),
              ("Week 3  (1-7 Oct)", "Report draft: architecture, ablation table, limitations; figures from runs.csv"),
              ("Week 4  (8-14 Oct)", "Demo rehearsal, README pass, final commit and tag")]},
    {"kind": "closing", "title": "Plan for publication",
     "lines": ["No publication is planned for this work.",
               "Deliverables: public repository (MIT), written report, live demo and results/runs.csv.",
               REPO]},
]


# ================================================================= PPTX
def rgb(h: str) -> RGBColor:
    return RGBColor.from_string(h)


def add_text(slide, x, y, w, h, text, size=14, bold=False, color=INK, font="Calibri", align=PP_ALIGN.LEFT,
             anchor=MSO_ANCHOR.TOP, italic=False):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    p = tf.paragraphs[0]
    p.alignment = align
    r = p.add_run()
    r.text = text
    r.font.size, r.font.bold, r.font.italic, r.font.name = Pt(size), bold, italic, font
    r.font.color.rgb = rgb(color)
    return tb


def add_bullets(slide, x, y, w, h, items, size=14, color=INK, gap=6):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(gap)
        r = p.add_run()
        r.text = "–  " + item
        r.font.size, r.font.name = Pt(size), "Calibri"
        r.font.color.rgb = rgb(color)
    return tb


def add_rect(slide, x, y, w, h, fill, line=None, shape=MSO_SHAPE.RECTANGLE):
    s = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    s.fill.solid()
    s.fill.fore_color.rgb = rgb(fill)
    if line:
        s.line.color.rgb = rgb(line)
        s.line.width = Pt(0.75)
    else:
        s.line.fill.background()
    s.shadow.inherit = False
    return s


def slide_frame(prs, title, dark=False):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    add_rect(s, 0, 0, 13.333, 7.5, DARK if dark else PAPER)
    if title:
        add_text(s, 0.6, 0.45, 12.1, 0.8, title, size=28, bold=True, color=PAPER if dark else INK, font="Cambria")
    add_text(s, 0.6, 7.05, 8, 0.3, f"{TITLE} · {DATE}", size=9, color="B8AFA6" if dark else INK2)
    return s


def build_pptx(path: Path) -> None:
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
    for n, sl in enumerate(SLIDES, start=1):
        kind = sl["kind"]
        if kind == "title":
            s = slide_frame(prs, "", dark=True)
            add_rect(s, 0.6, 2.05, 0.18, 0.18, ACCENT)
            add_text(s, 0.6, 2.35, 11.5, 1.2, TITLE, size=40, bold=True, color=PAPER, font="Cambria")
            add_text(s, 0.6, 3.55, 10.5, 1.0, SUBTITLE, size=18, color="D9D2CA")
            add_text(s, 0.6, 5.2, 10, 0.4, f"{AUTHOR} · {DATE}", size=14, color="D9D2CA")
            add_text(s, 0.6, 5.6, 10, 0.4, f"Code, data and results: {REPO}", size=12, color="B8AFA6")
        elif kind == "bullets":
            s = slide_frame(prs, sl["title"])
            add_bullets(s, 0.6, 1.45, 12.1, 5.0, sl["items"], size=15, gap=8)
            if sl.get("note"):
                add_text(s, 0.6, 6.55, 12.1, 0.4, sl["note"], size=11, color=INK2, italic=True)
        elif kind == "table":
            s = slide_frame(prs, sl["title"])
            rows, cols = len(sl["rows"]) + 1, len(sl["header"])
            avail_h = 5.1 if sl.get("note") else 5.5
            tbl = s.shapes.add_table(rows, cols, Inches(0.6), Inches(1.4), Inches(12.1), Inches(avail_h)).table
            for j, w in enumerate(sl["widths"]):
                tbl.columns[j].width = Inches(w * 12.1 / sum(sl["widths"]))
            fs = 10 if rows > 7 else 12
            for j, htxt in enumerate(sl["header"]):
                c = tbl.cell(0, j)
                c.fill.solid(); c.fill.fore_color.rgb = rgb(INK)
                c.text = htxt
                para = c.text_frame.paragraphs[0]
                para.runs[0].font.size, para.runs[0].font.bold, para.runs[0].font.name = Pt(fs), True, "Calibri"
                para.runs[0].font.color.rgb = rgb(PAPER)
            for i, row in enumerate(sl["rows"], start=1):
                for j, val in enumerate(row):
                    c = tbl.cell(i, j)
                    c.fill.solid(); c.fill.fore_color.rgb = rgb(PAPER if i % 2 else PAPER2)
                    c.text = str(val)
                    for para in c.text_frame.paragraphs:
                        for r in para.runs:
                            r.font.size, r.font.name = Pt(fs), "Calibri"
                            r.font.color.rgb = rgb(INK)
                            if j == 0:
                                r.font.bold = True
                    c.margin_left = c.margin_right = Inches(0.06)
                    c.margin_top = c.margin_bottom = Inches(0.03)
            if sl.get("note"):
                add_text(s, 0.6, 6.6, 12.1, 0.45, sl["note"], size=10, color=INK2, italic=True)
        elif kind == "two_col":
            s = slide_frame(prs, sl["title"])
            for k, (col_title, items) in enumerate(((sl["left_title"], sl["left"]), (sl["right_title"], sl["right"]))):
                x = 0.6 + k * 6.25
                add_rect(s, x, 1.45, 5.85, 5.4, PAPER2 if k else PAPER, line=None if k else "D9D2CA",
                         shape=MSO_SHAPE.ROUNDED_RECTANGLE).adjustments[0] = 0.03
                add_text(s, x + 0.3, 1.7, 5.3, 0.4, col_title, size=15, bold=True, color=ACCENT)
                add_bullets(s, x + 0.3, 2.2, 5.3, 4.5, items, size=12.5, gap=7)
        elif kind == "pipeline":
            s = slide_frame(prs, sl["title"])
            # discovery (network) box
            add_rect(s, 0.6, 1.5, 3.9, 4.3, PAPER2, line="D9D2CA", shape=MSO_SHAPE.ROUNDED_RECTANGLE).adjustments[0] = 0.03
            add_text(s, 0.85, 1.65, 3.5, 0.35, "Discovery layer (network)", size=14, bold=True, color=ACCENT)
            add_bullets(s, 0.85, 2.1, 3.5, 3.5, [
                "OpenAlex, arXiv (API + OAI-PMH), Crossref, Unpaywall, Europe PMC, DataCite; Semantic Scholar, CORE, PubMed with free keys",
                "Identity: DOI > OpenAlex > arXiv > S2 > title+year; union-find dedup across sources",
                "HEAD-probed open-access PDF download, per-paper timeout, status per record",
                "Every response cached; polite user-agent and per-source delays",
                "Explicit user action only; OFFLINE=1 disables it entirely"], size=11.5, gap=5)
            # handoff arrow
            arr = s.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(4.6), Inches(3.35), Inches(0.9), Inches(0.6))
            arr.fill.solid(); arr.fill.fore_color.rgb = rgb(ACCENT); arr.line.fill.background()
            add_text(s, 4.45, 4.0, 1.2, 0.5, "data/pdfs/\n(a folder)", size=9.5, color=INK2, align=PP_ALIGN.CENTER)
            # pipeline stages
            stages = [("Parse", "PyMuPDF4LLM to markdown; header/footer removal; references excised"),
                      ("Chunk", "Headings first, then 512-token windows with overlap; section title prepended"),
                      ("Index", "SQLite FTS5 + sqlite-vec (NumPy fallback); bge-small 384-d; embedding cache"),
                      ("Retrieve", "BM25 | dense | hybrid (weighted RRF) | optional MiniLM cross-encoder rerank"),
                      ("Generate", "Ollama Qwen3 on loopback; token-budgeted context; citation-forced prompt; integrity check")]
            for i, (name, desc) in enumerate(stages):
                y = 1.5 + i * 0.86
                add_rect(s, 5.7, y, 1.5, 0.7, INK, shape=MSO_SHAPE.ROUNDED_RECTANGLE).adjustments[0] = 0.15
                add_text(s, 5.7, y, 1.5, 0.7, name, size=13, bold=True, color=PAPER, align=PP_ALIGN.CENTER,
                         anchor=MSO_ANCHOR.MIDDLE)
                add_text(s, 7.35, y + 0.05, 5.35, 0.65, desc, size=11.5, color=INK, anchor=MSO_ANCHOR.MIDDLE)
            add_rect(s, 5.55, 5.95, 7.15, 0.75, ACCENT_SOFT, shape=MSO_SHAPE.ROUNDED_RECTANGLE).adjustments[0] = 0.1
            add_text(s, 5.75, 6.0, 6.8, 0.65,
                     "Privacy boundary (C1): no module from parse to generate may import an HTTP client; enforced by an AST test on every run. "
                     "Paper text, chunks, embeddings and answers never leave the machine.",
                     size=10.5, color=INK, anchor=MSO_ANCHOR.MIDDLE)
        elif kind == "results":
            s = slide_frame(prs, sl["title"])
            from pptx.chart.data import CategoryChartData
            from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION
            cd = CategoryChartData()
            cd.categories = [r[0].replace(" (baseline, 512-token chunks)", " (baseline)") for r in RETRIEVAL_ROWS]
            cd.add_series("Chunk recall@5", [float(r[1]) for r in RETRIEVAL_ROWS])
            cd.add_series("Chunk MRR", [float(r[2]) for r in RETRIEVAL_ROWS])
            gf = s.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, Inches(0.6), Inches(1.4), Inches(7.4), Inches(5.3), cd)
            ch = gf.chart
            ch.has_legend = True
            ch.legend.position = XL_LEGEND_POSITION.BOTTOM
            ch.legend.include_in_layout = False
            ch.legend.font.size = Pt(11)
            ch.value_axis.maximum_scale = 1.0
            ch.value_axis.minimum_scale = 0.0
            ch.value_axis.has_major_gridlines = True
            ch.value_axis.major_gridlines.format.line.color.rgb = rgb("E3DED7")
            ch.value_axis.tick_labels.font.size = Pt(10)
            ch.category_axis.tick_labels.font.size = Pt(10)
            ch.category_axis.reverse_order = True
            plot = ch.plots[0]
            plot.has_data_labels = True
            plot.data_labels.font.size = Pt(9)
            plot.data_labels.number_format = "0.00"
            plot.data_labels.number_format_is_linked = False
            plot.gap_width = 60
            for ser, col in zip(plot.series, (ACCENT, "B8AFA6")):
                ser.format.fill.solid(); ser.format.fill.fore_color.rgb = rgb(col)
            add_rect(s, 8.3, 1.4, 4.4, 5.3, PAPER2, shape=MSO_SHAPE.ROUNDED_RECTANGLE).adjustments[0] = 0.03
            add_text(s, 8.55, 1.6, 4.0, 0.4, "What the sweep says", size=15, bold=True, color=ACCENT)
            add_bullets(s, 8.55, 2.1, 3.95, 4.5, [
                "BM25 wins chunk recall@5 (0.93); dense alone trails (0.73); hybrid sits between and has the best MRR without reranking.",
                "Reranking lifts MRR to 0.73 but costs ~4 s per query on this CPU (vs 8-21 ms).",
                "256-token chunks halve generation time for a small recall loss; 1024-token chunks cost the most and gain nothing.",
                "Paper-level recall is 1.0 everywhere on 27 papers, which is why the chunk-level metric exists.",
                "Retrieval latency p50: 8-21 ms; index 12-18 MB; 1,133 chunks; first embedding pass 9 chunks/s warm."],
                size=11, gap=6)
        elif kind == "timeline":
            s = slide_frame(prs, sl["title"])
            for i, (when, what) in enumerate(sl["rows"]):
                y = 1.6 + i * 1.25
                add_rect(s, 0.6, y, 2.4, 0.9, ACCENT, shape=MSO_SHAPE.ROUNDED_RECTANGLE).adjustments[0] = 0.12
                add_text(s, 0.6, y, 2.4, 0.9, when, size=13, bold=True, color=PAPER, align=PP_ALIGN.CENTER,
                         anchor=MSO_ANCHOR.MIDDLE)
                add_rect(s, 3.15, y, 9.55, 0.9, PAPER2, shape=MSO_SHAPE.ROUNDED_RECTANGLE).adjustments[0] = 0.12
                add_text(s, 3.4, y, 9.1, 0.9, what, size=13, color=INK, anchor=MSO_ANCHOR.MIDDLE)
            add_text(s, 0.6, 6.65, 12, 0.35, "Effort is bounded by machine time: a full generation sweep with the 4B model takes 6-8 hours on this CPU.",
                     size=11, color=INK2, italic=True)
        elif kind == "closing":
            s = slide_frame(prs, "", dark=True)
            add_rect(s, 0.6, 2.05, 0.18, 0.18, ACCENT)
            add_text(s, 0.6, 2.35, 11.5, 0.9, sl["title"], size=34, bold=True, color=PAPER, font="Cambria")
            add_bullets(s, 0.6, 3.5, 11.5, 2.5, sl["lines"], size=16, color="D9D2CA", gap=10)
        add_text(s, 12.5, 7.05, 0.4, 0.3, str(n), size=9, color="B8AFA6" if kind in ("title", "closing") else INK2,
                 align=PP_ALIGN.RIGHT)
    prs.save(str(path))


# ================================================================= PDF (same content, rendered directly)
def build_pdf(path: Path) -> None:
    page = landscape((13.333 * inch, 7.5 * inch))
    doc = SimpleDocTemplate(str(path), pagesize=page, leftMargin=0.6 * inch, rightMargin=0.6 * inch,
                            topMargin=0.5 * inch, bottomMargin=0.45 * inch, title=TITLE, author=AUTHOR)
    ink, ink2, accent = colors.HexColor("#" + INK), colors.HexColor("#" + INK2), colors.HexColor("#" + ACCENT)
    H = ParagraphStyle("h", fontName="Times-Bold", fontSize=30, leading=36, textColor=ink, spaceAfter=18)
    B = ParagraphStyle("b", fontName="Helvetica", fontSize=16, leading=22, textColor=ink, leftIndent=16,
                       firstLineIndent=-16, spaceAfter=10)
    SUB = ParagraphStyle("sub", fontName="Helvetica-Bold", fontSize=17, leading=22, textColor=accent, spaceAfter=10)
    NOTE = ParagraphStyle("n", fontName="Helvetica-Oblique", fontSize=11.5, leading=15, textColor=ink2, spaceBefore=12)
    CELL = ParagraphStyle("c", fontName="Helvetica", fontSize=11, leading=14, textColor=ink)
    CELLB = ParagraphStyle("cb", parent=CELL, fontName="Helvetica-Bold")
    HEAD = ParagraphStyle("hd", parent=CELL, fontName="Helvetica-Bold", textColor=colors.white)
    TITLE_S = ParagraphStyle("t", fontName="Times-Bold", fontSize=44, leading=52, textColor=colors.white)
    TSUB = ParagraphStyle("ts", fontName="Helvetica", fontSize=19, leading=25, textColor=colors.HexColor("#D9D2CA"), spaceBefore=12)
    story = []

    def dark_page(canvas_, doc_):
        canvas_.saveState()
        canvas_.setFillColor(colors.HexColor("#" + DARK))
        canvas_.rect(0, 0, page[0], page[1], stroke=0, fill=1)
        canvas_.restoreState()

    def footer(canvas_, doc_):
        canvas_.saveState()
        canvas_.setFont("Helvetica", 8.5)
        canvas_.setFillColor(ink2)
        canvas_.drawString(0.6 * inch, 0.28 * inch, f"{TITLE} · {DATE}")
        canvas_.drawRightString(page[0] - 0.6 * inch, 0.28 * inch, str(doc_.page))
        canvas_.restoreState()

    def bullets(items, style=B):
        return [Paragraph("–  " + i, style) for i in items]

    dark_pages = set()
    for n, sl in enumerate(SLIDES, start=1):
        kind = sl["kind"]
        if kind in ("title", "closing"):
            dark_pages.add(n)
            story.append(Spacer(1, 1.6 * inch))
            story.append(Paragraph(TITLE if kind == "title" else sl["title"], TITLE_S))
            if kind == "title":
                story.append(Paragraph(SUBTITLE, TSUB))
                story.append(Paragraph(f"{AUTHOR} · {DATE}<br/>Code, data and results: {REPO}", TSUB))
            else:
                story.extend(bullets(sl["lines"], ParagraphStyle("bl", parent=B, textColor=colors.HexColor("#D9D2CA"), fontSize=15, leading=20)))
        elif kind == "bullets":
            story.append(Paragraph(sl["title"], H))
            story.extend(bullets(sl["items"]))
            if sl.get("note"):
                story.append(Paragraph(sl["note"], NOTE))
        elif kind == "table":
            story.append(Paragraph(sl["title"], H))
            data = [[Paragraph(h, HEAD) for h in sl["header"]]]
            for row in sl["rows"]:
                data.append([Paragraph(str(v), CELLB if j == 0 else CELL) for j, v in enumerate(row)])
            total = sum(sl["widths"])
            t = Table(data, colWidths=[w / total * 12.1 * inch for w in sl["widths"]], repeatRows=1)
            t.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), ink), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#" + PAPER2)]),
                ("LINEBELOW", (0, 0), (-1, -1), 0.25, colors.HexColor("#D9D2CA")),
                ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
            story.append(t)
            if sl.get("note"):
                story.append(Paragraph(sl["note"], NOTE))
        elif kind == "two_col":
            story.append(Paragraph(sl["title"], H))
            left = [Paragraph(sl["left_title"], SUB)] + bullets(sl["left"], ParagraphStyle("bs", parent=B, fontSize=14.5, leading=20))
            right = [Paragraph(sl["right_title"], SUB)] + bullets(sl["right"], ParagraphStyle("bs2", parent=B, fontSize=14.5, leading=20))
            t = Table([[left, right]], colWidths=[6.0 * inch, 6.0 * inch])
            t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 8),
                                   ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                                   ("BACKGROUND", (1, 0), (1, 0), colors.HexColor("#" + PAPER2))]))
            story.append(t)
        elif kind == "pipeline":
            story.append(Paragraph(sl["title"], H))
            left = [Paragraph("Discovery layer (network)", SUB)] + bullets([
                "OpenAlex, arXiv (API + OAI-PMH), Crossref, Unpaywall, Europe PMC, DataCite; Semantic Scholar, CORE, PubMed with free keys",
                "Identity: DOI > OpenAlex > arXiv > S2 > title+year; union-find dedup across sources",
                "HEAD-probed open-access PDF download, per-paper timeout, status per record",
                "Every response cached; polite user-agent and per-source delays",
                "Explicit user action only; OFFLINE=1 disables it entirely",
                "Hand-off is a folder: data/pdfs/"], ParagraphStyle("bp", parent=B, fontSize=13.5, leading=18))
            stages = [("Parse", "PyMuPDF4LLM to markdown; header/footer removal; references excised"),
                      ("Chunk", "Headings first, then 512-token windows with overlap; section title prepended"),
                      ("Index", "SQLite FTS5 + sqlite-vec (NumPy fallback); bge-small 384-d; embedding cache"),
                      ("Retrieve", "BM25 | dense | hybrid (weighted RRF) | optional MiniLM cross-encoder rerank"),
                      ("Generate", "Ollama Qwen3 on loopback; token-budgeted context; citation-forced prompt; integrity check")]
            right = [Paragraph("Local pipeline (offline)", SUB)] + [Paragraph(f"<b>{a}</b> — {b}", ParagraphStyle("st", parent=B, fontSize=13.5, leading=18, leftIndent=0, firstLineIndent=0)) for a, b in stages]
            right.append(Paragraph("Privacy boundary (C1): no module from parse to generate may import an HTTP client; enforced by an AST test on every run. "
                                   "Paper text, chunks, embeddings and answers never leave the machine.", NOTE))
            t = Table([[left, right]], colWidths=[5.2 * inch, 6.9 * inch])
            t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("BACKGROUND", (0, 0), (0, 0), colors.HexColor("#" + PAPER2)),
                                   ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8)]))
            story.append(t)
        elif kind == "results":
            story.append(Paragraph(sl["title"], H))
            data = [[Paragraph(h, HEAD) for h in ("Configuration", "Chunk recall@5", "Chunk MRR", "Retrieval p50")]]
            for r in RETRIEVAL_ROWS:
                data.append([Paragraph(r[0], CELLB)] + [Paragraph(v, CELL) for v in r[1:]])
            t = Table(data, colWidths=[3.6 * inch, 1.3 * inch, 1.1 * inch, 1.2 * inch], repeatRows=1)
            t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), ink),
                                   ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#" + PAPER2)]),
                                   ("LINEBELOW", (0, 0), (-1, -1), 0.25, colors.HexColor("#D9D2CA")),
                                   ("TOPPADDING", (0, 0), (-1, -1), 9), ("BOTTOMPADDING", (0, 0), (-1, -1), 9)]))
            right = [Paragraph("What the sweep says", SUB)] + bullets([
                "BM25 wins chunk recall@5 (0.93); dense alone trails (0.73); hybrid sits between and has the best MRR without reranking.",
                "Reranking lifts MRR to 0.73 but costs ~4 s per query on this CPU (vs 8-21 ms).",
                "256-token chunks halve generation time for a small recall loss; 1024-token chunks cost the most and gain nothing.",
                "Paper-level recall is 1.0 everywhere on 27 papers, which is why the chunk-level metric exists.",
                "Retrieval latency p50: 8-21 ms; index 12-18 MB; 1,133 chunks."], ParagraphStyle("br", parent=B, fontSize=13.5, leading=18))
            outer = Table([[t, right]], colWidths=[7.4 * inch, 4.7 * inch])
            outer.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("BACKGROUND", (1, 0), (1, 0), colors.HexColor("#" + PAPER2)),
                                       ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6)]))
            story.append(outer)
        elif kind == "timeline":
            story.append(Paragraph(sl["title"], H))
            data = [[Paragraph(w, ParagraphStyle("tw", parent=CELLB, fontSize=15, leading=19, textColor=colors.white)),
                     Paragraph(x, ParagraphStyle("tx", parent=CELL, fontSize=15, leading=19))] for w, x in sl["rows"]]
            t = Table(data, colWidths=[2.6 * inch, 9.5 * inch], rowHeights=[1.05 * inch] * len(data))
            t.setStyle(TableStyle([("BACKGROUND", (0, 0), (0, -1), accent), ("BACKGROUND", (1, 0), (1, -1), colors.HexColor("#" + PAPER2)),
                                   ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("LINEBELOW", (0, 0), (-1, -1), 4, colors.white),
                                   ("LEFTPADDING", (0, 0), (-1, -1), 10)]))
            story.append(t)
            story.append(Paragraph("Effort is bounded by machine time: a full generation sweep with the 4B model takes 6-8 hours on this CPU.", NOTE))
        if n < len(SLIDES):
            story.append(PageBreak())

    def on_page(canvas_, doc_):
        if doc_.page in dark_pages:
            dark_page(canvas_, doc_)
        else:
            footer(canvas_, doc_)

    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)


if __name__ == "__main__":
    pptx_path = OUT / "RAG_Research_Presentation.pptx"
    pdf_path = OUT / "RAG_Research_Presentation.pdf"
    build_pptx(pptx_path)
    build_pdf(pdf_path)
    print(f"wrote {pptx_path} ({pptx_path.stat().st_size // 1024} KB) and {pdf_path} ({pdf_path.stat().st_size // 1024} KB), "
          f"{len(SLIDES)} slides, {len(PAPERS)} papers")
