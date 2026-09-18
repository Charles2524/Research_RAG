"""Streamlit single-page UI (SPEC 8, Phase 8). Run with ``streamlit run app.py``.

Two clearly separated actions:
- Ask: retrieve + generate, entirely local. Never touches the network (C2).
- Discover: an explicit, user-triggered fetch of new papers (network), then parse/chunk/index.
"""

from __future__ import annotations

import logging
import time

import streamlit as st

import chunk as chunk_mod
import fetch
import generate
import index
import parse
import retrieve
from config import RETRIEVAL_MODES, ConfigError, load_config
from sources import SourceUnavailable

log = logging.getLogger(__name__)


@st.cache_resource
def get_cfg():
    cfg = load_config()
    _warm_up(cfg)                 # once per process: embedder + fast model, so the first question is not the slow one
    return cfg


def _warm_up(cfg) -> None:
    import uuid
    try:
        conn = retrieve.get_conn(cfg)
        if index.count_chunks(conn):
            index.embed_texts(cfg, conn, [f"warm-up {uuid.uuid4().hex}"])
    except Exception as e:                       # never block the page on a warm-up
        log.warning("embedder warm-up skipped: %s", e)
    try:
        generate.warm_up(cfg, cfg.llm_fallback_model, keep_alive="2h")
    except generate.GenerationError as e:
        log.warning("model warm-up skipped: %s", e)


def corpus_status(cfg) -> dict:
    conn = index.connect(cfg.index_path)
    try:
        stats = index.index_stats(cfg, conn)
    finally:
        conn.close()
    papers = fetch.load_corpus(cfg)
    stats["n_meta"] = len(papers)
    stats["n_fulltext"] = sum(1 for p in papers if p.has_full_text)
    stats["n_pdfs"] = len(list(cfg.pdf_dir.glob("*.pdf"))) if cfg.pdf_dir.exists() else 0
    return stats


def render_ask(cfg):
    st.subheader("Ask the corpus")
    q = st.text_input("Question", placeholder="e.g. How does RAG decide which passages to feed the generator?")
    col1, col2, col3, col4 = st.columns(4)
    mode = col1.selectbox("Retrieval", RETRIEVAL_MODES, index=RETRIEVAL_MODES.index(cfg.retrieval_mode))
    rerank = col2.checkbox("Rerank (cross-encoder, ~4 s)", value=cfg.rerank)
    model = col3.selectbox("Model", [cfg.llm_fallback_model, cfg.llm_model], index=0,      # fast model first for demos
                           format_func=lambda m: f"{m}  (fast)" if m == cfg.llm_fallback_model else f"{m}  (reasons first, slow)")
    k = col4.slider("Chunks retrieved", 3, 20, cfg.k)
    if st.button("Answer", type="primary", disabled=not q.strip()):
        t0 = time.perf_counter()
        with st.spinner(f"Retrieving ({mode}{' + rerank' if rerank else ''})..."):
            hits = retrieve.retrieve(cfg, q, mode=mode, k=k, rerank_=rerank)
        t_ret = time.perf_counter() - t0
        papers = {p.paper_id: p for p in fetch.load_corpus(cfg)}
        try:
            with st.spinner(f"Generating with {model} (CPU; the 4B model reasons first and can take a minute)..."):
                ans = generate.generate(cfg, q, hits, model=model, papers=papers)
        except generate.GenerationError as e:
            st.error(str(e))
            ans = None
        if ans:
            st.markdown("### Answer")
            st.write(ans.text)
            ctx = generate.assemble_context(cfg, hits)
            st.markdown("**Sources**")
            for i, r in enumerate(ctx, start=1):
                used = "cited" if r.chunk.chunk_id in ans.cited_ids else "not cited"
                st.markdown(f"- **[S{i}]** {generate.source_label(r, papers)} — `{r.chunk.chunk_id}` ({used})")
            if ans.invalid_citations:
                st.warning(f"Citations outside the retrieved set (counted, not hidden): {ans.invalid_citations}")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Citation integrity", f"{ans.citation_integrity:.2f}")
            m2.metric("Retrieval", f"{t_ret:.2f} s")
            m3.metric("Generation", f"{ans.latency_s:.1f} s", f"{ans.tokens_per_s:.1f} tok/s")
            m4.metric("Tokens", f"{ans.prompt_tokens}+{ans.completion_tokens}")
        with st.expander(f"Retrieved chunks ({len(hits)})"):
            for r in hits:
                st.markdown(f"**{r.rank}.** score {r.score:.4f} — {generate.source_label(r, papers)}")
                st.text(r.chunk.text[:800] + ("..." if len(r.chunk.text) > 800 else ""))


def render_discover(cfg):
    st.subheader("Discover and download papers (network)")
    st.caption("This is the only action that uses the network. Search terms are sent to public scholarly APIs; "
               "paper text, chunks and answers never leave this machine.")
    if cfg.offline:
        st.info("Offline mode is on (OFFLINE=1 / config.yaml). Discovery is disabled; the pipeline works unchanged.")
        return
    if not cfg.contact_email:
        st.warning("CONTACT_EMAIL is not set in .env; scholarly APIs require a contact address.")
        return
    query = st.text_input("Search query", value=cfg.fetch_query)
    limit = st.number_input("Max papers", 1, 200, min(cfg.fetch_max_papers, 20))
    if st.button("Discover, download and index", disabled=not query.strip()):
        status = st.status("Running discovery...", expanded=True)
        try:
            papers = fetch.discover(cfg, query, int(limit))
            status.write(f"discovered {len(papers)} papers")
            papers = fetch.download(cfg, papers)
            counts = {}
            for p in papers:
                counts[p.status] = counts.get(p.status, 0) + 1
            status.write(f"download: {counts}")
            written = parse.parse_all(cfg)
            status.write(f"parsed {len(written)} new PDFs")
            chunks = chunk_mod.chunk_all(cfg)
            status.write(f"{len(chunks)} chunks")
            conn = index.connect(cfg.index_path)
            stats = index.build_index(cfg, conn)
            conn.close()
            retrieve.close_conn(cfg)
            status.write(f"index: {stats['n_vectors']} vectors, backend {stats['vector_backend']}, "
                         f"{stats['embed_s']} s embedding")
            status.update(label="Done", state="complete")
        except SourceUnavailable as e:
            status.update(label=f"Discovery unavailable: {e}", state="error")
        except Exception as e:                       # surface, never hide
            log.exception("discovery failed")
            status.update(label=f"Failed: {type(e).__name__}: {e}", state="error")


def main() -> None:
    st.set_page_config(page_title="Local RAG Research Companion", layout="wide")
    st.title("Local RAG Research Companion")
    try:
        cfg = get_cfg()
    except ConfigError as e:
        st.error(str(e))
        return
    with st.sidebar:
        st.markdown("### Corpus")
        s = corpus_status(cfg)
        st.metric("Papers with full text", f"{s['n_fulltext']} / {s['n_meta']}")
        st.metric("Chunks indexed", s["n_chunks"])
        st.caption(f"Vector backend: **{s['vector_backend']}** · index {s['index_size_mb']} MB · "
                   f"embedder {cfg.embedding_model.split('/')[-1]}")
        st.caption(f"Offline mode: **{'on' if cfg.offline else 'off'}** · Ollama at {cfg.ollama_host}")
        try:
            st.caption("Ollama models: " + ", ".join(generate.available_models(cfg)))
        except generate.GenerationError as e:
            st.error(f"Ollama not reachable: start it with `ollama serve`. ({e})")
        if st.button("Refresh"):
            st.rerun()
    tab_ask, tab_discover = st.tabs(["Ask (local)", "Discover (network)"])
    with tab_ask:
        if s["n_chunks"] == 0:
            st.info("No chunks indexed yet. Use the Discover tab, or drop PDFs into data/pdfs and run "
                    "`python parse.py`, `python chunk.py`, `python index.py`.")
        else:
            render_ask(cfg)
    with tab_discover:
        render_discover(cfg)


if __name__ == "__main__":
    main()
else:                                 # `streamlit run app.py` imports the module and executes top level
    main()
