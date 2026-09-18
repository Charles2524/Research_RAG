/* Corpus — client. Vanilla JS, no build step. Talks only to the local server (server.py). */
(() => {
  "use strict";
  const $ = (s, el = document) => el.querySelector(s);
  const $$ = (s, el = document) => [...el.querySelectorAll(s)];
  const esc = (t) => String(t ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const fmtS = (s) => (s >= 10 ? `${Math.round(s)} s` : `${Number(s).toFixed(1)} s`);
  const fmtMs = (s) => (s < 1 ? `${Math.round(s * 1000)} ms` : fmtS(s));
  const num = (n) => Number(n).toLocaleString("en-US");
  const state = { status: null, papers: [], running: false, sources: [], mode: "hybrid", runs: [], current: null, page: 1, filter: "all", search: "", selected: null };
  const PAGE = 10;

  // ---------- theme ----------
  function setTheme(t) {
    document.documentElement.dataset.theme = t;
    try { localStorage.setItem("corpus-theme", t); } catch (_) {}
    for (const b of $$("[data-theme]")) if (b.tagName === "BUTTON") { const on = b.dataset.theme === t; b.setAttribute("aria-checked", String(on)); b.classList.toggle("is-on", on); }
  }
  let saved = "light";
  try { saved = localStorage.getItem("corpus-theme") || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light"); } catch (_) {}
  setTheme(saved);
  document.addEventListener("click", (e) => { const b = e.target.closest("button[data-theme]"); if (b) setTheme(b.dataset.theme); });

  // ---------- tabs ----------
  function setTab(t) {
    for (const v of $$(".view")) { v.hidden = v.id !== `view-${t}`; v.classList.toggle("is-on", v.id === `view-${t}`); }
    for (const b of $$(".nav-item")) b.classList.toggle("is-on", b.dataset.tab === t);
    if (t === "runs") loadRuns();
    window.scrollTo({ top: 0 });
  }
  document.addEventListener("click", (e) => { const b = e.target.closest("[data-tab]"); if (b) setTab(b.dataset.tab); });

  // ---------- modals ----------
  function openModal(id) { $(`#${id}`).hidden = false; const f = $(`#${id} input, #${id} select`); if (f) f.focus(); }
  function closeModals() { for (const m of $$(".modal")) m.hidden = true; }
  for (const id of ["open-add", "open-add-2"]) $(`#${id}`).addEventListener("click", () => openModal("modal-add"));
  for (const id of ["open-settings", "open-settings-2", "open-settings-3"]) $(`#${id}`).addEventListener("click", () => openModal("modal-settings"));
  document.addEventListener("click", (e) => { if (e.target.closest("[data-close]")) closeModals(); });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeModals(); });

  function toast(msg) {
    const t = document.createElement("div"); t.className = "toast"; t.textContent = msg; document.body.appendChild(t);
    setTimeout(() => t.remove(), 1800);
  }

  // ---------- status ----------
  async function loadStatus() {
    const r = await fetch("/api/status");
    const s = state.status = await r.json();
    const scope = s.fetch_query || "corpus";
    $("#crumb-scope span").textContent = s.corpus_name || scope;
    $("#idx-name").textContent = s.corpus_name || scope;
    $("#scope-name").textContent = scope;
    $("#scope-count").textContent = `(${num(s.n_papers)} papers indexed)`;
    $("#scope-embed").textContent = `Local embeddings: ${s.embedding_model.split("/").pop()}`;
    $("#nav-papers").textContent = num(s.n_papers);
    $("#idx-papers").textContent = num(s.n_papers);
    $("#idx-fulltext").textContent = `${s.n_fulltext} / ${s.n_papers}`;
    $("#idx-chunks").textContent = num(s.n_chunks);
    $("#idx-storage").textContent = `${Number(s.index_size_mb).toFixed(1)} MB`;
    $("#idx-backend").textContent = s.vector_backend;
    $("#lib-sub").textContent = `${num(s.n_papers)} indexed papers · ${s.n_fulltext} with full text · ${num(s.n_chunks)} chunks embedded with ${s.embedding_model.split("/").pop()} on this machine`;
    // models
    const sel = $("#model"); sel.innerHTML = "";
    const models = s.models.length ? s.models : [s.fallback_model, s.default_model];
    for (const m of models) { const o = document.createElement("option"); o.value = m; o.textContent = m + (m === s.fallback_model ? "  · fast" : m === s.default_model ? "  · reasons first, slow" : ""); sel.appendChild(o); }
    sel.value = models.includes(s.fallback_model) ? s.fallback_model : models[0];
    $("#k").value = s.k; $("#ctx").value = s.context_chunks; $("#rerank").checked = !!s.rerank;
    setMode(s.retrieval_mode);
    $("#discover-query").value = s.fetch_query || "";
    $("#drop-dir").textContent = `${s.corpus}/pdfs`;
    const pill = $("#status-pill");
    if (s.offline) { pill.classList.add("is-offline"); $("#status-pill-text").textContent = "Offline mode · network disabled"; }
    pill.classList.toggle("is-down", s.ollama !== "up"); $("#engine-dot").classList.toggle("dot-ok", s.ollama === "up");
    if (s.ollama !== "up") { $("#status-pill-text").textContent = "Ollama is not running · start it with `ollama serve`"; setTimeout(() => loadStatus().catch(() => {}), 10000); }   // it may still be starting
    else if (!s.offline) $("#status-pill-text").textContent = "Local model · nothing leaves this machine";
    $("#settings-kv").innerHTML = [
      ["Ollama", s.ollama === "up" ? "up · 127.0.0.1:11434 (loopback only)" : s.ollama],
      ["Embedder", s.embedding_model], ["Vector backend", s.vector_backend], ["Index", `${s.index_size_mb} MB · ${num(s.n_chunks)} chunks`],
      ["Network", s.offline ? "OFFLINE=1 · discovery disabled" : "only the Add Papers action"],
    ].map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join("");
    refreshMeta();
    $("#send").disabled = !$("#question").value.trim();
  }
  function refreshMeta() {
    $("#meta-engine").textContent = $("#model").value || "…";
    $("#hdr-model").lastElementChild.textContent = `${$("#model").value || "…"} · CPU`;
    $("#meta-k").textContent = $("#k").value;
    $("#meta-mode").textContent = { bm25: "BM25", dense: "Dense (bge)", hybrid: "Hybrid (BM25 + dense, RRF)" }[state.mode];
    $("#meta-rerank").textContent = $("#rerank").checked ? "MiniLM cross-encoder" : "off";
  }
  for (const id of ["model", "k", "ctx", "rerank"]) $(`#${id}`).addEventListener("change", refreshMeta);
  function setMode(m) {
    state.mode = m;
    for (const b of $$("#mode-seg button")) { const on = b.dataset.mode === m; b.classList.toggle("is-on", on); b.setAttribute("aria-checked", String(on)); }
    refreshMeta();
  }
  $("#mode-seg").addEventListener("click", (e) => { const b = e.target.closest("button"); if (b) setMode(b.dataset.mode); });

  // ---------- library ----------
  async function loadPapers() {
    const r = await fetch("/api/papers");
    state.papers = (await r.json()).papers.sort((a, b) => (b.has_full_text - a.has_full_text) || ((b.year || 0) - (a.year || 0)) || a.title.localeCompare(b.title));   // full text first
    renderPapers();
  }
  function filteredPapers() {
    const q = state.search.trim().toLowerCase();
    return state.papers.filter((p) => {
      if (q && !(p.title.toLowerCase().includes(q) || p.authors.join(" ").toLowerCase().includes(q) || (p.venue || "").toLowerCase().includes(q) || (p.doi || "").toLowerCase().includes(q) || String(p.year).includes(q))) return false;
      if (state.filter === "pdf") return p.has_full_text;
      if (state.filter === "meta") return !p.has_full_text;
      if (state.filter === "recent") return (p.year || 0) >= 2024;
      return true;
    });
  }
  function authorsShort(p) {
    const a = p.authors || [];
    if (!a.length) return "Unknown authors";
    const init = (n) => { const parts = n.trim().split(/\s+/); if (parts.length < 2) return n; const last = parts.length > 1 && /^[A-Z]{1,3}$/.test(parts[parts.length - 1]) ? parts[0] : parts[parts.length - 1]; const first = parts[0] === last ? parts[1] : parts[0]; return `${first[0]}. ${last}`; };
    return a.length > 2 ? `${init(a[0])}, ${init(a[1])} et al.` : a.map(init).join(", ");
  }
  function renderPapers() {
    const rows = filteredPapers();
    const pages = Math.max(1, Math.ceil(rows.length / PAGE));
    state.page = Math.min(state.page, pages);
    const start = (state.page - 1) * PAGE;
    const body = $("#paper-rows"); body.innerHTML = "";
    const chips = { all: state.papers.length, pdf: state.papers.filter((p) => p.has_full_text).length, meta: state.papers.filter((p) => !p.has_full_text).length, recent: state.papers.filter((p) => (p.year || 0) >= 2024).length };
    for (const b of $$("#lib-filters .chip")) { const base = b.textContent.replace(/\s*\(\d+\)$/, ""); b.textContent = `${base} (${chips[b.dataset.filter]})`; b.classList.toggle("is-on", b.dataset.filter === state.filter); }
    if (!rows.length) body.innerHTML = `<tr><td colspan="6" class="dim" style="cursor:default">No paper matches that. Try a word from the title, an author's surname or a DOI.</td></tr>`;
    rows.slice(start, start + PAGE).forEach((p, i) => {
      const tr = document.createElement("tr"); tr.dataset.id = p.paper_id;
      if (state.selected === p.paper_id) tr.classList.add("is-on");
      const emb = p.has_full_text ? (p.n_chunks ? `<span class="tag tag-ok">100% vectorized</span>` : `<span class="tag tag-warn">not chunked</span>`) : `<span class="tag tag-meta">metadata only</span>`;
      tr.innerHTML = `
        <td class="c p-idx">${String(start + i + 1).padStart(2, "0")}</td>
        <td><div class="p-title">${esc(p.title)}</div><div class="p-ref">${esc(authorsShort(p))} · ${p.doi ? `<span class="doi">doi:${esc(p.doi)}</span>` : `<span>${esc(p.paper_id)}</span>`}</div></td>
        <td><div class="p-venue">${esc(p.venue || (p.paper_id.includes("arxiv") ? "arXiv" : "—"))}</div><div class="p-year">${p.year ?? "—"}</div></td>
        <td class="c p-chunks">${p.n_chunks || "—"}</td>
        <td class="c">${emb}</td>
        <td class="r"><span class="row-actions"><button type="button" class="btn btn-icon btn-sm" data-ask title="Ask about this paper"><svg class="ic ic-accent"><use href="#i-sparkles"/></svg></button>${p.doi ? `<a class="btn btn-icon btn-sm" href="https://doi.org/${esc(p.doi)}" target="_blank" rel="noopener" title="Open on doi.org (external)"><svg class="ic"><use href="#i-external"/></svg></a>` : ""}</span></td>`;
      body.appendChild(tr);
    });
    $("#page-info").textContent = rows.length ? `Showing ${start + 1}–${Math.min(start + PAGE, rows.length)} of ${rows.length} papers` : "0 papers";
    const pager = $("#pager"); pager.innerHTML = "";
    const mk = (label, page, on = false, dis = false) => { const b = document.createElement("button"); b.type = "button"; b.textContent = label; b.disabled = dis; b.classList.toggle("is-on", on); b.addEventListener("click", () => { state.page = page; renderPapers(); }); pager.appendChild(b); };
    mk("Prev", state.page - 1, false, state.page <= 1);
    for (let p = 1; p <= pages; p++) mk(String(p), p, p === state.page);
    mk("Next", state.page + 1, false, state.page >= pages);
    if (!state.selected && rows.length) selectPaper(rows[0].paper_id, false);
  }
  function selectPaper(id, scroll = true) {
    state.selected = id;
    const p = state.papers.find((x) => x.paper_id === id); if (!p) return;
    for (const tr of $$("#paper-rows tr")) tr.classList.toggle("is-on", tr.dataset.id === id);
    const s = state.status || {};
    const share = s.n_chunks ? Math.min(100, Math.round(100 * p.n_chunks / Math.max(...state.papers.map((x) => x.n_chunks || 1)))) : 0;
    $("#insp-state").textContent = p.has_full_text ? "Indexed" : "Metadata";
    $("#insp-state").className = `badge mono ${p.has_full_text ? "badge-ok" : ""}`;
    $("#insp-body").innerHTML = `
      <div><span class="insp-label">Title</span><h3 class="insp-title">${esc(p.title)}</h3></div>
      <div class="insp-grid">
        <div class="insp-cell"><span>Authors</span><b title="${esc(p.authors.join(", "))}">${esc(p.authors.join(", ") || "Unknown")}</b></div>
        <div class="insp-cell"><span>Year / Venue</span><b>${p.year ?? "—"} · ${esc(p.venue || "—")}</b></div>
      </div>
      <div><span class="insp-label">Identifier / DOI</span><div class="insp-id"><span>${esc(p.doi || p.paper_id)}</span><button type="button" class="btn btn-icon btn-sm" data-copy="${esc(p.doi ? `https://doi.org/${p.doi}` : p.paper_id)}" title="Copy"><svg class="ic"><use href="#i-copy"/></svg></button></div></div>
      <div><span class="insp-label">Semantic Vector Profile</span><div class="insp-box">
        <div><span class="dim">Dense chunks:</span><b>${p.n_chunks} vectors</b></div>
        <div><span class="dim">Embedding model:</span><span class="hi">${esc((s.embedding_model || "").split("/").pop())}</span></div>
        <div><span class="dim">Vector dimension:</span><b>384-d float32</b></div>
        <div><span class="dim">Status:</span><b>${esc(p.status.replace(/_/g, " "))}${p.metadata_resolved ? "" : " · metadata unresolved"}</b></div>
        <div style="flex-direction:column;gap:4px;padding-top:2px"><div style="width:100%"><span class="dim">Chunk share vs. largest paper:</span><span>${share}%</span></div><div class="bar-wrap"><i style="width:${share}%"></i></div></div>
      </div></div>
      ${p.abstract ? `<div><span class="insp-label">Abstract</span><p class="insp-abs">${esc(p.abstract)}${p.abstract.length >= 600 ? "…" : ""}</p></div>` : ""}
      <div class="insp-actions">
        <button type="button" class="btn btn-primary" data-ask-paper><svg class="ic"><use href="#i-sparkles"/></svg>Synthesize Grounded Answer</button>
        <div class="two"><button type="button" class="btn btn-quiet" data-bibtex><svg class="ic"><use href="#i-copy"/></svg>Copy BibTeX</button>${p.doi ? `<a class="btn btn-quiet" href="https://doi.org/${esc(p.doi)}" target="_blank" rel="noopener"><svg class="ic"><use href="#i-external"/></svg>Open DOI</a>` : `<button type="button" class="btn btn-quiet" disabled>No DOI</button>`}</div>
      </div>`;
    if (scroll && window.innerWidth <= 1100) $("#inspector").scrollIntoView({ behavior: "smooth", block: "start" });
  }
  function bibtex(p) {
    const key = ((p.authors[0] || "anon").split(/\s+/).pop() || "anon").toLowerCase().replace(/[^a-z]/g, "") + (p.year || "");
    return `@article{${key},\n  title={${p.title}},\n  author={${p.authors.join(" and ")}},\n  journal={${p.venue || ""}},\n  year={${p.year || ""}},\n  doi={${p.doi || ""}}\n}`;
  }
  $("#paper-rows").addEventListener("click", (e) => {
    const tr = e.target.closest("tr[data-id]"); if (!tr) return;
    if (e.target.closest("a")) return;
    selectPaper(tr.dataset.id, false);
    if (e.target.closest("[data-ask]")) askAboutPaper(tr.dataset.id);
  });
  $("#inspector").addEventListener("click", async (e) => {
    const p = state.papers.find((x) => x.paper_id === state.selected); if (!p) return;
    if (e.target.closest("[data-ask-paper]")) askAboutPaper(p.paper_id);
    else if (e.target.closest("[data-bibtex]")) { await navigator.clipboard.writeText(bibtex(p)); toast("BibTeX copied"); }
    else if (e.target.closest("[data-copy]")) { await navigator.clipboard.writeText(e.target.closest("[data-copy]").dataset.copy); toast("Copied"); }
  });
  function askAboutPaper(id) {
    const p = state.papers.find((x) => x.paper_id === id); if (!p) return;
    setTab("ask");
    qEl.value = `What is the main contribution of "${p.title}"?`;
    qEl.dispatchEvent(new Event("input")); qEl.focus();
  }
  $("#library-filter").addEventListener("input", (e) => { state.search = e.target.value; state.page = 1; renderPapers(); });
  $("#lib-filters").addEventListener("click", (e) => { const b = e.target.closest(".chip"); if (b) { state.filter = b.dataset.filter; state.page = 1; renderPapers(); } });

  // ---------- composer ----------
  const qEl = $("#question");
  qEl.addEventListener("input", () => { $("#send").disabled = state.running || !qEl.value.trim(); autosize(); });
  qEl.addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); if (!$("#send").disabled) $("#composer").requestSubmit(); } });
  function autosize() { qEl.style.height = "auto"; qEl.style.height = Math.min(qEl.scrollHeight, 200) + "px"; }
  $("#examples").addEventListener("click", (e) => { const b = e.target.closest(".example"); if (b) { qEl.value = b.textContent; qEl.dispatchEvent(new Event("input")); qEl.focus(); } });
  $("#composer").addEventListener("submit", (e) => { e.preventDefault(); ask(qEl.value.trim()); });
  $("#rerun").addEventListener("click", () => { if (state.current && !state.running) ask(state.current.question); });

  // ---------- ask (SSE over POST) ----------
  function resetWork(question) {
    $("#work").hidden = false;
    $("#synth-title").textContent = question;
    const st = $(".synth-status"); st.className = "synth-status is-live";
    $("#synth-state").textContent = "Retrieving"; $("#synth-model").textContent = $("#model").value;
    $("#thinking").hidden = true; $("#thinking-text").textContent = ""; $("#thinking-count").textContent = "";
    $("#answer").className = "prose"; $("#answer").innerHTML = `<p class="placeholder"><span class="spinner"></span>Retrieving passages (${$("#meta-mode").textContent}${$("#rerank").checked ? " + rerank" : ""})…</p>`;
    $("#answer-error").hidden = true; $("#metrics").hidden = true; $("#synth-actions").hidden = true;
    $("#trace").open = false; $("#trace-lines").innerHTML = ""; $("#trace-sub").textContent = "";
    renderSources([], true); $("#evidence-foot").hidden = true;
    $("#nav-live").hidden = false;
    $("#send-label").textContent = "Synthesizing…";
  }
  async function ask(question) {
    if (!question || state.running) return;
    state.running = true; $("#send").disabled = true;
    const run = { id: state.runs.length + 1, question, started: Date.now(), mode: state.mode, rerank: $("#rerank").checked, k: Number($("#k").value), model: $("#model").value, ctx: Number($("#ctx").value), trace: [], sources: [], done: null, error: null, thinking: "" };
    state.current = run; state.runs.push(run);
    resetWork(question);
    let text = "", thinkChars = 0, t0 = performance.now();
    const trace = (msg, cls = "") => { const t = ((performance.now() - t0) / 1000).toFixed(3); run.trace.push({ t, msg, cls }); renderTrace(run); };
    trace(`Query → ${run.mode} retrieval, k=${run.k}${run.rerank ? ", cross-encoder rerank" : ""}, ${run.ctx} chunks to ${run.model}`);
    try {
      const res = await fetch("/api/ask", { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, mode: run.mode, rerank: run.rerank, k: run.k, model: run.model, context_chunks: run.ctx }) });
      if (!res.ok) throw new Error((await res.json()).error || res.statusText);
      for await (const ev of sse(res)) {
        if (ev.type === "sources") {
          run.sources = state.sources = ev.sources; run.retrieval_s = ev.retrieval_s; run.mode_label = ev.mode;
          renderSources(ev.sources);
          const n = ev.sources.filter((s) => s.in_context).length;
          trace(`${ev.mode}: ${ev.sources.length} candidates scored, top ${n} injected into the prompt`, "ok");
          $("#synth-state").textContent = "Reading"; $("#answer").innerHTML = `<p class="placeholder"><span class="spinner"></span>Reading ${n} passages with ${esc(run.model)}…</p>`;
          $("#trace-sub").textContent = `${n} chunks injected`;
        } else if (ev.type === "thinking") {
          if (thinkChars === 0) { $("#thinking").hidden = false; $("#thinking").open = true; $("#synth-state").textContent = "Reasoning"; trace("model is reasoning before answering (thinking-only build)"); }
          thinkChars += ev.delta.length; run.thinking += ev.delta;
          $("#thinking-text").textContent += ev.delta; $("#thinking-count").textContent = `${num(thinkChars)} chars`;
        } else if (ev.type === "token") {
          if (!text) { $("#synth-state").textContent = "Synthesizing"; $("#answer").classList.add("is-streaming"); $("#thinking").open = false; trace("first token received", "hi"); }
          text += ev.delta; renderProse(text, null);
        } else if (ev.type === "done") {
          run.done = ev; finish(run);
          trace(`done: ${ev.completion_tokens} tokens in ${fmtS(ev.eval_s || ev.latency_s)} (${ev.tokens_per_s} tok/s), ${ev.cited.length} citations, integrity ${ev.integrity.toFixed(2)}`, "hi");
        } else if (ev.type === "error") { run.error = ev.message; showError(ev.message); trace(`error: ${ev.message}`, "warn"); }
      }
    } catch (e) { run.error = e.message; showError(`Request failed: ${e.message}`); }
    finally {
      state.running = false; $("#send").disabled = !qEl.value.trim(); $("#send-label").textContent = "Synthesize"; $("#nav-live").hidden = true;
      $("#nav-runs").textContent = String(state.runs.length);
      $("#meta-run span").textContent = `Query run #${run.id}`;
      if (!run.done) { const st = $(".synth-status"); st.className = "synth-status is-error"; $("#synth-state").textContent = "Failed"; }
    }
  }
  async function* sse(res) {
    const reader = res.body.getReader(); const dec = new TextDecoder(); let buf = "";
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let i;
      while ((i = buf.indexOf("\n\n")) >= 0) {
        const raw = buf.slice(0, i); buf = buf.slice(i + 2);
        const line = raw.split("\n").find((l) => l.startsWith("data: "));
        if (line) yield JSON.parse(line.slice(6));
      }
    }
  }
  function showError(msg) { $("#answer").innerHTML = ""; $("#answer").classList.remove("is-streaming"); const e = $("#answer-error"); e.hidden = false; e.innerHTML = esc(msg).replace(/`([^`]+)`/g, "<code>$1</code>"); }

  // answer text -> paragraphs; at the end citations become chips. [S1], [S1, S3], (S2) -> [1] [3] (2)
  const CITE = /[\[(]\s*(S\d{1,3}(?:\s*[,;]\s*S\d{1,3})*)\s*[\])]/gi;
  function inline(t, byId) {
    let h = esc(t).replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>").replace(/(^|\s)\*([^*\n]+)\*(?=[\s.,;:)]|$)/g, "$1<em>$2</em>");
    if (!byId) return h;
    return h.replace(CITE, (m, innerIds) => innerIds.split(/[,;]/).map((x) => {
      const id = x.trim().toUpperCase(); const src = byId[id]; const n = id.slice(1);
      return src ? `<button type="button" class="cite" data-chunk="${esc(src.chunk_id)}" data-id="${id}" title="${esc(src.label)} · §${esc(src.section)}">[${n}]</button>`
                 : `<span class="cite is-invalid" title="Cites a passage that was not retrieved">[${esc(n)}]</span>`;
    }).join(""));
  }
  function renderProse(text, byId) {
    const blocks = text.replace(/\r/g, "").split(/\n\s*\n/).filter((b) => b.trim());
    const html = blocks.map((b) => {
      const lines = b.split("\n");
      if (lines.every((l) => /^\s*([-*•]|\d+[.)])\s+/.test(l))) return `<ul>${lines.map((l) => `<li>${inline(l.replace(/^\s*([-*•]|\d+[.)])\s+/, ""), byId)}</li>`).join("")}</ul>`;
      return `<p>${inline(b.replace(/\n/g, " "), byId)}</p>`;
    }).join("");
    $("#answer").innerHTML = html || `<p class="placeholder">The model returned an empty answer.</p>`;
  }
  function finish(run) {
    const ev = run.done;
    $("#answer").classList.remove("is-streaming");
    const byId = Object.fromEntries(state.sources.filter((s) => s.id).map((s) => [s.id, s]));
    renderProse(ev.answer, byId);
    const cited = new Set(ev.cited);
    for (const s of state.sources) s.cited = cited.has(s.chunk_id);
    renderSources(state.sources);
    const st = $(".synth-status"); st.className = "synth-status is-done";
    $("#synth-state").textContent = ev.cited.length ? "Grounded Synthesis" : "Synthesis · no citations";
    const inCtx = state.sources.filter((s) => s.in_context).length;
    const okInt = ev.integrity >= 1 && ev.cited.length > 0;
    $("#metrics").hidden = false;
    $("#metrics").innerHTML = `
      <span><span class="dim">Retrieval:</span><b>${esc(run.mode_label || run.mode)}</b> <span class="dim">${fmtMs(run.retrieval_s || 0)}</span></span><span class="bar">|</span>
      <span><span class="dim">Chunks:</span><b>${state.sources.length} evaluated (${inCtx} injected)</b></span><span class="bar">|</span>
      <span><span class="dim">Latency:</span><b>${fmtS(ev.eval_s || ev.latency_s)} (${ev.tokens_per_s} tok/s)</b></span><span class="bar">|</span>
      <span><span class="dim">Tokens:</span><b>${ev.prompt_tokens} prompt · ${ev.completion_tokens} answer${ev.thinking_chars ? ` · ${num(ev.thinking_chars)} reasoning chars` : ""}</b></span><span class="bar">|</span>
      <span class="${okInt ? "ok" : "warn"}"><svg class="ic"><use href="#${okInt ? "i-check-circle" : "i-alert"}"/></svg>Citation integrity ${ev.integrity.toFixed(2)}${ev.cited.length ? "" : " · no citations"}${ev.invalid.length ? ` · ${ev.invalid.length} outside the retrieved set` : ""}</span>`;
    $("#synth-actions").hidden = false;
    $("#evidence-foot").hidden = false;
    $("#evidence-foot-text").textContent = `${inCtx} of ${state.sources.length} candidates sent to the model`;
    const first = $("#answer .cite[data-id]"); if (first) activateCite(first.dataset.id, false);
  }
  function markdown(run) {
    const ev = run.done; if (!ev) return "";
    const srcs = run.sources.filter((s) => s.in_context).map((s) => `[${s.id.slice(1)}] ${s.label} — ${s.title} (§${s.section}, pp. ${s.pages.join("–")}) · ${s.chunk_id}`);
    return `## ${run.question}\n\n${ev.answer.replace(CITE, (m, ids) => ids.split(/[,;]/).map((x) => `[${x.trim().toUpperCase().slice(1)}]`).join(""))}\n\n### Sources\n${srcs.join("\n")}\n\n_${ev.model} · ${run.mode_label} · integrity ${ev.integrity.toFixed(2)} · ${fmtS(ev.eval_s || ev.latency_s)} at ${ev.tokens_per_s} tok/s_\n`;
  }
  for (const id of ["copy-md", "copy-md-2"]) $(`#${id}`).addEventListener("click", async () => { if (state.current?.done) { await navigator.clipboard.writeText(markdown(state.current)); toast("Markdown copied"); } });
  $("#copy-json").addEventListener("click", async () => { if (state.current?.done) { const r = state.current; await navigator.clipboard.writeText(JSON.stringify({ question: r.question, config: { mode: r.mode, rerank: r.rerank, k: r.k, context_chunks: r.ctx, model: r.model }, retrieval_s: r.retrieval_s, sources: r.sources, answer: r.done }, null, 2)); toast("JSON copied"); } });
  function renderTrace(run) {
    $("#trace-lines").innerHTML = run.trace.map((l) => `<li class="${l.cls}"><span class="t">[${l.t}s]</span>${esc(l.msg)}</li>`).join("");
  }

  // ---------- evidence ----------
  function renderSources(sources, loading = false) {
    const list = $("#source-list"); list.innerHTML = "";
    const inCtx = sources.filter((s) => s.in_context).length;
    const papers = new Set(sources.filter((s) => s.in_context).map((s) => s.paper_id)).size;
    $("#evidence-count").textContent = loading ? "retrieving…" : sources.length ? `${inCtx} chunks · ${papers} papers` : "";
    if (loading) { for (let i = 0; i < 3; i++) { const li = document.createElement("li"); li.className = "skeleton"; list.appendChild(li); } return; }
    const top = Math.max(...sources.map((s) => s.score), 1e-9);
    for (const s of sources) {
      const li = document.createElement("li");
      li.className = `ev ${s.in_context ? "" : "is-out"}`; li.dataset.chunk = s.chunk_id; if (s.id) li.dataset.id = s.id;
      li.tabIndex = 0; li.setAttribute("aria-expanded", "false");
      const pct = Math.round(100 * (s.score > 0 ? s.score / top : 0));
      const pages = s.pages[0] === s.pages[1] ? `p. ${s.pages[0]}` : `pp. ${s.pages[0]}–${s.pages[1]}`;
      const paper = state.papers.find((p) => p.paper_id === s.paper_id);
      const quote = s.text.replace(/^[^\n]*\n\n/, "").replace(/<!--[\s\S]*?-->/g, " ").replace(/<\/?[a-z][^>]*>/gi, "").replace(/\s+/g, " ").trim();
      li.innerHTML = `
        <div class="ev-head"><span>${s.id ? `<span class="ev-num">[${s.id.slice(1)}]</span>` : `<span class="ev-num dim">·</span>`}<span class="ev-match ${s.cited ? "is-cited" : ""}">${s.cited ? "cited · " : ""}${pct}% of top score</span></span><span class="ev-chunk">rank ${s.rank} · ${s.n_tokens} tok</span></div>
        <h3 class="ev-title">${esc(s.title)}</h3>
        <div class="ev-meta"><b>${esc(paper ? authorsShort(paper) : s.label)}</b><span>•</span><span>${s.year ?? "—"}</span>${paper?.venue ? `<span>•</span><i>${esc(paper.venue)}</i>` : ""}</div>
        <div class="ev-section"><span>§ ${esc(s.section)} · ${pages}</span><svg class="ic"><use href="#i-pin"/></svg></div>
        <p class="ev-quote">“${esc(quote)}”</p>
        <div class="ev-foot"><span class="mono dim">${esc(paper?.doi ? `doi:${paper.doi}` : s.chunk_id)}</span>${s.in_context ? `<span class="link">${s.cited ? "cited in answer" : "sent to model"}</span>` : `<span class="dim">retrieved, not sent</span>`}</div>`;
      const toggle = () => { const open = li.getAttribute("aria-expanded") === "true"; li.setAttribute("aria-expanded", String(!open)); if (s.id) activateCite(s.id, false); };
      li.addEventListener("click", toggle);
      li.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); toggle(); } });
      list.appendChild(li);
    }
  }
  function activateCite(id, scroll = true) {
    for (const c of $$("#answer .cite")) c.classList.toggle("is-on", c.dataset.id === id);
    for (const li of $$("#source-list .ev")) li.classList.toggle("is-on", li.dataset.id === id);
    const li = $(`#source-list .ev[data-id="${id}"]`);
    if (li && scroll) { li.setAttribute("aria-expanded", "true"); li.scrollIntoView({ behavior: "smooth", block: "center" }); }
  }
  $("#answer").addEventListener("click", (e) => { const c = e.target.closest(".cite[data-id]"); if (c) activateCite(c.dataset.id, true); });

  // ---------- runs ----------
  function showRun(run) {
    setTab("ask"); state.current = run; state.sources = run.sources;
    resetWork(run.question); renderTrace(run);
    $("#synth-model").textContent = run.model; $("#trace-sub").textContent = `${run.sources.filter((s) => s.in_context).length} chunks injected`;
    if (run.thinking) { $("#thinking").hidden = false; $("#thinking-text").textContent = run.thinking; $("#thinking-count").textContent = `${num(run.thinking.length)} chars`; }
    if (run.done) { renderSources(run.sources); finish(run); }
    else { renderSources(run.sources); if (run.error) showError(run.error); }
    $("#send-label").textContent = "Synthesize"; $("#nav-live").hidden = !state.running;
    $("#meta-run span").textContent = `Query run #${run.id}`;
  }
  async function loadRuns() {
    const ol = $("#session-runs"); ol.innerHTML = "";
    if (!state.runs.length) ol.innerHTML = `<li class="empty dim">Nothing asked yet. Runs appear here as soon as you synthesize an answer.</li>`;
    for (const r of [...state.runs].reverse()) {
      const li = document.createElement("li"); li.className = "run";
      const d = r.done, ago = Math.round((Date.now() - r.started) / 60000);
      li.innerHTML = `
        <div class="run-l"><div class="run-id"><b>run-${String(r.id).padStart(3, "0")}</b><span>•</span><span>${esc(r.model)}</span><span>•</span><span>${ago < 1 ? "just now" : `${ago} min ago`}</span></div><p class="run-q">${esc(r.question)}</p></div>
        <div class="run-r">
          <div><span>Latency / Speed</span><b>${d ? `${fmtS(d.eval_s || d.latency_s)} · ${d.tokens_per_s} tok/s` : "—"}</b></div>
          <div><span>Context</span><b>${r.sources.filter((s) => s.in_context).length} chunks${d ? ` (${d.prompt_tokens} tok)` : ""}</b></div>
          <span class="run-state ${d && d.integrity >= 1 && d.cited.length ? "" : "is-warn"}"><svg class="ic"><use href="#${d ? (d.integrity >= 1 && d.cited.length ? "i-check-circle" : "i-alert") : "i-clock"}"/></svg>${d ? (d.integrity >= 1 && d.cited.length ? "Grounded & verified" : `Integrity ${d.integrity.toFixed(2)}`) : r.error ? "Failed" : "Running"}</span>
          <svg class="ic ic-muted"><use href="#i-arrow"/></svg></div>`;
      li.addEventListener("click", () => showRun(r));
      ol.appendChild(li);
    }
    try {
      const r = await (await fetch("/api/runs?limit=40")).json();
      $("#runs-total").textContent = `${r.total} rows`; $("#runs-path").textContent = r.path.replace(/^.*[\\/](results[\\/])/, "$1");
      const f = (v, d = 2) => (v === undefined || v === "" ? "—" : Number.isFinite(Number(v)) ? Number(v).toFixed(d) : esc(v));
      $("#run-rows").innerHTML = r.runs.map((x) => `<tr><td>${esc((x.timestamp || "").replace("T", " "))}</td><td class="phase">${esc(x.phase || "—")}</td><td title="${esc(x.notes || "")}">${esc(x.run_name || "—")}</td><td>${esc(x.retrieval_mode || "—")}${x.rerank === "True" ? " + rerank" : ""}</td><td>${esc(x.llm_model || "—")}</td><td class="r">${f(x.recall_at_5)}</td><td class="r">${f(x.mrr)}</td><td class="r">${f(x.citation_integrity)}</td><td class="r">${x.gen_p50_latency_s || x.p50_latency_s ? fmtS(Number(x.gen_p50_latency_s || x.p50_latency_s)) : "—"}</td><td class="r">${x.peak_rss_mb ? `${Math.round(x.peak_rss_mb)} MB` : "—"}</td></tr>`).join("")
        || `<tr><td colspan="10" class="dim">No rows in the log yet.</td></tr>`;
    } catch (e) { $("#run-rows").innerHTML = `<tr><td colspan="10" class="dim">Could not read the log: ${esc(e.message)}</td></tr>`; }
  }

  // ---------- corpora ----------
  for (const id of ["crumb-scope", "open-corpora"]) $(`#${id}`).addEventListener("click", () => { openModal("modal-corpora"); loadCorpora(); });
  async function loadCorpora() {
    const list = $("#corpora-list");
    try {
      const r = await (await fetch("/api/corpora")).json();
      renderCorpora(r);
    } catch (e) { list.innerHTML = `<li class="dim">Could not list corpora: ${esc(e.message)}</li>`; }
  }
  function renderCorpora(r) {
    const list = $("#corpora-list"); list.innerHTML = "";
    for (const c of r.corpora) {
      const li = document.createElement("li"); li.className = `corpus ${c.active ? "is-on" : ""}`;
      li.innerHTML = `
        <div class="corpus-l"><div class="corpus-name">${esc(c.name)}${c.active ? `<span class="badge mono badge-ok">active</span>` : ""}</div>
          <div class="corpus-q">${c.query ? `“${esc(c.query)}”` : "<i>no search query set</i>"}</div>
          <div class="corpus-stats">${c.n_papers} papers · ${c.n_fulltext} full text · ${num(c.n_chunks)} chunks · ${c.index_size_mb} MB · ${esc(c.id)}/</div></div>
        ${c.active ? `<span class="dim mono" style="font-size:11px">in use</span>` : `<button type="button" class="btn btn-quiet" data-switch="${esc(c.id)}"><svg class="ic"><use href="#i-arrow"/></svg>Switch</button>`}`;
      list.appendChild(li);
    }
  }
  async function switchCorpus(body) {
    const res = await fetch("/api/corpus", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    const r = await res.json();
    if (!res.ok) { toast(r.error || res.statusText); return false; }
    renderCorpora(r);
    state.selected = null; state.page = 1; state.search = ""; $("#library-filter").value = "";
    $("#work").hidden = true; state.current = null; state.sources = [];
    await Promise.all([loadStatus(), loadPapers()]);
    toast(`Corpus: ${r.corpora.find((c) => c.active)?.name || r.active}`);
    return true;
  }
  $("#corpora-list").addEventListener("click", async (e) => {
    const b = e.target.closest("[data-switch]"); if (!b) return;
    b.disabled = true; await switchCorpus({ id: b.dataset.switch }); b.disabled = false;
  });
  $("#corpus-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    $("#corpus-create").disabled = true;
    const ok = await switchCorpus({ name: $("#corpus-name").value, query: $("#corpus-query").value });
    $("#corpus-create").disabled = false;
    if (ok) { $("#corpus-name").value = ""; $("#corpus-query").value = ""; closeModals(); openModal("modal-add"); }
  });

  // ---------- discover ----------
  $("#discover-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const steps = $("#discover-steps"); steps.innerHTML = ""; $("#discover-run").disabled = true;
    const add = (msg, cls = "is-live") => { const li = document.createElement("li"); li.className = cls; li.innerHTML = `<svg class="ic"><use href="#i-check-circle"/></svg><span>${esc(msg)}</span>`; steps.appendChild(li); return li; };
    let live = null;
    try {
      const res = await fetch("/api/discover", { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: $("#discover-query").value, limit: Number($("#discover-limit").value) }) });
      if (!res.ok) { add((await res.json()).error, "is-error"); return; }
      for await (const ev of sse(res)) {
        if (live) live.className = "";
        if (ev.type === "step") live = add(ev.message);
        else if (ev.type === "done") { add(ev.message, ""); await Promise.all([loadStatus(), loadPapers()]); toast("Corpus updated"); }
        else if (ev.type === "error") add(ev.message, "is-error");
      }
    } catch (err) { add(`Request failed: ${err.message}`, "is-error"); }
    finally { $("#discover-run").disabled = false; }
  });

  // ---------- boot ----------
  loadStatus().catch((e) => { $("#status-pill").classList.add("is-down"); $("#status-pill-text").textContent = `Server did not answer: ${e.message}`; });
  loadPapers().catch(() => { $("#paper-rows").innerHTML = `<tr><td colspan="6" class="dim">Could not load the library.</td></tr>`; });
})();
