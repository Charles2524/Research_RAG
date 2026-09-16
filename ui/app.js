/* Reading Room — client. Vanilla JS, no build step. Talks only to the local server (server.py). */
(() => {
  "use strict";
  const $ = (s, el = document) => el.querySelector(s);
  const $$ = (s, el = document) => [...el.querySelectorAll(s)];
  const state = { status: null, papers: [], running: false, sources: [], mode: "hybrid" };

  const fmtS = (s) => (s >= 10 ? `${Math.round(s)} s` : `${s.toFixed(1)} s`);
  const fmtMs = (s) => (s < 1 ? `${Math.round(s * 1000)} ms` : fmtS(s));
  const esc = (t) => t.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

  // ---------- status + library ----------
  async function loadStatus() {
    const r = await fetch("/api/status");
    state.status = await r.json();
    const s = state.status;
    $("#corpus-line").textContent =
      `${s.n_fulltext} papers with full text · ${s.n_chunks} passages · ${s.vector_backend}`;
    const pill = $("#privacy-pill");
    if (s.offline) {
      pill.classList.replace("pill-local", "pill-offline");
      $("#privacy-text").textContent = "Offline mode · network disabled";
      $("#open-discover").disabled = true;
    }
    const sel = $("#model");
    sel.innerHTML = "";
    const models = s.models.length ? s.models : [s.default_model, s.fallback_model];
    for (const m of models) {
      const o = document.createElement("option");
      o.value = m; o.textContent = m + (m === s.fallback_model ? " · fast" : m === s.default_model ? " · reasons" : "");
      sel.appendChild(o);
    }
    sel.value = models.includes(s.fallback_model) ? s.fallback_model : models[0];
    $("#k").value = s.k;
    $("#rerank").checked = !!s.rerank;
    setMode(s.retrieval_mode);
    if (s.ollama !== "up") showBanner(`Ollama isn't running. Start it with <code>ollama serve</code>, then ask again.`);
    $("#send").disabled = !$("#question").value.trim();
  }

  async function loadPapers() {
    const r = await fetch("/api/papers");
    state.papers = (await r.json()).papers;
    renderPapers();
  }

  function renderPapers(filter = "") {
    const list = $("#paper-list");
    const q = filter.trim().toLowerCase();
    const rows = state.papers.filter((p) => !q ||
      p.title.toLowerCase().includes(q) || String(p.year).includes(q) || p.authors.join(" ").toLowerCase().includes(q));
    $("#library-count").textContent = q ? `${rows.length} of ${state.papers.length}` : `${state.papers.length}`;
    list.innerHTML = "";
    if (!rows.length) {
      list.innerHTML = `<li class="empty-note">No paper matches that. Try a word from the title or an author's surname.</li>`;
      return;
    }
    for (const p of rows) {
      const li = document.createElement("li");
      li.className = "paper"; li.tabIndex = 0; li.setAttribute("aria-expanded", "false");
      const first = (p.authors[0] || "").trim();
      const authors = !first ? "Unknown authors" : first.length > 28 ? first.slice(0, 27) + "…" : first + (p.authors.length > 1 ? " et al." : "");
      li.innerHTML = `
        <div class="paper-title">${esc(p.title)}</div>
        <div class="paper-meta"><span class="year">${p.year ?? "—"}</span><span>${esc(authors)}</span>
          <span class="tag ${p.has_full_text ? "" : "tag-meta"}">${p.has_full_text ? `${p.n_chunks} passages` : "metadata only"}</span></div>`;
      const detail = document.createElement("div");
      detail.className = "paper-detail"; detail.hidden = true;
      detail.innerHTML = `${p.venue ? esc(p.venue) + " · " : ""}${p.doi ? `<span class="mono">${esc(p.doi)}</span>` : "no DOI"}
        ${p.metadata_resolved ? "" : " · metadata unresolved"}${p.abstract ? `<p style="margin-top:6px">${esc(p.abstract)}${p.abstract.length >= 600 ? "…" : ""}</p>` : ""}`;
      li.appendChild(detail);
      const toggle = () => { const open = li.getAttribute("aria-expanded") === "true"; li.setAttribute("aria-expanded", String(!open)); detail.hidden = open; };
      li.addEventListener("click", toggle);
      li.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); toggle(); } });
      list.appendChild(li);
    }
  }
  $("#library-filter").addEventListener("input", (e) => renderPapers(e.target.value));

  // ---------- composer ----------
  function setMode(m) {
    state.mode = m;
    for (const b of $$("#mode-seg button")) { const on = b.dataset.mode === m; b.classList.toggle("is-on", on); b.setAttribute("aria-checked", String(on)); }
  }
  $("#mode-seg").addEventListener("click", (e) => { const b = e.target.closest("button"); if (b) setMode(b.dataset.mode); });
  const qEl = $("#question");
  qEl.addEventListener("input", () => { $("#send").disabled = state.running || !qEl.value.trim(); autosize(); });
  qEl.addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); if (!$("#send").disabled) $("#composer").requestSubmit(); } });
  function autosize() { qEl.style.height = "auto"; qEl.style.height = Math.min(qEl.scrollHeight, 160) + "px"; }
  $("#examples").addEventListener("click", (e) => { const b = e.target.closest(".example"); if (b) { qEl.value = b.textContent; qEl.dispatchEvent(new Event("input")); qEl.focus(); } });
  $("#composer").addEventListener("submit", (e) => { e.preventDefault(); ask(qEl.value.trim()); });

  function showBanner(html) {
    let b = $("#banner");
    if (!b) { b = document.createElement("div"); b.id = "banner"; b.className = "error"; $("#thread").prepend(b); }
    b.innerHTML = html;
  }

  // ---------- ask (SSE over POST) ----------
  async function ask(question) {
    if (!question || state.running) return;
    state.running = true; $("#send").disabled = true;
    $("#empty-state")?.remove(); $("#banner")?.remove();
    const ex = document.createElement("article"); ex.className = "exchange";
    ex.innerHTML = `<h3 class="q">${esc(question)}</h3><div class="status"><span class="dot"></span><span class="status-text">Retrieving…</span></div>`;
    const thread = $("#thread"); thread.appendChild(ex); thread.scrollTop = thread.scrollHeight;
    const statusText = $(".status-text", ex);
    qEl.value = ""; autosize();
    renderSources([], "Retrieving passages…");
    setPanel("ask");
    let answerEl = null, thinkEl = null, thinkChars = 0, text = "";
    try {
      const res = await fetch("/api/ask", { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, mode: state.mode, rerank: $("#rerank").checked, k: Number($("#k").value), model: $("#model").value }) });
      if (!res.ok) throw new Error((await res.json()).error || res.statusText);
      for await (const ev of sse(res)) {
        if (ev.type === "sources") {
          state.sources = ev.sources; renderSources(ev.sources);
          const n = ev.sources.filter((s) => s.in_context).length;
          statusText.textContent = `Reading ${n} passages with ${$("#model").value.split(":")[0]} ${$("#model").value.split(":")[1]}…`;
          ex.dataset.retrieval = ev.retrieval_s; ex.dataset.mode = ev.mode;
        } else if (ev.type === "thinking") {
          if (!thinkEl) { thinkEl = document.createElement("details"); thinkEl.className = "thinking"; thinkEl.innerHTML = `<summary>Reasoning <span class="count"></span></summary><pre></pre>`; ex.appendChild(thinkEl); }
          thinkChars += ev.delta.length; $("pre", thinkEl).textContent += ev.delta; $(".count", thinkEl).textContent = `${thinkChars} chars`;
          statusText.textContent = "Reasoning before answering…";
        } else if (ev.type === "token") {
          if (!answerEl) { answerEl = document.createElement("p"); answerEl.className = "answer is-streaming"; ex.appendChild(answerEl); $(".status", ex).hidden = true; }
          text += ev.delta; answerEl.textContent = text;
          thread.scrollTop = thread.scrollHeight;
        } else if (ev.type === "done") {
          finish(ex, answerEl, ev);
        } else if (ev.type === "error") {
          $(".status", ex).hidden = true;
          const err = document.createElement("div"); err.className = "error"; err.innerHTML = esc(ev.message).replace(/`([^`]+)`/g, "<code>$1</code>"); ex.appendChild(err);
        }
      }
    } catch (e) {
      $(".status", ex).hidden = true;
      const err = document.createElement("div"); err.className = "error"; err.textContent = `Request failed: ${e.message}`; ex.appendChild(err);
    } finally {
      state.running = false; $("#send").disabled = !qEl.value.trim(); qEl.focus();
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

  function finish(ex, answerEl, ev) {
    if (!answerEl) { answerEl = document.createElement("p"); answerEl.className = "answer"; ex.appendChild(answerEl); }
    answerEl.classList.remove("is-streaming");
    $(".status", ex).hidden = true;
    const byId = Object.fromEntries(state.sources.filter((s) => s.id).map((s) => [s.id, s]));
    const citedIds = new Set(ev.cited);
    // [S1], [S1, S3], (S2) -> chips; **bold** / *italic* rendered; anything else stays literal text
    const html = esc(ev.answer).replace(/\*\*([^*
]+)\*\*/g, "<strong>$1</strong>").replace(/(^|\s)\*([^*
]+)\*(?=[\s.,;:)]|$)/g, "$1<em>$2</em>").replace(/[\[(]\s*(S\d{1,3}(?:\s*[,;]\s*S\d{1,3})*)\s*[\])]/gi, (m, inner) =>
      inner.split(/[,;]/).map((t) => {
        const id = t.trim().toUpperCase(); const src = byId[id];
        return src ? `<button type="button" class="cite" data-chunk="${esc(src.chunk_id)}" title="${esc(src.label)} · §${esc(src.section)}">${id}</button>`
                   : `<span class="cite is-invalid" title="Not among the retrieved passages">${esc(id)}</span>`;
      }).join(""));
    answerEl.innerHTML = html;
    for (const s of state.sources) s.cited = citedIds.has(s.chunk_id);
    renderSources(state.sources);
    const okInt = ev.integrity >= 1, cited = ev.cited.length;
    const v = document.createElement("div"); v.className = "verify";
    v.innerHTML = `
      <span title="Share of citations that point at a retrieved passage">Citation integrity <b class="${okInt ? "ok" : "warn"}">${ev.integrity.toFixed(2)}</b>${cited ? "" : ` <span class="warn">· no citations</span>`}${ev.invalid.length ? ` <span class="warn">· ${ev.invalid.length} outside the set</span>` : ""}</span>
      <span>Retrieval <b>${fmtMs(Number(ex.dataset.retrieval || 0))}</b> <span>${esc(ex.dataset.mode || "")}</span></span>
      <span>Generation <b>${fmtS(ev.eval_s || ev.latency_s)}</b> at <b>${ev.tokens_per_s}</b> tok/s</span>
      <span>Prompt <b>${ev.prompt_tokens}</b> · answer <b>${ev.completion_tokens}</b> tokens${ev.thinking_chars ? ` · reasoning <b>${ev.thinking_chars}</b> chars` : ""}</span>
      <span>${esc(ev.model)}</span>`;
    ex.appendChild(v);
    $("#thread").scrollTop = $("#thread").scrollHeight;
  }
  $("#thread").addEventListener("click", (e) => { const c = e.target.closest(".cite[data-chunk]"); if (c) revealSource(c.dataset.chunk); });

  // ---------- evidence ----------
  function renderSources(sources, note) {
    const list = $("#source-list");
    list.innerHTML = "";
    const inCtx = sources.filter((s) => s.in_context).length;
    $("#evidence-count").textContent = sources.length ? `${inCtx} sent · ${sources.length} retrieved` : "";
    $("#evidence-sub").textContent = note || (sources.length ? "Passages the model read, in rank order. Marked ones are cited in the answer." : "Passages retrieved for the current answer appear here.");
    if (!sources.length && note) { for (let i = 0; i < 4; i++) { const li = document.createElement("li"); li.className = "skeleton"; list.appendChild(li); } return; }
    for (const s of sources) {
      const li = document.createElement("li");
      li.className = `source ${s.cited ? "is-cited" : ""} ${s.in_context ? "" : "is-out"}`;
      li.dataset.chunk = s.chunk_id; li.tabIndex = 0; li.setAttribute("aria-expanded", "false");
      const pages = s.pages[0] === s.pages[1] ? `p. ${s.pages[0]}` : `pp. ${s.pages[0]}–${s.pages[1]}`;
      li.innerHTML = `
        <div class="source-head"><span class="source-id">${s.id ?? "·"}</span><span class="source-label">${esc(s.label)}</span><span class="source-where">§${esc(s.section)} · ${pages}</span></div>
        <div class="source-text">${esc(s.text.replace(/^[^\n]*\n\n/, ""))}</div>
        <div class="source-foot"><span>rank ${s.rank}</span><span>${s.n_tokens} tokens</span>${s.cited ? `<span class="tag tag-cited">cited</span>` : s.in_context ? `` : `<span class="tag tag-meta">retrieved, not sent</span>`}</div>`;
      const toggle = () => { const open = li.getAttribute("aria-expanded") === "true"; li.setAttribute("aria-expanded", String(!open)); };
      li.addEventListener("click", toggle);
      li.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); toggle(); } });
      list.appendChild(li);
    }
  }
  function revealSource(chunkId) {
    setPanel("evidence");
    const li = $(`.source[data-chunk="${CSS.escape(chunkId)}"]`);
    if (!li) return;
    li.setAttribute("aria-expanded", "true");
    li.scrollIntoView({ behavior: "smooth", block: "center" });
    li.classList.remove("is-marked"); void li.offsetWidth; li.classList.add("is-marked");
  }

  // ---------- panels (narrow screens) ----------
  function setPanel(p) {
    if (window.matchMedia("(min-width: 1181px)").matches) return;
    document.body.dataset.panel = p;
    for (const b of $$("#mobile-switch button")) b.classList.toggle("is-on", b.dataset.panel === p);
  }
  $("#mobile-switch").addEventListener("click", (e) => { const b = e.target.closest("button"); if (b) setPanel(b.dataset.panel); });

  // ---------- discover drawer ----------
  const drawer = $("#discover");
  $("#open-discover").addEventListener("click", () => { drawer.hidden = false; $("#discover-query").focus(); });
  drawer.addEventListener("click", (e) => { if (e.target.closest("[data-close]")) drawer.hidden = true; });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape" && !drawer.hidden) drawer.hidden = true; });
  $("#discover-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const steps = $("#discover-steps"); steps.innerHTML = ""; $("#discover-run").disabled = true;
    const add = (msg, cls = "is-live") => { const li = document.createElement("li"); li.className = cls; li.innerHTML = `<svg class="ic"><use href="#i-check"/></svg><span>${esc(msg)}</span>`; steps.appendChild(li); return li; };
    let live = null;
    try {
      const res = await fetch("/api/discover", { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: $("#discover-query").value, limit: Number($("#discover-limit").value) }) });
      if (!res.ok) { add((await res.json()).error, "is-error"); return; }
      for await (const ev of sse(res)) {
        if (live) live.className = "";
        if (ev.type === "step") live = add(ev.message);
        else if (ev.type === "done") { add(ev.message, ""); await Promise.all([loadStatus(), loadPapers()]); }
        else if (ev.type === "error") add(ev.message, "is-error");
      }
    } catch (err) { add(`Request failed: ${err.message}`, "is-error"); }
    finally { $("#discover-run").disabled = false; }
  });

  // ---------- boot ----------
  loadStatus().catch((e) => showBanner(`The local server did not answer: ${esc(e.message)}`));
  loadPapers().catch(() => { $("#paper-list").innerHTML = `<li class="empty-note">Could not load the library.</li>`; });
})();
