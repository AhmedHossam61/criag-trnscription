"use strict";

/* Lorekeeper: chat over the campaign archive.
   Conversations live in this browser (localStorage); documents and the index
   live on the server. The server streams each answer as server-sent events:
   route -> plan (one per search round) -> sources -> token... -> done | error. */

const $ = (sel, root = document) => root.querySelector(sel);
const el = (tag, attrs = {}, ...children) => {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null || v === false) continue;
    if (k === "class") node.className = v;
    else if (k === "text") node.textContent = v;
    else if (k === "html") node.innerHTML = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v === true ? "" : v);
  }
  for (const child of children.flat()) {
    if (child == null || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
};

const EXAMPLES = [
  { q: "What does the Ritual Adept feature let me do?", tag: "Rules" },
  { q: "Compare what the cult and the guardians want from the celestial shards.", tag: "Multi-step" },
  { q: "What happened at the altar in session 19?", tag: "Transcripts" },
  { q: "الـ DM قال ايه عن The Nameless Beyond؟", tag: "Arabic" },
];

/* ---------------------------------------------------------------- storage */

const STORE_KEY = "lorekeeper.v1";
const storage = {
  get(key, fallback) {
    try { const raw = localStorage.getItem(key); return raw ? JSON.parse(raw) : fallback; }
    catch (e) { return fallback; }
  },
  set(key, value) {
    try { localStorage.setItem(key, JSON.stringify(value)); } catch (e) { /* private mode or full */ }
  },
};

const state = {
  convs: [],          // newest first
  currentId: null,
  selected: null,     // index of the assistant message shown in the catalog
  controller: null,   // AbortController of the running answer
  status: null,
};

function loadState() {
  const saved = storage.get(STORE_KEY, null);
  if (saved && Array.isArray(saved.convs)) {
    state.convs = saved.convs;
    state.currentId = saved.currentId;
  }
  for (const c of state.convs) for (const m of c.messages) m.pending = false;
}
function saveState() {
  storage.set(STORE_KEY, { convs: state.convs.slice(0, 60), currentId: state.currentId });
}
const current = () => state.convs.find((c) => c.id === state.currentId) || null;

function newConversation() {
  const conv = { id: Date.now().toString(36) + Math.random().toString(36).slice(2, 6),
                 title: "", created: Date.now(), updated: Date.now(), messages: [] };
  state.convs.unshift(conv);
  state.currentId = conv.id;
  state.selected = null;
  return conv;
}

/* --------------------------------------------------------------- markdown */

const escapeHtml = (s) => s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

function inline(text, maxCite) {
  const codes = [];
  let s = escapeHtml(text).replace(/`([^`]+)`/g, (_, c) => { codes.push(c); return `\u0000${codes.length - 1}\u0000`; });
  s = s.replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
  s = s.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  s = s.replace(/(^|[^*\w])\*([^*\n]+)\*(?!\w)/g, "$1<em>$2</em>");
  s = s.replace(/\[(\d+(?:\s*[,،]\s*\d+)*)\]/g, (whole, nums) => {
    const parts = nums.split(/\s*[,،]\s*/).map(Number);
    if (!parts.every((n) => n >= 1 && n <= maxCite)) return whole;
    return parts.map((n) => `<button type="button" class="cite" data-n="${n}" aria-label="Source ${n}">${n}</button>`).join("");
  });
  return s.replace(/\u0000(\d+)\u0000/g, (_, i) => `<code>${codes[Number(i)]}</code>`);
}

function markdown(text, maxCite = 0) {
  const lines = text.replace(/\r/g, "").split("\n");
  const out = [];
  let para = [], list = null, quote = [];
  const flushPara = () => { if (para.length) { out.push(`<p dir="auto">${inline(para.join(" "), maxCite)}</p>`); para = []; } };
  const flushList = () => { if (list) { out.push(`<${list.tag} dir="auto">${list.items.map((i) => `<li>${inline(i, maxCite)}</li>`).join("")}</${list.tag}>`); list = null; } };
  const flushQuote = () => { if (quote.length) { out.push(`<blockquote dir="auto">${inline(quote.join(" "), maxCite)}</blockquote>`); quote = []; } };
  const flushAll = () => { flushPara(); flushList(); flushQuote(); };

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    if (/^```/.test(line)) {
      flushAll();
      const code = [];
      while (++i < lines.length && !/^```/.test(lines[i])) code.push(lines[i]);
      out.push(`<pre><code>${escapeHtml(code.join("\n"))}</code></pre>`);
      continue;
    }
    let m;
    if ((m = line.match(/^(#{1,6})\s+(.*)$/))) {
      flushAll();
      const level = m[1].length <= 2 ? 3 : 4;
      out.push(`<h${level} dir="auto">${inline(m[2], maxCite)}</h${level}>`);
    } else if ((m = line.match(/^\s*([-*+]|\d+[.)])\s+(.*)$/))) {
      flushPara(); flushQuote();
      const tag = /\d/.test(m[1]) ? "ol" : "ul";
      if (!list || list.tag !== tag) { flushList(); list = { tag, items: [] }; }
      list.items.push(m[2]);
    } else if ((m = line.match(/^>\s?(.*)$/))) {
      flushPara(); flushList(); quote.push(m[1]);
    } else if (!line.trim()) {
      flushAll();
    } else if (list && /^\s{2,}\S/.test(line)) {
      list.items[list.items.length - 1] += " " + line.trim();
    } else {
      flushList(); flushQuote(); para.push(line.trim());
    }
  }
  flushAll();
  return out.join("");
}

/* -------------------------------------------------------------- rendering */

const fmtDate = (ts) => {
  const d = new Date(ts), now = new Date();
  const time = d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  return d.toDateString() === now.toDateString() ? `Today ${time}` : d.toLocaleDateString([], { month: "short", day: "numeric" });
};

function confirmButton(button, label, action) {
  // First click arms the button; a second click within 4 s does it.
  let armed = null;
  button.addEventListener("click", (e) => {
    e.stopPropagation();
    if (armed) { clearTimeout(armed); armed = null; action(); return; }
    const original = { html: button.innerHTML, cls: button.className };
    button.textContent = label;
    button.className = "btn btn-danger confirm";
    armed = setTimeout(() => { button.innerHTML = original.html; button.className = original.cls; armed = null; }, 4000);
  });
  return button;
}

const xIcon = () => el("span", { html: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M7 7l10 10M17 7L7 17"/></svg>' });

function renderChatList() {
  const list = $("#chat-list");
  list.replaceChildren();
  const convs = state.convs.filter((c) => c.messages.length);
  if (!convs.length) {
    list.append(el("li", { class: "chat-empty", text: "Your questions will be listed here." }));
    return;
  }
  for (const conv of convs) {
    const del = el("button", { class: "icon-btn del", type: "button", "aria-label": `Delete “${conv.title}”` }, xIcon());
    confirmButton(del, "Delete", () => {
      state.convs = state.convs.filter((c) => c.id !== conv.id);
      if (state.currentId === conv.id) { state.currentId = null; state.selected = null; }
      saveState(); renderAll();
    });
    list.append(el("li", { class: "chat-item" + (conv.id === state.currentId ? " active" : "") },
      el("button", { class: "open", type: "button", onclick: () => openConversation(conv.id) },
        el("span", { class: "t", dir: "auto", text: conv.title || "Untitled" }),
        el("span", { class: "d", text: `${fmtDate(conv.updated)} · ${Math.ceil(conv.messages.length / 2)} question${conv.messages.length > 2 ? "s" : ""}` })),
      del));
  }
}

function routeChips(msg) {
  const chips = [];
  if (msg.route) {
    const multi = msg.route.mode === "multi";
    const searches = msg.plans.reduce((n, p) => n + p.queries.length, 0);
    chips.push(el("span", { class: "chip " + (multi ? "multi" : "single") },
      multi ? `Multi-step · ${searches} searches` : "One-step"));
    if (msg.route.by !== "you") {
      chips.push(el("span", { class: "chip", title: `Decided by the ${msg.route.by}` },
        "P(multi) ", el("span", { class: "num", text: msg.route.probability.toFixed(2) })));
    } else {
      chips.push(el("span", { class: "chip", text: "chosen by you" }));
    }
  }
  return el("span", { class: "route" }, chips);
}

function renderTurn(conv, i) {
  const msg = conv.messages[i];
  if (msg.role === "user") {
    return el("section", { class: "turn", id: `m-${i}` },
      el("div", { class: "who", text: "You asked" }),
      el("div", { class: "question-text", dir: "auto", text: msg.content }));
  }
  const nSources = msg.sources ? msg.sources.length : 0;
  let body;
  if (msg.content) {
    body = el("div", { class: "answer" + (state.selected === i ? " selected" : ""),
                       html: markdown(msg.content, nSources) + (msg.pending ? '<span class="caret"></span>' : "") });
  } else if (msg.pending) {
    const step = !msg.route ? "Reading the question…"
      : !msg.sources.length && msg.plans.length ? `Searching: ${msg.plans[msg.plans.length - 1].queries.join(" · ")}`
      : nSources ? `Writing the answer from ${nSources} passages…` : "Searching…";
    body = el("div", { class: "answer pending", dir: "auto", text: step });
  }
  const foot = el("div", { class: "answer-foot" });
  if (nSources) {
    foot.append(el("button", { type: "button", onclick: () => selectMessage(i, true) },
      `${nSources} source${nSources > 1 ? "s" : ""}`));
  }
  if (msg.model) foot.append(el("span", { text: msg.model }));
  if (msg.timing) foot.append(el("span", { class: "num", text: `${msg.timing.total_s}s` }));
  if (msg.stopped) foot.append(el("span", { text: "Stopped" }));

  return el("section", { class: "turn", id: `m-${i}` },
    el("div", { class: "who" }, "Archive", routeChips(msg)),
    body,
    msg.error ? el("div", { class: "error-box", role: "alert", text: msg.error }) : null,
    foot.childNodes.length ? foot : null);
}

function renderWelcome() {
  const empty = state.status && state.status.passages === 0;
  const box = el("div", { class: "welcome" },
    el("h2", { text: empty ? "The archive is empty" : "Ask the archive" }),
    el("p", { text: empty
      ? "Add your rulebooks, campaign notes or character sheets in the Library. Session transcripts in data/transcripts are added when you re-index."
      : "Answers come from your documents and session transcripts, with the passages they cite. The router picks one search for simple questions and several for questions with parts." }));
  if (empty) {
    box.append(el("div", {}, el("button", { class: "btn btn-primary", type: "button", onclick: () => { showTab("library"); openRail(); } }, "Open the Library")));
  } else {
    box.append(el("div", { class: "examples" }, EXAMPLES.map((ex) =>
      el("button", { class: "example", type: "button", onclick: () => ask(ex.q) },
        el("span", { class: "tag", text: ex.tag }),
        el("span", { class: "q", dir: "auto", text: ex.q })))));
  }
  return box;
}

function renderThread() {
  const thread = $("#thread");
  const conv = current();
  thread.replaceChildren();
  $("#chat-title").textContent = conv && conv.title ? conv.title : "New question";
  if (!conv || !conv.messages.length) { thread.append(renderWelcome()); return; }
  conv.messages.forEach((_, i) => thread.append(renderTurn(conv, i)));
}

function updateTurn(i) {
  const conv = current();
  const node = document.getElementById(`m-${i}`);
  if (!conv || !node) return;
  const thread = $("#thread");
  const atBottom = thread.scrollHeight - thread.scrollTop - thread.clientHeight < 80;
  node.replaceWith(renderTurn(conv, i));
  if (atBottom) thread.scrollTop = thread.scrollHeight;
}

function scoreBits(s) {
  const bits = [];
  if (s.rerank != null) bits.push(el("span", { title: "Reranker relevance (0–1)" }, "rel ",
    el("span", { class: "meter" }, el("i", { style: `width:${Math.round(s.rerank * 100)}%` })), ` ${s.rerank.toFixed(2)}`));
  if (s.dense != null) bits.push(el("span", { title: "Similarity of meaning (cosine)", text: `sim ${s.dense.toFixed(2)}` }));
  if (s.keyword != null) bits.push(el("span", { title: "Keyword match (BM25)", text: `kw ${s.keyword.toFixed(1)}` }));
  return el("span", { class: "scores" }, bits);
}

function renderCard(s) {
  const kindLabel = s.kind === "transcript" ? "Session" : s.kind.toUpperCase();
  const where = [kindLabel, s.location, s.heading].filter(Boolean).join(" · ");
  const text = el("div", { class: "card-text", dir: "auto", text: s.text });
  const card = el("li", { class: "card", id: `card-${s.n}` },
    el("div", { class: "card-head" },
      el("span", { class: "callno", text: s.n }),
      el("div", {},
        el("div", { class: "card-title", dir: "auto", text: s.title }),
        el("div", { class: "card-where", dir: "auto", text: where }))),
    text);
  const toggle = el("button", { type: "button", text: "Show all", onclick: () => {
    card.classList.toggle("open");
    toggle.textContent = card.classList.contains("open") ? "Show less" : "Show all";
  } });
  const context = el("button", { type: "button", text: "Show context", onclick: async () => {
    context.disabled = true;
    try {
      const p = await api(`/api/passages/${encodeURIComponent(s.id)}`);
      if (p.before) text.prepend(el("div", { class: "context", text: p.before }), "\n");
      if (p.after) text.append("\n", el("div", { class: "context", text: p.after }));
      card.classList.add("open");
      toggle.textContent = "Show less";
      context.remove();
    } catch (e) { context.textContent = e.message; }
  } });
  card.append(el("div", { class: "card-foot" }, toggle, context, scoreBits(s)));
  return card;
}

function renderCatalog() {
  const conv = current();
  const msg = conv && state.selected != null ? conv.messages[state.selected] : null;
  const trace = $("#trace"), cards = $("#cards");
  trace.replaceChildren(); cards.replaceChildren();
  if (!msg || msg.role !== "assistant") {
    cards.append(el("li", { class: "cards-empty", text: "The passages behind an answer show up here, numbered like the citations." }));
    return;
  }
  msg.plans.forEach((p) => {
    const label = p.hop === 1 ? (msg.route && msg.route.mode === "multi" ? `Searches${p.planner === "rules" ? " (split by rules: no LLM)" : ""}` : "Search")
                              : `Follow-up searches, round ${p.hop}`;
    trace.append(el("div", { class: "hop" },
      el("span", { class: "hop-label", text: label }),
      el("ul", {}, p.queries.map((q) => el("li", { dir: "auto", text: q })))));
  });
  if (!msg.sources.length) {
    cards.append(el("li", { class: "cards-empty", text: msg.pending ? "Searching…" : "Nothing matched in the archive." }));
    return;
  }
  msg.sources.forEach((s) => cards.append(renderCard(s)));
}

function renderAll() {
  renderChatList();
  renderThread();
  renderCatalog();
}

/* ----------------------------------------------------------------- panels */

const app = $("#app");
function closePanels() { app.classList.remove("rail-open", "catalog-open"); $("#scrim").hidden = true; }
function openRail() { if (matchMedia("(max-width: 760px)").matches) { app.classList.add("rail-open"); $("#scrim").hidden = false; } }
function openCatalog() { if (matchMedia("(max-width: 1180px)").matches) { app.classList.add("catalog-open"); $("#scrim").hidden = false; } }

function showTab(name) {
  const chats = name === "chats";
  $("#tab-chats").setAttribute("aria-selected", chats);
  $("#tab-library").setAttribute("aria-selected", !chats);
  $("#panel-chats").hidden = !chats;
  $("#panel-library").hidden = chats;
  if (!chats) loadDocuments();
}

function selectMessage(i, reveal = false) {
  state.selected = i;
  renderThread();
  renderCatalog();
  if (reveal) openCatalog();
}

function flashSource(i, n) {
  if (state.selected !== i) selectMessage(i);
  openCatalog();
  const card = document.getElementById(`card-${n}`);
  if (!card) return;
  card.scrollIntoView({ block: "nearest", behavior: "smooth" });
  card.classList.remove("flash"); void card.offsetWidth; card.classList.add("flash");
  setTimeout(() => card.classList.remove("flash"), 1600);
}

function openConversation(id) {
  if (state.controller) return;
  state.currentId = id;
  const conv = current();
  const last = conv ? conv.messages.map((m) => m.role).lastIndexOf("assistant") : -1;
  state.selected = last >= 0 ? last : null;
  saveState(); renderAll(); closePanels();
  $("#thread").scrollTop = $("#thread").scrollHeight;
}

/* -------------------------------------------------------------------- api */

async function api(path, options = {}) {
  const res = await fetch(path, options);
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try { const body = await res.json(); if (body.detail) detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail); } catch (e) {}
    throw new Error(detail);
  }
  return res.json();
}

async function streamChat(body, onEvent, signal) {
  const res = await fetch("/api/chat", { method: "POST", headers: { "Content-Type": "application/json" },
                                         body: JSON.stringify(body), signal });
  if (!res.ok) throw new Error(`The server refused the question (${res.status}).`);
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let cut;
    while ((cut = buffer.indexOf("\n\n")) >= 0) {
      const block = buffer.slice(0, cut);
      buffer = buffer.slice(cut + 2);
      for (const line of block.split("\n")) if (line.startsWith("data: ")) onEvent(JSON.parse(line.slice(6)));
    }
  }
}

/* -------------------------------------------------------------------- ask */

function setBusy(busy) {
  const send = $("#send");
  send.textContent = busy ? "Stop" : "Ask";
  send.classList.toggle("btn-primary", !busy);
  $("#new-chat").disabled = busy;
}

async function ask(question) {
  question = question.trim();
  if (!question || state.controller) return;
  const conv = current() || newConversation();
  if (!conv.title) conv.title = question.length > 70 ? question.slice(0, 68) + "…" : question;
  const history = conv.messages.filter((m) => m.content && !m.error).map((m) => ({ role: m.role, content: m.content }));
  conv.messages.push({ role: "user", content: question });
  const msg = { role: "assistant", content: "", plans: [], sources: [], pending: true };
  conv.messages.push(msg);
  const i = conv.messages.length - 1;
  conv.updated = Date.now();
  state.selected = i;
  saveState(); renderAll();
  $("#thread").scrollTop = $("#thread").scrollHeight;

  const mode = new FormData($("#composer")).get("mode") || "auto";
  state.controller = new AbortController();
  setBusy(true);
  let frame = 0;
  const refresh = () => { if (!frame) frame = requestAnimationFrame(() => { frame = 0; updateTurn(i); }); };

  try {
    await streamChat({ question, history, mode }, (ev) => {
      if (ev.type === "route") msg.route = ev;
      else if (ev.type === "plan") { msg.plans.push(ev); renderCatalog(); }
      else if (ev.type === "sources") { msg.sources = ev.sources; renderCatalog(); }
      else if (ev.type === "token") msg.content += ev.text;
      else if (ev.type === "done") { msg.content = ev.answer; msg.model = ev.model; msg.timing = ev.timing; }
      else if (ev.type === "error") msg.error = ev.message;
      refresh();
    }, state.controller.signal);
  } catch (e) {
    if (e.name === "AbortError") msg.stopped = true;
    else msg.error = `Lost the connection to the server: ${e.message}. Is scribe serve still running?`;
  } finally {
    msg.pending = false;
    state.controller = null;
    setBusy(false);
    conv.updated = Date.now();
    saveState();
    if (frame) cancelAnimationFrame(frame);
    updateTurn(i); renderChatList(); renderCatalog();
  }
}

/* ---------------------------------------------------------------- library */

function setNote(text, isError = false) {
  const note = $("#library-note");
  note.textContent = text;
  note.classList.toggle("error", isError);
}

function docRow(doc, deletable) {
  const kind = doc.kind === "transcript" ? "Session" : doc.kind;
  const row = el("li", { class: "doc" },
    el("span", { class: "kind" + (doc.kind === "transcript" ? " transcript" : ""), text: kind }),
    el("div", { style: "min-width:0" },
      el("div", { class: "name", dir: "auto", title: doc.path, text: doc.title }),
      el("div", { class: "meta", text: `${doc.passages} passages · ${doc.indexed || ""}` })));
  if (deletable) {
    const del = el("button", { class: "icon-btn del", type: "button", "aria-label": `Delete ${doc.title}` }, xIcon());
    confirmButton(del, "Delete", async () => {
      setNote(`Deleting ${doc.title} and re-indexing…`);
      try {
        const r = await api(`/api/documents?path=${encodeURIComponent(doc.path)}`, { method: "DELETE" });
        setNote(`Deleted. Index: ${r.summary}`);
      } catch (e) { setNote(`Could not delete: ${e.message}`, true); }
      loadDocuments(); refreshStatus();
    });
    row.append(del);
  } else {
    row.append(el("span"));
  }
  return row;
}

async function loadDocuments() {
  const groups = $("#doc-groups");
  try {
    const { documents, documents_dir } = await api("/api/documents");
    $("#docs-dir").textContent = documents_dir ? documents_dir.split(/[\\/]/).slice(-2).join("/") : "";
    const files = documents.filter((d) => d.kind !== "transcript");
    const sessions = documents.filter((d) => d.kind === "transcript");
    groups.replaceChildren(
      el("div", { class: "doc-group" }, el("h3", { text: `Documents · ${files.length}` }),
        files.length ? el("ul", { class: "doc-list" }, files.map((d) => docRow(d, true)))
                     : el("p", { class: "muted small", text: "None yet. Add files above." })),
      el("div", { class: "doc-group" }, el("h3", { text: `Session transcripts · ${sessions.length}` }),
        sessions.length ? el("ul", { class: "doc-list" }, sessions.map((d) => docRow(d, false)))
                        : el("p", { class: "muted small", text: "Transcripts in data/transcripts are added on re-index." })));
  } catch (e) {
    groups.replaceChildren(el("p", { class: "muted small", text: `Could not load the library: ${e.message}` }));
  }
}

async function upload(fileList) {
  const files = [...fileList];
  if (!files.length) return;
  const form = new FormData();
  files.forEach((f) => form.append("files", f, f.name));
  setNote(`Uploading ${files.length} file${files.length > 1 ? "s" : ""} and indexing… large PDFs take a minute.`);
  $("#reindex").disabled = true;
  try {
    const r = await api("/api/documents", { method: "POST", body: form });
    const rejected = Object.entries(r.rejected || {}).map(([n, why]) => `${n}: ${why}`);
    const failed = Object.entries(r.failed || {}).map(([n, why]) => `${n.split("/").pop()}: ${why}`);
    const parts = [];
    if (r.saved.length) parts.push(`Added ${r.saved.length}. Index: ${r.summary}.`);
    if (rejected.length) parts.push(`Not added: ${rejected.join("; ")}.`);
    if (failed.length) parts.push(`Could not read: ${failed.join("; ")}.`);
    setNote(parts.join(" "), !r.saved.length || failed.length > 0);
  } catch (e) {
    setNote(`Upload failed: ${e.message}`, true);
  } finally {
    $("#reindex").disabled = false;
    $("#file-input").value = "";
    loadDocuments(); refreshStatus();
  }
}

/* ----------------------------------------------------------------- status */

let statusTimer = null;
async function refreshStatus() {
  clearTimeout(statusTimer);
  let busy = false;
  try {
    const s = await api("/api/status");
    const firstLoad = !state.status;
    state.status = s;
    const dot = $("#status-dot"), main = $("#status-main");
    if (s.warmup_error) { dot.className = "dot bad"; main.textContent = "Models failed to load"; }
    else if (s.ingesting) { dot.className = "dot busy"; main.textContent = "Indexing…"; busy = true; }
    else if (!s.models_loaded) { dot.className = "dot busy"; main.textContent = "Loading models…"; busy = true; }
    else if (!s.passages) { dot.className = "dot"; main.textContent = "Archive is empty"; }
    else { dot.className = "dot ok"; main.textContent = "Ready"; }
    const rows = [
      ["Files", `${s.documents} (${Object.entries(s.kinds).map(([k, v]) => `${v} ${k}`).join(", ") || "none"})`],
      ["Passages", s.passages.toLocaleString()],
      ["Embedder", s.embedding_model.split("/").pop() + (s.index_stale ? " · re-index needed" : "")],
      ["Reranker", s.reranker ? s.reranker.split("/").pop() : "off"],
      ["Router", s.router],
      ["Answers", s.llm],
    ];
    if (s.warmup_error) rows.push(["Error", s.warmup_error]);
    $("#status-grid").replaceChildren(...rows.flatMap(([k, v]) => [el("dt", { text: k }), el("dd", { text: v })]));
    if (firstLoad && !current()?.messages.length) renderThread();
  } catch (e) {
    $("#status-dot").className = "dot bad";
    $("#status-main").textContent = "Server not reachable";
    busy = true;
  }
  statusTimer = setTimeout(refreshStatus, busy ? 3000 : 30000);
}

/* ------------------------------------------------------------------ theme */

const THEMES = ["system", "light", "dark"];
function applyTheme(name) {
  if (name === "system") delete document.documentElement.dataset.theme;
  else document.documentElement.dataset.theme = name;
  try { name === "system" ? localStorage.removeItem("lorekeeper.theme") : localStorage.setItem("lorekeeper.theme", name); } catch (e) {}
  $("#theme").title = `Theme: ${name}`;
}

/* ------------------------------------------------------------------- wire */

function wire() {
  const textarea = $("#question");
  const grow = () => {
    textarea.style.height = "auto";
    const max = innerHeight * 0.4;
    textarea.style.height = Math.min(textarea.scrollHeight + 2, max) + "px";
    textarea.style.overflowY = textarea.scrollHeight > max ? "auto" : "hidden";
  };
  grow();
  textarea.addEventListener("input", grow);
  textarea.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); $("#composer").requestSubmit(); }
  });
  $("#composer").addEventListener("submit", (e) => {
    e.preventDefault();
    if (state.controller) { state.controller.abort(); return; }
    const q = textarea.value;
    if (!q.trim()) { textarea.focus(); return; }
    textarea.value = ""; grow();
    ask(q);
  });

  $("#thread").addEventListener("click", (e) => {
    const cite = e.target.closest(".cite");
    if (!cite) return;
    const turn = cite.closest(".turn");
    flashSource(Number(turn.id.slice(2)), Number(cite.dataset.n));
  });

  $("#new-chat").addEventListener("click", () => {
    if (state.controller) return;
    const conv = current();
    if (!conv || conv.messages.length) newConversation();
    saveState(); renderAll(); showTab("chats"); closePanels(); textarea.focus();
  });
  $("#tab-chats").addEventListener("click", () => showTab("chats"));
  $("#tab-library").addEventListener("click", () => showTab("library"));
  $("#open-rail").addEventListener("click", () => { app.classList.add("rail-open"); $("#scrim").hidden = false; });
  $("#open-catalog").addEventListener("click", () => { app.classList.add("catalog-open"); $("#scrim").hidden = false; });
  $("#close-catalog").addEventListener("click", closePanels);
  $("#scrim").addEventListener("click", closePanels);
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") closePanels(); });

  $("#theme").addEventListener("click", () => {
    const now = document.documentElement.dataset.theme || "system";
    applyTheme(THEMES[(THEMES.indexOf(now) + 1) % THEMES.length]);
  });
  applyTheme(document.documentElement.dataset.theme || "system");

  const input = $("#file-input"), zone = $("#dropzone");
  input.addEventListener("change", () => upload(input.files));
  ["dragenter", "dragover"].forEach((t) => zone.addEventListener(t, (e) => { e.preventDefault(); zone.classList.add("over"); }));
  ["dragleave", "drop"].forEach((t) => zone.addEventListener(t, () => zone.classList.remove("over")));
  zone.addEventListener("drop", (e) => { e.preventDefault(); upload(e.dataTransfer.files); });

  $("#reindex").addEventListener("click", async () => {
    const btn = $("#reindex");
    btn.disabled = true;
    setNote("Indexing new and changed files…");
    refreshStatus();
    try { const r = await api("/api/reindex", { method: "POST" }); setNote(`Index: ${r.summary}`); }
    catch (e) { setNote(`Re-index failed: ${e.message}`, true); }
    finally { btn.disabled = false; loadDocuments(); refreshStatus(); }
  });
}

loadState();
wire();
{
  const conv = current();
  if (conv) { const last = conv.messages.map((m) => m.role).lastIndexOf("assistant"); state.selected = last >= 0 ? last : null; }
}
renderAll();
refreshStatus();
