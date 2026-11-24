"use strict";

/* The party's lore: chat over the campaign's documents and session transcripts.
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

const examples = (party) => [
  { q: "What does the Ritual Adept feature let me do?", tag: "Rules" },
  { q: `Who is in ${party}, and what has each of them done so far?`, tag: "Multi-step" },
  { q: "What happened at the altar in session 19?", tag: "Transcripts" },
  { q: "الـ DM قال ايه عن The Nameless Beyond؟", tag: "Arabic" },
];
const partyName = () => (state.status && state.status.party && state.status.party.name) || "the party";

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
  view: { kind: "chat" },   // chat | character (id) | companions | comics | comic (id); follows location.hash
  comics: null,             // from /api/comics
  party: null,              // /api/party
  partyError: null,
  canEdit: false,     // may upload, delete and re-index (an admin, or no logins at all)
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
  // The latest answer can be asked again: prominently when it failed, quietly when it didn't.
  const last = i === conv.messages.length - 1 && !msg.pending;
  const failed = Boolean(msg.error || msg.stopped);
  if (last && !failed) foot.append(el("button", { type: "button", onclick: () => regenerate(i) }, "Regenerate"));

  return el("section", { class: "turn", id: `m-${i}` },
    el("div", { class: "who" }, "Lore", routeChips(msg)),
    body,
    msg.error ? el("div", { class: "error-box", role: "alert", text: msg.error }) : null,
    last && failed ? el("div", { class: "retry-row" },
      el("button", { class: "btn", type: "button", onclick: () => regenerate(i) }, "Try again")) : null,
    foot.childNodes.length ? foot : null);
}

// Drop the latest question and its answer, then ask it again with the same history.
function regenerate(i) {
  const conv = current();
  if (!conv || state.controller || i !== conv.messages.length - 1) return;
  const question = conv.messages[i - 1] && conv.messages[i - 1].content;
  if (!question) return;
  conv.messages.splice(i - 1);
  ask(question);
}

function renderWelcome() {
  const empty = state.status && state.status.passages === 0;
  const box = el("div", { class: "welcome" },
    state.status && state.status.emblem ? el("img", { class: "crest", src: sizedImage(state.status.emblem, 384), alt: `${partyName()} crest` }) : null,
    el("h2", { text: empty ? `${partyName()}'s lore is empty` : `Ask about ${partyName()}'s lore` }),
    el("p", { text: empty
      ? "Add your lore, campaign notes, character sheets or rulebooks in the Library. Session transcripts in data/transcripts are added when you re-index."
      : "Answers come from your documents and session transcripts, with the passages they cite. The router picks one search for simple questions and several for questions with parts." }));
  if (empty) {
    box.append(el("div", {}, el("button", { class: "btn btn-primary", type: "button", onclick: () => { showTab("library"); openRail(); } }, "Open the Library")));
  } else {
    box.append(el("div", { class: "examples" }, examples(partyName()).map((ex) =>
      el("button", { class: "example", type: "button", onclick: () => ask(ex.q) },
        el("span", { class: "tag", text: ex.tag }),
        el("span", { class: "q", dir: "auto", text: ex.q })))));
  }
  if (mains().length || companions().length) {
    box.append(el("div", { class: "meet" },
      el("span", { class: "meet-label", text: "The party" }),
      el("div", { class: "meet-row" },
        mains().map((c) => el("a", { href: `#${c.id}` }, avatar(c, "sm"), c.name)),
        companions().length ? el("a", { href: "#companions", class: "comp" },
          el("span", { class: "avatar-stack" }, companions().slice(0, 3).map((c) => avatar(c, "xs"))), "Companions") : null,
        state.comics && state.comics.length ? el("a", { href: "#comics", class: "comp" },
          el("img", { class: "avatar xs cover", src: sizedImage(state.comics[0].pages[0], 96), alt: "", loading: "lazy" }), "Comics") : null)));
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
    cards.append(el("li", { class: "cards-empty", text: msg.pending ? "Searching…" : "Nothing in the lore matched." }));
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

// Side panel widths and which are hidden, remembered per browser. Wide screens
// only: below 1180px the sources, and below 760px the left panel, slide over
// the page instead.
const LAYOUT_KEY = "lorekeeper.layout";
const LAYOUT_DEFAULT = { rail: 272, catalog: 380, railHidden: false, catalogHidden: false };
const layout = { ...LAYOUT_DEFAULT, ...storage.get(LAYOUT_KEY, {}) };
const WIDE = "(min-width: 1181px)", NARROW = "(max-width: 760px)";
const DESK_MIN = 360;   // the questions column never gets narrower than this

function clampWidths() {
  const room = innerWidth - DESK_MIN;
  layout.rail = Math.round(Math.min(Math.max(layout.rail, 200), 480, room - (matchMedia(WIDE).matches ? 280 : 0)));
  layout.catalog = Math.round(Math.min(Math.max(layout.catalog, 280), room - (layout.railHidden ? 0 : layout.rail)));
}

function applyLayout(save = true) {
  clampWidths();
  app.style.setProperty("--rail-w", `${layout.rail}px`);
  app.style.setProperty("--catalog-w", `${layout.catalog}px`);
  app.classList.toggle("rail-collapsed", layout.railHidden);
  app.classList.toggle("catalog-collapsed", layout.catalogHidden);
  const widened = layout.catalog >= Math.round((innerWidth - (layout.railHidden ? 0 : layout.rail)) * 0.55) - 1;
  $("#widen-catalog").classList.toggle("on", widened);
  $("#widen-catalog").title = $("#widen-catalog").ariaLabel = widened ? "Narrow this panel" : "Widen this panel";
  if (save) storage.set(LAYOUT_KEY, layout);
}

function makeResizer(handle, side) {
  const widthAt = (x) => (side === "rail" ? x : innerWidth - x);
  handle.addEventListener("pointerdown", (e) => {
    e.preventDefault();
    handle.setPointerCapture(e.pointerId);
    handle.classList.add("active");
    app.classList.add("resizing");
    const move = (ev) => { layout[side] = widthAt(ev.clientX); applyLayout(false); };
    const up = () => {
      handle.removeEventListener("pointermove", move);
      handle.classList.remove("active");
      app.classList.remove("resizing");
      applyLayout();
    };
    handle.addEventListener("pointermove", move);
    handle.addEventListener("pointerup", up, { once: true });
    handle.addEventListener("pointercancel", up, { once: true });
  });
  handle.addEventListener("dblclick", () => { layout[side] = LAYOUT_DEFAULT[side]; applyLayout(); });
  handle.addEventListener("keydown", (e) => {
    const step = e.shiftKey ? 80 : 24;
    const grow = side === "rail" ? "ArrowRight" : "ArrowLeft", shrink = side === "rail" ? "ArrowLeft" : "ArrowRight";
    if (e.key !== grow && e.key !== shrink) return;
    e.preventDefault();
    layout[side] += e.key === grow ? step : -step;
    applyLayout();
  });
}

function closePanels() { app.classList.remove("rail-open", "catalog-open"); $("#scrim").hidden = true; }
function openRail() { if (matchMedia("(max-width: 760px)").matches) { app.classList.add("rail-open"); $("#scrim").hidden = false; } }
function openCatalog() { if (matchMedia("(max-width: 1180px)").matches) { app.classList.add("catalog-open"); $("#scrim").hidden = false; } }

function showTab(name) {
  for (const tab of ["chats", "party", "library"]) {
    $(`#tab-${tab}`).setAttribute("aria-selected", tab === name);
    $(`#panel-${tab}`).hidden = tab !== name;
  }
  if (name === "library") loadDocuments();
  if (name === "party") loadParty();
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
  saveState(); navigate(""); closePanels();
  $("#thread").scrollTop = $("#thread").scrollHeight;
}

/* -------------------------------------------------------------------- api */

async function api(path, options = {}) {
  const res = await fetch(path, options);
  if (res.status === 401) { location.assign("/login"); throw new Error("signed out"); }
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
  if (res.status === 401) { location.assign("/login"); throw new Error("signed out"); }
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
      else if (ev.type === "restart") msg.content = "";
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
        files.length ? el("ul", { class: "doc-list" }, files.map((d) => docRow(d, state.canEdit)))
                     : el("p", { class: "muted small", text: state.canEdit ? "None yet. Add files above." : "None yet." })),
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

/* ------------------------------------------------------------------ party */

const SHEET_PREVIEW = 4;      // sheet pages shown before "Show all"
const MENTION_PREVIEW = 6;    // mentions per session shown before "Show all"
const plural = (n, word) => `${n.toLocaleString()} ${word}${n === 1 ? "" : "s"}`;
const mains = () => (state.party ? state.party.characters.filter((c) => c.role === "main") : []);
const companions = () => (state.party ? state.party.characters.filter((c) => c.role === "companion") : []);

// The server shrinks images to the width asked for (96, 192, 384 or 800 px), so a
// 34px circle doesn't download a 600 KB painting. Twice the shown size stays sharp
// on high-density screens.
function sizedImage(url, width) {
  return `${url}${url.includes("?") ? "&" : "?"}w=${width}`;
}
const AVATAR_WIDTH = { xs: 96, sm: 96, "": 96, lg: 192, xl: 384 };

function avatar(c, size = "") {
  return c.portrait
    ? el("img", { class: `avatar ${size}`, src: sizedImage(c.portrait, AVATAR_WIDTH[size] || 192), alt: "", loading: "lazy" })
    : el("span", { class: `avatar mono ${size}`, "aria-hidden": "true", text: (c.name.trim()[0] || "?").toUpperCase() });
}

// The roster and comics hardly change, so one request serves every view for a
// minute, and calls made while it's on its way share it. Through the tunnel each
// request is a round trip worth saving.
const FRESH_MS = 60_000;
function once(load) {
  let pending = null, at = 0;
  return () => {
    if (!pending || Date.now() - at > FRESH_MS) { at = Date.now(); pending = load(); }
    return pending;
  };
}

const loadParty = once(async () => {
  try {
    state.party = await api("/api/party");
    state.partyError = null;
  } catch (e) {
    state.party = { characters: [] };
    state.partyError = e.message;
  }
  renderPartyList();
  return state.party;
});

function renderPartyList() {
  const view = state.view;
  const row = (c, extra = "") => el("li", {},
    el("a", { class: `party-row ${extra}` + (view.kind === "character" && view.id === c.id ? " active" : ""), href: `#${c.id}` },
      avatar(c, extra ? "xs" : "sm"),
      el("span", { class: "who-text" },
        el("span", { class: "n", text: c.name }),
        extra ? null : el("span", { class: "s", text: c.title || (c.player ? `played by ${c.player}` : "") }))));
  const items = [];
  if (state.partyError) items.push(el("li", { class: "chat-empty error", text: `config/party.toml: ${state.partyError}` }));
  items.push(...mains().map((c) => row(c)));
  items.push(el("li", { class: "party-sep" },
    el("a", { class: "party-row" + (view.kind === "companions" ? " active" : ""), href: "#companions" },
      el("span", { class: "who-text" }, el("span", { class: "n", text: "Companions" }),
        el("span", { class: "s", text: companions().length ? plural(companions().length, "companion") : "none yet" })))));
  items.push(...companions().map((c) => row(c, "sub")));
  const comics = state.comics || [];
  items.push(el("li", { class: "party-sep" },
    el("a", { class: "party-row" + (view.kind === "comics" || view.kind === "comic" ? " active" : ""), href: "#comics" },
      el("span", { class: "who-text" }, el("span", { class: "n", text: "Comics" }),
        el("span", { class: "s", text: comics.length ? plural(comics.length, "comic") : "none yet" })))));
  $("#party-list").replaceChildren(...items);
}

// Sheets come from PDFs as printed lines; join them back into paragraphs,
// lists and headings so they read like the document rather than a dump.
function reflow(text) {
  const lines = text.split("\n").map((l) => l.trim());
  const widths = lines.filter(Boolean).map((l) => l.length).sort((a, b) => a - b);
  const full = widths.length ? widths[Math.floor(widths.length * 0.8)] : 80;
  const blocks = [];
  let open = null;
  const close = () => { if (open) blocks.push(open); open = null; };
  for (const line of lines) {
    if (!line) { close(); continue; }
    const bullet = line.match(/^[-•]\s+(.*)$/);
    const short = line.length < full * 0.72;
    const ends = /[.!?:;…"”’)\]]$/.test(line);
    if (bullet) { close(); open = { type: "li", text: bullet[1] }; }
    else if (open) {
      open.text = /[a-z]-$/.test(open.text) ? open.text.slice(0, -1) + line : `${open.text} ${line}`;
    } else if (short && !ends && line.length < 64 && /^[\p{Lu}\p{N}"“]/u.test(line)) {
      blocks.push({ type: "h", text: line }); continue;
    } else open = { type: "p", text: line };
    if (short) close();
  }
  close();
  return blocks;
}

function renderBlocks(blocks) {
  const out = [];
  let list = null;
  for (const b of blocks) {
    if (b.type === "li") { if (!list) out.push(list = el("ul")); list.append(el("li", { dir: "auto", text: b.text })); continue; }
    list = null;
    out.push(b.type === "h" ? el("h4", { dir: "auto", text: b.text }) : el("p", { dir: "auto", text: b.text }));
  }
  return out;
}

function askFresh(question) {
  if (state.controller) return;
  newConversation();
  navigate("");
  showTab("chats");
  ask(question);
}

function characterQuestions(c) {
  const n = c.name;
  return c.role === "companion"
    ? [`Who is ${n}, and how did they come to travel with ${partyName()}?`, `What has ${n} done in the sessions?`,
       `What do the others say about ${n}?`]
    : [`Who is ${n}?`, `What has ${n} done in the sessions so far?`, `What do the others say about ${n}?`];
}

function renderSheet(c) {
  const section = el("section", { class: "sheet" });
  const label = el("h3", { class: "section-label" }, "Sheet");
  section.append(label);
  if (!c.sheet) {
    section.append(el("p", { class: "muted note", text: c.role === "main"
      ? `No sheet yet. Put ${c.name}'s PDF in data/documents/characters, then set sheet = "data/documents/characters/${c.name}.pdf" for them in config/party.toml.`
      : `No sheet yet. Notes about ${c.name} can go in data/documents/lore; set sheet in config/party.toml to show one here.` }));
    return section;
  }
  if (c.sheet.error) { section.append(el("div", { class: "error-box", text: `${c.sheet.path}: ${c.sheet.error}` })); return section; }
  const pages = c.sheet.sections;
  label.append(el("span", { class: "label-meta", text: `${c.sheet.path.split("/").pop()} · ${plural(pages.length, "section")}` }));
  const nodes = pages.map((p) => el("article", { class: "sheet-page" },
    el("div", { class: "page-no", text: p.location || "" }),
    el("div", { class: "sheet-text" }, p.heading ? el("h4", { dir: "auto", text: p.heading }) : null, renderBlocks(reflow(p.text)))));
  const body = el("div", { class: "sheet-body" }, nodes.slice(0, SHEET_PREVIEW));
  section.append(body);
  if (nodes.length > SHEET_PREVIEW) {
    const more = el("button", { class: "btn sheet-more", type: "button", text: `Show all ${pages.length}`,
      onclick: () => { body.append(...nodes.slice(SHEET_PREVIEW)); more.remove(); } });
    section.append(more);
  }
  return section;
}

function renderAppearances(c) {
  const box = $("#appearances");
  if (!c) { box.replaceChildren(); return; }
  const parts = [];
  if (!c.appearances.length) {
    parts.push(el("p", { class: "cards-empty", text: `${c.name} doesn't come up in the transcripts yet. If the transcripts spell the name another way, add that spelling to aliases in config/party.toml.` }));
  }
  for (const s of c.appearances) {
    const stats = [];
    if (s.spoken_lines) stats.push(`${plural(s.spoken_lines, "line")} spoken · ${s.spoken_minutes} min`);
    stats.push(s.mention_count ? `named ${plural(s.mention_count, "time")} by others` : "not named by others");
    const items = s.mentions.map((m) => el("li", { class: "mention" },
      el("span", { class: "clock", text: m.clock }),
      el("div", {}, el("span", { class: "speaker", text: m.speaker }), el("span", { class: "line", dir: "auto", text: m.text }))));
    const list = el("ol", { class: "mentions" }, items.slice(0, MENTION_PREVIEW));
    const group = el("section", { class: "app-group" },
      el("h3", { text: s.title }), el("div", { class: "app-stats", text: stats.join(" · ") }), items.length ? list : null);
    if (items.length > MENTION_PREVIEW) {
      const more = el("button", { class: "link-btn", type: "button",
        text: s.mention_count > items.length ? `Show the first ${items.length} of ${s.mention_count}` : `Show all ${items.length}`,
        onclick: () => { list.replaceChildren(...items); more.remove(); } });
      group.append(more);
    }
    parts.push(group);
  }
  if (c.documents.length) {
    parts.push(el("section", { class: "app-group" }, el("h3", { text: "In your documents" }),
      el("ol", { class: "mentions" }, c.documents.map((d) => el("li", { class: "doc-mention" },
        el("div", { class: "speaker" }, d.title, el("span", { class: "muted", text: ` · ${plural(d.count, "passage")}` })),
        d.snippets.map((sn) => el("div", { class: "line", dir: "auto" },
          sn.where ? el("span", { class: "clock", text: `${sn.where} ` }) : null, sn.text)))))));
  }
  box.replaceChildren(...parts);
}

async function renderProfile(id) {
  const box = $("#profile");
  box.replaceChildren(el("div", { class: "profile-inner" }, el("p", { class: "muted", text: "Opening the page…" })));
  renderAppearances(null);
  let c;
  try { c = await api(`/api/party/${encodeURIComponent(id)}`); }
  catch (e) {
    box.replaceChildren(el("div", { class: "profile-inner" }, el("div", { class: "error-box", role: "alert", text: e.message })));
    return;
  }
  if (state.view.kind !== "character" || state.view.id !== id) return;   // navigated away meanwhile
  $("#chat-title").textContent = c.name;
  document.title = `${c.name} · ${partyName()} Lore`;

  const facts = [c.role === "companion" ? `Companion of ${partyName()}` : c.player ? `Played by ${c.player}` : "Player character"];
  if (c.sessions) facts.push(`In ${plural(c.sessions, "session")}`);
  if (c.spoken_lines) facts.push(`${plural(c.spoken_lines, "line")} spoken`);
  if (c.mentions) facts.push(`Named ${plural(c.mentions, "time")} by others`);

  box.replaceChildren(el("div", { class: "profile-inner" },
    el("header", { class: "hero" + (c.portrait ? " has-portrait" : "") },
      avatar(c, "xl"),
      el("div", { class: "hero-text" },
        el("div", { class: "eyebrow", text: c.role === "companion" ? "Companion" : partyName() }),
        el("h2", { text: c.name }),
        c.full_name && c.full_name !== c.name ? el("div", { class: "full", text: c.full_name }) : null,
        c.title ? el("div", { class: "title-line", text: c.title }) : null,
        el("div", { class: "facts" }, facts.map((f) => el("span", { class: "chip", text: f }))),
        // Each spelling isolated, so a mix of Arabic and Latin names keeps its order.
        c.aliases.length ? el("div", { class: "aliases" }, "Also written ",
          ...c.aliases.flatMap((a, i) => [i ? " · " : "", el("bdi", { text: a })])) : null)),
    c.summary ? el("p", { class: "summary", dir: "auto", text: c.summary }) : null,
    el("div", { class: "ask-row" }, el("span", { class: "meet-label", text: "Ask" }),
      characterQuestions(c).map((q) => el("button", { class: "ask-chip", type: "button", onclick: () => askFresh(q) }, q))),
    renderSheet(c)));
  renderAppearances(c);
}

function renderCompanions() {
  $("#chat-title").textContent = "Companions";
  document.title = `Companions · ${partyName()} Lore`;
  const comps = companions();
  const grid = el("div", { class: "companion-grid" }, comps.map((c) => el("a", { class: "companion-card", href: `#${c.id}` },
    avatar(c, "lg"),
    el("div", { class: "cc-body" },
      el("h3", { text: c.name }),
      c.title ? el("div", { class: "cc-title", text: c.title }) : null,
      c.summary ? el("p", { dir: "auto", text: c.summary }) : null,
      el("div", { class: "cc-meta", text: c.mentions
        ? `Named ${plural(c.mentions, "time")} in ${plural(c.sessions, "session")}` : "Not in the transcripts yet" })))));
  $("#profile").replaceChildren(el("div", { class: "profile-inner" },
    el("header", { class: "page-head" },
      el("div", { class: "eyebrow", text: partyName() }),
      el("h2", { text: "Companions" }),
      el("p", { class: "muted", text: "Those who travel with the party. Each has a page with their notes and every time the sessions mention them." })),
    comps.length ? grid : el("p", { class: "muted note", text: 'No companions yet. Add them to config/party.toml with role = "companion".' })));
  $("#appearances").replaceChildren(el("p", { class: "cards-empty", text: "Open a companion to see where they come up in the sessions." }));
}

/* ----------------------------------------------------------------- comics */

const loadComics = once(async () => {
  try { state.comics = (await api("/api/comics")).comics; state.comicsError = null; }
  catch (e) { state.comics = []; state.comicsError = e.message; }
  renderPartyList();
  return state.comics;
});

// The comic's characters: their portrait and a link to their page when the roster knows them.
function castChips(comic, linked = true) {
  return comic.characters.map((who) => {
    const c = who.id && state.party ? state.party.characters.find((ch) => ch.id === who.id) : null;
    return c ? el(linked ? "a" : "span", { class: "cast", href: linked ? `#${c.id}` : null }, avatar(c, "xs"), c.name)
             : el("span", { class: "cast", text: who.name });
  });
}

function renderComics() {
  $("#chat-title").textContent = "Comics";
  document.title = `Comics · ${partyName()} Lore`;
  const comics = state.comics || [];
  const grid = el("div", { class: "comic-grid" }, comics.map((m) => el("a", { class: "comic-card", href: `#comics/${m.id}` },
    el("img", { class: "comic-cover", src: sizedImage(m.pages[0], 800), alt: "", loading: "lazy" }),
    el("div", { class: "cc-body" },
      el("h3", { text: m.title }),
      el("div", { class: "cc-meta", text: [plural(m.pages.length, "page"), m.date].filter(Boolean).join(" · ") }),
      m.summary ? el("p", { dir: "auto", text: m.summary }) : null,
      m.characters.length ? el("div", { class: "cast-row" }, castChips(m, false)) : null))));
  $("#profile").replaceChildren(el("div", { class: "profile-inner" },
    el("header", { class: "page-head" },
      el("div", { class: "eyebrow", text: partyName() }),
      el("h2", { text: "Comics" }),
      el("p", { class: "muted", text: "The party's adventures, drawn." })),
    state.comicsError ? el("p", { class: "note error", text: state.comicsError }) : null,
    comics.length ? grid : el("p", { class: "muted note",
      text: "No comics yet. Put each comic's pages in a folder of its own, assets/comics/<title>/, named 01, 02, 03 … so they read in order." })));
  $("#appearances").replaceChildren(el("p", { class: "cards-empty", text: "Open a comic to see its pages here." }));
}

function renderComic(id) {
  const comics = state.comics || [];
  const i = comics.findIndex((m) => m.id === id);
  if (i < 0) {
    $("#chat-title").textContent = "Comics";
    $("#profile").replaceChildren(el("div", { class: "profile-inner" },
      el("p", { class: "muted note", text: "That comic isn't in assets/comics any more." }),
      el("a", { href: "#comics", text: "All comics" })));
    $("#appearances").replaceChildren();
    return;
  }
  const m = comics[i];
  $("#chat-title").textContent = m.title;
  document.title = `${m.title} · ${partyName()} Lore`;
  const pages = m.pages.map((src, n) => el("figure", { class: "comic-page", id: `page-${n + 1}` },
    el("a", { href: src, target: "_blank", rel: "noopener", title: "Open full size" },
      el("img", { src, alt: `${m.title}, page ${n + 1}`, loading: n < 2 ? "eager" : "lazy" }))));
  const step = (j, label) => (comics[j] ? el("a", { class: "btn", href: `#comics/${comics[j].id}` }, `${label} ${comics[j].title}`) : el("span"));
  $("#profile").replaceChildren(el("div", { class: "profile-inner comic-reader" },
    el("header", { class: "page-head" },
      el("a", { class: "eyebrow", href: "#comics", text: "Comics" }),
      el("h2", { text: m.title }),
      m.summary ? el("p", { class: "summary", dir: "auto", text: m.summary }) : null,
      m.characters.length ? el("div", { class: "cast-row" }, castChips(m)) : null),
    el("div", { class: "comic-pages" }, pages),
    el("nav", { class: "comic-nav" }, step(i - 1, "←"), step(i + 1, "→"))));
  $("#appearances").replaceChildren(
    el("div", { class: "app-group" },
      el("h3", { text: plural(m.pages.length, "page") }),
      el("div", { class: "comic-thumbs" }, m.pages.map((src, n) => el("button", {
        class: "comic-thumb", type: "button", "aria-label": `Page ${n + 1}`,
        onclick: () => { closePanels(); $(`#page-${n + 1}`).scrollIntoView({ behavior: "smooth", block: "start" }); } },
        el("img", { src: sizedImage(src, 192), alt: "", loading: "lazy" }), el("span", { text: n + 1 }))))),
    m.characters.length ? el("div", { class: "app-group" }, el("h3", { text: "In this comic" }),
      el("div", { class: "cast-col" }, castChips(m))) : null);
}

/* ---------------------------------------------------------------- routing */

// The address bar holds the page: "" is the chat, #companions, #comics, #comics/<id>, or #<character>.
function applyView() {
  const chat = state.view.kind === "chat";
  $("#thread").hidden = !chat; $("#composer").hidden = !chat; $("#profile").hidden = chat;
  $("#trace").hidden = !chat; $("#cards").hidden = !chat; $("#appearances").hidden = chat;
  const comic = state.view.kind === "comics" || state.view.kind === "comic";
  $("#catalog-title").textContent = chat ? "Sources" : comic ? "Pages" : "In the sessions";
  $("#open-catalog").setAttribute("aria-label", chat ? "Show sources" : "Show where they come up");
  if (chat) document.title = `${partyName() === "the party" ? "" : partyName() + " "}Lore`.trim();
}

async function route() {
  const hash = decodeURIComponent(location.hash.slice(1));
  state.view = !hash ? { kind: "chat" }
    : hash === "companions" ? { kind: "companions" }
    : hash === "comics" ? { kind: "comics" }
    : hash.startsWith("comics/") ? { kind: "comic", id: hash.slice(7) }
    : { kind: "character", id: hash };
  applyView();
  closePanels();
  if (state.view.kind === "chat") { renderAll(); renderPartyList(); return; }
  showTab("party");
  $("#profile").scrollTop = 0;
  await loadParty();
  if (state.view.kind === "comics" || state.view.kind === "comic") {
    await loadComics();
    if (state.view.kind === "comics") renderComics(); else renderComic(state.view.id);
    return;
  }
  if (state.view.kind === "companions") renderCompanions();
  else if (state.view.kind === "character") await renderProfile(state.view.id);
}

function navigate(hash) {
  if (hash) { if (location.hash === `#${hash}`) route(); else location.hash = hash; return; }
  if (location.hash) history.pushState(null, "", location.pathname + location.search);
  route();
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
    const canEdit = !s.login || Boolean(s.user && s.user.admin);
    if (canEdit !== state.canEdit) {
      state.canEdit = canEdit;
      document.body.classList.toggle("read-only", !canEdit);
      if (!$("#panel-library").hidden) loadDocuments();
    }
    $("#account").hidden = !s.user;
    if (s.user) $("#account-name").textContent = `Signed in as ${s.user.name}${s.user.admin ? " · admin" : ""}`;
    const dot = $("#status-dot"), main = $("#status-main");
    if (s.warmup_error) { dot.className = "dot bad"; main.textContent = "Models failed to load"; }
    else if (s.ingesting) { dot.className = "dot busy"; main.textContent = "Indexing…"; busy = true; }
    else if (!s.models_loaded) { dot.className = "dot busy"; main.textContent = "Loading models…"; busy = true; }
    else if (!s.passages) { dot.className = "dot"; main.textContent = "No lore indexed yet"; }
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
    if (firstLoad && state.view.kind === "chat") { applyView(); if (!current()?.messages.length) renderThread(); }
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
    saveState(); navigate(""); showTab("chats"); closePanels(); textarea.focus();
  });
  $("#tab-chats").addEventListener("click", () => showTab("chats"));
  $("#tab-party").addEventListener("click", () => showTab("party"));
  $("#tab-library").addEventListener("click", () => showTab("library"));
  // On wide screens these hide and show the panels; on small ones they slide them over the page.
  $("#open-rail").addEventListener("click", () => {
    if (matchMedia(NARROW).matches) { app.classList.add("rail-open"); $("#scrim").hidden = false; }
    else { layout.railHidden = !layout.railHidden; applyLayout(); }
  });
  $("#open-catalog").addEventListener("click", () => {
    if (!matchMedia(WIDE).matches) { app.classList.add("catalog-open"); $("#scrim").hidden = false; }
    else { layout.catalogHidden = !layout.catalogHidden; applyLayout(); }
  });
  $("#close-catalog").addEventListener("click", () => {
    if (matchMedia(WIDE).matches) { layout.catalogHidden = true; applyLayout(); } else closePanels();
  });
  $("#widen-catalog").addEventListener("click", () => {
    const wide = Math.round((innerWidth - (layout.railHidden ? 0 : layout.rail)) * 0.55);
    layout.catalog = $("#widen-catalog").classList.contains("on") ? LAYOUT_DEFAULT.catalog : wide;
    applyLayout();
  });
  makeResizer($("#rail-resizer"), "rail");
  makeResizer($("#catalog-resizer"), "catalog");
  addEventListener("resize", () => applyLayout(false));
  applyLayout(false);
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
window.addEventListener("hashchange", route);
route();
refreshStatus();
Promise.all([loadParty(), loadComics()]).then(() => { if (state.view.kind === "chat" && !current()?.messages.length) renderThread(); });
