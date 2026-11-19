/* WordFlow dashboard.
 *
 * A dependency-free single-page front end. It never computes a probability itself:
 * every number shown comes from the Python engine through the JSON API in
 * wordflow/server.py, and every response carries the model version so the page can
 * tell when the model behind it has been retrained.
 */
"use strict";

// ------------------------------------------------------------------ utilities

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));

function esc(value) {
  return String(value ?? "").replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}

function fmtP(p) {
  if (p === null || p === undefined) return "—";
  if (p === 0) return "0";
  if (p >= 0.001) return p.toFixed(4);
  return p.toExponential(2);
}
const fmtN = (n) => (n ?? 0).toLocaleString("en-US");
const fmtF = (x, digits = 3) => (x === null || x === undefined ? "∞" : Number(x).toFixed(digits));
const fmtPct = (x) => `${(100 * (x || 0)).toFixed(1)}%`;
const SPECIAL = new Set(["<s>", "</s>", "<unk>"]);

function tok(token, extra = "") {
  const cls = token === "<unk>" ? "unk" : SPECIAL.has(token) ? "special" : "";
  return `<span class="tok ${cls} ${extra}">${esc(token)}</span>`;
}
function toks(tokens, empty = "(no context)") {
  if (!tokens || !tokens.length) return `<span class="muted small">${esc(empty)}</span>`;
  return tokens.map((t) => tok(t)).join("");
}

const ORDER_NAMES = { 1: "unigram", 2: "bigram", 3: "trigram" };
const orderName = (n) => ORDER_NAMES[n] || `${n}-gram`;
const cap = (s) => s.charAt(0).toUpperCase() + s.slice(1);

function debounce(fn, ms = 250) {
  let timer = null;
  return (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => fn(...args), ms);
  };
}

/** Wrap an async query so only the newest call's result is rendered. */
function latestOnly(fn) {
  let ticket = 0;
  return async (...args) => {
    const mine = ++ticket;
    const result = await fn(...args);
    return mine === ticket ? result : undefined;
  };
}

// ------------------------------------------------------------------ API layer

const store = { state: null };
let inflight = 0;

function setBusy(delta) {
  inflight = Math.max(0, inflight + delta);
  const bar = $("#progress");
  if (inflight > 0) {
    bar.classList.remove("done");
    bar.classList.add("on");
  } else {
    bar.classList.remove("on");
    bar.classList.add("done");
  }
}

async function api(method, route, payload) {
  let url = `/api/${route}`;
  const options = { method, headers: {} };
  if (payload && method === "GET") {
    url += `?${new URLSearchParams(payload)}`;
  } else if (payload) {
    options.headers["Content-Type"] = "application/json";
    options.body = JSON.stringify(payload);
  }
  setBusy(1);
  try {
    let response;
    try {
      response = await fetch(url, options);
    } catch {
      throw new Error("Cannot reach the WordFlow server. Is `python app.py` still running?");
    }
    let data;
    try {
      data = await response.json();
    } catch {
      data = { error: `Unexpected response (HTTP ${response.status})` };
    }
    if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
    if (data.version && store.state && data.version !== store.state.version && route !== "state") {
      // The model was retrained elsewhere (another tab or the CLI): resync everything.
      queueMicrotask(() => refreshState(true));
    }
    return data;
  } finally {
    setBusy(-1);
  }
}

let toastTimer = null;
function toast(message, ok = false) {
  const el = $("#toast");
  el.textContent = message;
  el.classList.toggle("ok", ok);
  el.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { el.hidden = true; }, ok ? 3500 : 6000);
}

function errorBox(error) {
  return `<div class="error-box">${esc(error.message || error)}</div>`;
}

async function refreshState(rerender = false) {
  const previous = store.state && store.state.version;
  store.state = await api("GET", "state");
  renderModelCard();
  if (rerender && previous !== store.state.version) {
    clampOrders();
    route();
  }
}

// ------------------------------------------------------------------ shared UI state

const maxOrder = () => store.state.model.max_order;
const defaultOrder = () => Math.min(3, maxOrder());

const ui = {
  predict: { text: "I love machine", order: 3, top_k: 5, smoothing: true, k: 1,
             max_words: 20, temperature: 1, seed: 7 },
  explorer: { order: 2, q: "", min_count: 1, include_special: true, limit: 50, ctx: "machine" },
  prob: { context: "the model", order: 2, k: 1, limit: 15 },
  gen: { seed_text: "The model", order: 3, max_words: 30, temperature: 1, seed: 42, k: 0, stop_at_eos: true },
  eval: { k: 1 },
  math: null,
  train: null,
};

function clampOrders() {
  const top = maxOrder();
  for (const key of ["predict", "explorer", "prob", "gen"]) {
    ui[key].order = Math.max(1, Math.min(ui[key].order, top));
  }
  ui.train = null;            // re-read the active training configuration
}

function orderSeg(name, current, min = 1) {
  let html = `<div class="seg" data-seg="${name}" role="group" aria-label="n-gram order">`;
  for (let n = 1; n <= maxOrder(); n += 1) {
    html += `<button type="button" data-value="${n}" class="${n === current ? "on" : ""}" ${n < min ? "disabled" : ""}>${cap(orderName(n))}</button>`;
  }
  return `${html}</div>`;
}

function bindSeg(root, name, onChange) {
  const seg = $(`[data-seg="${name}"]`, root);
  if (!seg) return;
  seg.addEventListener("click", (event) => {
    const button = event.target.closest("button[data-value]");
    if (!button || button.disabled) return;
    $$("button", seg).forEach((b) => b.classList.toggle("on", b === button));
    onChange(Number(button.dataset.value));
  });
}

/** Horizontal bar chart. rows: [{label, value, display, note, cls, inf}] */
function hbars(rows, { max = null, labelHtml = false } = {}) {
  if (!rows.length) return `<div class="empty">Nothing to show.</div>`;
  const top = max ?? Math.max(...rows.filter((r) => !r.inf).map((r) => r.value), 1e-12);
  return `<div class="bars">${rows.map((r) => {
    const width = r.inf ? 100 : Math.max(0, Math.min(100, (100 * r.value) / top));
    return `<div class="bar-row" title="${esc(r.title || "")}">
      <div class="bar-label">${labelHtml ? r.label : esc(r.label)}</div>
      <div class="bar-track"><div class="bar-fill ${r.inf ? "inf" : ""} ${r.cls || ""}" style="width:${width}%"></div></div>
      <div class="bar-value">${esc(r.display ?? fmtP(r.value))}${r.note ? ` <span class="muted">${esc(r.note)}</span>` : ""}</div>
    </div>`;
  }).join("")}</div>`;
}

function tile(label, value, hint = "", small = false) {
  return `<div class="tile"><div class="label">${esc(label)}</div>
    <div class="value ${small ? "sm" : ""}">${esc(value)}</div>
    ${hint ? `<div class="hint">${hint}</div>` : ""}</div>`;
}

function kvTable(pairs) {
  return `<div class="table-wrap"><table><tbody>${pairs.map(([k, v]) =>
    `<tr><td class="muted" style="width:40%">${esc(k)}</td><td>${v}</td></tr>`).join("")}</tbody></table></div>`;
}

/** Line chart in SVG. series: [{name, color, points: [{x, y}]}], x on a log10 scale. */
function lineChart(series, { xLabel = "", yLabel = "", logX = true } = {}) {
  const W = 680, H = 290, L = 64, R = 18, T = 14, B = 46;
  const pts = series.flatMap((s) => s.points).filter((p) => Number.isFinite(p.y));
  if (!pts.length) return `<div class="empty">No finite values to plot.</div>`;
  const tx = (x) => (logX ? Math.log10(x) : x);
  const xs = pts.map((p) => tx(p.x));
  const x0 = Math.min(...xs), x1 = Math.max(...xs);
  let y0 = Math.min(...pts.map((p) => p.y)), y1 = Math.max(...pts.map((p) => p.y));
  const span = y1 - y0 || Math.abs(y1) || 1;
  const step = niceStep(span / 5);
  y0 = Math.floor((y0 - span * 0.05) / step) * step;
  y1 = Math.ceil((y1 + span * 0.05) / step) * step;
  const sx = (x) => L + ((tx(x) - x0) / (x1 - x0 || 1)) * (W - L - R);
  const sy = (y) => T + (1 - (y - y0) / (y1 - y0 || 1)) * (H - T - B);
  let svg = `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(yLabel)} versus ${esc(xLabel)}">`;
  for (let y = y0; y <= y1 + 1e-9; y += step) {
    svg += `<line class="grid-line" x1="${L}" x2="${W - R}" y1="${sy(y)}" y2="${sy(y)}"/>`;
    svg += `<text x="${L - 8}" y="${sy(y) + 4}" text-anchor="end">${fmtTick(y)}</text>`;
  }
  const xTicks = [...new Set(pts.map((p) => p.x))].sort((a, b) => a - b);
  for (const x of xTicks) {
    svg += `<text x="${sx(x)}" y="${H - B + 18}" text-anchor="middle">${x}</text>`;
  }
  svg += `<line class="axis" x1="${L}" x2="${W - R}" y1="${H - B}" y2="${H - B}"/>`;
  svg += `<text x="${(L + W - R) / 2}" y="${H - 6}" text-anchor="middle">${esc(xLabel)}</text>`;
  svg += `<text transform="translate(14 ${(T + H - B) / 2}) rotate(-90)" text-anchor="middle">${esc(yLabel)}</text>`;
  for (const s of series) {
    const valid = s.points.filter((p) => Number.isFinite(p.y));
    if (!valid.length) continue;
    const d = valid.map((p, i) => `${i ? "L" : "M"}${sx(p.x).toFixed(1)},${sy(p.y).toFixed(1)}`).join(" ");
    svg += `<path d="${d}" fill="none" stroke="${s.color}" stroke-width="2.2" stroke-linejoin="round"/>`;
    const best = valid.reduce((a, b) => (b.y < a.y ? b : a));
    for (const p of valid) {
      const isBest = p === best;
      svg += `<circle cx="${sx(p.x)}" cy="${sy(p.y)}" r="${isBest ? 5.5 : 3.5}" fill="${isBest ? s.color : "var(--surface)"}" stroke="${s.color}" stroke-width="2"><title>${esc(s.name)}: k = ${p.x}, perplexity = ${p.y.toFixed(2)}</title></circle>`;
    }
  }
  return `${svg}</svg>`;
}
function niceStep(raw) {
  const pow = 10 ** Math.floor(Math.log10(raw));
  const f = raw / pow;
  return (f < 1.5 ? 1 : f < 3 ? 2 : f < 7 ? 5 : 10) * pow;
}
const fmtTick = (v) => (Math.abs(v) >= 1000 ? `${(v / 1000).toFixed(1)}k` : Number(v.toFixed(2)).toString());

const SERIES = ["var(--accent)", "var(--series-2)", "var(--series-3)", "var(--good)", "var(--bad)"];

// ------------------------------------------------------------------ sidebar

function renderModelCard() {
  const s = store.state;
  if (!s) return;
  const c = s.corpus, m = s.model;
  $("#model-card").innerHTML = `
    <div class="mc-title"><span>Active model</span><span class="badge accent">v${s.version}</span></div>
    <div class="mc-row"><span>Corpus</span><b>${esc(c.name)}</b></div>
    <div class="mc-row"><span>Source</span><b>${esc(c.source)}</b></div>
    <div class="mc-row"><span>Orders</span><b>1 – ${m.max_order}</b></div>
    <div class="mc-row"><span>Vocabulary V</span><b>${fmtN(m.vocab_size)}</b></div>
    <div class="mc-row"><span>Train / test</span><b>${fmtN(c.train_sentences)} / ${fmtN(c.test_sentences)}</b></div>`;
}

// ------------------------------------------------------------------ page 1: overview

function renderOverview(el) {
  const s = store.state, c = s.corpus, m = s.model;
  const flow = [
    ["Corpus", "Raw text: the sample or your .txt upload"],
    ["Preprocess", "Lower-case, clean, split into sentences"],
    ["Tokenize", "Words only; <code>&lt;unk&gt;</code> for unknowns"],
    ["Count", "n-gram tables with <code>Counter</code>"],
    ["Estimate", "MLE: C(context, w) / C(context)"],
    ["Smooth", "Add-k pseudo-counts for unseen pairs"],
    ["Predict / Generate", "Rank or sample the next word"],
    ["Evaluate", "Cross-entropy and perplexity on held-out text"],
  ];
  el.innerHTML = `<div class="stack">
    <section class="hero">
      <h2>Next-word prediction from counting, not neural networks</h2>
      <p class="lead">WordFlow is a statistical language model built from first principles in plain Python.
      It counts unigrams, bigrams and trigrams in a training corpus, turns the counts into conditional
      probabilities, and uses them to suggest the next word, generate text and measure perplexity on
      sentences it has never seen. Every number in this dashboard is computed live from the model
      that is currently loaded.</p>
      <div class="row"><a class="btn" href="#predict">Open the Prediction Studio</a>
      <a class="btn ghost" href="#math">Learn the mathematics</a></div>
    </section>

    <div class="quick">
      ${[
        ["predict", "I love machine", "Autocomplete a sentence", "Type, then press Tab to accept the top word. Keep pressing Tab to let the model write."],
        ["predict", "Quantum flux capacitor", "Watch the fallback", "Unknown words become &lt;unk&gt; and the model backs off to word frequencies."],
        ["generate", "The model", "Generate text", "Sample a sentence step by step and inspect every random choice."],
        ["evaluate", "", "Measure perplexity", "See why unsmoothed models fail on unseen text and which k works best."],
      ].map(([page, text, title, desc], i) => `<button class="quick-card" data-go="${page}" data-text="${esc(text)}">
        <span class="qn">${i + 1}</span><span class="qt">${title}</span><span class="qd">${desc}</span></button>`).join("")}
    </div>
    <p class="small muted" style="margin-top:-6px">Shortcuts: press <kbd>1</kbd>–<kbd>7</kbd> to switch pages, <kbd>/</kbd> to jump to the main input.</p>

    <div class="tiles">
      ${tile("Training corpus", c.name, `${esc(c.source)} · ${esc(c.line_mode)} mode`, true)}
      ${tile("Sentences", fmtN(c.sentences), `${fmtN(c.train_sentences)} train · ${fmtN(c.test_sentences)} test`)}
      ${tile("Training words", fmtN(c.train_words), `${fmtN(c.test_words)} held-out words`)}
      ${tile("Vocabulary V", fmtN(m.vocab_size), `${fmtN(m.vocab_words)} words + &lt;unk&gt; + &lt;/s&gt;`)}
      ${tile("Model order", `${m.max_order} (${orderName(m.max_order)})`, `orders 1–${m.max_order} available`, true)}
      ${tile("Model version", `v${s.version}`, esc(s.trained_at.replace("T", " ").replace("+00:00", " UTC")), true)}
    </div>

    <div class="grid grid-2">
      <div class="card"><h2>What is a language model?</h2>
        <p>A language model assigns a probability to a sequence of words. By the chain rule,
        <code>P(w₁ … wₙ) = Π P(wᵢ | w₁ … wᵢ₋₁)</code>, so a model that can say how likely each next
        word is can score whole sentences.</p>
        <p>An <b>n-gram model</b> makes the Markov assumption: only the last <code>n − 1</code> words
        matter. A bigram model conditions on one previous word, a trigram model on two. The
        probabilities are estimated by counting how often each word followed each context in the
        training text.</p>
        <p class="muted small">No neural network, embedding or pretrained model is involved anywhere.</p>
      </div>
      <div class="card"><h2>How next-word prediction works</h2>
        <ol style="margin:0;padding-left:20px">
          <li>Tokenize the sentence you are typing, exactly as the training text was tokenized.</li>
          <li>Keep the last <code>n − 1</code> tokens as the context; pad a short history with <code>&lt;s&gt;</code> and map unknown words to <code>&lt;unk&gt;</code>.</li>
          <li>If that context never occurred in training, back off to a shorter context.</li>
          <li>Look up every word that followed the context and compute <code>P(word | context)</code> with MLE or add-k smoothing.</li>
          <li>Rank the candidates by probability and return the top K with their counts.</li>
        </ol>
      </div>
    </div>

    <div class="card"><h2>Architecture</h2>
      <p class="card-sub">The same pipeline serves training, the dashboard and the command line. Training writes one model; inference only reads it.</p>
      <div class="flow">${flow.map(([t, d], i) => `${i ? `<div class="flow-arrow" aria-hidden="true">→</div>` : ""}
        <div class="flow-step"><div class="n">${String(i + 1).padStart(2, "0")}</div><div class="t">${t}</div><div class="d">${d}</div></div>`).join("")}</div>
    </div>

    <div class="grid grid-2">
      <div class="card"><h2>N-gram inventory</h2>
        <p class="card-sub">Counted on the training split only. "Events" counts every (context, word) occurrence including the final <code>&lt;/s&gt;</code>.</p>
        <div class="table-wrap"><table>
          <thead><tr><th>Order</th><th class="num">Distinct n-grams</th><th class="num">Events</th><th class="num">Avg. count</th></tr></thead>
          <tbody>${m.ngrams.map((g) => `<tr><td>${g.order} · ${cap(g.name)}</td><td class="num">${fmtN(g.distinct)}</td>
            <td class="num">${fmtN(g.total)}</td><td class="num">${(g.total / g.distinct).toFixed(2)}</td></tr>`).join("")}</tbody>
        </table></div>
        <p class="small muted" style="margin-top:10px">As the order grows, distinct n-grams approach the number of events: most longer n-grams are seen only once. That is data sparsity.</p>
      </div>
      <div class="card"><h2>Most frequent words</h2>
        ${hbars(m.top_words.map((w) => ({ label: w.word, value: w.count, display: fmtN(w.count) })))}
      </div>
    </div>

    <div class="card"><h2>Current training corpus <span class="badge">${esc(c.name)}</span></h2>
      <p class="card-sub">The first few training sentences. Load a different corpus on the <a href="#corpus">Corpus &amp; Training</a> page.</p>
      <ol class="small" style="margin:0;padding-left:20px">${c.preview.map((p) => `<li>${esc(p)}</li>`).join("")}</ol>
    </div>
  </div>`;
  $$("[data-go]", el).forEach((b) => b.addEventListener("click", () => {
    if (b.dataset.go === "predict" && b.dataset.text) ui.predict.text = b.dataset.text;
    if (b.dataset.go === "generate" && b.dataset.text) ui.gen.seed_text = b.dataset.text;
    location.hash = b.dataset.go;
  }));
}

// ------------------------------------------------------------------ page 2: prediction studio

function renderPredict(el) {
  const u = ui.predict;
  const examples = ["I love machine", "The weather is", "A language model", "Machine learning is",
                    "I am going to the", "Thank you for", "Quantum flux capacitor"];
  el.innerHTML = `<div class="stack">
    <div class="card">
      <label class="field" for="p-text">Type a sentence — suggestions update as you type</label>
      <textarea id="p-text" class="big" rows="2" spellcheck="false" aria-describedby="p-ghost">${esc(u.text)}</textarea>
      <div id="p-ghost" class="ghost-hint" aria-live="polite"></div>
      <div class="chips" style="margin:10px 0 14px">${examples.map((e) => `<button class="chip" data-example="${esc(e)}">${esc(e)}</button>`).join("")}</div>
      <div class="controls">
        <label class="field">Model order ${orderSeg("p-order", u.order)}</label>
        <label class="field">Suggestions (top K)<input id="p-topk" type="number" min="1" max="25" value="${u.top_k}"></label>
        <label class="field">Smoothing<span class="check" style="padding:7px 0"><input id="p-smooth" type="checkbox" ${u.smoothing ? "checked" : ""}> Add-k smoothing</span></label>
        <label class="field">k <span class="hint">(pseudo-count, &gt; 0)</span><input id="p-k" type="number" min="0.000001" max="100" step="any" value="${u.k}" ${u.smoothing ? "" : "disabled"}></label>
        <div><button class="btn" id="p-go">Predict</button></div>
      </div>
    </div>
    <div class="grid grid-2">
      <div class="card"><h2>Next-word suggestions</h2><div id="p-sugs"><div class="empty">…</div></div></div>
      <div class="card"><h2>How this prediction was made</h2><div id="p-why"><div class="empty">…</div></div></div>
    </div>
    <div class="card"><h2>Sentence surprise heatmap <span class="badge">per-word probability</span></h2>
      <p class="card-sub">Each word you typed, coloured by how surprised the model was to see it: surprisal = −log₂ P(word | context) bits. Hover a word for its numbers.</p>
      <div id="p-heat"></div>
    </div>
    <div class="card"><h2>Continue writing</h2>
      <p class="card-sub">Samples a continuation of your text from the same model and order (unsmoothed MLE distribution). Full controls live in the <a href="#generate">Text Generation Lab</a>.</p>
      <div class="controls">
        <label class="field">Max words <span class="range-val" id="pg-mw-v">${u.max_words}</span><input id="pg-mw" type="range" min="1" max="60" value="${u.max_words}"></label>
        <label class="field">Temperature <span class="range-val" id="pg-t-v">${u.temperature.toFixed(2)}</span><input id="pg-t" type="range" min="0.1" max="3" step="0.05" value="${u.temperature}"></label>
        <label class="field">Random seed<input id="pg-seed" type="number" min="0" value="${u.seed}"></label>
        <div class="row"><button class="btn" id="pg-go">Continue text</button><button class="btn ghost" id="pg-re">Try another seed</button></div>
      </div>
      <div id="pg-out" style="margin-top:14px"></div>
    </div>
  </div>`;

  const text = $("#p-text", el);
  const run = latestOnly(() => api("POST", "predict", {
    text: u.text, order: u.order, top_k: u.top_k, smoothing: u.smoothing, k: u.k,
  }));
  const runScore = latestOnly(() => api("POST", "score", {
    text: u.text, order: u.order, smoothing: u.smoothing, k: u.k,
  }));
  let current = null;
  const predict = async () => {
    try {
      const r = await run();
      if (r) { current = r; showPrediction(r); }
    } catch (error) {
      current = null;
      $("#p-sugs", el).innerHTML = errorBox(error);
      $("#p-why", el).innerHTML = "";
      $("#p-ghost", el).textContent = "";
    }
    try {
      const s = await runScore();
      if (s) $("#p-heat", el).innerHTML = heatmap(s);
    } catch (error) { $("#p-heat", el).innerHTML = errorBox(error); }
  };
  const accept = (word) => {
    const sep = u.text && !/\s$/.test(u.text) ? " " : "";
    u.text = `${u.text}${sep}${word}`;
    text.value = u.text;
    text.focus();
    text.setSelectionRange(u.text.length, u.text.length);
    predict();
  };
  const later = debounce(predict, 220);

  const showPrediction = (r) => {
    const sugs = r.suggestions;
    let html = "";
    if (!sugs.length) {
      html = `<div class="callout warn">No word candidates. Under MLE the only thing ever observed after this context is the end of the sentence
        (P(&lt;/s&gt;) = ${fmtP(r.hidden_mass["</s>"])}). Enable smoothing to give every word some probability.</div>`;
    } else {
      html += `<div class="chips" style="margin-bottom:14px">${sugs.map((s, i) =>
        `<button class="suggest-chip ${i === 0 ? "first" : ""}" data-word="${esc(s.word)}" title="Append to the text">
          <span class="w">${esc(s.word)}</span><span class="p">P = ${fmtP(s.probability)}</span></button>`).join("")}</div>`;
      html += hbars(sugs.map((s) => ({
        label: s.word, value: s.probability, cls: s.count ? "" : "faint",
        note: s.count ? `×${s.count}` : "unseen",
        title: `C(context, ${s.word}) = ${s.count}`,
      })));
      const unseen = sugs.filter((s) => !s.count).length;
      html += `<p class="small muted" style="margin-top:10px">
        ×n is how often the word followed the context in training. ${unseen ? `${unseen} suggestion(s) were never seen after this context;
        they only have probability because of add-k smoothing (pale bars), and unseen words tie, so they are ordered alphabetically.` : ""}
        Click a suggestion to append it.</p>`;
    }
    $("#p-sugs", el).innerHTML = html;
    $$(".suggest-chip", el).forEach((b) => b.addEventListener("click", () => accept(b.dataset.word)));
    $("#p-ghost", el).innerHTML = sugs.length
      ? `<kbd>Tab</kbd> accept <b>${esc(sugs[0].word)}</b> <span class="muted">(P = ${fmtP(sugs[0].probability)})</span>
         · <kbd>Alt</kbd>+<kbd>1</kbd>…<kbd>${Math.min(9, sugs.length)}</kbd> pick a suggestion`
      : `<span class="muted">No word suggestion for this context.</span>`;

    const fallback = r.backed_off
      ? `<span class="badge warn">Fallback: ${orderName(r.requested_order)} → ${orderName(r.used_order)}</span>`
      : `<span class="badge good">Full ${orderName(r.used_order)} context found</span>`;
    $("#p-why", el).innerHTML = kvTable([
      ["Tokens of current sentence", toks(r.input_tokens, "(start of a new sentence)")],
      [`Requested ${orderName(r.requested_order)} context`, toks(r.requested_context)],
      ["Context actually used", `${toks(r.used_context)} ${fallback}`],
      ["C(context) in training", `<span class="mono">${fmtN(r.context_count)}</span>`],
      ["Distinct words seen after it", `<span class="mono">${fmtN(r.distinct_followers)}</span>`],
      ["Unknown input words", r.unknown_words.length ? r.unknown_words.map((w) => tok(w, "unk")).join("") + ` <span class="muted small">→ &lt;unk&gt;</span>` : `<span class="muted">none</span>`],
      ["Estimator", r.smoothing ? `Add-k, k = ${r.k}, V = ${fmtN(r.vocab_size)}` : "Maximum likelihood (no smoothing)"],
      ["Mass not shown as suggestions", `P(&lt;/s&gt;) = <span class="mono">${fmtP(r.hidden_mass["</s>"])}</span>, P(&lt;unk&gt;) = <span class="mono">${fmtP(r.hidden_mass["<unk>"])}</span>`],
    ]) + `<div class="callout" style="margin-top:12px">${esc(r.explanation)}</div>`;
  };

  text.addEventListener("input", () => { u.text = text.value; later(); });
  text.addEventListener("keydown", (e) => {
    const sugs = current ? current.suggestions : [];
    if (e.key === "Tab" && !e.shiftKey && sugs.length) {
      e.preventDefault();
      accept(sugs[0].word);
    } else if (e.altKey && /^[1-9]$/.test(e.key) && sugs[+e.key - 1]) {
      e.preventDefault();
      accept(sugs[+e.key - 1].word);
    }
  });
  $$("[data-example]", el).forEach((b) => b.addEventListener("click", () => {
    u.text = b.dataset.example; text.value = u.text; predict();
  }));
  bindSeg(el, "p-order", (n) => { u.order = n; predict(); });
  $("#p-topk", el).addEventListener("change", (e) => {
    u.top_k = Math.max(1, Math.min(25, parseInt(e.target.value, 10) || 5));
    e.target.value = u.top_k; predict();
  });
  $("#p-smooth", el).addEventListener("change", (e) => {
    u.smoothing = e.target.checked; $("#p-k", el).disabled = !u.smoothing; predict();
  });
  $("#p-k", el).addEventListener("change", (e) => {
    const v = parseFloat(e.target.value);
    if (!(v > 0)) { toast("k must be a positive number."); e.target.value = u.k; return; }
    u.k = v; predict();
  });
  $("#p-go", el).addEventListener("click", predict);

  // continuation
  const mw = $("#pg-mw", el), temp = $("#pg-t", el), seed = $("#pg-seed", el);
  mw.addEventListener("input", () => { u.max_words = +mw.value; $("#pg-mw-v", el).textContent = mw.value; });
  temp.addEventListener("input", () => { u.temperature = +temp.value; $("#pg-t-v", el).textContent = (+temp.value).toFixed(2); });
  seed.addEventListener("change", () => { u.seed = Math.max(0, parseInt(seed.value, 10) || 0); seed.value = u.seed; });
  const cont = async () => {
    const out = $("#pg-out", el);
    try {
      const r = await api("POST", "generate", {
        seed_text: u.text, order: u.order, max_words: u.max_words, temperature: u.temperature, seed: u.seed, k: 0,
      });
      out.innerHTML = `<div class="gen-output">${renderGenTokens(r)}</div>
        <div class="row small muted" style="margin-top:8px">${genMeta(r)}<span class="spacer"></span>
        <button class="btn small ghost" id="pg-use">Use as input</button></div>`;
      $("#pg-use", el).addEventListener("click", () => {
        u.text = r.text; text.value = u.text; predict(); window.scrollTo({ top: 0, behavior: "smooth" });
      });
    } catch (error) { out.innerHTML = errorBox(error); }
  };
  $("#pg-go", el).addEventListener("click", cont);
  $("#pg-re", el).addEventListener("click", () => { u.seed = Math.floor(Math.random() * 100000); seed.value = u.seed; cont(); });

  predict();
}

/** Colour for a surprisal in bits: green (expected) → amber → red (very surprising). */
function heatColor(bits) {
  if (bits === null) return "var(--bad)";
  const t = Math.max(0, Math.min(1, bits / 12));
  const hue = 140 - 140 * t;                 // 140 = green, 0 = red
  return `hsl(${hue.toFixed(0)} 60% 45%)`;
}

function heatmap(s) {
  if (!s.tokens.length) return `<div class="empty">Start typing a sentence to see how surprising each word is.</div>`;
  const words = s.tokens.map((t) => {
    const bits = t.surprisal_bits;
    const title = `${t.word}${t.mapped !== t.word ? ` (→ ${t.mapped})` : ""}\nP(${t.mapped} | ${t.context.join(" ") || "∅"}) = ${fmtP(t.probability)}\nsurprisal = ${bits === null ? "∞" : bits.toFixed(2)} bits\nseen ${t.count}× after this context`;
    return `<span class="heat-word" style="--heat:${heatColor(bits)}" title="${esc(title)}">
      <span class="hw">${esc(t.word)}</span><span class="hb">${bits === null ? "∞" : bits.toFixed(1)}</span></span>`;
  }).join("");
  const summary = s.perplexity === null
    ? `<span class="badge bad">${s.zero_prob_tokens} word(s) with P = 0 → perplexity ∞</span> <span class="small muted">Turn smoothing on to score unseen words.</span>`
    : `<span class="badge accent">perplexity ${s.perplexity.toFixed(2)}</span> <span class="badge">${(s.cross_entropy / Math.LN2).toFixed(2)} bits / word</span>`;
  return `<div class="heat">${words}</div>
    <div class="row" style="margin-top:12px">${summary}<span class="spacer"></span>
      <span class="heat-scale small muted">expected <i></i> surprising</span></div>`;
}

// ------------------------------------------------------------------ page 3: n-gram explorer

function renderExplorer(el) {
  const u = ui.explorer;
  el.innerHTML = `<div class="stack">
    <div class="card">
      <div class="controls">
        <label class="field">N-gram order ${orderSeg("x-order", u.order)}</label>
        <label class="field">Search <span class="hint">(substring, e.g. "machine")</span><input id="x-q" type="text" value="${esc(u.q)}" placeholder="filter n-grams…"></label>
        <label class="field">Minimum count<input id="x-min" type="number" min="1" value="${u.min_count}"></label>
        <label class="field">Rows<select id="x-limit">${[25, 50, 100, 250, 500].map((n) => `<option ${n === u.limit ? "selected" : ""}>${n}</option>`).join("")}</select></label>
        <label class="field">Boundary tokens<span class="check" style="padding:7px 0"><input id="x-special" type="checkbox" ${u.include_special ? "checked" : ""}> include &lt;s&gt; / &lt;/s&gt;</span></label>
      </div>
    </div>
    <div class="grid grid-2">
      <div class="card"><h2 id="x-title">N-grams</h2><div id="x-summary" class="card-sub"></div><div id="x-table"></div></div>
      <div class="card"><h2>Context-conditioned next-word distribution</h2>
        <p class="card-sub">Pick a row on the left, or type a context. Shows the MLE distribution P(w | context) from raw counts.</p>
        <div class="row" style="margin-bottom:12px"><input id="x-ctx" type="text" value="${esc(u.ctx)}" placeholder="e.g. i love" style="flex:1"></div>
        <div id="x-dist"></div>
      </div>
    </div>
  </div>`;

  const runTable = latestOnly(() => api("GET", "ngrams", {
    order: u.order, q: u.q, limit: u.limit, min_count: u.min_count, include_special: u.include_special,
  }));
  const loadTable = async () => {
    try {
      const r = await runTable();
      if (!r) return;
      $("#x-title", el).textContent = `${cap(orderName(r.order))} counts`;
      $("#x-summary", el).innerHTML = `${fmtN(r.matches)} match(es) · showing ${fmtN(r.rows.length)} ·
        ${fmtN(r.distinct)} distinct ${orderName(r.order)}s, ${fmtN(r.total)} events in training`;
      if (!r.rows.length) { $("#x-table", el).innerHTML = `<div class="empty">No n-grams match these filters.</div>`; return; }
      const top = r.rows[0].count;
      $("#x-table", el).innerHTML = `<div class="table-wrap scroll-y"><table>
        <thead><tr><th>#</th><th>${cap(orderName(r.order))}</th><th class="num">Count</th><th class="num">C(context)</th><th class="num">MLE P(w | ctx)</th></tr></thead>
        <tbody>${r.rows.map((row, i) => `<tr class="clickable" data-ctx="${esc(row.ngram.slice(0, -1).join(" "))}">
          <td class="muted">${i + 1}</td>
          <td>${row.ngram.slice(0, -1).map((t) => tok(t)).join("")}${tok(row.ngram[row.ngram.length - 1], "hl")}</td>
          <td class="num"><div style="display:flex;align-items:center;gap:8px;justify-content:flex-end">
            <div class="bar-track" style="width:60px;height:8px"><div class="bar-fill" style="width:${(100 * row.count) / top}%"></div></div>${fmtN(row.count)}</div></td>
          <td class="num">${fmtN(row.context_count)}</td><td class="num">${fmtP(row.mle)}</td></tr>`).join("")}</tbody>
      </table></div>
      <p class="small muted" style="margin-top:8px">Highlighted token = the predicted word; the tokens before it are the context.${r.order === 1 ? " For unigrams the context is empty and C(context) is the total number of training tokens." : ""}</p>`;
      $$("tr[data-ctx]", el).forEach((tr) => tr.addEventListener("click", () => {
        u.ctx = tr.dataset.ctx; $("#x-ctx", el).value = u.ctx; loadDist();
      }));
    } catch (error) { $("#x-table", el).innerHTML = errorBox(error); }
  };

  const runDist = latestOnly(() => api("POST", "distribution", { context: u.ctx, order: u.order, k: 1, limit: 25 }));
  const loadDist = async () => {
    try {
      const r = await runDist();
      if (!r) return;
      let html = `<div class="row small" style="margin-bottom:10px">Context ${toks(r.context)}
        <span class="badge ${r.seen ? "good" : "bad"}">C(context) = ${fmtN(r.context_count)}</span>
        <span class="badge">${fmtN(r.distinct_followers)} distinct followers</span></div>`;
      if (r.unknown_words.length) html += `<div class="callout warn small" style="margin-bottom:10px">Unknown words mapped to &lt;unk&gt;: ${r.unknown_words.map((w) => esc(w)).join(", ")}</div>`;
      if (!r.seen) {
        html += `<div class="callout bad">This ${orderName(r.order)} context never occurs in the training data, so its MLE distribution is undefined (0 / 0).</div>`;
      } else {
        html += hbars(r.rows.map((row) => ({ label: row.word, value: row.mle, note: `${row.count}/${r.context_count}` })));
        if (r.rows_truncated) html += `<p class="small muted" style="margin-top:8px">+ ${r.rows_truncated} more follower(s) not shown.</p>`;
        html += `<p class="small muted" style="margin-top:8px">Σ P = ${r.checks.mle_sum.toFixed(6)} — the MLE distribution over observed followers sums to one.</p>`;
      }
      $("#x-dist", el).innerHTML = html;
    } catch (error) { $("#x-dist", el).innerHTML = errorBox(error); }
  };

  bindSeg(el, "x-order", (n) => { u.order = n; loadTable(); loadDist(); });
  $("#x-q", el).addEventListener("input", debounce((e) => { u.q = e.target.value; loadTable(); }, 250));
  $("#x-min", el).addEventListener("change", (e) => { u.min_count = Math.max(1, parseInt(e.target.value, 10) || 1); e.target.value = u.min_count; loadTable(); });
  $("#x-limit", el).addEventListener("change", (e) => { u.limit = +e.target.value; loadTable(); });
  $("#x-special", el).addEventListener("change", (e) => { u.include_special = e.target.checked; loadTable(); });
  $("#x-ctx", el).addEventListener("input", debounce((e) => { u.ctx = e.target.value; loadDist(); }, 250));
  loadTable();
  loadDist();
}

// ------------------------------------------------------------------ page 4: probability laboratory

const K_PRESETS = [0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5];

function renderProbability(el) {
  const u = ui.prob;
  if (u.order < 2 && maxOrder() >= 2) u.order = 2;
  const kIndex = Math.max(0, K_PRESETS.indexOf(u.k));
  el.innerHTML = `<div class="stack">
    <div class="card">
      <div class="controls">
        <label class="field" style="grid-column: span 2">Context <span class="hint">(the last ${u.order - 1} word(s) are used; type &lt;s&gt; for a sentence start)</span>
          <input id="b-ctx" type="text" value="${esc(u.context)}"></label>
        <label class="field">Order ${orderSeg("b-order", u.order)}</label>
        <label class="field">k = <span class="range-val" id="b-k-v">${u.k}</span><input id="b-k" type="range" min="0" max="${K_PRESETS.length - 1}" step="1" value="${kIndex}"></label>
        <label class="field">Rows<select id="b-limit">${[10, 15, 25, 50].map((n) => `<option ${n === u.limit ? "selected" : ""}>${n}</option>`).join("")}</select></label>
      </div>
      <div class="row small" style="margin-top:12px"><span class="muted">Frequent contexts:</span><span class="chips" id="b-popular"></span></div>
    </div>
    <div id="b-body"></div>
  </div>`;

  const run = latestOnly(() => api("POST", "distribution", { context: u.context, order: u.order, k: u.k, limit: u.limit }));
  const load = async () => {
    const body = $("#b-body", el);
    try {
      const r = await run();
      if (!r) return;
      $("#b-popular", el).innerHTML = r.popular_contexts.slice(0, 10).map((p) =>
        `<button class="chip" data-pc="${esc(p.context.join(" "))}">${esc(p.context.join(" "))} <span class="muted">${p.count}</span></button>`).join("");
      $$("[data-pc]", el).forEach((b) => b.addEventListener("click", () => {
        u.context = b.dataset.pc; $("#b-ctx", el).value = u.context; load();
      }));
      body.innerHTML = probBody(r);
    } catch (error) { body.innerHTML = errorBox(error); }
  };

  bindSeg(el, "b-order", (n) => { u.order = n; renderProbability(el); });
  $("#b-ctx", el).addEventListener("input", debounce((e) => { u.context = e.target.value; load(); }, 250));
  $("#b-k", el).addEventListener("input", (e) => {
    u.k = K_PRESETS[+e.target.value]; $("#b-k-v", el).textContent = u.k; load();
  });
  $("#b-limit", el).addEventListener("change", (e) => { u.limit = +e.target.value; load(); });
  load();
}

function probBody(r) {
  const C = r.context_count, V = r.vocab_size, k = r.k;
  const top = r.rows[0];
  const ctxLabel = r.context.length ? r.context.join(" ") : "∅";
  let worked;
  if (top) {
    worked = `<div class="formula">MLE    P(${esc(top.word)} | ${esc(ctxLabel)}) = C(${esc(ctxLabel)}, ${esc(top.word)}) / C(${esc(ctxLabel)})
                    = ${top.count} / ${C} = <span class="em">${fmtP(top.mle)}</span>

add-k  P(${esc(top.word)} | ${esc(ctxLabel)}) = (C(${esc(ctxLabel)}, ${esc(top.word)}) + k) / (C(${esc(ctxLabel)}) + k·V)
                    = (${top.count} + ${k}) / (${C} + ${k}·${V}) = <span class="em">${fmtP(top.smoothed)}</span>

unseen word w'      = (0 + ${k}) / (${C} + ${k}·${V}) = <span class="em">${fmtP(r.unseen.probability_each)}</span>   (MLE: 0)</div>`;
  } else {
    worked = `<div class="formula">MLE    P(w | ${esc(ctxLabel)}) = C(${esc(ctxLabel)}, w) / C(${esc(ctxLabel)}) = 0 / 0   → undefined

add-k  P(w | ${esc(ctxLabel)}) = (0 + ${k}) / (0 + ${k}·${V}) = 1 / V = <span class="em">${fmtP(r.unseen.probability_each)}</span>   (uniform)</div>`;
  }
  const maxP = Math.max(...r.rows.map((x) => Math.max(x.mle, x.smoothed)), r.unseen.probability_each, 1e-12);
  const pairs = r.rows.map((row) => `<div class="bar-row">
      <div class="bar-label">${esc(row.word)}</div>
      <div class="bar-pair">
        <div class="bar-track"><div class="bar-fill" style="width:${(100 * row.mle) / maxP}%"></div></div>
        <div class="bar-track"><div class="bar-fill alt" style="width:${(100 * row.smoothed) / maxP}%"></div></div>
      </div>
      <div class="bar-value">${fmtP(row.mle)}<br>${fmtP(row.smoothed)}</div></div>`).join("");
  const unseenRow = `<div class="bar-row"><div class="bar-label muted">any unseen</div>
      <div class="bar-pair"><div class="bar-track"></div>
      <div class="bar-track"><div class="bar-fill alt" style="width:${(100 * r.unseen.probability_each) / maxP}%"></div></div></div>
      <div class="bar-value">0<br>${fmtP(r.unseen.probability_each)}</div></div>`;

  const seenMassSmoothed = r.rows.reduce((a, x) => a + x.smoothed, 0);
  return `<div class="stack">
    <div class="tiles">
      ${tile("Context", ctxLabel, `${cap(orderName(r.order))} model`, true)}
      ${tile("C(context)", fmtN(C), r.seen ? "occurrences as a history" : "never seen in training")}
      ${tile("Distinct followers", fmtN(r.distinct_followers), "words with C(context, w) > 0")}
      ${tile("Unseen outcomes", fmtN(r.unseen.types), `of V = ${fmtN(V)} possible next tokens`)}
      ${tile("Mass moved to unseen", fmtPct(r.unseen.total_mass), `by add-${k} smoothing`)}
    </div>
    ${r.unknown_words.length ? `<div class="callout warn">Unknown words mapped to &lt;unk&gt;: ${r.unknown_words.map(esc).join(", ")}</div>` : ""}
    ${r.seen ? "" : `<div class="callout bad">This context never occurs in training. MLE has nothing to divide by, so it cannot assign any probability; add-k falls back to the uniform distribution 1/V.</div>`}
    <div class="grid grid-2">
      <div class="card"><h2>Worked calculation</h2>${worked}
        <p class="small muted">k = 1 is Laplace smoothing. Every one of the V possible next tokens (all words, &lt;unk&gt; and &lt;/s&gt;) receives k pseudo-counts, so the denominator grows by k·V and the distribution still sums to one.</p>
      </div>
      <div class="card"><h2>MLE vs. smoothed distribution</h2>
        <div class="legend"><span><i style="background:var(--accent)"></i>MLE</span><span><i style="background:var(--series-2)"></i>Add-${k}</span></div>
        ${r.rows.length ? `<div class="bars">${pairs}${unseenRow}</div>` : unseenRow}
      </div>
    </div>
    <div class="card"><h2>Count table</h2>
      <div class="table-wrap"><table>
        <thead><tr><th>Next word</th><th class="num">C(ctx, w)</th><th class="num">MLE</th><th class="num">Add-${k}</th><th class="num">Change</th></tr></thead>
        <tbody>${r.rows.map((row) => `<tr><td>${tok(row.word)}</td><td class="num">${row.count}</td><td class="num">${fmtP(row.mle)}</td>
          <td class="num">${fmtP(row.smoothed)}</td><td class="num">${row.smoothed >= row.mle ? "+" : "−"}${fmtP(Math.abs(row.smoothed - row.mle))}</td></tr>`).join("")}
          ${r.rows_truncated ? `<tr><td colspan="5" class="muted small">+ ${r.rows_truncated} more seen follower(s) not listed</td></tr>` : ""}
          <tr><td class="muted">each of ${fmtN(r.unseen.types)} unseen tokens</td><td class="num">0</td><td class="num">0</td><td class="num">${fmtP(r.unseen.probability_each)}</td><td class="num">+${fmtP(r.unseen.probability_each)}</td></tr>
          <tr class="total"><td>Σ over all V tokens</td><td class="num">${fmtN(C)}</td><td class="num">${r.seen ? r.checks.mle_sum.toFixed(6) : "undefined"}</td><td class="num">${r.checks.smoothed_sum.toFixed(6)}</td><td></td></tr>
        </tbody></table></div>
      <p class="small muted" style="margin-top:8px">The sums are computed by the engine over the full distributions (all ${fmtN(V)} tokens for add-k), not just the rows shown. Listed seen words keep ${fmtPct(seenMassSmoothed)} of the smoothed mass.</p>
    </div>
    <div class="card"><h2>What smoothing does, mathematically</h2>
      <p>MLE is the estimate that makes the training data as likely as possible, which is exactly why it gives
      <b>zero</b> probability to anything not in the training data. One unseen pair in a test sentence makes the whole sentence impossible.</p>
      <div class="formula">seen word:    (c + k) / (C + kV)  &lt;  c / C   whenever c / C &gt; 1 / V   (probability is discounted)
unseen word:  k / (C + kV)  &gt;  0                                    (probability is created)
total moved:  (V − distinct) · k / (C + kV)  =  <span class="em">${fmtPct(r.unseen.total_mass)}</span> for this context</div>
      <p class="small muted">Large k pushes every context towards the uniform 1/V; with a big vocabulary and a small corpus, add-1 moves far too much mass to unseen words. The Model Evaluation page shows held-out perplexity as a function of k.</p>
    </div>
  </div>`;
}

// ------------------------------------------------------------------ page 5: generation lab

function renderGenTokens(r) {
  const seed = r.seed_text.trim() ? `<span class="seed">${esc(r.seed_text.trim())}</span> ` : "";
  const words = r.generated_tokens.map((t, i) => {
    const delay = `style="animation-delay:${Math.min(i * 45, 2000)}ms"`;
    return t === "</s>" ? `<span class="w eos" ${delay}>&lt;/s&gt;</span>` : `<span class="w" ${delay}>${esc(t)}</span>`;
  }).join(" ");
  return seed + (words || `<span class="muted small">(nothing generated)</span>`);
}

const STOP_REASONS = { end_of_sentence: "reached </s>", max_words: "hit max words", no_candidates: "no candidates" };

function genMeta(r) {
  const backoffs = r.steps.filter((s) => s.backed_off).length;
  return `<span class="badge accent">${cap(orderName(r.order))}</span>
    <span class="badge">${r.steps.length} steps</span>
    <span class="badge">stopped: ${esc(STOP_REASONS[r.stop_reason] || r.stop_reason)}</span>
    <span class="badge">seed ${r.seed ?? "random"}</span>
    <span class="badge">T = ${r.temperature}</span>
    <span class="badge">${r.k > 0 ? `add-${r.k}` : "MLE"} sampling</span>
    ${backoffs ? `<span class="badge warn">${backoffs} fallback(s)</span>` : ""}`;
}

function renderGenerate(el) {
  const u = ui.gen;
  el.innerHTML = `<div class="stack">
    <div class="card">
      <div class="controls">
        <label class="field" style="grid-column: span 2">Seed text <span class="hint">(may be empty)</span><input id="g-seed-text" type="text" value="${esc(u.seed_text)}"></label>
        <label class="field">Order ${orderSeg("g-order", u.order)}</label>
        <label class="field">Max words <span class="range-val" id="g-mw-v">${u.max_words}</span><input id="g-mw" type="range" min="1" max="100" value="${u.max_words}"></label>
        <label class="field">Temperature <span class="range-val" id="g-t-v">${u.temperature.toFixed(2)}</span><input id="g-t" type="range" min="0.1" max="3" step="0.05" value="${u.temperature}"></label>
        <label class="field">Random seed<input id="g-seed" type="number" min="0" value="${u.seed}"></label>
        <label class="field">Sampling k <span class="hint">(0 = MLE)</span><input id="g-k" type="number" min="0" max="100" step="any" value="${u.k}"></label>
        <label class="field">Stopping<span class="check" style="padding:7px 0"><input id="g-eos" type="checkbox" ${u.stop_at_eos ? "checked" : ""}> stop at &lt;/s&gt;</span></label>
      </div>
      <div class="row" style="margin-top:14px">
        <button class="btn" id="g-go">Generate</button>
        <button class="btn ghost" id="g-re">Regenerate with a new seed</button>
        <span class="small muted">The same seed and settings always reproduce the same text.</span>
      </div>
    </div>
    <div class="card"><h2>Generated text</h2><div id="g-out"><div class="empty">Press Generate.</div></div></div>
    <div class="card"><h2>Step by step</h2>
      <p class="card-sub">At each step the model looks up the context, falls back to a shorter one if it was never seen, reshapes the distribution with the temperature and samples one token by inverse-CDF sampling.</p>
      <div id="g-steps"></div>
    </div>
    <div class="card"><h2>About temperature</h2>
      <div class="formula">p'(w) = p(w)^(1/T) / Σ_v p(v)^(1/T)</div>
      <p class="small">Temperature is a <b>sampling-time extension</b>, not part of Maximum Likelihood Estimation: the model's probabilities are
      estimated first, then reshaped. T = 1 samples from the model as estimated; T &lt; 1 sharpens towards the most likely word (T → 0 is greedy);
      T &gt; 1 flattens the distribution. With k = 0 only continuations that really occurred in training can be produced; with k &gt; 0 any vocabulary word can appear.</p>
    </div>
  </div>`;

  const generate = async () => {
    try {
      const r = await api("POST", "generate", {
        seed_text: u.seed_text, order: u.order, max_words: u.max_words, temperature: u.temperature,
        seed: u.seed, k: u.k, stop_at_eos: u.stop_at_eos,
      });
      $("#g-out", el).innerHTML = `<div class="gen-output">${renderGenTokens(r)}</div>
        <div class="row" style="margin-top:10px">${genMeta(r)}</div>
        <p class="small muted" style="margin-top:8px">Rendered: ${esc(r.text)}</p>`;
      $("#g-steps", el).innerHTML = r.steps.length ? `<div class="table-wrap scroll-y"><table>
        <thead><tr><th>#</th><th>Context used</th><th>Order</th><th class="num">Candidates</th><th>Chosen</th><th class="num">P′(chosen)</th><th>Top candidates (after temperature)</th></tr></thead>
        <tbody>${r.steps.map((s) => `<tr>
          <td class="muted">${s.index}</td><td>${toks(s.context)}</td>
          <td>${cap(orderName(s.order_used))}${s.backed_off ? ` <span class="badge warn">fallback</span>` : ""}</td>
          <td class="num">${fmtN(s.num_candidates)}</td><td>${tok(s.chosen, "hl")}</td><td class="num">${fmtP(s.probability)}</td>
          <td><div class="mini-cands">${s.top_candidates.map(([w, p]) => `<span class="${w === s.chosen ? "chosen" : ""}">${esc(w)} ${fmtP(p)}</span>`).join("")}</div></td>
        </tr>`).join("")}</tbody></table></div>` : `<div class="empty">No steps.</div>`;
    } catch (error) {
      $("#g-out", el).innerHTML = errorBox(error);
      $("#g-steps", el).innerHTML = "";
    }
  };

  $("#g-seed-text", el).addEventListener("input", (e) => { u.seed_text = e.target.value; });
  $("#g-seed-text", el).addEventListener("keydown", (e) => { if (e.key === "Enter") generate(); });
  bindSeg(el, "g-order", (n) => { u.order = n; generate(); });
  $("#g-mw", el).addEventListener("input", (e) => { u.max_words = +e.target.value; $("#g-mw-v", el).textContent = e.target.value; });
  $("#g-t", el).addEventListener("input", (e) => { u.temperature = +e.target.value; $("#g-t-v", el).textContent = (+e.target.value).toFixed(2); });
  $("#g-seed", el).addEventListener("change", (e) => { u.seed = Math.max(0, parseInt(e.target.value, 10) || 0); e.target.value = u.seed; });
  $("#g-k", el).addEventListener("change", (e) => {
    const v = parseFloat(e.target.value);
    if (!(v >= 0) || v > 100) { toast("k must be between 0 and 100."); e.target.value = u.k; return; }
    u.k = v;
  });
  $("#g-eos", el).addEventListener("change", (e) => { u.stop_at_eos = e.target.checked; });
  $("#g-go", el).addEventListener("click", generate);
  $("#g-re", el).addEventListener("click", () => {
    u.seed = Math.floor(Math.random() * 1000000); $("#g-seed", el).value = u.seed; generate();
  });
  generate();
}

// ------------------------------------------------------------------ page 6: evaluation

function renderEvaluate(el) {
  const u = ui.eval;
  el.innerHTML = `<div class="stack">
    <div class="callout"><b>Held-out evaluation.</b> The test sentences were split off before counting and are never seen during training.
      Training-set numbers are shown separately at the bottom; they measure fit, not quality.</div>
    <div class="card"><div class="row">
      <label class="field" style="max-width:220px">Smoothing k for the add-k models<input id="e-k" type="number" min="0.000001" max="100" step="any" value="${u.k}"></label>
      <div class="chips">${[0.01, 0.1, 0.5, 1].map((k) => `<button class="chip" data-k="${k}">k = ${k}</button>`).join("")}</div>
    </div></div>
    <div id="e-body"><div class="empty"><span class="spinner"></span> Evaluating…</div></div>
  </div>`;
  const load = async () => {
    try {
      const r = await api("GET", "evaluate", { k: u.k });
      $("#e-body", el).innerHTML = evalBody(r);
    } catch (error) { $("#e-body", el).innerHTML = errorBox(error); }
  };
  $("#e-k", el).addEventListener("change", (e) => {
    const v = parseFloat(e.target.value);
    if (!(v > 0)) { toast("k must be a positive number."); e.target.value = u.k; return; }
    u.k = v; load();
  });
  $$("[data-k]", el).forEach((b) => b.addEventListener("click", () => { u.k = +b.dataset.k; $("#e-k", el).value = u.k; load(); }));
  load();
}

function metricsTable(results) {
  return `<div class="table-wrap"><table>
    <thead><tr><th>Model</th><th class="num">Tokens N</th><th class="num">NLL (nats)</th><th class="num">H (nats)</th><th class="num">H (bits)</th>
      <th class="num">Perplexity</th><th class="num">Zero-P tokens</th><th class="num">OOV rate</th></tr></thead>
    <tbody>${results.map((r) => `<tr><td>${esc(r.label)}</td><td class="num">${fmtN(r.num_tokens)}</td><td class="num">${fmtF(r.total_nll, 1)}</td>
      <td class="num">${fmtF(r.cross_entropy, 4)}</td><td class="num">${fmtF(r.cross_entropy_bits, 4)}</td>
      <td class="num"><b>${fmtF(r.perplexity, 2)}</b></td>
      <td class="num">${r.zero_prob_tokens ? `<span class="badge bad">${fmtN(r.zero_prob_tokens)}</span>` : "0"}</td><td class="num">${fmtPct(r.oov_rate)}</td></tr>`).join("")}</tbody>
  </table></div>`;
}

function evalBody(r) {
  const c = r.configuration;
  if (!r.has_test) {
    return `<div class="callout warn">The current model was trained with a test fraction of 0, so there is no held-out data to evaluate on.
      Retrain on the <a href="#corpus">Corpus &amp; Training</a> page with a test fraction above 0.</div>`;
  }
  const held = r.held_out;
  const finite = held.filter((x) => x.finite);
  const best = finite.length ? finite.reduce((a, b) => (b.perplexity < a.perplexity ? b : a)) : null;
  const chart = hbars(held.map((x) => ({
    label: x.label, value: x.perplexity ?? 0, inf: !x.finite,
    display: x.finite ? x.perplexity.toFixed(1) : "∞", note: x.finite ? "" : `${x.zero_prob_tokens} zero-P`,
    cls: best && x.label === best.label ? "" : "faint",
  })));
  const sweep = lineChart(r.sweep.map((s, i) => ({
    name: cap(s.name), color: SERIES[i % SERIES.length], points: s.points.map((p) => ({ x: p.k, y: p.perplexity })),
  })), { xLabel: "smoothing parameter k (log scale)", yLabel: "held-out perplexity" });
  const bestK = r.sweep.map((s) => {
    const b = s.points.reduce((a, p) => (p.perplexity < a.perplexity ? p : a));
    return `${cap(s.name)}: best k = ${b.k} (PP ${b.perplexity.toFixed(1)})`;
  }).join(" · ");
  const trainByLabel = Object.fromEntries(r.training.map((x) => [x.label, x]));
  const gapRows = held.filter((x) => x.k > 0 && !x.backoff).map((x) => ({ held: x, train: trainByLabel[x.label] })).filter((g) => g.train);

  return `<div class="stack">
    <div class="tiles">
      ${tile("Training data", `${fmtN(c.train_sentences)} sent.`, `${fmtN(c.train_words)} words`)}
      ${tile("Test data (held out)", `${fmtN(c.test_sentences)} sent.`, `${fmtN(c.test_words)} words`)}
      ${tile("Vocabulary V", fmtN(c.vocab_size), "from training split only")}
      ${tile("Evaluated tokens N", fmtN(held[0].num_tokens), "test words + one &lt;/s&gt; per sentence")}
      ${tile("OOV rate", fmtPct(held[0].oov_rate), `${fmtN(held[0].oov_tokens)} test words scored as &lt;unk&gt;`)}
      ${tile("Best held-out model", best ? best.label : "none finite", best ? `perplexity ${best.perplexity.toFixed(2)}` : "", true)}
    </div>
    <div class="grid grid-2">
      <div class="card"><h2>Held-out perplexity <span class="badge">lower is better</span></h2>
        ${chart}
        <p class="small muted" style="margin-top:10px">Hatched bars are infinite: an unsmoothed model gave at least one test token probability 0, so log P = −∞.
        Perplexity is never patched with an artificial floor.</p>
      </div>
      <div class="card"><h2>Perplexity as a function of k</h2>
        <div class="legend">${r.sweep.map((s, i) => `<span><i style="background:${SERIES[i % SERIES.length]}"></i>${cap(s.name)} + add-k</span>`).join("")}</div>
        ${sweep}
        <p class="small muted" style="margin-top:6px">${esc(bestK)}. Filled markers show each curve's minimum.</p>
      </div>
    </div>
    <div class="card"><h2>Held-out metrics</h2>${metricsTable(held)}
      <p class="small muted" style="margin-top:10px">H = −(1/N) Σ ln P(wᵢ | contextᵢ); perplexity = exp(H) = 2^(H in bits).
      "+ backoff" uses the longest context seen in training before applying add-k (an extension beyond plain add-k).</p>
    </div>
    <div class="card"><h2>How to read these results</h2>
      <ul style="margin:0;padding-left:20px" class="small">
        <li><b>Unsmoothed (MLE) models are infinite</b> on held-out text: every unseen n-gram, and every &lt;unk&gt; token (which never occurs in training when min count = 1), gets probability 0. This is the zero-probability problem smoothing solves.</li>
        <li><b>Add-1 is a blunt instrument.</b> With V = ${fmtN(c.vocab_size)} and few events per context, Laplace smoothing hands most of the mass to unseen words. On a small corpus that can make higher-order models look <i>worse</i> than the unigram model, because their contexts are sparser.</li>
        <li><b>Smaller k usually helps</b>: the sweep shows where each order's held-out perplexity bottoms out. A larger corpus, backoff or interpolation would let the trigram's longer context pay off.</li>
        <li>A perplexity of P means the model is, on average, as uncertain as if it were choosing uniformly between P words.</li>
      </ul>
    </div>
    <div class="card"><h2>Training-set metrics <span class="badge warn">not a quality measure</span></h2>
      <p class="card-sub">The same models scored on the sentences they were trained on. These are optimistic by construction — MLE is finite here because every training n-gram was seen.</p>
      ${metricsTable(r.training)}
      ${gapRows.length ? `<h3>Train vs. held-out perplexity (add-${c.k})</h3><div class="table-wrap"><table>
        <thead><tr><th>Model</th><th class="num">Training PP</th><th class="num">Held-out PP</th><th class="num">Ratio</th></tr></thead>
        <tbody>${gapRows.map((g) => `<tr><td>${esc(g.held.label)}</td><td class="num">${fmtF(g.train.perplexity, 2)}</td><td class="num">${fmtF(g.held.perplexity, 2)}</td>
          <td class="num">${g.train.perplexity && g.held.perplexity ? (g.held.perplexity / g.train.perplexity).toFixed(2) + "×" : "—"}</td></tr>`).join("")}</tbody></table></div>
        <p class="small muted" style="margin-top:8px">The gap grows with the order: higher-order models fit their training data more closely and generalise less.</p>` : ""}
    </div>
    <div class="card"><h2>Evaluation configuration</h2>${kvTable([
      ["Corpus", esc(c.corpus)],
      ["Split", `deterministic, test fraction ${c.test_fraction}, seed ${c.seed}; duplicate sentences always land on the same side`],
      ["Vocabulary", esc(c.vocabulary)],
      ["Minimum word count", String(c.min_count)],
      ["Predicted tokens", esc(c.predicted_tokens)],
      ["Logarithm", esc(c.log_base)],
      ["Smoothing k", String(c.k)],
    ])}</div>
  </div>`;
}

// ------------------------------------------------------------------ page 7: mathematics

const TOY_DEFAULT = {
  corpus_text: "I love Python\nI love machine learning\nI love AI",
  test_sentence: "I love machine learning",
  context_word: "love",
  target_word: "machine",
  k: 1,
};

function renderMath(el) {
  if (!ui.math) ui.math = { ...TOY_DEFAULT };
  const u = ui.math;
  const sections = ["Unigrams", "Bigrams", "Trigrams", "Conditional probability", "Maximum likelihood", "Laplace smoothing", "Cross-entropy", "Perplexity"];
  el.innerHTML = `<div class="stack">
    <div class="card">
      <h2>Toy corpus for the worked examples</h2>
      <p class="card-sub">Every number below is computed by the real engine on this toy corpus (one sentence per line). Edit it and watch the numbers change. The main model is not affected.</p>
      <div class="grid grid-side">
        <label class="field">Toy corpus<textarea id="m-corpus" rows="6" spellcheck="false">${esc(u.corpus_text)}</textarea></label>
        <div class="controls">
          <label class="field">Context word<input id="m-ctx" type="text" value="${esc(u.context_word)}"></label>
          <label class="field">Target word<input id="m-tgt" type="text" value="${esc(u.target_word)}"></label>
          <label class="field">k<input id="m-k" type="number" min="0.000001" step="any" value="${u.k}"></label>
          <label class="field" style="grid-column:1/-1">Sentence to score<input id="m-test" type="text" value="${esc(u.test_sentence)}"></label>
          <div class="row" style="grid-column:1/-1">
            <button class="btn ghost small" id="m-reset">Reset example</button>
            <button class="chip" data-try="I love machine translation">try: I love machine translation</button>
            <button class="chip" data-try="I love Python">try: I love Python</button>
          </div>
        </div>
      </div>
      <div class="toc" style="margin-top:14px">${sections.map((s, i) => `<a class="chip" href="#math" data-jump="m-sec-${i + 1}">${i + 1}. ${s}</a>`).join("")}</div>
    </div>
    <div id="m-body"><div class="empty"><span class="spinner"></span></div></div>
  </div>`;

  const run = latestOnly(() => api("POST", "learn", u));
  const load = async () => {
    try {
      const r = await run();
      if (r) $("#m-body", el).innerHTML = mathBody(r);
    } catch (error) { $("#m-body", el).innerHTML = errorBox(error); }
  };
  const later = debounce(load, 300);
  const bind = (id, key, parse = (v) => v) => $(id, el).addEventListener("input", (e) => { u[key] = parse(e.target.value); later(); });
  bind("#m-corpus", "corpus_text");
  bind("#m-ctx", "context_word");
  bind("#m-tgt", "target_word");
  bind("#m-test", "test_sentence");
  $("#m-k", el).addEventListener("change", (e) => {
    const v = parseFloat(e.target.value);
    if (!(v > 0)) { toast("k must be a positive number."); e.target.value = u.k; return; }
    u.k = v; load();
  });
  $("#m-reset", el).addEventListener("click", () => { ui.math = { ...TOY_DEFAULT }; renderMath(el); });
  $$("[data-try]", el).forEach((b) => b.addEventListener("click", () => { u.test_sentence = b.dataset.try; $("#m-test", el).value = u.test_sentence; load(); }));
  $$("[data-jump]", el).forEach((a) => a.addEventListener("click", (e) => {
    e.preventDefault(); document.getElementById(a.dataset.jump)?.scrollIntoView({ behavior: "smooth" });
  }));
  load();
}

function ngramTable(rows, total = null) {
  return `<div class="table-wrap scroll-y" style="max-height:300px"><table>
    <thead><tr><th>n-gram</th><th class="num">Count</th>${total ? `<th class="num">P = C / N</th>` : ""}</tr></thead>
    <tbody>${rows.map((g) => `<tr><td>${g.ngram.map((t, i) => tok(t, i === g.ngram.length - 1 && g.ngram.length > 1 ? "hl" : "")).join("")}</td>
      <td class="num">${g.count}</td>${total ? `<td class="num">${g.count} / ${total} = ${fmtP(g.count / total)}</td>` : ""}</tr>`).join("")}</tbody></table></div>`;
}

function section(n, title, body) {
  return `<div class="card math-section" id="m-sec-${n}"><h2><span class="sec-num">${n}</span>${esc(title)}</h2>${body}</div>`;
}

function scoreTable(s, label) {
  const res = s.result;
  return `<h3>${esc(label)}</h3><div class="table-wrap"><table>
    <thead><tr><th>Context</th><th>Word</th><th class="num">P(w | ctx)</th><th class="num">−ln P</th></tr></thead>
    <tbody>${s.rows.map((row) => `<tr><td>${toks(row.context)}</td><td>${tok(row.word, "hl")}</td><td class="num">${fmtP(row.probability)}</td>
      <td class="num">${row.neg_log === null ? `<span class="badge bad">∞</span>` : row.neg_log.toFixed(4)}</td></tr>`).join("")}
      <tr class="total"><td colspan="3">Σ −ln P (NLL) over N = ${res.num_tokens} tokens</td><td class="num">${fmtF(res.total_nll, 4)}</td></tr></tbody>
  </table></div>`;
}

function mathBody(r) {
  const f = r.focus, V = r.vocab_size, k = r.k, N = r.unigram_total;
  const ctx = f.context_label;
  const sumSmoothed = f.followers.reduce((a, x) => a + x.smoothed, 0) + f.unseen_types * f.unseen_each;
  const mle = r.scored.bigram_mle, sm = r.scored.bigram_smoothed;
  const padded = (rows) => rows.map((p) => `<div style="margin:3px 0">${p.map((t) => tok(t)).join("")}</div>`).join("");

  const s1 = section(1, "Unigrams", `
    <p>A unigram model ignores context: the probability of a word is its share of all training tokens.</p>
    <div class="formula">P(w) = C(w) / N        N = total predicted tokens = ${N}</div>
    <p class="small muted">N counts every word plus one &lt;/s&gt; per sentence (the model must also predict where a sentence ends). &lt;s&gt; is padding, never predicted.</p>
    <div class="grid grid-2">${ngramTable(r.unigrams, N)}
      <div><div class="formula">P(${esc(f.word)}) = C(${esc(f.word)}) / N
           = ${f.unigram_count} / ${N}
           = <span class="em">${fmtP(f.unigram_probability)}</span></div>
      <p class="small">Vocabulary (${r.vocab_words.length} words): ${r.vocab_words.map((w) => tok(w)).join("")}</p>
      <p class="small muted">V = ${r.vocab_words.length} words + &lt;unk&gt; + &lt;/s&gt; = <b>${V}</b> possible next tokens.</p></div>
    </div>`);

  const s2 = section(2, "Bigrams", `
    <p>Each sentence is padded with one &lt;s&gt; at the start and &lt;/s&gt; at the end, then every adjacent pair is counted.
    Sentences are padded <i>separately</i>, so no pair ever spans two sentences.</p>
    <div class="formula">P(wᵢ | wᵢ₋₁) = C(wᵢ₋₁, wᵢ) / C(wᵢ₋₁)</div>
    <div class="grid grid-2"><div><h3>Padded sentences</h3>${padded(r.padded.bigram)}</div><div><h3>Bigram counts</h3>${ngramTable(r.bigrams)}</div></div>`);

  const s3 = section(3, "Trigrams", `
    <p>A trigram model conditions on the previous two words, so sentences get two &lt;s&gt; tokens of padding.</p>
    <div class="formula">P(wᵢ | wᵢ₋₂, wᵢ₋₁) = C(wᵢ₋₂, wᵢ₋₁, wᵢ) / C(wᵢ₋₂, wᵢ₋₁)</div>
    <div class="grid grid-2"><div><h3>Padded sentences</h3>${padded(r.padded.trigram)}</div><div><h3>Trigram counts</h3>${ngramTable(r.trigrams)}</div></div>
    <p class="small muted" style="margin-top:10px">Longer contexts are more specific but rarer: the same corpus gives fewer repeated trigrams than bigrams.</p>`);

  const s4 = section(4, "Conditional probability", `
    <p>The probability of B given that A happened is the joint probability divided by the probability of A. With counts the
    common total cancels, leaving a ratio of counts.</p>
    <div class="formula">P(B | A) = P(A, B) / P(A)

P(${esc(f.word)} | ${esc(ctx)}) = [C(${esc(ctx)}, ${esc(f.word)}) / N₂] / [C(${esc(ctx)}) / N₂]
                     = C(${esc(ctx)}, ${esc(f.word)}) / C(${esc(ctx)})
                     = ${f.pair_count} / ${f.context_count} = <span class="em">${f.context_count ? fmtP(f.mle) : "undefined (0 / 0)"}</span></div>
    ${f.known ? "" : `<div class="callout warn small">"${esc(f.word)}" is not in the toy vocabulary, so it is looked up as &lt;unk&gt;.</div>`}
    <p class="small muted">C(${esc(ctx)}) counts how often "${esc(ctx)}" occurs as a history, i.e. the sum of the counts of every word that followed it.</p>`);

  const s5 = section(5, "Maximum likelihood estimation", `
    <p>For a fixed context with follower counts c<sub>w</sub> (total C), choose probabilities p<sub>w</sub> that maximise the likelihood of the training data:</p>
    <div class="formula">maximise   L(p) = Σ_w c_w · ln p_w      subject to   Σ_w p_w = 1
Lagrangian: ∂/∂p_w [ Σ c_w ln p_w − λ(Σ p_w − 1) ] = c_w / p_w − λ = 0   ⇒   p_w = c_w / λ
constraint: Σ c_w / λ = 1  ⇒  λ = C        ⇒   <span class="em">p_w = c_w / C</span></div>
    <p>So MLE <i>is</i> relative frequency. Next words after "${esc(ctx)}":</p>
    ${f.followers.length ? `<div class="table-wrap"><table><thead><tr><th>Word</th><th class="num">C(${esc(ctx)}, w)</th><th class="num">MLE</th></tr></thead>
      <tbody>${f.followers.map((x) => `<tr><td>${tok(x.word)}</td><td class="num">${x.count}</td><td class="num">${x.count} / ${f.context_count} = ${fmtP(x.mle)}</td></tr>`).join("")}
      <tr class="total"><td>Σ</td><td class="num">${f.context_count}</td><td class="num">${f.followers.reduce((a, x) => a + x.mle, 0).toFixed(6)}</td></tr></tbody></table></div>`
      : `<div class="callout bad">The context "${esc(ctx)}" never occurs in the toy corpus, so MLE is undefined here.</div>`}
    <p class="small muted" style="margin-top:8px">Any word that never followed "${esc(ctx)}" gets MLE probability exactly 0.</p>`);

  const s6 = section(6, "Laplace (add-k) smoothing", `
    <p>Pretend every one of the V possible next tokens was seen k extra times after every context. k = 1 is Laplace smoothing.</p>
    <div class="formula">P_k(w | ctx) = (C(ctx, w) + k) / (C(ctx) + k·V)

P_${k}(${esc(f.word)} | ${esc(ctx)}) = (${f.pair_count} + ${k}) / (${f.context_count} + ${k}·${V}) = <span class="em">${fmtP(f.smoothed)}</span>     (MLE: ${f.context_count ? fmtP(f.mle) : "undefined"})
unseen word        = (0 + ${k}) / (${f.context_count} + ${k}·${V}) = <span class="em">${fmtP(f.unseen_each)}</span>     (MLE: 0)</div>
    <div class="table-wrap"><table><thead><tr><th>Word</th><th class="num">Count</th><th class="num">MLE</th><th class="num">Add-${k}</th></tr></thead>
      <tbody>${f.followers.map((x) => `<tr><td>${tok(x.word)}</td><td class="num">${x.count}</td><td class="num">${fmtP(x.mle)}</td><td class="num">${fmtP(x.smoothed)}</td></tr>`).join("")}
      <tr><td class="muted">${f.unseen_types} unseen tokens, each</td><td class="num">0</td><td class="num">0</td><td class="num">${fmtP(f.unseen_each)}</td></tr>
      <tr class="total"><td>Σ over all V = ${V}</td><td class="num">${f.context_count}</td><td class="num">${f.followers.length ? "1.000000" : "—"}</td><td class="num">${sumSmoothed.toFixed(6)}</td></tr></tbody></table></div>
    <p class="small" style="margin-top:8px">Normalisation check: Σ_w (C(ctx,w) + k) = C(ctx) + k·V, which is exactly the denominator, so the smoothed values always sum to 1.</p>`);

  const s7 = section(7, "Cross-entropy", `
    <p>Score the sentence "${esc(r.test_sentence)}" with the bigram model: one probability per predicted token (each word, then &lt;/s&gt;).</p>
    <div class="formula">H = −(1/N) Σᵢ ln P(wᵢ | contextᵢ)        (nats; divide by ln 2 for bits)</div>
    ${r.test_sentence ? `<div class="callout warn small">This toy example scores against a model trained on the whole toy corpus. If the sentence is one of the training lines,
      this illustrates the arithmetic only. It is not a held-out evaluation (see Model Evaluation for that).</div>
    <div class="grid grid-2"><div>${scoreTable(mle, "Bigram, MLE")}</div><div>${scoreTable(sm, `Bigram, add-${k}`)}</div></div>
    <div class="formula">MLE:     H = ${fmtF(mle.result.total_nll, 4)} / ${mle.result.num_tokens} = <span class="em">${fmtF(mle.result.cross_entropy, 4)}</span> nats = ${fmtF(mle.result.cross_entropy_bits, 4)} bits
add-${k}:  H = ${fmtF(sm.result.total_nll, 4)} / ${sm.result.num_tokens} = <span class="em">${fmtF(sm.result.cross_entropy, 4)}</span> nats = ${fmtF(sm.result.cross_entropy_bits, 4)} bits</div>
    ${mle.result.finite ? "" : `<div class="callout bad small">The MLE model assigns probability 0 to ${mle.result.zero_prob_tokens} token(s), so ln 0 = −∞ and the cross-entropy is infinite. Smoothing fixes this.</div>`}`
    : `<div class="empty">Type a sentence to score.</div>`}`);

  const s8 = section(8, "Perplexity", `
    <p>Perplexity is the exponential of the cross-entropy: the effective number of equally likely choices the model is torn between at each step.</p>
    <div class="formula">PP = exp(H) = 2^(H in bits) = P(w₁ … w_N)^(−1/N)
${r.test_sentence ? `
MLE:     PP = exp(${fmtF(mle.result.cross_entropy, 4)}) = <span class="em">${fmtF(mle.result.perplexity, 3)}</span>
add-${k}:  PP = exp(${fmtF(sm.result.cross_entropy, 4)}) = <span class="em">${fmtF(sm.result.perplexity, 3)}</span>` : ""}

uniform model over V = ${V} tokens:  H = ln ${V},  PP = <span class="em">${V}</span></div>
    <p class="small">A model that knows nothing has perplexity V; a model that always predicts the right word with certainty has perplexity 1. Lower is better,
    but only when measured on text the model was not trained on, with the same vocabulary and tokenization.</p>`);

  return `<div class="stack">${s1}${s2}${s3}${s4}${s5}${s6}${s7}${s8}</div>`;
}

// ------------------------------------------------------------------ corpus & training

function renderCorpus(el) {
  const s = store.state, c = s.corpus;
  if (!ui.train) ui.train = { source: "sample", line_mode: "auto", ...s.config, file: null };
  const u = ui.train;
  el.innerHTML = `<div class="stack">
    <div class="tiles">
      ${tile("Loaded corpus", c.name, `source: ${esc(c.source)}`, true)}
      ${tile("Sentence layout", c.line_mode, c.line_mode === "line" ? "newlines end sentences" : "blank lines end paragraphs", true)}
      ${tile("Sentences", fmtN(c.sentences), `${fmtN(c.train_sentences)} train · ${fmtN(c.test_sentences)} test`)}
      ${tile("Words", fmtN(c.train_words + c.test_words), `${fmtN(c.train_words)} train · ${fmtN(c.test_words)} test`)}
      ${tile("Model", `v${s.version}`, `order ${s.config.max_order}, min count ${s.config.min_count}`, true)}
    </div>
    <div class="grid grid-2">
      <div class="card"><h2>Train a model</h2>
        <form id="t-form" class="stack" style="gap:12px" novalidate>
          <div class="seg" data-seg="t-src">
            <button type="button" data-value="1" class="${u.source === "sample" ? "on" : ""}">Sample corpus</button>
            <button type="button" data-value="2" class="${u.source === "upload" ? "on" : ""}">Upload a .txt file</button>
          </div>
          <div id="t-upload" ${u.source === "upload" ? "" : "hidden"}>
            <label class="field">Plain-text file (.txt, UTF-8, up to 8 MB)<input id="t-file" type="file" accept=".txt,text/plain"></label>
            <div class="small muted" id="t-file-info" style="margin-top:4px">${u.file ? `${esc(u.file.name)} · ${fmtN(u.file.size)} bytes` : "No file chosen."}</div>
          </div>
          <div class="controls">
            <label class="field">Sentence layout<select id="t-line">
              <option value="auto" ${u.line_mode === "auto" ? "selected" : ""}>Auto-detect</option>
              <option value="line" ${u.line_mode === "line" ? "selected" : ""}>One sentence per line</option>
              <option value="paragraph" ${u.line_mode === "paragraph" ? "selected" : ""}>Wrapped prose (paragraphs)</option></select></label>
            <label class="field">Max n-gram order<input id="t-order" type="number" min="1" max="5" value="${u.max_order}"></label>
            <label class="field">Min word count <span class="hint">(rarer → &lt;unk&gt;)</span><input id="t-min" type="number" min="1" value="${u.min_count}"></label>
            <label class="field">Test fraction<input id="t-frac" type="number" min="0" max="0.9" step="0.05" value="${u.test_fraction}"></label>
            <label class="field">Split seed<input id="t-seed" type="number" min="0" value="${u.seed}"></label>
          </div>
          <div class="row"><button class="btn" id="t-go" type="submit">Train model</button><span id="t-status" class="small muted"></span></div>
        </form>
      </div>
      <div class="card"><h2>How the data is handled</h2>
        <ul class="small" style="margin:0;padding-left:20px">
          <li>The file is checked to be non-empty plain text; control characters are removed and line endings normalised.</li>
          <li>Text is split into sentences. "One sentence per line" treats each line as a sentence; "wrapped prose" joins lines within a paragraph first (needed for books such as Project Gutenberg texts).</li>
          <li>The split is <b>deterministic</b> (seeded) and made at sentence level. Sentences with identical tokens are grouped first, so a test sentence can never also appear in training.</li>
          <li>The vocabulary and all counts come from the training split only. Test words outside it are scored as &lt;unk&gt;.</li>
          <li>Training replaces the active model in one step; every page then reads from the new model (the version number in the sidebar changes).</li>
        </ul>
        <h3>Using a larger corpus</h3>
        <p class="small">Download any public-domain English text (for example a book from Project Gutenberg), save it as <code>.txt</code>, and upload it here
        with "Wrapped prose", or start the app with <code>python app.py --corpus path/to/book.txt</code>. Nothing is downloaded automatically.</p>
      </div>
    </div>
    <div class="card"><h2>Training sentences (first ${c.preview.length})</h2>
      <ol class="small" style="margin:0;padding-left:20px">${c.preview.map((p) => `<li>${esc(p)}</li>`).join("")}</ol></div>
  </div>`;

  bindSeg(el, "t-src", (n) => {
    u.source = n === 1 ? "sample" : "upload";
    $("#t-upload", el).hidden = u.source !== "upload";
  });
  $("#t-file", el).addEventListener("change", (e) => {
    u.file = e.target.files[0] || null;
    $("#t-file-info", el).textContent = u.file ? `${u.file.name} · ${fmtN(u.file.size)} bytes` : "No file chosen.";
  });

  $("#t-form", el).addEventListener("submit", async (event) => {
    event.preventDefault();
    const status = $("#t-status", el), button = $("#t-go", el);
    const intIn = (id, lo, hi) => {
      const v = Number($(id, el).value);
      if (!Number.isInteger(v) || v < lo || v > hi) throw new Error(`${$(id, el).closest("label").firstChild.textContent.trim()} must be a whole number between ${lo} and ${hi}.`);
      return v;
    };
    let payload;
    try {
      const frac = Number($("#t-frac", el).value);
      if (!(frac >= 0 && frac <= 0.9)) throw new Error("Test fraction must be between 0 and 0.9.");
      payload = {
        source: u.source, line_mode: $("#t-line", el).value,
        max_order: intIn("#t-order", 1, 5), min_count: intIn("#t-min", 1, 1000),
        test_fraction: frac, seed: intIn("#t-seed", 0, 2 ** 31 - 1),
      };
      Object.assign(u, payload);
      if (u.source === "upload") {
        if (!u.file) throw new Error("Choose a .txt file to upload first.");
        if (!/\.txt$/i.test(u.file.name)) throw new Error("Only plain-text .txt files are supported.");
        if (u.file.size === 0) throw new Error("The file is empty.");
        if (u.file.size > 8 * 1024 * 1024) throw new Error("The file is larger than 8 MB.");
        payload.filename = u.file.name;
        payload.content_base64 = await fileToBase64(u.file);
      }
    } catch (error) { toast(error.message); return; }

    button.disabled = true;
    status.innerHTML = `<span class="spinner"></span> Training…`;
    try {
      store.state = await api("POST", "train", payload);
      renderModelCard();
      clampOrders();
      toast(`Trained model v${store.state.version} on "${store.state.corpus.name}".`, true);
      renderCorpus(el);
    } catch (error) {
      status.textContent = "";
      toast(`Training failed: ${error.message}`);
    } finally {
      button.disabled = false;
    }
  });
}

function fileToBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result).split(",", 2)[1] || "");
    reader.onerror = () => reject(new Error("The file could not be read."));
    reader.readAsDataURL(file);
  });
}

// ------------------------------------------------------------------ router

const PAGES = [
  { id: "overview", num: "1", title: "Overview", eyebrow: "Page 1 · Project and model", render: renderOverview },
  { id: "predict", num: "2", title: "Live Prediction Studio", eyebrow: "Page 2 · Next-word suggestions", render: renderPredict },
  { id: "explorer", num: "3", title: "N-gram Explorer", eyebrow: "Page 3 · Raw counts", render: renderExplorer },
  { id: "probability", num: "4", title: "Probability Laboratory", eyebrow: "Page 4 · MLE and smoothing", render: renderProbability },
  { id: "generate", num: "5", title: "Text Generation Lab", eyebrow: "Page 5 · Sampling", render: renderGenerate },
  { id: "evaluate", num: "6", title: "Model Evaluation", eyebrow: "Page 6 · Held-out metrics", render: renderEvaluate },
  { id: "math", num: "7", title: "Learn the Mathematics", eyebrow: "Page 7 · Worked examples", render: renderMath },
  { id: "corpus", num: "◆", title: "Corpus & Training", eyebrow: "Data · Retrain the model", render: renderCorpus, sep: true },
];

function renderNav(active) {
  $("#nav").innerHTML = PAGES.map((p) => `${p.sep ? `<div class="sep"></div>` : ""}
    <a href="#${p.id}" class="${p.id === active ? "active" : ""}" ${p.id === active ? 'aria-current="page"' : ""}>
      <span class="num">${p.num}</span>${esc(p.title)}</a>`).join("");
}

function route() {
  const id = location.hash.replace("#", "") || "overview";
  const page = PAGES.find((p) => p.id === id) || PAGES[0];
  renderNav(page.id);
  $("#page-title").textContent = page.title;
  $("#page-eyebrow").textContent = page.eyebrow;
  document.title = `${page.title} · WordFlow`;
  $("#sidebar").classList.remove("open");
  const el = $("#page");
  try {
    page.render(el);
  } catch (error) {
    el.innerHTML = errorBox(error);
    console.error(error);
  }
}

async function boot() {
  $("#menu-btn").addEventListener("click", () => $("#sidebar").classList.toggle("open"));
  window.addEventListener("hashchange", () => { route(); window.scrollTo(0, 0); });
  document.addEventListener("keydown", (e) => {
    const typing = e.target.closest && e.target.closest("input, textarea, select, [contenteditable]");
    if (typing || e.ctrlKey || e.metaKey || e.altKey) return;
    if (/^[1-7]$/.test(e.key)) {
      location.hash = PAGES[+e.key - 1].id;
    } else if (e.key === "/") {
      const input = $("#page textarea, #page input[type=text]");
      if (input) { e.preventDefault(); input.focus(); }
    }
  });
  try {
    await refreshState();
    ui.predict.order = ui.gen.order = defaultOrder();
    clampOrders();
    route();
  } catch (error) {
    $("#page").innerHTML = errorBox(error);
    $("#model-card").textContent = "No model loaded.";
  }
}

boot();
