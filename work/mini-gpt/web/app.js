"use strict";
/* mini-GPT Playground — vanilla JS, no dependencies. */

const $ = (sel, el = document) => el.querySelector(sel);
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const vis = (s) => s.replace(/\n/g, "↵").replace(/ /g, "·");          // make spaces visible in labels
const pct = (p) => (p >= 0.995 ? (p * 100).toFixed(0) : p >= 0.1 ? (p * 100).toFixed(1) : (p * 100).toFixed(2)) + "%";
let INFO = null;

async function api(path, body) {
  const res = await fetch(path, body === undefined ? {} : {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || res.statusText);
  return data;
}
function toast(msg) {
  const t = $("#toast"); t.textContent = msg; t.hidden = false;
  clearTimeout(toast.t); toast.t = setTimeout(() => (t.hidden = true), 3500);
}
function debounce(fn, ms) { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; }
// Only the latest call's result is applied (typing fast must not show stale results).
function latest(fn) { let n = 0; return async (...a) => { const me = ++n; const r = await fn(...a); return me === n ? r : null; }; }

/* ---------- tooltip ---------- */
const tip = document.createElement("div"); tip.className = "tip"; tip.hidden = true; document.body.appendChild(tip);
function showTip(e, html) { tip.innerHTML = html; tip.hidden = false; moveTip(e); }
function moveTip(e) { tip.style.left = Math.min(e.clientX + 14, innerWidth - 300) + "px"; tip.style.top = e.clientY + 16 + "px"; }
function hideTip() { tip.hidden = true; }

/* ---------- colour for a probability: red = surprising, green = confident ---------- */
function probColor(p) {
  const s = Math.max(0, Math.min(1, 1 + Math.log10(Math.max(p, 1e-4)) / 3)); // 1e-3..1 -> 0..1
  const hue = 10 + s * 140;
  return `hsla(${hue}, 75%, 45%, ${0.42 - s * 0.3})`;
}

/* ---------- shared sampling controls ---------- */
const S = { strategy: "top_p", temperature: 0.8, top_k: 40, top_p: 0.9, repetition_penalty: 1.0, max_new_tokens: 120, seed: "" };
function controlsHTML(withLength = true) {
  const strat = ["greedy", "temperature", "top_k", "top_p"].map((s) =>
    `<button data-s="${s}" class="${S.strategy === s ? "on" : ""}">${s.replace("_", "-")}</button>`).join("");
  const r = (key, label, min, max, step) =>
    `<div class="ctl" data-ctl="${key}"><label>${label} <span class="val">${S[key]}</span></label>
     <input type="range" min="${min}" max="${max}" step="${step}" value="${S[key]}" data-k="${key}"></div>`;
  return `<div class="row"><label>Decoding</label><div class="seg" id="strat">${strat}</div></div>
  <div class="controls">
    ${r("temperature", "Temperature", 0.1, 2, 0.05)}
    ${r("top_k", "Top-k", 1, 100, 1)}
    ${r("top_p", "Top-p (nucleus)", 0.05, 1, 0.01)}
    ${r("repetition_penalty", "Repetition penalty", 1, 2, 0.05)}
    ${withLength ? r("max_new_tokens", "Max new tokens", 10, 300, 10) : ""}
    <div class="ctl"><label>Seed (blank = random)</label><input type="number" id="seed" value="${S.seed}" placeholder="random"></div>
  </div>`;
}
function wireControls(root, onChange) {
  const sync = () => {
    root.querySelectorAll(".ctl[data-ctl]").forEach((c) => {
      const k = c.dataset.ctl;
      const off = (S.strategy === "greedy" && ["temperature", "top_k", "top_p"].includes(k)) ||
        (S.strategy === "temperature" && ["top_k", "top_p"].includes(k)) ||
        (S.strategy === "top_k" && k === "top_p") || (S.strategy === "top_p" && k === "top_k");
      c.classList.toggle("off", off);
    });
  };
  root.querySelectorAll("#strat button").forEach((b) => b.onclick = () => {
    S.strategy = b.dataset.s;
    root.querySelectorAll("#strat button").forEach((x) => x.classList.toggle("on", x === b));
    sync(); onChange && onChange();
  });
  root.querySelectorAll("input[type=range]").forEach((inp) => inp.oninput = () => {
    S[inp.dataset.k] = Number(inp.value);
    inp.previousElementSibling.querySelector(".val").textContent = inp.value;
    onChange && onChange();
  });
  const seed = $("#seed", root); if (seed) seed.oninput = () => { S.seed = seed.value; onChange && onChange(); };
  sync();
}
const samplingBody = () => ({ strategy: S.strategy, temperature: S.temperature, top_k: S.top_k, top_p: S.top_p,
  repetition_penalty: S.repetition_penalty, seed: S.seed === "" ? null : Number(S.seed) });

/* =====================================================================  GENERATE */
const PROMPTS = ["Once upon a time, a little mouse named Pip", "Tom had a red ball.", "One day, Lily and Max were playing outside.",
  "Luna the curious cat wanted to make a gift for her mom.", "The bear said,"];
let lastGen = null;

function pageGenerate(root) {
  root.innerHTML = `
  <h1>Make it talk</h1>
  <p class="lead">Type a prompt and the model continues it <b>one token at a time</b>. Each token is coloured by how
   likely the model thought it was. Hover a token to see the alternatives it was choosing between.</p>
  <div class="grid" style="grid-template-columns: minmax(0,1fr)">
   <div class="card">
    <textarea id="prompt" rows="2">${esc(PROMPTS[0])}</textarea>
    <div class="row" style="margin-top:8px">${PROMPTS.map((p, i) => `<button class="btn ghost" data-p="${i}" style="padding:4px 10px;font-size:12px">${esc(p.slice(0, 28))}…</button>`).join("")}</div>
    ${controlsHTML(true)}
    <div class="row"><button class="btn" id="go">▶ Generate</button>
     <button class="btn ghost" id="cmp">Compare all 4 strategies</button>
     <span class="hint"><kbd>Ctrl</kbd>+<kbd>Enter</kbd> to generate</span></div>
   </div>
   <div class="card"><h3>Output</h3><div id="out" class="output"></div>
    <div class="legend">surprising <div class="bar"></div> confident</div>
    <div id="gstats" class="stats"></div></div>
   <div id="cmpbox"></div>
  </div>`;
  wireControls(root);
  root.querySelectorAll("[data-p]").forEach((b) => b.onclick = () => { $("#prompt").value = PROMPTS[b.dataset.p]; run(); });
  $("#prompt").onkeydown = (e) => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); run(); } };
  $("#go").onclick = run;
  $("#cmp").onclick = compare;
  if (lastGen) render(lastGen.prompt, lastGen.res, false);

  async function run() {
    const prompt = $("#prompt").value;
    $("#go").disabled = true;
    try {
      const res = await api("/api/generate", { prompt, max_new_tokens: S.max_new_tokens, ...samplingBody() });
      lastGen = { prompt, res };
      await render(prompt, res, true);
    } catch (e) { toast(e.message); } finally { $("#go").disabled = false; }
  }
  async function render(prompt, res, animate) {
    const out = $("#out");
    out.innerHTML = `<span class="prompt">${esc(prompt)}</span>`;
    const cur = document.createElement("span"); cur.className = "cursor"; out.appendChild(cur);
    for (const s of res.steps) {
      if (s.piece === "⏎END") break;
      const span = document.createElement("span");
      span.className = "tk"; span.textContent = s.piece; span.style.background = probColor(s.p_model);
      span.onmousemove = (e) => showTip(e,
        `<b>${esc(vis(s.piece))}</b> &nbsp; model p = ${pct(s.p_model)}<br>sampled from p = ${pct(s.p)}<hr style="border:0;border-top:1px solid #888">` +
        s.candidates.map(([c, q]) => `${esc(vis(c)).padEnd(14, " ")} ${pct(q)}`).join("<br>"));
      span.onmouseleave = hideTip;
      out.insertBefore(span, cur);
      if (animate) await new Promise((r) => setTimeout(r, 18));
    }
    cur.remove();
    const ended = res.steps.length && res.steps[res.steps.length - 1].piece === "⏎END";
    const ppl = res.avg_nll == null ? "–" : Math.exp(res.avg_nll).toFixed(2);
    $("#gstats").innerHTML = [
      [res.prompt_tokens, "prompt tokens"], [res.steps.length - (ended ? 1 : 0), "generated tokens"],
      [ppl, "perplexity of own output"], [ended ? "yes" : "no", "chose to stop (<|endoftext|>)"],
    ].map(([b, s]) => `<div class="stat"><b>${b}</b><span>${s}</span></div>`).join("");
  }
  async function compare() {
    const prompt = $("#prompt").value; const box = $("#cmpbox");
    box.innerHTML = `<div class="card">Running greedy, temperature, top-k and top-p…</div>`;
    try {
      const strategies = ["greedy", "temperature", "top_k", "top_p"];
      const results = [];
      for (const st of strategies) // sequential: the CPU server handles one request at a time anyway
        results.push(await api("/api/generate", { prompt, max_new_tokens: S.max_new_tokens, ...samplingBody(), strategy: st }));
      box.innerHTML = `<div class="grid">${results.map((r, i) => `<div class="card"><h3>${strategies[i].replace("_", "-")}
        <span class="hint">· ppl ${r.avg_nll == null ? "–" : Math.exp(r.avg_nll).toFixed(2)}</span></h3>
        <div class="output" style="min-height:0;font-size:14px"><span class="prompt">${esc(prompt)}</span>${esc(r.text)}</div></div>`).join("")}</div>
        <p class="hint">Greedy always picks the top token, so it is deterministic and tends to repeat itself. Sampling adds variety;
        top-k / top-p cut off the unreliable tail of the distribution.</p>`;
    } catch (e) { toast(e.message); box.innerHTML = ""; }
  }
}

/* =====================================================================  NEXT TOKEN */
let nextText = "One day, Ben";
function pageNext(root) {
  root.innerHTML = `
  <h1>What comes next?</h1>
  <p class="lead">This is everything a language model does: given the context, output a probability for <b>every</b> token
   in the vocabulary. Type, and watch the distribution change live. <b>Click a bar</b> (or press <kbd>Tab</kbd>) to accept a token
   and keep building the sentence yourself.</p>
  <div class="grid" style="grid-template-columns: minmax(0,1.2fr) minmax(0,1fr)">
   <div class="card"><textarea id="ctx" rows="3">${esc(nextText)}</textarea>
     ${controlsHTML(false)}
     <div id="nstats" class="stats"></div></div>
   <div class="card"><h3>Next-token distribution <span class="hint">· dark = what decoding samples from, light = raw model</span></h3>
     <div id="bars" class="bars"></div></div>
  </div>`;
  const ctx = $("#ctx");
  const fetchNext = latest((body) => api("/api/next", body));
  let top = null;
  const update = debounce(async () => {
    nextText = ctx.value;
    if (!ctx.value.trim()) { $("#bars").innerHTML = `<p class="hint">Type something.</p>`; return; }
    try {
      const r = await fetchNext({ text: ctx.value, k: 15, ...samplingBody() });
      if (!r) return;
      top = r.candidates[0];
      const max = r.candidates[0].p;
      $("#bars").innerHTML = r.candidates.map((c) => `
        <div class="barrow ${c.p_sampled === 0 ? "cut" : ""}" data-piece="${esc(c.piece)}" title="logit ${c.logit.toFixed(2)}">
          <span class="lbl">${esc(vis(c.piece))}</span>
          <span class="track"><span class="fill2" style="width:${(c.p / max) * 100}%"></span>
            <span class="fill" style="width:${(c.p_sampled / Math.max(max, ...r.candidates.map((x) => x.p_sampled))) * 100}%"></span></span>
          <span class="num">${pct(c.p)}</span></div>`).join("");
      $("#bars").querySelectorAll(".barrow").forEach((b) => b.onclick = () => accept(b.dataset.piece));
      $("#nstats").innerHTML = [
        [r.context_tokens, "context tokens"], [r.entropy_bits.toFixed(2), "entropy (bits)"],
        [(2 ** r.entropy_bits).toFixed(1), "effective choices"], [r.kept, "tokens kept by decoding"],
      ].map(([b, s]) => `<div class="stat"><b>${b}</b><span>${s}</span></div>`).join("");
    } catch (e) { toast(e.message); }
  }, 120);
  function accept(piece) {
    if (piece === "⏎END") { toast("The model thinks the story ends here."); return; }
    ctx.value += piece; update(); ctx.focus();
  }
  ctx.oninput = update;
  ctx.onkeydown = (e) => { if (e.key === "Tab" && top) { e.preventDefault(); accept(top.piece); } };
  wireControls(root, update);
  update();
}

/* =====================================================================  ATTENTION */
let attText = "Lily had a shiny key. She loved the key very much.";
function pageAttention(root) {
  const L = INFO.config.n_layer, H = INFO.config.n_head;
  root.innerHTML = `
  <h1>Look back, never forward</h1>
  <p class="lead">Every head in every layer decides how much each token should <b>look at</b> each earlier token.
   The upper triangle is always empty: that is the causal mask. Pick a head, then click a word to see what it attends to.</p>
  <div class="card"><div class="row"><input type="text" id="atxt" value="${esc(attText)}" style="flex:1">
   <button class="btn" id="ago">Analyse</button></div>
   <p class="hint">Try a pronoun: in the trained checkpoint, layer 1 · head 2 sends <code>She</code>'s strongest attention back to <code>L</code>, the first token of <code>Lily</code>. Click <code>She</code> to see it.</p></div>
  <div class="grid" style="margin-top:16px;grid-template-columns:minmax(0,1fr) minmax(0,1.3fr)">
   <div class="card"><h3>All ${L * H} heads <span class="hint">(${L} layers × ${H} heads)</span></h3><div id="heads" class="heads"></div></div>
   <div class="card"><h3 id="htitle"></h3><div id="focus" class="focus"></div><div id="heat" style="margin-top:14px"></div></div>
  </div>`;
  let data = null, sel = [0, Math.min(1, H - 1)], q = null;
  const go = async () => {
    attText = $("#atxt").value;
    try { data = await api("/api/attention", { text: attText }); q = data.pieces.length - 1; draw(); } catch (e) { toast(e.message); }
  };
  $("#ago").onclick = go; $("#atxt").onkeydown = (e) => e.key === "Enter" && go();
  function draw() {
    const heads = $("#heads"); heads.innerHTML = "";
    data.layers.forEach((layer, l) => layer.forEach((m, h) => {
      const d = document.createElement("div"); d.className = "mini" + (l === sel[0] && h === sel[1] ? " sel" : "");
      const c = document.createElement("canvas"); const n = m.length; c.width = n; c.height = n;
      const g = c.getContext("2d"); const img = g.createImageData(n, n);
      m.forEach((row, i) => row.forEach((v, j) => { const k = (i * n + j) * 4; const a = Math.sqrt(v);
        img.data[k] = 91; img.data[k + 1] = 91; img.data[k + 2] = 214; img.data[k + 3] = j > i ? 0 : 25 + a * 230; }));
      g.putImageData(img, 0, 0);
      d.appendChild(c); d.insertAdjacentHTML("beforeend", `<div>layer ${l + 1} · head ${h + 1}</div>`);
      d.onclick = () => { sel = [l, h]; draw(); };
      heads.appendChild(d);
    }));
    const m = data.layers[sel[0]][sel[1]];
    $("#htitle").textContent = `Layer ${sel[0] + 1}, head ${sel[1] + 1} — token “${vis(data.pieces[q])}” attends to:`;
    const f = $("#focus"); f.innerHTML = "";
    data.pieces.forEach((p, j) => {
      const s = document.createElement("span"); s.textContent = p;
      const w = j <= q ? m[q][j] : 0;
      s.style.background = j <= q ? `rgba(91,91,214,${Math.min(0.9, Math.sqrt(w))})` : "transparent";
      if (j > q) s.style.opacity = 0.3;
      if (j === q) s.className = "me";
      s.onmousemove = (e) => showTip(e, j <= q ? `attention ${pct(w)}` : "future token: masked (−∞ → 0)");
      s.onmouseleave = hideTip;
      s.onclick = () => { q = j; draw(); };
      f.appendChild(s);
    });
    // full matrix
    const n = data.pieces.length, cell = n > 28 ? 14 : 22;
    let html = `<div class="heat" style="grid-template-columns: 90px repeat(${n}, ${cell}px)"><div></div>`;
    html += data.pieces.map((p) => `<div class="colh">${esc(vis(p))}</div>`).join("");
    m.forEach((row, i) => {
      html += `<div class="lab">${esc(vis(data.pieces[i]))}</div>`;
      html += row.map((v, j) => `<div class="c" data-i="${i}" data-j="${j}" style="height:${cell}px;background:${j > i ? "repeating-linear-gradient(45deg,transparent 0 3px,rgba(128,128,128,.15) 3px 5px)" : `rgba(91,91,214,${Math.sqrt(v)})`}"></div>`).join("");
    });
    $("#heat").innerHTML = html + "</div><p class='hint'>Rows = the token doing the looking (query), columns = what it looks at (key). Hatched = masked future.</p>";
    $("#heat").querySelectorAll(".c").forEach((c) => {
      const i = +c.dataset.i, j = +c.dataset.j;
      c.onmousemove = (e) => showTip(e, `“${esc(vis(data.pieces[i]))}” → “${esc(vis(data.pieces[j]))}”: ${j > i ? "masked" : pct(m[i][j])}`);
      c.onmouseleave = hideTip; c.onclick = () => { q = i; draw(); };
    });
  }
  go();
}

/* =====================================================================  TOKENIZER */
let tokText = "Once upon a time, a curious cat named Luna found a shiny key. Unbelievably, ünïcödé and emoji 🙂 still work!";
function pageTokenizer(root) {
  root.innerHTML = `
  <h1>Text → tokens</h1>
  <p class="lead">The model never sees letters. A byte-level <b>BPE</b> tokenizer, trained on the corpus, cuts text into
   frequent chunks. Common words become a single token; rare words fall apart into pieces. Any text, even emoji, can be encoded,
   because the 256 raw bytes are always in the vocabulary.</p>
  <div class="card"><textarea id="ttxt" rows="3">${esc(tokText)}</textarea><div id="tstats" class="stats"></div>
   <div id="pieces" class="pieces" style="margin-top:14px"></div></div>
  <div class="card" style="margin-top:16px"><h3>The first merges it learned</h3>
   <p class="hint">Training repeatedly fuses the most frequent adjacent pair. ${INFO.merges} merges + 256 bytes + 1 special token = ${INFO.vocab_size} tokens.</p>
   <div class="pieces">${INFO.sample_merges.map(([a, b, n], i) => `<span class="piece" style="background:var(--bg)"><small>#${i + 1}</small>${esc(vis(a))} + ${esc(vis(b))} → <b>${esc(vis(n))}</b></span>`).join("")}</div></div>`;
  const colors = ["#5b5bd633", "#12805c33", "#b5470833", "#c4320a2b", "#0e749033", "#7c3aed2b"];
  const fetchTok = latest((b) => api("/api/tokenize", b));
  const upd = debounce(async () => {
    tokText = $("#ttxt").value;
    try {
      const r = await fetchTok({ text: tokText }); if (!r) return;
      $("#pieces").innerHTML = r.pieces.map((p, i) => `<span class="piece" style="background:${colors[i % colors.length]}">${esc(vis(p))}<small>${r.ids[i]}</small></span>`).join("");
      $("#tstats").innerHTML = [[r.chars, "characters"], [r.bytes, "UTF-8 bytes"], [r.ids.length, "tokens"],
        [r.ids.length ? (r.chars / r.ids.length).toFixed(2) : "–", "chars per token"]]
        .map(([b, s]) => `<div class="stat"><b>${b}</b><span>${s}</span></div>`).join("");
    } catch (e) { toast(e.message); }
  }, 80);
  $("#ttxt").oninput = upd; upd();
}

/* =====================================================================  TRAINING */
function lineChart(series, { w = 760, h = 300, ylab = "", refs = [] } = {}) {
  const pad = { l: 46, r: 12, t: 12, b: 30 };
  const xs = series.flatMap((s) => s.x), ys = series.flatMap((s) => s.y).concat(refs.map((r) => r.y));
  const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = 0, y1 = Math.max(...ys) * 1.05;
  const X = (v) => pad.l + ((v - x0) / (x1 - x0 || 1)) * (w - pad.l - pad.r);
  const Y = (v) => h - pad.b - ((v - y0) / (y1 - y0 || 1)) * (h - pad.t - pad.b);
  let g = "";
  for (let i = 0; i <= 5; i++) { const v = y0 + (i / 5) * (y1 - y0);
    g += `<line x1="${pad.l}" x2="${w - pad.r}" y1="${Y(v)}" y2="${Y(v)}" stroke="currentColor" opacity=".1"/><text x="${pad.l - 6}" y="${Y(v) + 4}" text-anchor="end">${v.toFixed(1)}</text>`; }
  for (let i = 0; i <= 5; i++) { const v = x0 + (i / 5) * (x1 - x0); g += `<text x="${X(v)}" y="${h - 10}" text-anchor="middle">${Math.round(v)}</text>`; }
  refs.forEach((r) => g += `<line x1="${pad.l}" x2="${w - pad.r}" y1="${Y(r.y)}" y2="${Y(r.y)}" stroke="${r.c}" stroke-dasharray="5 4"/><text x="${w - pad.r - 4}" y="${Y(r.y) - 5}" text-anchor="end">${r.label} ${r.y.toFixed(2)}</text>`);
  series.forEach((s) => {
    const d = s.x.map((x, i) => `${i ? "L" : "M"}${X(x).toFixed(1)},${Y(s.y[i]).toFixed(1)}`).join("");
    g += `<path d="${d}" fill="none" stroke="${s.c}" stroke-width="${s.wd || 2}" opacity="${s.op || 1}"/>`;
  });
  return `<svg viewBox="0 0 ${w} ${h}" style="width:100%;height:auto;color:var(--ink)">${g}<text x="12" y="${h / 2}" transform="rotate(-90 12 ${h / 2})" text-anchor="middle">${ylab}</text></svg>`;
}
function pageTraining(root) {
  const h = INFO.history, m = INFO.meta, b = m.baselines || {};
  const best = m.best_val_loss ?? Math.min(...h.val_loss);
  const bl = h.batch_loss || [];
  const smooth = bl.map((_, i) => { const a = bl.slice(Math.max(0, i - 24), i + 1); return a.reduce((x, y) => x + y, 0) / a.length; });
  root.innerHTML = `
  <h1>Watch the loss fall</h1>
  <p class="lead">Training minimises next-token <b>cross-entropy</b>: −log p(correct next token), averaged over every position.
   Validation text was split off <i>before</i> the tokenizer and model saw anything, so the val curve measures generalisation.</p>
  <div class="stats" style="margin-bottom:16px">
   ${[[b.uniform?.toFixed(3), "random guess: ln(V)"], [b.unigram?.toFixed(3), "unigram baseline"], [best.toFixed(3), "best val loss (nats)"],
      [Math.exp(best).toFixed(2), "val perplexity"], [m.step, "steps trained"], [m.elapsed_s ? Math.round(m.elapsed_s) + "s" : "–", "CPU time"],
      [(m.data?.train_tokens || 0).toLocaleString(), "training tokens"]]
     .map(([v, s]) => `<div class="stat"><b>${v}</b><span>${s}</span></div>`).join("")}</div>
  <div class="card"><h3>Loss (nats per token)</h3>
   ${lineChart([
     { x: bl.map((_, i) => i), y: smooth, c: "var(--muted)", wd: 1.5, op: 0.6 },
     { x: h.step, y: h.train_loss, c: "#5b5bd6" }, { x: h.step, y: h.val_loss, c: "#12805c" },
   ], { ylab: "loss", refs: [{ y: b.uniform || 0, c: "#c4320a", label: "uniform" }, { y: b.unigram || 0, c: "#b54708", label: "unigram" }] })}
   <div class="legend"><span style="color:#5b5bd6">━ train</span><span style="color:#12805c">━ validation</span><span>━ batch loss (smoothed)</span>
   <span style="color:#c4320a">┅ uniform</span><span style="color:#b54708">┅ unigram</span></div></div>
  <div class="grid" style="margin-top:16px">
   <div class="card"><h3>Learning-rate schedule</h3>${lineChart([{ x: h.step, y: h.lr.map((v) => v * 1000), c: "#5b5bd6" }], { h: 200, ylab: "lr × 1000" })}
    <p class="hint">Linear warm-up, then cosine decay. AdamW, weight decay on matrices only, gradient clipping at 1.0.</p></div>
   <div class="card"><h3>Checkpoints</h3><table>${h.step.map((s, i) => `<tr><td>step ${s}</td><td class="n">${h.train_loss[i].toFixed(3)}</td><td class="n">${h.val_loss[i].toFixed(3)}</td><td class="n">${Math.exp(h.val_loss[i]).toFixed(2)}</td></tr>`).join("")}
    </table><p class="hint">columns: train loss · val loss · val perplexity</p></div></div>`;
  $("table", root).insertAdjacentHTML("afterbegin", "<tr><th>checkpoint</th><th>train</th><th>val</th><th>ppl</th></tr>");
}

/* =====================================================================  ARCHITECTURE */
function pageArchitecture(root) {
  const c = INFO.config;
  const groups = Object.entries(INFO.param_groups);
  root.innerHTML = `
  <h1>Inside the model</h1>
  <p class="lead">A decoder-only transformer, the same shape as GPT-2, just small enough to train on a laptop CPU in minutes.
   Every piece below is hand-written in <code>minigpt/model.py</code>.</p>
  <div class="grid">
   <div class="card arch">
    <div class="box">token ids <code>(B, T≤${c.block_size})</code></div><div class="arrow">↓</div>
    <div class="box">token embedding (${c.vocab_size} × ${c.n_embd}) <b>+</b> ${c.pos_encoding} positional encoding</div><div class="arrow">↓</div>
    <div class="box blk"><b>× ${c.n_layer} transformer blocks</b><br>x = x + MultiHeadAttention(LayerNorm(x)) — ${c.n_head} heads × ${c.n_embd / c.n_head} dims, causal mask<br>
      x = x + MLP(LayerNorm(x)) — ${c.n_embd} → ${4 * c.n_embd} → ${c.n_embd}, GELU</div><div class="arrow">↓</div>
    <div class="box">final LayerNorm</div><div class="arrow">↓</div>
    <div class="box">linear head (tied to the embedding) → logits over ${c.vocab_size} tokens</div><div class="arrow">↓</div>
    <div class="box">softmax → p(next token)</div>
   </div>
   <div class="card"><h3>${INFO.num_params.toLocaleString()} parameters</h3>
    <table>${groups.map(([k, v]) => `<tr><td>${k}</td><td class="n">${v.toLocaleString()}</td><td class="n">${((v / INFO.num_params) * 100).toFixed(1)}%</td></tr>`).join("")}</table>
    <h3 style="margin-top:18px">Attention, in one line</h3>
    <p><code>softmax(QKᵀ / √d<sub>k</sub> + M) V</code>, where <code>M</code> is 0 on and below the diagonal and −∞ above it,
     so after softmax a token gives exactly 0 weight to the future.</p>
    <p class="hint">For scale: GPT-2 small has 124M parameters, ${Math.round(124e6 / INFO.num_params)}× more. The architecture is the same.</p></div>
  </div>`;
}

/* ---------- router ---------- */
const PAGES = { generate: pageGenerate, next: pageNext, attention: pageAttention, tokenizer: pageTokenizer,
  training: pageTraining, architecture: pageArchitecture };
function route() {
  const name = (location.hash || "#generate").slice(1);
  const fn = PAGES[name] || pageGenerate;
  document.querySelectorAll("#nav a").forEach((a) => a.classList.toggle("active", a.dataset.page === name));
  hideTip();
  fn($("#page"));
}
document.addEventListener("keydown", (e) => {
  if (e.altKey && /^[1-6]$/.test(e.key)) { location.hash = Object.keys(PAGES)[+e.key - 1]; e.preventDefault(); }
});
(async () => {
  try {
    INFO = await api("/api/info");
    $("#modelchip").textContent = `${(INFO.num_params / 1e6).toFixed(2)}M params · ${INFO.config.n_layer}L ${INFO.config.n_head}H · val ppl ${Math.exp(INFO.meta.best_val_loss ?? INFO.meta.val_loss).toFixed(2)}`;
    window.onhashchange = route; route();
  } catch (e) { $("#page").innerHTML = `<div class="card">Could not reach the model server: ${esc(e.message)}</div>`; }
})();
