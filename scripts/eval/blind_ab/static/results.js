"use strict";
// Results page: variants table, quality-vs-speed scatter, head-to-head matrix.

const $ = (id) => document.getElementById(id);
const REFRESH_MS = 10000;
const SVG_NS = "http://www.w3.org/2000/svg";

function el(tag, attrs, text) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) node.setAttribute(k, v);
  if (text !== undefined && text !== null) node.textContent = text;
  return node;
}
function svg(tag, attrs, text) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs || {})) node.setAttribute(k, v);
  if (text !== undefined) node.textContent = text;
  return node;
}
const pct = (x) => (x === null || x === undefined ? "" : `${(100 * x).toFixed(1)}%`);
const num = (x, d) => (x === null || x === undefined ? "" : Number(x).toFixed(d));

function renderArms(data) {
  const table = $("arms-table");
  table.replaceChildren();
  const labels = [...new Set(data.arms.filter((r) => r.seconds_per_clip).map((r) => r.time_label || ""))];
  const timeLabel = labels.length === 1 && labels[0] ? labels[0] : "Time per clip";
  const head = ["Variant", "Votes", "Wins / Can't tell / Losses / Both bad", "Win rate [95% range]", "",
    "Ranking score", timeLabel, `Speedup vs ${data.baseline || "-"}`, "Size"];
  const tr = el("tr");
  head.forEach((h) => tr.appendChild(el("th", {}, h)));
  table.appendChild(el("thead")).appendChild(tr);
  const body = table.appendChild(el("tbody"));
  for (const r of data.arms) {
    const row = el("tr");
    const name = el("td", { title: r.notes || "" });
    name.appendChild(el("div", {}, r.display_name));
    name.appendChild(el("div", { class: "muted small mono" }, r.slug));
    row.appendChild(name);
    row.appendChild(el("td", { class: "num" }, r.votes));
    row.appendChild(el("td", { class: "num" }, `${r.wins} / ${r.ties} / ${r.losses} / ${r.both_bad}`));
    row.appendChild(el("td", { class: "num" },
      r.votes ? `${pct(r.win_rate)} [${pct(r.win_rate_ci_low)}, ${pct(r.win_rate_ci_high)}]` : "-"));
    row.appendChild(ciBar(r));
    row.appendChild(el("td", { class: "num strong" }, num(r.bt_score, 0)));
    row.appendChild(el("td", { class: "num", title: r.hardware || "" },
      r.seconds_per_clip ? `${num(r.seconds_per_clip, 2)} s` : ""));
    row.appendChild(el("td", { class: "num strong" }, r.speedup_vs_baseline ? `${num(r.speedup_vs_baseline, 2)}x` : ""));
    row.appendChild(el("td", { class: "num" }, r.size || ""));
    body.appendChild(row);
  }
}

function ciBar(r) {
  const td = el("td", { class: "cibar" });
  if (!r.votes) return td;
  const box = svg("svg", { width: 120, height: 14, viewBox: "0 0 120 14" });
  box.appendChild(svg("line", { x1: 60, x2: 60, y1: 0, y2: 14, class: "ref" }));
  box.appendChild(svg("line", { x1: 120 * r.win_rate_ci_low, x2: 120 * r.win_rate_ci_high, y1: 7, y2: 7, class: "ci" }));
  box.appendChild(svg("circle", { cx: 120 * r.win_rate, cy: 7, r: 3.5, class: "pt" }));
  td.appendChild(box);
  return td;
}

function renderScatter(data) {
  const host = $("scatter");
  host.replaceChildren();
  const pts = data.arms.filter((r) => r.speedup_vs_baseline);
  if (!pts.length) {
    host.appendChild(el("p", { class: "muted" }, "No variants have timing data (speed.seconds_per_clip in arms.json)."));
    return;
  }
  const W = 640, H = 320, M = { l: 56, r: 24, t: 16, b: 40 };
  const xs = pts.map((r) => Math.log2(r.speedup_vs_baseline));
  const ys = pts.map((r) => r.bt_score);
  const [x0, x1] = [Math.min(-0.25, ...xs) - 0.25, Math.max(0.25, ...xs) + 0.25];
  const [y0, y1] = [Math.min(...ys) - 30, Math.max(...ys) + 30];
  const sx = (x) => M.l + ((x - x0) / (x1 - x0)) * (W - M.l - M.r);
  const sy = (y) => H - M.b - ((y - y0) / (y1 - y0)) * (H - M.t - M.b);
  const s = svg("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}`, class: "chart" });
  s.appendChild(svg("line", { x1: sx(0), x2: sx(0), y1: M.t, y2: H - M.b, class: "ref" }));
  s.appendChild(svg("line", { x1: M.l, x2: W - M.r, y1: sy(1000), y2: sy(1000), class: "ref" }));
  for (let e = Math.ceil(x0); e <= Math.floor(x1); e += 1) {
    s.appendChild(svg("text", { x: sx(e), y: H - M.b + 16, class: "tick", "text-anchor": "middle" },
      `${Math.pow(2, e) >= 1 ? Math.pow(2, e) : `1/${Math.pow(2, -e)}`}x`));
  }
  s.appendChild(svg("text", { x: (W + M.l) / 2, y: H - 6, class: "axis", "text-anchor": "middle" },
    `speedup vs ${data.baseline} (faster →)`));
  s.appendChild(svg("text", { x: 14, y: H / 2, class: "axis", "text-anchor": "middle",
    transform: `rotate(-90 14 ${H / 2})` }, "ranking score (better →)"));
  [y0, (y0 + y1) / 2, y1].forEach((y) => s.appendChild(svg("text", { x: M.l - 6, y: sy(y) + 4, class: "tick",
    "text-anchor": "end" }, y.toFixed(0))));
  pts.forEach((r, i) => {
    const g = svg("g");
    g.appendChild(svg("title", {}, `${r.display_name}: ${num(r.seconds_per_clip, 2)} s/clip, ranking score ${num(r.bt_score, 0)}`));
    g.appendChild(svg("circle", { cx: sx(xs[i]), cy: sy(ys[i]), r: 5, class: "pt" }));
    g.appendChild(svg("text", { x: sx(xs[i]) + 8, y: sy(ys[i]) + 4, class: "label" }, r.display_name));
    s.appendChild(g);
  });
  host.appendChild(s);
}

function renderMatrix(data) {
  const table = $("matrix-table");
  table.replaceChildren();
  const order = data.arms.map((r) => r.slug);
  const names = Object.fromEntries(data.arms.map((r) => [r.slug, r.display_name]));
  const head = el("tr");
  head.appendChild(el("th", {}, "row vs col"));
  order.forEach((s) => head.appendChild(el("th", { title: s }, names[s])));
  table.appendChild(el("thead")).appendChild(head);
  const body = table.appendChild(el("tbody"));
  for (const a of order) {
    const tr = el("tr");
    tr.appendChild(el("th", { title: a }, names[a]));
    for (const b of order) {
      if (a === b) { tr.appendChild(el("td", { class: "diag" }, "")); continue; }
      const c = data.matrix[a][b];
      const ties = c.ties + c.both_bad;
      const n = c.wins + c.losses + ties;
      const rate = n ? (c.wins + 0.5 * ties) / n : null;
      const td = el("td", { class: "num", title: n ? `${pct(rate)} of ${n} votes (${c.both_bad} both bad)` : "no votes" },
        n ? `${c.wins}-${ties}-${c.losses}` : "-");
      if (rate !== null) td.style.background = shade(rate);
      tr.appendChild(td);
    }
    body.appendChild(tr);
  }
}

function shade(rate) {
  // Diverging: blue when the row arm wins, orange when it loses, neutral at 50%.
  const d = Math.min(1, Math.abs(rate - 0.5) * 2);
  const hue = rate >= 0.5 ? 212 : 28;
  return `hsla(${hue}, 70%, 50%, ${(0.08 + 0.42 * d).toFixed(2)})`;
}

let currentBaseline = new URLSearchParams(location.search).get("baseline") || "";

function fillBaseline(data) {
  const sel = $("baseline");
  const withSpeed = data.arms.filter((r) => r.seconds_per_clip);
  const want = data.baseline || "";
  sel.replaceChildren(...(withSpeed.length ? withSpeed : [{ slug: "", display_name: "(no speed data)" }])
    .map((r) => el("option", { value: r.slug }, r.display_name)));
  sel.value = want;
}

async function refresh() {
  const q = currentBaseline ? `?baseline=${encodeURIComponent(currentBaseline)}` : "";
  try {
    const resp = await fetch(`/api/results${q}`);
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.error || resp.statusText);
    $("summary").textContent = `${data.bundle}: ${data.total_votes} votes from ${data.voters.length} voters `
      + `on ${data.clips} clips${data.ignored_votes ? ` (${data.ignored_votes} votes on retired variants not counted)` : ""}`;
    $("csv-link").href = `/api/results.csv${q}`;
    $("json-link").href = `/api/results.json${q}`;
    fillBaseline(data);
    renderArms(data);
    renderScatter(data);
    renderMatrix(data);
    $("status").textContent = `Updated ${new Date().toLocaleTimeString()}`;
  } catch (err) {
    $("status").textContent = `Could not load results: ${err.message}`;
  }
}

$("baseline").addEventListener("change", (e) => {
  currentBaseline = e.target.value;
  history.replaceState(null, "", currentBaseline ? `?baseline=${encodeURIComponent(currentBaseline)}` : "/results");
  refresh();
});
setInterval(() => { if ($("auto-chk").checked && !document.hidden) refresh(); }, REFRESH_MS);
refresh();
