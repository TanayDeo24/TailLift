"use strict";

const EXAMPLE_QUERIES = [
  "nfldraftcountdown",
  "tcf national bank",
  "wells fargo.",
  "lexmark x63 printer fax",
  "species of trees",
];

const form = document.getElementById("query-form");
const input = document.getElementById("query-input");
const mapBtn = document.getElementById("map-btn");
const statusLine = document.getElementById("status-line");
const examplesEl = document.getElementById("examples");
const resultsCard = document.getElementById("results-card");
const resultsList = document.getElementById("results-list");
const resultsMeta = document.getElementById("results-meta");
const compareSection = document.getElementById("compare-section");
const compareGrid = document.getElementById("compare-grid");
const heroMeta = document.getElementById("hero-meta");
const evidenceGrid = document.getElementById("evidence-grid");

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

function setStatus(msg, isError) {
  statusLine.textContent = msg || "";
  statusLine.classList.toggle("error", Boolean(isError));
}

function renderExamples() {
  examplesEl.innerHTML = "";
  for (const q of EXAMPLE_QUERIES) {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "example-chip";
    chip.textContent = q;
    chip.addEventListener("click", () => {
      input.value = q;
      form.requestSubmit();
    });
    examplesEl.appendChild(chip);
  }
}

async function fetchJson(url, options) {
  const res = await fetch(url, options);
  let body;
  try {
    body = await res.json();
  } catch (_) {
    body = null;
  }
  if (!res.ok) {
    const detail = (body && body.detail) || `Request failed (${res.status})`;
    throw new Error(detail);
  }
  return body;
}

function fmtScore(n) {
  return typeof n === "number" ? n.toFixed(3) : "—";
}

function renderMapResults(data) {
  resultsList.innerHTML = "";
  data.results.forEach((r, i) => {
    const li = document.createElement("li");
    li.className = "result-row";
    li.style.animationDelay = `${i * 45}ms`;
    li.innerHTML = `
      <span class="result-index">${String(r.rank).padStart(2, "0")}</span>
      <span class="result-main">
        <span class="head-query">${escapeHtml(r.head_query)}</span>
        <div class="result-sub">semantic ${fmtScore(r.semantic_score)} · lexical ${fmtScore(r.lexical_score)} · support ${fmtScore(r.support_score)}</div>
      </span>
      <span class="result-score">${fmtScore(r.final_score)}</span>
    `;
    resultsList.appendChild(li);
  });
  resultsMeta.textContent = `Inference ${data.latency_ms.toFixed(1)} ms · ${data.model.embedding_model.split("/").pop()} · α ${data.model.alpha} · β ${data.model.beta}`;
  resultsCard.classList.remove("hidden");
}

function renderCompareColumn(title, block) {
  const col = document.createElement("div");
  col.className = "compare-col";
  const items = block.results
    .map(
      (r) =>
        `<div class="compare-item"><span><span class="ci-rank">${r.rank}</span>${escapeHtml(r.head_query)}</span><span class="ci-score">${fmtScore(r.score)}</span></div>`
    )
    .join("");
  col.innerHTML = `
    <h3>${title}</h3>
    <div class="col-latency">${block.latency_ms.toFixed(1)} ms</div>
    ${items || '<div class="compare-item"><span class="ci-score">no candidates</span></div>'}
  `;
  return col;
}

function renderCompare(data) {
  compareGrid.innerHTML = "";
  compareGrid.appendChild(renderCompareColumn("Lexical baseline", data.lexical_baseline));
  compareGrid.appendChild(renderCompareColumn("Semantic only", data.semantic_only));
  compareGrid.appendChild(renderCompareColumn("Final system", data.final_system));
  compareSection.classList.remove("hidden");
}

async function runQuery(query) {
  setStatus("");
  mapBtn.disabled = true;
  const original = mapBtn.textContent;
  mapBtn.textContent = "Mapping…";
  try {
    const [mapData, compareData] = await Promise.all([
      fetchJson("/api/map", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, k: 5 }),
      }),
      fetchJson("/api/compare", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, k: 5 }),
      }),
    ]);
    renderMapResults(mapData);
    renderCompare(compareData);
  } catch (err) {
    setStatus(err.message || "Something went wrong. Please try again.", true);
    resultsCard.classList.add("hidden");
    compareSection.classList.add("hidden");
  } finally {
    mapBtn.disabled = false;
    mapBtn.textContent = original;
  }
}

form.addEventListener("submit", (e) => {
  e.preventDefault();
  const query = input.value.trim();
  if (!query) {
    setStatus("Enter a query before mapping.", true);
    return;
  }
  runQuery(query);
});

function renderHeroMeta(meta) {
  const rows = heroMeta.querySelectorAll(".hm-value");
  rows[0].textContent = meta.head_index_size.toLocaleString();
  rows[1].textContent = meta.held_out_test_size.toLocaleString();
  rows[2].textContent = meta.final_mrr.toFixed(3);
}

function evidenceItem(label, value, emph) {
  return `<div class="evidence-item${emph ? " emph" : ""}"><div class="ev-label">${label}</div><div class="ev-value">${value}</div></div>`;
}

function renderEvidence(meta) {
  evidenceGrid.innerHTML = [
    evidenceItem("Baseline MRR", meta.baseline_mrr.toFixed(4)),
    evidenceItem("Final MRR", meta.final_mrr.toFixed(4), true),
    evidenceItem("Relative lift", `+${meta.relative_improvement_pct.toFixed(1)}%`, true),
    evidenceItem("Held-out queries", meta.held_out_test_size.toLocaleString()),
    evidenceItem(
      "Local P95 latency",
      meta.p95_local_inference_benchmark_ms ? `${meta.p95_local_inference_benchmark_ms.toFixed(1)} ms` : "n/a"
    ),
  ].join("");
}

async function loadMetadata() {
  try {
    const meta = await fetchJson("/api/metadata");
    renderHeroMeta(meta);
    renderEvidence(meta);
  } catch (err) {
    evidenceGrid.innerHTML = `<div class="evidence-item"><div class="ev-label">Status</div><div class="ev-value" style="font-size:16px;color:var(--ink-dim)">Backend not ready</div></div>`;
  }
}

async function checkHealth() {
  try {
    const h = await fetchJson("/health");
    if (!h.model_loaded) {
      setStatus("Model is still loading on the server — please wait a moment and try again.", true);
    }
  } catch (_) {
    setStatus("Backend is unreachable.", true);
  }
}

function initReveal() {
  const targets = document.querySelectorAll(".reveal");
  if (!("IntersectionObserver" in window)) {
    return; // .reveal is visible by default -- no JS needed for content to show
  }
  // Opt into the hidden pre-reveal state only once JS + observers are
  // confirmed available, so a failure anywhere upstream never leaves
  // content permanently invisible.
  targets.forEach((t) => t.classList.add("pending"));
  const io = new IntersectionObserver(
    (entries) => {
      for (const entry of entries) {
        if (entry.isIntersecting) {
          entry.target.classList.add("in-view");
          io.unobserve(entry.target);
        }
      }
    },
    { threshold: 0.12 }
  );
  targets.forEach((t) => io.observe(t));
  // Safety net: force-reveal anything still pending after 2.5s (covers
  // edge cases like instant anchor jumps or unusual viewport states).
  setTimeout(() => {
    document.querySelectorAll(".reveal.pending:not(.in-view)").forEach((t) => t.classList.add("in-view"));
  }, 2500);
}

renderExamples();
initReveal();
checkHealth();
loadMetadata();
