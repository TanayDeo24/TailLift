"use strict";

const EXAMPLE_QUERIES = [
  "nfldraftcountdown",
  "tcf national bank",
  "gwen stefani's clothing line",
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
const compareCard = document.getElementById("compare-card");
const compareGrid = document.getElementById("compare-grid");
const infoGrid = document.getElementById("info-grid");
const offlineBadge = document.getElementById("offline-badge");

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

function renderMapResults(data) {
  resultsList.innerHTML = "";
  for (const r of data.results) {
    const row = document.createElement("div");
    row.className = "result-row";
    row.innerHTML = `
      <div class="rank-badge">${r.rank}</div>
      <div class="result-main">
        <div class="head-query">${escapeHtml(r.head_query)}</div>
        <div class="scores">lexical ${r.lexical_score.toFixed(3)} · semantic ${r.semantic_score.toFixed(3)} · support ${r.support_score.toFixed(3)}</div>
      </div>
      <div class="final-score">${r.final_score.toFixed(3)}</div>
    `;
    resultsList.appendChild(row);
  }
  resultsMeta.textContent = `Inference latency: ${data.latency_ms.toFixed(1)} ms · model: ${data.model.embedding_model} · alpha ${data.model.alpha} · beta ${data.model.beta}`;
  resultsCard.classList.remove("hidden");
}

function renderCompareColumn(title, block) {
  const col = document.createElement("div");
  col.className = "compare-col";
  const items = block.results
    .map(
      (r) =>
        `<div class="compare-item"><span class="cq">${r.rank}. ${escapeHtml(r.head_query)}</span><span class="cs">${r.score.toFixed(3)}</span></div>`
    )
    .join("");
  col.innerHTML = `
    <h3>${title}</h3>
    <div class="system-latency">${block.latency_ms.toFixed(1)} ms</div>
    ${items || '<div class="compare-item cs">no candidates</div>'}
  `;
  return col;
}

function renderCompare(data) {
  compareGrid.innerHTML = "";
  compareGrid.appendChild(renderCompareColumn("Lexical Baseline", data.lexical_baseline));
  compareGrid.appendChild(renderCompareColumn("Semantic Only", data.semantic_only));
  compareGrid.appendChild(renderCompareColumn("Final System (Level 5)", data.final_system));
  compareCard.classList.remove("hidden");
}

async function runQuery(query) {
  setStatus("");
  mapBtn.disabled = true;
  const original = mapBtn.textContent;
  mapBtn.innerHTML = '<span class="spinner"></span>Mapping…';
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
    compareCard.classList.add("hidden");
  } finally {
    mapBtn.disabled = false;
    mapBtn.textContent = original;
  }
}

form.addEventListener("submit", (e) => {
  e.preventDefault();
  const query = input.value.trim();
  if (!query) {
    setStatus("Please enter a query before mapping.", true);
    return;
  }
  runQuery(query);
});

async function loadMetadata() {
  try {
    const meta = await fetchJson("/api/metadata");
    offlineBadge.classList.remove("hidden");
    infoGrid.innerHTML = `
      <div class="info-item"><div class="label">Frozen model</div><div class="value">${escapeHtml(meta.frozen_model.split("/").pop())}</div></div>
      <div class="info-item"><div class="label">Candidate index</div><div class="value">${meta.head_index_size.toLocaleString()} head queries</div></div>
      <div class="info-item"><div class="label">Held-out evaluation</div><div class="value">${meta.held_out_test_size.toLocaleString()} tail queries</div></div>
      <div class="info-item"><div class="label">Baseline MRR</div><div class="value">${meta.baseline_mrr.toFixed(4)}</div></div>
      <div class="info-item"><div class="label">Final MRR</div><div class="value">${meta.final_mrr.toFixed(4)}</div></div>
      <div class="info-item"><div class="label">Improvement</div><div class="value">+${meta.relative_improvement_pct.toFixed(1)}%</div></div>
      <div class="info-item"><div class="label">Local P95 latency</div><div class="value">${meta.p95_local_inference_benchmark_ms ? meta.p95_local_inference_benchmark_ms.toFixed(1) + " ms" : "n/a"}</div></div>
    `;
  } catch (err) {
    infoGrid.innerHTML = `<div class="status-line error">Backend not ready: ${escapeHtml(err.message)}</div>`;
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

renderExamples();
checkHealth();
loadMetadata();
