/* ══════════════════════════════════════════════════════════════════════════
   PlaceIQ — app.js
   Tabs · Predict · Batch · Analytics · Model Info
   ══════════════════════════════════════════════════════════════════════════ */

"use strict";

// ── Utilities ────────────────────────────────────────────────────────────────

const esc = (s) =>
  String(s).replace(/[&<>"]/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c])
  );

const fmt   = (v, d = 3) => v == null ? "–" : Number(v).toFixed(d);
const fmtPct = (v)       => v == null ? "–" : (Number(v) * 100).toFixed(1) + "%";
const sigmoid = (x)      => 1 / (1 + Math.exp(-x));

function showError(msg) {
  const box = document.getElementById("error-toast");
  box.textContent = msg;
  box.hidden = false;
  clearTimeout(showError._t);
  showError._t = setTimeout(() => (box.hidden = true), 7000);
}

async function api(path, options = {}) {
  const res = await fetch(path, options);
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body.detail || `HTTP ${res.status}`);
  return body;
}

// ── Tab switching ─────────────────────────────────────────────────────────────

const TABS = [
  { btn: "tab-predict",   panel: "panel-predict",   onActivate: null },
  { btn: "tab-batch",     panel: "panel-batch",     onActivate: null },
  { btn: "tab-analytics", panel: "panel-analytics", onActivate: loadAnalytics },
  { btn: "tab-model",     panel: "panel-model",     onActivate: loadModelInfo },
];

TABS.forEach(({ btn, panel, onActivate }) => {
  const btnEl = document.getElementById(btn);
  btnEl.addEventListener("click", () => {
    TABS.forEach(({ btn: b, panel: p }) => {
      const bEl = document.getElementById(b);
      const pEl = document.getElementById(p);
      const isActive = b === btn;
      bEl.classList.toggle("active", isActive);
      bEl.setAttribute("aria-selected", String(isActive));
      pEl.hidden = !isActive;
    });
    if (onActivate) onActivate();
  });
});

// ── Model meta chips ──────────────────────────────────────────────────────────

async function loadMeta() {
  try {
    const meta = await api("/api/model/meta");
    const chips = document.getElementById("meta-chips");
    const m = meta.test_metrics || {};
    const roc = parseFloat(m.roc_auc);
    chips.innerHTML = [
      `<span class="chip"><b>${esc(meta.model || "?")}</b> model</span>`,
      `<span class="chip ${roc >= 0.85 ? "great" : roc >= 0.75 ? "good" : ""}">ROC-AUC <b>${fmt(m.roc_auc)}</b></span>`,
      `<span class="chip">F1 <b>${fmt(m.f1)}</b></span>`,
      `<span class="chip">Brier <b>${fmt(m.brier)}</b></span>`,
      `<span class="chip">thr <b>${fmt(meta.threshold)}</b></span>`,
      `<span class="chip">SHAP <b>${meta.explanation_mode || "exact"}</b></span>`,
      meta.salary_model ? `<span class="chip good">salary model ✓</span>` : "",
    ].join("");
    document.getElementById("run-line").textContent =
      `run ${meta.run_id || "?"} · ${meta.n_runs || 0} experiment(s) · ${meta.n_total || "?"} training rows`;
    // Pre-populate metric cards in result section
    renderMetricCards(m);
  } catch (e) { /* best-effort */ }
}

async function loadDrift() {
  try {
    const d = await api("/api/monitor/psi?window=200");
    const line = document.getElementById("drift-line");
    if (!d.available) { line.textContent = "monitoring: no reference distribution yet"; return; }
    document.getElementById("pred-count").textContent = `${d.n_samples} logged`;
    const alerts = d.alerts?.length ? d.alerts.join(", ") : null;
    line.className = "drift-line " + (alerts ? "warn" : "good");
    line.textContent = alerts
      ? `⚠ drift alerts: ${alerts}`
      : `✓ feature drift stable across ${d.n_samples} recent predictions`;
  } catch (e) { /* best-effort */ }
}

// ── Prediction form ───────────────────────────────────────────────────────────

const EXAMPLE = {
  gender: "M", tenth_p: 67.0, tenth_b: "Others", twelfth_p: 91.0, twelfth_b: "Others",
  twelfth_s: "Commerce", degree_p: 58.0, degree_t: "Sci&Tech", workex: "No",
  etest_p: 55.0, mba_p: 58.8, has_masters: "Yes", masters_type: "MBA",
  specialisation: "Mkt&HR",
};

const form = document.getElementById("predict-form");
const hasMastersEl = document.getElementById("f-has_masters");

function syncMastersFields() {
  const yes = hasMastersEl.value === "Yes";
  document.getElementById("masters-type-field").hidden = !yes;
  document.getElementById("specialisation-field").hidden = !yes;
  if (!yes) document.getElementById("f-mba_p").value = 0;
}

hasMastersEl.addEventListener("change", syncMastersFields);
syncMastersFields();

document.getElementById("prefill").addEventListener("click", () => {
  Object.entries(EXAMPLE).forEach(([k, v]) => {
    const el = form.elements[k];
    if (el) el.value = v;
  });
  syncMastersFields();
});

document.getElementById("clear").addEventListener("click", () => {
  form.reset();
  document.getElementById("result").hidden = true;
  document.getElementById("cf-result").hidden = true;
  document.getElementById("salary-badge").hidden = true;
});

function formData() {
  const data = Object.fromEntries(new FormData(form).entries());
  // Parse numeric fields
  for (const k of ["tenth_p", "twelfth_p", "degree_p", "etest_p", "mba_p"]) {
    if (data[k] !== "") data[k] = parseFloat(data[k]);
  }
  // Keep masters fields internally consistent
  if (data.has_masters !== "Yes") {
    data.has_masters = "No";
    data.masters_type = "None";
    data.mba_p = 0;
  } else if (!data.masters_type || data.masters_type === "None") {
    data.masters_type = "MBA";
  }
  // Optional Kumbhar fields — only include if non-empty
  for (const k of ["cgpa", "aptitude_score", "soft_skills"]) {
    if (data[k] === "" || data[k] == null) delete data[k];
    else data[k] = parseFloat(data[k]);
  }
  for (const k of ["internships", "projects", "workshops"]) {
    if (data[k] === "" || data[k] == null) delete data[k];
    else data[k] = parseInt(data[k], 10);
  }
  return data;
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const btn = document.getElementById("submit");
  btn.disabled = true;
  btn.innerHTML = `<svg viewBox="0 0 20 20" fill="none" style="animation:spin .8s linear infinite"><path d="M10 3a7 7 0 1 0 7 7" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/></svg> Predicting…`;
  document.getElementById("cf-result").hidden = true;
  try {
    const result = await api("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(formData()),
    });
    renderResult(result);
    loadDrift();
  } catch (err) {
    showError(err.message);
  } finally {
    btn.disabled = false;
    btn.innerHTML = `<svg viewBox="0 0 20 20" fill="none"><path d="M10 3l7 7-7 7M3 10h14" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg> Predict Placement`;
  }
});

// ── Result rendering ──────────────────────────────────────────────────────────

function renderResult(p) {
  const placed = p.prediction === "Placed";

  // Prediction badge
  const badge = document.getElementById("prediction");
  badge.textContent = p.prediction;
  badge.className = "pred-badge " + (placed ? "yes" : "no");

  // Gauge
  const pct = Math.round(p.probability * 100);
  updateGauge(p.probability, placed);

  // Note line
  let note = `${esc(p.model)} · cal: ${esc(p.calibration || "none")}`;
  if (p.explanation_mode === "surrogate") note += ` · SHAP via ${esc(p.explanation_model || "")}`;
  document.getElementById("prediction-note").textContent = note;

  // Salary badge
  renderSalaryBadge(p, placed);

  // Metric cards
  renderMetricCards(p);

  // Waterfall + contributors
  renderWaterfall(p);
  renderContributors(p.contributors);

  // Counterfactual ready
  const cfBtn = document.getElementById("cf");
  cfBtn.disabled = !p.contributors?.length;

  // Show panel
  const resultEl = document.getElementById("result");
  resultEl.hidden = false;
  resultEl.scrollIntoView({ behavior: "smooth", block: "start" });
}

function updateGauge(probability, placed) {
  const ARC_LENGTH = 157; // half-circle circumference for r=50 (≈ π * 50)
  const offset = ARC_LENGTH * (1 - probability);
  const arc = document.getElementById("gauge-arc");
  const label = document.getElementById("gauge-pct");
  arc.style.strokeDashoffset = offset;
  arc.style.stroke = placed ? "var(--emerald)" : "var(--danger)";
  label.textContent = Math.round(probability * 100) + "%";
  label.style.fill = placed ? "var(--emerald)" : "var(--danger)";
}

function renderSalaryBadge(p, placed) {
  const badge = document.getElementById("salary-badge");
  if (!placed || !p.salary_estimate) { badge.hidden = true; return; }
  document.getElementById("salary-amount").textContent =
    "₹" + (p.salary_estimate / 100000).toFixed(1) + " LPA";
  const bandEl = document.getElementById("salary-band");
  bandEl.textContent = p.salary_band?.toUpperCase() || "";
  bandEl.className = "salary-band " + (p.salary_band || "mid");
  if (p.salary_range?.length === 2) {
    document.getElementById("salary-range").textContent =
      `Range: ₹${(p.salary_range[0] / 100000).toFixed(1)}–${(p.salary_range[1] / 100000).toFixed(1)} LPA`;
  }
  badge.hidden = false;
}

function renderMetricCards(m) {
  const host = document.getElementById("metric-cards");
  if (!m) return;
  const cards = [
    ["ROC-AUC",   fmt(m.roc_auc)],
    ["PR-AUC",    fmt(m.pr_auc)],
    ["F1",        fmt(m.f1)],
    ["Precision", fmt(m.precision)],
    ["Recall",    fmt(m.recall)],
    ["Brier",     fmt(m.brier)],
  ];
  host.innerHTML = cards.map(([k, v]) =>
    `<div class="metric"><strong>${v ?? "–"}</strong><span>${k}</span></div>`
  ).join("");
}

// ── SHAP waterfall (SVG) ─────────────────────────────────────────────────────

function renderWaterfall(p) {
  const host = document.getElementById("waterfall");
  const base = p.base_probability ?? 0.5;
  const contributors = (p.contributors ?? []).slice(0, 10);
  if (!contributors.length) { host.innerHTML = ""; return; }

  let logit = Math.log(base / (1 - base));
  const runs = [{ label: "Base", start: base, end: base, shap: 0 }];
  for (const c of contributors) {
    const s = logit, e = logit + c.shap;
    logit = e;
    runs.push({ label: c.feature, start: sigmoid(s), end: sigmoid(e), shap: c.shap });
  }

  const W = 540, rowH = 30, padL = 160, padR = 60;
  const H = 30 + runs.length * rowH + 10;
  const bw = W - padL - padR;
  const yOf = (idx) => 20 + idx * rowH;

  let svg = `<svg viewBox="0 0 ${W} ${H}" style="width:100%;min-width:340px" aria-hidden="true">`;

  // Grid lines
  for (let i = 0; i <= 5; i++) {
    const x = padL + (bw * i) / 5;
    svg += `<line x1="${x}" y1="12" x2="${x}" y2="${H - 5}" stroke="#252d55" stroke-width=".7"/>`;
    svg += `<text x="${x}" y="9" fill="#7b87b8" font-size="9" text-anchor="middle">${(i * 20)}%</text>`;
  }

  runs.forEach((r, i) => {
    const y = yOf(i);
    const x1 = padL + bw * Math.min(r.start, r.end);
    const barW = Math.max(bw * Math.abs(r.end - r.start), 3);
    const color = r.shap > 0 ? "#34d39988" : r.shap < 0 ? "#f8717188" : "#404a78";
    svg += `<rect x="${x1}" y="${y + 6}" width="${barW}" height="${rowH - 10}" rx="3" fill="${color}"/>`;
    const label = r.label.length > 22 ? r.label.slice(0, 21) + "…" : r.label;
    svg += `<text x="${padL - 8}" y="${y + rowH / 2 + 4}" fill="${i === 0 ? "#7b87b8" : "#e2e8f8"}" font-size="10.5" text-anchor="end">${esc(label)}</text>`;
    const pctTxt = (r.end * 100).toFixed(0) + "%";
    svg += `<text x="${x1 + barW + 4}" y="${y + rowH / 2 + 4}" fill="${r.shap >= 0 ? "#34d399" : "#f87171"}" font-size="9.5" font-weight="700">${esc(pctTxt)}</text>`;
  });

  svg += "</svg>";
  host.innerHTML = svg;
}

// ── SHAP contributor bars ────────────────────────────────────────────────────

function renderContributors(contributors) {
  const list = document.getElementById("contributors");
  list.innerHTML = "";
  if (!contributors?.length) return;
  const max = Math.max(...contributors.map((c) => Math.abs(c.shap)), 1e-9);
  contributors.slice(0, 8).forEach((c) => {
    const li = document.createElement("li");
    const track = document.createElement("div");
    track.className = "row-track";
    const bar = document.createElement("div");
    const isPos = c.shap >= 0;
    bar.className = "row-bar " + (isPos ? "positive" : "negative");
    bar.style.width = Math.max((Math.abs(c.shap) / max) * 100, 2) + "%";
    if (isPos) bar.style.float = "right";
    track.appendChild(bar);
    const meta = document.createElement("div");
    meta.className = "row-meta";
    meta.innerHTML = `<span>${esc(c.feature)}</span><b>${isPos ? "+" : ""}${c.shap.toFixed(3)} · ${isPos ? "supports ✓" : "opposes ✗"}</b>`;
    li.appendChild(track);
    li.appendChild(meta);
    list.appendChild(li);
  });
}

// ── Counterfactual ────────────────────────────────────────────────────────────

const cfBtn = document.getElementById("cf");
const cfResult = document.getElementById("cf-result");

cfBtn.addEventListener("click", async () => {
  cfBtn.disabled = true;
  cfBtn.textContent = "Searching…";
  try {
    const res = await api("/api/explain/counterfactual", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(formData()),
    });
    cfResult.hidden = false;
    if (res.already_true || !res.steps.length) {
      cfResult.innerHTML = `<p class="note">No single-edit path to <b>${esc(res.target)}</b> found.</p>`;
      return;
    }
    const diff = ((res.to_probability - res.from_probability) * 100).toFixed(1);
    let html = `<p class="note">Edits to reach <b>${esc(res.target)}</b> 
      (${(res.from_probability * 100).toFixed(0)}% → <span class="delta">${(res.to_probability * 100).toFixed(0)}%</span>, Δ${diff}pp). 
      ${res.achieved ? "Outcome flips. ✓" : "Partial improvement."}</p>`;
    res.steps.forEach((s) => {
      html += `<div class="cf-step">
        <span>${esc(s.feature)}: <b>${esc(s.from)}</b> → <b>${esc(s.to)}</b></span>
        <span class="delta">${((s.probability_after - s.probability_before) * 100).toFixed(0)}pp</span>
        <button class="btn ghost" data-feature="${esc(s.feature)}" data-value="${esc(s.to)}">Apply</button>
      </div>`;
    });
    cfResult.innerHTML = html;
    cfResult.querySelectorAll("button[data-feature]").forEach((b) =>
      b.addEventListener("click", () => {
        const el = form.elements[b.dataset.feature];
        if (el) el.value = b.dataset.value;
        form.querySelector("#submit").click();
      })
    );
  } catch (err) { showError(err.message); }
  finally {
    cfBtn.disabled = false;
    cfBtn.innerHTML = `<svg viewBox="0 0 20 20" fill="none"><path d="M4 4h5v5M16 16h-5v-5M4 16l12-12" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/></svg> Suggest changes to flip the outcome`;
  }
});

// ── Batch ─────────────────────────────────────────────────────────────────────

const BATCH_EXAMPLE_EXTRA = {
  ...EXAMPLE, tenth_p: 50, twelfth_p: 62, degree_p: 52, mba_p: 50, etest_p: 40,
};

document.getElementById("batch-example").addEventListener("click", () => {
  document.getElementById("batch-input").value =
    JSON.stringify(EXAMPLE) + "\n" + JSON.stringify(BATCH_EXAMPLE_EXTRA);
});

document.getElementById("batch-run").addEventListener("click", async () => {
  const raw = document.getElementById("batch-input").value.trim();
  if (!raw) return;
  let items;
  try {
    const parsed = JSON.parse(raw);
    items = Array.isArray(parsed) ? parsed : raw.split("\n").map((l) => JSON.parse(l.trim())).filter(Boolean);
  } catch {
    items = raw.split("\n").map((l) => { try { return JSON.parse(l.trim()); } catch { return null; } }).filter(Boolean);
  }
  const btn = document.getElementById("batch-run");
  btn.disabled = true;
  try {
    const res = await api("/api/predict/batch", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ items }),
    });
    const rows = res.results.map((r, i) => {
      const salaryCell = r.salary_band
        ? `<span class="salary-band ${r.salary_band}">${r.salary_band.toUpperCase()}</span>`
        : "—";
      return `<tr>
        <td>${i + 1}</td>
        <td><b style="color:${r.prediction === "Placed" ? "var(--emerald)" : "var(--danger)"}">${esc(r.prediction)}</b></td>
        <td>${(r.probability * 100).toFixed(0)}%</td>
        <td>${(r.confidence * 100).toFixed(0)}%</td>
        <td>${salaryCell}</td>
      </tr>`;
    }).join("");
    document.getElementById("batch-result").innerHTML = `
      <table>
        <thead><tr><th>#</th><th>Prediction</th><th>Prob</th><th>Confidence</th><th>Salary Band</th></tr></thead>
        <tbody>${rows}</tbody>
        <tfoot><tr class="sum-row">
          <td>${res.n}</td><td>${res.placed} placed</td>
          <td>${res.not_placed} not placed</td>
          <td>mean ${(res.mean_probability * 100).toFixed(0)}%</td><td></td>
        </tr></tfoot>
      </table>`;
    loadDrift();
  } catch (err) { showError(err.message); }
  finally { btn.disabled = false; }
});

// ── Analytics tab ─────────────────────────────────────────────────────────────

let analyticsLoaded = false;

async function loadAnalytics() {
  if (analyticsLoaded) return;
  analyticsLoaded = true;
  try {
    const data = await api("/api/admin/analytics?window=500");
    renderKPIs(data);
    if (data.available) {
      renderDailyChart(data.daily_counts);
      renderSHAPChart(data.top_shap);
    } else {
      document.getElementById("daily-note").hidden = false;
    }
  } catch (e) { showError("Analytics load failed: " + e.message); }
}

function renderKPIs(data) {
  const n = data.n_predictions || 0;
  const rate = data.placement_rate ?? null;
  const placed = rate != null ? Math.round(n * rate) : null;
  const notPlaced = placed != null ? n - placed : null;
  document.getElementById("kpi-total").querySelector("strong").textContent = n;
  document.getElementById("kpi-rate").querySelector("strong").textContent =
    rate != null ? (rate * 100).toFixed(1) + "%" : "—";
  document.getElementById("kpi-placed").querySelector("strong").textContent = placed ?? "—";
  document.getElementById("kpi-notplaced").querySelector("strong").textContent = notPlaced ?? "—";
}

function renderDailyChart(daily) {
  const host = document.getElementById("daily-chart");
  if (!daily?.length) { document.getElementById("daily-note").hidden = false; return; }

  const max = Math.max(...daily.map((d) => d.placed + d.not_placed), 1);
  const W = 560, H = 180, padL = 30, padB = 28, barW = Math.max(16, Math.min(40, (W - padL - 20) / daily.length - 4));
  let svg = `<svg viewBox="0 0 ${W} ${H}" style="width:100%" aria-hidden="true">`;

  daily.forEach((d, i) => {
    const x = padL + i * ((W - padL - 20) / daily.length);
    const placedH = ((d.placed / max) * (H - padB - 10));
    const notH = ((d.not_placed / max) * (H - padB - 10));
    const y1 = H - padB - placedH;
    const y2 = y1 - notH;
    svg += `<rect x="${x}" y="${y1}" width="${barW}" height="${placedH}" rx="3" fill="#34d399aa"/>`;
    svg += `<rect x="${x}" y="${y2}" width="${barW}" height="${notH}" rx="3" fill="#f87171aa"/>`;
    if (i % Math.max(1, Math.floor(daily.length / 6)) === 0) {
      const lbl = d.date?.slice(5) || "";
      svg += `<text x="${x + barW / 2}" y="${H}" fill="#7b87b8" font-size="9" text-anchor="middle">${esc(lbl)}</text>`;
    }
  });

  // Legend
  svg += `<rect x="${W - 90}" y="10" width="10" height="10" rx="2" fill="#34d399aa"/>`;
  svg += `<text x="${W - 76}" y="19" fill="#7b87b8" font-size="9">Placed</text>`;
  svg += `<rect x="${W - 90}" y="26" width="10" height="10" rx="2" fill="#f87171aa"/>`;
  svg += `<text x="${W - 76}" y="35" fill="#7b87b8" font-size="9">Not Placed</text>`;
  svg += "</svg>";
  host.innerHTML = svg;
}

function renderSHAPChart(top) {
  const host = document.getElementById("shap-chart");
  if (!top?.length) { host.innerHTML = `<p class="note">Train a model first to see global SHAP importance.</p>`; return; }

  const maxV = top[0].importance;
  const rowH = 22, padL = 160, padR = 50, W = 520;
  const H = top.length * rowH + 20;
  let svg = `<svg viewBox="0 0 ${W} ${H}" style="width:100%" aria-hidden="true">`;
  top.forEach((f, i) => {
    const y = i * rowH + 8;
    const barW = (f.importance / maxV) * (W - padL - padR);
    svg += `<rect x="${padL}" y="${y}" width="${barW}" height="${rowH - 6}" rx="3" fill="#7c6cffaa"/>`;
    const label = f.feature.length > 22 ? f.feature.slice(0, 21) + "…" : f.feature;
    svg += `<text x="${padL - 8}" y="${y + rowH / 2 - 1}" fill="#e2e8f8" font-size="10.5" text-anchor="end">${esc(label)}</text>`;
    svg += `<text x="${padL + barW + 5}" y="${y + rowH / 2 - 1}" fill="#7c6cff" font-size="9.5" font-weight="700">${f.importance.toFixed(4)}</text>`;
  });
  svg += "</svg>";
  host.innerHTML = svg;
}

// ── Model Info tab ────────────────────────────────────────────────────────────

let modelInfoLoaded = false;

async function loadModelInfo() {
  if (modelInfoLoaded) return;
  modelInfoLoaded = true;
  try {
    const [meta, cmp] = await Promise.allSettled([
      api("/api/model/meta"),
      api("/api/model/comparison"),
    ]);

    if (meta.status === "fulfilled") {
      renderLearningCurve(meta.value.learning_curve);
      renderRunsTable(meta.value.runs, meta.value.run_id);
    }
    if (cmp.status === "fulfilled") {
      renderComparisonTable(cmp.value.rows, meta.value?.run_id);
    } else {
      document.getElementById("comparison-table").innerHTML =
        `<p class="note">Train the model first to see the comparison.</p>`;
    }
  } catch (e) { showError("Model info load failed: " + e.message); }
}

function renderComparisonTable(rows, championRunId) {
  if (!rows?.length) return;
  const host = document.getElementById("comparison-table");
  const cols = Object.keys(rows[0]).filter((k) => k !== "index");
  let html = `<table><thead><tr>${cols.map((c) => `<th>${esc(c.replace(/_/g, " "))}</th>`).join("")}</tr></thead><tbody>`;
  rows.forEach((row) => {
    const isChamp = row.model === rows.find((r) =>
      parseFloat(r.cv_roc_auc_mean) === Math.max(...rows.map((x) => parseFloat(x.cv_roc_auc_mean) || 0))
    )?.model;
    html += `<tr class="${isChamp ? "champion" : ""}">`;
    html += cols.map((c) => {
      const v = row[c];
      return `<td>${typeof v === "number" ? v.toFixed(4) : esc(String(v ?? "—"))}</td>`;
    }).join("");
    html += `</tr>`;
  });
  html += "</tbody></table>";
  host.innerHTML = html;
}

function renderLearningCurve(lc) {
  const host = document.getElementById("learning-curve-chart");
  if (!lc?.train_sizes?.length) {
    host.innerHTML = `<p class="note">Learning curve data will appear after training.</p>`;
    return;
  }
  const { train_sizes, train, cv } = lc;
  const W = 520, H = 160, padL = 40, padB = 28, padR = 20;
  const allVals = [...train, ...cv];
  const minV = Math.min(...allVals) * 0.97;
  const maxV = Math.max(...allVals) * 1.005;
  const xOf = (sz) => padL + ((sz - train_sizes[0]) / (train_sizes.at(-1) - train_sizes[0])) * (W - padL - padR);
  const yOf = (v) => H - padB - ((v - minV) / (maxV - minV)) * (H - padB - 10);

  let svg = `<svg viewBox="0 0 ${W} ${H}" style="width:100%" aria-hidden="true">`;
  // Grid
  [0.25, 0.5, 0.75, 1].forEach((t) => {
    const y = H - padB - t * (H - padB - 10);
    svg += `<line x1="${padL}" y1="${y}" x2="${W - padR}" y2="${y}" stroke="#252d55" stroke-width=".5"/>`;
    svg += `<text x="${padL - 4}" y="${y + 3}" fill="#7b87b8" font-size="9" text-anchor="end">${(minV + t * (maxV - minV)).toFixed(2)}</text>`;
  });
  // Train line
  const trainPts = train_sizes.map((sz, i) => `${xOf(sz)},${yOf(train[i])}`).join(" ");
  svg += `<polyline points="${trainPts}" fill="none" stroke="#7c6cff" stroke-width="2" stroke-linejoin="round"/>`;
  // CV line
  const cvPts = train_sizes.map((sz, i) => `${xOf(sz)},${yOf(cv[i])}`).join(" ");
  svg += `<polyline points="${cvPts}" fill="none" stroke="#34d399" stroke-width="2" stroke-linejoin="round"/>`;
  // Dots
  train_sizes.forEach((sz, i) => {
    svg += `<circle cx="${xOf(sz)}" cy="${yOf(train[i])}" r="3" fill="#7c6cff"/>`;
    svg += `<circle cx="${xOf(sz)}" cy="${yOf(cv[i])}" r="3" fill="#34d399"/>`;
    svg += `<text x="${xOf(sz)}" y="${H}" fill="#7b87b8" font-size="9" text-anchor="middle">${Math.round(sz)}</text>`;
  });
  // Legend
  svg += `<circle cx="${W - padR - 70}" cy="14" r="4" fill="#7c6cff"/>`;
  svg += `<text x="${W - padR - 62}" y="18" fill="#7b87b8" font-size="9">Train</text>`;
  svg += `<circle cx="${W - padR - 30}" cy="14" r="4" fill="#34d399"/>`;
  svg += `<text x="${W - padR - 22}" y="18" fill="#7b87b8" font-size="9">CV</text>`;
  svg += "</svg>";
  host.innerHTML = svg;
}

function renderRunsTable(runs, currentRunId) {
  const host = document.getElementById("runs-table");
  if (!runs?.length) { host.innerHTML = `<p class="note">No runs recorded yet.</p>`; return; }
  let html = `<table><thead><tr>
    <th>Run ID</th><th>Model</th><th>CV AUC</th><th>F1</th><th>Calibration</th>
  </tr></thead><tbody>`;
  runs.forEach((r) => {
    const isCurrent = r.run_id === currentRunId;
    html += `<tr class="${isCurrent ? "champion" : ""}">
      <td style="font-size:11px;font-family:monospace">${esc(r.run_id || "—")}</td>
      <td>${esc(r.model_name || "—")}</td>
      <td>${fmt(r.cv_roc_auc_mean)}</td>
      <td>${fmt(r.test_metrics?.f1)}</td>
      <td>${esc(r.calibration || "none")}</td>
    </tr>`;
  });
  html += "</tbody></table>";
  host.innerHTML = html;
}

// ── Spinner keyframe (injected) ───────────────────────────────────────────────
const style = document.createElement("style");
style.textContent = `@keyframes spin { to { transform: rotate(360deg); } }`;
document.head.appendChild(style);

// ── Init ──────────────────────────────────────────────────────────────────────
loadMeta();
loadDrift();