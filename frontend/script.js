// ── Fixed dataset colors ──
const CLASS_COLORS = {
    "B.subtilis":    "#1f77b4",
    "C.albicans":    "#17becf",
    "Contamination": "#2ca02c",
    "E.coli":        "#0d1b6b",
    "P.aeruginosa":  "#e377c2",
    "S.aureus":      "#ff4d4d"
};

function getColor(cls) {
    return CLASS_COLORS[cls] || "#ffffff";
}

// ── Store last result for report ──
let _lastData     = null;
let _lastImageSrc = null;

// ── Color Reference ──
function renderColorReference() {
    const ref = document.getElementById("color-ref");
    ref.innerHTML = Object.entries(CLASS_COLORS).map(([cls, color]) => `
        <div class="ref-item">
            <div class="dot" style="background:${color}"></div>
            <span>${cls}</span>
        </div>
    `).join("");
}

// ── Class Breakdown Chips ──
// Uses class_counts directly from backend response (already computed there)
function renderSummary(classCounts) {
    const grid = document.getElementById("summary-grid");
    const sorted = Object.entries(classCounts).sort((a, b) => b[1] - a[1]);

    grid.innerHTML = sorted.map(([cls, n]) => `
        <div class="summary-chip">
            <div class="chip-dot" style="background:${getColor(cls)}"></div>
            <span class="chip-label">${cls}</span>
            <span class="chip-count">${n}</span>
        </div>
    `).join("");

    document.getElementById("summary-section").style.display = "block";
}

// ── Set loading state ──
function setLoading(on) {
    const btn     = document.getElementById("run-btn");
    const text    = document.getElementById("btn-text");
    const spinner = document.getElementById("spinner");
    btn.disabled       = on;
    text.textContent   = on ? "Analyzing…" : "Run Analysis";
    spinner.style.display = on ? "block" : "none";
}

// ── Main Analysis ──
async function analyze() {
    const file = document.getElementById("file").files[0];
    const mode = document.getElementById("mode").value;
    if (!file) { alert("Please upload an image first."); return; }

    setLoading(true);

    const fd = new FormData();
    fd.append("file", file);
    fd.append("mode", mode);

    let data;
    try {
        const res = await fetch("http://localhost:8000/analyze", {
            method: "POST",
            body:   fd
        });
        data = await res.json();
    } catch (e) {
        setLoading(false);
        alert("Could not reach backend at localhost:8000. Make sure the server is running.");
        return;
    }

    setLoading(false);

    if (data.error) { alert("Backend error: " + data.error); return; }

    // Image
    const img = document.getElementById("res-img");
    img.src = data.image_base64;
    img.style.display = "block";
    document.getElementById("img-placeholder").style.display = "none";

    // Total count
    document.getElementById("count").innerText = data.total_colonies;

    // Class breakdown chips  (use backend-computed class_counts)
    renderSummary(data.class_counts);

    // Table rows
    const tbody = document.getElementById("det-tbody");
    tbody.innerHTML = data.detections.map(d => `
        <tr>
            <td>${d.id}</td>
            <td style="color:${getColor(d.class)}"><b>${d.class}</b></td>
            <td>${d.confidence.toFixed(3)}</td>
            <td>${d.bbox.join(", ")}</td>
        </tr>
    `).join("");

    // Store for report
    _lastData     = data;
    _lastImageSrc = data.image_base64;
    document.getElementById("dl-btn").disabled = false;
}

// ── Download Report ──
function downloadReport() {
    if (!_lastData) return;

    const { total_colonies, detections, class_counts, mode } = _lastData;
    const modeLabel = document.getElementById("mode").selectedOptions[0].text;
    const timestamp = new Date().toLocaleString();

    // Class breakdown HTML for report
    const breakdownHTML = Object.entries(class_counts)
        .sort((a, b) => b[1] - a[1])
        .map(([cls, n]) => `
            <div style="display:flex;align-items:center;gap:12px;padding:10px 16px;
                        background:#f5f7fa;border-radius:8px;margin:5px 0;">
                <div style="width:12px;height:12px;border-radius:50%;background:${getColor(cls)};flex-shrink:0;"></div>
                <span style="flex:1;font-weight:600;color:#1a1a2e;font-size:0.95rem;">${cls}</span>
                <span style="font-family:monospace;font-weight:700;color:#f97316;font-size:1.1rem;">${n}</span>
            </div>
        `).join("");

    // Table rows for report
    const rowsHTML = detections.map(d => `
        <tr>
            <td style="padding:10px 12px;border-bottom:1px solid #eee;color:#555;">${d.id}</td>
            <td style="padding:10px 12px;border-bottom:1px solid #eee;font-weight:600;color:${getColor(d.class)};">${d.class}</td>
            <td style="padding:10px 12px;border-bottom:1px solid #eee;font-family:monospace;">${d.confidence.toFixed(3)}</td>
            <td style="padding:10px 12px;border-bottom:1px solid #eee;font-family:monospace;font-size:0.85rem;">${d.bbox.join(", ")}</td>
        </tr>
    `).join("");

    const html = `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Colony Analysis Report - ${timestamp}</title>
<link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;700&display=swap" rel="stylesheet">
<style>
  *{margin:0;padding:0;box-sizing:border-box;}
  body{font-family:'DM Sans',sans-serif;background:#fff;color:#1a1a2e;padding:48px;max-width:960px;margin:auto;}
  header{border-bottom:3px solid #f97316;padding-bottom:22px;margin-bottom:32px;display:flex;justify-content:space-between;align-items:flex-end;flex-wrap:wrap;gap:12px;}
  .brand{font-family:'Space Mono',monospace;color:#f97316;font-size:0.75rem;letter-spacing:0.14em;margin-bottom:6px;}
  .report-title{font-family:'Space Mono',monospace;font-size:1.6rem;color:#0a0e1a;}
  .report-meta{font-size:0.82rem;color:#888;text-align:right;line-height:2;}
  section{margin-bottom:36px;}
  h2{font-family:'Space Mono',monospace;font-size:0.78rem;letter-spacing:0.14em;color:#f97316;margin-bottom:14px;text-transform:uppercase;border-bottom:1px solid #f0f0f0;padding-bottom:8px;}
  .total-badge{
    display:inline-block;background:linear-gradient(135deg,#f97316,#fb923c);
    color:#fff;font-family:'Space Mono',monospace;font-size:2.6rem;font-weight:700;
    padding:20px 40px;border-radius:16px;letter-spacing:0.04em;
  }
  .detection-img{max-width:100%;border-radius:12px;border:2px solid #eee;display:block;}
  table{width:100%;border-collapse:collapse;font-size:0.9rem;}
  thead tr{background:#0a0e1a;}
  th{color:#e2eaf6;padding:13px 12px;text-align:left;font-size:0.78rem;letter-spacing:0.06em;font-weight:600;}
  tr:nth-child(even) td{background:#f9f9fb;}
  footer{margin-top:48px;padding-top:16px;border-top:1px solid #eee;font-size:0.76rem;color:#aaa;font-family:'Space Mono',monospace;text-align:center;}
  @media print{body{padding:20px;}button{display:none;}}
</style>
</head>
<body>

<header>
  <div>
    <div class="brand">🧫 COLONY ANALYZER · EDI2 PROJECT</div>
    <div class="report-title">Detection Report</div>
  </div>
  <div class="report-meta">
    <div><b>Generated:</b> ${timestamp}</div>
    <div><b>Mode:</b> ${modeLabel}</div>
    <div><b>Total Colonies:</b> ${total_colonies}</div>
  </div>
</header>

<section>
  <h2>Total Colonies Detected</h2>
  <div class="total-badge">${total_colonies}</div>
</section>

<section>
  <h2>Class Breakdown</h2>
  ${breakdownHTML}
</section>

<section>
  <h2>Detection Image</h2>
  <img class="detection-img" src="${_lastImageSrc}" alt="Detection Output">
</section>

<section>
  <h2>Detection Table (${total_colonies} colonies)</h2>
  <table>
    <thead>
      <tr><th>ID</th><th>Class</th><th>Confidence</th><th>Bounding Box</th></tr>
    </thead>
    <tbody>${rowsHTML}</tbody>
  </table>
</section>

<footer>Colony Analyzer · EDI2 Project · YOLOv8 + UNet · Auto-generated ${timestamp}</footer>

</body>
</html>`;

    const blob = new Blob([html], { type: "text/html" });
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement("a");
    a.href     = url;
    a.download = `colony-report-${Date.now()}.html`;
    a.click();
    URL.revokeObjectURL(url);
}

// Init
window.onload = renderColorReference;