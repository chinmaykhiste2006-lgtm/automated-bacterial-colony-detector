// 🎯 Fixed dataset colors (same as your Kaggle chart)
const CLASS_COLORS = {
    "B.subtilis": "#1f77b4",
    "C.albicans": "#17becf",
    "Contamination": "#2ca02c",
    "E.coli": "#0d1b6b",
    "P.aeruginosa": "#e377c2",
    "S.aureus": "#ff4d4d"
};

// fallback
function getColor(cls) {
    return CLASS_COLORS[cls] || "#ffffff";
}

// 🎨 Color Reference Section
function renderColorReference() {
    const ref = document.getElementById("color-ref");

    ref.innerHTML = Object.entries(CLASS_COLORS).map(([cls, color]) => `
        <div class="ref-item">
            <div class="ref-color" style="background:${color}"></div>
            <b>${cls}</b> → ${color}
        </div>
    `).join("");
}

// 🚀 Main Analysis
async function analyze() {
    const file = document.getElementById("file").files[0];
    const mode = document.getElementById("mode").value;

    if (!file) return alert("Please upload image");

    const fd = new FormData();
    fd.append("file", file);
    fd.append("mode", mode);

    const res = await fetch("http://localhost:8000/analyze", {
        method: "POST",
        body: fd
    });

    const data = await res.json();

    if (data.error) return alert(data.error);

    // Image
    document.getElementById("res-img").src = data.image_base64;

    // Count
    document.getElementById("count").innerText = data.total_colonies;

    // Table
    const tbody = document.querySelector("tbody");

    tbody.innerHTML = data.detections.map(d => `
        <tr>
            <td>${d.id}</td>
            <td style="color:${getColor(d.class)}"><b>${d.class}</b></td>
            <td>${d.confidence.toFixed(3)}</td>
            <td>${d.bbox.join(", ")}</td>
        </tr>
    `).join("");
}

// Load reference on start
window.onload = () => {
    renderColorReference();
};