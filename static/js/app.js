const charts = {};

function showResult(payload) {
  const box = document.getElementById("resultBox");
  box.textContent = JSON.stringify(payload, null, 2);
}

async function postJson(url, data) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
  return res.json();
}

async function postFormData(url, formData) {
  const res = await fetch(url, {
    method: "POST",
    body: formData,
  });
  return res.json();
}

function upsertChart(key, canvasId, config) {
  if (charts[key]) {
    charts[key].destroy();
  }
  const ctx = document.getElementById(canvasId);
  charts[key] = new Chart(ctx, config);
}

function updateRecentTable(rows) {
  const tbody = document.getElementById("recentTableBody");
  tbody.innerHTML = "";

  rows.forEach((row) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
            <td>${row.source_type}</td>
            <td>${row.prediction_label}</td>
            <td>${(row.confidence * 100).toFixed(2)}%</td>
            <td>${row.response_time_ms.toFixed(2)}</td>
            <td>${new Date(row.created_at).toLocaleString()}</td>
        `;
    tbody.appendChild(tr);
  });
}

function buildConfusionDataset(confusion) {
  const models = Object.keys(confusion);
  return {
    labels: models,
    datasets: [
      {
        label: "TP",
        data: models.map((m) => confusion[m].tp),
        backgroundColor: "rgba(34,197,94,0.7)",
      },
      {
        label: "TN",
        data: models.map((m) => confusion[m].tn),
        backgroundColor: "rgba(56,189,248,0.7)",
      },
      {
        label: "FP",
        data: models.map((m) => confusion[m].fp),
        backgroundColor: "rgba(245,158,11,0.7)",
      },
      {
        label: "FN",
        data: models.map((m) => confusion[m].fn),
        backgroundColor: "rgba(239,68,68,0.7)",
      },
    ],
  };
}

async function refreshDashboard() {
  const res = await fetch("/api/dashboard-data");
  const data = await res.json();

  const mm = data.model_metrics;

  upsertChart("accuracy", "accuracyChart", {
    type: "bar",
    data: {
      labels: mm.models,
      datasets: [
        {
          label: "Accuracy",
          data: mm.accuracy,
          backgroundColor: "rgba(20,184,166,0.8)",
        },
      ],
    },
    options: { responsive: true, scales: { y: { min: 0, max: 1 } } },
  });

  upsertChart("prf", "prfChart", {
    type: "bar",
    data: {
      labels: mm.models,
      datasets: [
        {
          label: "Precision",
          data: mm.precision,
          backgroundColor: "rgba(59,130,246,0.75)",
        },
        {
          label: "Recall",
          data: mm.recall,
          backgroundColor: "rgba(245,158,11,0.75)",
        },
        { label: "F1", data: mm.f1, backgroundColor: "rgba(168,85,247,0.75)" },
      ],
    },
    options: { responsive: true, scales: { y: { min: 0, max: 1 } } },
  });

  upsertChart("roc", "rocChart", {
    type: "line",
    data: {
      labels: mm.models,
      datasets: [
        {
          label: "ROC-AUC",
          data: mm.roc_auc,
          borderColor: "rgba(250,204,21,0.95)",
          backgroundColor: "rgba(250,204,21,0.35)",
          tension: 0.35,
          fill: true,
        },
      ],
    },
    options: { responsive: true, scales: { y: { min: 0, max: 1 } } },
  });

  const confusionData = buildConfusionDataset(data.confusion);
  upsertChart("confusion", "confusionChart", {
    type: "bar",
    data: confusionData,
    options: { responsive: true, scales: { y: { beginAtZero: true } } },
  });

  const labels = Object.keys(data.label_distribution);
  const values = Object.values(data.label_distribution);
  upsertChart("distribution", "distributionChart", {
    type: "doughnut",
    data: {
      labels,
      datasets: [
        { data: values, backgroundColor: ["#ef4444", "#22c55e", "#38bdf8"] },
      ],
    },
    options: { responsive: true },
  });

  upsertChart("trend", "trendChart", {
    type: "line",
    data: {
      labels: data.trends.map((x) => x.date),
      datasets: [
        {
          label: "Detections Over Time",
          data: data.trends.map((x) => x.count),
          borderColor: "rgba(14,165,233,0.9)",
          tension: 0.35,
        },
      ],
    },
    options: { responsive: true },
  });

  upsertChart("response", "responseChart", {
    type: "bar",
    data: {
      labels: data.response_times.map((x) => x.source),
      datasets: [
        {
          label: "Avg Response Time (ms)",
          data: data.response_times.map((x) => x.avg_ms),
          backgroundColor: "rgba(251,146,60,0.75)",
        },
      ],
    },
    options: { responsive: true },
  });

  updateRecentTable(data.recent_logs);
}

function attachHandlers() {
  document.getElementById("urlForm").addEventListener("submit", async (e) => {
    e.preventDefault();
    const formData = new FormData(e.target);
    const payload = { url: formData.get("url") };
    const result = await postJson("/api/detect/url", payload);
    showResult(result);
    refreshDashboard();
  });

  document.getElementById("emailForm").addEventListener("submit", async (e) => {
    e.preventDefault();
    const formData = new FormData(e.target);
    const payload = { email_text: formData.get("email_text") };
    const result = await postJson("/api/detect/email", payload);
    showResult(result);
    refreshDashboard();
  });

  document.getElementById("smsForm").addEventListener("submit", async (e) => {
    e.preventDefault();
    const formData = new FormData(e.target);
    const payload = { sms_text: formData.get("sms_text") };
    const result = await postJson("/api/detect/sms", payload);
    showResult(result);
    refreshDashboard();
  });

  document.getElementById("qrForm").addEventListener("submit", async (e) => {
    e.preventDefault();
    const formData = new FormData(e.target);
    const result = await postFormData("/api/detect/qr", formData);
    showResult(result);
    refreshDashboard();
  });

  document
    .getElementById("deepfakeForm")
    .addEventListener("submit", async (e) => {
      e.preventDefault();
      const formData = new FormData(e.target);
      const result = await postFormData("/api/detect/deepfake", formData);
      showResult(result);
      refreshDashboard();
    });

  document
    .getElementById("fusionForm")
    .addEventListener("submit", async (e) => {
      e.preventDefault();
      const formData = new FormData(e.target);
      const payload = {
        url: formData.get("url") || null,
        email_text: formData.get("email_text") || null,
        sms_text: formData.get("sms_text") || null,
        qr_url: formData.get("qr_url") || null,
        deepfake_probability: formData.get("deepfake_probability")
          ? Number(formData.get("deepfake_probability"))
          : null,
      };

      const result = await postJson("/api/detect/fusion", payload);
      showResult(result);
      refreshDashboard();
    });
}

window.addEventListener("DOMContentLoaded", () => {
  attachHandlers();
  refreshDashboard();
});
