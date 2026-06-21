// Configure Chart.js globally for high contrast dark-mode readability
if (typeof Chart !== 'undefined') {
  Chart.defaults.color = '#e2e8f0'; // slate-200 (tick labels, general text)
  Chart.defaults.borderColor = 'rgba(255, 255, 255, 0.12)'; // subtle grid lines
  Chart.defaults.font.family = "'Space Grotesk', system-ui, -apple-system, sans-serif";
  Chart.defaults.font.size = 12;

  // Customizing Legends
  Chart.defaults.plugins.legend.labels.color = '#cbd5e1'; // slate-300
  Chart.defaults.plugins.legend.labels.boxWidth = 15;
  Chart.defaults.plugins.legend.labels.font = {
    family: "'Space Grotesk', system-ui, -apple-system, sans-serif",
    size: 13,
    weight: '600'
  };

  // Customizing Tooltips
  Chart.defaults.plugins.tooltip.titleColor = '#ffffff';
  Chart.defaults.plugins.tooltip.titleFont = {
    family: "'Space Grotesk', system-ui, -apple-system, sans-serif",
    size: 13,
    weight: '700'
  };
  Chart.defaults.plugins.tooltip.bodyColor = '#f1f5f9'; // slate-100
  Chart.defaults.plugins.tooltip.bodyFont = {
    family: "'Space Grotesk', system-ui, -apple-system, sans-serif",
    size: 12
  };
  Chart.defaults.plugins.tooltip.padding = 10;
  Chart.defaults.plugins.tooltip.cornerRadius = 6;
}

const charts = {};

function showResult(payload) {
  const box = document.getElementById("resultBox");
  
  // Create a more readable display instead of raw JSON
  let html = '';
  
  if (payload.prediction_label) {
    const isPhishing = payload.prediction_label === 'Phishing';
    const confidence = payload.confidence ? (payload.confidence * 100).toFixed(2) : 'N/A';
    const probability = payload.phishing_probability !== undefined ? (payload.phishing_probability * 100).toFixed(2) : confidence;
    
    html += `<div style="padding: 15px; border-radius: 8px; margin-bottom: 10px; background: ${isPhishing ? 'rgba(239, 68, 68, 0.15)' : 'rgba(16, 185, 129, 0.15)'}; border: 2px solid ${isPhishing ? '#ef4444' : '#10b981'};">`;
    html += `<h3 style="color: ${isPhishing ? '#ef4444' : '#10b981'}; margin: 0 0 10px 0;">${payload.prediction_label}</h3>`;
    html += `<div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px;">`;
    html += `<div><strong>Confidence:</strong> ${confidence}%</div>`;
    html += `<div><strong>Probability:</strong> ${probability}%</div>`;
    
    if (payload.response_time_ms) {
      html += `<div><strong>Response Time:</strong> ${payload.response_time_ms.toFixed(2)} ms</div>`;
    }
    
    if (payload.source_type) {
      html += `<div><strong>Source:</strong> ${payload.source_type}</div>`;
    }
    
    html += `</div></div>`;
  }
  
  // Show additional details
  if (payload.debug) {
    html += `<div style="padding: 15px; border-radius: 8px; background: rgba(59, 130, 246, 0.1); border: 1px solid rgba(59, 130, 246, 0.3);">`;
    html += `<h4 style="margin: 0 0 10px 0; color: #3b82f6;">Debug Information</h4>`;
    
    if (payload.debug.url_probability !== undefined) {
      html += `<div><strong>URL Probability:</strong> ${(payload.debug.url_probability * 100).toFixed(2)}%</div>`;
    }
    if (payload.debug.qr_model_probability !== undefined) {
      html += `<div><strong>QR Model Probability:</strong> ${(payload.debug.qr_model_probability * 100).toFixed(2)}%</div>`;
    }
    if (payload.debug.fused_probability !== undefined) {
      html += `<div><strong>Fused Probability:</strong> ${(payload.debug.fused_probability * 100).toFixed(2)}%</div>`;
    }
    if (payload.debug.decision_threshold !== undefined) {
      html += `<div><strong>Decision Threshold:</strong> ${payload.debug.decision_threshold}</div>`;
    }
    if (payload.debug.model_status) {
      html += `<div><strong>Model Status:</strong> ${payload.debug.model_status}</div>`;
    }
    if (payload.debug.fusion_weights) {
      html += `<div><strong>Fusion Weights:</strong> URL: ${payload.debug.fusion_weights.url || 0}, QR: ${payload.debug.fusion_weights.qr || 0}</div>`;
    }
    
    html += `</div>`;
  }
  
  // Show raw JSON for debugging (collapsible)
  html += `<details style="margin-top: 10px;">`;
  html += `<summary style="cursor: pointer; color: #64748b; font-size: 12px;">Show Raw JSON</summary>`;
  html += `<pre style="margin-top: 10px; padding: 10px; background: rgba(0,0,0,0.05); border-radius: 4px; font-size: 11px; overflow-x: auto;">${JSON.stringify(payload, null, 2)}</pre>`;
  html += `</details>`;
  
  box.innerHTML = html;
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
          backgroundColor: "rgba(16,185,129,0.8)",
        },
      ],
    },
    options: { 
      responsive: true, 
      scales: { 
        y: { 
          min: 0, 
          max: 1,
          ticks: { color: '#64748b' },
          grid: { color: 'rgba(0,0,0,0.1)' }
        },
        x: {
          ticks: { color: '#64748b' },
          grid: { color: 'rgba(0,0,0,0.1)' }
        }
      },
      plugins: {
        legend: { labels: { color: '#1e293b' } }
      }
    },
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
          backgroundColor: "rgba(239,68,68,0.75)",
        },
        { label: "F1", data: mm.f1, backgroundColor: "rgba(16,185,129,0.75)" },
      ],
    },
    options: { 
      responsive: true, 
      scales: { 
        y: { 
          min: 0, 
          max: 1,
          ticks: { color: '#64748b' },
          grid: { color: 'rgba(0,0,0,0.1)' }
        },
        x: {
          ticks: { color: '#64748b' },
          grid: { color: 'rgba(0,0,0,0.1)' }
        }
      },
      plugins: {
        legend: { labels: { color: '#1e293b' } }
      }
    },
  });

  upsertChart("roc", "rocChart", {
    type: "line",
    data: {
      labels: mm.models,
      datasets: [
        {
          label: "ROC-AUC",
          data: mm.roc_auc,
          borderColor: "rgba(16,185,129,0.95)",
          backgroundColor: "rgba(16,185,129,0.35)",
          tension: 0.35,
          fill: true,
        },
      ],
    },
    options: { 
      responsive: true, 
      scales: { 
        y: { 
          min: 0, 
          max: 1,
          ticks: { color: '#64748b' },
          grid: { color: 'rgba(0,0,0,0.1)' }
        },
        x: {
          ticks: { color: '#64748b' },
          grid: { color: 'rgba(0,0,0,0.1)' }
        }
      },
      plugins: {
        legend: { labels: { color: '#1e293b' } }
      }
    },
  });

  const confusionData = buildConfusionDataset(data.confusion);
  upsertChart("confusion", "confusionChart", {
    type: "bar",
    data: confusionData,
    options: { 
      responsive: true, 
      scales: { 
        y: { 
          beginAtZero: true,
          ticks: { color: '#64748b' },
          grid: { color: 'rgba(0,0,0,0.1)' }
        },
        x: {
          ticks: { color: '#64748b' },
          grid: { color: 'rgba(0,0,0,0.1)' }
        }
      },
      plugins: {
        legend: { labels: { color: '#1e293b' } }
      }
    },
  });

  const labels = Object.keys(data.label_distribution);
  const values = Object.values(data.label_distribution);
  upsertChart("distribution", "distributionChart", {
    type: "doughnut",
    data: {
      labels,
      datasets: [
        { data: values, backgroundColor: ["#ef4444", "#10b981", "#3b82f6"] },
      ],
    },
    options: { 
      responsive: true,
      plugins: {
        legend: { labels: { color: '#1e293b' } }
      }
    },
  });

  upsertChart("trend", "trendChart", {
    type: "line",
    data: {
      labels: data.trends.map((x) => x.date),
      datasets: [
        {
          label: "Detections Over Time",
          data: data.trends.map((x) => x.count),
          borderColor: "rgba(16,185,129,0.9)",
          backgroundColor: "rgba(16,185,129,0.2)",
          tension: 0.35,
          fill: true,
        },
      ],
    },
    options: { 
      responsive: true,
      scales: {
        y: {
          ticks: { color: '#64748b' },
          grid: { color: 'rgba(0,0,0,0.1)' }
        },
        x: {
          ticks: { color: '#64748b' },
          grid: { color: 'rgba(0,0,0,0.1)' }
        }
      },
      plugins: {
        legend: { labels: { color: '#1e293b' } }
      }
    },
  });

  upsertChart("response", "responseChart", {
    type: "bar",
    data: {
      labels: data.response_times.map((x) => x.source),
      datasets: [
        {
          label: "Avg Response Time (ms)",
          data: data.response_times.map((x) => x.avg_ms),
          backgroundColor: "rgba(59,130,246,0.75)",
        },
      ],
    },
    options: { 
      responsive: true,
      scales: {
        y: {
          ticks: { color: '#64748b' },
          grid: { color: 'rgba(0,0,0,0.1)' }
        },
        x: {
          ticks: { color: '#64748b' },
          grid: { color: 'rgba(0,0,0,0.1)' }
        }
      },
      plugins: {
        legend: { labels: { color: '#1e293b' } }
      }
    },
  });

  updateRecentTable(data.recent_logs);
}

function detectInputType(content) {
  const trimmedContent = content.trim();

  // URL detection - check if content is primarily a URL
  // More flexible regex that allows URLs with or without trailing whitespace
  const urlPattern = /^https?:\/\/[^\s]+/i;
  const wwwPattern = /^www\.[^\s]+/i;
  if (urlPattern.test(trimmedContent) || wwwPattern.test(trimmedContent)) {
    return "url";
  }

  // Email detection (contains @, has email-like structure)
  if (trimmedContent.includes("@") && trimmedContent.match(/^[^\s@]+@[^\s@]+\.[^\s@]+$/)) {
    return "email";
  }

  // SMS detection (short text, typically < 160 chars, no @, no URL pattern)
  if (trimmedContent.length < 160 && !trimmedContent.includes("@") && !urlPattern.test(trimmedContent) && !wwwPattern.test(trimmedContent)) {
    return "sms";
  }

  // Default to email for longer text
  return "email";
}

function detectFileType(file) {
  if (!file) return null;

  const fileType = file.type.toLowerCase();
  const fileName = file.name.toLowerCase();

  // QR code detection (image files)
  if (fileType.startsWith("image/") || fileName.match(/\.(png|jpg|jpeg|gif|bmp|webp)$/i)) {
    return "qr";
  }

  // Deepfake detection (video files)
  if (fileType.startsWith("video/") || fileName.match(/\.(mp4|avi|mov|mkv|webm|flv)$/i)) {
    return "deepfake";
  }

  // Voice detection (audio files)
  if (fileType.startsWith("audio/") || fileName.match(/\.(mp3|wav|ogg|flac|m4a|aac)$/i)) {
    return "voice";
  }

  return null;
}

function attachHandlers() {
  // Handle input type selection change
  document.getElementById("inputType").addEventListener("change", (e) => {
    const inputType = e.target.value;
    const textContainer = document.getElementById("textInputContainer");
    const fileContainer = document.getElementById("fileInputContainer");
    const fileInput = document.getElementById("fileInput");
    
    if (inputType === "auto") {
      // Auto-detect mode - show text input by default
      textContainer.style.display = "block";
      fileContainer.style.display = "none";
    } else if (["qr", "deepfake", "voice"].includes(inputType)) {
      // File-based inputs
      textContainer.style.display = "none";
      fileContainer.style.display = "block";
      
      // Update file accept attribute
      if (inputType === "qr") {
        fileInput.accept = "image/*";
      } else if (inputType === "deepfake") {
        fileInput.accept = "video/*,image/*";
      } else if (inputType === "voice") {
        fileInput.accept = "audio/*";
      }
    } else {
      // Text-based inputs
      textContainer.style.display = "block";
      fileContainer.style.display = "none";
    }
  });

  // Unified form handler
  document.getElementById("unifiedForm").addEventListener("submit", async (e) => {
    e.preventDefault();
    const formData = new FormData(e.target);
    const inputType = formData.get("input_type");
    const textContent = formData.get("text_content");
    const fileContent = formData.get("file_content");
    
    let result;
    
    if (inputType === "auto") {
      // Auto-detect based on content
      if (fileContent && fileContent.size > 0) {
        // File-based auto-detection using improved detection
        const detectedFileType = detectFileType(fileContent);
        if (detectedFileType === "qr") {
          result = await postFormData("/api/detect/qr", formData);
        } else if (detectedFileType === "deepfake") {
          result = await postFormData("/api/detect/deepfake", formData);
        } else if (detectedFileType === "voice") {
          result = await postFormData("/api/detect/voice", formData);
        } else {
          showResult({ error: "Unsupported file type. Please upload an image (QR), video (deepfake), or audio (voice) file." });
          return;
        }
      } else if (textContent) {
        // Text-based auto-detection
        const detectedType = detectInputType(textContent);
        if (detectedType === "url") {
          result = await postJson("/api/detect/url", { url: textContent });
        } else if (detectedType === "email") {
          result = await postJson("/api/detect/email", { email_text: textContent });
        } else if (detectedType === "sms") {
          result = await postJson("/api/detect/sms", { sms_text: textContent });
        }
      } else {
        showResult({ error: "Please provide text or file content" });
        return;
      }
    } else if (inputType === "url") {
      result = await postJson("/api/detect/url", { url: textContent });
    } else if (inputType === "email") {
      result = await postJson("/api/detect/email", { email_text: textContent });
    } else if (inputType === "sms") {
      result = await postJson("/api/detect/sms", { sms_text: textContent });
    } else if (inputType === "qr") {
      const qrFormData = new FormData();
      qrFormData.append("qr_file", fileContent);
      result = await postFormData("/api/detect/qr", qrFormData);
    } else if (inputType === "deepfake") {
      const deepfakeFormData = new FormData();
      deepfakeFormData.append("media_file", fileContent);
      result = await postFormData("/api/detect/deepfake", deepfakeFormData);
    } else if (inputType === "voice") {
      const voiceFormData = new FormData();
      voiceFormData.append("voice_file", fileContent);
      result = await postFormData("/api/detect/voice", voiceFormData);
    }
    
    showResult(result);
    refreshDashboard();
  });

  document.getElementById("voiceForm").addEventListener("submit", async (e) => {
    e.preventDefault();
    const formData = new FormData(e.target);
    const result = await postFormData("/api/detect/voice", formData);
    showResult(result);
    refreshDashboard();
  });
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
      };

      // Handle video (deepfake) upload
      const mediaFile = formData.get("fusion_media_file");
      if (mediaFile && mediaFile.size > 0) {
        const mediaForm = new FormData();
        mediaForm.append("media_file", mediaFile);
        const deepfakeRes = await postFormData(
          "/api/detect/deepfake",
          mediaForm,
        );
        if (
          deepfakeRes &&
          typeof deepfakeRes.phishing_probability === "number"
        ) {
          payload.deepfake_probability = deepfakeRes.phishing_probability;
        }
      }

      // Handle audio (voice) upload
      const voiceFile = formData.get("fusion_voice_file");
      if (voiceFile && voiceFile.size > 0) {
        const voiceForm = new FormData();
        voiceForm.append("voice_file", voiceFile);
        const voiceRes = await postFormData("/api/detect/voice", voiceForm);
        if (voiceRes && typeof voiceRes.phishing_probability === "number") {
          payload.voice_probability = voiceRes.phishing_probability;
        }
      }

      const result = await postJson("/api/detect/fusion", payload);
      showResult(result);
      refreshDashboard();
    });
}

window.addEventListener("DOMContentLoaded", () => {
  attachHandlers();
  refreshDashboard();
});
