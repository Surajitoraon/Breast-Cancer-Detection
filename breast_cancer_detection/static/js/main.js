/**
 * main.js
 * --------
 * Client-side logic for MammoAI.
 * Handles:
 *  - Drag-and-drop + click-to-browse file selection
 *  - Image preview
 *  - Async fetch to /predict
 *  - Loading / result / error states
 */

"use strict";

// ── DOM refs ─────────────────────────────────────────────────────
const dropZone      = document.getElementById("drop-zone");
const fileInput     = document.getElementById("file-input");
const previewSec    = document.getElementById("preview-section");
const previewImg    = document.getElementById("preview-img");
const previewName   = document.getElementById("preview-name");
const clearBtn      = document.getElementById("clear-btn");
const analyzeBtn    = document.getElementById("analyze-btn");
const resultSec     = document.getElementById("result-section");
const resultBadge   = document.getElementById("result-badge");
const confidenceBar = document.getElementById("confidence-bar");
const confidenceVal = document.getElementById("confidence-value");
const probBar       = document.getElementById("prob-bar");
const probVal       = document.getElementById("prob-value");
const errorCard     = document.getElementById("error-card");
const errorMsg      = document.getElementById("error-msg");
const errorDismiss  = document.getElementById("error-dismiss");
const resetBtn      = document.getElementById("reset-btn");

/** Currently selected File object */
let selectedFile = null;

// ── Utility helpers ───────────────────────────────────────────────

function show(el) { el.classList.remove("hidden"); }
function hide(el) { el.classList.add("hidden"); }

function showError(msg) {
  errorMsg.textContent = msg;
  show(errorCard);
}
function hideError() { hide(errorCard); }

function setLoading(isLoading) {
  if (isLoading) {
    analyzeBtn.classList.add("btn--loading");
    analyzeBtn.disabled = true;
  } else {
    analyzeBtn.classList.remove("btn--loading");
    analyzeBtn.disabled = false;
  }
}

/** Animate a progress bar to `percent` (0–100) */
function animateBar(barEl, percent) {
  // Reset first so animation always fires
  barEl.style.width = "0%";
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      barEl.style.width = `${percent}%`;
    });
  });
}

// ── File selection ────────────────────────────────────────────────

function handleFile(file) {
  if (!file) return;

  const allowed = /\.(png|jpe?g|bmp|tiff?|avif)$/i;
  if (!allowed.test(file.name)) {
    showError(`Unsupported file type "${file.name}". Please upload a PNG, JPG, BMP, TIFF, or AVIF.`);
    return;
  }

  selectedFile = file;
  hideError();
  hide(resultSec);

  // Show preview
  const url = URL.createObjectURL(file);
  previewImg.src = url;
  previewImg.onload = () => URL.revokeObjectURL(url);
  previewName.textContent = file.name;
  show(previewSec);
  show(analyzeBtn);
}

// Drop-zone click → open file picker
dropZone.addEventListener("click", () => fileInput.click());
dropZone.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileInput.click(); }
});

fileInput.addEventListener("change", () => {
  if (fileInput.files.length) handleFile(fileInput.files[0]);
});

// Drag & drop events
["dragenter", "dragover"].forEach(evt =>
  dropZone.addEventListener(evt, (e) => { e.preventDefault(); dropZone.classList.add("drag-over"); })
);
["dragleave", "dragend"].forEach(evt =>
  dropZone.addEventListener(evt, () => dropZone.classList.remove("drag-over"))
);
dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.classList.remove("drag-over");
  if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
});

// Clear button
clearBtn.addEventListener("click", resetUI);

// ── Analyze ───────────────────────────────────────────────────────

analyzeBtn.addEventListener("click", async () => {
  if (!selectedFile) return;

  hideError();
  hide(resultSec);
  setLoading(true);

  const formData = new FormData();
  formData.append("file", selectedFile);

  try {
    const res = await fetch("/predict", { method: "POST", body: formData });
    const data = await res.json();

    if (!res.ok) {
      showError(data.error || `Server error (${res.status}). Please try again.`);
      return;
    }

    showResult(data);
  } catch (err) {
    showError("Network error — could not reach the server. Is the Flask app running?");
    console.error(err);
  } finally {
    setLoading(false);
  }
});

// ── Show result ───────────────────────────────────────────────────

function showResult(data) {
  const { predicted_class, probability_malignant, confidence } = data;

  // Badge
  const isMalignant = predicted_class.toLowerCase() === "malignant";
  resultBadge.textContent = predicted_class.toUpperCase();
  resultBadge.className = "result__badge " +
    (isMalignant ? "result__badge--malignant" : "result__badge--benign");

  // Bars (animate after element is visible)
  show(resultSec);
  setTimeout(() => {
    animateBar(confidenceBar, confidence);
    confidenceVal.textContent = `${confidence.toFixed(1)}%`;

    animateBar(probBar, probability_malignant);
    probVal.textContent = `${probability_malignant.toFixed(1)}%`;
  }, 50);

  // Scroll into view
  resultSec.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

// ── Reset ─────────────────────────────────────────────────────────

function resetUI() {
  selectedFile = null;
  fileInput.value = "";
  previewImg.src = "";
  previewName.textContent = "";
  hide(previewSec);
  hide(analyzeBtn);
  hide(resultSec);
  hideError();
  analyzeBtn.classList.remove("btn--loading");
  analyzeBtn.disabled = false;
}

resetBtn.addEventListener("click", resetUI);
errorDismiss.addEventListener("click", hideError);
