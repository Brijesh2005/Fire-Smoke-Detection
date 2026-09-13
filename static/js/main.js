/**
 * Fire & Smoke Detection - Web Dashboard Controller
 * Handles image upload, sample gallery, video analysis, live webcam stream,
 * Web Audio API sirens, and API communication.
 */

document.addEventListener("DOMContentLoaded", () => {
  // ==========================================
  // 1. Audio Siren Synthesizer (Web Audio API)
  // ==========================================
  let audioCtx = null;
  let isMuted = false;
  let isAlarmPlaying = false;
  let alarmInterval = null;

  function initAudio() {
    if (!audioCtx) {
      const AudioContext = window.AudioContext || window.webkitAudioContext;
      if (AudioContext) {
        audioCtx = new AudioContext();
      }
    }
  }

  function playAlertBeep(frequency = 880, duration = 0.25) {
    if (isMuted) return;
    initAudio();
    if (!audioCtx) return;

    if (audioCtx.state === "suspended") {
      audioCtx.resume();
    }

    try {
      const osc = audioCtx.createOscillator();
      const gain = audioCtx.createGain();

      osc.type = "sawtooth";
      osc.frequency.setValueAtTime(frequency, audioCtx.currentTime);
      osc.frequency.exponentialRampToValueAtTime(frequency * 1.5, audioCtx.currentTime + duration);

      gain.gain.setValueAtTime(0.2, audioCtx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.01, audioCtx.currentTime + duration);

      osc.connect(gain);
      gain.connect(audioCtx.destination);

      osc.start();
      osc.stop(audioCtx.currentTime + duration);
    } catch (e) {
      console.warn("Audio alert error:", e);
    }
  }

  function triggerAlarmSiren() {
    if (isMuted || isAlarmPlaying) return;
    isAlarmPlaying = true;
    playAlertBeep(900, 0.3);

    alarmInterval = setTimeout(() => {
      isAlarmPlaying = false;
    }, 1200);
  }

  // Audio mute button toggle
  const audioToggleBtn = document.getElementById("audioToggleBtn");
  if (audioToggleBtn) {
    audioToggleBtn.addEventListener("click", () => {
      initAudio();
      isMuted = !isMuted;
      audioToggleBtn.classList.toggle("active", !isMuted);
      audioToggleBtn.innerHTML = isMuted
        ? "🔇 Alerts: Off"
        : "🔊 Alerts: On";
      if (!isMuted) {
        playAlertBeep(600, 0.15);
      }
    });
  }

  // ==========================================
  // 2. Tab Navigation
  // ==========================================
  const tabBtns = document.querySelectorAll(".tab-btn");
  const tabPanels = document.querySelectorAll(".tab-panel");

  tabBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      const targetId = btn.getAttribute("data-tab");

      tabBtns.forEach((b) => b.classList.remove("active"));
      tabPanels.forEach((p) => p.classList.remove("active"));

      btn.classList.add("active");
      const targetPanel = document.getElementById(targetId);
      if (targetPanel) {
        targetPanel.classList.add("active");
      }

      // Stop webcam stream if navigating away from Webcam tab
      if (targetId !== "webcamTabPanel" && isWebcamRunning) {
        stopWebcam();
      }
    });
  });

  // ==========================================
  // 3. System Health Check
  // ==========================================
  async function fetchHealth() {
    try {
      const res = await fetch("/api/health");
      if (res.ok) {
        const data = await res.json();
        const devElem = document.getElementById("deviceBadge");
        if (devElem) {
          devElem.textContent = `Device: ${data.device.toUpperCase()}`;
        }
        const modelElem = document.getElementById("modelBadge");
        if (modelElem) {
          modelElem.textContent = `Model: ${data.model}`;
        }
      }
    } catch (err) {
      console.warn("Healthcheck failed:", err);
    }
  }
  fetchHealth();

  // ==========================================
  // 4. Sample Images Gallery
  // ==========================================
  const samplesGrid = document.getElementById("samplesGrid");

  async function loadSamples() {
    if (!samplesGrid) return;
    try {
      const res = await fetch("/api/samples");
      if (!res.ok) return;
      const samples = await res.json();

      samplesGrid.innerHTML = "";
      samples.forEach((sample) => {
        const card = document.createElement("div");
        card.className = "sample-card";
        card.title = `Click to test ${sample.filename} (${sample.hint})`;
        card.innerHTML = `
          <img src="${sample.url}" alt="${sample.filename}" loading="lazy" />
          <div class="sample-label">${sample.hint}</div>
        `;
        card.addEventListener("click", () => {
          predictSampleImage(sample);
        });
        samplesGrid.appendChild(card);
      });
    } catch (e) {
      console.error("Error loading sample gallery:", e);
    }
  }
  loadSamples();

  async function predictSampleImage(sample) {
    const previewImg = document.getElementById("imagePreview");
    const previewContainer = document.getElementById("imagePreviewContainer");
    const placeholder = document.getElementById("previewPlaceholder");

    if (placeholder) placeholder.style.display = "none";
    if (previewImg) {
      previewImg.src = sample.url;
      previewImg.style.display = "block";
    }

    setResultLoading();

    try {
      const res = await fetch("/api/predict/image", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sample_filename: sample.filename }),
      });
      const data = await res.json();
      renderImageResult(data);
    } catch (e) {
      renderError("Failed to run prediction on sample image.");
    }
  }

  // ==========================================
  // 5. Image Upload & Drag-and-Drop
  // ==========================================
  const imageDropzone = document.getElementById("imageDropzone");
  const imageFileInput = document.getElementById("imageFileInput");
  const imagePreview = document.getElementById("imagePreview");
  const previewPlaceholder = document.getElementById("previewPlaceholder");

  if (imageDropzone && imageFileInput) {
    imageDropzone.addEventListener("click", () => imageFileInput.click());

    imageDropzone.addEventListener("dragover", (e) => {
      e.preventDefault();
      imageDropzone.classList.add("dragover");
    });

    imageDropzone.addEventListener("dragleave", () => {
      imageDropzone.classList.remove("dragover");
    });

    imageDropzone.addEventListener("drop", (e) => {
      e.preventDefault();
      imageDropzone.classList.remove("dragover");
      if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
        handleImageFile(e.dataTransfer.files[0]);
      }
    });

    imageFileInput.addEventListener("change", (e) => {
      if (e.target.files && e.target.files.length > 0) {
        handleImageFile(e.target.files[0]);
      }
    });
  }

  function handleImageFile(file) {
    if (!file.type.startsWith("image/")) {
      alert("Please upload a valid image file (PNG, JPG, WEBP).");
      return;
    }

    const reader = new FileReader();
    reader.onload = (event) => {
      if (previewPlaceholder) previewPlaceholder.style.display = "none";
      if (imagePreview) {
        imagePreview.src = event.target.result;
        imagePreview.style.display = "block";
      }
    };
    reader.readAsDataURL(file);

    uploadAndPredictImage(file);
  }

  async function uploadAndPredictImage(file) {
    setResultLoading();

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch("/api/predict/image", {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      if (data.status === "error") {
        renderError(data.message || "Prediction error");
      } else {
        renderImageResult(data);
      }
    } catch (err) {
      renderError("Failed to upload image for inference.");
    }
  }

  function setResultLoading() {
    const banner = document.getElementById("imageResultBanner");
    const heading = document.getElementById("hazardHeading");
    const sub = document.getElementById("hazardSub");
    const icon = document.getElementById("hazardIcon");
    const conf = document.getElementById("confidenceChip");

    if (banner) {
      banner.className = "result-banner";
    }
    if (heading) heading.textContent = "Analyzing Image...";
    if (sub) sub.textContent = "Running ResNet-50 PyTorch inference";
    if (icon) icon.textContent = "⏳";
    if (conf) conf.textContent = "--%";
  }

  function renderImageResult(data) {
    const banner = document.getElementById("imageResultBanner");
    const heading = document.getElementById("hazardHeading");
    const sub = document.getElementById("hazardSub");
    const icon = document.getElementById("hazardIcon");
    const conf = document.getElementById("confidenceChip");

    const pred = data.prediction || "Neutral";
    const confidence = data.confidence || 0.0;
    const isHazard = data.is_hazard;

    // Trigger audio alarm if fire or smoke
    if (isHazard) {
      triggerAlarmSiren();
    }

    // Update banner styling
    if (banner) {
      banner.className = "result-banner";
      if (pred === "Fire") {
        banner.classList.add("hazard-fire");
      } else if (pred === "Smoke") {
        banner.classList.add("hazard-smoke");
      } else {
        banner.classList.add("hazard-safe");
      }
    }

    // Update text
    if (heading) heading.textContent = `${pred.toUpperCase()} DETECTED`;
    if (sub) {
      sub.textContent = isHazard
        ? `EMERGENCY ALERT: ${data.hazard_level || "HAZARD"} condition detected.`
        : "STATUS NORMAL: No fire or smoke detected in scene.";
    }

    if (icon) {
      icon.textContent = pred === "Fire" ? "🔥" : pred === "Smoke" ? "💨" : "🛡️";
    }
    if (conf) {
      conf.textContent = `${confidence.toFixed(1)}%`;
      conf.style.color = data.color_hex || "#fff";
    }

    // Update distribution bars
    const probs = data.probabilities || {};
    updateProgressBar("probFire", "barFire", probs["Fire"] || 0.0);
    updateProgressBar("probSmoke", "barSmoke", probs["Smoke"] || 0.0);
    updateProgressBar("probNeutral", "barNeutral", probs["Neutral"] || 0.0);

    // Update metadata box
    const metaLatency = document.getElementById("metaLatency");
    if (metaLatency) {
      metaLatency.textContent = `Latency: ${data.latency_ms || "--"} ms`;
    }
    const metaSource = document.getElementById("metaSource");
    if (metaSource) {
      metaSource.textContent = `Source: ${data.filename || data.source || "upload"}`;
    }
  }

  function updateProgressBar(textId, barId, percentage) {
    const txt = document.getElementById(textId);
    const bar = document.getElementById(barId);
    if (txt) txt.textContent = `${percentage.toFixed(1)}%`;
    if (bar) bar.style.width = `${Math.min(100, Math.max(0, percentage))}%`;
  }

  function renderError(msg) {
    const heading = document.getElementById("hazardHeading");
    const sub = document.getElementById("hazardSub");
    const icon = document.getElementById("hazardIcon");
    if (heading) heading.textContent = "Inference Failed";
    if (sub) sub.textContent = msg;
    if (icon) icon.textContent = "⚠️";
  }

  // ==========================================
  // 6. Video File Analysis
  // ==========================================
  const videoDropzone = document.getElementById("videoDropzone");
  const videoFileInput = document.getElementById("videoFileInput");
  const videoProgressSection = document.getElementById("videoProgressSection");
  const videoResultsSection = document.getElementById("videoResultsSection");

  if (videoDropzone && videoFileInput) {
    videoDropzone.addEventListener("click", () => videoFileInput.click());

    videoDropzone.addEventListener("dragover", (e) => {
      e.preventDefault();
      videoDropzone.classList.add("dragover");
    });

    videoDropzone.addEventListener("dragleave", () => {
      videoDropzone.classList.remove("dragover");
    });

    videoDropzone.addEventListener("drop", (e) => {
      e.preventDefault();
      videoDropzone.classList.remove("dragover");
      if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
        handleVideoFile(e.dataTransfer.files[0]);
      }
    });

    videoFileInput.addEventListener("change", (e) => {
      if (e.target.files && e.target.files.length > 0) {
        handleVideoFile(e.target.files[0]);
      }
    });
  }

  function handleVideoFile(file) {
    if (!file.type.startsWith("video/")) {
      alert("Please upload a valid video file (MP4, AVI, MOV).");
      return;
    }

    if (videoProgressSection) videoProgressSection.style.display = "block";
    if (videoResultsSection) videoResultsSection.style.display = "none";

    const formData = new FormData();
    formData.append("file", file);
    formData.append("sample_interval", "0.5"); // sample every 0.5s

    fetch("/api/predict/video", {
      method: "POST",
      body: formData,
    })
      .then((res) => res.json())
      .then((data) => {
        if (videoProgressSection) videoProgressSection.style.display = "none";
        if (data.status === "error") {
          alert(`Video Error: ${data.message}`);
        } else {
          renderVideoResult(data);
        }
      })
      .catch((err) => {
        if (videoProgressSection) videoProgressSection.style.display = "none";
        alert("Failed to analyze video: " + err.message);
      });
  }

  function renderVideoResult(data) {
    if (!videoResultsSection) return;
    videoResultsSection.style.display = "block";

    const overallPred = document.getElementById("videoOverallPred");
    if (overallPred) {
      overallPred.textContent = (data.overall_prediction || "Neutral").toUpperCase();
      overallPred.style.color =
        data.overall_prediction === "Fire"
          ? "var(--danger-red)"
          : data.overall_prediction === "Smoke"
          ? "var(--warning-amber)"
          : "var(--safe-green)";
    }

    const fireStat = document.getElementById("videoFirePercent");
    if (fireStat) fireStat.textContent = `${data.stats?.fire_percent || 0}%`;

    const smokeStat = document.getElementById("videoSmokePercent");
    if (smokeStat) smokeStat.textContent = `${data.stats?.smoke_percent || 0}%`;

    const neutralStat = document.getElementById("videoNeutralPercent");
    if (neutralStat) neutralStat.textContent = `${data.stats?.neutral_percent || 0}%`;

    // Render Timeline Items
    const timelineContainer = document.getElementById("videoTimeline");
    if (timelineContainer && data.timeline) {
      timelineContainer.innerHTML = "";
      data.timeline.forEach((item) => {
        const row = document.createElement("div");
        row.className = "timeline-item";
        const tagClass =
          item.prediction === "Fire"
            ? "tag-fire"
            : item.prediction === "Smoke"
            ? "tag-smoke"
            : "tag-neutral";

        row.innerHTML = `
          <span>⏱️ ${item.timestamp.toFixed(1)}s (Frame ${item.frame})</span>
          <span class="timeline-tag ${tagClass}">${item.prediction} (${item.confidence.toFixed(1)}%)</span>
        `;
        timelineContainer.appendChild(row);
      });
    }

    if (data.is_hazard) {
      triggerAlarmSiren();
    }
  }

  // ==========================================
  // 7. Live Webcam Surveillance
  // ==========================================
  const startCameraBtn = document.getElementById("startCameraBtn");
  const stopCameraBtn = document.getElementById("stopCameraBtn");
  const webcamVideo = document.getElementById("webcamVideo");
  const webcamCanvas = document.getElementById("webcamCanvas");
  const webcamBanner = document.getElementById("webcamBanner");
  const webcamStatusText = document.getElementById("webcamStatusText");
  const webcamConfidence = document.getElementById("webcamConfidence");
  const webcamFpsText = document.getElementById("webcamFpsText");

  let webcamStream = null;
  let webcamInterval = null;
  let isWebcamRunning = false;
  let isPredictingFrame = false;
  let frameCount = 0;
  let lastFpsTime = Date.now();

  if (startCameraBtn) {
    startCameraBtn.addEventListener("click", startWebcam);
  }
  if (stopCameraBtn) {
    stopCameraBtn.addEventListener("click", stopWebcam);
  }

  async function startWebcam() {
    initAudio();
    try {
      webcamStream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 } },
        audio: false,
      });
      webcamVideo.srcObject = webcamStream;
      await webcamVideo.play();

      isWebcamRunning = true;
      startCameraBtn.style.display = "none";
      stopCameraBtn.style.display = "inline-flex";

      if (webcamStatusText) webcamStatusText.textContent = "MONITORING ACTIVE";

      // Loop frame capture every 400ms for responsive near-realtime inference
      webcamInterval = setInterval(captureAndSendWebcamFrame, 400);
    } catch (err) {
      alert("Unable to access camera: " + err.message);
    }
  }

  function stopWebcam() {
    if (webcamInterval) {
      clearInterval(webcamInterval);
      webcamInterval = null;
    }
    if (webcamStream) {
      webcamStream.getTracks().forEach((track) => track.stop());
      webcamStream = null;
    }
    if (webcamVideo) {
      webcamVideo.srcObject = null;
    }

    isWebcamRunning = false;
    startCameraBtn.style.display = "inline-flex";
    stopCameraBtn.style.display = "none";

    if (webcamBanner) {
      webcamBanner.className = "result-banner";
    }
    if (webcamStatusText) webcamStatusText.textContent = "CAMERA OFFLINE";
    if (webcamConfidence) webcamConfidence.textContent = "--%";
  }

  async function captureAndSendWebcamFrame() {
    if (!isWebcamRunning || isPredictingFrame || !webcamVideo || !webcamCanvas) return;

    const ctx = webcamCanvas.getContext("2d");
    webcamCanvas.width = 320; // Reduced resolution for fast network transmission
    webcamCanvas.height = 240;

    ctx.drawImage(webcamVideo, 0, 0, webcamCanvas.width, webcamCanvas.height);
    const base64Data = webcamCanvas.toDataURL("image/jpeg", 0.7);

    isPredictingFrame = true;
    try {
      const res = await fetch("/api/predict/frame", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ frame: base64Data }),
      });
      const data = await res.json();
      renderWebcamResult(data);

      frameCount++;
      const now = Date.now();
      if (now - lastFpsTime >= 1000) {
        if (webcamFpsText) {
          webcamFpsText.textContent = `Inference: ${(frameCount * 1000 / (now - lastFpsTime)).toFixed(1)} FPS`;
        }
        frameCount = 0;
        lastFpsTime = now;
      }
    } catch (e) {
      console.warn("Webcam frame predict error:", e);
    } finally {
      isPredictingFrame = false;
    }
  }

  function renderWebcamResult(data) {
    if (!data || !webcamBanner) return;

    const pred = data.prediction || "Neutral";
    const confidence = data.confidence || 0.0;
    const isHazard = data.is_hazard;

    webcamBanner.className = "result-banner";
    if (pred === "Fire") {
      webcamBanner.classList.add("hazard-fire");
    } else if (pred === "Smoke") {
      webcamBanner.classList.add("hazard-smoke");
    } else {
      webcamBanner.classList.add("hazard-safe");
    }

    if (webcamStatusText) {
      webcamStatusText.textContent = `${pred.toUpperCase()} (${confidence.toFixed(1)}%)`;
    }
    if (webcamConfidence) {
      webcamConfidence.textContent = `${confidence.toFixed(1)}%`;
      webcamConfidence.style.color = data.color_hex || "#fff";
    }

    if (isHazard) {
      triggerAlarmSiren();
    }
  }
});
