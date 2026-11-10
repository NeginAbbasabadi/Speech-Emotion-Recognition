
const $ = (id) => document.getElementById(id);

const audioInput = $("audioFile");
const analyzeBtn = $("analyzeBtn");
const compareBtn = $("compareBtn");
const modelSelect = $("modelSelect");
const fileName = $("fileName");
const statusText = $("status");

const startBtn = $("startRecordingBtn");
const stopBtn = $("stopRecordingBtn");
const recordingStatus = $("recordingStatus");
const recordingTimer = $("recordingTimer");
const recordingIndicator = $("recordingIndicator");
const recordingLabel = $("recordingLabel");

const canvas = $("waveformCanvas");
const canvasContext = canvas.getContext("2d");
const audioPreview = $("audioPreview");
const previewSection = $("audioPreviewSection");

const resultSection = $("resultSection");
const emotionLabel = $("emotionLabel");
const emotionIcon = $("emotionIcon");
const confidenceLabel = $("confidenceLabel");
const probabilitiesDiv = $("probabilities");
const resultModelName = $("resultModelName");

const comparisonSection = $("comparisonSection");
const comparisonGrid = $("comparisonGrid");

const MAX_SIZE = 10 * 1024 * 1024;
const SAMPLE_RATE = 16000;

const EMOTIONS = {
    happy:   { icon: "☺", color: "#F8C66B" },
    sad:     { icon: "☂", color: "#66B8F2" },
    angry:   { icon: "⚡", color: "#FF819C" },
    neutral: { icon: "◉", color: "#A99BFF" }
};

let selectedFile = null;
let previewURL = null;

let recorder = null;
let stream = null;
let chunks = [];
let audioContext = null;
let analyser = null;
let waveformFrame = null;

let timerInterval = null;
let recordingStartedAt = 0;
let busy = false;
let recording = false;
let recordingProcessing = false;


// ---------- Interface helpers ----------

function setStatus(message, error = false) {
    statusText.textContent = message;
    statusText.classList.toggle("error", error);
}

function updateButtons() {
    const canAnalyze =
        selectedFile !== null &&
        !busy && !recording && !recordingProcessing;

    analyzeBtn.disabled = !canAnalyze;
    compareBtn.disabled = !canAnalyze;
    startBtn.disabled = busy || recording || recordingProcessing;
    stopBtn.disabled = !recording;
}

function hideResults() {
    resultSection.classList.add("hidden");
    comparisonSection.classList.add("hidden");
}

function clearPreview() {
    audioPreview.pause();
    audioPreview.removeAttribute("src");
    audioPreview.load();

    if (previewURL) {
        URL.revokeObjectURL(previewURL);
        previewURL = null;
    }

    previewSection.classList.add("hidden");
}

function setSelectedAudio(file) {
    selectedFile = file;
    fileName.textContent = file ? file.name : "";

    clearPreview();
    hideResults();

    if (file) {
        previewURL = URL.createObjectURL(file);
        audioPreview.src = previewURL;
        previewSection.classList.remove("hidden");
    }

    updateButtons();
}


// ---------- WAV upload ----------

audioInput.addEventListener("change", () => {
    const file = audioInput.files[0];

    setSelectedAudio(null);
    setStatus("");

    if (!file) return;

    if (!file.name.toLowerCase().endsWith(".wav")) {
        setStatus("Please choose a WAV file.", true);
        return;
    }

    if (file.size > MAX_SIZE) {
        setStatus("File must be under 10 MB.", true);
        return;
    }

    setSelectedAudio(file);
});


// ---------- Recording timer ----------

function formatTime(milliseconds) {
    const seconds = Math.floor(milliseconds / 1000);
    const minutes = Math.floor(seconds / 60);

    return (
        String(minutes).padStart(2, "0") + ":" +
        String(seconds % 60).padStart(2, "0")
    );
}

function startTimer() {
    recordingStartedAt = performance.now();

    recordingTimer.textContent = "00:00";

    timerInterval = setInterval(() => {
        recordingTimer.textContent = formatTime(
            performance.now() - recordingStartedAt
        );
    }, 100);
}

function stopTimer() {
    clearInterval(timerInterval);
    timerInterval = null;
}


// ---------- Live waveform ----------

function resizeCanvas() {
    const ratio = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();

    canvas.width = Math.max(1, Math.round(rect.width * ratio));
    canvas.height = Math.max(1, Math.round(rect.height * ratio));

    canvasContext.setTransform(ratio, 0, 0, ratio, 0, 0);
}

function drawWaveform() {
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;

    canvasContext.clearRect(0, 0, width, height);

    if (!analyser) {
        canvasContext.strokeStyle = "#514B81";
        canvasContext.lineWidth = 2;
        canvasContext.beginPath();
        canvasContext.moveTo(0, height / 2);
        canvasContext.lineTo(width, height / 2);
        canvasContext.stroke();
        return;
    }

    const data = new Uint8Array(analyser.fftSize);
    analyser.getByteTimeDomainData(data);

    canvasContext.beginPath();
    canvasContext.lineWidth = 2.5;
    canvasContext.strokeStyle = "#A99BFF";
    canvasContext.shadowBlur = 12;
    canvasContext.shadowColor = "#8C7CF0";

    for (let i = 0; i < data.length; i++) {
        const x = (i / (data.length - 1)) * width;
        const y = (data[i] / 255) * height;

        if (i === 0) {
            canvasContext.moveTo(x, y);
        } else {
            canvasContext.lineTo(x, y);
        }
    }

    canvasContext.stroke();
    canvasContext.shadowBlur = 0;

    waveformFrame = requestAnimationFrame(drawWaveform);
}

async function stopAudioVisualization() {
    if (waveformFrame !== null) {
        cancelAnimationFrame(waveformFrame);
        waveformFrame = null;
    }

    analyser = null;

    if (audioContext) {
        const context = audioContext;
        audioContext = null;

        if (context.state !== "closed") {
            await context.close();
        }
    }

    resizeCanvas();
    drawWaveform();
}

window.addEventListener("resize", () => {
    resizeCanvas();

    if (!recording) drawWaveform();
});

resizeCanvas();
drawWaveform();


// ---------- Microphone recording ----------

startBtn.addEventListener("click", async () => {
    if (busy || recording || recordingProcessing) return;

    if (!navigator.mediaDevices?.getUserMedia ||
        typeof MediaRecorder === "undefined") {
        recordingStatus.textContent =
            "Your browser does not support microphone recording.";
        return;
    }

    startBtn.disabled = true;
    audioInput.value = "";
    setSelectedAudio(null);
    setStatus("");

    try {
        stream = await navigator.mediaDevices.getUserMedia({
            audio: true
        });

        const AudioContextClass =
            window.AudioContext || window.webkitAudioContext;

        audioContext = new AudioContextClass();

        const source = audioContext.createMediaStreamSource(stream);
        analyser = audioContext.createAnalyser();

        analyser.fftSize = 1024;
        source.connect(analyser);

        chunks = [];
        recorder = new MediaRecorder(stream);

        recorder.ondataavailable = (event) => {
            if (event.data.size > 0) {
                chunks.push(event.data);
            }
        };

        recorder.onstop = async () => {
            recording = false;
            recordingProcessing = true;
            updateButtons();

            stream?.getTracks().forEach(track => track.stop());
            stream = null;

            await stopAudioVisualization();

            try {
                recordingStatus.textContent =
                    "Preparing your WAV recording...";

                const recordedBlob = new Blob(chunks, {
                    type: recorder.mimeType ||
                          "application/octet-stream"
                });

                const wavBlob = await convertToWav(recordedBlob);

                if (wavBlob.size > MAX_SIZE) {
                    throw new Error("Recording exceeds 10 MB.");
                }

                const file = new File(
                    [wavBlob],
                    "voice_recording.wav",
                    { type: "audio/wav" }
                );

                setSelectedAudio(file);
                recordingStatus.textContent =
                    "Recording ready. Listen or analyze.";

            } catch (error) {
                recordingStatus.textContent =
                    "Could not prepare recording: " + error.message;
                setStatus("Recording conversion failed.", true);

            } finally {
                recorder = null;
                chunks = [];
                recordingProcessing = false;
                recordingIndicator.classList.remove("active");
                recordingLabel.textContent = "READY";
                updateButtons();
            }
        };

        recorder.onerror = () => {
            recordingStatus.textContent =
                "A microphone recording error occurred.";

            if (recorder?.state === "recording") {
                recorder.stop();
            }
        };

        recorder.start();
        recording = true;

        recordingIndicator.classList.add("active");
        recordingLabel.textContent = "RECORDING";

        recordingStatus.textContent = "Speak now...";
        startTimer();
        drawWaveform();
        updateButtons();

    } catch (error) {
        stream?.getTracks().forEach(track => track.stop());
        stream = null;

        recording = false;
        recordingProcessing = false;
        stopTimer();
        await stopAudioVisualization();

        recordingStatus.textContent =
            "Microphone unavailable: " + error.message;

        updateButtons();
    }
});


stopBtn.addEventListener("click", () => {
    if (!recorder || recorder.state !== "recording") return;

    stopTimer();
    recording = false;
    recordingProcessing = true;
    recordingStatus.textContent = "Processing recording...";

    updateButtons();
    recorder.stop();
});


// ---------- Convert browser audio to WAV ----------

async function convertToWav(blob) {
    const AudioContextClass =
        window.AudioContext || window.webkitAudioContext;

    const context = new AudioContextClass();

    let decoded;

    try {
        decoded = await context.decodeAudioData(
            await blob.arrayBuffer()
        );
    } finally {
        await context.close();
    }

    if (!decoded.length) {
        throw new Error("Empty recording.");
    }

    const mono = new Float32Array(decoded.length);

    for (let channel = 0;
         channel < decoded.numberOfChannels;
         channel++) {

        const data = decoded.getChannelData(channel);

        for (let i = 0; i < decoded.length; i++) {
            mono[i] += data[i] / decoded.numberOfChannels;
        }
    }

    const targetLength = Math.ceil(
        decoded.duration * SAMPLE_RATE
    );

    const offline = new OfflineAudioContext(
        1, targetLength, SAMPLE_RATE
    );

    const buffer = offline.createBuffer(
        1, mono.length, decoded.sampleRate
    );

    buffer.copyToChannel(mono, 0);

    const source = offline.createBufferSource();
    source.buffer = buffer;
    source.connect(offline.destination);
    source.start();

    const rendered = await offline.startRendering();

    return encodeWav(rendered.getChannelData(0));
}

function encodeWav(samples) {
    const buffer = new ArrayBuffer(44 + samples.length * 2);
    const view = new DataView(buffer);

    function write(offset, text) {
        for (let i = 0; i < text.length; i++) {
            view.setUint8(offset + i, text.charCodeAt(i));
        }
    }

    write(0, "RIFF");
    view.setUint32(4, 36 + samples.length * 2, true);
    write(8, "WAVE");
    write(12, "fmt ");
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, SAMPLE_RATE, true);
    view.setUint32(28, SAMPLE_RATE * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    write(36, "data");
    view.setUint32(40, samples.length * 2, true);

    for (let i = 0; i < samples.length; i++) {
        const value = Math.max(-1, Math.min(1, samples[i]));
        view.setInt16(
            44 + i * 2,
            value < 0 ? value * 32768 : value * 32767,
            true
        );
    }

    return new Blob([buffer], { type: "audio/wav" });
}


// ---------- Prediction requests ----------

async function predictWithModel(file, modelName) {
    const formData = new FormData();

    formData.append("audio", file);
    formData.append("model", modelName);

    const response = await fetch("/api/predict", {
        method: "POST",
        body: formData
    });

    const data = await response.json();

    if (!response.ok) {
        throw new Error(data.error || "Prediction failed.");
    }

    return data;
}

function setBusy(value) {
    busy = value;
    updateButtons();
}

analyzeBtn.addEventListener("click", async () => {
    if (!selectedFile || busy) return;

    const file = selectedFile;
    setBusy(true);
    hideResults();
    setStatus("Analyzing your speech...");

    try {
        const result = await predictWithModel(
            file, modelSelect.value
        );

        displaySingleResult(result);
        setStatus("Analysis complete.");

    } catch (error) {
        setStatus(error.message, true);

    } finally {
        setBusy(false);
    }
});

compareBtn.addEventListener("click", async () => {
    if (!selectedFile || busy) return;

    const file = selectedFile;
    setBusy(true);
    hideResults();
    setStatus("Comparing CNN and Wav2Vec2...");

    const results = await Promise.allSettled([
        predictWithModel(file, "cnn"),
        predictWithModel(file, "wav2vec2")
    ]);

    const successful = [];
    const failures = [];

    results.forEach((result, index) => {
        const name = index === 0 ? "CNN" : "Wav2Vec2";

        if (result.status === "fulfilled") {
            successful.push({
                name,
                result: result.value
            });
        } else {
            failures.push(name + ": " + result.reason.message);
        }
    });

    if (successful.length) {
        displayComparison(successful);
    }

    if (failures.length) {
        setStatus(failures.join(" | "), true);
    } else {
        setStatus("Model comparison complete.");
    }

    setBusy(false);
});


// ---------- Emotion result visualization ----------

function emotionInfo(emotion) {
    return EMOTIONS[emotion] ||
        { icon: "✦", color: "#A99BFF" };
}

function formatEmotion(name) {
    return name.charAt(0).toUpperCase() + name.slice(1);
}

function createScoreBars(probabilities) {
    const wrapper = document.createElement("div");

    for (const [name, probability] of Object.entries(
        probabilities
    )) {
        const row = document.createElement("div");
        row.className = "score-row";

        const heading = document.createElement("div");
        heading.className = "score-header";

        const label = document.createElement("span");
        label.textContent = formatEmotion(name);

        const value = document.createElement("span");
        value.textContent =
            (probability * 100).toFixed(1) + "%";

        heading.append(label, value);

        const track = document.createElement("div");
        track.className = "score-track";

        const fill = document.createElement("div");
        fill.className = "score-fill";
        fill.style.width = "0%";
        fill.style.background = emotionInfo(name).color;

        track.append(fill);
        row.append(heading, track);
        wrapper.append(row);

        requestAnimationFrame(() => {
            requestAnimationFrame(() => {
                fill.style.width =
                    Math.max(0, Math.min(100, probability * 100))
                    + "%";
            });
        });
    }

    return wrapper;
}

function displaySingleResult(data) {
    const info = emotionInfo(data.emotion);

    emotionLabel.textContent = formatEmotion(data.emotion);
    emotionLabel.style.color = info.color;

    emotionIcon.textContent = info.icon;
    emotionIcon.style.color = info.color;

    confidenceLabel.textContent =
        (data.confidence * 100).toFixed(1) + "%";

    resultModelName.textContent =
        "Analyzed using " +
        (data.model === "cnn" ? "CNN" : "Wav2Vec2");

    probabilitiesDiv.replaceChildren(
        createScoreBars(data.probabilities)
    );

    resultSection.classList.remove("hidden");
}

function displayComparison(items) {
    comparisonGrid.replaceChildren();

    for (const item of items) {
        const data = item.result;
        const info = emotionInfo(data.emotion);

        const card = document.createElement("article");
        card.className = "comparison-model";

        const name = document.createElement("h3");
        name.textContent = item.name;

        const summary = document.createElement("div");
        summary.className = "comparison-emotion";
        summary.style.color = info.color;

        summary.textContent =
            info.icon + " " + formatEmotion(data.emotion);

        const confidence = document.createElement("p");
        confidence.className = "comparison-confidence";
        confidence.textContent =
            (data.confidence * 100).toFixed(1) +
            "% model confidence";

        card.append(
            name,
            summary,
            confidence,
            createScoreBars(data.probabilities)
        );

        comparisonGrid.append(card);
    }

    comparisonSection.classList.remove("hidden");
}
