# VoxSense AI — Software Requirements

## 1. Project overview

VoxSense AI is a full-stack speech emotion recognition research prototype. A user records or uploads a short speech clip; the application predicts one of four learned labels: **happy, sad, angry or neutral**. The original MVP used a PyTorch CNN with Flask and a browser frontend. The current implementation additionally supports a fine-tuned Wav2Vec2 classifier and side-by-side model comparison.

Predictions describe learned patterns from acted speech, not a person's actual emotional or psychological state.

## 2. Functional requirements

| ID | Requirement | Implementation scope |
| --- | --- | --- |
| FR-01 | Users shall be able to record a short speech clip using their microphone. | Browser recording controls |
| FR-02 | Users shall be able to upload a supported audio recording. | WAV upload and preview |
| FR-03 | The application shall preprocess audio consistently with the selected model's training pipeline. | CNN: log-Mel features; Wav2Vec2: normalized waveform |
| FR-04 | The system shall predict one of four emotion labels using the CNN. | Original MVP |
| FR-05 | The application shall display predicted label, confidence score and per-class probabilities. | Both models |
| FR-06 | The application shall show meaningful errors for invalid audio, denied microphone permission and prediction failures. | UI/API error handling |
| FR-07 | Users shall be able to select the CNN or fine-tuned Wav2Vec2 model. | Extension beyond original MVP |
| FR-08 | Users shall be able to compare the two models using the same audio clip. | Extension beyond original MVP |

## 3. Non-functional requirements

| ID | Requirement | Verification / note |
| --- | --- | --- |
| NFR-01 | The interface should be responsive, accessible and usable on desktop and mobile browsers. | Responsive design implemented; formal accessibility audit not yet documented |
| NFR-02 | Initial target: respond within five seconds of receiving a valid short clip on the development machine. | **Target only**; benchmark measurements are needed to claim compliance |
| NFR-03 | Separate routing, preprocessing, model inference and frontend concerns. | Modular Python and frontend files |
| NFR-04 | Reject invalid, empty and unsupported audio without crashing the service. | Unit and API tests |
| NFR-05 | Provide setup instructions, API documentation, architecture diagrams and relevant code comments. | README and architecture documentation |
| NFR-06 | Do not permanently store recordings server-side by default. | Intended privacy behavior; confirm by reviewing the deployed server configuration |
| NFR-07 | Provide automated regression tests for the main application flow. | pytest, Playwright and GitHub Actions |
| NFR-08 | Make production model checkpoints retrievable without committing large binary files to Git. | Hugging Face Hub and download script |

## 4. Machine learning requirements

- **Dataset:** RAVDESS speech recordings for the four selected emotions.
- **CNN features:** 64-band log-Mel spectrogram input.
- **Wav2Vec2 input:** standardized and normalized waveform.
- **Framework:** PyTorch; pretrained Wav2Vec2 via Hugging Face Transformers.
- **Training:** local Apple Silicon acceleration where available.
- **Evaluation:** accuracy, macro F1-score and confusion matrix on actor-separated evaluation splits.
- **Robustness:** investigate augmentation and performance outside acted RAVDESS speech.

Previously recorded held-out RAVDESS results: CNN accuracy **50.89%**, macro F1 **48.72%**; fine-tuned Wav2Vec2 accuracy **79.46%**, macro F1 **78.08%**. These results do not establish performance on natural speech.

## 5. Scope

### Original MVP

The original MVP comprised microphone recording or audio upload, Flask request processing, CNN feature extraction and classification, and a visible prediction result.

### Extensions implemented after MVP

- Selection between CNN and fine-tuned Wav2Vec2 inference.
- Side-by-side prediction comparison.
- GitHub Actions CI and downloadable Python coverage reports.
- Public model-checkpoint distribution through Hugging Face Hub.

### Outside the current scope

- Continuous live-stream emotion classification.
- User accounts, authentication or persistent prediction history.
- Database storage of audio recordings.
- Additional emotion labels or multilingual performance guarantees.

## 6. Acceptance criteria

1. A user can record or upload a supported clip in the browser.
2. The frontend sends the WAV audio and selected model name to the Flask endpoint.
3. The backend validates audio and applies the selected preprocessing pipeline.
4. The selected model returns a label, confidence score and class probabilities in JSON.
5. The frontend displays the result; comparison mode can display both models' outputs.
6. Invalid audio inputs lead to appropriate errors without crashing the service.
7. Fast unit/API tests and browser tests pass in CI; real-model inference is checked separately with local model files and test recordings.
8. A fresh installation can retrieve the production checkpoints using the documented download script.

**Verification status:** Automated CI and checkpoint-download/hash checks have been reported successful. A fully clean installation, measured latency benchmark and formal accessibility assessment should be documented separately rather than assumed from those checks.
