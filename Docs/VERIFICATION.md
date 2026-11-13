# VoxSense AI — Verification Report

**Project:** Speech Emotion Recognition Web Application  
**Status:** Core functional verification completed; additional quality checks remain  
**Last updated:** 13 November 2026

## 1. Purpose and scope

This report records evidence that the VoxSense AI application behaves as intended. It distinguishes local fast unit/API tests, tests using actual model checkpoints, GitHub Actions CI, real HTTP performance measurements, and manual browser acceptance testing. Only observed results are reported as passed; the absence of a failure in a test does not establish that all quality attributes are satisfied.

The application accepts microphone recordings and uploaded WAV files; provides CNN and fine-tuned Wav2Vec2 inference, four output labels (**neutral**, **happy**, **sad**, **angry**), and side-by-side model comparisons. Predictions reflect patterns learned from acted RAVDESS speech and must not be interpreted as reliable estimates of a person's true emotional state.

## 2. Verification environment and evidence

| Item | Observed configuration or evidence |
| --- | --- |
| Development environment | macOS; Python 3.13.3; project `.venv` |
| Python test framework | pytest 9.2.0 (reported in local test output) |
| Web application | Flask on `http://127.0.0.1:5002/` |
| Browser acceptance tests | Manual checks in Chrome on the local app (screenshots supplied for negative-input scenarios) |
| CI | GitHub Actions, `VoxSense AI - Continuous Integration`, branch `main` |
| Latest observed successful CI | Run **#13**, commit **`48b028a`**, 12 November 2026, duration **2m 16s** (workflow list screenshot) |
| Model checkpoints | `models/emotion_cnn.pt`, `models/wav2vec2_finetuned/` |
| Checkpoint distribution | [Hugging Face — NeginAbbasabadi/voxsense-ai](https://huggingface.co/NeginAbbasabadi/voxsense-ai) |
| Real API test input | `data/ravdess/Actor_21/03-01-03-01-01-01-21.wav` |

The tests and manual checks were reported during the November 2026 verification session. Exact timestamps for individual local commands were not recorded. The GitHub Actions screenshot confirms the **overall workflow status** but does not, on its own, show each step's logs or whether its coverage artifact was uploaded.

## 3. Verification summary

| Verification activity | Observed evidence | Status |
| --- | --- | --- |
| Fast Python unit/API regression suite | **56 passed**, **0 failed**, **4.40 s**; browser and real-checkpoint integration tests excluded | **Passed locally** |
| Real-model integration suite | **8 passed**: checkpoint presence, CNN/Wav2Vec2 inference, repeatability, and real Flask prediction for both models | **Passed locally** |
| GitHub Actions CI | Workflow **#13** on `main` at `48b028a`, green overall result | **Passed overall** |
| CI step-level logs and artifact inspection | Not reviewed for run #13 | **Pending** |
| Actual API predictions | CNN and Wav2Vec2 both returned HTTP 200 with label, confidence and four class scores | **Passed locally** |
| Warm-request API latency | Five repeated HTTP 200 responses per model; all below 5 s | **Measured; target met for tested requests** |
| Confirmed cold-start latency | First requests occurred while Flask was already running; model-cache state not established | **Not measured** |
| Manual browser acceptance scenarios | Six scenarios completed successfully; details in Section 6 | **Passed in reported manual checks** |
| Fresh clone/install + checkpoint download + real inference | Complete clean-environment sequence not observed | **Pending** |
| Formal accessibility evaluation | No recorded accessibility audit | **Pending** |

## 4. Automated test evidence

### 4.1 Fast Python unit/API regression suite

Command executed:

```bash
python -m pytest tests/ -v \
  --ignore=tests/e2e \
  --ignore=tests/test_model_inference.py
```

**Observed result:** `56 passed in 4.40s` (0 failures), on macOS with Python 3.13.3 and pytest 9.2.0.

The passing cases include Flask routes and request validation, silence/low-activity handling, audio preprocessing, augmentation, CNN architecture, and mocked or isolated CNN/Wav2Vec2 inference logic. This run did **not** execute browser end-to-end tests or real-checkpoint integration tests.

### 4.2 Integration tests with production model checkpoints

Command executed:

```bash
python -m pytest tests/test_model_inference.py -v
```

**Observed result:** All **8 collected tests passed**:

- `test_cnn_checkpoint_exists`
- `test_wav2vec2_checkpoint_exists`
- `test_cnn_real_inference`
- `test_cnn_prediction_is_deterministic`
- `test_wav2vec2_real_inference`
- `test_wav2vec2_prediction_is_deterministic`
- `test_real_flask_prediction[cnn]`
- `test_real_flask_prediction[wav2vec2]`

The reported output did not include the final duration line. These tests establish successful loading and prediction in the tested environment, not generalization to spontaneous speech.

### 4.3 GitHub Actions CI

A GitHub Actions workflow-list screenshot shows **run #13** (`ci: use CPU-only PyTorch to reduce runner disk usage`) on `main`, commit `48b028a`, as successful on **12 November 2026** (duration **2m 16s**). The preceding **run #12** for `4f1ace4` is shown as failed. This sequence can be used as evidence of iterative CI maintenance, but the screenshot alone does not establish the root cause of the failure.

**Follow-up:** Review run #13's step-level logs to confirm the Python and Playwright steps and inspect the coverage artifact for the latest results. The repository's previously reported **31% combined statement/branch coverage** is a **10 November 2026 baseline**, not a current figure. Training/evaluation scripts and locally dependent model tests are not fully exercised by the standard CI workflow; coverage must not be confused with test pass rate or model accuracy.

## 5. API response and performance verification

A valid WAV recording was submitted to `POST /api/predict` for each model using `curl`. Both requests returned **HTTP 200**, an `emotion` label, a `confidence` score, and `probabilities` for all four emotion classes.

| First observed request in this check | CNN | Fine-tuned Wav2Vec2 |
| --- | --- | --- |
| HTTP status | 200 | 200 |
| Total HTTP request time | **0.697032 s** | **0.319587 s** |
| Returned label | `angry` | `happy` |
| Returned confidence | 0.5228305 | 0.8519108 |
| Four class scores present | Yes | Yes |

The application was already running; these two measurements are **not confirmed cold-start times**. Model predictions can differ for the same recording, and confidence values are not calibrated probabilities of correctness.

### 5.1 Repeated warm-request measurements

With Flask running and the checkpoints already used, five additional requests were submitted for each model using the same file and `curl -w '%{time_total}'`. This measures **total HTTP request duration**, including request handling and model inference, not isolated neural-network inference time.

| Run | CNN (s) | Wav2Vec2 (s) |
| --- | ---: | ---: |
| 1 | 0.016571 | 0.057110 |
| 2 | 0.010868 | 0.026877 |
| 3 | 0.009852 | 0.026977 |
| 4 | 0.008643 | 0.027173 |
| 5 | 0.008345 | 0.027647 |
| **Mean** | **0.010856** | **0.033157** |
| **Median** | **0.009852** | **0.027173** |
| **Minimum** | **0.008345** | **0.026877** |
| **Maximum** | **0.016571** | **0.057110** |

**HTTP status:** 200 for all 10 repeated requests. **Five-second warm-request target:** met for every measured request on the development machine. These samples are too limited to establish performance under load, across devices or across varied audio files. The results do not imply that CNN is universally faster than Wav2Vec2.

**Outstanding measurement:** Restart the service in a controlled state and record separately the first request to each model; document the definition of “cold” (fresh process with empty in-process model cache). Do not claim the five-second target has been verified for cold requests.

## 6. Manual browser acceptance tests

Manual acceptance checks were reported as successful in the local browser application. The screenshots for silence rejection, corrupted audio, and microphone permissions were supplied during the verification session. Other successful scenarios were confirmed by the tester in conversation.

| Scenario | Observed result | Status |
| --- | --- | --- |
| Upload valid WAV and predict with CNN | Uploaded recording could be previewed; emotion, confidence and four class scores displayed | **Passed** |
| Record speech and predict with Wav2Vec2 | Microphone capture, timer, playback, and Wav2Vec2 result worked | **Passed** |
| Compare both models | Both model results displayed for the same WAV recording | **Passed** |
| Submit a quiet/silent microphone recording | No emotion prediction; visible error: **“No audible speech detected. Please record your voice and try again.”** | **Passed in tested conditions** |
| Upload malformed `.wav` file | No prediction; visible error: **“Unable to decode audio.”** | **Passed** |
| Block then restore microphone permission | Visible error: **“Microphone unavailable: Permission denied”**; restoring permission returned to **“Ready to capture your voice”** | **Passed** |

The intentionally malformed test file was created locally with text content under a `.wav` filename; rejection therefore confirms that the application did not treat the file extension alone as sufficient evidence of valid audio.

**Limitation:** RMS-based low-activity detection is not speech recognition. It may allow loud non-speech signals or reject very quiet speech. Manual acceptance demonstrates behavior only under the tested browser and recording conditions.

## 7. Remaining verification work and limitations

1. **Inspect CI internals:** Confirm the successful run's individual Python/Playwright steps and coverage artifact; update the coverage figure only after reading the actual artifact.
2. **Measure cold starts:** Record model-loading latency after a controlled server restart and distinguish it from warm-request latency.
3. **Verify clean installation:** Clone into a fresh directory or environment, install dependencies, download both production checkpoints, start Flask, and execute a real prediction.
4. **Assess accessibility:** Document keyboard use, focus, labels, feedback accessibility and other applicable checks; responsive design alone is not an accessibility audit.
5. **Assess privacy behavior:** Verify request handling and logging before claiming that uploaded recordings are never retained; no production deployment was evaluated here.
6. **Check robustness beyond RAVDESS:** Domain shift, accents, recording conditions, natural speech, and class-score calibration remain open research concerns.

The previously recorded RAVDESS test results (CNN **50.89% accuracy / 48.72% macro-F1**; fine-tuned Wav2Vec2 **79.46% accuracy / 78.08% macro-F1**) are **model-evaluation results**, not acceptance-test or real-world accuracy guarantees. The test set was examined during development and should not be described as a completely untouched benchmark.

## 8. Sign-off checklist

- [x] Fast Python unit/API regression suite passes locally (56/56)
- [x] Real-model integration tests pass locally (8/8)
- [x] Actual Flask prediction responses verified for both models
- [x] Repeated warm-request latency measured; tested requests satisfy five-second target
- [x] Six manual browser acceptance scenarios completed successfully
- [x] GitHub Actions workflow #13 is green overall at commit `48b028a`
- [ ] Review run #13 step logs and coverage artifact
- [ ] Measure controlled cold-start latency
- [ ] Verify a fresh-clone installation and actual inference
- [ ] Complete formal accessibility and relevant privacy checks

**Final assessment:** Core functional and integration checks have passed in the tested development environment. Performance under cold-start or production conditions, complete installation reproducibility, formal accessibility, and several quality attributes remain unverified. These limitations are documented rather than treated as passed.
