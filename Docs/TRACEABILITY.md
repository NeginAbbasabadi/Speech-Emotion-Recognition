# VoxSense AI — Requirements Traceability Matrix

**Prepared:** 13 November 2026  
**Project:** Speech Emotion Recognition Web Application  
**Source of requirements:** `docs/Requirements.md`  
**Purpose:** Link each requirement to implementation, available tests, observed evidence, and remaining verification gaps.

## 1. How to read this matrix

- **Verified (observed):** Relevant behavior was observed in a reported test run or manual acceptance test. The status applies only to that evidence and environment.
- **Partially verified:** Some aspects have evidence, but the complete requirement, quality attribute, or acceptance criterion has not been established.
- **Implemented / not independently verified:** Code or documentation supports implementation, but direct test evidence for the whole requirement has not been inspected.
- **Pending:** Required evaluation or review has not been performed or supplied.

**Evidence sources:** `backend/app.py`, three inspected test files (`tests/test_models_api.py`, `tests/test_predict.py`, `tests/test_audio_activity.py`), the supplied test run outputs, manual browser checks on 13 November 2026, README and architecture documentation, and `docs/VERIFICATION.md`. Other files named below are **candidate implementation/test locations derived from the repository listing or documentation**, not source-code-reviewed mappings.

## 2. Functional requirements

| ID | Requirement (summary) | Implementation / interface | Verification evidence | Status and gap |
| --- | --- | --- | --- | --- |
| FR-01 | Record speech in browser | `frontend/script.js`, microphone/recording UI *(candidate; not source reviewed)* | Manual microphone recording, timer, playback, and Wav2Vec2 prediction passed | **Verified (observed)** for tested browser; permission behavior separately covered under FR-06 |
| FR-02 | Upload supported recording | `frontend/`; `backend/app.py` reads multipart `audio` and checks filename extension | Manual WAV upload and CNN prediction passed; `test_invalid_file_extension`, `test_invalid_wav_content` | **Verified (observed)** for a valid WAV and selected invalid inputs; not every possible file format |
| FR-03 | Model-consistent audio preprocessing | `backend/preprocessing.py`, `backend/inference.py`, `backend/wav2vec2_inference.py` *(candidate module details)* | Previously reported preprocessing test passes; real-checkpoint tests passed; `test_audio_activity.py` verifies activity rejection | **Partially verified**: specific training/inference preprocessing equivalence not independently audited |
| FR-04 | CNN four-class prediction | `backend/app.py` dispatches to `predict_emotion`; `backend/inference.py` | `test_cnn_prediction` (mocked dispatch); reported `test_cnn_real_inference` and `test_real_flask_prediction[cnn]` passed; live CNN API HTTP 200 | **Verified (observed)** for exercised checkpoint and recording |
| FR-05 | Show emotion, confidence, four class scores | `backend/app.py` returns prediction JSON plus `model`; frontend result display *(candidate)* | `test_prediction_response_has_all_probabilities` passed with mock; real API JSON included all four classes; manual result display passed | **Verified (observed)** for test cases; confidence calibration is not claimed |
| FR-06 | Meaningful errors for invalid audio, denied microphone, prediction failures | `backend/app.py` returns 400/413/500; browser error UI *(candidate)* | Inspected tests for missing/invalid input and model exception; manual silent WAV/mic error, invalid WAV, microphone permission denied passed | **Partially verified**: observed error scenarios pass; full recovery/error-message coverage not established |
| FR-07 | Select CNN or fine-tuned Wav2Vec2 | `backend/app.py` `model` dispatch; frontend selection *(candidate)* | `test_cnn_prediction`, `test_wav2vec2_prediction`, default-model test; both real Flask prediction tests passed; manual selection passed | **Verified (observed)** |
| FR-08 | Compare models on same audio | Browser comparison workflow *(candidate)*; two model-specific `/api/predict` requests described by architecture | Manual comparison displayed both outputs for the same chosen recording | **Verified (observed)** in manual browser test; browser test source not yet inspected |

## 3. Non-functional requirements

| ID | Requirement (summary) | Implementation / evidence location | Verification evidence | Status and gap |
| --- | --- | --- | --- | --- |
| NFR-01 | Responsive, accessible desktop/mobile interface | `frontend/index.html`, `frontend/style.css`, `frontend/script.js` *(candidate)* | Desktop/browser workflows passed manually | **Partially verified**: no documented mobile-device or formal accessibility audit |
| NFR-02 | Initial target: valid short-audio request within 5 s on development machine | Flask `/api/predict`, cached inference loaders described in architecture | Both initial observed requests under 5 s; 5 additional warm HTTP requests/model all under 5 s | **Partially verified**: limited one-file, local repeated benchmark; true cold starts, broader audio cases, and percentile objective not measured |
| NFR-03 | Separate routing, preprocessing, inference, frontend concerns | `backend/app.py` imports CNN/Wav2Vec2 inference; separate modules and frontend files | Source review confirms API-level delegation; architecture document lists boundaries | **Implemented / not independently verified** for full modularity and maintainability quality |
| NFR-04 | Reject invalid, empty, unsupported audio without service crash | `backend/app.py` validation, 413 handler, inference validation | Invalid extension/content and missing-file tests; `test_audio_upload_exceeds_size_limit`; silence tests; manual invalid/silent requests | **Verified (observed)** for exercised inputs; exhaustive malformed-media and recovery coverage not established |
| NFR-05 | Setup/API/architecture docs and comments | `README.md`, `docs/Requirements.md`, `docs/architecture.md`, `docs/VERIFICATION.md` | Documents reviewed; README includes install instructions, API example, diagrams | **Partially verified**: clean-install reproduction still pending |
| NFR-06 | Do not persist recordings server-side by default | `backend/app.py` reads bytes and does not explicitly write recordings; other modules/deployment not audited | No end-to-end data-retention review or deployment configuration evidence | **Partially verified**: API route inspected only; cannot yet assert system-wide privacy behavior |
| NFR-07 | Automated regression tests for main application flow | `tests/`, `tests/e2e/`, `.github/workflows/ci.yml` | 58 fast tests passed locally after new tests; 8 real-model integration tests passed locally earlier; CI #13 passed on prior commit | **Partially verified**: new 58-test state has not yet been matched to a confirmed CI run; E2E job details not directly inspected |
| NFR-08 | Retrieve production checkpoints without Git binaries | `scripts/download_models.py`, Hugging Face repository | Earlier reported successful download and matching Wav2Vec2 SHA-256; real-model tests passed with available checkpoints | **Partially verified**: clean clone/download/inference end-to-end and CNN checkpoint integrity not fully evidenced |

## 4. Evidence log (observed results)

| Evidence ID | Date | Result | Limits |
| --- | --- | --- | --- |
| EV-01 | 13 Nov 2026 conversation | 56/56 fast Python tests passed before two additional API tests | Excludes browser and checkpoint-backed suite |
| EV-02 | 13 Nov 2026 conversation | 8/8 checkpoint-backed integration tests passed | Uses existing local checkpoints and test data |
| EV-03 | Screenshot shared 13 Nov 2026 | GitHub Actions CI #13 passed at commit `48b028a` | This run predates two newly added tests; workflow individual steps/artifact not independently reviewed |
| EV-04 | 13 Nov 2026 conversation | Initial CNN HTTP 200: 0.697032 s; Wav2Vec2 HTTP 200: 0.319587 s | Not proven to be cold-start measurements |
| EV-05 | 13 Nov 2026 conversation | Five warm HTTP 200 requests per model. CNN: 16.571, 10.868, 9.852, 8.643, 8.345 ms. Wav2Vec2: 57.110, 26.877, 26.977, 27.173, 27.647 ms | One WAV, local machine, sequential requests; no load testing |
| EV-06 | 13 Nov 2026 conversation | Manual WAV upload/CNN result; microphone/Wav2Vec2 result; side-by-side comparison; silent-audio rejection; malformed-WAV rejection; denied-microphone error/recovery passed | Manual user-reported/screenshot-backed checks, not a formal cross-browser study |
| EV-07 | 13 Nov 2026 conversation | 12/12 `test_models_api.py` passed after adding full probability structure and upload-size tests | Uses mocked inference for API success cases |
| EV-08 | 13 Nov 2026 conversation | 58/58 fast Python tests passed after new tests | Local run; excludes E2E and real-model tests |
| EV-09 | Git status supplied | `7d49a2f` verification report commit on `main` and `origin/main` | Two separate tracked experiment files remain deleted in local working tree; not part of this evidence |

## 5. Requirements-to-tests examples (source-reviewed)

| Test | Direct assertion demonstrated | Linked requirement |
| --- | --- | --- |
| `test_cnn_prediction` / `test_wav2vec2_prediction` | Correct model selected; status 200; selected model returned | FR-04, FR-07 |
| `test_default_model_is_wav2vec2` | Default dispatch without model parameter | FR-07 |
| `test_prediction_response_has_all_probabilities` | Four expected class keys with numeric 0–1 values from mocked prediction | FR-05 |
| `test_missing_audio_file`, `test_invalid_file_extension`, `test_empty_filename` | Invalid request rejected with 400 | FR-02, FR-06, NFR-04 |
| `test_invalid_wav_content` | Invalid WAV contents rejected | FR-06, NFR-04 |
| `test_model_value_error` / `test_internal_model_error` | 400 for known bad input; sanitized 500 for internal failure | FR-06 |
| `test_audio_upload_exceeds_size_limit` | Oversized request rejected with 413 and descriptive JSON error | FR-06, NFR-04 |
| `test_silent_audio_rejected`, `test_low_level_noise_rejected` | Low-energy audio rejected by activity validator | FR-06, NFR-04 |
| `test_audible_signal_accepted` | A sufficiently strong synthetic signal is accepted | FR-03 (limited); **not speech detection** |

## 6. Open verification and engineering actions

1. Inspect the real Playwright test implementations and confirm which specific requirements they assert.
2. Inspect inference/preprocessing code to confirm training/inference preprocessing consistency (FR-03).
3. Confirm the new 58-test suite passes in the GitHub Actions workflow after committing the tests; inspect workflow steps and coverage artifact.
4. Perform a reproducible fresh-clone/model-download/real-inference check (NFR-05, NFR-08).
5. Measure true cold-start latency with an explicitly restarted service; consider repeated recordings and an agreed percentile metric (NFR-02).
6. Review frontend accessibility and responsive behavior, and document what was actually tested (NFR-01).
7. Review all audio-handling code and any eventual deployment configuration before making a system-wide non-retention claim (NFR-06).
8. Record architecture trade-offs, requirement changes, risk responses, and relevant literature in separate engineering-decision and reflection evidence; this matrix alone does not satisfy every Grade 5 learning goal.

## 7. Interpretation

Traceability connects **requirement → implementation → verification evidence → remaining risk**. A test pass is evidence for a specific behavior under specific conditions, not proof of universal correctness or software quality. The matrix should be updated when implementations, requirements, or verification results change.
