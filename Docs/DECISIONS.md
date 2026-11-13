# VoxSense AI — Architectural Decision Records

**Updated:** 13 November 2026  
**Project:** Speech Emotion Recognition course prototype  
**Status:** Retrospective documentation of implemented decisions

## How to read these records

These records document the project team's stated motivations, observed implementation, alternatives, consequences, and verification evidence. **Alternatives are retrospective comparisons**, not claims that every option was formally assessed during original development. "Accepted" means implemented, not formally approved at a historical meeting. See [Requirements](Requirements.md), [Architecture](architecture.md), and [Verification](VERIFICATION.md) for supporting context.

| ID | Decision | Status |
| --- | --- | --- |
| ADR-001 | Flask REST API | Accepted |
| ADR-002 | Retain CNN and Wav2Vec2 | Accepted |
| ADR-003 | Cache loaded models | Accepted |
| ADR-004 | Local execution scope | Current scope; original deployment intention unknown |
| ADR-005 | Automated tests and CI | Accepted |
| ADR-006 | RMS-based audio activity check | Accepted |

---

## ADR-001 — Flask backend

**Context.** The browser must submit audio to CNN and Wav2Vec2 models built with Python and PyTorch. The backend handles requests, model choice, validation, and JSON results.

**Decision and motivation.** We chose Flask because it is lightweight, easy to integrate with Python ML code, and suitable for a simple REST-style API. `backend/app.py` routes requests while separate inference modules handle predictions.

**Retrospective alternatives.** FastAPI offers typed validation and automatic API documentation but adds conventions not needed for the current endpoints. Django offers an ORM and authentication but is larger than required by this prototype. Flask requires manual implementation of some validation and documentation.

**Consequences.** Clear separation and straightforward Flask test-client tests; some API facilities must be coded manually. `app.run(debug=True)` is suitable for local development only, not public production hosting.

**Evidence.** `backend/app.py`; `tests/test_models_api.py`; `tests/test_predict.py`. Related: FR-02, FR-04–FR-07, NFR-03–NFR-04.

## ADR-002 — Two separate ML inference pipelines

**Context.** The initial MVP used a custom CNN. Fine-tuned Wav2Vec2 was added later. The application is both an ML experiment and a Software Engineering project: comparing approaches matters, even though Wav2Vec2 obtained better recorded RAVDESS results.

**Decision and motivation.** Retain both models and expose them through the same Flask API. CNN uses log-Mel features and spectrogram normalization; Wav2Vec2 uses silence removal, standardized waveform input, and waveform normalization. Both return `emotion`, `confidence`, and class `probabilities`; Flask adds the model name. The user can compare results from the same clip.

**Retrospective alternatives.** Replacing CNN with Wav2Vec2 simplifies maintenance but loses the baseline comparison. Using only CNN sacrifices the higher recorded accuracy of Wav2Vec2. An ensemble could be researched but was neither implemented nor shown to improve performance.

**Consequences.** Enables scientific and interactive comparison; increases maintenance, testing, dependency and memory demands. A shared response schema decouples frontend rendering from individual model implementations.

**Recorded evidence (project-specific, not general guarantees).**

| Metric | CNN | Fine-tuned Wav2Vec2 |
| --- | ---: | ---: |
| RAVDESS test accuracy | 50.89% | 79.46% |
| Macro-F1 | 48.72% | 78.08% |
| Mean of five warmed HTTP requests, same local WAV | 10.86 ms | 33.16 ms |

The RAVDESS test data had been examined during development, so the result is not an untouched benchmark. Local warmed latencies on one recording do not prove relative performance across all inputs/devices.

**Evidence.** `backend/inference.py`, `backend/wav2vec2_inference.py`, `backend/app.py`, `tests/test_model_inference.py`, `results/`, `docs/VERIFICATION.md`. Related: FR-03–FR-05, FR-07–FR-08, NFR-03.

## ADR-003 — Cache model instances after first use

**Context.** Checkpoint loading requires reading weights, creating model objects, moving them to MPS or CPU, and setting evaluation mode. The team reports approximate checkpoint sizes of 167 KB (CNN) and 361 MB (Wav2Vec2); these are not total runtime memory sizes.

**Decision and motivation.** Use `@lru_cache(maxsize=1)` on `load_model()` and `load_wav2vec2()` to reuse each model in the current Python process after a successful first load. The primary objective was lower latency and avoiding repeated disk loading, improving responsiveness.

**Retrospective alternatives.** Load on every request (less resident memory but repeated initialization); preload at startup (less first-request delay but slower startup and initial memory usage); separate model services (independent scaling but much greater operational complexity).

**Consequences.** Less repeated initialization overhead, at the cost of memory held for each loaded model. Each worker process could have a separate cache. Updated checkpoint files do not automatically replace cached instances; restart or cache invalidation is needed.

**Observed measurement.** Five successful warm requests per model gave mean local total HTTP latencies of **10.86 ms (CNN)** and **33.16 ms (Wav2Vec2)**, both below the five-second target. This does **not** isolate caching's performance contribution; no cached-versus-uncached controlled benchmark or true cold-start measurement was conducted.

**Evidence.** Both inference modules; local benchmark in `docs/VERIFICATION.md`. Related: NFR-02, NFR-03, NFR-08.

## ADR-004 — Keep this course prototype locally executable

**Context.** The main objective was a complete frontend, Flask API, two trained models, and software testing. Public hosting would add infrastructure, security, memory, and operating-cost considerations.

**Current scope and rationale.** The system was completed as a local research prototype so effort could focus on model development, UI, integration, verification, and CI. This scope is reasonable for the timeline and resources. **There is no evidence whether cloud deployment was originally planned and later abandoned; this ADR does not claim such a historical decision.**

**Retrospective alternatives.** Hosted Flask would provide public access but add resource and operational requirements; local containerization could improve environment consistency but require container builds; an independent hosted model service could support scaling but add network and deployment complexity.

**Consequences.** No cloud hosting expense and straightforward local development, but no public endpoint and users must set up compatible Python dependencies and models. The current `debug=True` development server is not a production deployment.

**Reproducibility evidence and limits.** The source and setup instructions reside in GitHub, checkpoints are distributed on Hugging Face, and `scripts/download_models.py` fetches them. Downloads and local real-model tests were observed successful; **a clean-clone install through real inference has not yet been independently verified**.

**Evidence.** `README.md`, `requirements.txt`, `scripts/download_models.py`, `docs/VERIFICATION.md`. Related: NFR-05, NFR-08 and project scope.

## ADR-005 — Layered automated testing and continuous integration

**Context.** Adding Wav2Vec2, model comparison, microphone recording, and stronger validation made repeated manual verification increasingly inefficient and raised regression risk.

**Decision and motivation.** Introduce unit, API, real-checkpoint integration, and Playwright browser tests; use GitHub Actions for standard tests on push and pull request to `main`. The primary aim was improved reliability and regression detection. Meeting course objectives on CI and automated quality assurance was an additional benefit; collaboration was not the main reason.

**Retrospective alternatives.** Manual-only tests are easy to start but hard to repeat. Local automation without CI relies on developers remembering to run tests. Real-checkpoint testing on every CI run would increase coverage but requires distributing large models and test audio and more runner resources.

**Observed evidence.**

- **58** fast Python tests passed locally after adding response-schema and >10 MB upload tests.
- **8** checkpoint-backed integration tests passed locally.
- A focused inference run reported **24** passing tests.
- **Six** manual browser scenarios passed: WAV upload, microphone recording, model comparison, silence rejection, malformed WAV rejection, and permission-denied handling.
- Screenshot evidence showed GitHub Actions CI #13 passing on commit `48b028a` after switching to CPU-only PyTorch to reduce runner disk use. It does not confirm later uninspected CI runs.

**Consequences.** Repeatable checks and faster discovery of regressions, with test and CI maintenance costs. Mocked model responses in standard CI do not prove real-checkpoint inference or model accuracy; local integration tests fill part of that gap. Passing tests never guarantee absence of defects.

**Evidence.** `tests/`, `tests/e2e/`, `.github/workflows/ci.yml`, `docs/VERIFICATION.md`. Related: NFR-07, NFR-04 and other tested requirements.

## ADR-006 — RMS-based silence and low-energy rejection

**Context.** During browser testing, microphone recordings without speech still produced emotion classifications because some background noise passed original input checks. General validation was a requirement, but robust silence detection was not specified initially.

**Decision and motivation.** Add shared `validate_audio_activity()` with an RMS-based energy threshold before either model's inference. Reject tested silent and low-energy recordings and provide useful feedback rather than an arbitrary class label.

**Retrospective alternatives.** Dedicated voice activity detection (VAD) could better identify speech but adds dependency, cost, and evaluation work. UI-only volume advice is easier but does not enforce validation. No activity check preserves the original failure.

**Consequences.** Clearer handling of low-energy cases, plus reusable validation and regression protection. **RMS detects energy, not speech**: loud non-speech sounds can pass and quiet speech might be rejected. A future VAD comparison should measure false rejects/accepts and latency.

**Observed verification.** `test_silent_audio_rejected`, `test_low_level_noise_rejected`, and `test_audible_signal_accepted` passed. The last uses an **audible synthetic tone**, not proof of speech recognition. Manual silent-browser recording produced: “No audible speech detected. Please record your voice and try again.” The observed defect led to requirements refinement, implementation, and new regression tests.

**Evidence.** `backend/preprocessing.py`, both inference modules, `tests/test_audio_activity.py`, `tests/test_inference_unit.py`, `docs/VERIFICATION.md`. Related: FR-06, NFR-04.

---

## Cross-cutting lessons and outstanding evidence

The decision records show how requirements, architectural boundaries, performance, and quality assurance influence one another. The ability to compare two models drove separate preprocessing paths and a common response contract. Cached loading improves repeated-request efficiency but retains memory. Expanding functionality created regression risk, motivating CI; testing also uncovered an incomplete silence-validation requirement and led to its refinement.

| Outstanding evidence | Reason it matters |
| --- | --- |
| Fully clean installation and actual inference | Reproducibility is not yet end-to-end verified |
| True cold-start latency and diverse recordings | Current speed data is limited to warmed local requests |
| Formal accessibility testing | Responsive UI alone does not establish accessibility |
| External natural-speech evaluation and calibration | RAVDESS performance does not establish emotion validity in real-world speech |
| Latest CI run following all recent changes | Prior successful run does not automatically validate later commits |
| Additional noise/speech detection evaluation | RMS threshold does not distinguish speech from non-speech |

## Links to course learning goals

| Goal | Representative decisions |
| --- | --- |
| LG1 — Computer Science vs Software Engineering | ADR-001 and ADR-002 distinguish model development from integrating and maintaining a functioning software system |
| LG2 — Planning and execution | ADR-004 and ADR-005 show scope and regression-risk management |
| LG3 — Development lifecycle | ADR-002 and ADR-006 connect changed requirements to design, implementation, and verification |
| LG4 — Methods and quality control | ADR-001, ADR-003, ADR-005 connect technical choices, quality requirements, and tests |
| LG5 — Process improvement | ADR-005 and ADR-006 show a concrete feedback-to-fix-to-regression-test cycle |

These project-specific records are **not a substitute for the scientific references and broader methodology comparisons** required in the final Grade 5 reflection.
