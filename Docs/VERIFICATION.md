# VoxSense AI — Verification Report

**Project:** Speech Emotion Recognition Web Application  
**Status:** Verification in progress  
**Last updated:** 11 November 2026

## 1. Purpose and scope

This report records evidence that the VoxSense AI web application behaves as intended. It separates automated CI checks, local tests with real model checkpoints, manual browser checks, and performance measurements. **Only observed results should be marked as passed.**

The application supports microphone recording and WAV uploads, Flask `/api/predict` requests, model selection (CNN or fine-tuned Wav2Vec2), four-class predictions (neutral, happy, sad, angry), and model comparison. Emotion labels are predictions of patterns learned from acted speech in RAVDESS, not measurements of a speaker's actual emotional state.

## 2. Verification environment

| Item | Environment / configuration |
| --- | --- |
| Local development | macOS, Python 3.13.3, project `.venv` (observed in prior pytest output) |
| Web API | Flask, local port 5002 |
| Tests | pytest; Playwright for browser tests |
| CI | GitHub Actions on pushes / pull requests targeting `main` |
| Checkpoints | `models/emotion_cnn.pt` and `models/wav2vec2_finetuned/` |
| Checkpoint distribution | [Hugging Face: NeginAbbasabadi/voxsense-ai](https://huggingface.co/NeginAbbasabadi/voxsense-ai) |

## 3. Confirmed evidence

| Check | Evidence | Status |
| --- | --- | --- |
| GitHub Actions CI pipeline | Successful run shown in GitHub Actions screenshot, including a coverage artifact | **Passed (earlier run)** |
| Baseline Python coverage | Generated HTML report measured **31% combined statement/branch coverage** across `backend` and `ml` on 10 Nov 2026 | **Measured (baseline)** |
| Model download script | Local execution printed `All production models are available!` | **Passed locally** |
| Wav2Vec2 checkpoint integrity | Original and temporary-download `model.safetensors` files had identical SHA-256 hashes | **Passed** |
| Silence activity unit tests | Three `tests/test_audio_activity.py` tests passed in supplied local test output | **Passed in supplied run** |
| Fast Python tests after silence update | Supplied test output: 54 passed, 2 failed because existing tests expected previous error text | **Re-run needed after test correction** |
| Real-checkpoint inference tests after changes | No updated execution log supplied | **Not yet verified** |
| Manual browser regression checks after changes | No completed checklist/results supplied | **Not yet verified** |
| API performance against 5-second target | No timing measurements supplied | **Not yet measured** |

### Checkpoint integrity detail

The local `models/wav2vec2_finetuned/model.safetensors` and `/tmp/voxsense-download-test/wav2vec2_finetuned/model.safetensors` both produced this SHA-256 digest:

`ac6830cf005fc79f3d44485aab5cb264fee3a62d398cb61cc4397e47947b6dfe`

Matching digests demonstrate file integrity, not inference accuracy.

## 4. Automated regression tests

Run fast tests after completing the silence-error-message test adjustment:

```bash
python -m pytest tests/ -v \
  --ignore=tests/e2e \
  --ignore=tests/test_model_inference.py
```

**Result:** _Pending updated output._  
**Evidence to retain:** test summary and GitHub Actions run URL.

Run real-model integration tests (local checkpoints and RAVDESS recordings required):

```bash
python -m pytest tests/test_model_inference.py -v
```

**Result:** _Pending execution evidence._

## 5. API and latency verification

Start Flask in a separate terminal:

```bash
python -m backend.app
```

For each model, send the same known test WAV to `/api/predict`. Record HTTP status, returned emotion, whether all four probabilities are present, and request duration. Measure **cold** (first model load) and **warm** (subsequent) requests separately.

```bash
for model in cnn wav2vec2; do
  echo "=== $model ==="
  curl -sS -w '\nHTTP %{http_code} | total %{time_total}s\n' \
    -X POST http://127.0.0.1:5002/api/predict \
    -F "model=$model" \
    -F 'audio=@data/ravdess/Actor_21/03-01-03-01-01-01-21.wav'
done
```

| Model | Cold request (s) | Warm request (s) | HTTP code | Predicted emotion | Meets 5s warm target? |
| --- | ---: | ---: | --- | --- | --- |
| CNN | Pending | Pending | Pending | Pending | Pending |
| Wav2Vec2 | Pending | Pending | Pending | Pending | Pending |

**Note:** The `curl` figure is total HTTP request duration, not pure neural-network inference time. The five-second limit is a requirement target, not a verified achievement.

## 6. Manual browser acceptance tests

| Scenario | Expected result | Status |
| --- | --- | --- |
| Upload valid WAV | Preview plays; prediction renders | Pending |
| Record speech | Recording controls and timer work; captured audio can be analyzed | Pending |
| Select CNN | CNN label and four scores returned | Pending |
| Select Wav2Vec2 | Wav2Vec2 label and four scores returned | Pending |
| Compare models | Both predictions shown for the same audio | Pending |
| Submit silent audio | User-friendly validation error; no invented emotion | Pending recheck |
| Submit malformed or empty WAV | Clear error response without server crash | Pending |
| Deny microphone access | Informative permission error | Pending |

## 7. Limitations and interpretation

- The model is trained on acted RAVDESS speech; accuracy on spontaneous speech, varying languages, devices, and backgrounds is not established.
- Softmax confidence is not calibrated certainty about a person's emotions.
- RMS-based low-energy rejection detects quiet audio, **not speech presence**; loud non-speech sounds may still be classified and very quiet speech may be rejected.
- Standard GitHub Actions checks mock real model inference and do not establish accuracy of the hosted checkpoints.
- Initial coverage included training and evaluation scripts intentionally absent from the fast-test execution path; no minimum coverage target is enforced.

## 8. Sign-off checklist

- [ ] Fast Python suite passes after the silence-message test update
- [ ] Real-model integration tests pass with downloaded checkpoints
- [ ] Both real API model responses verified
- [ ] Cold and warm latency measured and recorded
- [ ] Manual browser scenarios verified
- [ ] GitHub Actions green after latest changes
- [ ] Remaining limitations and failures explicitly documented

**Final assessment:** _Pending the checks above. A previously successful CI run alone does not validate the entire application._
