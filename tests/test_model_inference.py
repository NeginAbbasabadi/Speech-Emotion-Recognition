
from pathlib import Path

import pytest

from backend.inference import predict_emotion
from backend.wav2vec2_inference import predict_wav2vec2
from backend.app import app


# --------------------------------------------------
# Project paths and model configurations
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CNN_CHECKPOINT = (
    PROJECT_ROOT / "models" / "emotion_cnn.pt"
)

WAV2VEC2_CHECKPOINT = (
    PROJECT_ROOT / "models" / "wav2vec2_finetuned"
)

RAVDESS_AUDIO = (
    PROJECT_ROOT
    / "data"
    / "ravdess"
    / "Actor_21"
    / "03-01-03-01-01-01-21.wav"
)

EMOTIONS = {"neutral", "happy", "sad", "angry"}


# --------------------------------------------------
# Shared test fixtures
# --------------------------------------------------

@pytest.fixture(scope="module")
def real_audio():
    """Load a real RAVDESS recording once per test module."""

    assert RAVDESS_AUDIO.is_file(), (
        f"RAVDESS recording not found: {RAVDESS_AUDIO}"
    )

    return RAVDESS_AUDIO.read_bytes()


@pytest.fixture(scope="module")
def cnn_prediction(real_audio):
    """Run the actual CNN inference pipeline once."""
    return predict_emotion(real_audio)


@pytest.fixture(scope="module")
def wav2vec2_prediction(real_audio):
    """Run actual Wav2Vec2 inference once."""
    return predict_wav2vec2(real_audio)


# --------------------------------------------------
# Reusable prediction validation
# --------------------------------------------------

def validate_prediction(result):
    assert isinstance(result, dict)

    assert "emotion" in result
    assert "confidence" in result
    assert "probabilities" in result

    assert result["emotion"] in EMOTIONS

    confidence = result["confidence"]

    assert isinstance(confidence, (int, float))
    assert 0.0 <= confidence <= 1.0

    probabilities = result["probabilities"]

    assert isinstance(probabilities, dict)
    assert set(probabilities.keys()) == EMOTIONS

    for probability in probabilities.values():
        assert isinstance(probability, (int, float))
        assert 0.0 <= probability <= 1.0

    assert sum(probabilities.values()) == pytest.approx(
        1.0, abs=1e-4
    )

    predicted_probability = probabilities[result["emotion"]]

    assert confidence == pytest.approx(
        predicted_probability, abs=1e-4
    )

    assert confidence == pytest.approx(
        max(probabilities.values()), abs=1e-4
    )


# --------------------------------------------------
# Test model checkpoint availability
# --------------------------------------------------

def test_cnn_checkpoint_exists():
    assert CNN_CHECKPOINT.is_file(), (
        f"CNN checkpoint missing: {CNN_CHECKPOINT}"
    )


def test_wav2vec2_checkpoint_exists():
    assert (
        WAV2VEC2_CHECKPOINT / "config.json"
    ).is_file()

    assert (
        WAV2VEC2_CHECKPOINT / "model.safetensors"
    ).is_file()


# --------------------------------------------------
# Actual CNN inference tests
# --------------------------------------------------

def test_cnn_real_inference(cnn_prediction):
    validate_prediction(cnn_prediction)


def test_cnn_prediction_is_deterministic(
    cnn_prediction, real_audio
):
    second_prediction = predict_emotion(real_audio)

    assert second_prediction["emotion"] == (
        cnn_prediction["emotion"]
    )

    for emotion in EMOTIONS:
        assert second_prediction["probabilities"][
            emotion
        ] == pytest.approx(
            cnn_prediction["probabilities"][emotion],
            abs=1e-4,
        )


# --------------------------------------------------
# Actual Wav2Vec2 inference tests
# --------------------------------------------------

def test_wav2vec2_real_inference(wav2vec2_prediction):
    validate_prediction(wav2vec2_prediction)


def test_wav2vec2_prediction_is_deterministic(
    wav2vec2_prediction, real_audio
):
    second_prediction = predict_wav2vec2(real_audio)

    assert second_prediction["emotion"] == (
        wav2vec2_prediction["emotion"]
    )

    for emotion in EMOTIONS:
        assert second_prediction["probabilities"][
            emotion
        ] == pytest.approx(
            wav2vec2_prediction["probabilities"][emotion],
            abs=1e-4,
        )


# --------------------------------------------------
# Flask integration with real inference
# --------------------------------------------------

@pytest.mark.parametrize(
    "model_name",
    ["cnn", "wav2vec2"],
)
def test_real_flask_prediction(model_name, real_audio):
    client = app.test_client()

    response = client.post(
        "/api/predict",
        data={
            "audio": (
                __import__("io").BytesIO(real_audio),
                "ravdess_test.wav",
            ),
            "model": model_name,
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 200, (
        response.get_json()
    )

    result = response.get_json()

    assert result["model"] == model_name

    validate_prediction(result)
