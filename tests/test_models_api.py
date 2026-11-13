
from io import BytesIO
from unittest.mock import patch

import pytest

from backend.app import app


# --------------------------------------------------
# Test configuration
# --------------------------------------------------

@pytest.fixture
def client():
    app.config["TESTING"] = True

    with app.test_client() as test_client:
        yield test_client


@pytest.fixture
def mock_prediction():
    return {
        "emotion": "happy",
        "confidence": 0.85,
        "probabilities": {
            "neutral": 0.03,
            "happy": 0.85,
            "sad": 0.07,
            "angry": 0.05,
        },
    }


def upload_audio(client, model="wav2vec2"):
    return client.post(
        "/api/predict",
        data={
            "audio": (
                BytesIO(b"fake wav data"),
                "test.wav",
            ),
            "model": model,
        },
        content_type="multipart/form-data",
    )


# --------------------------------------------------
# Health endpoint
# --------------------------------------------------

def test_health_endpoint(client):
    response = client.get("/api/health")

    assert response.status_code == 200


# --------------------------------------------------
# Model selection tests
# --------------------------------------------------

def test_wav2vec2_prediction(client, mock_prediction):
    with patch(
        "backend.app.predict_wav2vec2",
        return_value=mock_prediction.copy(),
    ) as mock_model:

        response = upload_audio(client, "wav2vec2")

    assert response.status_code == 200

    result = response.get_json()

    assert result["model"] == "wav2vec2"
    assert result["emotion"] == "happy"
    assert result["confidence"] == 0.85

    mock_model.assert_called_once()


def test_cnn_prediction(client, mock_prediction):
    with patch(
        "backend.app.predict_emotion",
        return_value=mock_prediction.copy(),
    ) as mock_model:

        response = upload_audio(client, "cnn")

    assert response.status_code == 200

    result = response.get_json()

    assert result["model"] == "cnn"
    assert result["emotion"] == "happy"

    mock_model.assert_called_once()


def test_default_model_is_wav2vec2(
    client,
    mock_prediction,
):
    with patch(
        "backend.app.predict_wav2vec2",
        return_value=mock_prediction.copy(),
    ) as mock_model:

        response = client.post(
            "/api/predict",
            data={
                "audio": (
                    BytesIO(b"fake wav data"),
                    "test.wav",
                ),
            },
            content_type="multipart/form-data",
        )

    assert response.status_code == 200
    assert response.get_json()["model"] == "wav2vec2"

    mock_model.assert_called_once()


# --------------------------------------------------
# Invalid request tests
# --------------------------------------------------

def test_unsupported_model(client):
    response = upload_audio(client, "invalid_model")

    assert response.status_code == 400
    assert "error" in response.get_json()


def test_missing_audio_file(client):
    response = client.post(
        "/api/predict",
        data={"model": "cnn"},
    )

    assert response.status_code == 400
    assert "error" in response.get_json()


def test_invalid_file_extension(client):
    response = client.post(
        "/api/predict",
        data={
            "audio": (
                BytesIO(b"not audio"),
                "test.txt",
            ),
            "model": "cnn",
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 400
    assert "error" in response.get_json()


def test_empty_filename(client):
    response = client.post(
        "/api/predict",
        data={
            "audio": (
                BytesIO(b"fake wav data"),
                "",
            ),
            "model": "cnn",
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 400


# --------------------------------------------------
# Error handling
# --------------------------------------------------

def test_model_value_error(client):
    with patch(
        "backend.app.predict_wav2vec2",
        side_effect=ValueError("Invalid audio."),
    ):
        response = upload_audio(client, "wav2vec2")

    assert response.status_code == 400
    assert response.get_json()["error"] == "Invalid audio."


def test_internal_model_error(client):
    with patch(
        "backend.app.predict_wav2vec2",
        side_effect=RuntimeError("Model failure"),
    ):
        response = upload_audio(client, "wav2vec2")

    assert response.status_code == 500

    result = response.get_json()
    assert "error" in result
    assert "Model failure" not in result["error"]


def test_prediction_response_has_all_probabilities(
    client,
    mock_prediction,
):
    with patch(
        "backend.app.predict_wav2vec2",
        return_value=mock_prediction.copy(),
    ):
        response = upload_audio(client, "wav2vec2")

    assert response.status_code == 200

    result = response.get_json()

    assert set(result["probabilities"]) == {
        "neutral",
        "happy",
        "sad",
        "angry",
    }

    assert all(
        isinstance(value, (int, float))
        and 0 <= value <= 1
        for value in result["probabilities"].values()
    )


def test_audio_upload_exceeds_size_limit(client):
    oversized_audio = b"x" * (10 * 1024 * 1024 + 1)

    response = client.post(
        "/api/predict",
        data={
            "audio": (
                BytesIO(oversized_audio),
                "oversized.wav",
            ),
            "model": "cnn",
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 413
    assert (
        response.get_json()["error"]
        == "Audio file exceeds the 10 MB limit."
    )
