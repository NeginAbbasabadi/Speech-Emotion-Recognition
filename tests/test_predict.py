
from io import BytesIO
from unittest.mock import patch

from backend.app import app


def test_missing_audio():
    client = app.test_client()

    response = client.post("/api/predict")

    assert response.status_code == 400
    assert "error" in response.get_json()


def test_invalid_extension():
    client = app.test_client()

    response = client.post(
        "/api/predict",
        data={
            "audio": (
                BytesIO(b"not audio"),
                "test.txt"
            )
        },
        content_type="multipart/form-data"
    )

    assert response.status_code == 400


def test_prediction_response():
    client = app.test_client()

    mock_result = {
        "emotion": "happy",
        "confidence": 0.75,
        "probabilities": {
            "neutral": 0.05,
            "happy": 0.75,
            "sad": 0.10,
            "angry": 0.10
        }
    }

    with patch(
        "backend.app.predict_emotion",
        return_value=mock_result
    ):
        response = client.post(
            "/api/predict",
            data={
                "audio": (
                    BytesIO(b"mock audio"),
                    "test.wav"
            ),
        "model": "cnn"
            },

            content_type="multipart/form-data"
        )

    assert response.status_code == 200

    data = response.get_json()

    assert data["emotion"] == "happy"
    assert data["confidence"] == 0.75


def test_invalid_wav_content():
    client = app.test_client()

    response = client.post(
        "/api/predict",
        data={
            "audio": (
                BytesIO(b"not a real wav file"),
                "invalid.wav"
            )
        },
        content_type="multipart/form-data"
    )

    assert response.status_code == 400
    assert "error" in response.get_json()
