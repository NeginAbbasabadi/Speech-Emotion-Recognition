
from functools import lru_cache
from io import BytesIO
from pathlib import Path

import librosa
import numpy as np
import torch

from backend.preprocessing import (
    SAMPLE_RATE,
    extract_features,
)
from ml.dataset import normalize_spectrogram
from ml.model import EmotionCNN, EMOTIONS


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "experiment_05"
    / "emotion_cnn.pt"
)


def get_device():
    """Select MPS when available, otherwise CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


@lru_cache(maxsize=1)
def load_model():
    """
    Load and cache the trained CNN.

    The model is loaded once per Flask process,
    not once per prediction.
    """
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model checkpoint not found: {MODEL_PATH}"
        )

    device = get_device()

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True
    )

    if checkpoint["emotions"] != EMOTIONS:
        raise ValueError(
            "Checkpoint emotion labels do not match."
        )

    model = EmotionCNN(
        num_classes=len(EMOTIONS)
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)
    model.eval()

    return model, device


def predict_emotion(audio_bytes: bytes):
    """
    Predict an emotion from a WAV recording.

    Returns a dictionary containing the
    predicted emotion and probabilities.
    """
    if not audio_bytes:
        raise ValueError("Empty audio recording.")

    # Load the uploaded WAV directly from memory.
    try:
        audio, _ = librosa.load(
            BytesIO(audio_bytes),
            sr=SAMPLE_RATE,
            mono=True
        )
    except Exception as exc:
        raise ValueError(
            "Audio could not be decoded as a supported WAV."
        ) from exc

    if audio.size == 0 or not np.all(np.isfinite(audio)):
        raise ValueError("Invalid or empty audio.")

    if np.max(np.abs(audio)) < 1e-5:
        raise ValueError("Audio appears to be silent.")

    # Use the same preprocessing as training.
    features = extract_features(audio)
    features = normalize_spectrogram(features)

    # (64, 188) -> (1, 1, 64, 188)
    tensor = torch.from_numpy(
        features
    ).unsqueeze(0).unsqueeze(0)

    model, device = load_model()
    tensor = tensor.to(device)

    with torch.inference_mode():
        logits = model(tensor)

        probabilities = torch.softmax(
            logits,
            dim=1
        )[0].cpu().numpy()

    predicted_index = int(
        np.argmax(probabilities)
    )

    return {
        "emotion": EMOTIONS[predicted_index],
        "confidence": float(
            probabilities[predicted_index]
        ),
        "probabilities": {
            emotion: float(probability)
            for emotion, probability in zip(
                EMOTIONS,
                probabilities
            )
        }
    }
