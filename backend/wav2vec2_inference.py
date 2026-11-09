
from functools import lru_cache
from io import BytesIO
from pathlib import Path

import librosa
import numpy as np
import torch
from transformers import Wav2Vec2ForSequenceClassification

from backend.preprocessing import (
    SAMPLE_RATE,
    remove_silence,
    standardize_audio,
)
from ml.model import EMOTIONS


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "wav2vec2_finetuned"


@lru_cache(maxsize=1)
def load_wav2vec2():
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        MODEL_DIR
    )

    for index, emotion in enumerate(EMOTIONS):
        if model.config.id2label[index] != emotion:
            raise ValueError("Emotion label order mismatch.")

    model.to(device)
    model.eval()

    return model, device


def predict_wav2vec2(audio_bytes: bytes):
    if not audio_bytes:
        raise ValueError("Empty audio recording.")

    try:
        audio, _ = librosa.load(
            BytesIO(audio_bytes),
            sr=SAMPLE_RATE,
            mono=True,
        )
    except Exception as exc:
        raise ValueError("Unable to decode audio.") from exc

    if audio.size == 0 or not np.all(np.isfinite(audio)):
        raise ValueError("Invalid audio recording.")

    if np.max(np.abs(audio)) < 1e-5:
        raise ValueError("Audio appears to be silent.")

    # Match our Wav2Vec2 training preprocessing.
    audio = remove_silence(audio)
    audio = standardize_audio(audio)

    tensor = torch.from_numpy(
        audio.astype(np.float32)
    ).unsqueeze(0)

    # Same per-recording normalization as training.
    mean = tensor.mean(dim=1, keepdim=True)
    std = tensor.std(
        dim=1, keepdim=True, unbiased=False
    )

    tensor = (tensor - mean) / std.clamp(min=1e-7)

    model, device = load_wav2vec2()
    tensor = tensor.to(device)

    with torch.inference_mode():
        logits = model(input_values=tensor).logits
        probabilities = torch.softmax(
            logits, dim=1
        )[0].cpu().numpy()

    predicted_index = int(np.argmax(probabilities))

    return {
        "emotion": EMOTIONS[predicted_index],
        "confidence": float(probabilities[predicted_index]),
        "probabilities": {
            emotion: float(probability)
            for emotion, probability in zip(
                EMOTIONS, probabilities
            )
        },
    }
