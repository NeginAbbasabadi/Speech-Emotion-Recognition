
from pathlib import Path

import librosa
import numpy as np


# Audio configuration
SAMPLE_RATE = 16000
DURATION = 3
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 256

TARGET_SAMPLES = SAMPLE_RATE * DURATION
TARGET_FRAMES = 1 + TARGET_SAMPLES // HOP_LENGTH


def load_audio(file_path: str | Path) -> np.ndarray:
    """
    Load audio, convert it to mono, and resample
    it to 16 kHz.
    """
    audio, _ = librosa.load(
        file_path,
        sr=SAMPLE_RATE,
        mono=True
    )

    if audio.size == 0 or not np.all(np.isfinite(audio)):
        raise ValueError("Audio is empty or invalid.")

    return audio


def standardize_audio(audio: np.ndarray) -> np.ndarray:
    """
    Trim or pad audio to exactly 3 seconds.
    """
    audio = np.asarray(audio, dtype=np.float32)

    if audio.ndim != 1 or audio.size == 0:
        raise ValueError("Expected non-empty mono audio.")

    if not np.all(np.isfinite(audio)):
        raise ValueError("Audio contains invalid values.")

    if len(audio) > TARGET_SAMPLES:
        audio = audio[:TARGET_SAMPLES]

    elif len(audio) < TARGET_SAMPLES:
        padding = TARGET_SAMPLES - len(audio)

        audio = np.pad(
            audio,
            (0, padding),
            mode="constant"
        )

    return audio


def extract_features(audio: np.ndarray) -> np.ndarray:
    """
    Extract a log-Mel spectrogram.

    Returns:
        NumPy array with shape (64, 188).
    """
    audio = standardize_audio(audio)

    mel_spectrogram = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=2.0
    )

    log_mel = librosa.power_to_db(
        mel_spectrogram,
        ref=1.0
    )

    # Deterministically enforce the target shape.
    log_mel = log_mel[:, :TARGET_FRAMES]

    if log_mel.shape[1] < TARGET_FRAMES:
        log_mel = np.pad(
            log_mel,
            ((0, 0), (0, TARGET_FRAMES - log_mel.shape[1]))
        )

    return log_mel.astype(np.float32)


def preprocess_audio(file_path: str | Path) -> np.ndarray:
    """
    Complete pipeline:
    file -> audio -> standardized audio -> features
    """
    audio = load_audio(file_path)
    return extract_features(audio)

