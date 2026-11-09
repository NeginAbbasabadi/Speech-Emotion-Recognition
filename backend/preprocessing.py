
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

# Silence detection
TRIM_DB = 30
TRIM_FRAME_LENGTH = 512
TRIM_HOP_LENGTH = 128


def load_audio(file_path: str | Path) -> np.ndarray:
    """
    Load an audio file, convert to mono,
    and resample to 16 kHz.
    """
    audio, _ = librosa.load(
        file_path,
        sr=SAMPLE_RATE,
        mono=True
    )

    if audio.size == 0:
        raise ValueError("Audio file is empty.")

    if not np.all(np.isfinite(audio)):
        raise ValueError("Audio contains invalid values.")

    return audio.astype(np.float32)


def remove_silence(audio: np.ndarray) -> np.ndarray:
    """
    Trim low-energy silence at the beginning
    and end of a recording.
    """
    audio = np.asarray(audio, dtype=np.float32)

    if audio.ndim != 1 or audio.size == 0:
        raise ValueError("Expected non-empty mono audio.")

    if not np.all(np.isfinite(audio)):
        raise ValueError("Audio contains invalid values.")

    # Avoid trimming an entirely silent recording.
    if np.max(np.abs(audio)) < 1e-5:
        return audio

    trimmed_audio, _ = librosa.effects.trim(
        audio,
        top_db=TRIM_DB,
        frame_length=TRIM_FRAME_LENGTH,
        hop_length=TRIM_HOP_LENGTH
    )

    if trimmed_audio.size == 0:
        return audio

    return trimmed_audio


def standardize_audio(audio: np.ndarray) -> np.ndarray:
    """
    Center-pad or center-crop audio so its
    duration is exactly 3 seconds.
    """
    audio = np.asarray(audio, dtype=np.float32)

    if audio.ndim != 1 or audio.size == 0:
        raise ValueError("Expected non-empty mono audio.")

    if not np.all(np.isfinite(audio)):
        raise ValueError("Audio contains invalid values.")

    current_length = len(audio)

    if current_length > TARGET_SAMPLES:
        # Crop equally from the beginning and end.
        start = (current_length - TARGET_SAMPLES) // 2

        audio = audio[
            start:start + TARGET_SAMPLES
        ]

    elif current_length < TARGET_SAMPLES:
        # Add silence equally to both sides.
        total_padding = TARGET_SAMPLES - current_length
        left_padding = total_padding // 2
        right_padding = total_padding - left_padding

        audio = np.pad(
            audio,
            (left_padding, right_padding),
            mode="constant"
        )

    return audio


def extract_features(
    audio: np.ndarray,
    trim_silence: bool = True
) -> np.ndarray:
    """
    Convert audio into a log-Mel spectrogram.

    Output shape: (64, 188)
    """
    if trim_silence:
        audio = remove_silence(audio)

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

    # Ensure the model always receives
    # the same spectrogram dimensions.
    log_mel = log_mel[:, :TARGET_FRAMES]

    if log_mel.shape[1] < TARGET_FRAMES:
        log_mel = np.pad(
            log_mel,
            (
                (0, 0),
                (0, TARGET_FRAMES - log_mel.shape[1])
            )
        )

    return log_mel.astype(np.float32)


def preprocess_audio(file_path: str | Path) -> np.ndarray:
    """
    Complete preprocessing pipeline.
    """
    audio = load_audio(file_path)
    return extract_features(audio)