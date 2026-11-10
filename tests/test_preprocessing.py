"""Unit tests for VoxSense audio loading and preprocessing."""

import numpy as np
import pytest
import soundfile as sf

from backend.preprocessing import (
    SAMPLE_RATE,
    TARGET_SAMPLES,
    extract_features,
    load_audio,
    remove_silence,
    standardize_audio,
)
from ml.dataset import normalize_spectrogram


@pytest.fixture
def sample_audio():
    """One second of synthetic 440 Hz mono audio."""
    t = np.arange(SAMPLE_RATE) / SAMPLE_RATE
    return (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


def test_sample_rate():
    assert SAMPLE_RATE == 16000
    assert TARGET_SAMPLES == SAMPLE_RATE * 3


def test_load_audio_mono(tmp_path, sample_audio):
    path = tmp_path / "mono.wav"
    sf.write(path, sample_audio, SAMPLE_RATE)

    audio = load_audio(path)
    assert audio.ndim == 1
    assert len(audio) == SAMPLE_RATE


def test_stereo_audio_becomes_mono(tmp_path, sample_audio):
    path = tmp_path / "stereo.wav"
    sf.write(path, np.stack([sample_audio, sample_audio], axis=1), SAMPLE_RATE)

    audio = load_audio(path)
    assert audio.ndim == 1
    assert len(audio) == SAMPLE_RATE


def test_audio_is_resampled(tmp_path):
    original_sample_rate = 22050
    t = np.arange(original_sample_rate) / original_sample_rate
    signal = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    path = tmp_path / "audio_22050.wav"
    sf.write(path, signal, original_sample_rate)

    audio = load_audio(path)
    assert audio.ndim == 1
    assert len(audio) == SAMPLE_RATE


def test_remove_silence(sample_audio):
    silence = np.zeros(SAMPLE_RATE, dtype=np.float32)
    audio = np.concatenate([silence, sample_audio, silence])

    trimmed = remove_silence(audio)
    assert 0 < len(trimmed) < len(audio)
    assert np.max(np.abs(trimmed)) > 0.4


def test_short_audio_is_padded(sample_audio):
    result = standardize_audio(sample_audio)
    assert len(result) == TARGET_SAMPLES
    assert np.count_nonzero(result) > 0


def test_center_padding():
    audio = np.ones(SAMPLE_RATE, dtype=np.float32)
    result = standardize_audio(audio)

    assert len(result) == TARGET_SAMPLES
    assert result[0] == 0
    assert result[-1] == 0
    assert result[TARGET_SAMPLES // 2] == 1


def test_long_audio_is_cropped(sample_audio):
    result = standardize_audio(np.tile(sample_audio, 5))
    assert len(result) == TARGET_SAMPLES


def test_center_cropping():
    audio = np.arange(SAMPLE_RATE * 5, dtype=np.float32)
    result = standardize_audio(audio)

    assert len(result) == TARGET_SAMPLES
    assert result[0] == SAMPLE_RATE


def test_exact_length_unchanged(sample_audio):
    audio = np.tile(sample_audio, 3)
    result = standardize_audio(audio)

    assert len(result) == TARGET_SAMPLES
    np.testing.assert_allclose(result, audio, atol=1e-6)


def test_spectrogram_shape(sample_audio):
    features = extract_features(sample_audio)
    assert features.shape == (64, 188)
    assert np.all(np.isfinite(features))


def test_spectrogram_normalization(sample_audio):
    features = extract_features(sample_audio)
    normalized = normalize_spectrogram(features)

    assert normalized.shape == (64, 188)
    assert normalized.dtype == np.float32
    assert normalized.min() >= 0.0
    assert normalized.max() <= 1.0


def test_all_silent_audio_is_handled():
    audio = np.zeros(SAMPLE_RATE, dtype=np.float32)
    trimmed = remove_silence(audio)

    assert trimmed is not None
    assert np.all(np.isfinite(trimmed))


def test_empty_audio_standardization():
    audio = np.array([], dtype=np.float32)
    with pytest.raises(ValueError, match="Expected non-empty mono audio"):
        standardize_audio(audio)
