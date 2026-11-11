
import numpy as np
import pytest

from backend.preprocessing import validate_audio_activity


def test_silent_audio_rejected():
    audio = np.zeros(16000, dtype=np.float32)

    with pytest.raises(ValueError, match="No audible speech"):
        validate_audio_activity(audio)


def test_low_level_noise_rejected():
    rng = np.random.default_rng(42)
    audio = rng.normal(0, 0.001, 16000).astype(np.float32)

    with pytest.raises(ValueError, match="No audible speech"):
        validate_audio_activity(audio)


def test_audible_signal_accepted():
    t = np.arange(16000) / 16000
    audio = (0.1 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)

    result = validate_audio_activity(audio)

    np.testing.assert_array_equal(result, audio)
