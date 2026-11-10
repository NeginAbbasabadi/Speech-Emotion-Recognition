
import numpy as np
import pytest

from backend.preprocessing import SAMPLE_RATE
from ml.augmentation import (
    augment_audio,
    add_background_noise,
    change_volume,
    time_shift,
)
from ml.dataset import RAVDESSDataset


def test_augmentation_preserves_shape():
    audio = np.ones(
        SAMPLE_RATE * 3,
        dtype=np.float32
    )

    augmented = augment_audio(audio)

    assert augmented.shape == audio.shape
    assert augmented.dtype == np.float32
    assert np.all(np.isfinite(augmented))


def test_noise_changes_signal():
    audio = np.ones(
        SAMPLE_RATE,
        dtype=np.float32
    )

    noisy = add_background_noise(audio)

    assert noisy.shape == audio.shape
    assert not np.array_equal(noisy, audio)


def test_volume_preserves_shape():
    audio = np.ones(
        SAMPLE_RATE,
        dtype=np.float32
    )

    adjusted = change_volume(audio)

    assert adjusted.shape == audio.shape


def test_time_shift_preserves_length():
    audio = np.ones(
        SAMPLE_RATE,
        dtype=np.float32
    )

    shifted = time_shift(audio)

    assert len(shifted) == len(audio)


def test_validation_augmentation_rejected():
    with pytest.raises(ValueError):
        RAVDESSDataset(
            "validation",
            augment=True
        )
