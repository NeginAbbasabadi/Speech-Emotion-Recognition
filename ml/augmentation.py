
import numpy as np

from backend.preprocessing import SAMPLE_RATE


def add_background_noise(audio, min_snr=18, max_snr=30):
    """
    Add mild Gaussian noise at a random
    signal-to-noise ratio (SNR).
    """
    audio = audio.astype(np.float32, copy=True)

    signal_rms = np.sqrt(np.mean(audio ** 2))

    if signal_rms < 1e-8:
        return audio

    snr_db = np.random.uniform(min_snr, max_snr)

    noise_rms = signal_rms / (10 ** (snr_db / 20))

    noise = np.random.normal(
        0.0,
        noise_rms,
        size=audio.shape
    ).astype(np.float32)

    return audio + noise


def change_volume(audio):
    """
    Apply a random volume multiplier.
    """
    gain = np.random.uniform(0.8, 1.2)

    return (audio * gain).astype(np.float32)


def time_shift(audio, max_shift_seconds=0.08):
    """
    Shift the waveform while preserving
    its original length.

    Unlike np.roll, this does not wrap
    samples around the recording.
    """
    max_samples = int(
        SAMPLE_RATE * max_shift_seconds
    )

    shift = np.random.randint(
        -max_samples,
        max_samples + 1
    )

    result = np.zeros_like(audio)

    if shift > 0:
        result[shift:] = audio[:-shift]

    elif shift < 0:
        result[:shift] = audio[-shift:]

    else:
        result = audio.copy()

    return result


def augment_audio(audio):
    """
    Apply random audio augmentation.

    Each transformation has its own
    probability of being applied.
    """
    augmented = audio.copy()

    # 60% probability of noise
    if np.random.random() < 0.60:
        augmented = add_background_noise(augmented)

    # 50% probability of volume changes
    if np.random.random() < 0.50:
        augmented = change_volume(augmented)

    # 50% probability of time shifting
    if np.random.random() < 0.50:
        augmented = time_shift(augmented)

    return augmented.astype(np.float32)
