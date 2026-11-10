"""Fast, checkpoint-free unit tests for the two prediction pipelines."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

import backend.inference as cnn
import backend.wav2vec2_inference as wav2vec2
from ml.model import EMOTIONS


@pytest.fixture
def waveform():
    t = np.arange(16000, dtype=np.float32) / 16000
    return (0.4 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


@pytest.fixture(params=[("cnn", cnn, "predict_emotion"),
                        ("wav2vec2", wav2vec2, "predict_wav2vec2")], ids=["cnn", "wav2vec2"])
def pipeline(request):
    return request.param


def assert_prediction(result):
    assert result["emotion"] in EMOTIONS
    assert set(result["probabilities"]) == set(EMOTIONS)
    assert all(0 <= p <= 1 for p in result["probabilities"].values())
    assert sum(result["probabilities"].values()) == pytest.approx(1.0, abs=1e-6)
    assert result["confidence"] == pytest.approx(result["probabilities"][result["emotion"]])
    assert result["confidence"] == pytest.approx(max(result["probabilities"].values()))


@pytest.mark.parametrize("invalid", [b"", None])
def test_empty_bytes_rejected(pipeline, invalid):
    _, _, func_name = pipeline
    module = pipeline[1]
    with pytest.raises(ValueError, match="Empty audio recording"):
        getattr(module, func_name)(invalid)


def test_decode_failure_is_clear(pipeline, monkeypatch):
    _, module, func_name = pipeline
    def fail_decode(*args, **kwargs):
        raise RuntimeError("unreadable")
    monkeypatch.setattr(module.librosa, "load", fail_decode)
    with pytest.raises(ValueError, match="decod"):
        getattr(module, func_name)(b"bad wav bytes")


@pytest.mark.parametrize("samples", [
    np.array([], dtype=np.float32),
    np.array([np.nan, 0.5], dtype=np.float32),
    np.array([np.inf, 0.5], dtype=np.float32),
    np.zeros(16000, dtype=np.float32),
])
def test_invalid_decoded_audio_rejected(pipeline, monkeypatch, samples):
    _, module, func_name = pipeline
    monkeypatch.setattr(module.librosa, "load", lambda *args, **kwargs: (samples, module.SAMPLE_RATE))
    with pytest.raises(ValueError, match="Invalid|silent"):
        getattr(module, func_name)(b"nonempty")


def test_cnn_prediction_output_and_tensor_shape(monkeypatch, waveform):
    monkeypatch.setattr(cnn.librosa, "load", lambda *args, **kwargs: (waveform, 16000))
    monkeypatch.setattr(cnn, "extract_features", lambda audio: np.ones((64, 188), dtype=np.float32))
    monkeypatch.setattr(cnn, "normalize_spectrogram", lambda features: features)

    def forward(x):
        assert x.shape == (1, 1, 64, 188)
        assert x.dtype == torch.float32
        return torch.tensor([[0.1, 3.0, 0.2, -0.5]], dtype=torch.float32)

    mock_model = Mock(side_effect=forward)
    monkeypatch.setattr(cnn, "load_model", lambda: (mock_model, torch.device("cpu")))

    result = cnn.predict_emotion(b"sample wav")
    assert_prediction(result)
    assert result["emotion"] == EMOTIONS[1]
    mock_model.assert_called_once()


def test_wav2vec2_prediction_normalization(monkeypatch, waveform):
    monkeypatch.setattr(wav2vec2.librosa, "load", lambda *args, **kwargs: (waveform, 16000))
    monkeypatch.setattr(wav2vec2, "remove_silence", lambda audio: audio)
    monkeypatch.setattr(wav2vec2, "standardize_audio", lambda audio: np.pad(audio, (16000, 16000)))

    def forward(*, input_values):
        assert input_values.shape == (1, 48000)
        assert torch.isfinite(input_values).all()
        assert input_values.mean().item() == pytest.approx(0.0, abs=1e-5)
        assert input_values.std(unbiased=False).item() == pytest.approx(1.0, abs=1e-4)
        return SimpleNamespace(logits=torch.tensor([[0.0, -1.0, 2.0, 0.5]]))
