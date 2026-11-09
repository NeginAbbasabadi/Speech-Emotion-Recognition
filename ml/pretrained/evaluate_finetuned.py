
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader
from transformers import Wav2Vec2ForSequenceClassification

from ml.model import EMOTIONS
from ml.pretrained.dataset import Wav2Vec2Dataset


ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models" / "wav2vec2_finetuned"
RESULTS_DIR = ROOT / "results" / "wav2vec2_finetuned"


def normalize_audio(audio):
    mean = audio.mean(dim=1, keepdim=True)
    std = audio.std(dim=1, keepdim=True, unbiased=False)
    return (audio - mean) / std.clamp(min=1e-7)


def main():
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    print("Device:", device)
    print("Loading best fine-tuned checkpoint...")

    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        MODEL_DIR
    )

    for i, emotion in enumerate(EMOTIONS):
        if model.config.id2label[i] != emotion:
            raise ValueError("Emotion label order mismatch")

    model.to(device)
    model.eval()

    dataset = Wav2Vec2Dataset("test")
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )

    all_predictions = []
    all_labels = []

    with torch.inference_mode():
        for audio, labels in loader:
            audio = normalize_audio(audio.to(device))

            outputs = model(input_values=audio)
            predicted = outputs.logits.argmax(dim=1)

            all_predictions.extend(
                predicted.cpu().tolist()
            )
            all_labels.extend(labels.tolist())

    accuracy = accuracy_score(
        all_labels, all_predictions
    )

    macro_f1 = f1_score(
        all_labels,
        all_predictions,
        labels=list(range(len(EMOTIONS))),
        average="macro",
        zero_division=0,
    )

    report = classification_report(
        all_labels,
        all_predictions,
        labels=list(range(len(EMOTIONS))),
        target_names=EMOTIONS,
        digits=4,
        zero_division=0,
    )

    matrix = confusion_matrix(
        all_labels,
        all_predictions,
        labels=list(range(len(EMOTIONS))),
    )

    print("\n===== WAV2VEC2 TEST RESULTS =====")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Macro F1: {macro_f1:.4f}")
    print("\nClassification Report:")
    print(report)
    print("Confusion matrix:\n", matrix)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    metrics = {
        "test_accuracy": float(accuracy),
        "test_macro_f1": float(macro_f1),
        "test_samples": len(all_labels),
        "emotions": EMOTIONS,
        "confusion_matrix": matrix.tolist(),
    }

    (RESULTS_DIR / "test_metrics.json").write_text(
        json.dumps(metrics, indent=4)
    )

    (RESULTS_DIR / "classification_report.txt").write_text(
        report
    )

    print("\nResults saved to:", RESULTS_DIR)


if __name__ == "__main__":
    main()
