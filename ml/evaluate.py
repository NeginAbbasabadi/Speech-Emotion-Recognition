import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
)

from ml.dataset import RAVDESSDataset
from ml.model import EmotionCNN, EMOTIONS


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "models" / "emotion_cnn.pt"
RESULTS_DIR = PROJECT_ROOT / "results"

REPORT_PATH = RESULTS_DIR / "classification_report.txt"
METRICS_PATH = RESULTS_DIR / "test_metrics.json"
MATRIX_PATH = RESULTS_DIR / "confusion_matrix.png"


def get_device():
    """Use Apple Silicon GPU when available."""
    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def load_trained_model(device):
    """
    Load the best model checkpoint saved during training.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {MODEL_PATH}"
        )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True
    )

    if checkpoint["emotions"] != EMOTIONS:
        raise ValueError(
            "Checkpoint emotion labels do not match the model."
        )

    model = EmotionCNN(
        num_classes=len(EMOTIONS)
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)
    model.eval()

    return model, checkpoint


def evaluate():
    """Evaluate the CNN on held-out test speakers."""

    device = get_device()
    print(f"Using device: {device}")

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Load only test recordings.
    test_dataset = RAVDESSDataset("test")

    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0
    )

    model, checkpoint = load_trained_model(device)

    print(
        f"Evaluating checkpoint from epoch "
        f"{checkpoint['best_epoch']}"
    )

    all_labels = []
    all_predictions = []

    # Disable gradient tracking during evaluation.
    with torch.inference_mode():

        for features, labels in test_loader:
            features = features.to(device)

            outputs = model(features)

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            all_labels.extend(
                labels.cpu().numpy().tolist()
            )

            all_predictions.extend(
                predictions.cpu().numpy().tolist()
            )

    # Classification metrics
    class_ids = list(range(len(EMOTIONS)))

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    macro_f1 = f1_score(
        all_labels,
        all_predictions,
        labels=class_ids,
        average="macro",
        zero_division=0
    )

    report = classification_report(
        all_labels,
        all_predictions,
        labels=class_ids,
        target_names=EMOTIONS,
        digits=4,
        zero_division=0
    )

    print("\n===== TEST RESULTS =====")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Macro F1: {macro_f1:.4f}")

    print("\nClassification Report:")
    print(report)

    # Save text report.
    REPORT_PATH.write_text(
        report,
        encoding="utf-8"
    )

    # Save machine-readable results.
    metrics = {
        "checkpoint_epoch": checkpoint["best_epoch"],
        "test_samples": len(all_labels),
        "test_accuracy": float(accuracy),
        "test_macro_f1": float(macro_f1),
        "emotions": EMOTIONS,
    }

    with open(METRICS_PATH, "w") as file:
        json.dump(metrics, file, indent=4)

    # Generate confusion matrix.
    matrix = confusion_matrix(
        all_labels,
        all_predictions,
        labels=class_ids
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=EMOTIONS
    )

    fig, ax = plt.subplots(figsize=(8, 6))

    display.plot(
        ax=ax,
        cmap="Blues",
        values_format="d",
        colorbar=False
    )

    ax.set_title("Speech Emotion Recognition - Test Set")

    fig.tight_layout()
    fig.savefig(MATRIX_PATH, dpi=150)

    plt.close(fig)

    print("\nEvaluation files saved:")
    print(REPORT_PATH)
    print(METRICS_PATH)
    print(MATRIX_PATH)


if __name__ == "__main__":
    evaluate()
