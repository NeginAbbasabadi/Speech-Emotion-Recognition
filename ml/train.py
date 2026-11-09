import copy
import csv
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ml.dataset import RAVDESSDataset
from ml.model import EmotionCNN, EMOTIONS


# Training configuration
EPOCHS = 50
BATCH_SIZE = 16
LEARNING_RATE = 0.001
PATIENCE = 12
SEED = 42

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "models" / "emotion_cnn.pt"
RESULTS_DIR = PROJECT_ROOT / "results"

HISTORY_PATH = RESULTS_DIR / "training_history.csv"
CURVES_PATH = RESULTS_DIR / "learning_curves.png"


def set_seed(seed):
    """Improve experiment reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device():
    """Use Apple Silicon GPU when available."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def calculate_macro_f1(y_true, y_pred, num_classes):
    """Calculate F1 for each emotion, then average."""
    scores = []

    for cls in range(num_classes):
        tp = sum(
            t == cls and p == cls
            for t, p in zip(y_true, y_pred)
        )
        fp = sum(
            t != cls and p == cls
            for t, p in zip(y_true, y_pred)
        )
        fn = sum(
            t == cls and p != cls
            for t, p in zip(y_true, y_pred)
        )

        denominator = 2 * tp + fp + fn
        f1 = 2 * tp / denominator if denominator else 0.0
        scores.append(f1)

    return float(np.mean(scores))


def run_epoch(model, loader, criterion, device, optimizer=None):
    """Perform one training or validation epoch."""
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    correct = 0
    total = 0

    all_targets = []
    all_predictions = []

    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)

        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):
            outputs = model(features)
            loss = criterion(outputs, labels)

            if training:
                loss.backward()
                optimizer.step()

        predictions = outputs.argmax(dim=1)
        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size
        correct += (predictions == labels).sum().item()
        total += batch_size

        all_targets.extend(labels.cpu().tolist())
        all_predictions.extend(predictions.cpu().tolist())

    return {
        "loss": total_loss / total,
        "accuracy": correct / total,
        "f1": calculate_macro_f1(
            all_targets,
            all_predictions,
            len(EMOTIONS)
        )
    }




def save_history(history):
    """Save all training metrics to CSV."""

    if not history:
        return

    # Automatically get column names from the
    # training metrics dictionary.
    fieldnames = list(history[0].keys())

    with open(HISTORY_PATH, "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(history)



def plot_history(history):
    """Plot training and validation learning curves."""
    epochs = [row["epoch"] for row in history]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    metrics = [
        ("loss", "Loss"),
        ("accuracy", "Accuracy"),
        ("f1", "Macro F1")
    ]

    for ax, (metric, title) in zip(axes, metrics):
        train_values = [
            row[f"train_{metric}"] for row in history
        ]
        val_values = [
            row[f"val_{metric}"] for row in history
        ]

        ax.plot(epochs, train_values, label="Training")
        ax.plot(epochs, val_values, label="Validation")

        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(CURVES_PATH, dpi=150)
    plt.close(fig)


def save_best_model(model, epoch, val_f1):
    """Save the best model weights and metadata."""
    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Store weights on CPU for portable loading.
    best_state = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }

    checkpoint = {
        "model_state_dict": best_state,
        "emotions": EMOTIONS,
        "best_epoch": epoch,
        "validation_macro_f1": val_f1,
        "sample_rate": 16000,
        "duration_seconds": 3,
        "n_mels": 64,
    }

    torch.save(checkpoint, MODEL_PATH)


def train():
    set_seed(SEED)

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    device = get_device()
    print(f"Using device: {device}")

    train_dataset = RAVDESSDataset(
    "train",
    augment=False
    )

    val_dataset = RAVDESSDataset(
    "validation",
    augment=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    model = EmotionCNN().to(device)

    class_counts = torch.bincount(
        train_dataset.labels,
        minlength=len(EMOTIONS)
    ).float()

    class_weights = class_counts.sum() / (
        len(EMOTIONS) * class_counts.clamp(min=1)
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device)
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=3,
    min_lr=1e-5
    )

    history = []
    best_f1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0

    print("\nStarting training...\n")

    for epoch in range(1, EPOCHS + 1):

        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer
        )

        val_metrics = run_epoch(
            model,
            val_loader,
            criterion,
            device
        )

        epoch_record = {
             "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_f1": train_metrics["f1"],
            "val_loss": val_metrics["loss"],
                    "val_accuracy": val_metrics["accuracy"],
            "val_f1": val_metrics["f1"],
            "learning_rate": optimizer.param_groups[0]["lr"]
        }

        history.append(epoch_record)

        # Persist metrics after every epoch.
        save_history(history)
        scheduler.step(val_metrics["loss"])

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Train Acc: {train_metrics['accuracy']:.3f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.3f} | "
            f"Val F1: {val_metrics['f1']:.3f}"
        )

        # Save the best model immediately.
        if val_metrics["f1"] > best_f1 + 1e-6:
            best_f1 = val_metrics["f1"]
            best_epoch = epoch
            epochs_without_improvement = 0

            save_best_model(model, epoch, best_f1)
            print("  -> New best model saved!")

        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            print("\nEarly stopping triggered.")
            break

    plot_history(history)

    print("\nTraining completed!")
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation macro F1: {best_f1:.4f}")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"History saved to: {HISTORY_PATH}")
    print(f"Curves saved to: {CURVES_PATH}")


if __name__ == "__main__":
    train()
