
import csv
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader
from transformers import Wav2Vec2ForSequenceClassification

from ml.model import EMOTIONS
from ml.pretrained.dataset import Wav2Vec2Dataset


SEED = 42
MODEL_NAME = "facebook/wav2vec2-base"
EPOCHS = 10
BATCH_SIZE = 4
LEARNING_RATE = 1e-3
PATIENCE = 4

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models" / "wav2vec2"
RESULTS_DIR = ROOT / "results" / "wav2vec2"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)


def prepare_waveforms(audio):
    # Wav2Vec2 expects raw waveforms with appropriate scaling.
    # Standardize each recording independently.
    mean = audio.mean(dim=1, keepdim=True)
    std = audio.std(dim=1, keepdim=True, unbiased=False)
    return (audio - mean) / std.clamp(min=1e-7)


def run_epoch(model, loader, criterion, optimizer=None):
    training = optimizer is not None

    if training:
        model.train()
        # Keep the frozen speech encoder in evaluation mode.
        model.wav2vec2.eval()
    else:
        model.eval()

    total_loss = 0.0
    labels_all = []
    predictions_all = []

    for audio, labels in loader:
        audio = prepare_waveforms(audio.to(device))
        labels = labels.to(device)

        with torch.set_grad_enabled(training):
            logits = model(input_values=audio).logits
            loss = criterion(logits, labels)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * labels.size(0)

        predictions_all.extend(
            logits.argmax(dim=1).detach().cpu().tolist()
        )
        labels_all.extend(labels.cpu().tolist())

    return {
        "loss": total_loss / len(loader.dataset),
        "accuracy": accuracy_score(
            labels_all, predictions_all
        ),
        "macro_f1": f1_score(
            labels_all,
            predictions_all,
            labels=list(range(len(EMOTIONS))),
            average="macro",
            zero_division=0,
        ),
    }


def main():
    print("Device:", device)

    train_dataset = Wav2Vec2Dataset("train")
    val_dataset = Wav2Vec2Dataset("validation")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    print("Loading Wav2Vec2...")
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(EMOTIONS),
        id2label={i: e for i, e in enumerate(EMOTIONS)},
        label2id={e: i for i, e in enumerate(EMOTIONS)},
    )

    # Freeze all pretrained speech encoder parameters.
    model.wav2vec2.requires_grad_(False)
    model.to(device)

    trainable = [
        p for p in model.parameters() if p.requires_grad
    ]

    print(
        "Trainable parameters:",
        sum(p.numel() for p in trainable),
    )

    optimizer = torch.optim.AdamW(
        trainable,
        lr=LEARNING_RATE,
        weight_decay=0.01,
    )

    # Compensate for unequal class frequencies.
    counts = torch.bincount(
        train_dataset_labels(train_dataset),
        minlength=len(EMOTIONS),
    ).float()
    weights = counts.sum() / (
        len(EMOTIONS) * counts.clamp(min=1)
    )

    criterion = nn.CrossEntropyLoss(
        weight=weights.to(device)
    )

    best_f1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, EPOCHS + 1):
        train_metrics = run_epoch(
            model, train_loader, criterion, optimizer
        )
        val_metrics = run_epoch(
            model, val_loader, criterion
        )

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
        }
        history.append(row)

        print(
            f"Epoch {epoch:02d} | "
            f"Train F1: {train_metrics['macro_f1']:.4f} | "
            f"Val F1: {val_metrics['macro_f1']:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f}"
        )

        with open(
            RESULTS_DIR / "training_history.csv",
            "w",
            newline="",
        ) as file:
            writer = csv.DictWriter(
                file, fieldnames=list(row.keys())
            )
            writer.writeheader()
            writer.writerows(history)

        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            epochs_without_improvement = 0

            # Save model configuration and weights.
            model.save_pretrained(MODEL_DIR)

            with open(RESULTS_DIR / "best_model.txt", "w") as file:
                file.write(f"Best epoch: {best_epoch}\n")
                file.write(f"Validation macro F1: {best_f1:.4f}\n")

            print("  Saved best checkpoint.")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            print("Early stopping.")
            break

    print("\nTraining finished.")
    print("Best epoch:", best_epoch)
    print("Best validation macro F1:", round(best_f1, 4))


def train_dataset_labels(dataset):
    return torch.tensor(
        [
            list(EMOTIONS).index(emotion)
            for emotion in dataset.records["emotion"]
        ],
        dtype=torch.long,
    )


if __name__ == "__main__":
    main()
