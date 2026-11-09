import csv
import json
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


# Experiment configuration
SEED = 42
EPOCHS = 8
PATIENCE = 3
BATCH_SIZE = 2

ENCODER_LR = 1e-5
HEAD_LR = 1e-4

ROOT = Path(__file__).resolve().parents[2]

SOURCE_MODEL = (
    ROOT / "models" / "experiments" / "wav2vec2_frozen"
)

MODEL_DIR = ROOT / "models" / "wav2vec2_finetuned"
RESULTS_DIR = ROOT / "results" / "wav2vec2_finetuned"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def normalize_audio(audio):
    mean = audio.mean(dim=1, keepdim=True)
    std = audio.std(
        dim=1, keepdim=True, unbiased=False
    )
    return (audio - mean) / std.clamp(min=1e-7)


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None

    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    predictions = []
    targets = []

    for audio, labels in loader:
        audio = normalize_audio(audio.to(device))
        labels = labels.to(device)

        with torch.set_grad_enabled(training):
            outputs = model(input_values=audio)
            logits = outputs.logits

            loss = criterion(logits, labels)

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()

                # Prevent excessively large updates.
                torch.nn.utils.clip_grad_norm_(
                    (p for p in model.parameters()
                     if p.requires_grad),
                    max_norm=1.0,
                )

                optimizer.step()

        total_loss += loss.item() * labels.size(0)

        predictions.extend(
            logits.detach().argmax(dim=1).cpu().tolist()
        )
        targets.extend(labels.cpu().tolist())

    return {
        "loss": total_loss / len(loader.dataset),
        "accuracy": accuracy_score(targets, predictions),
        "macro_f1": f1_score(
            targets,
            predictions,
            labels=list(range(len(EMOTIONS))),
            average="macro",
            zero_division=0,
        ),
    }


def main():
    set_seed(SEED)

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    print("Device:", device)
    print("Starting partial fine-tuning...")

    if not SOURCE_MODEL.exists():
        raise FileNotFoundError(SOURCE_MODEL)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

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

    # Start from the frozen-model experiment.
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        SOURCE_MODEL
    )

    # Freeze pretrained speech encoder.
    model.wav2vec2.requires_grad_(False)

    # Unfreeze only its last two transformer layers.
    for layer in model.wav2vec2.encoder.layers[-2:]:
        layer.requires_grad_(True)

    # Train classification head.
    model.projector.requires_grad_(True)
    model.classifier.requires_grad_(True)

    model.to(device)

    encoder_params = [
        p for p in model.wav2vec2.parameters()
        if p.requires_grad
    ]

    head_params = (
        list(model.projector.parameters())
        + list(model.classifier.parameters())
    )

    optimizer = torch.optim.AdamW(
        [
            {
                "params": encoder_params,
                "lr": ENCODER_LR,
            },
            {
                "params": head_params,
                "lr": HEAD_LR,
            },
        ],
        weight_decay=0.01,
    )

    print(
        "Trainable parameters:",
        sum(p.numel() for p in model.parameters()
            if p.requires_grad)
    )

    # Calculate class weights from training labels only.
    label_ids = [
        EMOTIONS.index(emotion)
        for emotion in train_dataset.records["emotion"]
    ]

    counts = torch.bincount(
        torch.tensor(label_ids),
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
    no_improvement = 0
    history = []

    for epoch in range(1, EPOCHS + 1):
        train_metrics = run_epoch(
            model, train_loader, criterion,
            device, optimizer
        )

        val_metrics = run_epoch(
            model, val_loader, criterion, device
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
            f"Val Acc: {val_metrics['accuracy']:.4f}",
            flush=True,
        )

        # Save training history each epoch.
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

        if not np.isfinite(val_metrics["macro_f1"]):
            raise RuntimeError("Non-finite validation metric.")

        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            no_improvement = 0

            model.save_pretrained(MODEL_DIR)

            metadata = {
                "best_epoch": best_epoch,
                "validation_macro_f1": float(best_f1),
                "source_model": str(SOURCE_MODEL),
                "unfrozen_encoder_layers": [10, 11],
                "seed": SEED,
                "emotions": EMOTIONS,
            }

            with open(
                RESULTS_DIR / "best_model.json",
                "w",
            ) as file:
                json.dump(metadata, file, indent=4)

            print("  Saved best fine-tuned checkpoint.")

        else:
            no_improvement += 1

        if no_improvement >= PATIENCE:
            print("Early stopping.")
            break

    print("\nFine-tuning finished.")
    print("Best epoch:", best_epoch)
    print("Best validation macro F1:", round(best_f1, 4))


if __name__ == "__main__":
    main()
