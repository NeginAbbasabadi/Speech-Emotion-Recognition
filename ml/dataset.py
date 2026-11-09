
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from backend.preprocessing import (
    load_audio,
    remove_silence,
    standardize_audio,
    extract_features,
)

from ml.augmentation import augment_audio
from ml.model import EMOTIONS


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MANIFEST_PATH = (
    PROJECT_ROOT / "data" / "processed" / "dataset_manifest.csv"
)

EMOTION_TO_ID = {
    emotion: index
    for index, emotion in enumerate(EMOTIONS)
}


def normalize_spectrogram(features):
    """
    Apply the same fixed spectrogram scaling
    for training, validation, and inference.
    """
    return np.clip(
        (features + 80.0) / 80.0,
        0.0,
        1.0
    ).astype(np.float32)


class RAVDESSDataset(Dataset):
    """
    Dataset supporting optional training-only
    waveform augmentation.
    """

    def __init__(self, split, augment=False):
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"Invalid split: {split}")

        if augment and split != "train":
            raise ValueError(
                "Augmentation is only permitted for training."
            )

        self.split = split
        self.augment = augment

        df = pd.read_csv(MANIFEST_PATH)

        df = df[
            df["split"] == split
        ].reset_index(drop=True)

        if df.empty:
            raise ValueError(
                f"No recordings found for split: {split}"
            )

        self.audio_data = []
        self.features = []
        self.labels = []

        print(f"Loading {split} recordings...")

        for row in df.itertuples(index=False):
            audio_path = PROJECT_ROOT / row.file_path

            audio = load_audio(audio_path)
            audio = remove_silence(audio)
            audio = standardize_audio(audio)

            if augment:
                # Store standardized waveform.
                # Augmentation will happen dynamically.
                self.audio_data.append(audio)

            else:
                # Clean spectrograms are computed once.
                features = extract_features(
                    audio,
                    trim_silence=False
                )

                features = normalize_spectrogram(
                    features
                )

                self.features.append(
                    torch.from_numpy(features).unsqueeze(0)
                )

            self.labels.append(
                EMOTION_TO_ID[row.emotion]
            )

        self.labels = torch.tensor(
            self.labels,
            dtype=torch.long
        )

        if not augment:
            self.features = torch.stack(self.features)

        print(
            f"Loaded {len(self.labels)} recordings "
            f"(augmentation={augment})."
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        label = self.labels[index]

        if self.augment:
            audio = self.audio_data[index]

            audio = augment_audio(audio)

            features = extract_features(
                audio,
                trim_silence=False
            )

            features = normalize_spectrogram(
                features
            )

            tensor = torch.from_numpy(
                features
            ).unsqueeze(0)

            return tensor, label

        return self.features[index], label
