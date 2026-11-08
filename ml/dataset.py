
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from backend.preprocessing import preprocess_audio
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
    Scale log-Mel values for the CNN.

    This exact transformation must also be used
    during web application inference.
    """
    return np.clip(
        (features + 80.0) / 80.0,
        0.0,
        1.0
    ).astype(np.float32)


class RAVDESSDataset(Dataset):
    """
    PyTorch dataset for the RAVDESS recordings.

    Each sample returns:
        features: Tensor (1, 64, 188)
        label: Integer emotion class
    """

    def __init__(self, split):
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"Invalid split: {split}")

        df = pd.read_csv(MANIFEST_PATH)
        df = df[df["split"] == split].reset_index(drop=True)

        if df.empty:
            raise ValueError(f"No recordings for split: {split}")

        self.features = []
        self.labels = []

        print(f"Loading {split} recordings...")

        for row in df.itertuples(index=False):
            audio_path = PROJECT_ROOT / row.file_path

            # Load and extract the log-Mel spectrogram.
            features = preprocess_audio(audio_path)
            features = normalize_spectrogram(features)

            # Add CNN channel dimension:
            # (64, 188) -> (1, 64, 188)
            tensor = torch.from_numpy(
                features
            ).unsqueeze(0)

            label = EMOTION_TO_ID[row.emotion]

            self.features.append(tensor)
            self.labels.append(label)

        # Precompute features once instead of
        # repeating Librosa processing each epoch.
        self.features = torch.stack(self.features)

        self.labels = torch.tensor(
            self.labels,
            dtype=torch.long
        )

        print(f"Loaded {len(self.labels)} recordings.")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]
