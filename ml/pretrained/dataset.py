
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from backend.preprocessing import (
    load_audio,
    remove_silence,
    standardize_audio,
)
from ml.dataset import MANIFEST_PATH, EMOTION_TO_ID


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Wav2Vec2Dataset(Dataset):
    def __init__(self, split):
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"Invalid split: {split}")

        df = pd.read_csv(MANIFEST_PATH)
        self.records = df[df["split"] == split].reset_index(drop=True)

        if self.records.empty:
            raise ValueError(f"No recordings found for {split}")

        print(f"{split}: {len(self.records)} recordings")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records.iloc[index]
        audio_path = PROJECT_ROOT / row["file_path"]

        # Load and prepare the 16 kHz waveform.
        audio = load_audio(audio_path)
        audio = remove_silence(audio)
        audio = standardize_audio(audio)

        # Wav2Vec2 expects raw waveform values, not a spectrogram.
        waveform = torch.tensor(audio, dtype=torch.float32)
        label = torch.tensor(
            EMOTION_TO_ID[row["emotion"]],
            dtype=torch.long
        )

        return waveform, label
