from pathlib import Path

import pandas as pd


# Project directories
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_DIR = PROJECT_ROOT / "data" / "ravdess"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

# RAVDESS emotion codes
EMOTION_MAP = {
    "01": "neutral",
    "03": "happy",
    "04": "sad",
    "05": "angry",
}

# Speaker-independent splits
TRAIN_ACTORS = set(range(1, 17))
VAL_ACTORS = set(range(17, 21))
TEST_ACTORS = set(range(21, 25))


def get_split(actor_id):
    """Assign an actor to exactly one dataset split."""

    if actor_id in TRAIN_ACTORS:
        return "train"

    if actor_id in VAL_ACTORS:
        return "validation"

    if actor_id in TEST_ACTORS:
        return "test"

    raise ValueError(f"Unknown actor: {actor_id}")


def prepare_dataset():
    """Build a labeled manifest of RAVDESS recordings."""

    if not DATASET_DIR.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {DATASET_DIR}"
        )

    records = []

    audio_files = sorted(DATASET_DIR.rglob("*.wav"))

    if not audio_files:
        raise ValueError("No WAV files found in RAVDESS.")

    for audio_path in audio_files:
        # Example:
        # 03-01-05-01-02-01-12.wav
        parts = audio_path.stem.split("-")

        if len(parts) != 7:
            continue

        # Keep speech recordings only
        if parts[1] != "01":
            continue

        emotion_code = parts[2]

        if emotion_code not in EMOTION_MAP:
            continue

        actor_id = int(parts[6])
        emotion = EMOTION_MAP[emotion_code]
        split = get_split(actor_id)

        records.append({
            "file_path": str(
                audio_path.relative_to(PROJECT_ROOT)
            ),
            "actor_id": actor_id,
            "emotion": emotion,
            "split": split,
        })

    if not records:
        raise ValueError("No matching speech recordings found.")

    df = pd.DataFrame(records)

    # Verify that each actor belongs to one split
    actor_split_counts = df.groupby("actor_id")["split"].nunique()

    assert (actor_split_counts == 1).all(), (
        "An actor appears in multiple splits!"
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = OUTPUT_DIR / "dataset_manifest.csv"

    df.to_csv(output_path, index=False)

    print("\nDataset preparation completed!")
    print(f"Total recordings: {len(df)}")

    print("\nEmotion distribution:")
    print(df["emotion"].value_counts().sort_index())

    print("\nDataset splits:")
    print(df["split"].value_counts())

    print("\nSplit and emotion distribution:")
    print(pd.crosstab(df["split"], df["emotion"]))

    print(f"\nManifest saved to: {output_path}")


if __name__ == "__main__":
    prepare_dataset()
