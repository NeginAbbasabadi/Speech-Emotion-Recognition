
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = "NeginAbbasabadi/voxsense-ai"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"

MODEL_FILES = [
    "emotion_cnn.pt",
    "wav2vec2_finetuned/config.json",
    "wav2vec2_finetuned/model.safetensors",
]


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for filename in MODEL_FILES:
        print(f"Downloading {filename}...")

        hf_hub_download(
            repo_id=REPO_ID,
            filename=filename,
            local_dir=MODELS_DIR,
        )

    print("All production models are available!")


if __name__ == "__main__":
    main()
