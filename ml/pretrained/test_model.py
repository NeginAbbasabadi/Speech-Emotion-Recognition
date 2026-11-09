
import torch
from transformers import Wav2Vec2ForSequenceClassification

MODEL_NAME = "facebook/wav2vec2-base"

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

print("Using device:", device)
print("Loading pretrained Wav2Vec2...")

model = Wav2Vec2ForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=4,
)

model = model.to(device)
model.eval()

# Synthetic 1-second audio, sampled at 16 kHz
audio = torch.zeros(1, 16000, device=device)

with torch.inference_mode():
    outputs = model(input_values=audio)

print("Output shape:", outputs.logits.shape)
print("Wav2Vec2 forward pass successful!")

