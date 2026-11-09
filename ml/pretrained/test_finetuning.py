import torch
from torch.utils.data import DataLoader
from transformers import Wav2Vec2ForSequenceClassification

from ml.pretrained.dataset import Wav2Vec2Dataset


MODEL_PATH = "models/experiments/wav2vec2_frozen"

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

print("Device:", device)
print("Loading frozen-experiment checkpoint...")

model = Wav2Vec2ForSequenceClassification.from_pretrained(
    MODEL_PATH
)

# Start by freezing the entire speech encoder.
model.wav2vec2.requires_grad_(False)

# Unfreeze the final two transformer layers.
layers = model.wav2vec2.encoder.layers

for layer in layers[-2:]:
    layer.requires_grad_(True)

# Classification layers remain trainable.
model.projector.requires_grad_(True)
model.classifier.requires_grad_(True)

print("Total transformer layers:", len(layers))
print("Unfrozen transformer layers:", list(
    range(len(layers) - 2, len(layers))
))

print(
    "Trainable parameters:",
    sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
)

model.to(device)
model.train()

dataset = Wav2Vec2Dataset("train")
loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True,
    num_workers=0
)

audio, labels = next(iter(loader))

audio = audio.to(device)
labels = labels.to(device)

# Match the waveform normalization used in our
# previous frozen-encoder training experiment.
mean = audio.mean(dim=1, keepdim=True)
std = audio.std(dim=1, keepdim=True, unbiased=False)
audio = (audio - mean) / std.clamp(min=1e-7)

optimizer = torch.optim.AdamW(
    [
        {
            "params": [
                p for p in model.wav2vec2.parameters()
                if p.requires_grad
            ],
            "lr": 1e-5,
        },
        {
            "params": list(model.projector.parameters())
            + list(model.classifier.parameters()),
            "lr": 1e-4,
        },
    ]
)

optimizer.zero_grad()

outputs = model(input_values=audio)
loss = torch.nn.functional.cross_entropy(
    outputs.logits, labels
)

print("Loss before update:", loss.item())

loss.backward()

last_layer = layers[-1]
gradient_found = any(
    p.grad is not None
    for p in last_layer.parameters()
    if p.requires_grad
)

print("Last encoder layer has gradients:", gradient_found)

optimizer.step()

print("Partial fine-tuning smoke test successful!")
