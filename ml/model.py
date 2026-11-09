
import torch
import torch.nn as nn


EMOTIONS = ["neutral", "happy", "sad", "angry"]
NUM_CLASSES = len(EMOTIONS)


class EmotionCNN(nn.Module):
    """
    CNN with Batch Normalization for SER.

    Input:  (batch, 1, 64, 188)
    Output: (batch, 4)
    """

    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()

        self.features = nn.Sequential(

            # Block 1
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 2
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 3
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Compatible with our MPS input dimensions
            nn.AdaptiveAvgPool2d((2, 1))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


if __name__ == "__main__":
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    model = EmotionCNN().to(device)
    model.eval()

    dummy_input = torch.randn(
        8, 1, 64, 188,
        device=device
    )

    with torch.no_grad():
        output = model(dummy_input)

    print("Device:", device)
    print("Output shape:", output.shape)
