
import torch
import torch.nn as nn


# Emotion classes (fixed ordering)
EMOTIONS = ["neutral", "happy", "sad", "angry"]

NUM_CLASSES = len(EMOTIONS)


class EmotionCNN(nn.Module):
    """
    CNN for Speech Emotion Recognition.

    Input shape:
        (batch_size, 1, 64, 188)

    Output shape:
        (batch_size, 4)
    """

    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()

        # Feature extraction
        self.features = nn.Sequential(

            # Convolutional Block 1
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # Convolutional Block 2
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # Convolutional Block 3
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # Fixed spatial output size
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # Classification layers
        self.classifier = nn.Sequential(
            nn.Flatten(),

            nn.Linear(64, 128),
            nn.ReLU(),

            nn.Dropout(p=0.3),

            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        """
        Forward pass through the network.
        """
        x = self.features(x)
        x = self.classifier(x)

        return x


if __name__ == "__main__":

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    model = EmotionCNN().to(device)

    # Simulate a batch of 8 spectrograms.
    dummy_input = torch.randn(
        8, 1, 64, 188,
        device=device
    )

    model.eval()

    with torch.no_grad():
        output = model(dummy_input)

    print("Device:", device)
    print("Input shape:", dummy_input.shape)
    print("Output shape:", output.shape)

    print(
        "Trainable parameters:",
        sum(p.numel() for p in model.parameters()
            if p.requires_grad)
    )
