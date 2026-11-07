
import torch

from ml.model import EmotionCNN, NUM_CLASSES


def test_model_output_shape():
    model = EmotionCNN()
    model.eval()

    dummy_input = torch.randn(8, 1, 64, 188)

    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == (8, NUM_CLASSES)


def test_model_output_is_finite():
    model = EmotionCNN()
    model.eval()

    dummy_input = torch.randn(2, 1, 64, 188)

    with torch.no_grad():
        output = model(dummy_input)

    assert torch.isfinite(output).all()


def test_model_backward_pass():
    model = EmotionCNN()

    dummy_input = torch.randn(4, 1, 64, 188)
    labels = torch.tensor([0, 1, 2, 3])

    outputs = model(dummy_input)

    loss = torch.nn.CrossEntropyLoss()(outputs, labels)
    loss.backward()

    assert torch.isfinite(loss)
    assert any(
        parameter.grad is not None
        for parameter in model.parameters()
    )
