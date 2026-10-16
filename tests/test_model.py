"""Basic unit tests for AlexNet."""

import torch
import pytest

from alexnet import AlexNet, count_parameters


def test_alexnet_forward_shape():
    model = AlexNet(num_classes=1000)
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 1000)


def test_alexnet_custom_classes():
    model = AlexNet(num_classes=10)
    model.eval()
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 10)


def test_parameter_count():
    model = AlexNet(num_classes=1000)
    params = count_parameters(model)
    # AlexNet typically has ~60M parameters
    assert 50_000_000 < params < 70_000_000


def test_eval_mode():
    model = AlexNet()
    model.eval()
    assert not model.training


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
