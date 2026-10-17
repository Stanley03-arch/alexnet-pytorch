"""Unit tests for AlexNet."""

import torch
import pytest

from alexnet import AlexNet, alexnet, count_parameters


def test_forward_shape_default():
    model = AlexNet(num_classes=1000)
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 1000)


def test_forward_shape_custom_classes():
    model = AlexNet(num_classes=10)
    model.eval()
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 10)


def test_parameter_count():
    model = AlexNet(num_classes=1000)
    params = count_parameters(model)
    # Classic AlexNet is around 60M parameters
    assert 50_000_000 < params < 70_000_000


def test_eval_mode():
    model = AlexNet()
    model.eval()
    assert not model.training


def test_train_mode():
    model = AlexNet()
    model.train()
    assert model.training


def test_different_batch_sizes():
    model = AlexNet(num_classes=10)
    model.eval()
    for bs in [1, 4, 8]:
        x = torch.randn(bs, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (bs, 10)


def test_factory_function():
    model = alexnet(num_classes=10)
    assert isinstance(model, AlexNet)
    model.eval()
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 10)


def test_dropout_argument():
    model = AlexNet(num_classes=10, dropout=0.3)
    assert isinstance(model, AlexNet)


def test_output_is_finite():
    model = AlexNet(num_classes=10)
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    assert torch.isfinite(out).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
