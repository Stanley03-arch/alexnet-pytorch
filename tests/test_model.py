"""Unit tests for AlexNet."""

import torch
import pytest

from alexnet import AlexNet, alexnet, count_parameters, count_flops


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


def test_extract_features():
    model = AlexNet(num_classes=10)
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        feats = model.extract_features(x)
    assert feats.shape == (2, 256 * 6 * 6)


def test_freeze_unfreeze():
    model = AlexNet(num_classes=10)
    total_before = count_parameters(model)

    model.freeze_features()
    trainable_frozen = count_parameters(model)
    assert trainable_frozen < total_before

    model.unfreeze_features()
    assert count_parameters(model) == total_before


def test_count_flops():
    model = AlexNet(num_classes=10)
    model.eval()
    flops = count_flops(model)
    # AlexNet is typically in the ~1–2 GFLOP range for 224×224
    assert flops > 100_000_000  # at least 0.1 GFLOP


def test_use_lrn():
    """Test that LRN variant constructs and runs correctly."""
    model = AlexNet(num_classes=10, use_lrn=True)
    assert model.use_lrn is True
    model.eval()
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 10)
    assert torch.isfinite(out).all()

    # Factory also accepts the flag
    model2 = alexnet(num_classes=10, use_lrn=True)
    assert model2.use_lrn is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
