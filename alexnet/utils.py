"""Utility functions for AlexNet."""

from typing import Tuple

import torch
from torch import nn


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size_mb(model: nn.Module) -> float:
    """Estimate model size in megabytes (parameters only)."""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / (1024 ** 2)


def summary(model: nn.Module, input_size: Tuple[int, int, int] = (3, 224, 224)) -> None:
    """Print a simple model summary."""
    device = next(model.parameters()).device
    x = torch.randn(1, *input_size).to(device)

    print("=" * 60)
    print(f"{'Layer':<30} {'Output Shape':<25}")
    print("=" * 60)

    def hook(module, input, output):
        class_name = module.__class__.__name__
        output_shape = str(tuple(output.shape))
        print(f"{class_name:<30} {output_shape:<25}")

    handles = []
    for layer in model.children():
        handles.append(layer.register_forward_hook(hook))

    model.eval()
    with torch.no_grad():
        model(x)

    for h in handles:
        h.remove()

    print("=" * 60)
    print(f"Total trainable parameters : {count_parameters(model):,}")
    print(f"Approximate model size     : {get_model_size_mb(model):.2f} MB")
    print("=" * 60)
