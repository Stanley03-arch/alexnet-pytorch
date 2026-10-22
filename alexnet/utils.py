"""Utility functions for AlexNet."""

from __future__ import annotations

from typing import Tuple

import torch
from torch import nn


def get_device() -> torch.device:
    """Return the best available device (cuda > cpu)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size_mb(model: nn.Module) -> float:
    """Estimate model size in megabytes (parameters + buffers)."""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / (1024 ** 2)


def count_flops(
    model: nn.Module,
    input_size: Tuple[int, int, int] = (3, 224, 224),
) -> int:
    """Approximate FLOPs for a single forward pass (multiply-adds counted as 2 FLOPs).

    Uses a lightweight hook-based estimator that covers Conv2d and Linear layers.
    """
    device = next(model.parameters()).device
    x = torch.randn(1, *input_size, device=device)

    total_flops = 0

    def conv_hook(module: nn.Conv2d, inputs, output):
        nonlocal total_flops
        # output: (N, C_out, H, W)
        n, c_out, h, w = output.shape
        # kernel ops per position
        kernel_ops = module.kernel_size[0] * module.kernel_size[1] * (module.in_channels // module.groups)
        # FLOPs = 2 * MAC (mul + add) * spatial * channels * batch
        flops = 2 * kernel_ops * c_out * h * w * n
        if module.bias is not None:
            flops += c_out * h * w * n
        total_flops += flops

    def linear_hook(module: nn.Linear, inputs, output):
        nonlocal total_flops
        # output: (N, out_features)
        n = output.shape[0]
        flops = 2 * module.in_features * module.out_features * n
        if module.bias is not None:
            flops += module.out_features * n
        total_flops += flops

    handles = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            handles.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            handles.append(m.register_forward_hook(linear_hook))

    was_training = model.training
    model.eval()
    with torch.no_grad():
        model(x)

    for h in handles:
        h.remove()

    if was_training:
        model.train()

    return total_flops


def summary(
    model: nn.Module,
    input_size: Tuple[int, int, int] = (3, 224, 224),
) -> None:
    """Print a simple layer-by-layer model summary."""
    device = next(model.parameters()).device
    x = torch.randn(1, *input_size).to(device)

    print("=" * 65)
    print(f"{'Layer':<30} {'Output Shape':<25}")
    print("=" * 65)

    def hook(module, input, output):
        class_name = module.__class__.__name__
        if isinstance(output, torch.Tensor):
            output_shape = str(tuple(output.shape))
        else:
            output_shape = str(type(output))
        print(f"{class_name:<30} {output_shape:<25}")

    handles = []
    for layer in model.children():
        handles.append(layer.register_forward_hook(hook))

    model.eval()
    with torch.no_grad():
        model(x)

    for h in handles:
        h.remove()

    print("=" * 65)
    print(f"Total trainable parameters : {count_parameters(model):,}")
    print(f"Approximate model size     : {get_model_size_mb(model):.2f} MB")
    try:
        flops = count_flops(model, input_size)
        print(f"Approx. FLOPs (1 forward)  : {flops / 1e9:.2f} GFLOPs")
    except Exception:
        pass
    print(f"Device                     : {device}")
    print("=" * 65)
