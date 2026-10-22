"""
Visualize the first-layer convolutional filters of AlexNet.

Usage:
    python examples/visualize_filters.py
    python examples/visualize_filters.py --use-lrn
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from alexnet import AlexNet


def visualize_first_layer_filters(model: AlexNet, save_path: str | None = None):
    """Plot the 96 filters of the first convolutional layer."""
    # First conv is always the first module in features
    first_conv = None
    for module in model.features.modules():
        if isinstance(module, nn.Conv2d):
            first_conv = module
            break

    if first_conv is None:
        raise RuntimeError("Could not find first Conv2d layer")

    weights = first_conv.weight.data.cpu()  # (96, 3, 11, 11)

    # Normalize each filter for visualization
    weights = weights - weights.min()
    weights = weights / (weights.max() + 1e-8)

    n_filters = weights.shape[0]
    n_cols = 12
    n_rows = (n_filters + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 12))
    axes = axes.flatten()

    for i in range(n_filters):
        # Convert (3, 11, 11) → (11, 11, 3)
        filt = weights[i].permute(1, 2, 0).numpy()
        axes[i].imshow(filt)
        axes[i].axis("off")
        axes[i].set_title(f"{i}", fontsize=6)

    for i in range(n_filters, len(axes)):
        axes[i].axis("off")

    plt.suptitle("AlexNet – First Layer Filters (96 × 11×11)", fontsize=14)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved filter visualization to {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize AlexNet first-layer filters")
    parser.add_argument("--use-lrn", action="store_true", help="Use LRN variant")
    parser.add_argument("--save", type=str, default="artifacts/alexnet_filters.png")
    args = parser.parse_args()

    model = AlexNet(num_classes=1000, use_lrn=args.use_lrn)
    model.eval()
    visualize_first_layer_filters(model, save_path=args.save)


if __name__ == "__main__":
    main()
