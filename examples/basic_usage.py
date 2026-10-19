"""
Basic usage example for alexnet-pytorch.

Demonstrates:
- Creating the model
- Counting parameters
- Running a forward pass
- Switching between train/eval modes
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from alexnet import AlexNet, count_parameters, get_device, summary


def main() -> None:
    device = get_device()
    print(f"Using device: {device}\n")

    # Create model for ImageNet-style classification
    model = AlexNet(num_classes=1000).to(device)
    print(f"Parameters: {count_parameters(model):,}")

    # Show architecture summary
    summary(model)

    # Dummy input
    x = torch.randn(4, 3, 224, 224).to(device)

    # Eval mode
    model.eval()
    with torch.no_grad():
        out = model(x)
    print(f"\nEval mode output shape: {tuple(out.shape)}")

    # Train mode
    model.train()
    out_train = model(x)
    print(f"Train mode output shape: {tuple(out_train.shape)}")

    print("\nDone.")


if __name__ == "__main__":
    main()
