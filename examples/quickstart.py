"""
Quickstart example for AlexNet.
Run from the project root:
    python examples/quickstart.py
"""

import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from alexnet import AlexNet
from alexnet.utils import summary, count_parameters


def main() -> None:
    print("Loading AlexNet...")
    model = AlexNet(num_classes=1000)
    model.eval()

    print(f"Trainable parameters: {count_parameters(model):,}")
    print()

    # Model summary
    summary(model)

    # Forward pass
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(x)

    print(f"\nInput shape  : {tuple(x.shape)}")
    print(f"Output shape : {tuple(out.shape)}")
    print("\nAlexNet is ready.")


if __name__ == "__main__":
    main()
