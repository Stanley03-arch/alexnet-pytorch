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
from alexnet import AlexNet, summary, count_parameters


def main() -> None:
    print("=" * 60)
    print("AlexNet Quickstart")
    print("=" * 60)

    print("\n1. Creating model (1000 classes)...")
    model = AlexNet(num_classes=1000)
    model.eval()

    params = count_parameters(model)
    print(f"   Trainable parameters : {params:,}")

    print("\n2. Model summary:")
    summary(model)

    print("\n3. Running forward pass...")
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(x)

    print(f"   Input shape  : {tuple(x.shape)}")
    print(f"   Output shape : {tuple(out.shape)}")
    print(f"   Output sample (first 5 logits): {out[0, :5].tolist()}")

    print("\nAlexNet is ready to use.")
    print("=" * 60)


if __name__ == "__main__":
    main()
