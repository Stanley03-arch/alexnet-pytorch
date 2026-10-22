"""
Transfer-learning example for AlexNet.

Demonstrates:
  - Freezing the convolutional backbone
  - Replacing / training only the classifier head
  - Extracting intermediate features

Run from the project root:
    python examples/transfer_learning.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
from alexnet import AlexNet, count_parameters


def main() -> None:
    print("=" * 60)
    print("AlexNet Transfer-Learning Example")
    print("=" * 60)

    # 1. Create a model originally for ImageNet (1000 classes)
    model = AlexNet(num_classes=1000)
    print(f"\nOriginal parameters : {count_parameters(model):,}")

    # 2. Freeze the feature extractor
    model.freeze_features()
    print(f"After freeze_features : {count_parameters(model):,} trainable")

    # 3. Replace the classifier for a new task (e.g. 10 classes)
    in_features = 256 * 6 * 6  # 9216
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features, 4096),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5),
        nn.Linear(4096, 10),  # new head
    )
    print(f"After new classifier  : {count_parameters(model):,} trainable")

    # 4. Feature extraction (backbone only)
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        features = model.extract_features(x)
        logits = model(x)

    print(f"\nFeature vector shape  : {tuple(features.shape)}  (N, 9216)")
    print(f"Logits shape          : {tuple(logits.shape)}   (N, 10)")

    # 5. Optionally unfreeze for fine-tuning
    model.unfreeze_features()
    print(f"\nAfter unfreeze_features: {count_parameters(model):,} trainable")

    print("\nTransfer learning helpers are ready to use.")
    print("=" * 60)


if __name__ == "__main__":
    main()
