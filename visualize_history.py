"""
Visualize training history saved by train_cifar10.py.

Usage:
    python visualize_history.py
    python visualize_history.py --history checkpoints/training_history.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib is required. Install with: pip install matplotlib")
    raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize AlexNet training history")
    parser.add_argument(
        "--history",
        type=str,
        default="checkpoints/training_history.pt",
        help="Path to training_history.pt",
    )
    parser.add_argument(
        "--save",
        type=str,
        default="checkpoints/training_curves.png",
        help="Where to save the plot",
    )
    args = parser.parse_args()

    history_path = Path(args.history)
    if not history_path.exists():
        print(f"History file not found: {history_path}")
        print("Train the model first with: python train_cifar10.py")
        return

    history = torch.load(history_path, weights_only=False)

    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    test_loss = [h["test_loss"] for h in history]
    train_acc = [h["train_acc"] for h in history]
    test_acc = [h["test_acc"] for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss plot
    axes[0].plot(epochs, train_loss, label="Train Loss", marker="o")
    axes[0].plot(epochs, test_loss, label="Test Loss", marker="s")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Curves")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy plot
    axes[1].plot(epochs, train_acc, label="Train Acc", marker="o")
    axes[1].plot(epochs, test_acc, label="Test Acc", marker="s")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Accuracy Curves")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    save_path = Path(args.save)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    print(f"Plot saved to: {save_path}")

    plt.show()


if __name__ == "__main__":
    main()
