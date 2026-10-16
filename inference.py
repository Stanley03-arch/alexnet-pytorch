"""
Inference example for AlexNet.

Usage:
    # Random input (no checkpoint)
    python inference.py

    # Load a trained checkpoint
    python inference.py --checkpoint checkpoints/alexnet_cifar10_best.pth --num-classes 10
"""

import argparse
from pathlib import Path

import torch
from alexnet import AlexNet


def main() -> None:
    parser = argparse.ArgumentParser(description="AlexNet inference")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint (.pth)")
    parser.add_argument("--num-classes", type=int, default=1000, help="Number of classes")
    parser.add_argument("--device", type=str, default=None, help="cuda / cpu (auto if not set)")
    args = parser.parse_args()

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    model = AlexNet(num_classes=args.num_classes)

    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state)
        print(f"Loaded checkpoint: {ckpt_path}")

    model.to(device)
    model.eval()

    # Dummy input
    x = torch.randn(1, 3, 224, 224).to(device)

    with torch.no_grad():
        output = model(x)
        probs = torch.softmax(output, dim=1)
        pred = output.argmax(dim=1).item()
        confidence = probs[0, pred].item()

    print(f"Output shape   : {tuple(output.shape)}")
    print(f"Predicted class: {pred}")
    print(f"Confidence     : {confidence:.4f}")


if __name__ == "__main__":
    main()
