"""
Inference script for AlexNet.

Examples:
    # Random tensor (sanity check)
    python inference.py

    # Load checkpoint + random input
    python inference.py --checkpoint checkpoints/alexnet_cifar10_best.pth --num-classes 10

    # Run on a real image
    python inference.py --image path/to/photo.jpg --checkpoint checkpoints/alexnet_cifar10_best.pth --num-classes 10
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from alexnet import AlexNet

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2023, 0.1994, 0.2010),
        ),
    ])


def load_image(path: str | Path, device: torch.device) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    tensor = get_transform()(img).unsqueeze(0).to(device)
    return tensor


def main() -> None:
    parser = argparse.ArgumentParser(description="AlexNet inference")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint (.pth)")
    parser.add_argument("--image", type=str, default=None, help="Path to an image file")
    parser.add_argument("--num-classes", type=int, default=1000, help="Number of classes")
    parser.add_argument("--topk", type=int, default=5, help="Show top-k predictions")
    parser.add_argument("--device", type=str, default=None, help="cuda / cpu (auto if not set)")
    args = parser.parse_args()

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device        : {device}")

    model = AlexNet(num_classes=args.num_classes)

    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        state = torch.load(ckpt_path, map_location=device, weights_only=False)

        # Support both plain state_dict and full checkpoint dict
        if isinstance(state, dict) and "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"])
            print(f"Loaded checkpoint: {ckpt_path} (epoch {state.get('epoch', '?')}, acc {state.get('best_acc', 0):.2f}%)")
        else:
            model.load_state_dict(state)
            print(f"Loaded checkpoint: {ckpt_path}")

    model.to(device)
    model.eval()

    # Prepare input
    if args.image:
        x = load_image(args.image, device)
        print(f"Image         : {args.image}")
    else:
        x = torch.randn(1, 3, 224, 224).to(device)
        print("Image         : random tensor (no --image provided)")

    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)[0]

    topk = min(args.topk, args.num_classes)
    values, indices = torch.topk(probs, k=topk)

    print(f"\nTop-{topk} predictions:")
    print("-" * 40)

    for rank, (prob, idx) in enumerate(zip(values, indices), start=1):
        class_id = idx.item()
        if args.num_classes == 10 and class_id < len(CIFAR10_CLASSES):
            name = CIFAR10_CLASSES[class_id]
        else:
            name = f"class_{class_id}"
        print(f"{rank}. {name:<15} {prob.item()*100:6.2f}%")


if __name__ == "__main__":
    main()
