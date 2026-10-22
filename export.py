"""
Export AlexNet to TorchScript and/or ONNX.

Usage:
    python export.py --checkpoint checkpoints/alexnet_cifar10_best.pth --num-classes 10
    python export.py --format torchscript --output alexnet.pt
    python export.py --format onnx --output alexnet.onnx --num-classes 10
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from alexnet import AlexNet


def load_model(checkpoint: str | None, num_classes: int, device: torch.device) -> AlexNet:
    model = AlexNet(num_classes=num_classes)
    if checkpoint:
        ckpt_path = Path(checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        if isinstance(state, dict) and "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"])
            print(f"Loaded checkpoint: {ckpt_path} (epoch {state.get('epoch', '?')})")
        else:
            model.load_state_dict(state)
            print(f"Loaded checkpoint: {ckpt_path}")
    model.to(device)
    model.eval()
    return model


def export_torchscript(model: AlexNet, output: Path, device: torch.device) -> None:
    dummy = torch.randn(1, 3, 224, 224, device=device)
    traced = torch.jit.trace(model, dummy)
    traced.save(str(output))
    print(f"TorchScript model saved to: {output}")


def export_onnx(model: AlexNet, output: Path, device: torch.device) -> None:
    dummy = torch.randn(1, 3, 224, 224, device=device)
    torch.onnx.export(
        model,
        dummy,
        str(output),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
        opset_version=17,
    )
    print(f"ONNX model saved to: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export AlexNet")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument(
        "--format",
        type=str,
        choices=["torchscript", "onnx", "both"],
        default="both",
    )
    parser.add_argument("--output", type=str, default=None, help="Base output path (extension added automatically)")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    model = load_model(args.checkpoint, args.num_classes, device)

    base = Path(args.output) if args.output else Path("alexnet_export")

    if args.format in ("torchscript", "both"):
        out = base.with_suffix(".pt") if args.output else Path("alexnet.pt")
        export_torchscript(model, out, device)

    if args.format in ("onnx", "both"):
        out = base.with_suffix(".onnx") if args.output else Path("alexnet.onnx")
        export_onnx(model, out, device)

    print("Export finished.")


if __name__ == "__main__":
    main()
