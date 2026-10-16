"""Simple inference example for AlexNet."""

import torch
from alexnet import AlexNet


def main() -> None:
    model = AlexNet(num_classes=1000)
    model.eval()

    # Dummy input (batch_size=1, channels=3, height=224, width=224)
    x = torch.randn(1, 3, 224, 224)

    with torch.no_grad():
        output = model(x)

    print(f"Output shape: {output.shape}")
    print(f"Predicted class index: {output.argmax(dim=1).item()}")


if __name__ == "__main__":
    main()
