"""
AlexNet implementation in PyTorch
=================================

Faithful re-implementation of the architecture described in:

    Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012).
    ImageNet Classification with Deep Convolutional Neural Networks.
    Advances in Neural Information Processing Systems, 25.

Notes
-----
- Input size expected: (N, 3, 224, 224)
- The original paper used Local Response Normalization (LRN).
  Modern practice usually replaces LRN with BatchNorm or omits it.
  This implementation follows the common modern variant without LRN
  for better compatibility and performance.
- torchvision.models.alexnet uses a different channel progression
  (64 → 192 → 384 …). This implementation stays closer to the
  original paper (96 → 256 → 384 …).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class AlexNet(nn.Module):
    """AlexNet convolutional neural network.

    Args:
        num_classes: Number of output classes. Default: 1000 (ImageNet).
        dropout: Dropout probability applied before the fully-connected layers.
                 Default: 0.5 (as in the original paper).
    """

    def __init__(self, num_classes: int = 1000, dropout: float = 0.5) -> None:
        super().__init__()

        # -----------------------------------------------------------------
        # Feature extractor (5 convolutional layers)
        # -----------------------------------------------------------------
        self.features = nn.Sequential(
            # Layer 1: 11×11 conv, stride 4 → 96 filters
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            # Layer 2: 5×5 conv → 256 filters
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            # Layer 3: 3×3 conv → 384 filters
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # Layer 4: 3×3 conv → 384 filters
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # Layer 5: 3×3 conv → 256 filters + pool
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        # Adaptive pooling guarantees a fixed 6×6 spatial size
        # regardless of slight input size variations.
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        # -----------------------------------------------------------------
        # Classifier (3 fully-connected layers)
        # -----------------------------------------------------------------
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the flattened feature vector before the classifier.

        Useful for transfer learning, visualization, or as a fixed backbone.
        Output shape: (N, 256 * 6 * 6) = (N, 9216)
        """
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def freeze_features(self) -> None:
        """Freeze the convolutional feature extractor (for transfer learning)."""
        for param in self.features.parameters():
            param.requires_grad = False

    def unfreeze_features(self) -> None:
        """Unfreeze the convolutional feature extractor."""
        for param in self.features.parameters():
            param.requires_grad = True

    def freeze_classifier(self) -> None:
        """Freeze the fully-connected classifier head."""
        for param in self.classifier.parameters():
            param.requires_grad = False

    def unfreeze_classifier(self) -> None:
        """Unfreeze the fully-connected classifier head."""
        for param in self.classifier.parameters():
            param.requires_grad = True

    def _initialize_weights(self) -> None:
        """Weight initialization following common modern practice."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def alexnet(
    pretrained: bool = False,
    num_classes: int = 1000,
    **kwargs,
) -> AlexNet:
    """Convenience factory for AlexNet.

    Args:
        pretrained: If True, attempts to load torchvision weights.
                    Note that torchvision uses a different channel layout,
                    so a warning is printed and random init is used instead.
        num_classes: Number of output classes.
        **kwargs: Extra arguments passed to the AlexNet constructor.

    Returns:
        An instance of :class:`AlexNet`.
    """
    model = AlexNet(num_classes=num_classes, **kwargs)

    if pretrained:
        try:
            from torchvision.models import alexnet as tv_alexnet, AlexNet_Weights

            _ = tv_alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)
            print(
                "[Warning] torchvision pretrained weights use a different "
                "channel layout (64→192→…).\n"
                "          This implementation stays faithful to the original "
                "paper (96→256→…).\n"
                "          Using randomly initialized weights."
            )
        except Exception as e:
            print(f"[Warning] Could not load pretrained weights: {e}")

    return model
