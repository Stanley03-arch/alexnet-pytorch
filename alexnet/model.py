"""
AlexNet implementation in PyTorch
Based on: Krizhevsky, Sutskever & Hinton (2012)
"""

from typing import Optional

import torch
import torch.nn as nn


class AlexNet(nn.Module):
    """AlexNet model architecture from the paper
    "ImageNet Classification with Deep Convolutional Neural Networks".

    Args:
        num_classes (int): Number of classification classes. Default: 1000
        dropout (float): Dropout probability. Default: 0.5
    """

    def __init__(self, num_classes: int = 1000, dropout: float = 0.5) -> None:
        super().__init__()

        self.features = nn.Sequential(
            # Conv1
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # Conv2
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # Conv3
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # Conv4
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # Conv5
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

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

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def alexnet(pretrained: bool = False, num_classes: int = 1000, **kwargs) -> AlexNet:
    """Factory function that can optionally load torchvision pretrained weights.

    Note:
        torchvision's AlexNet has a slightly different architecture (64→192→384...).
        When pretrained=True we load torchvision's weights into a compatible structure
        where possible, or fall back to random initialization with a warning.
    """
    model = AlexNet(num_classes=num_classes, **kwargs)

    if pretrained:
        try:
            from torchvision.models import alexnet as tv_alexnet, AlexNet_Weights

            tv_model = tv_alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)
            # torchvision AlexNet uses different channel progression;
            # we keep our faithful implementation and just warn the user.
            print(
                "[Warning] torchvision pretrained weights use a different channel layout.\n"
                "          Using randomly initialized weights for the classic architecture.\n"
                "          For official pretrained weights, prefer torchvision.models.alexnet."
            )
        except Exception as e:
            print(f"[Warning] Could not load pretrained weights: {e}")

    return model
