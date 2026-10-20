"""Common image transforms for AlexNet."""

from __future__ import annotations

from torchvision import transforms

# ImageNet-style normalization (used by original AlexNet)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# CIFAR-10 normalization
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)


def get_train_transforms(dataset: str = "cifar10", image_size: int = 224):
    """Return training transforms with data augmentation."""
    if dataset.lower() == "cifar10":
        mean, std = CIFAR10_MEAN, CIFAR10_STD
    else:
        mean, std = IMAGENET_MEAN, IMAGENET_STD

    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(image_size, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def get_eval_transforms(dataset: str = "cifar10", image_size: int = 224):
    """Return evaluation / inference transforms (no augmentation)."""
    if dataset.lower() == "cifar10":
        mean, std = CIFAR10_MEAN, CIFAR10_STD
    else:
        mean, std = IMAGENET_MEAN, IMAGENET_STD

    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
