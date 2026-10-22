"""
Train AlexNet on CIFAR-10.

Usage:
    python train_cifar10.py
    python train_cifar10.py --epochs 20 --batch-size 128 --lr 0.01
    python train_cifar10.py --epochs 30 --patience 5   # early stopping
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from alexnet import AlexNet
from alexnet.utils import count_parameters


def get_dataloaders(batch_size: int = 128, num_workers: int = 2):
    """CIFAR-10 dataloaders with standard transforms."""
    transform_train = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(224, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_set = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train)
    test_set = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, test_loader


def train_one_epoch(model, loader, criterion, optimizer, device, epoch: int, total_epochs: int):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch:02d}/{total_epochs} [Train]", leave=False)
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        pbar.set_postfix({
            "loss": f"{running_loss / (pbar.n + 1):.4f}",
            "acc": f"{100.0 * correct / total:.2f}%"
        })

    return running_loss / len(loader), 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device, epoch: int, total_epochs: int):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch:02d}/{total_epochs} [Test ]", leave=False)
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        pbar.set_postfix({
            "loss": f"{running_loss / (pbar.n + 1):.4f}",
            "acc": f"{100.0 * correct / total:.2f}%"
        })

    return running_loss / len(loader), 100.0 * correct / total


def main():
    parser = argparse.ArgumentParser(description="Train AlexNet on CIFAR-10")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument(
        "--patience",
        type=int,
        default=0,
        help="Early stopping patience (0 = disabled). Stop if test acc does not improve for N epochs.",
    )
    parser.add_argument(
        "--freeze-features",
        action="store_true",
        help="Freeze the convolutional backbone (train only the classifier).",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device          : {device}")
    print(f"Epochs          : {args.epochs}")
    print(f"Batch size      : {args.batch_size}")
    print(f"Learning rate   : {args.lr}")
    print(f"Early stopping  : {'patience=' + str(args.patience) if args.patience > 0 else 'disabled'}")
    print(f"Freeze features : {args.freeze_features}")

    train_loader, test_loader = get_dataloaders(args.batch_size, args.num_workers)

    model = AlexNet(num_classes=10).to(device)

    if args.freeze_features:
        model.freeze_features()
        print("Feature extractor frozen — only classifier will be trained.")

    print(f"Parameters      : {count_parameters(model):,} (trainable)")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    best_acc = 0.0
    epochs_without_improvement = 0
    history = []

    print("\nStarting training...\n")
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, args.epochs
        )
        test_loss, test_acc = evaluate(
            model, test_loader, criterion, device, epoch, args.epochs
        )
        scheduler.step()

        current_lr = scheduler.get_last_lr()[0]
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "lr": current_lr,
        })

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:5.2f}% | "
            f"Test Loss: {test_loss:.4f} Acc: {test_acc:5.2f}% | "
            f"LR: {current_lr:.6f}"
        )

        if test_acc > best_acc:
            best_acc = test_acc
            epochs_without_improvement = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_acc": best_acc,
            }, save_dir / "alexnet_cifar10_best.pth")
            print(f"  → Saved best model (acc: {best_acc:.2f}%)")
        else:
            epochs_without_improvement += 1
            if args.patience > 0 and epochs_without_improvement >= args.patience:
                print(
                    f"\nEarly stopping triggered after {epoch} epochs "
                    f"(no improvement for {args.patience} epochs)."
                )
                break

    total_time = time.time() - start_time
    print(f"\nTraining finished in {total_time / 60:.1f} minutes.")
    print(f"Best test accuracy: {best_acc:.2f}%")

    # Save simple history
    history_path = save_dir / "training_history.pt"
    torch.save(history, history_path)
    print(f"Training history saved to {history_path}")


if __name__ == "__main__":
    main()
