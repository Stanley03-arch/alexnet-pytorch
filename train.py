"""
Basic training script for AlexNet.
This is a minimal example. For real training use a proper dataset (ImageNet, CIFAR, etc.).
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from alexnet import AlexNet


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dummy data for demonstration (replace with real dataset)
    x = torch.randn(64, 3, 224, 224)
    y = torch.randint(0, 10, (64,))
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = AlexNet(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

    model.train()
    for epoch in range(2):
        running_loss = 0.0
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch + 1} | Loss: {running_loss / len(loader):.4f}")

    print("Training finished (demo).")


if __name__ == "__main__":
    main()
