# AlexNet Colab Quickstart

Copy-paste these cells into a new [Google Colab](https://colab.research.google.com/) notebook.

## 1. Install

```python
!pip install -q torch torchvision
!pip install -q git+https://github.com/Stanley03-arch/alexnet-pytorch.git
```

## 2. Load the model

```python
from alexnet import AlexNet, summary, count_parameters, count_flops
import torch

# Modern variant (default)
model = AlexNet(num_classes=1000)
model.eval()

# Paper-faithful variant with Local Response Normalization
model_lrn = AlexNet(num_classes=1000, use_lrn=True)

print(f"Parameters: {count_parameters(model):,}")
print(f"FLOPs: {count_flops(model) / 1e9:.2f} GFLOPs")
summary(model)
```

## 3. Forward pass

```python
x = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    out = model(x)
print("Output shape:", out.shape)  # torch.Size([1, 1000])
```

## 4. Feature extraction (transfer learning)

```python
model.freeze_features()
features = model.extract_features(x)
print("Feature shape:", features.shape)  # (1, 9216)
```

## 5. (Optional) Quick CIFAR-10 training snippet

```python
# Uncomment and run if you want a short training demo
# !python -c "
# from alexnet import AlexNet
# import torch, torch.nn as nn, torch.optim as optim
# from torchvision import datasets, transforms
# from torch.utils.data import DataLoader
# 
# transform = transforms.Compose([
#     transforms.Resize(224),
#     transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
# ])
# trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
# loader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)
# 
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = AlexNet(num_classes=10).to(device)
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
# 
# model.train()
# for i, (inputs, targets) in enumerate(loader):
#     if i >= 5: break  # just a few batches for demo
#     inputs, targets = inputs.to(device), targets.to(device)
#     optimizer.zero_grad()
#     loss = criterion(model(inputs), targets)
#     loss.backward()
#     optimizer.step()
#     print(f'Batch {i}: loss={loss.item():.4f}')
# print('Demo training finished')
# "
```

That's it! You now have a working paper-faithful AlexNet in Colab.
