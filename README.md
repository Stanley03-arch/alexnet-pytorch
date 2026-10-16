# AlexNet (PyTorch)

A clean, modern, and well-documented implementation of **AlexNet** in PyTorch.

> Original paper: [ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html) (Krizhevsky, Sutskever & Hinton, 2012)

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---

## Overview

AlexNet was the breakthrough model that won the 2012 ImageNet competition and helped launch the modern deep learning era. This repository provides a faithful yet clean re-implementation suitable for learning, experimentation, and transfer learning.

### Key Features

- Faithful architecture to the original paper
- Modern PyTorch practices (type hints, clean code)
- Easy to use for both training and inference
- Utility helpers (parameter count, model summary)
- Ready for transfer learning / fine-tuning
- CIFAR-10 training script included
- Gradio web demo included
- Basic unit tests included
- Well-documented and structured

---

## Architecture

```
Input (3 × 224 × 224)
    ↓
Conv1 (96, 11×11, stride 4) → ReLU → MaxPool
    ↓
Conv2 (256, 5×5) → ReLU → MaxPool
    ↓
Conv3 (384, 3×3) → ReLU
    ↓
Conv4 (384, 3×3) → ReLU
    ↓
Conv5 (256, 3×3) → ReLU → MaxPool
    ↓
Flatten
    ↓
FC1 (4096) → ReLU → Dropout
    ↓
FC2 (4096) → ReLU → Dropout
    ↓
FC3 (1000)
```

---

## Installation

```bash
git clone https://github.com/Stanley03-arch/alexnet-pytorch.git
cd alexnet-pytorch
pip install -r requirements.txt
```

---

## Quick Start

### Load the model

```python
from alexnet import AlexNet
import torch

model = AlexNet(num_classes=1000)
model.eval()

x = torch.randn(1, 3, 224, 224)
output = model(x)
print(output.shape)  # torch.Size([1, 1000])
```

### Model summary

```python
from alexnet import AlexNet, summary, count_parameters

model = AlexNet()
print(f"Parameters: {count_parameters(model):,}")
summary(model)
```

### Run the example

```bash
python examples/quickstart.py
```

---

## Training on CIFAR-10

```bash
python train_cifar10.py --epochs 10 --batch-size 128 --lr 0.01
```

The best model will be saved to `checkpoints/alexnet_cifar10_best.pth`.

---

## Web Demo (Gradio)

```bash
python app.py
```

Open the local URL shown in the terminal to upload images and see predictions.

---

## Testing

```bash
pytest tests/ -v
```

---

## Project Structure

```
alexnet-pytorch/
├── alexnet/
│   ├── __init__.py
│   ├── model.py          # AlexNet implementation
│   └── utils.py          # Helpers (summary, param count)
├── examples/
│   └── quickstart.py
├── tests/
│   └── test_model.py
├── train.py
├── train_cifar10.py
├── inference.py
├── app.py                # Gradio demo
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

---

## Citation

```bibtex
@inproceedings{krizhevsky2012imagenet,
  title={ImageNet classification with deep convolutional neural networks},
  author={Krizhevsky, Alex and Sutskever, Ilya and Hinton, Geoffrey E},
  booktitle={Advances in neural information processing systems},
  pages={1097--1105},
  year={2012}
}
```

---

## License

This project is licensed under the MIT License.

---

**Built with ❤️ by [Stanley Samwel](https://github.com/Stanley03-arch)**
