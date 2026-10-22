# AlexNet (PyTorch)

**Clean, modern, and paper-faithful implementation of AlexNet in PyTorch.**

> Original paper: [ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html) (Krizhevsky, Sutskever & Hinton, 2012)

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![GitHub stars](https://img.shields.io/github/stars/Stanley03-arch/alexnet-pytorch?style=social)](https://github.com/Stanley03-arch/alexnet-pytorch)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Stanley03-arch/alexnet-pytorch/blob/main/examples/colab_quickstart.ipynb)

⭐ **If this implementation helps you, please consider starring the repository!** It helps others discover it.

---

## Why this implementation?

Most public AlexNet re-implementations (including `torchvision`) use a modified channel progression (64 → 192 → …).  
This repo stays **closer to the original 2012 paper**:

- Channels: **96 → 256 → 384 → 384 → 256**
- Optional **Local Response Normalization (LRN)** exactly as described in the paper
- Clean modern PyTorch (type hints, no legacy code)
- Ready for transfer learning, export, and experimentation

---

## Key Features

- Faithful architecture to the original paper (96 → 256 → 384 channels)
- Optional Local Response Normalization (`use_lrn=True`) for paper fidelity
- Modern PyTorch practices (type hints, clean code)
- Easy to use for both training and inference
- Utility helpers (parameter count, model summary, **FLOPs**)
- **Feature extraction** and **freeze/unfreeze** helpers for transfer learning
- CIFAR-10 training script with progress bars and **early stopping**
- Training history visualization
- Gradio web demo
- **Export to TorchScript / ONNX**
- Unit tests
- Installable as a proper Python package
- **Google Colab quickstart**

---

## Architecture

```
Input (3 × 224 × 224)
    ↓
Conv1 (96, 11×11, stride 4) → ReLU → [LRN] → MaxPool
    ↓
Conv2 (256, 5×5) → ReLU → [LRN] → MaxPool
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

`[LRN]` is enabled only when `use_lrn=True`.

---

## Installation

### Option 1 — Clone and install locally

```bash
git clone https://github.com/Stanley03-arch/alexnet-pytorch.git
cd alexnet-pytorch
pip install -e .
```

### Option 2 — Install dependencies only

```bash
pip install -r requirements.txt
```

### Option 3 — Google Colab (zero setup)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Stanley03-arch/alexnet-pytorch/blob/main/examples/colab_quickstart.ipynb)

---

## Quick Start

### Load the model

```python
from alexnet import AlexNet
import torch

# Modern variant (default)
model = AlexNet(num_classes=1000)
model.eval()

# Paper-faithful variant with Local Response Normalization
model_lrn = AlexNet(num_classes=1000, use_lrn=True)

x = torch.randn(1, 3, 224, 224)
output = model(x)
print(output.shape)  # torch.Size([1, 1000])
```

### Model summary (includes FLOPs)

```python
from alexnet import AlexNet, summary, count_parameters, count_flops

model = AlexNet()
print(f"Parameters: {count_parameters(model):,}")
print(f"FLOPs: {count_flops(model) / 1e9:.2f} GFLOPs")
summary(model)
```

### Feature extraction & transfer learning

```python
model = AlexNet(num_classes=1000)
model.freeze_features()          # freeze conv backbone
features = model.extract_features(x)  # (N, 9216)
```

See `examples/transfer_learning.py` for a full example.

### Run the example

```bash
python examples/quickstart.py
python examples/transfer_learning.py
```

---

## Training on CIFAR-10

```bash
python train_cifar10.py --epochs 10 --batch-size 128 --lr 0.01

# With early stopping and frozen backbone
python train_cifar10.py --epochs 30 --patience 5 --freeze-features
```

The best model will be saved to `checkpoints/alexnet_cifar10_best.pth`.

### Visualize training curves

```bash
python visualize_history.py
```

---

## Inference

```bash
# On a real image
python inference.py --image path/to/photo.jpg --checkpoint checkpoints/alexnet_cifar10_best.pth --num-classes 10

# Random tensor (sanity check)
python inference.py
```

---

## Export (TorchScript / ONNX)

```bash
python export.py --checkpoint checkpoints/alexnet_cifar10_best.pth --num-classes 10
# or
python export.py --format onnx --output alexnet.onnx --num-classes 10
```

---

## Web Demo (Gradio)

```bash
python app.py
```

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
│   ├── model.py
│   ├── utils.py
│   └── transforms.py
├── examples/
│   ├── quickstart.py
│   ├── transfer_learning.py
│   └── colab_quickstart.ipynb
├── tests/
│   └── test_model.py
├── train.py
├── train_cifar10.py
├── inference.py
├── export.py
├── visualize_history.py
├── app.py
├── pyproject.toml
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

⭐ **Star the repo if you found this useful — it really helps!**
