# Contributing to alexnet-pytorch

Thanks for your interest in improving this AlexNet implementation.

## Setup

```bash
git clone https://github.com/Stanley03-arch/alexnet-pytorch.git
cd alexnet-pytorch
pip install -e ".[dev]"
# or: pip install -r requirements.txt pytest
```

## Guidelines

- Keep the architecture **paper-faithful** (96 → 256 → 384 → 384 → 256 channels).
- Prefer small, focused PRs with clear commit messages.
- Update `CHANGELOG.md` and bump `__version__` when behavior or public API changes.
- Run tests before opening a PR:

```bash
pytest tests/ -v
```

## Useful areas

1. Extra unit tests (Grad-CAM, export, transfer helpers)
2. Documentation / Colab notebook polish
3. Training script improvements (logging, metrics)
4. Performance notes or FLOPs edge cases

## Style

- Type hints encouraged
- Match existing formatting (Black-friendly)
- Avoid unnecessary dependencies

By contributing, you agree that your contributions are licensed under the MIT License.
