# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1] - 2026-10-19

### Added
- Training history visualization script (`visualize_history.py`)
- Progress bars in CIFAR-10 training (`tqdm`)
- Support for real image inference (`--image` flag)
- Top-k predictions with class names
- Expanded unit tests
- Proper package installation via `pyproject.toml`
- Gradio web demo improvements

### Changed
- Improved model documentation and architecture comments
- Better checkpoint loading (supports full training checkpoints)
- Stronger data augmentation in training
- Updated README with full usage examples

## [0.1.0] - 2026-10-16

### Added
- Initial AlexNet implementation (faithful to original paper)
- CIFAR-10 training script
- Basic inference script
- Gradio demo
- Unit tests
- Utility helpers (parameter count, model summary)
- MIT License
