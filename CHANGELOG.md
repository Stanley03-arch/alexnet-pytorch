# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.1] - 2026-10-28

### Changed
- Package version bump for consistency across docs and packaging

## [0.3.0] - 2026-10-23

### Added
- **Grad-CAM** explainability module (`alexnet.gradcam`) – pure PyTorch, no extra deps
- Completely redesigned Gradio demo with tabs:
  - Classify + Grad-CAM visual explanation
  - Model Info (params, FLOPs, status)
  - About
- `overlay_cam_on_image` helper for beautiful heatmaps

### Changed
- Major demo upgrade from basic Interface to full Blocks UI
- Bumped package version to 0.3.0

## [0.2.1] - 2026-10-23

### Added
- Optional Local Response Normalization (LRN) support via `use_lrn=True`
  (matches the original 2012 paper after the first two convolutional layers)
- Improved README with clearer differentiation, star CTA, and LRN documentation
- Google Colab quickstart notebook + badge

### Changed
- Bumped package version to 0.2.1

## [0.2.0] - 2026-10-22

### Added
- FLOPs / MACs counter (`count_flops`) and inclusion in model summary
- Feature extraction method (`model.extract_features()`)
- Freeze / unfreeze helpers for transfer learning (`freeze_features`, `unfreeze_features`, etc.)
- Early stopping support in CIFAR-10 training (`--patience`)
- Option to freeze the backbone during training (`--freeze-features`)
- Model export script (`export.py`) for TorchScript and ONNX
- Transfer-learning example (`examples/transfer_learning.py`)
- Unit tests for new utilities and methods

### Fixed
- Gradio demo now correctly loads full training checkpoints (dict with `model_state_dict`)

### Changed
- Bumped package version to 0.2.0

## [0.1.2] - 2026-10-20

### Changed
- Bumped package version to 0.1.2

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
