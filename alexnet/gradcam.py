"""
Lightweight Grad-CAM implementation for AlexNet.
No external dependencies beyond PyTorch + torchvision.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


class GradCAM:
    """Gradient-weighted Class Activation Mapping for AlexNet."""

    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        self.model = model
        self.model.eval()

        # Default to the last convolutional layer in features
        if target_layer is None:
            # Find the last Conv2d in the features Sequential
            for module in reversed(list(model.features.modules())):
                if isinstance(module, nn.Conv2d):
                    target_layer = module
                    break
        self.target_layer = target_layer

        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None

        # Register hooks
        self._forward_handle = self.target_layer.register_forward_hook(self._save_activation)
        self._backward_handle = self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap.

        Args:
            input_tensor: (1, C, H, W) preprocessed image tensor
            target_class: Class index to explain. If None, uses the predicted class.

        Returns:
            heatmap: (H, W) numpy array normalized to [0, 1]
        """
        self.model.zero_grad()
        input_tensor = input_tensor.clone().requires_grad_(True)

        # Forward
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Backward on the target score
        score = output[0, target_class]
        score.backward()

        # Grad-CAM computation
        gradients = self.gradients  # (1, C, h, w)
        activations = self.activations  # (1, C, h, w)

        # Global average pool the gradients → weights
        weights = gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Weighted combination of activation maps
        cam = (weights * activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = F.relu(cam)  # only positive contributions

        # Normalize to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam

    def remove_hooks(self):
        self._forward_handle.remove()
        self._backward_handle.remove()


def overlay_cam_on_image(
    image: Image.Image,
    cam: np.ndarray,
    alpha: float = 0.45,
) -> Image.Image:
    """Overlay a Grad-CAM heatmap on a PIL image."""
    import matplotlib.cm as cm

    # Resize CAM to image size
    cam_resized = np.array(
        Image.fromarray((cam * 255).astype(np.uint8)).resize(image.size, Image.BILINEAR)
    ).astype(np.float32) / 255.0

    # Apply colormap
    heatmap = cm.jet(cam_resized)[:, :, :3]  # RGB
    heatmap = (heatmap * 255).astype(np.uint8)

    # Blend
    img_np = np.array(image.convert("RGB")).astype(np.float32)
    overlay = (1 - alpha) * img_np + alpha * heatmap
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    return Image.fromarray(overlay)
