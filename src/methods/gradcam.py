"""src/methods/gradcam.py — Gradient-weighted Class Activation Mapping."""
from __future__ import annotations
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM:
    """Gradient-weighted Class Activation Mapping (Selvaraju et al., 2017).

    Computes a coarse saliency map by back-propagating gradient signals into
    the final convolutional feature maps, then global-average-pooling the
    gradients to produce per-channel weights.

    Args:
        model:      A CNN nn.Module.
        target_layer: The Conv2d layer whose activations and gradients we hook.

    Usage::

        gradcam = GradCAM(model, model.layer4[-1].conv2)
        cam, pred_class = gradcam.generate(img_tensor, target_class=None)
        # cam: (H, W) numpy array in [0, 1]
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self._activations: Optional[torch.Tensor] = None
        self._gradients:   Optional[torch.Tensor] = None
        self._hooks = []

        self._hooks.append(
            target_layer.register_forward_hook(self._save_activation)
        )
        self._hooks.append(
            target_layer.register_full_backward_hook(self._save_gradient)
        )

    def _save_activation(self, module, inp, output) -> None:
        self._activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output) -> None:
        self._gradients = grad_output[0].detach()

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> Tuple[torch.Tensor, int]:
        """Compute Grad-CAM for a single input image.

        Args:
            input_tensor: Shape (1, C, H, W).
            target_class: Class index to explain. Uses predicted class if None.

        Returns:
            (cam, pred_class) where cam is a (H, W) float tensor in [0, 1].
        """
        self.model.eval()
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        self.model.zero_grad()
        output[0, target_class].backward()

        # Global average pool of gradients → per-channel importance weights.
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)   # (1, C, 1, 1)
        cam = (weights * self._activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = F.relu(cam)

        # Upsample to input resolution.
        H, W = input_tensor.shape[2], input_tensor.shape[3]
        cam = F.interpolate(cam, size=(H, W), mode="bilinear", align_corners=False)
        cam = cam.squeeze()

        # Normalise to [0, 1].
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)

        return cam, int(target_class)

    def remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()

    def __del__(self):
        self.remove_hooks()
