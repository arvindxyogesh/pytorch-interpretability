"""src/methods/saliency.py — Vanilla and SmoothGrad saliency maps."""
from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn


class VanillaSaliency:
    """Vanilla gradient saliency: |∂output/∂input|.

    The simplest gradient-based attribution method. Computes the absolute
    gradient of the target class score with respect to the input.
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def attribute(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> torch.Tensor:
        self.model.eval()
        x = input_tensor.clone().requires_grad_(True)
        output = self.model(x)
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        self.model.zero_grad()
        output[0, target_class].backward()
        saliency = x.grad.abs()
        return saliency


class SmoothGrad:
    """SmoothGrad: average saliency over noisy copies of input.

    Reduces noise in gradient maps by averaging saliency across N
    copies of the input with added Gaussian noise.

    Args:
        model:      nn.Module.
        n_samples:  Number of noisy samples to average.
        noise_level: Standard deviation of noise relative to input range.
    """

    def __init__(
        self,
        model: nn.Module,
        n_samples: int = 25,
        noise_level: float = 0.1,
    ) -> None:
        self.model = model
        self.n_samples = n_samples
        self.noise_level = noise_level
        self._vanilla = VanillaSaliency(model)

    def attribute(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            out = self.model(input_tensor)
        if target_class is None:
            target_class = out.argmax(dim=1).item()

        input_range = input_tensor.max() - input_tensor.min()
        sigma = self.noise_level * input_range.item()

        sum_grads = torch.zeros_like(input_tensor)
        for _ in range(self.n_samples):
            noisy = input_tensor + torch.randn_like(input_tensor) * sigma
            sal = self._vanilla.attribute(noisy, target_class=target_class)
            sum_grads += sal

        return sum_grads / self.n_samples
