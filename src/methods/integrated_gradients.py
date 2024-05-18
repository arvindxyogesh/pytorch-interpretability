"""src/methods/integrated_gradients.py — Integrated Gradients (Sundararajan et al., 2017)."""
from __future__ import annotations
from typing import Optional, Tuple

import torch
import torch.nn as nn


class IntegratedGradients:
    """Integrated Gradients attribution method.

    Approximates the path integral of gradients from a baseline (usually zeros
    or black image) to the actual input. Attribution = (input - baseline) *
    mean_gradient_along_path.

    Args:
        model:   nn.Module to explain.
        steps:   Number of interpolation steps (higher → more accurate).

    Usage::

        ig = IntegratedGradients(model, steps=50)
        attributions = ig.attribute(input_tensor, target_class=1)
        # attributions: same shape as input_tensor
    """

    def __init__(self, model: nn.Module, steps: int = 50) -> None:
        self.model = model
        self.steps = steps

    def attribute(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        baseline: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute Integrated Gradients attributions.

        Args:
            input_tensor: Shape (1, C, H, W) or (1, F).
            target_class: Class to explain. Defaults to argmax of output.
            baseline:     Starting point of the path. Defaults to zeros.

        Returns:
            Attributions tensor with the same shape as input_tensor.
        """
        if baseline is None:
            baseline = torch.zeros_like(input_tensor)

        self.model.eval()
        with torch.no_grad():
            out = self.model(input_tensor)
        if target_class is None:
            target_class = out.argmax(dim=1).item()

        # Build interpolated inputs: baseline + alpha * (input - baseline)
        alphas = torch.linspace(0.0, 1.0, self.steps, device=input_tensor.device)
        delta  = input_tensor - baseline

        grads = []
        for alpha in alphas:
            interp = (baseline + alpha * delta).requires_grad_(True)
            output = self.model(interp)
            self.model.zero_grad()
            output[0, target_class].backward()
            grads.append(interp.grad.detach().clone())

        # Average gradients via trapezoidal rule and multiply by delta.
        avg_grads = torch.stack(grads).mean(dim=0)
        attributions = delta * avg_grads
        return attributions

    def signed_summary(self, attributions: torch.Tensor) -> Tuple[float, float, float]:
        """Return (min, max, l1_norm) of attributions."""
        return (
            attributions.min().item(),
            attributions.max().item(),
            attributions.abs().sum().item(),
        )
