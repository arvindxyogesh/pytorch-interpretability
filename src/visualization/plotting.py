"""src/visualization/plotting.py — Saliency / CAM visualisation utilities."""
from __future__ import annotations
from typing import List, Optional, Tuple
import numpy as np


def cam_to_heatmap(cam: "torch.Tensor", colormap: str = "jet") -> "np.ndarray":
    """Convert a (H, W) float tensor in [0,1] to an RGB heat-map array.

    Args:
        cam:       2-D pytorch tensor output from GradCAM.generate().
        colormap:  matplotlib colormap name.

    Returns:
        (H, W, 3) uint8 numpy array.
    """
    import matplotlib.pyplot as plt
    import torch

    arr = cam.cpu().numpy() if hasattr(cam, "cpu") else np.array(cam)
    cmap = plt.get_cmap(colormap)
    rgba = cmap(arr)                  # (H, W, 4)
    rgb  = (rgba[:, :, :3] * 255).astype(np.uint8)
    return rgb


def overlay_cam(
    image: "torch.Tensor",
    cam: "torch.Tensor",
    alpha: float = 0.5,
) -> "np.ndarray":
    """Overlay a CAM heat-map on an input image.

    Args:
        image: (C, H, W) float tensor, normalised to [0,1].
        cam:   (H, W) float tensor in [0,1].
        alpha: Blending factor for the heat-map.

    Returns:
        (H, W, 3) blended uint8 numpy array.
    """
    import torch
    img_np = image.cpu().permute(1, 2, 0).numpy()
    img_np = np.clip((img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8), 0, 1)
    img_rgb = (img_np * 255).astype(np.uint8)

    heat = cam_to_heatmap(cam)
    blended = (alpha * heat + (1 - alpha) * img_rgb).astype(np.uint8)
    return blended


def plot_attributions(
    image: "torch.Tensor",
    attributions: "torch.Tensor",
    title: str = "Attributions",
    save_path: Optional[str] = None,
) -> None:
    """Plot input image alongside its attribution map."""
    import matplotlib.pyplot as plt
    import torch

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Original image
    img_np = image.cpu().squeeze().permute(1, 2, 0).numpy()
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
    axes[0].imshow(img_np)
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    # Attribution magnitude
    attr = attributions.cpu().abs().mean(dim=1).squeeze().numpy()
    axes[1].imshow(attr, cmap="hot")
    axes[1].set_title("Attribution Magnitude")
    axes[1].axis("off")

    # Positive attributions only
    pos = attributions.cpu().clamp(min=0).mean(dim=1).squeeze().numpy()
    axes[2].imshow(pos, cmap="Blues")
    axes[2].set_title("Positive Attributions")
    axes[2].axis("off")

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
