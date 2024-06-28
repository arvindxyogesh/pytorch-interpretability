"""examples/explain_image_classifier.py — End-to-end XAI demo."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import logging
import torch
from src.models.simple_cnn import SimpleCNN
from src.methods.gradcam import GradCAM
from src.methods.integrated_gradients import IntegratedGradients
from src.methods.saliency import VanillaSaliency, SmoothGrad

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)-8s %(message)s")
logger = logging.getLogger(__name__)


def main():
    torch.manual_seed(42)
    model = SimpleCNN(n_classes=10)
    model.eval()

    # Synthetic input (batch=1, 3 channels, 32x32)
    img = torch.randn(1, 3, 32, 32)
    logger.info(f"Input shape: {img.shape}")

    # ── Grad-CAM ──────────────────────────────────────────────────────────────
    logger.info("Computing Grad-CAM ...")
    gcam = GradCAM(model, model.last_conv)
    cam, pred_class = gcam.generate(img)
    gcam.remove_hooks()
    logger.info(f"  Predicted class   : {pred_class}")
    logger.info(f"  CAM shape         : {cam.shape}")
    logger.info(f"  CAM range         : [{cam.min():.3f}, {cam.max():.3f}]")

    # ── Integrated Gradients ──────────────────────────────────────────────────
    logger.info("Computing Integrated Gradients ...")
    ig = IntegratedGradients(model, steps=30)
    attrs = ig.attribute(img, target_class=pred_class)
    lo, hi, norm = ig.signed_summary(attrs)
    logger.info(f"  Attributions range: [{lo:.3f}, {hi:.3f}]")
    logger.info(f"  L1 norm           : {norm:.3f}")

    # ── Vanilla Saliency ──────────────────────────────────────────────────────
    logger.info("Computing Vanilla Saliency ...")
    sal = VanillaSaliency(model)
    sal_map = sal.attribute(img, target_class=pred_class)
    logger.info(f"  Saliency max      : {sal_map.max():.4f}")

    # ── SmoothGrad ────────────────────────────────────────────────────────────
    logger.info("Computing SmoothGrad (n=10) ...")
    sg = SmoothGrad(model, n_samples=10, noise_level=0.15)
    sg_map = sg.attribute(img, target_class=pred_class)
    logger.info(f"  SmoothGrad max    : {sg_map.max():.4f}")

    logger.info("All attribution methods completed successfully.")

    # ── Optional: save visualisation if matplotlib is available ───────────────
    try:
        from src.visualization.plotting import plot_attributions
        os.makedirs("outputs", exist_ok=True)
        plot_attributions(img, attrs, title=f"IG  (class={pred_class})",
                          save_path="outputs/ig_attribution.png")
        logger.info("Saved attribution plot to outputs/ig_attribution.png")
    except Exception as e:
        logger.debug(f"Visualisation skipped: {e}")


if __name__ == "__main__":
    main()
