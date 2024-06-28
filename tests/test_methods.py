"""tests/test_methods.py — Unit tests for interpretability methods."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pytest
import torch
import torch.nn as nn
from src.methods.gradcam import GradCAM
from src.methods.integrated_gradients import IntegratedGradients
from src.methods.saliency import VanillaSaliency, SmoothGrad
from src.models.simple_cnn import SimpleCNN


@pytest.fixture
def cnn():
    model = SimpleCNN(n_classes=5)
    model.eval()
    return model


@pytest.fixture
def dummy_image():
    torch.manual_seed(0)
    return torch.randn(1, 3, 32, 32)


def test_gradcam_output_shape(cnn, dummy_image):
    gcam = GradCAM(cnn, cnn.last_conv)
    cam, pred_class = gcam.generate(dummy_image)
    assert cam.shape == (32, 32), f"Expected (32, 32), got {cam.shape}"
    assert 0 <= pred_class < 5
    gcam.remove_hooks()


def test_gradcam_normalised(cnn, dummy_image):
    gcam = GradCAM(cnn, cnn.last_conv)
    cam, _ = gcam.generate(dummy_image)
    assert cam.min().item() >= -1e-6
    assert cam.max().item() <= 1.0 + 1e-6
    gcam.remove_hooks()


def test_gradcam_target_class(cnn, dummy_image):
    gcam = GradCAM(cnn, cnn.last_conv)
    cam, pred = gcam.generate(dummy_image, target_class=2)
    assert pred == 2
    gcam.remove_hooks()


def test_integrated_gradients_shape(cnn, dummy_image):
    ig = IntegratedGradients(cnn, steps=5)
    attrs = ig.attribute(dummy_image)
    assert attrs.shape == dummy_image.shape


def test_integrated_gradients_baseline(cnn, dummy_image):
    ig = IntegratedGradients(cnn, steps=5)
    baseline = torch.zeros_like(dummy_image)
    attrs = ig.attribute(dummy_image, baseline=baseline)
    assert attrs.shape == dummy_image.shape


def test_vanilla_saliency_shape(cnn, dummy_image):
    sal = VanillaSaliency(cnn)
    attrs = sal.attribute(dummy_image)
    assert attrs.shape == dummy_image.shape


def test_vanilla_saliency_nonnegative(cnn, dummy_image):
    """Vanilla saliency returns absolute gradients — should all be ≥ 0."""
    sal = VanillaSaliency(cnn)
    attrs = sal.attribute(dummy_image)
    assert attrs.min().item() >= 0.0


def test_smoothgrad_shape(cnn, dummy_image):
    sg = SmoothGrad(cnn, n_samples=3, noise_level=0.1)
    attrs = sg.attribute(dummy_image)
    assert attrs.shape == dummy_image.shape


def test_signed_summary(cnn, dummy_image):
    ig = IntegratedGradients(cnn, steps=5)
    attrs = ig.attribute(dummy_image)
    lo, hi, norm = ig.signed_summary(attrs)
    assert lo <= hi
    assert norm >= 0.0
