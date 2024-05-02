# pytorch-interpretability

Neural network explainability toolkit implemented in PyTorch. Implements **Grad-CAM**, **Integrated Gradients**, **Vanilla Saliency**, and **SmoothGrad** — the four most widely used gradient-based attribution methods, plus visualisation utilities for overlaying attributions on input images.

## Architecture

```
pytorch-interpretability/
├── src/
│   ├── methods/
│   │   ├── gradcam.py              # GradCAM (Selvaraju et al., 2017)
│   │   ├── integrated_gradients.py # IG (Sundararajan et al., 2017)
│   │   └── saliency.py             # VanillaSaliency + SmoothGrad
│   ├── models/
│   │   └── simple_cnn.py           # 3-layer CNN target model
│   └── visualization/
│       └── plotting.py             # Overlay, heatmap, and attribution plots
├── examples/
│   └── explain_image_classifier.py
├── tests/
│   └── test_methods.py
└── config/config.yaml
```

## Quick Start

```bash
pip install -r requirements.txt
python examples/explain_image_classifier.py
pytest tests/ -v
```

## Method Comparison

| Method | Gradient type | Noise | Spatial |
|---|---|---|---|
| **VanillaSaliency** | Input gradient | High | Per-pixel |
| **SmoothGrad** | Averaged noisy gradients | Low | Per-pixel |
| **IntegratedGradients** | Path-integrated gradient | Low | Per-pixel |
| **GradCAM** | Feature-map gradient | Low | Coarse |

## Usage Examples

### Grad-CAM

```python
from src.methods.gradcam import GradCAM
from src.models.simple_cnn import SimpleCNN

model = SimpleCNN(n_classes=10)
gcam  = GradCAM(model, model.last_conv)
cam, predicted_class = gcam.generate(img_tensor)  # cam: (H, W) in [0, 1]
```

### Integrated Gradients

```python
from src.methods.integrated_gradients import IntegratedGradients

ig = IntegratedGradients(model, steps=50)
attrs = ig.attribute(img_tensor, target_class=3)
```

### Saliency Maps

```python
from src.methods.saliency import VanillaSaliency, SmoothGrad

saliency = VanillaSaliency(model)
sal_map  = saliency.attribute(img_tensor)

smooth = SmoothGrad(model, n_samples=25, noise_level=0.1)
smooth_map = smooth.attribute(img_tensor)
```

## Running Tests

```bash
pytest tests/ -v
```
