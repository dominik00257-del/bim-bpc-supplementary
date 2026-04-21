# BIM-informed Bayesian post-hoc correction for deep learning-based concrete defect classification: eliminating cross-element misclassifications through element-type priors

## Supplementary Materials

**Paper:** BIM-informed Bayesian post-hoc correction for deep learning-based concrete defect classification: eliminating cross-element misclassifications through element-type priors

---

## Key Results

| Architecture | Params | Baseline CE | BIM-BPC CE | Acc Gain | p-value |
|:---|---:|---:|---:|---:|---:|
| YOLOv8s-cls | 5.1M | 24 | 0 | +0.27% | <0.001 |
| EfficientNet-B0 | 4.0M | 18 | 0 | +0.21% | <0.001 |
| ResNet-18 | 11.2M | 52 | 0 | +0.58% | <0.001 |

CE = cross-element errors, SDNET2018 test set (n=8,420)

## Repository Structure

- `data/` — JSON results from all experiments
- `figures/` — Publication-quality figures (PNG)
- `models/` — Trained model weights (.pt)
- `tables/` — Result tables (CSV)
- `src/` — Core Python modules
- `notebooks/` — Jupyter notebooks for reproduction

## Datasets

- **SDNET2018**: https://digitalcommons.usu.edu/all_datasets/48/
- **CODEBRIM**: https://zenodo.org/record/2620293

## Quick Start

```python
from src.bpc_correction import correct_predictions
import numpy as np

softmax = np.load('your_softmax.npy')
elements = ['Deck', 'Wall', 'Pavement', ...]
corrected = correct_predictions(softmax, elements)
```

## License

MIT
