# BIM-informed Bayesian post-hoc correction for deep learning-based concrete defect classification: eliminating cross-element misclassifications through element-type priors

## Supplementary Materials

**Paper:** BIM-informed Bayesian post-hoc correction for deep learning-based concrete defect classification: eliminating cross-element misclassifications through element-type priors

**Journal:** Automation in Construction (Elsevier)

---

## Key Results

| Architecture | Params | Baseline CE | BIM-BPC CE | Acc Gain | p-value |
|:---|---:|---:|---:|---:|---:|
| YOLOv8s-cls | 5.1M | 24 | 0 | +0.27% | <0.001 |
| EfficientNet-B0 | 4.0M | 18 | 0 | +0.21% | <0.001 |
| ResNet-18 | 11.2M | 52 | 0 | +0.58% | <0.001 |

CE = cross-element errors, SDNET2018 test set (n=8,420)

---

## Repository Structure

├── data/               JSON results from all experiments
├── figures/            Publication-quality figures (PNG)
├── models/             Trained model weights (.pt)
├── notebooks/          Jupyter notebooks for reproduction (see below)
├── src/                Core Python modules
├── tables/             Result tables (CSV)
├── requirements.txt    Python dependencies
└── README.md

## Notebooks

| # | File | Manuscript Section | Description |
|---|---|---|---|
| 01 | 01_experiment_A_method_comparison.ipynb | §5.1 | SDNET2018 6-class method comparison |
| 02 | 02_experiment_B_bootstrap_ci.ipynb | §5.2 | Bootstrap confidence intervals (B=2,000) |
| 03 | 03_experiment_C_error_decomposition.ipynb | §5.3 | Cross-element vs intra-element error analysis |
| 04 | 04_experiment_D_prior_sensitivity.ipynb | §5.4 | Prior degradation alpha sweep |
| 05 | 05_experiment_E_cross_architecture.ipynb | §5.5 | Validation across YOLOv8/EfficientNet/ResNet |
| 06 | 06_experiment_F_scalability.ipynb | §5.6 | BIM spatial mapping benchmark (15-2000 elements) |
| 07 | 07_supplementary_codebrim.ipynb | §5.7 | CODEBRIM 5-class (simulated BIM context) |

## Datasets

- SDNET2018: https://digitalcommons.usu.edu/all_datasets/48/
- CODEBRIM: https://zenodo.org/record/2620293

## Quick Start

    from src.bpc_correction import correct_predictions
    import numpy as np
    softmax = np.load('data/sdnet6_test_softmax.npz')['softmax']
    elements = ['Deck', 'Wall', 'Pavement', ...]
    corrected = correct_predictions(softmax, elements)

## Citation

If you use this code, please cite:

    @article{metelski2025bimbpc,
      title={BIM-informed Bayesian post-hoc correction},
      author={Metelski, Dominik and others},
      journal={Automation in Construction},
      year={2025},
      publisher={Elsevier}
    }

## License

MIT
