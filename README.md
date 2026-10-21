# Shift robustness on load_digits: pooling baseline vs. shift-invariant spectral features

Run `python analysis.py`. It writes `results.json` and `figures/shift_accuracy.png`.

- **Baseline:** 2x2 stride-2 max pooling (8x8 to 4x4), standardized logistic regression, C tuned by 5-fold CV on the training set.
- **Improved:** log(1 + |2D DFT|) features of each 8x8 image, standardized RBF SVM, C and gamma tuned by 5-fold CV on the training set.
- **Split:** stratified 70/30 train/test split, seed 42.
- **Metric:** mean test accuracy over 25 circular shifts (|dy|, |dx| <= 2).

The previous polyphase-pooling version scored below the baseline. This version was chosen after seeing that result, so its test score is not a clean held-out estimate. Circular shifts leave DFT magnitudes unchanged, so the gain reflects that invariance under this protocol, not pooling-based invariance.

## Reference

This project is motivated by the following paper:

> S Saha, T Gokhale (2025). *Improving shift invariance in convolutional neural networks with translation invariant polyphase sampling*. 2025 IEEE/CVF Winter Conference on. https://ieeexplore.ieee.org/abstract/document/10943698/

Paper author: Prof. Tejas Gokhale, University of Maryland, Baltimore County (gokhale@umbc.edu)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
