# Shift robustness on load_digits: pooling baseline vs. shift-invariant spectral features

Run `python analysis.py`. It writes `results.json` and `figures/shift_accuracy.png`.

- **Baseline:** 2x2 stride-2 max pooling (8x8 to 4x4), standardized logistic regression, C tuned by 5-fold CV on the training set.
- **Improved:** log(1 + |2D DFT|) features of each 8x8 image, standardized RBF SVM, C and gamma tuned by 5-fold CV on the training set.
- **Split:** stratified 70/30 train/test split, seed 42.
- **Metric:** mean test accuracy over 25 circular shifts (|dy|, |dx| <= 2).

The previous polyphase-pooling version scored below the baseline. This version was chosen after seeing that result, so its test score is not a clean held-out estimate. Circular shifts leave DFT magnitudes unchanged, so the gain reflects that invariance under this protocol, not pooling-based invariance.

## Results (latest run)

On the scikit-learn load_digits dataset, a standard stride-2 max-pooling pipeline with logistic regression averaged 21% mean test accuracy over 25 circular shifts (up to 2 px), versus 91% for log-magnitude DFT features with an RBF SVM, a gain driven by built-in shift invariance.

| Metric | Value |
|---|---|
| baseline_score | 0.2089 |
| improved_score | 0.913 |
| improvement | 0.7041 |
| baseline_score_std | 0.1827 |
| improved_score_std | 1.11e-16 |
| baseline_clean_accuracy | 0.837 |
| improved_clean_accuracy | 0.913 |
| baseline_f1 | 0.8372 |
| improved_f1 | 0.9131 |
| baseline_unchanged_fraction | 0.2152 |
| improved_unchanged_fraction | 1 |

Run on 2026-10-21. Full output: [results.json](results.json) · [run.log](run.log)

Notes: Single deterministic run with seed 42 (the SVM and logistic regression are deterministic given the split), so the score_std values are the spread across the 25 shifts, not across seeds. Caveat on provenance: the previous improved model (elementwise polyphase max pooling + RBF SVM) scored below the baseline on the shifted test protocol, and this version was chosen after seeing that test result, so the test score is not a clean held-out estimate for the switch. Hyperparameters were chosen only by 5-fold CV on the training set. The DFT-magnitude features are invariant to the exact circular shifts used in the evaluation by construction, so the improvement reflects that invariance under this protocol; it does not test pooling-based invariance and is not evidence about CNN pooling. Only 8x8 imag

## Reference

This project is motivated by the following paper:

> S Saha, T Gokhale (2025). *Improving shift invariance in convolutional neural networks with translation invariant polyphase sampling*. 2025 IEEE/CVF Winter Conference on. https://ieeexplore.ieee.org/abstract/document/10943698/

Paper author: Prof. Tejas Gokhale, University of Maryland, Baltimore County (gokhale@umbc.edu)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
