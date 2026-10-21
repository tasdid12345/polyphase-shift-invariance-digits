# Shift robustness: polyphase vs. stride-2 max pooling on load_digits

Question: does translation-invariant polyphase max pooling keep a classifier's accuracy more stable under small input shifts than standard stride-2 max pooling?

Setup:
- Data: scikit-learn `load_digits` (bundled, no download), stratified 70/30 split with seed 42 (1257 train / 540 test).
- Evaluation: mean test accuracy over 25 circular shifts (every (dy, dx) with |dy|, |dx| <= 2).
- Baseline: 2x2 stride-2 max pooling (16 features) + standardized logistic regression, C tuned by 5-fold CV on the training set.
- Improved: dense stride-1 polyphase max pooling (elementwise max of the four sub-grids, plus the sub-grids themselves; 80 features) + standardized RBF SVM, C and gamma tuned by 5-fold CV on the training set.

The test set is used only for final scoring.

Run:
```
pip install -r requirements.txt
python analysis.py
```
Outputs `results.json` and `figures/shift_accuracy.png`.

Caveat: the improved model changes both the features and the classifier, so the gain cannot be attributed to pooling alone.

## Reference

This project is motivated by the following paper:

> S Saha, T Gokhale (2025). *Improving shift invariance in convolutional neural networks with translation invariant polyphase sampling*. 2025 IEEE/CVF Winter Conference on. https://ieeexplore.ieee.org/abstract/document/10943698/

Paper author: Prof. Tejas Gokhale, University of Maryland, Baltimore County (gokhale@umbc.edu)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
