# Shift Robustness of Polyphase Max Pooling on Handwritten Digits

**Motivation.** Standard stride-2 pooling discards spatial information, so a convolutional network's predictions can change when its input is shifted by a pixel or two, even though the digit is unchanged. The paper "Improving shift invariance in convolutional neural networks with translation invariant polyphase sampling" (2025) proposes a translation-invariant polyphase sampling scheme for pooling layers to reduce this sensitivity. This repository tests the pooling idea in a much smaller setting.

**Question.** Does translation-invariant polyphase max pooling keep a classifier's accuracy more stable under small input shifts than standard stride-2 max pooling?

**Dataset.** The bundled scikit-learn `load_digits` set (UCI Optical Recognition of Handwritten Digits), 8x8 images, with a stratified hold-out split and a fixed seed. This is not the paper's own dataset.

**Baseline vs. improved.** Both models use the same pooled features, the same standardized logistic regression with C chosen by cross-validation on the training set, and the same evaluation on shifted test images. The baseline uses standard 2x2 max pooling with stride 2. The improved model uses a dense stride-1 2x2 max pool whose four stride-2 polyphase outputs are combined by an elementwise maximum. The baseline is a stand-in for CNN pooling, not a re-implementation of the paper's method.

**How to run**

```
pip install -r requirements.txt
python analysis.py
```

**Results.** Results are written to `results.json`; a chart is saved to `figures/shift_accuracy.png`.

## Reference

This project is motivated by the following paper:

> S Saha, T Gokhale (2025). *Improving shift invariance in convolutional neural networks with translation invariant polyphase sampling*. 2025 IEEE/CVF Winter Conference on. https://ieeexplore.ieee.org/abstract/document/10943698/

Paper author: Prof. Tejas Gokhale, University of Maryland, Baltimore County (gokhale@umbc.edu)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
