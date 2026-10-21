"""Shift robustness: standard stride-2 max pooling vs. dense stride-1 polyphase max pooling on load_digits."""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

SEED = 42
# 25 circular shifts: every (dy, dx) with |dy|, |dx| <= 2, including the unshifted image (0, 0)
SHIFTS = [(dy, dx) for dy in range(-2, 3) for dx in range(-2, 3)]
C_GRID = [0.01, 0.1, 1.0, 10.0]  # baseline logistic-regression grid, selected by CV on training data only
SVC_GRID = {"svc__C": [0.1, 1.0, 10.0, 100.0], "svc__gamma": ["scale", 0.01, 0.1]}  # CV on training data only


def baseline_pool(X):
    """Standard 2x2 max pooling with stride 2: (n, 8, 8) -> (n, 4, 4)."""
    n = X.shape[0]
    return X.reshape(n, 4, 2, 4, 2).max(axis=(2, 4))


def _dense_phases(X):
    """Dense stride-1 2x2 max pool (edge-padded so output stays 8x8), then the four
    stride-2 polyphase sub-grids, each (n, 4, 4)."""
    Xp = np.pad(X, ((0, 0), (0, 1), (0, 1)), mode="edge")
    dense = np.maximum.reduce([Xp[:, :-1, :-1], Xp[:, 1:, :-1], Xp[:, :-1, 1:], Xp[:, 1:, 1:]])
    return [dense[:, a::2, b::2] for a in (0, 1) for b in (0, 1)]


def polyphase_pool(X):
    """Polyphase max: elementwise maximum of the four stride-2 sub-grids: (n, 8, 8) -> (n, 4, 4)."""
    return np.maximum.reduce(_dense_phases(X))


def polyphase_features(X):
    """Improved features: the polyphase-max map (16 values) plus all four polyphase sub-grids
    kept separately (64 values), giving 80 features per image. Uses only the same images."""
    n = X.shape[0]
    phases = _dense_phases(X)
    maxed = np.maximum.reduce(phases).reshape(n, -1)
    stacked = np.stack(phases, axis=1).reshape(n, -1)
    return np.concatenate([maxed, stacked], axis=1)


def run_model(name, featurize, estimator, grid, Xtr, ytr, Xte, yte):
    """Fit classifier on unshifted train features, tune by 5-fold CV on training set, evaluate on shifted test images."""
    print(f"[{name}] tuning hyperparameters with 5-fold CV on training set ...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    search = GridSearchCV(estimator, grid, cv=cv)
    search.fit(featurize(Xtr), ytr)
    model = search.best_estimator_
    print(f"[{name}] best params = {search.best_params_}  (CV acc = {search.best_score_:.4f})")

    clean_pred = model.predict(featurize(Xte))
    accs, unchanged = [], []
    for dy, dx in SHIFTS:
        Xs = np.roll(Xte, shift=(dy, dx), axis=(1, 2))  # circular shift of each test image
        pred = model.predict(featurize(Xs))
        accs.append(accuracy_score(yte, pred))
        unchanged.append(np.mean(pred == clean_pred))  # agreement with the unshifted prediction
    print(f"[{name}] mean shifted accuracy = {np.mean(accs):.4f}")
    return {
        "score": float(np.mean(accs)),
        "clean_acc": float(accs[SHIFTS.index((0, 0))]),
        "f1": float(f1_score(yte, clean_pred, average="macro")),
        "unchanged": float(np.mean(unchanged)),
        "score_std": float(np.std(accs)),
    }


def main():
    print("Loading load_digits (bundled with scikit-learn, no download) ...")
    data = load_digits()
    X, y = data.images, data.target  # (1797, 8, 8), integer pixel values 0-16
    if len(X) == 0:
        print("No data loaded", file=sys.stderr)
        sys.exit(1)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=SEED)
    print(f"Split: {len(Xtr)} train / {len(Xte)} test (stratified, seed {SEED})")

    base_est = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, random_state=SEED))
    base = run_model("baseline", lambda Z: baseline_pool(Z).reshape(len(Z), -1),
                     base_est, {"logisticregression__C": C_GRID}, Xtr, ytr, Xte, yte)

    impr_est = make_pipeline(StandardScaler(), SVC(kernel="rbf", random_state=SEED))
    impr = run_model("improved", polyphase_features, impr_est, SVC_GRID, Xtr, ytr, Xte, yte)

    vals = [base["score"], impr["score"], base["clean_acc"], impr["clean_acc"]]
    if not np.all(np.isfinite(vals)):
        print("Non-finite metrics; aborting", file=sys.stderr)
        sys.exit(1)

    # Chart: mean shifted accuracy and clean accuracy, baseline vs improved
    import os
    os.makedirs("figures", exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = ["mean over shifts", "clean (unshifted)"]
    w = 0.35
    x = np.arange(2)
    ax.bar(x - w / 2, [base["score"], base["clean_acc"]], w, label="baseline: stride-2 max pool + logreg")
    ax.bar(x + w / 2, [impr["score"], impr["clean_acc"]], w, label="improved: polyphase features + RBF SVM")
    ax.set_xticks(x, labels)
    ax.set_ylabel("test accuracy")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("figures/shift_accuracy.png", dpi=120)
    plt.close(fig)

    results = {
        "question": "Does translation-invariant polyphase max pooling keep a classifier's accuracy more "
                    "stable under small input shifts than standard stride-2 max pooling on the digits dataset?",
        "method": "Baseline: standard stride-2 max-pooled features with standardized logistic regression (C tuned by "
                  "5-fold CV on training data). Improved: polyphase max-pooled features (elementwise max of the four "
                  "stride-2 sub-grids plus the sub-grids themselves, 80 features) with a standardized RBF SVM whose "
                  "C and gamma are tuned by 5-fold CV on training data. Both trained on unshifted train images; test "
                  "accuracy averaged over 25 circular shifts (up to 2 px).",
        "dataset": {
            "name": "Optical Recognition of Handwritten Digits (scikit-learn load_digits)",
            "source_url": "https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits",
            "n_rows": int(len(X)),
            "n_train": int(len(Xtr)),
            "n_test": int(len(Xte)),
            "same_as_paper": False,
            "baseline_is_papers_method": False,
        },
        "metric_name": "mean test accuracy over 25 circular shifts (up to 2 px in each direction)",
        "higher_is_better": True,
        "baseline": "Standard 2x2 max pooling with stride 2 (8x8 -> 4x4) followed by standardized logistic regression.",
        "improved": "Dense stride-1 2x2 max pooling with polyphase sub-grids (elementwise max plus separate sub-grid "
                    "features, 80 values per image) followed by a standardized RBF SVM tuned by CV on training data.",
        "metrics": {
            "baseline_score": base["score"],
            "improved_score": impr["score"],
            "improvement": impr["score"] - base["score"],
            "baseline_score_std": base["score_std"],
            "improved_score_std": impr["score_std"],
            "baseline_clean_accuracy": base["clean_acc"],
            "improved_clean_accuracy": impr["clean_acc"],
            "baseline_f1": base["f1"],
            "improved_f1": impr["f1"],
            "baseline_unchanged_fraction": base["unchanged"],
            "improved_unchanged_fraction": impr["unchanged"],
        },
        "notes": "Single deterministic run with seed 42 (the SVM and logistic regression are deterministic given the "
                 "split), so the score_std values are the spread across the 25 shifts, not across seeds. The baseline "
                 "is a stand-in for CNN pooling; the paper's exact polyphase selection rule was not re-implemented. "
                 "The improved model changes both the pooled features and the classifier (RBF SVM), so any gain "
                 "reflects the combination and not pooling alone; the elementwise max over the four phases also "
                 "enlarges the effective pooling window to about 3x3. Hyperparameters were chosen only by 5-fold CV "
                 "on the training set; the test set was used once for final scores. Only 8x8 images were used, so "
                 "results may not transfer to deep CNNs.",
    }
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Baseline mean shifted acc={base['score']:.4f}  Improved={impr['score']:.4f}  "
          f"Improvement={impr['score'] - base['score']:+.4f}")
    print("Wrote results.json and figures/shift_accuracy.png")


if __name__ == "__main__":
    main()
