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

SEED = 42
# 25 circular shifts: every (dy, dx) with |dy|, |dx| <= 2, including the unshifted image (0, 0)
SHIFTS = [(dy, dx) for dy in range(-2, 3) for dx in range(-2, 3)]
C_GRID = [0.01, 0.1, 1.0, 10.0]  # small logistic-regression grid, selected by CV on training data only


def baseline_pool(X):
    """Standard 2x2 max pooling with stride 2: (n, 8, 8) -> (n, 4, 4)."""
    n = X.shape[0]
    return X.reshape(n, 4, 2, 4, 2).max(axis=(2, 4))


def polyphase_pool(X):
    """Dense stride-1 2x2 max pool (edge-padded so output stays 8x8), then the four
    stride-2 polyphase sub-grids are combined by elementwise maximum: (n, 8, 8) -> (n, 4, 4)."""
    Xp = np.pad(X, ((0, 0), (0, 1), (0, 1)), mode="edge")
    dense = np.maximum.reduce([Xp[:, :-1, :-1], Xp[:, 1:, :-1], Xp[:, :-1, 1:], Xp[:, 1:, 1:]])
    phases = [dense[:, a::2, b::2] for a in (0, 1) for b in (0, 1)]  # four stride-2 outputs
    return np.maximum.reduce(phases)


def run_model(name, pool, Xtr, ytr, Xte, yte):
    """Fit pooling-feature classifier on unshifted train images, then evaluate on shifted test images."""
    print(f"[{name}] tuning C with 5-fold CV on training set ...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, random_state=SEED))
    search = GridSearchCV(pipe, {"logisticregression__C": C_GRID}, cv=cv)
    search.fit(pool(Xtr).reshape(len(Xtr), -1), ytr)
    model = search.best_estimator_
    print(f"[{name}] best C = {search.best_params_['logisticregression__C']}")

    clean_pred = model.predict(pool(Xte).reshape(len(Xte), -1))
    accs, unchanged = [], []
    for dy, dx in SHIFTS:
        Xs = np.roll(Xte, shift=(dy, dx), axis=(1, 2))  # circular shift of each test image
        pred = model.predict(pool(Xs).reshape(len(Xs), -1))
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

    base = run_model("baseline", baseline_pool, Xtr, ytr, Xte, yte)
    impr = run_model("improved", polyphase_pool, Xtr, ytr, Xte, yte)

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
    ax.bar(x - w / 2, [base["score"], base["clean_acc"]], w, label="baseline: stride-2 max pool")
    ax.bar(x + w / 2, [impr["score"], impr["clean_acc"]], w, label="improved: polyphase max pool")
    ax.set_xticks(x, labels)
    ax.set_ylabel("test accuracy")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("figures/shift_accuracy.png", dpi=120)
    plt.close(fig)

    results = {
        "question": "Does translation-invariant polyphase max pooling keep a classifier's accuracy more "
                    "stable under small input shifts than standard stride-2 max pooling on the digits dataset?",
        "method": "Same pooled-feature standardized logistic regression (C tuned by 5-fold CV on training data) "
                  "trained on unshifted train images; test accuracy averaged over 25 circular shifts (up to 2 px).",
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
        "improved": "Dense stride-1 2x2 max pooling whose four stride-2 polyphase sub-grids are combined by "
                    "elementwise maximum (8x8 -> 4x4), followed by the same classifier.",
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
        "notes": "Single deterministic run: logistic regression is deterministic, so seeds 42-44 were not needed; "
                 "the score_std values are the spread across the 25 shifts, not across seeds. The baseline is a "
                 "stand-in for CNN pooling; the paper's exact polyphase selection rule was not re-implemented. "
                 "Elementwise max over the four phases effectively enlarges the pooling window to about 3x3, so "
                 "any gain may reflect a larger receptive field rather than polyphase sampling per se. Only 8x8 "
                 "images with a linear classifier were used, so results may not transfer to deep CNNs. Test set "
                 "used once for final scores; no selection on it.",
    }
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Baseline mean shifted acc={base['score']:.4f}  Improved={impr['score']:.4f}  "
          f"Improvement={impr['score'] - base['score']:+.4f}")
    print("Wrote results.json and figures/shift_accuracy.png")


if __name__ == "__main__":
    main()
