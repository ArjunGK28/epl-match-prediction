"""Confusion matrices and per-class precision/recall (the paper's draw-prediction story).

Run from the repository root, after src/build_features.py:  python src/analysis.py
Writes results/per_class_metrics.csv and results/confusion_matrices.png.
Uses the same split as run_experiments.py (seed 0), so numbers match results/results.csv.
"""
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiment_utils import LABELS, RESULTS, load_matches, per_class_report, split_feature_sets
from run_experiments import SEED, make_models

NAMES = ["Home win", "Draw", "Home loss"]
# (model, feature set): the best small-feature models versus large-feature models
CASES = [("NN", "F1"), ("SVM (poly5)", "F1"), ("SVM (linear)", "F3"), ("SVM (RBF)", "F3")]


def main():
    RESULTS.mkdir(exist_ok=True)
    rows, y = load_matches()
    feature_sets = split_feature_sets(rows, y, SEED)
    models = make_models()

    report, matrices = [], []
    for model_name, fs in CASES:
        X_tr, y_tr, X_te, y_te = feature_sets[fs]
        pred = models[model_name]().fit(X_tr, y_tr).predict(X_te)
        cm, precision, recall = per_class_report(y_te, pred)
        matrices.append((f"{model_name} on {fs}", cm, (pred == y_te).mean()))
        for k, name in enumerate(NAMES):
            report.append({"model": model_name, "features": fs, "class": name,
                           "precision": f"{precision[k]:.4f}", "recall": f"{recall[k]:.4f}",
                           "support": int(cm[k].sum())})
        print(f"{model_name} on {fs}: accuracy {(pred == y_te).mean():.4f}")
        print(cm)

    with open(RESULTS / "per_class_metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(report[0]))
        writer.writeheader()
        writer.writerows(report)

    fig, axes = plt.subplots(1, len(matrices), figsize=(4.2 * len(matrices), 4))
    for ax, (title, cm, acc) in zip(np.atleast_1d(axes), matrices):
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(3), NAMES, fontsize=8)
        ax.set_yticks(range(3), NAMES, fontsize=8)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"{title}\naccuracy {acc:.2%}", fontsize=9)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(RESULTS / "confusion_matrices.png", dpi=150)
    print("wrote results/per_class_metrics.csv and results/confusion_matrices.png")


if __name__ == "__main__":
    main()
