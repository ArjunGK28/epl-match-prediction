"""Repeat the random 80/20 split over many seeds and report mean and std of test accuracy.

Run from the repository root, after src/build_features.py:
    python src/multi_seed.py            # 10 seeds (a few minutes, the F3 neural network is the slow part)
    python src/multi_seed.py 3          # quick check with 3 seeds
Writes results/multi_seed.csv and results/multi_seed.png.
"""
import csv
import sys
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.exceptions import ConvergenceWarning

from experiment_utils import RESULTS, load_matches, split_feature_sets
from run_experiments import SKIP, make_models


def main():
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    RESULTS.mkdir(exist_ok=True)
    warnings.simplefilter("ignore", ConvergenceWarning)
    rows, y = load_matches()
    scores = {}  # (model, feature set) -> list of test accuracies
    for seed in range(n_seeds):
        feature_sets = split_feature_sets(rows, y, seed)
        for model_name, build in make_models().items():
            for fs, (X_tr, y_tr, X_te, y_te) in feature_sets.items():
                if (model_name, fs) in SKIP:
                    continue
                try:
                    acc = (build().fit(X_tr, y_tr).predict(X_te) == y_te).mean()
                except np.linalg.LinAlgError:
                    continue  # GDA on F3: singular covariance
                scores.setdefault((model_name, fs), []).append(acc)
        print(f"seed {seed} done", flush=True)

    home_win = (y == 1).mean()
    out = []
    for (model_name, fs), accs in scores.items():
        out.append({"model": model_name, "features": fs, "n_seeds": len(accs),
                    "mean_test_acc": f"{np.mean(accs):.4f}", "std_test_acc": f"{np.std(accs, ddof=1):.4f}"})
        print(f"{model_name:17s} {fs}  {np.mean(accs):.4f} +/- {np.std(accs, ddof=1):.4f}")
    with open(RESULTS / "multi_seed.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(out[0]))
        writer.writeheader()
        writer.writerows(out)

    models = list(dict.fromkeys(m for m, _ in scores))
    fig, ax = plt.subplots(figsize=(11, 4.5))
    width = 0.26
    for j, (fs, colour) in enumerate([("F1", "#1f77b4"), ("F2", "#2ca02c"), ("F3", "#d62728")]):
        means = [np.mean(scores[(m, fs)]) if (m, fs) in scores else 0 for m in models]
        stds = [np.std(scores[(m, fs)], ddof=1) if (m, fs) in scores else 0 for m in models]
        ax.bar(np.arange(len(models)) + (j - 1) * width, means, width, yerr=stds, capsize=3, color=colour, label=fs)
    ax.axhline(home_win, color="black", linestyle="--", linewidth=1, label="always home win")
    ax.axhline(0.5889, color="grey", linestyle=":", linewidth=1, label="paper best (58.89%)")
    ax.set_xticks(np.arange(len(models)), models)
    ax.set_ylim(0.4, 0.65)
    ax.set_ylabel("Test accuracy (mean +/- std)")
    ax.set_title(f"Test accuracy over {n_seeds} random splits")
    ax.legend(ncol=5, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(RESULTS / "multi_seed.png", dpi=150)
    print("wrote results/multi_seed.csv and results/multi_seed.png")


if __name__ == "__main__":
    main()
