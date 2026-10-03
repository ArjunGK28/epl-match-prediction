"""Tune the degree-5 polynomial SVM (the paper's best model, our weakest) with cross-validation.

Run from the repository root, after src/build_features.py:  python src/tune_svm.py
The grid search sees only the training set (5-fold CV); the test set is used once at the end.
Writes results/tune_svm.csv.
"""
import csv
import warnings

import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from experiment_utils import RESULTS, load_matches, per_class_report, split_feature_sets
from run_experiments import SEED

GRID = {
    "svc__degree": [2, 3, 5],
    "svc__C": [0.1, 1, 10],
    "svc__gamma": ["scale", 0.1],
    "svc__coef0": [0, 1],  # the untuned baseline uses coef0 = 0
}


def main():
    RESULTS.mkdir(exist_ok=True)
    warnings.simplefilter("ignore")
    rows, y = load_matches()
    feature_sets = split_feature_sets(rows, y, SEED)
    out = []
    for fs in ("F1", "F2"):
        X_tr, y_tr, X_te, y_te = feature_sets[fs]
        pipe = make_pipeline(StandardScaler(), SVC(kernel="poly", cache_size=500))

        baseline = make_pipeline(StandardScaler(), SVC(kernel="poly", degree=5)).fit(X_tr, y_tr)
        base_acc = (baseline.predict(X_te) == y_te).mean()

        search = GridSearchCV(pipe, GRID, cv=5, n_jobs=-1).fit(X_tr, y_tr)
        pred = search.predict(X_te)
        _, _, recall = per_class_report(y_te, pred)
        best = {k.replace("svc__", ""): v for k, v in search.best_params_.items()}
        print(f"{fs}: untuned poly5 test {base_acc:.4f} | tuned CV {search.best_score_:.4f} "
              f"test {(pred == y_te).mean():.4f} | best {best} | draw recall {recall[1]:.4f}")
        out.append({"features": fs, "untuned_test_acc": f"{base_acc:.4f}",
                    "tuned_cv_acc": f"{search.best_score_:.4f}", "tuned_test_acc": f"{(pred == y_te).mean():.4f}",
                    "draw_recall": f"{recall[1]:.4f}", "best_params": best})

    with open(RESULTS / "tune_svm.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(out[0]))
        writer.writeheader()
        writer.writerows(out)
    print("wrote results/tune_svm.csv")


if __name__ == "__main__":
    main()
