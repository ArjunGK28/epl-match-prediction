"""Train every model on feature sets F1, F2 and F3 and report train/test accuracy (Figure 3 of the paper).

Run from the repository root, after src/build_features.py:  python src/run_experiments.py
"""
import csv
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVC

from build_features import F1_COLUMNS, F2_COLUMNS, F3_COLUMNS
from gda import GDA

ROOT = Path(__file__).resolve().parent.parent
MATCHES = ROOT / "data" / "processed" / "matches.csv"
RESULTS = ROOT / "results"
TEST_SIZE = 0.2
SEED = 0
LABELS = [1, 0, -1]  # home win, draw, home loss


def load_feature_sets():
    with open(MATCHES, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    y = np.array([int(r["label"]) for r in rows])
    # one random 80/20 split, stratified by outcome and shared by all feature sets
    train, test = train_test_split(np.arange(len(y)), test_size=TEST_SIZE, stratify=y, random_state=SEED)
    feature_sets = {}
    for name, columns in [("F1", F1_COLUMNS), ("F2", F2_COLUMNS), ("F3", F3_COLUMNS)]:
        X = np.array([[float(r[c]) for c in columns] for r in rows])
        feature_sets[name] = (X[train], y[train], X[test], y[test])
    return feature_sets


def make_models():
    # every model except GDA gets standardised inputs; the scaler is fitted on the training set only
    return {
        "GDA": lambda: GDA(),
        "SVM (linear)": lambda: make_pipeline(StandardScaler(), SVC(kernel="linear")),
        "SVM (poly5)": lambda: make_pipeline(StandardScaler(), SVC(kernel="poly", degree=5)),
        "SVM (RBF)": lambda: make_pipeline(StandardScaler(), SVC(kernel="rbf")),
        "SoftMax (linear)": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
        "SoftMax (quad.)": lambda: make_pipeline(
            StandardScaler(), PolynomialFeatures(2, include_bias=False), StandardScaler(),
            LogisticRegression(max_iter=5000)),
        # one hidden layer trained with regularised SGD; the number of hidden nodes is chosen
        # by 5-fold cross-validation on the training set
        "NN": lambda: GridSearchCV(
            make_pipeline(StandardScaler(), MLPClassifier(solver="sgd", alpha=1.0, activation="tanh",
                                                          max_iter=5000, random_state=SEED)),
            {"mlpclassifier__hidden_layer_sizes": [(n,) for n in (4, 8, 16)]}, cv=5, n_jobs=-1),
    }


SKIP = {("SoftMax (quad.)", "F3"): "not run (about 12,000 quadratic features; the paper also stopped it)"}


def recall(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    return cm.diagonal() / cm.sum(axis=1)


def main():
    RESULTS.mkdir(exist_ok=True)
    warnings.simplefilter("ignore", ConvergenceWarning)
    feature_sets = load_feature_sets()
    rows = []
    for model_name, build in make_models().items():
        for fs_name, (X_tr, y_tr, X_te, y_te) in feature_sets.items():
            row = {"model": model_name, "features": fs_name, "n_features": X_tr.shape[1]}
            if (model_name, fs_name) in SKIP:
                row["note"] = SKIP[(model_name, fs_name)]
            else:
                start = time.time()
                try:
                    model = build().fit(X_tr, y_tr)
                    pred_te = model.predict(X_te)
                    win, draw, loss = recall(y_te, pred_te)
                    row.update(train_acc=(model.predict(X_tr) == y_tr).mean(), test_acc=(pred_te == y_te).mean(),
                               recall_win=win, recall_draw=draw, recall_loss=loss,
                               draws_predicted=int((pred_te == 0).sum()))
                    if isinstance(model, GridSearchCV):
                        row["note"] = f"hidden nodes = {model.best_params_['mlpclassifier__hidden_layer_sizes'][0]}"
                except np.linalg.LinAlgError as err:
                    row["note"] = f"failed: {err}"
                row["seconds"] = round(time.time() - start, 1)
            rows.append(row)
            acc = f"train {row['train_acc']:.4f}  test {row['test_acc']:.4f}" if "test_acc" in row else ""
            print(f"{model_name:17s} {fs_name}  {acc}  {row.get('note', '')}", flush=True)

    fields = ["model", "features", "n_features", "train_acc", "test_acc", "recall_win", "recall_draw",
              "recall_loss", "draws_predicted", "seconds", "note"]
    with open(RESULTS / "results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (f"{v:.4f}" if isinstance(v, float) and k != "seconds" else v) for k, v in row.items()})

    y_all = np.concatenate([feature_sets["F1"][1], feature_sets["F1"][3]])
    home_win_rate = (y_all == 1).mean()
    print(f"\nmatches: {len(y_all)}  home win {home_win_rate:.4f}  draw {(y_all == 0).mean():.4f}  "
          f"home loss {(y_all == -1).mean():.4f}")
    plot(rows, home_win_rate)


def plot(rows, home_win_rate):
    models = list(dict.fromkeys(r["model"] for r in rows))
    fig, ax = plt.subplots(figsize=(12, 4.5))
    colours = {"F1": "#1f77b4", "F2": "#2ca02c", "F3": "#d62728"}
    width = 0.13
    for j, fs in enumerate(colours):
        for k, (key, hatch) in enumerate([("train_acc", None), ("test_acc", "//")]):
            values = [next((r.get(key, 0) for r in rows if r["model"] == m and r["features"] == fs), 0) for m in models]
            offset = (2 * j + k - 2.5) * width
            ax.bar(np.arange(len(models)) + offset, values, width, color=colours[fs], alpha=0.55 if k == 0 else 1.0,
                   hatch=hatch, edgecolor="white", label=f"{fs} {'train' if k == 0 else 'test'}")
    ax.axhline(home_win_rate, color="black", linestyle="--", linewidth=1, label="always home win")
    ax.set_xticks(np.arange(len(models)), models)
    ax.set_ylim(0.4, 1.02)
    ax.set_ylabel("Accuracy")
    ax.set_title("Training and test accuracy by model and feature set")
    ax.legend(ncol=4, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(RESULTS / "accuracy.png", dpi=150)


if __name__ == "__main__":
    main()
