"""Does re-weighting classes help the models predict draws?  (extension beyond the paper)

Run from the repository root, after src/build_features.py:  python src/class_weight_experiment.py
Compares class_weight=None with class_weight='balanced' for three models on F1 and F2.
Writes results/class_weight.csv.
"""
import csv
import warnings

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from experiment_utils import RESULTS, load_matches, per_class_report, split_feature_sets
from run_experiments import SEED

MODELS = {
    "SVM (linear)": lambda cw: make_pipeline(StandardScaler(), SVC(kernel="linear", class_weight=cw)),
    "SVM (RBF)": lambda cw: make_pipeline(StandardScaler(), SVC(kernel="rbf", class_weight=cw)),
    "SoftMax (linear)": lambda cw: make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, class_weight=cw)),
}


def main():
    RESULTS.mkdir(exist_ok=True)
    warnings.simplefilter("ignore")
    rows, y = load_matches()
    feature_sets = split_feature_sets(rows, y, SEED)
    out = []
    for fs in ("F1", "F2"):
        X_tr, y_tr, X_te, y_te = feature_sets[fs]
        for name, build in MODELS.items():
            for cw in (None, "balanced"):
                pred = build(cw).fit(X_tr, y_tr).predict(X_te)
                _, precision, recall = per_class_report(y_te, pred)
                acc = (pred == y_te).mean()
                print(f"{name:17s} {fs} class_weight={str(cw):8s} acc {acc:.4f}  draw recall {recall[1]:.4f}  "
                      f"draw precision {precision[1]:.4f}  draws predicted {(pred == 0).sum()}")
                out.append({"model": name, "features": fs, "class_weight": cw or "none", "test_acc": f"{acc:.4f}",
                            "draw_recall": f"{recall[1]:.4f}", "draw_precision": f"{precision[1]:.4f}",
                            "draws_predicted": int((pred == 0).sum())})
    with open(RESULTS / "class_weight.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(out[0]))
        writer.writeheader()
        writer.writerows(out)
    print("wrote results/class_weight.csv")


if __name__ == "__main__":
    main()
