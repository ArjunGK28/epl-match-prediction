"""Shared helpers for the extra experiments (tuning, multi-seed, analysis, class weights).

Reuses the feature columns from build_features.py and the model definitions from
run_experiments.py, so the extra experiments stay consistent with the base pipeline.
"""
import csv
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from build_features import F1_COLUMNS, F2_COLUMNS, F3_COLUMNS

ROOT = Path(__file__).resolve().parent.parent
MATCHES = ROOT / "data" / "processed" / "matches.csv"
RESULTS = ROOT / "results"
LABELS = [1, 0, -1]  # home win, draw, home loss
FEATURE_COLUMNS = {"F1": F1_COLUMNS, "F2": F2_COLUMNS, "F3": F3_COLUMNS}


def load_matches():
    """Return (rows, y) for every match in data/processed/matches.csv."""
    with open(MATCHES, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    y = np.array([int(r["label"]) for r in rows])
    return rows, y


def split_feature_sets(rows, y, seed, test_size=0.2):
    """Stratified random 80/20 split for a given seed; returns {name: (X_tr, y_tr, X_te, y_te)}."""
    train, test = train_test_split(np.arange(len(y)), test_size=test_size, stratify=y, random_state=seed)
    out = {}
    for name, columns in FEATURE_COLUMNS.items():
        X = np.array([[float(r[c]) for c in columns] for r in rows])
        out[name] = (X[train], y[train], X[test], y[test])
    return out


def per_class_report(y_true, y_pred):
    """Precision and recall for home win / draw / home loss (0 when a class is never predicted)."""
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    recall = cm.diagonal() / cm.sum(axis=1)
    predicted = cm.sum(axis=0)
    precision = np.divide(cm.diagonal(), predicted, out=np.zeros(len(LABELS)), where=predicted > 0)
    return cm, precision, recall
