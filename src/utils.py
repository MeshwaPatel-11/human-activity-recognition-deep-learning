import json
import os
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def compute_metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }

def save_json(obj, path: str):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def save_confusion(y_true, y_pred, labels, path: str):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    ensure_dir(os.path.dirname(path))
    np.savetxt(path, cm, fmt="%d", delimiter=",")
    return cm

def save_classification_report(y_true, y_pred, target_names, path: str):
    rep = classification_report(y_true, y_pred, target_names=target_names, digits=4)
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        f.write(rep)
    return rep