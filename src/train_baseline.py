import argparse
import numpy as np
from src.dataset import load_ucihar, select_sensors, normalize_train_apply
from src.splits import subject_independent_split, check_no_subject_overlap
from src.models.baselines import train_logreg, predict_logreg
from src.utils import compute_metrics, save_json, save_confusion, save_classification_report

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/UCI_HAR_Dataset")
    parser.add_argument("--split", type=str, choices=["standard", "subject"], default="standard")
    parser.add_argument("--sensors", type=str, choices=["both", "accel", "gyro"], default="both")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_data, test_data = load_ucihar(args.data_dir)

    # Select sensors (ablation ready)
    X_train = select_sensors(train_data.X, args.sensors)
    y_train = train_data.y
    s_train = train_data.subjects

    X_test = select_sensors(test_data.X, args.sensors)
    y_test = test_data.y
    s_test = test_data.subjects

    label_map = train_data.label_map
    class_labels = sorted(list(label_map.keys()))
    class_names = [label_map[i] for i in class_labels]

    if args.split == "standard":
        # normalize using train stats only, apply to test
        X_train_n, X_test_n, mean, std = normalize_train_apply(X_train, X_test)

        model = train_logreg(X_train_n, y_train, seed=args.seed)
        y_pred = predict_logreg(model, X_test_n)

        metrics = compute_metrics(y_test, y_pred)

        out_prefix = f"outputs/results/baseline_logreg_standard_{args.sensors}_seed{args.seed}"
        save_json({"args": vars(args), "metrics": metrics}, out_prefix + ".json")
        save_confusion(y_test, y_pred, class_labels, out_prefix + "_cm.csv")
        save_classification_report(y_test, y_pred, class_names, out_prefix + "_report.txt")

        print("STANDARD SPLIT RESULT:", metrics)

    else:
        # SUBJECT split: combine train+test first then resplit by subjects
        X_all = np.concatenate([X_train, X_test], axis=0)
        y_all = np.concatenate([y_train, y_test], axis=0)
        s_all = np.concatenate([s_train, s_test], axis=0)

        split = subject_independent_split(s_all, seed=args.seed)
        check_no_subject_overlap(s_all, split)

        tr_idx, va_idx, te_idx = split["train_idx"], split["val_idx"], split["test_idx"]

        X_tr, y_tr = X_all[tr_idx], y_all[tr_idx]
        X_va, y_va = X_all[va_idx], y_all[va_idx]
        X_te, y_te = X_all[te_idx], y_all[te_idx]

        # Normalize using train stats only, apply to val/test
        X_tr_n, X_va_n, mean, std = normalize_train_apply(X_tr, X_va)
        X_tr_n2, X_te_n, _, _ = normalize_train_apply(X_tr, X_te)  # reuse train stats
        # Normalize_train_apply recomputes mean/std, so we use same stats:
        # We'll recompute correctly below:

        # Correct normalization reuse:
        mean = X_tr.mean(axis=(0, 1), keepdims=True)
        std = X_tr.std(axis=(0, 1), keepdims=True) + 1e-8
        X_tr_n = (X_tr - mean) / std
        X_va_n = (X_va - mean) / std
        X_te_n = (X_te - mean) / std

        model = train_logreg(X_tr_n, y_tr, seed=args.seed)

        # Evaluate on val + test
        y_va_pred = predict_logreg(model, X_va_n)
        y_te_pred = predict_logreg(model, X_te_n)

        metrics_val = compute_metrics(y_va, y_va_pred)
        metrics_test = compute_metrics(y_te, y_te_pred)

        out_prefix = f"outputs/results/baseline_logreg_subject_{args.sensors}_seed{args.seed}"
        save_json({
            "args": vars(args),
            "val_metrics": metrics_val,
            "test_metrics": metrics_test,
            "n_train": int(len(tr_idx)),
            "n_val": int(len(va_idx)),
            "n_test": int(len(te_idx)),
            "train_subjects": int(len(set(s_all[tr_idx]))),
            "val_subjects": int(len(set(s_all[va_idx]))),
            "test_subjects": int(len(set(s_all[te_idx]))),
        }, out_prefix + ".json")

        save_confusion(y_te, y_te_pred, class_labels, out_prefix + "_cm.csv")
        save_classification_report(y_te, y_te_pred, class_names, out_prefix + "_report.txt")

        print("SUBJECT SPLIT VAL:", metrics_val)
        print("SUBJECT SPLIT TEST:", metrics_test)

if __name__ == "__main__":
    main()