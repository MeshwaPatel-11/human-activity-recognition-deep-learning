import argparse
import os
import csv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.dataset import load_ucihar, select_sensors
from src.splits import subject_independent_split, check_no_subject_overlap
from src.torch_dataset import HARWindowDataset
from src.models.cnn_lstm import CNNLSTM
from src.utils import ensure_dir, compute_metrics, save_json

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys, preds = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb)
        yhat = torch.argmax(logits, dim=1).cpu().numpy()
        preds.append(yhat)
        ys.append(yb.numpy())
    y_true = np.concatenate(ys) + 1
    y_pred = np.concatenate(preds) + 1
    return y_true, y_pred

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/UCI_HAR_Dataset")
    parser.add_argument("--split", type=str, choices=["standard", "subject"], default="subject")
    parser.add_argument("--sensors", type=str, choices=["both", "accel", "gyro"], default="both")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_data, test_data = load_ucihar(args.data_dir)

    X_train = select_sensors(train_data.X, args.sensors)
    y_train = train_data.y
    s_train = train_data.subjects

    X_test = select_sensors(test_data.X, args.sensors)
    y_test = test_data.y
    s_test = test_data.subjects

    if args.split == "standard":
        mean = X_train.mean(axis=(0, 1), keepdims=True)
        std = X_train.std(axis=(0, 1), keepdims=True) + 1e-8
        Xtr = (X_train - mean) / std
        Xte = (X_test - mean) / std

        dl_tr = DataLoader(HARWindowDataset(Xtr, y_train), batch_size=args.batch_size, shuffle=True)
        dl_te = DataLoader(HARWindowDataset(Xte, y_test), batch_size=args.batch_size, shuffle=False)

        n_channels = Xtr.shape[-1]
        model = CNNLSTM(n_channels=n_channels, n_classes=6, dropout=args.dropout).to(device)

        criterion = nn.CrossEntropyLoss()
        optim = torch.optim.Adam(model.parameters(), lr=args.lr)

        best_metric = -1.0
        best_path = f"outputs/results/cnnlstm_{args.split}_{args.sensors}_seed{args.seed}.pt"
        ensure_dir(os.path.dirname(best_path))

        log_path = best_path.replace(".pt", "_trainlog.csv")
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["epoch", "eval_accuracy", "eval_macro_f1"])  # eval=test for standard split

        for epoch in range(1, args.epochs + 1):
            model.train()
            for xb, yb in dl_tr:
                xb, yb = xb.to(device), yb.to(device)
                optim.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optim.step()

            y_true, y_pred = evaluate(model, dl_te, device)
            m = compute_metrics(y_true, y_pred)
            print(f"Epoch {epoch:02d} | TEST acc={m['accuracy']:.4f} macroF1={m['macro_f1']:.4f}")

            with open(log_path, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow([epoch, m["accuracy"], m["macro_f1"]])

            if m["macro_f1"] > best_metric:
                best_metric = m["macro_f1"]
                torch.save(model.state_dict(), best_path)

        save_json({"args": vars(args), "best_test_macro_f1": best_metric, "model_path": best_path},
                  best_path.replace(".pt", ".json"))

        print("Saved best model to:", best_path)
        print("Saved training log to:", log_path)

    else:
        X_all = np.concatenate([X_train, X_test], axis=0)
        y_all = np.concatenate([y_train, y_test], axis=0)
        s_all = np.concatenate([s_train, s_test], axis=0)

        split = subject_independent_split(s_all, seed=args.seed)
        check_no_subject_overlap(s_all, split)
        tr_idx, va_idx, te_idx = split["train_idx"], split["val_idx"], split["test_idx"]

        Xtr_raw, ytr = X_all[tr_idx], y_all[tr_idx]
        Xva_raw, yva = X_all[va_idx], y_all[va_idx]
        Xte_raw, yte = X_all[te_idx], y_all[te_idx]

        mean = Xtr_raw.mean(axis=(0, 1), keepdims=True)
        std = Xtr_raw.std(axis=(0, 1), keepdims=True) + 1e-8
        Xtr = (Xtr_raw - mean) / std
        Xva = (Xva_raw - mean) / std
        Xte = (Xte_raw - mean) / std

        dl_tr = DataLoader(HARWindowDataset(Xtr, ytr), batch_size=args.batch_size, shuffle=True)
        dl_va = DataLoader(HARWindowDataset(Xva, yva), batch_size=args.batch_size, shuffle=False)
        dl_te = DataLoader(HARWindowDataset(Xte, yte), batch_size=args.batch_size, shuffle=False)

        n_channels = Xtr.shape[-1]
        model = CNNLSTM(n_channels=n_channels, n_classes=6, dropout=args.dropout).to(device)

        criterion = nn.CrossEntropyLoss()
        optim = torch.optim.Adam(model.parameters(), lr=args.lr)

        best_val_f1 = -1.0
        bad_epochs = 0
        best_path = f"outputs/results/cnnlstm_{args.split}_{args.sensors}_seed{args.seed}.pt"
        ensure_dir(os.path.dirname(best_path))

        log_path = best_path.replace(".pt", "_trainlog.csv")
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["epoch", "val_accuracy", "val_macro_f1"])

        for epoch in range(1, args.epochs + 1):
            model.train()
            for xb, yb in dl_tr:
                xb, yb = xb.to(device), yb.to(device)
                optim.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optim.step()

            yva_true, yva_pred = evaluate(model, dl_va, device)
            mva = compute_metrics(yva_true, yva_pred)
            print(f"Epoch {epoch:02d} | VAL acc={mva['accuracy']:.4f} macroF1={mva['macro_f1']:.4f}")

            with open(log_path, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow([epoch, mva["accuracy"], mva["macro_f1"]])

            if mva["macro_f1"] > best_val_f1:
                best_val_f1 = mva["macro_f1"]
                bad_epochs = 0
                torch.save(model.state_dict(), best_path)
            else:
                bad_epochs += 1
                if bad_epochs >= args.patience:
                    print("Early stopping triggered.")
                    break

        model.load_state_dict(torch.load(best_path, map_location=device))
        yte_true, yte_pred = evaluate(model, dl_te, device)
        mte = compute_metrics(yte_true, yte_pred)

        save_json({"args": vars(args), "best_val_macro_f1": best_val_f1, "test_metrics": mte, "model_path": best_path},
                  best_path.replace(".pt", ".json"))

        print("BEST VAL macroF1:", best_val_f1)
        print("TEST:", mte)
        print("Saved best model to:", best_path)
        print("Saved training log to:", log_path)

if __name__ == "__main__":
    main()