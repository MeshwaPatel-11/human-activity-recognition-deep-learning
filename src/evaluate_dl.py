import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from src.dataset import load_ucihar, select_sensors
from src.splits import subject_independent_split, check_no_subject_overlap
from src.torch_dataset import HARWindowDataset
from src.utils import compute_metrics, save_json, save_confusion, save_classification_report
from src.models.cnn1d import CNN1D
from src.models.rnn import RNNModel
from src.models.cnn_lstm import CNNLSTM

@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    ys, preds = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb)
        yhat = torch.argmax(logits, dim=1).cpu().numpy()
        preds.append(yhat)
        ys.append(yb.numpy())
    y_true = np.concatenate(ys)
    y_pred = np.concatenate(preds)
    return y_true + 1, y_pred + 1  # back to 1..6

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/UCI_HAR_Dataset")
    parser.add_argument("--sensors", type=str, choices=["both", "accel", "gyro"], default="both")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", type=str, choices=["cnn1d", "gru", "lstm", "cnnlstm"], default="cnnlstm")
    parser.add_argument("--ckpt", type=str, required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_data, test_data = load_ucihar(args.data_dir)

    X_train = select_sensors(train_data.X, args.sensors)
    y_train = train_data.y
    s_train = train_data.subjects

    X_test = select_sensors(test_data.X, args.sensors)
    y_test = test_data.y
    s_test = test_data.subjects

    # Subject split from combined data
    X_all = np.concatenate([X_train, X_test], axis=0)
    y_all = np.concatenate([y_train, y_test], axis=0)
    s_all = np.concatenate([s_train, s_test], axis=0)

    split = subject_independent_split(s_all, seed=args.seed)
    check_no_subject_overlap(s_all, split)
    tr_idx, te_idx = split["train_idx"], split["test_idx"]

    Xtr_raw = X_all[tr_idx]
    Xte_raw = X_all[te_idx]
    yte = y_all[te_idx]

    # Normalize using TRAIN stats
    mean = Xtr_raw.mean(axis=(0, 1), keepdims=True)
    std = Xtr_raw.std(axis=(0, 1), keepdims=True) + 1e-8
    Xte = (Xte_raw - mean) / std

    dl_te = DataLoader(HARWindowDataset(Xte, yte), batch_size=256, shuffle=False)

    n_channels = Xte.shape[-1]
    if args.model == "cnn1d":
        model = CNN1D(n_channels=n_channels, n_classes=6).to(device)
    elif args.model in ["gru", "lstm"]:
        model = RNNModel(
            n_channels=n_channels, n_classes=6, hidden_size=128, num_layers=2,
            dropout=0.3, rnn_type=args.model
        ).to(device)
    else:
        model = CNNLSTM(n_channels=n_channels, n_classes=6, dropout=0.3).to(device)

    model.load_state_dict(torch.load(args.ckpt, map_location=device))

    y_true, y_pred = predict(model, dl_te, device)
    metrics = compute_metrics(y_true, y_pred)

    label_map = train_data.label_map
    class_labels = sorted(list(label_map.keys()))
    class_names = [label_map[i] for i in class_labels]

    base = args.ckpt.replace(".pt", "")
    save_json({"args": vars(args), "metrics": metrics}, base + "_eval.json")
    save_confusion(y_true, y_pred, class_labels, base + "_cm.csv")
    save_classification_report(y_true, y_pred, class_names, base + "_report.txt")

    print("EVAL METRICS:", metrics)
    print("Saved:", base + "_cm.csv")
    print("Saved:", base + "_report.txt")

if __name__ == "__main__":
    main()