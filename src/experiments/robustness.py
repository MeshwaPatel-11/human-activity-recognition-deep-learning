import argparse
import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.dataset import load_ucihar, select_sensors
from src.splits import subject_independent_split, check_no_subject_overlap
from src.torch_dataset import HARWindowDataset
from src.utils import compute_metrics, save_json

from src.models.cnn1d import CNN1D
from src.models.rnn import RNNModel
from src.models.cnn_lstm import CNNLSTM


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(model_name: str, n_channels: int, device: str):
    model_name = model_name.lower()
    if model_name == "cnn1d":
        return CNN1D(n_channels=n_channels, n_classes=6).to(device)
    if model_name == "gru":
        return RNNModel(n_channels=n_channels, n_classes=6, hidden_size=128, num_layers=2, dropout=0.3, rnn_type="gru").to(device)
    if model_name == "lstm":
        return RNNModel(n_channels=n_channels, n_classes=6, hidden_size=128, num_layers=2, dropout=0.3, rnn_type="lstm").to(device)
    if model_name == "cnnlstm":
        return CNNLSTM(n_channels=n_channels, n_classes=6, dropout=0.3).to(device)
    raise ValueError("model must be one of: cnn1d, gru, lstm, cnnlstm")


@torch.no_grad()
def predict(model, X, y, device, batch_size=256):
    loader = DataLoader(HARWindowDataset(X, y), batch_size=batch_size, shuffle=False)
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


def add_noise(X, sigma):
    return X + np.random.normal(0.0, sigma, size=X.shape).astype(np.float32)


def time_mask(X, p):
    """
    Randomly zero-out p fraction of timesteps (same mask across channels).
    """
    X = X.copy()
    N, T, C = X.shape
    mask = (np.random.rand(N, T) < p)  # True means drop timestep
    for c in range(C):
        X[:, :, c][mask] = 0.0
    return X


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/UCI_HAR_Dataset")
    parser.add_argument("--sensors", type=str, choices=["both", "accel", "gyro"], default="both")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--models", nargs="+", default=["cnnlstm", "gru"])
    parser.add_argument("--noise_levels", nargs="+", type=float, default=[0.0, 0.05, 0.10, 0.20])
    parser.add_argument("--mask_levels", nargs="+", type=float, default=[0.0, 0.05, 0.10, 0.20])
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

    # Subject split from combined data (same as training)
    X_all = np.concatenate([X_train, X_test], axis=0)
    y_all = np.concatenate([y_train, y_test], axis=0)
    s_all = np.concatenate([s_train, s_test], axis=0)

    split = subject_independent_split(s_all, seed=args.seed)
    check_no_subject_overlap(s_all, split)

    tr_idx, te_idx = split["train_idx"], split["test_idx"]
    Xtr_raw = X_all[tr_idx]
    Xte_raw = X_all[te_idx]
    yte = y_all[te_idx]

    # Normalize using TRAIN stats only
    mean = Xtr_raw.mean(axis=(0, 1), keepdims=True)
    std = Xtr_raw.std(axis=(0, 1), keepdims=True) + 1e-8
    Xte = (Xte_raw - mean) / std

    results = {}

    for model_name in args.models:
        ckpt = f"outputs/results/{model_name}_subject_{args.sensors}_seed{args.seed}.pt"
        if not os.path.exists(ckpt):
            print(f"[WARN] Missing checkpoint: {ckpt} (skip)")
            continue

        model = build_model(model_name, n_channels=Xte.shape[-1], device=device)
        model.load_state_dict(torch.load(ckpt, map_location=device))

        model_res = {"noise": [], "mask": []}

        # Noise robustness
        for sigma in args.noise_levels:
            Xn = add_noise(Xte, sigma) if sigma > 0 else Xte
            yt, yp = predict(model, Xn, yte, device)
            m = compute_metrics(yt, yp)
            model_res["noise"].append({"sigma": sigma, **m})
            print(f"{model_name} | noise sigma={sigma:.2f} -> acc={m['accuracy']:.4f}, macroF1={m['macro_f1']:.4f}")

        # Mask robustness
        for p in args.mask_levels:
            Xm = time_mask(Xte, p) if p > 0 else Xte
            yt, yp = predict(model, Xm, yte, device)
            m = compute_metrics(yt, yp)
            model_res["mask"].append({"mask_p": p, **m})
            print(f"{model_name} | mask p={p:.2f} -> acc={m['accuracy']:.4f}, macroF1={m['macro_f1']:.4f}")

        results[model_name] = model_res

    out = f"outputs/results/robustness_subject_{args.sensors}_seed{args.seed}.json"
    save_json({"args": vars(args), "results": results}, out)
    print("\nSaved robustness results to:", out)


if __name__ == "__main__":
    main()