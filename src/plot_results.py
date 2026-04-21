import os
import json
import numpy as np
import matplotlib.pyplot as plt

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def bar_plot(labels, values, title, ylabel, out_path):
    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(111)
    ax.bar(range(len(labels)), values)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)

def line_plot(x, ys, labels, title, xlabel, ylabel, out_path):
    fig = plt.figure(figsize=(7, 4))
    ax = fig.add_subplot(111)
    for y, lab in zip(ys, labels):
        ax.plot(x, y, marker="o", label=lab)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    fig.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)

def main():
    out_dir = "outputs/figures"
    ensure_dir(out_dir)

    # Main comparison (subject split)
    main_files = [
        ("LogReg", "outputs/results/baseline_logreg_subject_both_seed42.json", "test_metrics"),
        ("CNN1D", "outputs/results/cnn1d_subject_both_seed42.json", "test_metrics"),
        ("GRU", "outputs/results/gru_subject_both_seed42.json", "test_metrics"),
        ("CNN-LSTM", "outputs/results/cnnlstm_subject_both_seed42.json", "test_metrics"),
    ]

    labels = []
    accs = []
    f1s = []
    for name, path, key in main_files:
        d = read_json(path)
        m = d[key]
        labels.append(name)
        accs.append(m["accuracy"])
        f1s.append(m["macro_f1"])

    bar_plot(labels, f1s, "Model comparison (Subject Split) - Macro-F1", "Macro-F1",
             f"{out_dir}/results_main_macroF1.png")
    bar_plot(labels, accs, "Model comparison (Subject Split) - Accuracy", "Accuracy",
             f"{out_dir}/results_main_accuracy.png")

    # Sensor ablation (CNN-LSTM)
    ab_files = [
        ("Gyro only", "outputs/results/cnnlstm_subject_gyro_seed42_eval.json"),
        ("Accel only", "outputs/results/cnnlstm_subject_accel_seed42_eval.json"),
        ("Both", "outputs/results/cnnlstm_subject_both_seed42_eval.json"),
    ]
    labels = []
    f1s = []
    for name, path in ab_files:
        d = read_json(path)
        labels.append(name)
        f1s.append(d["metrics"]["macro_f1"])

    bar_plot(labels, f1s, "Sensor ablation (CNN-LSTM) - Macro-F1", "Macro-F1",
             f"{out_dir}/results_ablation_macroF1.png")

    # Robustness curves
    rob = read_json("outputs/results/robustness_subject_both_seed42.json")["results"]

    # Noise
    sigmas = [r["sigma"] for r in rob["cnnlstm"]["noise"]]
    cnn_noise_f1 = [r["macro_f1"] for r in rob["cnnlstm"]["noise"]]
    gru_noise_f1 = [r["macro_f1"] for r in rob["gru"]["noise"]]
    line_plot(sigmas, [cnn_noise_f1, gru_noise_f1], ["CNN-LSTM", "GRU"],
              "Robustness to Gaussian noise (Macro-F1)",
              "Noise sigma", "Macro-F1",
              f"{out_dir}/results_robust_noise_macroF1.png")

    # Masking
    ps = [r["mask_p"] for r in rob["cnnlstm"]["mask"]]
    cnn_mask_f1 = [r["macro_f1"] for r in rob["cnnlstm"]["mask"]]
    gru_mask_f1 = [r["macro_f1"] for r in rob["gru"]["mask"]]
    line_plot(ps, [cnn_mask_f1, gru_mask_f1], ["CNN-LSTM", "GRU"],
              "Robustness to time-step masking (Macro-F1)",
              "Mask fraction p", "Macro-F1",
              f"{out_dir}/results_robust_mask_macroF1.png")

    print("Saved results plots to outputs/figures/ (results_*.png)")

if __name__ == "__main__":
    main()