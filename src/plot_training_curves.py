import os
import csv
import matplotlib.pyplot as plt

def read_csv(path):
    epochs, accs, f1s = [], [], []
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            epochs.append(int(row["epoch"]))
            accs.append(float(row["val_accuracy"]))
            f1s.append(float(row["val_macro_f1"]))
    return epochs, accs, f1s

def plot_curves(files, out_path, title):
    fig = plt.figure(figsize=(7, 4))
    ax = fig.add_subplot(111)

    for label, path in files:
        epochs, accs, f1s = read_csv(path)
        ax.plot(epochs, f1s, marker="o", label=f"{label} (val Macro-F1)")

    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Macro-F1")
    ax.legend()
    fig.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)

def main():
    out_dir = "outputs/figures"
    os.makedirs(out_dir, exist_ok=True)

    files = [
        ("CNN1D", "outputs/results/cnn1d_subject_both_seed42_trainlog.csv"),
        ("GRU", "outputs/results/gru_subject_both_seed42_trainlog.csv"),
        ("CNN-LSTM", "outputs/results/cnnlstm_subject_both_seed42_trainlog.csv"),
    ]

    plot_curves(files, f"{out_dir}/training_curves_val_macroF1.png",
                "Training curves (Validation Macro-F1 vs Epoch)")

    print("Saved training curve plot to outputs/figures/training_curves_val_macroF1.png")

if __name__ == "__main__":
    main()