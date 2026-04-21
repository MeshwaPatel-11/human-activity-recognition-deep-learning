import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from src.dataset import load_ucihar

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cm_csv",
        type=str,
        default="outputs/results/cnnlstm_subject_both_seed42_cm.csv",
        help="Path to confusion matrix CSV saved by evaluate_dl.py",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/UCI_HAR_Dataset",
        help="UCI HAR dataset folder (for class label names)",
    )
    parser.add_argument(
        "--out_png",
        type=str,
        default="outputs/figures/cnnlstm_subject_both_confusion.png",
        help="Where to save the PNG",
    )
    parser.add_argument("--title", type=str, default="CNN-LSTM Confusion Matrix (Subject Split)")
    args = parser.parse_args()

    # Load confusion matrix
    cm = np.loadtxt(args.cm_csv, delimiter=",", dtype=int)

    # Get class names from dataset label map (1..6)
    train_data, _ = load_ucihar(args.data_dir)
    label_map = train_data.label_map
    class_labels = sorted(label_map.keys())
    class_names = [label_map[i] for i in class_labels]

    os.makedirs(os.path.dirname(args.out_png), exist_ok=True)

    # Plot
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)

    im = ax.imshow(cm, interpolation="nearest")  # no explicit colors requested

    ax.set_title(args.title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    # Add numbers in cells
    thresh = cm.max() * 0.6 if cm.max() > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=9
            )

    fig.tight_layout()
    plt.savefig(args.out_png, dpi=200)
    plt.close(fig)

    print("Saved confusion matrix image to:", args.out_png)

if __name__ == "__main__":
    main()