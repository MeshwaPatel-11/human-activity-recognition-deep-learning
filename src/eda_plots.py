import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from src.dataset import load_ucihar, select_sensors

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def plot_class_distribution(y, label_map, out_path, title):
    labels = sorted(label_map.keys())
    counts = [(y == k).sum() for k in labels]
    names = [label_map[k] for k in labels]
    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(111)
    ax.bar(range(len(labels)), counts)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("Count")
    ax.set_title(title)
    fig.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)

def plot_subject_distribution(subjects, out_path, title):
    uniq = np.unique(subjects)
    counts = np.array([(subjects == s).sum() for s in uniq])
    fig = plt.figure(figsize=(9, 4))
    ax = fig.add_subplot(111)
    ax.bar(uniq, counts)
    ax.set_xlabel("Subject ID")
    ax.set_ylabel("#Windows")
    ax.set_title(title)
    fig.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)

def plot_example_windows(X, y, label_map, out_path, title, n_per_class=1, seed=42):
    """
    Plot example window per class using:
    - accel magnitude (channels 0..5)
    - gyro magnitude (channels 6..8 if present)
    """
    rng = np.random.default_rng(seed)
    labels = sorted(label_map.keys())
    fig = plt.figure(figsize=(10, 10))
    nrows = len(labels)
    axlist = fig.subplots(nrows=nrows, ncols=1, sharex=True)

    for i, k in enumerate(labels):
        idx = np.where(y == k)[0]
        idx_pick = rng.choice(idx, size=min(n_per_class, len(idx)), replace=False)
        j = idx_pick[0]

        # accel magnitude
        accel = X[j, :, :6]
        accel_mag = np.sqrt((accel ** 2).sum(axis=1))
        ax = axlist[i]
        ax.plot(accel_mag, label="Accel magnitude")

        if X.shape[-1] >= 9:
            gyro = X[j, :, 6:9]
            gyro_mag = np.sqrt((gyro ** 2).sum(axis=1))
            ax.plot(gyro_mag, label="Gyro magnitude")

        ax.set_title(f"{label_map[k]} (sample index {j})")
        ax.legend(loc="upper right")

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(out_path, dpi=200)
    plt.close(fig)

def plot_mean_signal_per_class(X, y, label_map, out_path, title):
    """
    For each class, plot mean accel magnitude over time and gyro magnitude if available
    """
    labels = sorted(label_map.keys())
    T = X.shape[1]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)

    for k in labels:
        idx = np.where(y == k)[0]
        Xk = X[idx]
        accel_mag = np.sqrt((Xk[:, :, :6] ** 2).sum(axis=2))  # (N,T)
        accel_mean = accel_mag.mean(axis=0)
        ax.plot(np.arange(T), accel_mean, label=f"{label_map[k]} (accel)")

    ax.set_title(title + " (Accel magnitude mean)")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Magnitude (normalized units)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)

    # Gyro plot
    if X.shape[-1] >= 9:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
        for k in labels:
            idx = np.where(y == k)[0]
            Xk = X[idx]
            gyro_mag = np.sqrt((Xk[:, :, 6:9] ** 2).sum(axis=2))
            gyro_mean = gyro_mag.mean(axis=0)
            ax.plot(np.arange(T), gyro_mean, label=f"{label_map[k]} (gyro)")
        ax.set_title(title + " (Gyro magnitude mean)")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Magnitude (normalized units)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        plt.savefig(out_path.replace(".png", "_gyro.png"), dpi=200)
        plt.close(fig)

def plot_channel_correlation(X, out_path, title):
    """
    Correlation across channels using flattened N*T per channel.
    """
    N, T, C = X.shape
    flat = X.reshape(N * T, C)
    corr = np.corrcoef(flat, rowvar=False)
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111)
    im = ax.imshow(corr, interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("Channel")
    ax.set_ylabel("Channel")
    ax.set_xticks(np.arange(C))
    ax.set_yticks(np.arange(C))
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)

def plot_pca(X, y, label_map, out_path, title, max_points=4000, seed=42):
    """
    PCA on flattened windows T*C. For visualization sample points.
    """
    rng = np.random.default_rng(seed)
    N = X.shape[0]
    idx = np.arange(N)
    if N > max_points:
        idx = rng.choice(idx, size=max_points, replace=False)

    Xs = X[idx].reshape(len(idx), -1)
    ys = y[idx]

    pca = PCA(n_components=2, random_state=seed)
    Z = pca.fit_transform(Xs)

    labels = sorted(label_map.keys())
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    for k in labels:
        m = ys == k
        ax.scatter(Z[m, 0], Z[m, 1], s=8, alpha=0.7, label=label_map[k])

    ax.set_title(title + f" (PCA 2D, n={len(idx)})")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(fontsize=8)
    fig.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)

def main():
    out_dir = "outputs/figures"
    ensure_dir(out_dir)

    train, test = load_ucihar("data/UCI_HAR_Dataset")
    label_map = train.label_map

    # Combined for EDA
    X_all = np.concatenate([train.X, test.X], axis=0)
    y_all = np.concatenate([train.y, test.y], axis=0)
    s_all = np.concatenate([train.subjects, test.subjects], axis=0)

    # Choose sensors for EDA
    X_all = select_sensors(X_all, "both")

    plot_class_distribution(y_all, label_map, f"{out_dir}/eda_class_distribution.png",
                            "Class distribution (All data)")
    plot_subject_distribution(s_all, f"{out_dir}/eda_subject_distribution.png",
                              "Subject distribution (#windows per subject)")

    plot_example_windows(X_all, y_all, label_map, f"{out_dir}/eda_example_windows.png",
                         "Example windows per class")

    plot_mean_signal_per_class(X_all, y_all, label_map, f"{out_dir}/eda_mean_signal_accel.png",
                               "Per-class mean signal")

    plot_channel_correlation(X_all, f"{out_dir}/eda_channel_correlation.png",
                             "Channel correlation heatmap (flattened over N*T)")

    plot_pca(X_all, y_all, label_map, f"{out_dir}/eda_pca.png",
             "Low-dimensional visualization")

    print("Saved EDA figures to outputs/figures/ (eda_*.png)")

if __name__ == "__main__":
    main()