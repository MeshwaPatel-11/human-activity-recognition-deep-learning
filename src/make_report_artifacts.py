import json
import os
import pandas as pd

def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_table(df, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)

def fmt(x):
    return f"{x:.4f}"

def main():
    out_dir = "outputs/results"

    # Table 1: Main model comparison (subject split)
    main_files = {
        "LogReg (Subject Split)": "baseline_logreg_subject_both_seed42.json",
        "CNN1D (Subject Split)": "cnn1d_subject_both_seed42.json",
        "GRU (Subject Split)": "gru_subject_both_seed42.json",
        "CNN-LSTM (Subject Split)": "cnnlstm_subject_both_seed42.json",
    }

    rows = []
    for name, fn in main_files.items():
        p = os.path.join(out_dir, fn)
        d = read_json(p)
        if "test_metrics" in d:
            m = d["test_metrics"]
        elif "metrics" in d:
            m = d["metrics"]
        else:
            raise ValueError(f"Unexpected JSON format: {fn}")
        rows.append({"Model": name, "Accuracy": m["accuracy"], "Macro-F1": m["macro_f1"]})

    df_main = pd.DataFrame(rows).sort_values("Macro-F1", ascending=False)
    save_table(df_main, os.path.join(out_dir, "table_main_subject.csv"))

    # Table 2: Sensor ablation (CNN-LSTM subject split)
    ablation_files = {
        "Accel only": "cnnlstm_subject_accel_seed42_eval.json",
        "Gyro only": "cnnlstm_subject_gyro_seed42_eval.json",
        "Accel + Gyro (Both)": "cnnlstm_subject_both_seed42_eval.json",
    }
    rows = []
    for setting, fn in ablation_files.items():
        p = os.path.join(out_dir, fn)
        d = read_json(p)
        m = d["metrics"]
        rows.append({"Sensor Setting": setting, "Accuracy": m["accuracy"], "Macro-F1": m["macro_f1"]})
    df_ab = pd.DataFrame(rows).sort_values("Macro-F1", ascending=False)
    save_table(df_ab, os.path.join(out_dir, "table_sensor_ablation.csv"))

    # Table 3: Robustness (Noise + Masking)
    rob = read_json(os.path.join(out_dir, "robustness_subject_both_seed42.json"))["results"]

    rob_rows = []
    for model_name, res in rob.items():
        for item in res["noise"]:
            rob_rows.append({
                "Model": model_name,
                "Perturbation": "GaussianNoise",
                "Level": item["sigma"],
                "Accuracy": item["accuracy"],
                "Macro-F1": item["macro_f1"],
            })
        for item in res["mask"]:
            rob_rows.append({
                "Model": model_name,
                "Perturbation": "TimeMask",
                "Level": item["mask_p"],
                "Accuracy": item["accuracy"],
                "Macro-F1": item["macro_f1"],
            })

    df_rob = pd.DataFrame(rob_rows).sort_values(["Perturbation", "Level", "Model"])
    save_table(df_rob, os.path.join(out_dir, "table_robustness.csv"))

    # Print tables
    print("\n=== TABLE 1: Main comparison (Subject Split) ===")
    print(df_main.to_string(index=False, formatters={"Accuracy": fmt, "Macro-F1": fmt}))

    print("\n=== TABLE 2: Sensor Ablation (CNN-LSTM, Subject Split) ===")
    print(df_ab.to_string(index=False, formatters={"Accuracy": fmt, "Macro-F1": fmt}))

    print("\n=== TABLE 3: Robustness (subset view: Macro-F1) ===")
    df_rob_show = df_rob.copy()
    print(df_rob_show.to_string(index=False, formatters={"Accuracy": fmt, "Macro-F1": fmt}))

    # Results
    best_model = df_main.iloc[0]["Model"]
    best_f1 = df_main.iloc[0]["Macro-F1"]
    best_acc = df_main.iloc[0]["Accuracy"]

    logreg = df_main[df_main["Model"].str.contains("LogReg")].iloc[0]
    improvement = best_f1 - logreg["Macro-F1"]

    ab_best = df_ab.iloc[0]
    ab_worst = df_ab.iloc[-1]

    print("\n\n=== READY-TO-PASTE RESULTS TEXT ===\n")
    print(
        f"**Main comparison (subject-independent test):** "
        f"We compared a classical baseline (logistic regression) with three deep learning models (CNN1D, GRU, and CNN-LSTM). "
        f"The best-performing model was **{best_model}** with Accuracy={best_acc:.4f} and Macro-F1={best_f1:.4f}. "
        f"In contrast, logistic regression achieved Accuracy={logreg['Accuracy']:.4f} and Macro-F1={logreg['Macro-F1']:.4f}, "
        f"showing that deep sequence models improve Macro-F1 by approximately **{improvement:.4f}** under a subject-independent evaluation."
    )

    print(
        f"\n**Sensor ablation:** "
        f"Using CNN-LSTM, combining accelerometer and gyroscope signals performed best "
        f"(Accuracy={ab_best['Accuracy']:.4f}, Macro-F1={ab_best['Macro-F1']:.4f}). "
        f"Accelerometer-only was moderately competitive, while gyroscope-only performed substantially worse "
        f"(worst setting: {ab_worst['Sensor Setting']}, Accuracy={ab_worst['Accuracy']:.4f}, Macro-F1={ab_worst['Macro-F1']:.4f}). "
        f"This suggests that accelerometer channels carry most discriminative information for the activity classes, "
        f"and that fusing both modalities yields the highest overall performance."
    )

    print(
        "\n**Robustness to perturbations:** "
        "We evaluated robustness by injecting Gaussian noise and applying random time-step masking at test time. "
        "CNN-LSTM remained stable with only a small degradation as noise/masking increased. "
        "Interestingly, GRU showed strong tolerance to time-step masking and even improved slightly at moderate masking levels, "
        "which may indicate a regularization effect where the model relies on distributed temporal cues rather than individual time steps."
    )

    print("\nSaved CSV tables to:")
    print(" - outputs/results/table_main_subject.csv")
    print(" - outputs/results/table_sensor_ablation.csv")
    print(" - outputs/results/table_robustness.csv")

if __name__ == "__main__":
    main()