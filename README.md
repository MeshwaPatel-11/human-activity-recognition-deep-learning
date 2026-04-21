# Human Activity Recognition (HAR)

**Project Title:** Human Activity Recognition (HAR) from Smartphone Sensors: Deep Sequence Models vs Classical Baselines with Subject-Generalization, Sensor Ablation, and Robustness Testing  
**Course:** DS8013 – Deep Learning  
**Team:** Avikumar Patel (501376903), Meshwa Patel (501390663)

## 1) Overview

This project performs an empirical analysis on the **UCI HAR** dataset using:

- **Baseline:** Logistic Regression  
- **Deep models:** CNN1D, GRU, CNN-LSTM  
- **Primary evaluation:** Subject-independent split (disjoint subjects across train/val/test)  
- **Additional experiments:** Sensor ablation (accel vs gyro vs both), robustness testing (Gaussian noise + time-step masking)

**My Contribution:**

Built and trained deep learning models for HAR
Performed preprocessing and dataset handling
Conducted evaluation, robustness analysis, and ablation experiments
Generated visualizations and result summaries

**Dataset source (UCI):**  
https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones

## 2) Environment / Requirements

### Python
- Python **3.10+** (tested on Windows)

### Libraries
Install required packages:

```bash
pip install -r requirements.txt
```

`requirements.txt` contains:

* numpy
* pandas
* scikit-learn
* matplotlib
* torch
* tqdm

## 3) Dataset Setup

Download and extract the dataset so that you have:

```text
ds8013_har_project/
  data/
    UCI_HAR_Dataset/
      train/
      test/
      activity_labels.txt
      ...
```

## 4) Project Structure (Key Files)

```
src/
  dataset.py                # load UCI HAR inertial signals
  splits.py                 # subject-independent split utilities
  torch_dataset.py          # PyTorch Dataset wrapper
  utils.py                  # metrics + saving helpers

  train_baseline.py         # Logistic Regression baseline (standard or subject split)
  train_dl.py               # CNN1D training (standard or subject split)
  train_gru.py              # GRU training (standard or subject split)
  train_cnnlstm.py          # CNN-LSTM training (standard or subject split)

  evaluate_dl.py            # evaluation for DL checkpoints (cm + report)
  plot_confusion.py         # create confusion matrix PNG from saved cm CSV

  collect_results.py        # print main results from outputs/results/*.json
  make_report_artifacts.py  # generate CSV tables + ready-to-paste report text

  experiments/
    ablation_sensors.py     # train/eval CNN-LSTM for accel/gyro/both
    robustness.py           # robustness: noise + masking at inference
```

All outputs are saved under:

* `outputs/results/`
* `outputs/figures/`

## 5) How to Run (Commands)

Run commands from the **project root**: `ds8013_har_project/`

### A) Quick dataset check

```bash
python -m src.quick_check
```

### B) Baseline (Logistic Regression)

**Standard split:**

```bash
python -m src.train_baseline --split standard --sensors both
```

**Subject-independent split (primary):**

```bash
python -m src.train_baseline --split subject --sensors both
```

### C) Deep models (Subject-independent split)

**CNN1D:**

```bash
python -m src.train_dl --split subject --sensors both --epochs 30 --batch_size 128 --seed 42
```

**GRU:**

```bash
python -m src.train_gru --split subject --sensors both --epochs 30 --batch_size 128 --seed 42
```

**CNN-LSTM:**

```bash
python -m src.train_cnnlstm --split subject --sensors both --epochs 30 --batch_size 128 --seed 42
```

### D) Evaluate a trained DL checkpoint (Confusion Matrix + Classification Report)

Example for CNN-LSTM:

```bash
python -m src.evaluate_dl --model cnnlstm --sensors both --seed 42 --ckpt outputs/results/cnnlstm_subject_both_seed42.pt
```

This generates:

* `outputs/results/cnnlstm_subject_both_seed42_cm.csv`
* `outputs/results/cnnlstm_subject_both_seed42_report.txt`
* `outputs/results/cnnlstm_subject_both_seed42_eval.json`

### E) Sensor ablation (CNN-LSTM)

Runs accel-only, gyro-only, and both:

```bash
python -m src.experiments.ablation_sensors --epochs 30 --batch_size 128 --seed 42
```

### F) Robustness testing (CNN-LSTM vs GRU)

```bash
python -m src.experiments.robustness --models cnnlstm gru --sensors both --seed 42
```

Saves:

* `outputs/results/robustness_subject_both_seed42.json`

### G) Generate final result tables for report

```bash
pip install pandas
python -m src.make_report_artifacts
```

Saves:

* `outputs/results/table_main_subject.csv`
* `outputs/results/table_sensor_ablation.csv`
* `outputs/results/table_robustness.csv`

### H) Plot confusion matrix image (PNG)

```bash
python -m src.plot_confusion
```

Saves:

* `outputs/figures/cnnlstm_subject_both_confusion.png`

## 6) Generate Figures for the Report (EDA + Results + Training Curves)

### A) EDA Figures

```bash
python -m src.eda_plots
```

Outputs:

* `outputs/figures/eda_*.png`

### B) Results Visuals (bar charts + robustness curves)

```bash
python -m src.plot_results
```

Outputs:

* `outputs/figures/results_*.png`

### C) Training Curves (Validation Macro-F1 vs Epoch)

```bash
python -m src.plot_training_curves
```

Output:

* `outputs/figures/training_curves_val_macroF1.png`

### D) Confusion Matrix PNG
```bash
python -m src.plot_confusion
```

Output:

* `outputs/figures/cnnlstm_subject_both_confusion.png`

## 7) Notes

* Subject-independent split uses disjoint subject IDs across train, val, test (seed=42).
* Normalization is computed using training split statistics only.
* Deep models use early stopping based on validation Macro-F1.
