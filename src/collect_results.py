import glob
import json
import os

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    paths = sorted(glob.glob("outputs/results/*.json"))
    rows = []

    for p in paths:
        name = os.path.basename(p).replace(".json", "")
        data = load_json(p)

        if "test_metrics" in data:
            m = data["test_metrics"]
        elif "metrics" in data:
            m = data["metrics"]
        elif "test_metrics" in data:
            m = data["test_metrics"]
        else:
            continue

        rows.append((name, m.get("accuracy", None), m.get("macro_f1", None)))

    print("\n=== Model Comparison (from outputs/results/*.json) ===")
    print(f"{'Model':45s} {'Accuracy':>10s} {'Macro-F1':>10s}")
    print("-" * 70)
    for name, acc, f1 in rows:
        if acc is None or f1 is None:
            continue
        print(f"{name:45s} {acc:10.4f} {f1:10.4f}")

if __name__ == "__main__":
    main()