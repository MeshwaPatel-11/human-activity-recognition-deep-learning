import argparse
import subprocess
import sys

def run(cmd):
    print("\n>>>", " ".join(cmd))
    subprocess.check_call(cmd)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Train CNN-LSTM for each sensor setting
    for sensors in ["accel", "gyro", "both"]:
        run([
            sys.executable, "-m", "src.train_cnnlstm",
            "--split", "subject",
            "--sensors", sensors,
            "--epochs", str(args.epochs),
            "--batch_size", str(args.batch_size),
            "--seed", str(args.seed),
        ])

        # Evaluate and save cm/report
        ckpt = f"outputs/results/cnnlstm_subject_{sensors}_seed{args.seed}.pt"
        run([
            sys.executable, "-m", "src.evaluate_dl",
            "--model", "cnnlstm",
            "--sensors", sensors,
            "--seed", str(args.seed),
            "--ckpt", ckpt
        ])

if __name__ == "__main__":
    main()