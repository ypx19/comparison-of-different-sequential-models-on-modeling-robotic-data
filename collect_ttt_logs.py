import os
import re
from typing import List, Dict

import matplotlib.pyplot as plt
import pandas as pd


LOG_DIR = "log"
SUMMARY_TXT = os.path.join(LOG_DIR, "ttt_summary.txt")
SUMMARY_CSV = os.path.join(LOG_DIR, "ttt_summary.csv")
SUMMARY_FIG_MAE = os.path.join(LOG_DIR, "ttt_error_vs_H_MAE.png")
TRAIN_CURVE_FIG_K10_H10 = os.path.join(LOG_DIR, "ttt_training_curve_K10_H10.png")
K_ERROR_FIG_H10 = os.path.join(LOG_DIR, "ttt_K_vs_error_H10.png")


SUMMARY_RE = re.compile(
    r">> SUMMARY \(K=(\d+), H=(\d+)\).*TTT_test_MSE=([0-9eE+\-\.]+)\s*\|\s*TTT_test_MAE=([0-9eE+\-\.]+)"
)

TRAIN_LINE_RE = re.compile(
    r"TTT \(single\) \| K=(\d+) \| H=(\d+) \| Epoch (\d+)"
    r" \| train_MSE=([0-9eE+\-\.]+)"
    r" \| val_MSE=([0-9eE+\-\.]+)"
    r" \| val_MAE=([0-9eE+\-\.]+)"
)


def parse_log_file(path: str) -> Dict[str, float] | None:
    """Parse a single log file and extract K, H, TTT_test_MSE, TTT_test_MAE."""
    with open(path, "r") as f:
        lines = f.readlines()

    for line in reversed(lines):
        m = SUMMARY_RE.search(line)
        if m:
            K = int(m.group(1))
            H = int(m.group(2))
            ttt_mse = float(m.group(3))
            ttt_mae = float(m.group(4))
            return {
                "K": K,
                "H": H,
                "TTT_test_MSE": ttt_mse,
                "TTT_test_MAE": ttt_mae,
            }
    return None


def collect_all_logs() -> pd.DataFrame:
    os.makedirs(LOG_DIR, exist_ok=True)
    rows: List[Dict[str, float]] = []

    for fname in os.listdir(LOG_DIR):
        if not fname.startswith("ttt_K") or not fname.endswith(".log"):
            continue
        path = os.path.join(LOG_DIR, fname)
        rec = parse_log_file(path)
        if rec is None:
            print(f"[WARN] No SUMMARY line found in {fname}, skipping.")
            continue
        rows.append(rec)

    if not rows:
        raise RuntimeError("No valid TTT SUMMARY lines found in log files.")

    df = pd.DataFrame(rows).sort_values(["K", "H"]).reset_index(drop=True)
    return df


def save_txt(df: pd.DataFrame, path: str) -> None:
    with open(path, "w") as f:
        f.write("K\tH\tTTT_test_MSE\tTTT_test_MAE\n")
        for _, row in df.iterrows():
            f.write(
                f"{int(row['K'])}\t{int(row['H'])}\t"
                f"{row['TTT_test_MSE']:.6f}\t{row['TTT_test_MAE']:.6f}\n"
            )
    print(f"Saved summary TXT to {path}")


def plot_mae_vs_H(df: pd.DataFrame, path: str) -> None:
    """Plot TTT_test_MAE vs H for each K with dual-panel layout.

    Left panel: all H values on log-y scale to show full dynamic range.
    Right panel: H >= 3 on linear scale (zoomed in) to reveal fine differences.
    """
    Ks = sorted(df["K"].unique())
    Hs = sorted(df["H"].unique())
    Hs_zoom = [h for h in Hs if h >= 3]

    # Each K gets a distinct color + linestyle + marker combo for max legibility
    styles = [
        {"color": "#1f77b4", "linestyle": "-",  "marker": "o"},   # K=10
        {"color": "#ff7f0e", "linestyle": "--", "marker": "s"},   # K=50
        {"color": "#2ca02c", "linestyle": "-.", "marker": "^"},   # K=100
    ]

    fig, ax_zoom = plt.subplots(1, 1, figsize=(7, 4))

    for K, sty in zip(Ks, styles):
        sub = df[df["K"] == K].sort_values("H")
        sub_zoom = sub[sub["H"] >= 3]
        ax_zoom.plot(sub_zoom["H"].values, sub_zoom["TTT_test_MAE"].values,
                     label=f"K={K}", linewidth=1.8, markersize=6, **sty)

    ax_zoom.set_xlabel("H (horizon)")
    ax_zoom.set_ylabel("TTT test MAE")
    ax_zoom.set_title("TTT: Error Growth vs Horizon (Direct multi-step)")
    ax_zoom.set_xticks(Hs_zoom)
    ax_zoom.grid(True, alpha=0.3)
    ax_zoom.legend()

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved MAE-vs-H figure to {path}")


def parse_training_curve(path: str, K_target: int = 10, H_target: int = 10) -> pd.DataFrame:
    """Extract per-epoch train/val MSE/MAE for a specific (K,H) from one log."""
    epochs: List[int] = []
    train_mse: List[float] = []
    val_mse: List[float] = []
    val_mae: List[float] = []

    with open(path, "r") as f:
        for line in f:
            m = TRAIN_LINE_RE.search(line)
            if not m:
                continue
            K = int(m.group(1))
            H = int(m.group(2))
            if K != K_target or H != H_target:
                continue
            epoch = int(m.group(3))
            t_mse = float(m.group(4))
            v_mse = float(m.group(5))
            v_mae = float(m.group(6))
            epochs.append(epoch)
            train_mse.append(t_mse)
            val_mse.append(v_mse)
            val_mae.append(v_mae)

    if not epochs:
        raise RuntimeError(f"No training lines found for K={K_target}, H={H_target} in {path}")

    df = pd.DataFrame(
        {
            "epoch": epochs,
            "train_MSE": train_mse,
            "val_MSE": val_mse,
            "val_MAE": val_mae,
        }
    ).sort_values("epoch")
    return df


def plot_training_curve(df: pd.DataFrame, path: str, K: int = 10, H: int = 10) -> None:
    """Plot training curve for a given (K,H) using train_MSE and val_MSE."""
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    ax.plot(df["epoch"], df["train_MSE"], color="#1f77b4", linestyle="-", marker="o", label="train MSE")
    ax.plot(df["epoch"], df["val_MSE"], color="#ff7f0e", linestyle="-", marker="o", label="val MSE")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.set_title(f"Training curve (K={K}, H={H})")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved training curve for K={K}, H={H} to {path}")


def plot_K_vs_error_H10(df: pd.DataFrame, path: str, H_target: int = 10) -> None:
    """For H=10, plot TTT test MAE and RMSE vs K on a single axis."""
    sub = df[df["H"] == H_target].sort_values("K")
    if sub.empty:
        print(f"[WARN] No entries found for H={H_target} when plotting K-vs-error.")
        return

    Ks = sub["K"].values
    mae = sub["TTT_test_MAE"].values
    rmse = sub["TTT_test_MSE"].values ** 0.5

    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    ax.plot(Ks, mae,  color="#1f77b4", linestyle="-", marker="o", linewidth=1.8, markersize=6, label="Test MAE")
    ax.plot(Ks, rmse, color="#ff7f0e", linestyle="-", marker="o", linewidth=1.8, markersize=6, label="Test RMSE")

    ax.set_xlabel("History block length K")
    ax.set_ylabel("Error (normalized)")
    ax.set_title(f"TTT: Test Error vs K (Non-overlapping blocks, H={H_target})")
    ax.set_xticks(Ks)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved K-vs-error (H={H_target}) figure to {path}")


def main():
    df = collect_all_logs()
    # Save as txt and csv for convenience
    save_txt(df, SUMMARY_TXT)
    df.to_csv(SUMMARY_CSV, index=False)
    print(f"Saved summary CSV to {SUMMARY_CSV}")
    # Plot figure (MAE on y-axis)
    plot_mae_vs_H(df, SUMMARY_FIG_MAE)

    # Plot training curve for (K=10, H=10) using its log file
    log_path_10_10 = os.path.join(LOG_DIR, "ttt_K10_H10.log")
    if os.path.exists(log_path_10_10):
        try:
            train_df = parse_training_curve(log_path_10_10, K_target=10, H_target=10)
            plot_training_curve(train_df, TRAIN_CURVE_FIG_K10_H10, K=10, H=10)
        except Exception as e:
            print(f"[WARN] Failed to plot training curve for K=10,H=10: {e}")
    else:
        print(f"[WARN] Log file not found for K=10,H=10 at {log_path_10_10}")

    # Plot K vs error (MAE & MSE) for H=10
    plot_K_vs_error_H10(df, K_ERROR_FIG_H10, H_target=10)


if __name__ == "__main__":
    main()


