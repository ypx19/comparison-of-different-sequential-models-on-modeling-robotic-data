import os
import math
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    return device


def load_series(csv_path="../../robotic_arm_dataset_multiple_trajectories.csv"):
    df = pd.read_csv(csv_path)
    data = torch.from_numpy(
        df[["Axis_0_Angle", "Axis_1_Angle", "Axis_2_Angle"]].values.astype(
            np.float32
        )
    )
    T, d = data.shape
    print(f"Loaded ONE long trajectory: T={T}, d={d}")
    return data, d


def make_splits(series, train_frac=0.7, val_frac=0.15):
    T = len(series)
    train_end = int(train_frac * T)
    val_end = int((train_frac + val_frac) * T)
    train = series[:train_end]
    val = series[train_end:val_end]
    test = series[val_end:]
    return train, val, test


class SlidingWindowDataset(Dataset):
    def __init__(self, series, K):
        self.series = series
        self.K = K

    def __len__(self):
        return len(self.series) - self.K

    def __getitem__(self, idx):
        x = self.series[idx : idx + self.K]
        y = self.series[idx + self.K]
        return x, y


class RNNRegressor(nn.Module):
    def __init__(self, d_in, hidden_size):
        super().__init__()
        self.rnn = nn.GRU(d_in, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, d_in)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    n = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        batch_size = x.size(0)
        total_loss += loss.item() * batch_size
        n += batch_size
    return total_loss / n


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    n = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        pred = model(x)
        loss = criterion(pred, y)
        batch_size = x.size(0)
        total_loss += loss.item() * batch_size
        n += batch_size
    return total_loss / n


def evaluate_with_ttt(base_model, test_series, K, device, adapt_lr=1e-4, adapt_steps=1):
    """
    One-step-ahead TTT along the test trajectory, returning MSE & MAE.
    """
    import copy

    model = copy.deepcopy(base_model).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=adapt_lr)
    criterion_mse = nn.MSELoss()
    model.train()
    T_test = len(test_series)
    preds = []
    targets = []
    for start in range(T_test - K):
        window = test_series[start : start + K].unsqueeze(0).to(device)
        target = test_series[start + K].unsqueeze(0).to(device)
        for _ in range(adapt_steps):
            optimizer.zero_grad()
            pred = model(window)
            loss = criterion_mse(pred, target)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            pred = model(window)
        preds.append(pred.squeeze(0).cpu().numpy())
        targets.append(target.squeeze(0).cpu().numpy())
    preds = np.stack(preds)
    targets = np.stack(targets)
    mse = ((preds - targets) ** 2).mean()
    mae = np.abs(preds - targets).mean()
    return float(mse), float(mae)


def run_config(
    K,
    H,
    num_epochs=20,
    batch_size=256,
    adapt_lr=1e-4,
    adapt_steps=1,
    csv_path="../../robotic_arm_dataset_multiple_trajectories.csv",
):
    """
    Train a single (K, H) configuration and print final metrics (MSE & MAE).
    """
    device = get_device()
    series, d = load_series(csv_path)
    train_series, val_series, test_series = make_splits(series)

    train_ds = SlidingWindowDataset(train_series, K)
    val_ds = SlidingWindowDataset(val_series, K)
    test_ds = SlidingWindowDataset(test_series, K)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    base_test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    criterion_mse = nn.MSELoss()
    criterion_mae = nn.L1Loss()

    model = RNNRegressor(d_in=d, hidden_size=H).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_val_mse = float("inf")
    best_state = None

    print(f"\n=== Running TTT config: K={K}, H={H} ===")
    for epoch in range(1, num_epochs + 1):
        train_mse = train_one_epoch(model, train_loader, optimizer, criterion_mse, device)
        val_mse = eval_epoch(model, val_loader, criterion_mse, device)
        val_mae = eval_epoch(model, val_loader, criterion_mae, device)
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        print(
            f"TTT (single) | K={K} | H={H} | Epoch {epoch:02d} | "
            f"train_MSE={train_mse:.6f} | val_MSE={val_mse:.6f} | "
            f"val_MAE={val_mae:.6f} | best_val_MSE={best_val_mse:.6f}"
        )

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    # Base test metrics (no TTT)
    base_test_mse = eval_epoch(model, base_test_loader, criterion_mse, device)
    base_test_mae = eval_epoch(model, base_test_loader, criterion_mae, device)

    # TTT metrics on full test trajectory (separate from sliding-window test_ds)
    ttt_mse, ttt_mae = evaluate_with_ttt(
        model, test_series, K, device, adapt_lr=adapt_lr, adapt_steps=adapt_steps
    )

    print(
        f">> SUMMARY (K={K}, H={H}) | "
        f"val_best_MSE={best_val_mse:.6f} | "
        f"base_test_MSE={base_test_mse:.6f} | base_test_MAE={base_test_mae:.6f} | "
        f"TTT_test_MSE={ttt_mse:.6f} | TTT_test_MAE={ttt_mae:.6f}"
    )

    return {
        "K": K,
        "H": H,
        "val_best_MSE": float(best_val_mse),
        "base_test_MSE": float(base_test_mse),
        "base_test_MAE": float(base_test_mae),
        "TTT_test_MSE": float(ttt_mse),
        "TTT_test_MAE": float(ttt_mae),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run a single TTT configuration (K, H).")
    parser.add_argument("--K", type=int, required=True, help="History length K.")
    parser.add_argument("--H", type=int, required=True, help="Hidden size H.")
    parser.add_argument(
        "--num_epochs", type=int, default=20, help="Number of training epochs."
    )
    parser.add_argument(
        "--batch_size", type=int, default=256, help="Training batch size."
    )
    parser.add_argument(
        "--adapt_lr", type=float, default=1e-4, help="TTT adaptation learning rate."
    )
    parser.add_argument(
        "--adapt_steps",
        type=int,
        default=1,
        help="Number of gradient steps per TTT step.",
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default="../../robotic_arm_dataset_multiple_trajectories.csv",
        help="Path to the robotic arm CSV.",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="log",
        help="Directory to store stdout logs when used with a launcher script.",
    )

    args = parser.parse_args()

    # Just run; logging to files is handled by the launcher via stdout redirection.
    run_config(
        K=args.K,
        H=args.H,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        adapt_lr=args.adapt_lr,
        adapt_steps=args.adapt_steps,
        csv_path=args.csv_path,
    )

