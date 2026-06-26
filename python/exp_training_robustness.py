"""Exp 4: Effect of training budget on Division vs Reference.

Question: at what training budget does the reference overtake subnetworks?

Usage:
    /media/javier/intenso-tools/home-moved/venvs/lazy-training-venv/bin/python python/exp_training_robustness.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import csv


def load_glass():
    X, y = [], []
    with open("csv/glass.csv") as f:
        for row in csv.reader(f):
            if not row: continue
            X.append([float(v) for v in row[1:10]])
            y.append(int(row[10]))
    X, y = np.array(X), np.array(y)
    mapping = {1:0, 2:1, 3:2, 5:3, 6:4, 7:5}
    y = np.array([mapping[c] for c in y])
    return X, y, 6


def train_with_budget(n_in, n_hidden, n_out, x_train, y_train, x_val, y_val,
                      max_epochs, lr=0.001, device="cuda", seed=42):
    """Train for exactly max_epochs (no early stopping). Return accuracy at checkpoints."""
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(n_in, n_hidden), nn.ReLU(), nn.Linear(n_hidden, n_out)).to(device)
    x_tr, y_tr = x_train.to(device), y_train.to(device)
    x_va, y_va = x_val.to(device), y_val.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(x_tr, y_tr), batch_size=64, shuffle=True)

    checkpoints = {}
    check_at = [10, 25, 50, 100, 200, 500, 1000, 2000]

    for epoch in range(1, max_epochs + 1):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            optimizer.step()
        if epoch in check_at:
            model.eval()
            with torch.no_grad():
                acc = (model(x_va).argmax(1) == y_va).float().mean().item()
            checkpoints[epoch] = acc

    return checkpoints


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print("=" * 60)
    print("  Exp 4: Training budget effect (Glass)")
    print("  How many epochs until reference overtakes subnetwork?")
    print("=" * 60)

    X, y, n_classes = load_glass()
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    sc = StandardScaler().fit(X_tr)
    X_tr, X_va = sc.transform(X_tr), sc.transform(X_va)

    x_tr = torch.tensor(X_tr, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.long)
    x_va = torch.tensor(X_va, dtype=torch.float32)
    y_va_t = torch.tensor(y_va, dtype=torch.long)

    n_feat = 9
    n_hidden = 16
    seeds = [42, 123, 7, 2024, 31415]

    # Subnetwork: use only feature 0 (Refractive Index) — the dominant one from experiments
    sub_feat = [0]  # RI
    n_sub_in = len(sub_feat)

    epochs_list = [10, 25, 50, 100, 200, 500, 1000, 2000]

    print(f"\nReference: 9→16→6 | Subnetwork: 1→16→6 (RI only)")
    print(f"Seeds: {len(seeds)}\n")

    print(f"{'Epochs':>6} | {'Ref mean':>8} | {'Sub mean':>8} | {'Δ':>8} | {'Winner':>8}")
    print("-" * 50)

    for max_ep in epochs_list:
        ref_accs = []
        sub_accs = []
        for seed in seeds:
            # Reference: all features
            ref_ckpts = train_with_budget(n_feat, n_hidden, n_classes,
                x_tr, y_tr_t, x_va, y_va_t, max_ep, device=device, seed=seed)
            ref_accs.append(ref_ckpts.get(max_ep, 0))

            # Subnetwork: RI only
            sub_ckpts = train_with_budget(n_sub_in, n_hidden, n_classes,
                x_tr[:, sub_feat], y_tr_t, x_va[:, sub_feat], y_va_t, max_ep, device=device, seed=seed)
            sub_accs.append(sub_ckpts.get(max_ep, 0))

        ref_mean = np.mean(ref_accs) * 100
        sub_mean = np.mean(sub_accs) * 100
        delta = sub_mean - ref_mean
        winner = "SUB" if delta > 0 else "REF" if delta < -1 else "TIE"

        print(f"{max_ep:>6} | {ref_mean:>7.1f}% | {sub_mean:>7.1f}% | {delta:>+7.1f} | {winner:>8}")

    print("\nInterpretation:")
    print("  When Sub > Ref: subnetwork generalizes better (regularization effect)")
    print("  When Ref > Sub: full network has converged, more data helps")


if __name__ == "__main__":
    main()
