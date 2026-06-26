"""Neural Division for Regression — California Housing.

8 features, 1 output (continuous), 20640 samples.
Metric: R² score instead of accuracy.
Exhaustive exploration feasible: (2^8-1)×1 = 255 subconfigurations.

Usage:
    /media/javier/intenso-tools/home-moved/venvs/lazy-training-venv/bin/python python/exp_regression.py
"""

import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from itertools import combinations
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def r2_score(y_true, y_pred):
    """R² score."""
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    return 1 - (ss_res / ss_tot)


@torch.no_grad()
def evaluate_subconfig_regression(w0, b0, w1, b1, x_val, y_val, input_idx):
    """Forward pass with random weights, return R² (can be negative)."""
    x = x_val[:, input_idx]
    h = torch.relu(x @ w0[input_idx, :] + b0)
    out = (h @ w1 + b1).squeeze()
    return r2_score(y_val, out).item()


def train_subnet_regression(n_in, n_hidden, x_train, y_train, x_val, y_val,
                             epochs=500, lr=0.001, batch_size=256, device="cuda", seed=42):
    """Train regression subnetwork. Returns (pre_r2, post_r2)."""
    torch.manual_seed(seed)
    model = nn.Sequential(
        nn.Linear(n_in, n_hidden), nn.ReLU(),
        nn.Linear(n_hidden, 1),
    ).to(device)

    x_tr = x_train.to(device)
    y_tr = y_train.to(device)
    x_va = x_val.to(device)
    y_va = y_val.to(device)

    # Pre R²
    model.eval()
    pre_r2 = r2_score(y_va, model(x_va).squeeze()).item()

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(TensorDataset(x_tr, y_tr), batch_size=batch_size, shuffle=True)

    best_loss, patience, max_patience = float('inf'), 0, 50
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb).squeeze(), yb)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(x_va).squeeze(), y_va).item()
        if val_loss < best_loss - 1e-5:
            best_loss = val_loss
            patience = 0
        else:
            patience += 1
            if patience >= max_patience:
                break

    model.eval()
    post_r2 = r2_score(y_va, model(x_va).squeeze()).item()
    return pre_r2, post_r2


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print("=" * 60)
    print("  Exp. Regression: California Housing")
    print("  8 features → precio mediano de vivienda")
    print("=" * 60)

    # Load data
    data = fetch_california_housing()
    X, y = data.data, data.target
    feature_names = data.feature_names
    # ponytail: subsample for speed (5K train, 1K val)
    X_sub, _, y_sub, _ = train_test_split(X, y, train_size=6000, random_state=42, shuffle=True)
    X_train, X_val, y_train, y_val = train_test_split(X_sub, y_sub, test_size=0.2, random_state=42)

    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)

    x_train = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    x_val = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)

    n_feat = 8
    n_hidden = 32
    seeds = [42, 123, 7, 2024, 31415, 1, 99, 256, 777, 1337]

    print(f"\nTrain: {x_train.shape[0]}, Val: {x_val.shape[0]}")
    print(f"Network: 8→{n_hidden}→1 | Subconfiguraciones: 255\n")

    # --- Exploration ---
    print("--- Exploration (CPU, R² metric) ---\n")

    print(f"{'Seed':>6} | {'Ref R²':>7} | {'Sub R²':>7} | {'Sub feat':>8} | {'Selected':>10}")
    print("-" * 55)

    ref_r2s = []
    sub_r2s = []
    sub_feats_all = []
    selections = []

    for seed in seeds:
        torch.manual_seed(seed)
        w0 = torch.randn(n_feat, n_hidden)
        b0 = torch.randn(n_hidden)
        w1 = torch.randn(n_hidden, 1)
        b1 = torch.randn(1)

        # Reference
        ref_r2_explore = evaluate_subconfig_regression(w0, b0, w1, b1, x_val, y_val_t, list(range(n_feat)))

        # Explore all subsets (255)
        best_sub = None
        best_r2 = -float('inf')
        best_idx = None
        for n_in in range(1, n_feat + 1):
            for combo in combinations(range(n_feat), n_in):
                idx = list(combo)
                r2 = evaluate_subconfig_regression(w0, b0, w1, b1, x_val, y_val_t, idx)
                n_active = n_in + n_hidden + 1
                # Simpler is better if R² similar
                if r2 > best_r2 or (r2 > best_r2 - 0.01 and n_in < (len(best_idx) if best_idx else n_feat)):
                    best_r2 = r2
                    best_idx = idx

        # Train reference
        ref_pre, ref_post = train_subnet_regression(n_feat, n_hidden,
            x_train, y_train_t, x_val, y_val_t,
            epochs=500, lr=0.001, device=device, seed=seed)

        # Train best subnetwork
        if best_idx and len(best_idx) < n_feat:
            sub_pre, sub_post = train_subnet_regression(len(best_idx), n_hidden,
                x_train[:, best_idx], y_train_t, x_val[:, best_idx], y_val_t,
                epochs=500, lr=0.001, device=device, seed=seed)
        else:
            sub_post = ref_post
            best_idx = list(range(n_feat))

        # Score selection (80% R², 20% efficiency)
        max_neur = n_feat + n_hidden + 1
        sub_neur = len(best_idx) + n_hidden + 1
        ref_score = 0.8 * max(ref_post, 0)
        sub_score = 0.8 * max(sub_post, 0) + 0.2 * (1 - sub_neur / max_neur)
        selected = "subnetwork" if sub_score > ref_score else "reference"

        ref_r2s.append(ref_post)
        sub_r2s.append(sub_post)
        sub_feats_all.append(best_idx)
        selections.append(selected)

        feat_str = ",".join(feature_names[i][:5] for i in best_idx) if len(best_idx) <= 4 else f"{len(best_idx)} feat"
        print(f"{seed:>6} | {ref_post:>6.3f} | {sub_post:>6.3f} | {feat_str:>8} | {selected:>10}")

    # Summary
    print("-" * 55)
    print(f"{'Mean':>6} | {np.mean(ref_r2s):>6.3f} | {np.mean(sub_r2s):>6.3f} |          | ref:{selections.count('reference')}")
    print(f"{'Std':>6} | {np.std(ref_r2s):>6.3f} | {np.std(sub_r2s):>6.3f} |          | sub:{selections.count('subnetwork')}")

    # Feature frequency
    print("\n  Feature frequency in subnetworks:")
    freq = np.zeros(n_feat)
    n_sub = 0
    for idx in sub_feats_all:
        if len(idx) < n_feat:
            for i in idx:
                freq[i] += 1
            n_sub += 1
    if n_sub > 0:
        order = np.argsort(freq)[::-1]
        for i in order:
            if freq[i] > 0:
                print(f"    {feature_names[i]:>12}: {int(freq[i])}/{n_sub} ({freq[i]/n_sub*100:.0f}%)")


if __name__ == "__main__":
    main()
