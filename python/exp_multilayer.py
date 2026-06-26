"""Neural Division with multi-layer networks (2 hidden layers).

Tests if deeper networks benefit from division, and if Monks-2 becomes solvable.

Usage:
    /media/javier/intenso-tools/home-moved/venvs/lazy-training-venv/bin/python python/exp_multilayer.py
"""

import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from itertools import combinations
from sklearn.datasets import load_wine, load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@torch.no_grad()
def evaluate_subconfig_2layer(w0, b0, w1, b1, w2, b2, x_val, y_val, input_idx, output_idx):
    """Forward pass with 2 hidden layers, random weights."""
    x = x_val[:, input_idx]
    h1 = torch.relu(x @ w0[input_idx, :] + b0)
    h2 = torch.relu(h1 @ w1 + b1)
    out = h2 @ w2[:, output_idx] + b2[output_idx]
    if len(output_idx) == 1:
        pred = (torch.sigmoid(out.squeeze()) > 0.5).long()
        target = (y_val == output_idx[0]).long()
        return (pred == target).float().mean().item()
    else:
        pred_local = out.argmax(dim=1)
        pred_global = torch.tensor([output_idx[i] for i in pred_local.tolist()])
        return (pred_global == y_val).float().mean().item()


def explore_2layer(n_in, n_out, h1_size, h2_size, x_val, y_val, threshold=0.4,
                   input_range=None, seed=42):
    """Explore with 2 hidden layers."""
    torch.manual_seed(seed)
    w0 = torch.randn(n_in, h1_size)
    b0 = torch.randn(h1_size)
    w1 = torch.randn(h1_size, h2_size)
    b1 = torch.randn(h2_size)
    w2 = torch.randn(h2_size, n_out)
    b2 = torch.randn(n_out)

    all_in = list(range(n_in))
    all_out = list(range(n_out))
    min_in, max_in = input_range if input_range else (1, n_in)

    ref_acc = evaluate_subconfig_2layer(w0, b0, w1, b1, w2, b2, x_val, y_val, all_in, all_out)

    best_sub_acc = 0.0
    best_sub_idx = None
    best_sub_out = None

    n_explored = 0
    for n_i in range(min_in, max_in + 1):
        for in_combo in combinations(all_in, n_i):
            in_idx = list(in_combo)
            for n_o in range(1, n_out + 1):
                for out_combo in combinations(all_out, n_o):
                    out_idx = list(out_combo)
                    acc = evaluate_subconfig_2layer(w0, b0, w1, b1, w2, b2, x_val, y_val, in_idx, out_idx)
                    n_explored += 1
                    if acc > threshold:
                        n_active = len(in_idx) + h1_size + h2_size + len(out_idx)
                        if best_sub_idx is None or len(in_idx) < len(best_sub_idx) or acc > best_sub_acc:
                            best_sub_acc = acc
                            best_sub_idx = in_idx
                            best_sub_out = out_idx

    return ref_acc, best_sub_acc, best_sub_idx, best_sub_out, n_explored


def train_2layer(n_in, h1, h2, n_out, x_train, y_train, x_val, y_val,
                 epochs=500, lr=0.001, batch_size=128, device="cuda", seed=42):
    """Train 2-hidden-layer network."""
    torch.manual_seed(seed)
    model = nn.Sequential(
        nn.Linear(n_in, h1), nn.ReLU(),
        nn.Linear(h1, h2), nn.ReLU(),
        nn.Linear(h2, n_out),
    ).to(device)

    x_tr, y_tr = x_train.to(device), y_train.to(device)
    x_va, y_va = x_val.to(device), y_val.to(device)

    model.eval()
    pre_acc = (model(x_va).argmax(1) == y_va).float().mean().item()

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(x_tr, y_tr), batch_size=batch_size, shuffle=True)

    best_loss, patience = float('inf'), 0
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            vl = criterion(model(x_va), y_va).item()
        if vl < best_loss - 1e-5:
            best_loss = vl
            patience = 0
        else:
            patience += 1
            if patience >= 50:
                break

    model.eval()
    post_acc = (model(x_va).argmax(1) == y_va).float().mean().item()
    return pre_acc, post_acc


def load_glass():
    """Glass dataset."""
    import csv
    path = "csv/glass.csv"
    X, y = [], []
    with open(path) as f:
        for row in csv.reader(f):
            if not row:
                continue
            X.append([float(v) for v in row[1:10]])
            y.append(int(row[10]))
    X, y = np.array(X), np.array(y)
    # Remap classes 1,2,3,5,6,7 → 0,1,2,3,4,5
    mapping = {1:0, 2:1, 3:2, 5:3, 6:4, 7:5}
    y = np.array([mapping[c] for c in y])
    return X, y, 6


def load_monks2():
    """Monks-2 dataset."""
    def parse_monks(path):
        X, y = [], []
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                parts = line.split()
                y.append(int(parts[0]))
                X.append([float(parts[i]) for i in range(1, 7)])
        return np.array(X), np.array(y)
    X_tr, y_tr = parse_monks("csv/monks2_train.csv")
    X_te, y_te = parse_monks("csv/monks2_test.csv")
    return X_tr, y_tr, X_te, y_te, 2


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print("=" * 60)
    print("  Exp. Multi-layer: 2 hidden layers")
    print("=" * 60)

    seeds = [42, 123, 7, 2024, 31415, 1, 99, 256, 777, 1337]
    h1, h2 = 32, 16  # Two hidden layers

    # --- Glass ---
    print("\n--- Glass (9→32→16→6) ---\n")
    X_g, y_g, n_c_g = load_glass()
    X_tr_g, X_va_g, y_tr_g, y_va_g = train_test_split(X_g, y_g, test_size=0.2, random_state=42, stratify=y_g)
    sc_g = StandardScaler().fit(X_tr_g)
    X_tr_g, X_va_g = sc_g.transform(X_tr_g), sc_g.transform(X_va_g)

    x_tr_g = torch.tensor(X_tr_g, dtype=torch.float32)
    y_tr_g_t = torch.tensor(y_tr_g, dtype=torch.long)
    x_va_g = torch.tensor(X_va_g, dtype=torch.float32)
    y_va_g_t = torch.tensor(y_va_g, dtype=torch.long)

    print(f"{'Seed':>6} | {'Ref':>6} | {'Sub(explore)':>12} | {'Ref(train)':>10}")
    print("-" * 50)
    for seed in seeds:
        ref_exp, sub_exp, sub_idx, sub_out, _ = explore_2layer(
            9, n_c_g, h1, h2, x_va_g, y_va_g_t, threshold=0.3, seed=seed)
        _, ref_post = train_2layer(9, h1, h2, n_c_g, x_tr_g, y_tr_g_t, x_va_g, y_va_g_t,
                                    epochs=500, device=device, seed=seed)
        feat_str = str(sub_idx) if sub_idx and len(sub_idx) <= 3 else f"{len(sub_idx) if sub_idx else 9} feat"
        print(f"{seed:>6} | {ref_exp:.3f} | {sub_exp:.3f} {feat_str:>12} | {ref_post*100:>9.1f}%")

    # --- Monks-2 ---
    print("\n--- Monks-2 (6→32→16→2, 'exactly 2 of 6') ---\n")
    X_tr_m, y_tr_m, X_te_m, y_te_m, n_c_m = load_monks2()
    # Normalize
    sc_m = StandardScaler().fit(X_tr_m)
    X_tr_m, X_te_m = sc_m.transform(X_tr_m), sc_m.transform(X_te_m)
    x_tr_m = torch.tensor(X_tr_m, dtype=torch.float32)
    y_tr_m_t = torch.tensor(y_tr_m, dtype=torch.long)
    x_te_m = torch.tensor(X_te_m, dtype=torch.float32)
    y_te_m_t = torch.tensor(y_te_m, dtype=torch.long)

    print(f"{'Seed':>6} | {'Ref(train)':>10} | {'1-layer ref':>11}")
    print("-" * 40)
    for seed in seeds:
        _, ref_post = train_2layer(6, h1, h2, n_c_m, x_tr_m, y_tr_m_t, x_te_m, y_te_m_t,
                                    epochs=1000, device=device, seed=seed)
        # Compare with 1 layer
        torch.manual_seed(seed)
        model_1l = nn.Sequential(nn.Linear(6, 32), nn.ReLU(), nn.Linear(32, 2)).to(device)
        optimizer = optim.Adam(model_1l.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        loader = DataLoader(TensorDataset(x_tr_m.to(device), y_tr_m_t.to(device)), batch_size=64, shuffle=True)
        for _ in range(1000):
            model_1l.train()
            for xb, yb in loader:
                optimizer.zero_grad()
                criterion(model_1l(xb), yb).backward()
                optimizer.step()
        model_1l.eval()
        acc_1l = (model_1l(x_te_m.to(device)).argmax(1) == y_te_m_t.to(device)).float().mean().item()
        print(f"{seed:>6} | {ref_post*100:>9.1f}% | {acc_1l*100:>10.1f}%")


if __name__ == "__main__":
    main()
