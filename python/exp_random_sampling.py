"""Exp 6: Random sampling for high-dimensional feature selection.

Instead of exhaustive enumeration, sample N random subsets of features.
Test on Ionosphere (34 features) where exhaustive is infeasible.

Usage:
    /media/javier/intenso-tools/home-moved/venvs/lazy-training-venv/bin/python python/exp_random_sampling.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import csv
import time


def load_ionosphere():
    X, y = [], []
    with open("csv/ionosphere.csv") as f:
        for row in csv.reader(f):
            if not row or len(row) < 35: continue
            try:
                feats = [float(v) for v in row[:34]]
                label = 1 if row[34].strip() == 'g' else 0
                X.append(feats)
                y.append(label)
            except ValueError:
                continue
    return np.array(X), np.array(y), 2


def load_sonar():
    X, y = [], []
    with open("csv/sonar.csv") as f:
        for row in csv.reader(f):
            if not row or len(row) < 61: continue
            try:
                feats = [float(v) for v in row[:60]]
                label = 1 if row[60].strip() == 'M' else 0
                X.append(feats)
                y.append(label)
            except ValueError:
                continue
    return np.array(X), np.array(y), 2


@torch.no_grad()
def evaluate_random_subconfig(w0, b0, w1, b1, x_val, y_val, input_idx):
    """Forward pass with random weights on feature subset."""
    x = x_val[:, input_idx]
    h = torch.relu(x @ w0[input_idx, :] + b0)
    out = torch.sigmoid(h @ w1 + b1).squeeze()
    pred = (out > 0.5).long()
    return (pred == y_val).float().mean().item()


def train_subset(input_idx, n_hidden, x_tr, y_tr, x_va, y_va, epochs=300, device="cuda", seed=42):
    """Train binary classifier on feature subset."""
    torch.manual_seed(seed)
    n_in = len(input_idx)
    model = nn.Sequential(nn.Linear(n_in, n_hidden), nn.ReLU(), nn.Linear(n_hidden, 2)).to(device)
    xt = x_tr[:, input_idx].to(device)
    yt = y_tr.to(device)
    xv = x_va[:, input_idx].to(device)
    yv = y_va.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(xt, yt), batch_size=64, shuffle=True)
    best_loss, patience = float('inf'), 0
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            vl = criterion(model(xv), yv).item()
        if vl < best_loss - 1e-5:
            best_loss = vl
            patience = 0
        else:
            patience += 1
            if patience >= 30: break
    model.eval()
    with torch.no_grad():
        return (model(xv).argmax(1) == yv).float().mean().item()


def random_exploration(n_feat, n_hidden, x_val, y_val, n_samples=5000,
                       size_range=(2, 15), threshold=0.5, seed=42):
    """Sample random feature subsets and evaluate with random weights."""
    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)
    w0 = torch.randn(n_feat, n_hidden)
    b0 = torch.randn(n_hidden)
    w1 = torch.randn(n_hidden, 1)
    b1 = torch.randn(1)

    best_acc = 0.0
    best_idx = None

    for _ in range(n_samples):
        k = rng.randint(size_range[0], size_range[1] + 1)
        idx = sorted(rng.choice(n_feat, size=k, replace=False).tolist())
        acc = evaluate_random_subconfig(w0, b0, w1, b1, x_val, y_val, idx)
        if acc > best_acc:
            best_acc = acc
            best_idx = idx

    return best_acc, best_idx


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print("=" * 60)
    print("  Exp 6: Random sampling for high-dim feature selection")
    print("=" * 60)

    seeds = [42, 123, 7, 2024, 31415]
    n_hidden = 32
    n_samples_list = [1000, 5000, 10000]

    for name, load_fn in [("Ionosphere (34 feat)", load_ionosphere),
                           ("Sonar (60 feat)", load_sonar)]:
        X, y, n_classes = load_fn()
        n_feat = X.shape[1]
        X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        sc = StandardScaler().fit(X_tr)
        X_tr, X_va = sc.transform(X_tr), sc.transform(X_va)
        x_tr = torch.tensor(X_tr, dtype=torch.float32)
        y_tr_t = torch.tensor(y_tr, dtype=torch.long)
        x_va = torch.tensor(X_va, dtype=torch.float32)
        y_va_t = torch.tensor(y_va, dtype=torch.long)

        print(f"\n--- {name} ---\n")

        # Reference accuracy
        ref_accs = []
        for seed in seeds:
            acc = train_subset(list(range(n_feat)), n_hidden, x_tr, y_tr_t, x_va, y_va_t, device=device, seed=seed)
            ref_accs.append(acc)
        print(f"  Reference (all {n_feat} features): {np.mean(ref_accs)*100:.1f}% ± {np.std(ref_accs)*100:.1f}%\n")

        print(f"{'N samples':>10} | {'Explore acc':>11} | {'Features':>8} | {'Trained acc':>11} | {'Δ vs ref':>8} | {'Time':>5}")
        print("-" * 70)

        for n_samples in n_samples_list:
            explore_accs = []
            trained_accs = []
            feat_counts = []
            times = []

            for seed in seeds:
                t0 = time.time()
                exp_acc, best_idx = random_exploration(n_feat, n_hidden, x_va, y_va_t,
                    n_samples=n_samples, size_range=(2, min(15, n_feat//2)), seed=seed)
                t_exp = time.time() - t0

                # Train the best found subset
                if best_idx:
                    tr_acc = train_subset(best_idx, n_hidden, x_tr, y_tr_t, x_va, y_va_t, device=device, seed=seed)
                else:
                    tr_acc = 0.0
                    best_idx = list(range(n_feat))

                explore_accs.append(exp_acc)
                trained_accs.append(tr_acc)
                feat_counts.append(len(best_idx))
                times.append(t_exp)

            delta = (np.mean(trained_accs) - np.mean(ref_accs)) * 100
            print(f"{n_samples:>10} | {np.mean(explore_accs)*100:>10.1f}% | {np.mean(feat_counts):>7.1f} | {np.mean(trained_accs)*100:>10.1f}% | {delta:>+7.1f} | {np.mean(times):>4.1f}s")

        # Feature frequency
        print(f"\n  Most selected features (10K samples, 5 seeds):")
        freq = np.zeros(n_feat)
        for seed in seeds:
            _, idx = random_exploration(n_feat, n_hidden, x_va, y_va_t,
                n_samples=10000, size_range=(2, min(15, n_feat//2)), seed=seed)
            if idx:
                for i in idx: freq[i] += 1
        top = np.argsort(freq)[::-1][:10]
        for i in top:
            if freq[i] > 0:
                print(f"    Feature {i:>2}: {int(freq[i])}/5")


if __name__ == "__main__":
    main()
