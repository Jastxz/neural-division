"""Neural Division Ensemble — combine partial subnetworks.

Each partial subnetwork specializes in distinguishing a subset of classes.
Combine predictions via voting/averaging for the full problem.

Usage:
    /media/javier/intenso-tools/home-moved/venvs/lazy-training-venv/bin/python python/exp_ensemble.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from itertools import combinations
from sklearn.datasets import load_wine, load_iris
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
    return X, y, 6, ["RI","Na","Mg","Al","Si","K","Ca","Ba","Fe"]


def load_seeds():
    X, y = [], []
    with open("csv/seeds.csv") as f:
        for line in f:
            if not line.strip(): continue
            parts = line.split()
            X.append([float(v) for v in parts[:7]])
            y.append(int(parts[7]) - 1)  # 0-indexed
    return np.array(X), np.array(y), 3, ["Area","Perim","Compact","KernLen","KernWid","Asym","Groove"]


def train_specialist(n_in, n_hidden, n_out, x_train, y_train, x_val, y_val,
                     epochs=300, lr=0.001, device="cuda", seed=42):
    """Train a specialist network for a subset of classes."""
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(n_in, n_hidden), nn.ReLU(), nn.Linear(n_hidden, n_out)).to(device)
    x_tr, y_tr = x_train.to(device), y_train.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(x_tr, y_tr), batch_size=64, shuffle=True)
    best_loss, patience = float('inf'), 0
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            vl = criterion(model(x_val.to(device)), y_val.to(device)).item()
        if vl < best_loss - 1e-5:
            best_loss = vl
            patience = 0
        else:
            patience += 1
            if patience >= 30: break
    return model


def ensemble_predict(specialists, x, n_classes, device="cuda"):
    """Combine specialist predictions via score accumulation.

    Each specialist votes for its classes with its softmax probabilities.
    """
    scores = torch.zeros(x.shape[0], n_classes, device=device)
    x_dev = x.to(device)

    for model, class_indices, input_indices in specialists:
        model.eval()
        with torch.no_grad():
            out = model(x_dev[:, input_indices])
            probs = torch.softmax(out, dim=1)  # (N, n_specialist_classes)
            for local_idx, global_idx in enumerate(class_indices):
                scores[:, global_idx] += probs[:, local_idx]

    return scores.argmax(dim=1)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print("=" * 60)
    print("  Exp. Ensemble: combining partial subnetworks")
    print("=" * 60)

    seeds = [42, 123, 7, 2024, 31415, 1, 99, 256, 777, 1337]
    n_hidden = 16

    for name, load_fn in [("Glass", load_glass), ("Seeds", load_seeds)]:
        X, y, n_classes, feat_names = load_fn()
        X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        sc = StandardScaler().fit(X_tr)
        X_tr, X_va = sc.transform(X_tr), sc.transform(X_va)

        x_tr = torch.tensor(X_tr, dtype=torch.float32)
        y_tr_t = torch.tensor(y_tr, dtype=torch.long)
        x_va = torch.tensor(X_va, dtype=torch.float32)
        y_va_t = torch.tensor(y_va, dtype=torch.long)
        n_feat = x_tr.shape[1]

        print(f"\n--- {name} ({n_feat} feat, {n_classes} classes) ---\n")
        print(f"{'Seed':>6} | {'Ref':>6} | {'Ensemble':>8} | {'Best partial':>12} | {'Δ(ens-ref)':>10}")
        print("-" * 55)

        ref_accs = []
        ens_accs = []

        for seed in seeds:
            # Train full reference
            ref_model = train_specialist(n_feat, n_hidden * 2, n_classes, x_tr, y_tr_t, x_va, y_va_t,
                                         epochs=500, device=device, seed=seed)
            ref_model.eval()
            with torch.no_grad():
                ref_acc = (ref_model(x_va.to(device)).argmax(1) == y_va_t.to(device)).float().mean().item()

            # Train one-vs-rest specialists (each class gets a binary specialist)
            # Plus pairwise specialists for top combinations
            specialists = []

            # One-vs-all: each class
            for c in range(n_classes):
                mask_tr = (y_tr_t == c) | True  # all samples, binary target
                y_binary_tr = (y_tr_t == c).long()
                y_binary_va = (y_va_t == c).long()
                model = train_specialist(n_feat, n_hidden, 2, x_tr, y_binary_tr, x_va, y_binary_va,
                                         epochs=200, device=device, seed=seed + c)
                # This specialist predicts "class c or not" — use as score for class c
                specialists.append((model, [c], list(range(n_feat)), "binary"))

            # Ensemble prediction: for each sample, each specialist gives P(class=c)
            # Use the "yes" probability from each binary specialist
            scores = torch.zeros(x_va.shape[0], n_classes, device=device)
            x_va_dev = x_va.to(device)
            for model, class_indices, _, _ in specialists:
                model.eval()
                with torch.no_grad():
                    out = torch.softmax(model(x_va_dev), dim=1)
                    # Column 1 = P(yes this class)
                    scores[:, class_indices[0]] += out[:, 1]

            ens_pred = scores.argmax(dim=1).cpu()
            ens_acc = (ens_pred == y_va_t).float().mean().item()

            # Best individual partial (one-vs-rest)
            best_partial = 0.0
            for model, cidx, _, _ in specialists:
                model.eval()
                with torch.no_grad():
                    p = torch.softmax(model(x_va_dev), dim=1)[:, 1]
                    partial_acc = ((p > 0.5).long() == (y_va_t.to(device) == cidx[0]).long()).float().mean().item()
                    best_partial = max(best_partial, partial_acc)

            ref_accs.append(ref_acc)
            ens_accs.append(ens_acc)
            delta = (ens_acc - ref_acc) * 100

            print(f"{seed:>6} | {ref_acc*100:>5.1f}% | {ens_acc*100:>7.1f}% | {best_partial*100:>11.1f}% | {delta:>+9.1f} pp")

        print("-" * 55)
        print(f"{'Mean':>6} | {np.mean(ref_accs)*100:>5.1f}% | {np.mean(ens_accs)*100:>7.1f}% |              | {(np.mean(ens_accs)-np.mean(ref_accs))*100:>+9.1f} pp")


if __name__ == "__main__":
    main()
