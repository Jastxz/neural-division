"""Exp 5: Comparison with Lottery Ticket Hypothesis (iterative magnitude pruning).

Simple LTH: train → prune smallest X% weights → reset to init → retrain → repeat.
Compare with Neural Division on same datasets.

Usage:
    /media/javier/intenso-tools/home-moved/venvs/lazy-training-venv/bin/python python/exp_lth_comparison.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import csv
import copy


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


def train_model(model, x_tr, y_tr, x_va, y_va, epochs=300, lr=0.001, device="cuda"):
    """Train and return test accuracy."""
    x_tr, y_tr = x_tr.to(device), y_tr.to(device)
    x_va, y_va = x_va.to(device), y_va.to(device)
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
            vl = criterion(model(x_va), y_va).item()
        if vl < best_loss - 1e-5:
            best_loss = vl
            patience = 0
        else:
            patience += 1
            if patience >= 50: break
    model.eval()
    with torch.no_grad():
        return (model(x_va).argmax(1) == y_va).float().mean().item()


def count_nonzero_params(model):
    """Count non-zero parameters."""
    total = 0
    nonzero = 0
    for p in model.parameters():
        total += p.numel()
        nonzero += (p != 0).sum().item()
    return nonzero, total


def lth_prune(model, prune_pct=0.2):
    """Prune smallest magnitude weights globally. Returns mask."""
    all_weights = []
    for name, p in model.named_parameters():
        if 'weight' in name:
            all_weights.append(p.data.abs().flatten())
    all_w = torch.cat(all_weights)
    threshold = torch.quantile(all_w[all_w > 0], prune_pct)

    masks = {}
    for name, p in model.named_parameters():
        if 'weight' in name:
            mask = (p.data.abs() >= threshold).float()
            p.data *= mask
            masks[name] = mask
    return masks


def apply_masks(model, masks):
    """Zero-out pruned weights."""
    for name, p in model.named_parameters():
        if name in masks:
            p.data *= masks[name]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print("=" * 60)
    print("  Exp 5: Neural Division vs Lottery Ticket Hypothesis")
    print("=" * 60)

    seeds = [42, 123, 7, 2024, 31415]

    for name, load_fn in [("Wine", lambda: (*load_wine(return_X_y=True), 3)),
                           ("Glass", load_glass)]:
        X, y, n_classes = load_fn()
        X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        sc = StandardScaler().fit(X_tr)
        X_tr, X_va = sc.transform(X_tr), sc.transform(X_va)
        x_tr = torch.tensor(X_tr, dtype=torch.float32)
        y_tr_t = torch.tensor(y_tr, dtype=torch.long)
        x_va = torch.tensor(X_va, dtype=torch.float32)
        y_va_t = torch.tensor(y_va, dtype=torch.long)
        n_feat = x_tr.shape[1]
        n_hidden = 32

        print(f"\n--- {name} ({n_feat}→{n_hidden}→{n_classes}) ---\n")
        print(f"{'Method':<20} | {'Acc%':>6} | {'Params':>8} | {'Sparsity':>8}")
        print("-" * 55)

        # Full network (baseline)
        full_accs = []
        for seed in seeds:
            torch.manual_seed(seed)
            model = nn.Sequential(nn.Linear(n_feat, n_hidden), nn.ReLU(), nn.Linear(n_hidden, n_classes)).to(device)
            acc = train_model(model, x_tr, y_tr_t, x_va, y_va_t, device=device)
            full_accs.append(acc)
            nz, total = count_nonzero_params(model)

        print(f"{'Full network':<20} | {np.mean(full_accs)*100:>5.1f}% | {total:>8d} | {'0%':>8}")

        # LTH: iterative pruning at 20%, 40%, 60%, 80%
        for target_sparsity in [0.2, 0.4, 0.6, 0.8]:
            lth_accs = []
            lth_params = []
            for seed in seeds:
                torch.manual_seed(seed)
                model = nn.Sequential(nn.Linear(n_feat, n_hidden), nn.ReLU(), nn.Linear(n_hidden, n_classes)).to(device)
                init_state = copy.deepcopy(model.state_dict())

                # Iterative pruning: prune 20% each round until target
                n_rounds = max(1, int(target_sparsity / 0.2))
                masks = {}
                for _ in range(n_rounds):
                    # Train
                    train_model(model, x_tr, y_tr_t, x_va, y_va_t, epochs=200, device=device)
                    # Prune
                    masks = lth_prune(model, prune_pct=0.2)
                    # Reset to init with mask
                    model.load_state_dict(init_state)
                    apply_masks(model, masks)

                # Final train with mask
                acc = train_model(model, x_tr, y_tr_t, x_va, y_va_t, epochs=300, device=device)
                nz, total = count_nonzero_params(model)
                lth_accs.append(acc)
                lth_params.append(nz)

            sparsity_actual = 1 - np.mean(lth_params) / total
            print(f"{'LTH ' + f'{target_sparsity*100:.0f}%':<20} | {np.mean(lth_accs)*100:>5.1f}% | {int(np.mean(lth_params)):>8d} | {sparsity_actual*100:>6.0f}%")

        # Neural Division result (from Julia experiments)
        div_acc = {"Wine": 93.3, "Glass": 94.4}
        div_neur = {"Wine": 21, "Glass": 18}
        # Approximate params: for subnetwork with k inputs, params ≈ k*hidden + hidden*n_out + biases
        k_feat = {"Wine": 4, "Glass": 1}
        div_params = k_feat[name] * n_hidden + n_hidden * n_classes + n_hidden + n_classes
        print(f"{'Neural Division':<20} | {div_acc[name]:>5.1f}% | {div_params:>8d} | {'N/A':>8}")
        print(f"  (from Julia: {k_feat[name]} features, {div_neur[name]} neurons)")


if __name__ == "__main__":
    main()
