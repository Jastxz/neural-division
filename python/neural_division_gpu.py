"""Neural Division Method — PyTorch GPU implementation.

Exploration on CPU (forward pass with random weights), training on GPU.
Reuses the lazy-training pattern for GPU training of selected subnetworks.

Usage:
    python python/neural_division_gpu.py --dataset wine --hidden 32 --epochs 200
    python python/neural_division_gpu.py --dataset mnist --hidden 128 --epochs 30 --input-range 2,10
    python python/neural_division_gpu.py --dataset cifar10 --hidden 256 --epochs 30 --input-range 5,20

Requirements: torch, torchvision, scikit-learn (for UCI datasets)
"""

import argparse
import time
import sys
from itertools import combinations
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


# --- Types ---

@dataclass
class Subconfiguration:
    input_indices: List[int]
    output_indices: List[int]
    n_active: int

@dataclass
class MapEntry:
    subconfig: Optional[Subconfiguration]
    accuracy: float
    accuracy_pre: float
    accuracy_post: float


# --- Exploration (CPU, fast) ---

@torch.no_grad()
def evaluate_subconfig(w0, b0, w1, b1, x_val, y_val, input_idx, output_idx):
    """Single forward pass with random weights. Returns accuracy."""
    x = x_val[:, input_idx]
    h = torch.sigmoid(x @ w0[input_idx, :] + b0)
    out = h @ w1[:, output_idx] + b1[output_idx]
    # ponytail: argmax over output subset, map back to global class
    if len(output_idx) == 1:
        pred_local = (torch.sigmoid(out.squeeze()) > 0.5).long()
        correct = (y_val == output_idx[0]).long()
        return (pred_local == correct).float().mean().item()
    else:
        pred_local = out.argmax(dim=1)
        pred_global = torch.tensor([output_idx[i] for i in pred_local.tolist()])
        return (pred_global == y_val).float().mean().item()


def explore(n_in, n_out, n_hidden, x_val, y_val, threshold=0.4,
            input_range=None, seed=42):
    """Exhaustive exploration on CPU."""
    torch.manual_seed(seed)
    w0 = torch.randn(n_in, n_hidden)
    b0 = torch.randn(n_hidden)
    w1 = torch.randn(n_hidden, n_out)
    b1 = torch.randn(n_out)

    all_in = list(range(n_in))
    all_out = list(range(n_out))
    min_in, max_in = input_range if input_range else (1, n_in)

    # Reference
    ref_acc = evaluate_subconfig(w0, b0, w1, b1, x_val, y_val, all_in, all_out)
    ref = MapEntry(Subconfiguration(all_in, all_out, n_in + n_hidden + n_out),
                   ref_acc, 0.0, 0.0)

    global_ = MapEntry(None, 0.0, 0.0, 0.0)
    partials: Dict[tuple, MapEntry] = {}
    for r in range(1, n_out + 1):
        for c in combinations(all_out, r):
            partials[c] = MapEntry(None, 0.0, 0.0, 0.0)

    n_explored = 0
    for n_i in range(min_in, max_in + 1):
        for in_combo in combinations(all_in, n_i):
            in_idx = list(in_combo)
            for n_o in range(1, n_out + 1):
                for out_combo in combinations(all_out, n_o):
                    out_idx = list(out_combo)
                    n_active = n_i + n_hidden + n_o
                    acc = evaluate_subconfig(w0, b0, w1, b1, x_val, y_val, in_idx, out_idx)
                    n_explored += 1
                    if acc <= threshold:
                        continue
                    sc = Subconfiguration(in_idx, out_idx, n_active)
                    # Update global
                    if tuple(out_idx) == tuple(all_out):
                        if _better(sc, acc, global_):
                            global_.subconfig = sc
                            global_.accuracy = acc
                    # Update partial
                    key = tuple(out_idx)
                    if key in partials and _better(sc, acc, partials[key]):
                        partials[key].subconfig = sc
                        partials[key].accuracy = acc

    print(f"  Explored {n_explored} subconfigurations")
    return ref, global_, partials


def _better(new, new_acc, entry):
    if entry.subconfig is None:
        return True
    if new.n_active < entry.subconfig.n_active:
        return True
    return new.n_active == entry.subconfig.n_active and new_acc > entry.accuracy


# --- GPU Training ---

def train_subnet(n_in_sub, n_hidden, n_out_sub, x_train, y_train, x_val, y_val,
                 epochs=200, lr=0.001, batch_size=128, device="cuda", seed=42):
    """Train a subnetwork on GPU with early stopping. Returns (pre_acc, post_acc)."""
    torch.manual_seed(seed)
    model = nn.Sequential(
        nn.Linear(n_in_sub, n_hidden), nn.ReLU(),
        nn.Linear(n_hidden, n_out_sub),
    ).to(device)

    x_tr, y_tr = x_train.to(device), y_train.to(device)
    x_va, y_va = x_val.to(device), y_val.to(device)

    # Pre accuracy
    pre_acc = _acc(model, x_va, y_va, n_out_sub)

    criterion = nn.CrossEntropyLoss() if n_out_sub > 1 else nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(TensorDataset(x_tr, y_tr), batch_size=batch_size, shuffle=True)

    best_loss, patience, patience_max = float('inf'), 0, 30
    for _ in range(epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out.squeeze(), yb.float() if n_out_sub == 1 else yb.long())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg = total_loss / len(loader)
        if avg < best_loss - 1e-4:
            best_loss = avg
            patience = 0
        else:
            patience += 1
            if patience >= patience_max:
                break

    post_acc = _acc(model, x_va, y_va, n_out_sub)
    return pre_acc, post_acc


@torch.no_grad()
def _acc(model, x, y, n_out):
    model.eval()
    out = model(x)
    if n_out == 1:
        pred = (out.squeeze() > 0).float()
        return (pred == y).float().mean().item()
    return (out.argmax(1) == y).float().mean().item()


# --- Datasets ---

def load_data(name):
    """Returns x_train, y_train, x_val, y_val (CPU tensors), n_classes."""
    from sklearn.datasets import load_wine, load_iris, load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    loaders = {"wine": load_wine, "iris": load_iris, "cancer": load_breast_cancer}

    if name in loaders:
        data = loaders[name]()
        X, y = data.data, data.target
        X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        sc = StandardScaler().fit(X_tr)
        X_tr, X_va = sc.transform(X_tr), sc.transform(X_va)
        return (torch.tensor(X_tr, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.long),
                torch.tensor(X_va, dtype=torch.float32), torch.tensor(y_va, dtype=torch.long),
                len(np.unique(y)))
    elif name == "mnist":
        from torchvision import datasets, transforms
        ds_tr = datasets.MNIST("~/.cache/torch", train=True, download=True, transform=transforms.ToTensor())
        ds_te = datasets.MNIST("~/.cache/torch", train=False, transform=transforms.ToTensor())
        x_tr = ds_tr.data.float().reshape(-1, 784) / 255.0
        x_va = ds_te.data.float().reshape(-1, 784) / 255.0
        return x_tr, ds_tr.targets, x_va, ds_te.targets, 10
    elif name == "cifar10":
        from torchvision import datasets, transforms
        ds_tr = datasets.CIFAR10("~/.cache/torch", train=True, download=True, transform=transforms.ToTensor())
        ds_te = datasets.CIFAR10("~/.cache/torch", train=False, transform=transforms.ToTensor())
        x_tr = torch.tensor(np.array(ds_tr.data), dtype=torch.float32).reshape(-1, 3072) / 255.0
        x_va = torch.tensor(np.array(ds_te.data), dtype=torch.float32).reshape(-1, 3072) / 255.0
        return x_tr, torch.tensor(ds_tr.targets), x_va, torch.tensor(ds_te.targets), 10
    raise ValueError(f"Unknown dataset: {name}")


# --- Main ---

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="wine")
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--input-range", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    x_train, y_train, x_val, y_val, n_classes = load_data(args.dataset)
    n_feat = x_train.shape[1]
    print(f"Dataset: {args.dataset} | {n_feat} features, {n_classes} classes")
    print(f"Train: {x_train.shape[0]}, Val: {x_val.shape[0]}")

    input_range = None
    if args.input_range:
        parts = args.input_range.split(",")
        input_range = (int(parts[0]), int(parts[1]))

    # Explore (CPU)
    print(f"\n--- Exploration (CPU) ---")
    t0 = time.time()
    ref, global_, partials = explore(n_feat, n_classes, args.hidden, x_val, y_val,
                                      threshold=args.threshold, input_range=input_range, seed=args.seed)
    print(f"  Time: {time.time()-t0:.1f}s")

    # Train on GPU
    print(f"\n--- Training ({device}) ---")
    all_in, all_out = list(range(n_feat)), list(range(n_classes))

    # Reference
    x_tr_ref, y_tr_ref = x_train, y_train
    x_va_ref, y_va_ref = x_val, y_val
    ref_pre, ref_post = train_subnet(n_feat, args.hidden, n_classes,
        x_tr_ref, y_tr_ref, x_va_ref, y_va_ref,
        epochs=args.epochs, lr=args.lr, batch_size=args.batch_size, device=device, seed=args.seed)
    print(f"  Reference: {ref_pre*100:.1f}% → {ref_post*100:.1f}% ({n_feat+args.hidden+n_classes} neur)")

    # Global subnetwork
    if global_.subconfig is not None:
        sc = global_.subconfig
        # Filter samples for output subset
        mask_tr = torch.tensor([y.item() in sc.output_indices for y in y_train])
        mask_va = torch.tensor([y.item() in sc.output_indices for y in y_val])
        label_map = {orig: i for i, orig in enumerate(sc.output_indices)}
        x_tr_g = x_train[mask_tr][:, sc.input_indices]
        y_tr_g = torch.tensor([label_map[y.item()] for y in y_train[mask_tr]])
        x_va_g = x_val[mask_va][:, sc.input_indices]
        y_va_g = torch.tensor([label_map[y.item()] for y in y_val[mask_va]])
        n_out_g = len(sc.output_indices)

        g_pre, g_post = train_subnet(len(sc.input_indices), args.hidden, n_out_g,
            x_tr_g, y_tr_g, x_va_g, y_va_g,
            epochs=args.epochs, lr=args.lr, batch_size=args.batch_size, device=device, seed=args.seed)
        global_.accuracy_pre = g_pre
        global_.accuracy_post = g_post
        print(f"  Global:    {g_pre*100:.1f}% → {g_post*100:.1f}% ({sc.n_active} neur, in={sc.input_indices})")

    # Score & select
    max_neur = n_feat + args.hidden + n_classes
    ref_score = 0.8 * ref_post
    best_type, best_score, best_acc, best_neur = "reference", ref_score, ref_post, max_neur

    if global_.subconfig is not None:
        g_score = 0.8 * global_.accuracy_post + 0.2 * (1 - global_.subconfig.n_active / max_neur)
        if g_score > best_score:
            best_type, best_score, best_acc = "global", g_score, global_.accuracy_post
            best_neur = global_.subconfig.n_active

    print(f"\n--- Result ---")
    print(f"  Reference: {ref_post*100:.1f}% ({max_neur} neurons)")
    if global_.subconfig:
        print(f"  Subnetwork: {global_.accuracy_post*100:.1f}% ({global_.subconfig.n_active} neurons)")
    print(f"  Selected: {best_type} → {best_acc*100:.1f}% ({best_neur} neurons)")
    print(f"  Total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
