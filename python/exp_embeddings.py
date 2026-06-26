"""Exp 7: Neural Division on learned embeddings.

1. Train a network on MNIST.
2. Extract intermediate layer activations as "features" (embeddings).
3. Apply Neural Division on these embeddings.
4. Discover which learned neurons are necessary for which digits.

Usage:
    /media/javier/intenso-tools/home-moved/venvs/lazy-training-venv/bin/python python/exp_embeddings.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from itertools import combinations
import time


def train_encoder(device="cuda"):
    """Train a simple MNIST network and return it."""
    from torchvision import datasets, transforms
    ds_tr = datasets.MNIST("~/.cache/torch", train=True, download=True, transform=transforms.ToTensor())
    ds_te = datasets.MNIST("~/.cache/torch", train=False, transform=transforms.ToTensor())
    x_tr = ds_tr.data.float().reshape(-1, 784).to(device) / 255.0
    y_tr = ds_tr.targets.to(device)
    x_te = ds_te.data.float().reshape(-1, 784).to(device) / 255.0
    y_te = ds_te.targets.to(device)

    # Architecture: 784 → 128 → 32 → 10
    model = nn.Sequential(
        nn.Linear(784, 128), nn.ReLU(),
        nn.Linear(128, 32), nn.ReLU(),
        nn.Linear(32, 10)
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(x_tr, y_tr), batch_size=512, shuffle=True)

    print("  Training MNIST encoder (784→128→32→10)...")
    for epoch in range(15):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        acc = (model(x_te).argmax(1) == y_te).float().mean().item()
    print(f"  Full network accuracy: {acc*100:.1f}%")

    return model, x_tr, y_tr, x_te, y_te


def extract_embeddings(model, x, layer_idx=2):
    """Extract activations after layer_idx (0=first linear, 2=second linear)."""
    # Get activations from the 32-dim layer (after second ReLU)
    model.eval()
    with torch.no_grad():
        h = x
        for i, layer in enumerate(model):
            h = layer(h)
            if i == layer_idx + 1:  # after ReLU following layer_idx
                return h
    return h


@torch.no_grad()
def evaluate_embedding_subconfig(w_out, b_out, embeddings, y_val, neuron_idx, output_idx):
    """Evaluate using subset of embedding neurons for subset of classes."""
    emb = embeddings[:, neuron_idx]
    out = emb @ w_out[neuron_idx][:, output_idx] + b_out[output_idx]
    if len(output_idx) == 1:
        pred = (torch.sigmoid(out.squeeze()) > 0.5).long()
        target = (y_val == output_idx[0]).long()
        return (pred == target).float().mean().item()
    else:
        pred_local = out.argmax(dim=1)
        pred_global = torch.tensor([output_idx[i] for i in pred_local.cpu().tolist()], device=y_val.device)
        return (pred_global == y_val).float().mean().item()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print("=" * 60)
    print("  Exp 7: Neural Division on learned embeddings (MNIST)")
    print("=" * 60)

    model, x_tr, y_tr, x_te, y_te = train_encoder(device)

    # Extract 32-dim embeddings
    print("\n  Extracting 32-dim embeddings...")
    emb_te = extract_embeddings(model, x_te, layer_idx=2)
    print(f"  Embedding shape: {emb_te.shape}")

    # The last layer weights: (32, 10)
    last_layer = model[4]  # Linear(32, 10)
    w_out = last_layer.weight.data.T  # (32, 10)
    b_out = last_layer.bias.data  # (10,)

    # Now apply Neural Division: which of the 32 neurons are needed for each digit?
    n_neurons = 32
    n_classes = 10
    threshold = 0.5

    print(f"\n  Exploring subsets of 32 embedding neurons...")
    print(f"  Question: which neurons are necessary for each digit pair?\n")

    # For interpretability, focus on digit pairs (0 vs 1, 3 vs 8, etc.)
    digit_pairs = [(0, 1), (3, 8), (4, 9), (2, 7), (5, 6)]

    print(f"{'Pair':>8} | {'Full 32':>7} | {'Best K':>6} | {'K neurons':>9} | {'Acc':>6} | {'Key neurons'}")
    print("-" * 70)

    for d1, d2 in digit_pairs:
        output_idx = [d1, d2]
        # Filter test set for these two digits
        mask = (y_te == d1) | (y_te == d2)
        emb_pair = emb_te[mask]
        y_pair = y_te[mask]
        # Remap: d1→0, d2→1
        y_binary = (y_pair == d2).long()

        # Full 32 neurons accuracy
        out_full = emb_pair @ w_out[:, output_idx] + b_out[output_idx]
        pred_full = out_full.argmax(1)
        full_acc = (pred_full == y_binary).float().mean().item()

        # Search: try subsets of size 1 to 10
        best_acc = 0.0
        best_neurons = None
        for k in range(1, min(11, n_neurons + 1)):
            # ponytail: for k>5, sample random subsets instead of exhaustive
            if k <= 5:
                combos = list(combinations(range(n_neurons), k))
            else:
                rng = np.random.RandomState(42)
                combos = [sorted(rng.choice(n_neurons, k, replace=False).tolist()) for _ in range(500)]

            for neurons in combos:
                neurons_list = list(neurons)
                emb_sub = emb_pair[:, neurons_list]
                w_sub = w_out[neurons_list][:, output_idx]
                b_sub = b_out[output_idx]
                out_sub = emb_sub @ w_sub + b_sub
                pred_sub = out_sub.argmax(1)
                acc = (pred_sub == y_binary).float().mean().item()
                if acc > best_acc:
                    best_acc = acc
                    best_neurons = neurons_list

            if best_acc >= 0.99:
                break  # Found near-perfect with k neurons

        print(f"  {d1} vs {d2} | {full_acc*100:>5.1f}% | {len(best_neurons):>6} | {best_neurons[:6]} | {best_acc*100:>5.1f}% |")

    # Global: which neurons are most important across all pairs?
    print("\n  Neuron importance (frequency in best subsets):")
    freq = np.zeros(n_neurons)
    for d1, d2 in digit_pairs:
        mask = (y_te == d1) | (y_te == d2)
        emb_pair = emb_te[mask]
        y_binary = (y_te[mask] == d2).long()
        # Find best subset of size 3
        best_acc = 0.0
        best_neurons = None
        for neurons in combinations(range(n_neurons), 3):
            neurons_list = list(neurons)
            emb_sub = emb_pair[:, neurons_list]
            w_sub = w_out[neurons_list][:, [d1, d2]]
            out_sub = emb_sub @ w_sub + b_out[[d1, d2]]
            acc = (out_sub.argmax(1) == y_binary).float().mean().item()
            if acc > best_acc:
                best_acc = acc
                best_neurons = neurons_list
        if best_neurons:
            for n in best_neurons:
                freq[n] += 1

    top = np.argsort(freq)[::-1][:10]
    for n in top:
        if freq[n] > 0:
            print(f"    Neuron {n:>2}: {int(freq[n])}/5 pairs")


if __name__ == "__main__":
    main()
