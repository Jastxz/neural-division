"""Run full benchmark suite with GPU training — multiple seeds per dataset.

Outputs results to python/results/ as CSV.

Usage:
    /media/javier/intenso-tools/home-moved/venvs/lazy-training-venv/bin/python python/run_suite.py
"""

import subprocess
import sys
import csv
import time
from pathlib import Path

PYTHON = "/media/javier/intenso-tools/home-moved/venvs/lazy-training-venv/bin/python"
SCRIPT = "python/neural_division_gpu.py"
RESULTS_DIR = Path("python/results")
RESULTS_DIR.mkdir(exist_ok=True)

SEEDS = [42, 123, 7, 2024, 31415, 1, 99, 256, 777, 1337,
         5555, 8080, 9999, 12345, 54321, 65536, 100000, 271828, 314159, 999999]

# Datasets feasible for exhaustive exploration (≤13 features)
CONFIGS = [
    # (name, hidden, epochs, threshold, input_range)
    ("wine",   16, 300, 0.4, None),
    ("wine",   32, 300, 0.4, None),
    ("iris",   16, 300, 0.4, None),
    ("cancer", 16, 300, 0.5, None),
    ("cancer", 32, 300, 0.5, None),
]

def run_one(dataset, hidden, epochs, threshold, input_range, seed):
    """Run a single experiment and parse output."""
    cmd = [PYTHON, SCRIPT,
           "--dataset", dataset,
           "--hidden", str(hidden),
           "--epochs", str(epochs),
           "--threshold", str(threshold),
           "--seed", str(seed),
           "--batch-size", "128"]
    if input_range:
        cmd += ["--input-range", input_range]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

    # Parse output
    ref_acc, sub_acc, sub_neur, selected = 0.0, 0.0, 0, "reference"
    for line in result.stdout.split("\n"):
        if "Reference:" in line and "→" in line:
            parts = line.split("→")[1].strip().split("%")[0]
            ref_acc = float(parts)
        elif "Global:" in line and "→" in line:
            parts = line.split("→")[1].strip().split("%")[0]
            sub_acc = float(parts)
            if "neur" in line:
                sub_neur = int(line.split("(")[1].split(" ")[0])
        elif "Selected:" in line:
            selected = line.split("→")[0].split(":")[1].strip()
            sel_acc = float(line.split("→")[1].strip().split("%")[0])

    return {"ref_acc": ref_acc, "sub_acc": sub_acc, "sub_neur": sub_neur,
            "selected": selected, "seed": seed}


def main():
    print("Neural Division GPU Suite")
    print(f"Seeds: {len(SEEDS)} | Configs: {len(CONFIGS)}")
    print("=" * 60)

    all_results = []

    for dataset, hidden, epochs, threshold, input_range in CONFIGS:
        config_name = f"{dataset}_h{hidden}"
        print(f"\n--- {config_name} (threshold={threshold}) ---")

        results = []
        for seed in SEEDS:
            try:
                r = run_one(dataset, hidden, epochs, threshold, input_range, seed)
                results.append(r)
                status = "✓" if r["ref_acc"] > 0 else "✗"
                print(f"  Seed {seed:>6}: ref={r['ref_acc']:.1f}% sub={r['sub_acc']:.1f}% → {r['selected']} {status}")
            except Exception as e:
                print(f"  Seed {seed:>6}: ERROR {e}")
                results.append({"ref_acc": 0, "sub_acc": 0, "sub_neur": 0, "selected": "error", "seed": seed})

        # Summary
        refs = [r["ref_acc"] for r in results if r["ref_acc"] > 0]
        subs = [r["sub_acc"] for r in results if r["sub_acc"] > 0]
        n_sub = sum(1 for r in results if r["selected"] != "reference")

        if refs:
            import numpy as np
            print(f"  Ref mean: {np.mean(refs):.1f}% ± {np.std(refs):.1f}%")
        if subs:
            print(f"  Sub mean: {np.mean(subs):.1f}% ± {np.std(subs):.1f}%")
        print(f"  Subnetwork selected: {n_sub}/{len(results)}")

        # Save CSV
        csv_path = RESULTS_DIR / f"{config_name}.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["seed", "ref_acc", "sub_acc", "sub_neur", "selected"])
            w.writeheader()
            w.writerows(results)

        for r in results:
            r["config"] = config_name
        all_results.extend(results)

    # Global summary CSV
    csv_path = RESULTS_DIR / "all_results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["config", "seed", "ref_acc", "sub_acc", "sub_neur", "selected"])
        w.writeheader()
        w.writerows(all_results)

    print(f"\n{'='*60}")
    print(f"Results saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
