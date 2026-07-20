#!/usr/bin/env python3
"""run_folds.py -- 3-fold cross-validation launcher

Runs train.py sequentially for each fold (fold_V0, fold_V1, fold_V2).
Any extra command-line arguments are passed through to train.py.

Usage:
    python run_folds.py                    # Basic run
    python run_folds.py --use-swa          # Pass --use-swa to each train.py
    python run_folds.py --epochs 50        # Pass --epochs 50 to each train.py
"""
import subprocess
import sys

folds = ['fold_V0', 'fold_V1', 'fold_V2']

# Collect any extra arguments passed to this script
extra_args = sys.argv[1:]

for i, fold in enumerate(folds):
    print(f"\n[{i+1}/{len(folds)}] Starting {fold}...")
    cmd = [sys.executable, 'train.py', fold] + extra_args
    print(f"[CMD] {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd='.')
    if result.returncode != 0:
        print(f"[ERROR] {fold} failed with code {result.returncode}")
        break
    print(f"[{i+1}/{len(folds)}] {fold} completed.")

print("\n[Done] All folds processed.")
