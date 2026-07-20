#!/usr/bin/env python3
"""
run_folds.py -- 3折串行训练 wrapper
=========================================
- 避免 Python 21层嵌套限制
- 通过子进程串行运行每个 fold
- 用法: python run_folds.py
"""

import subprocess
import sys

folds = ['fold_V0', 'fold_V1', 'fold_V2']

print("=" * 50)
print("  3-Fold Cross Validation Launcher")
print("=" * 50)

for i, fold in enumerate(folds):
    print(f"\n[{i+1}/{len(folds)}] Starting {fold}...")
    print("-" * 50)
    
    # 启动子进程运行当前 fold
    result = subprocess.run(
        [sys.executable, 'train.py', fold],
        cwd='.'  # 在 BS_experiments 目录下运行
    )
    
    if result.returncode != 0:
        print(f"[ERROR] {fold} failed with code {result.returncode}")
        break
    
    print(f"[{i+1}/{len(folds)}] {fold} completed.")

print("\n" + "=" * 50)
print("  All folds completed!")
print("=" * 50)
