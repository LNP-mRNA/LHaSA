#!/usr/bin/env python3
"""
run_experiments.py -- A1/A2/A3 三组实验统一调度脚本
========================================================
按顺序执行所有子实验，通过环境变量传参，不修改 train.py 文件。

用法:
    cd LOSS_experiments
    python run_experiments.py

机制:
    - 通过 COMET_* 环境变量覆盖 train.py 中 CONFIG_GRID 的参数
    - train.py 中 exp_name 自动缩短前缀 + 包含 K 值 + 实验标记
    - 日志保存到 ./logs/A1_A2_A3/ 目录，按实验编号命名
    - 原始 train.py 只读不写，永不修改
"""

import subprocess
import sys
import os
import shutil
from datetime import datetime

# ==================== 路径配置 ====================

EXP_DIR = os.path.abspath(".")
COMET_ROOT = os.path.dirname(EXP_DIR)
EXP_NAME = os.path.basename(EXP_DIR).replace("_experiments", "")
UNIMOL_DIR = f"{EXP_NAME}_unimol"
UNIMOL_FULL_PATH = os.path.join(COMET_ROOT, UNIMOL_DIR)
SYMLINK_PATH = os.path.join(COMET_ROOT, "unimol")
LOG_ROOT = "./logs/A1_A2_A3"


# ==================== 实验配置 ====================

A1_EXPERIMENTS = [
    {"id": "A1-0", "name": "alpha_0.0",   "lambdarank_alpha": 0.0, "grad_clip_norm": 1.0, "k": 10},
    {"id": "A1-1", "name": "alpha_0.1",   "lambdarank_alpha": 0.1, "grad_clip_norm": 1.0, "k": 10},
    {"id": "A1-2", "name": "alpha_0.3",   "lambdarank_alpha": 0.3, "grad_clip_norm": 1.0, "k": 10},
    {"id": "A1-3", "name": "alpha_0.5",   "lambdarank_alpha": 0.5, "grad_clip_norm": 1.0, "k": 10},
    {"id": "A1-4", "name": "alpha_0.7",   "lambdarank_alpha": 0.7, "grad_clip_norm": 1.0, "k": 10},
    {"id": "A1-5", "name": "alpha_1.0",   "lambdarank_alpha": 1.0, "grad_clip_norm": 1.0, "k": 10},
]

A2_EXPERIMENTS = [
    {"id": "A2-0", "name": "clip_0.0",    "lambdarank_alpha": 0.3, "grad_clip_norm": 0.0, "k": 10},
    {"id": "A2-1", "name": "clip_1.0",    "lambdarank_alpha": 0.3, "grad_clip_norm": 1.0, "k": 10},
    {"id": "A2-2", "name": "clip_0.5",    "lambdarank_alpha": 0.3, "grad_clip_norm": 0.5, "k": 10},
    {"id": "A2-3", "name": "clip_2.0",    "lambdarank_alpha": 0.3, "grad_clip_norm": 2.0, "k": 10},
]

A3_EXPERIMENTS = [
    {"id": "A3-4", "name": "K_50", "lambdarank_alpha": 0.3, "grad_clip_norm": 1.0, "k": 50},
]


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def setup_symlink():
    """创建软链接 unimol -> {VERSION}_unimol"""
    if os.path.islink(SYMLINK_PATH):
        os.unlink(SYMLINK_PATH)
    elif os.path.exists(SYMLINK_PATH):
        backup = SYMLINK_PATH + f"_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        shutil.move(SYMLINK_PATH, backup)
    if os.path.exists(UNIMOL_FULL_PATH):
        os.symlink(UNIMOL_FULL_PATH, SYMLINK_PATH)
        print(f"[INFO] 软链接: unimol -> {UNIMOL_DIR}")


def cleanup_symlink():
    if os.path.islink(SYMLINK_PATH):
        os.unlink(SYMLINK_PATH)


def run_exp(exp_id, exp_name, alpha, clip_norm, k, epoch, patience, folds, log_file):
    """执行单个子实验，通过环境变量传参"""
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"\n{'='*60}")
    print(f"  [{ts}] {exp_id} | {exp_name}")
    print(f"  alpha={alpha}, clip={clip_norm}, K={k}, epoch={epoch}, patience={patience}")
    print(f"{'='*60}")

    ensure_dir(os.path.dirname(log_file))

    with open(log_file, "w") as log_fh:
        log_fh.write(f"# {exp_id} | alpha={alpha} clip={clip_norm} K={k}\n")
        log_fh.write(f"# epoch={epoch} patience={patience} folds={folds}\n")
        log_fh.write(f"# start={datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_fh.write("=" * 60 + "\n\n")
        log_fh.flush()

        for fold in folds:
            fold_ts = datetime.now().strftime("%H:%M:%S")
            header = f"\n[{fold_ts}] {exp_id} | {fold} ...\n"
            print(header, end="")
            log_fh.write(header)
            log_fh.flush()

            # 环境变量传参
            env = os.environ.copy()
            env["PYTHONPATH"] = ".."
            env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            env["COMET_EXP_ID"] = exp_id
            env["COMET_EXP_TAG"] = f"-{exp_id}"
            env["COMET_LAMBDARANK_ALPHA"] = str(alpha)
            env["COMET_GRAD_CLIP_NORM"] = str(clip_norm)
            env["COMET_K"] = str(k)
            env["COMET_LAMBDARANK_K"] = str(k)  # LambdaRank 损失函数内部用这个
            env["COMET_EPOCH"] = str(epoch)
            env["COMET_PATIENCE"] = str(patience)
            env["COMET_USE_LAMBDARANK"] = "1" if alpha > 0 else "0"

            process = subprocess.Popen(
                [sys.executable, "train.py", fold],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env
            )

            for line in process.stdout:
                print(line, end="")
                log_fh.write(line)
                log_fh.flush()

            process.wait()

            if process.returncode != 0:
                err = f"\n[ERROR] {exp_id} {fold} failed code={process.returncode}\n"
                print(err)
                log_fh.write(err)
                return False

            done = f"\n[{datetime.now().strftime('%H:%M:%S')}] {exp_id} {fold} DONE\n"
            print(done)
            log_fh.write(done)
            log_fh.flush()

        log_fh.write(f"\n# end={datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_fh.write("# STATUS: SUCCESS\n")

    return True


def run_group(name, exps, epoch, patience, folds):
    print(f"\n{'#'*60}")
    print(f"# {name}: {len(exps)} 个实验")
    print(f"# epoch={epoch} patience={patience} folds={folds}")
    print(f"{'#'*60}")

    results = []
    for i, exp in enumerate(exps, 1):
        log_file = os.path.join(LOG_ROOT, f"{exp['id']}_{exp['name']}.log")
        ok = run_exp(
            exp["id"], exp["name"],
            exp["lambdarank_alpha"], exp["grad_clip_norm"], exp["k"],
            epoch, patience, folds, log_file
        )
        results.append((exp["id"], ok))
        print(f"\n[{i}/{len(exps)}] {exp['id']} {'OK' if ok else 'FAIL'}")
        if not ok:
            break

    print(f"\n{'-'*40}")
    print(f"  {name} 汇总")
    for eid, ok in results:
        print(f"  {'OK' if ok else 'FAIL'} {eid}")
    print(f"{'-'*40}")
    return all(ok for _, ok in results)


def main():
    start = datetime.now()
    print("=" * 60)
    print("  A1/A2/A3 实验调度")
    print(f"  start={start.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  logs={os.path.abspath(LOG_ROOT)}")
    print("=" * 60)

    ensure_dir(LOG_ROOT)
    if not os.path.exists("./train.py"):
        print("[ERROR] train.py 不存在")
        sys.exit(1)
    if not os.path.exists(UNIMOL_FULL_PATH):
        print(f"[ERROR] {UNIMOL_FULL_PATH} 不存在")
        sys.exit(1)

    setup_symlink()
    all_ok = []

    try:
        # A1 已完成
        print("\n[INFO] A1 已完成，跳过")
        all_ok.append(("A1", True))

        # A2 已完成
        print("[INFO] A2 已完成，跳过")
        all_ok.append(("A2", True))

        # A3: K=50
        ok = run_group("A3 (NDCG@K)", A3_EXPERIMENTS, epoch=150, patience=1000,
                       folds=["fold_V0", "fold_V1", "fold_V2"])
        all_ok.append(("A3", ok))

    finally:
        cleanup_symlink()

    duration = datetime.now() - start
    print(f"\n{'='*60}")
    print(f"  总用时: {duration}")
    for g, ok in all_ok:
        print(f"  {'OK' if ok else 'FAIL'} {g}")
    if os.path.exists(LOG_ROOT):
        logs = sorted(f for f in os.listdir(LOG_ROOT) if f.endswith('.log'))
        print(f"\n  日志文件 ({len(logs)}个):")
        for lf in logs:
            fp = os.path.join(LOG_ROOT, lf)
            print(f"    {lf}  ({os.path.getsize(fp)/1024:.1f} KB)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
