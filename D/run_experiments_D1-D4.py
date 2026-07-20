#!/usr/bin/env python3
"""
run_experiments_D.py -- D1/D2/D3/D4 四组实验统一调度脚本
===========================================================
按顺序执行所有子实验，通过环境变量传参，不修改 train.py 文件。

用法:
    cd AUG_experiments
    python run_experiments_D.py

机制:
    - 通过 COMET_AUG_* 环境变量覆盖 train.py 中结构化增强参数
    - 日志保存到 ./logs/D1_D2_D3_D4/ 目录，按实验编号命名
    - 原始 train.py 只读不写，永不修改
    - D1-0/D2-0/D3-0 为同一基线（无增强），只跑一次
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
LOG_ROOT = "./logs/D1_D2_D3_D4"


# ==================== 实验配置 ====================

# D1: 三类增强操作独立贡献（8个组合）
# 基线 D1-0（全关）在所有组中只跑一次
D1_EXPERIMENTS = [
    # D1-0 为基线，在 D0（共用基线）中跑
    {"id": "D1-1", "name": "replace_only",         "aug": True,  "sch": True,  "prob": 0.5, "r": True,  "p": False, "d": False, "sig": 0.05},
    {"id": "D1-2", "name": "perturb_only",         "aug": True,  "sch": True,  "prob": 0.5, "r": False, "p": True,  "d": False, "sig": 0.05},
    {"id": "D1-3", "name": "dropout_only",         "aug": True,  "sch": True,  "prob": 0.5, "r": False, "p": False, "d": True,  "sig": 0.05},
    {"id": "D1-4", "name": "replace_perturb",      "aug": True,  "sch": True,  "prob": 0.5, "r": True,  "p": True,  "d": False, "sig": 0.05},
    {"id": "D1-5", "name": "replace_dropout",      "aug": True,  "sch": True,  "prob": 0.5, "r": True,  "p": False, "d": True,  "sig": 0.05},
    {"id": "D1-6", "name": "perturb_dropout",      "aug": True,  "sch": True,  "prob": 0.5, "r": False, "p": True,  "d": True,  "sig": 0.05},
    {"id": "D1-7", "name": "all_three_full",       "aug": True,  "sch": True,  "prob": 0.5, "r": True,  "p": True,  "d": True,  "sig": 0.05},
]

# D2: 增强概率调度策略消融
# D2-0 基线复用 D0
D2_EXPERIMENTS = [
    {"id": "D2-1", "name": "fixed_0.3",            "aug": True,  "sch": False, "prob": 0.3, "r": True,  "p": True,  "d": True,  "sig": 0.05},
    {"id": "D2-2", "name": "fixed_0.5",            "aug": True,  "sch": False, "prob": 0.5, "r": True,  "p": True,  "d": True,  "sig": 0.05},
    {"id": "D2-3", "name": "curriculum",           "aug": True,  "sch": True,  "prob": 0.5, "r": True,  "p": True,  "d": True,  "sig": 0.05},
    {"id": "D2-4", "name": "fixed_low_0.1",        "aug": True,  "sch": False, "prob": 0.1, "r": True,  "p": True,  "d": True,  "sig": 0.05},
]

# D3: 比例扰动 sigma 值消融
# D3-0 基线复用 D0
D3_EXPERIMENTS = [
    {"id": "D3-1", "name": "sigma_0.02",           "aug": True,  "sch": True,  "prob": 0.5, "r": True,  "p": True,  "d": True,  "sig": 0.02},
    {"id": "D3-2", "name": "sigma_0.05_optimal",   "aug": True,  "sch": True,  "prob": 0.5, "r": True,  "p": True,  "d": True,  "sig": 0.05},
    {"id": "D3-3", "name": "sigma_0.10",           "aug": True,  "sch": True,  "prob": 0.5, "r": True,  "p": True,  "d": True,  "sig": 0.10},
    {"id": "D3-4", "name": "sigma_0.15",           "aug": True,  "sch": True,  "prob": 0.5, "r": True,  "p": True,  "d": True,  "sig": 0.15},
]

# D4: 结构化增强 vs Uni-Mol 噪声
# D4-0 无增强 → 复用 D0 基线
D4_EXPERIMENTS = [
    {"id": "D4-1", "name": "unimol_noise_only",     "aug": False, "sch": False, "prob": 0.0, "r": False, "p": False, "d": False, "sig": 0.0,  "noise": 0.1},
    {"id": "D4-2", "name": "structural_aug_only",   "aug": True,  "sch": True,  "prob": 0.5, "r": True,  "p": True,  "d": True,  "sig": 0.05, "noise": 0.0},
    {"id": "D4-3", "name": "both_combined",         "aug": True,  "sch": True,  "prob": 0.5, "r": True,  "p": True,  "d": True,  "sig": 0.05, "noise": 0.1},
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


def ensure_utils_init():
    """确保 utils/ 目录有 __init__.py，避免 import 被 AUG_unimol/utils 覆盖"""
    init_file = os.path.join(".", "utils", "__init__.py")
    if not os.path.exists(init_file):
        with open(init_file, "w") as f:
            f.write("")
        print(f"[INFO] 创建 {init_file}")


def run_exp(exp_id, exp_name, aug, sch, prob, r, p, d, sig,
            epoch, patience, folds, log_file, noise=0.1):
    """执行单个子实验，通过环境变量传参"""
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"\n{'='*60}")
    print(f"  [{ts}] {exp_id} | {exp_name}")
    print(f"  aug={aug} sch={sch} prob={prob} r={r} p={p} d={d} sig={sig} noise={noise}")
    print(f"{'='*60}")

    ensure_dir(os.path.dirname(log_file))

    with open(log_file, "w") as log_fh:
        log_fh.write(f"# {exp_id} | {exp_name}\n")
        log_fh.write(f"# aug={aug} sch={sch} prob={prob} replace={r} perturb={p} dropout={d} sigma={sig}\n")
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
            _exp_dir = os.path.abspath(".")
            # _exp_dir 放最前面，确保 AUG_experiments/utils 优先于 AUG_unimol/utils
            env["PYTHONPATH"] = f"{_exp_dir}:../AUG_unimol:.."
            env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            env["COMET_EXP_ID"] = exp_id
            env["COMET_EXP_TAG"] = f"-{exp_id}"
            env["COMET_AUG_ENABLED"] = "1" if aug else "0"
            env["COMET_AUG_SCHEDULER"] = "1" if sch else "0"
            env["COMET_AUG_STATIC_PROB"] = str(prob)
            env["COMET_AUG_ENABLE_REPLACE"] = "1" if r else "0"
            env["COMET_AUG_ENABLE_PERTURB"] = "1" if p else "0"
            env["COMET_AUG_ENABLE_DROPOUT"] = "1" if d else "0"
            env["COMET_AUG_SIGMA"] = str(sig)
            env["COMET_PERCENT_NOISE"] = str(noise)

            # 确保 cwd 是 AUG_experiments 绝对路径
            _cwd = os.path.abspath(".")
            process = subprocess.Popen(
                [sys.executable, "train.py", fold],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
                cwd=_cwd
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
        _noise = exp.get("noise", 0.1)  # 默认0.1，D4实验可覆盖
        ok = run_exp(
            exp["id"], exp["name"],
            exp["aug"], exp["sch"], exp["prob"],
            exp["r"], exp["p"], exp["d"], exp["sig"],
            epoch, patience, folds, log_file,
            noise=_noise
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
    print("  D1/D2/D3/D4 实验调度 (结构化数据增强消融)")
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
    ensure_utils_init()
    all_ok = []

    try:
        # D0: 共用基线（无增强，所有组复用）
        ok = run_exp(
            "D0", "baseline_no_aug",
            aug=False, sch=False, prob=0.0,
            r=False, p=False, d=False, sig=0.0,
            epoch=150, patience=1000,
            folds=["fold_V0", "fold_V1", "fold_V2"],
            log_file=os.path.join(LOG_ROOT, "D0_baseline_no_aug.log")
        )
        all_ok.append(("D0 (共用基线)", ok))

        # D1: 三类增强独立贡献
        if ok:
            ok = run_group("D1 (增强操作独立贡献)", D1_EXPERIMENTS,
                           epoch=150, patience=1000,
                           folds=["fold_V0", "fold_V1", "fold_V2"])
        all_ok.append(("D1", ok))

        # D2: 调度策略消融
        if ok:
            ok = run_group("D2 (调度策略消融)", D2_EXPERIMENTS,
                           epoch=150, patience=1000,
                           folds=["fold_V0", "fold_V1", "fold_V2"])
        all_ok.append(("D2", ok))

        # D3: sigma 值消融
        if ok:
            ok = run_group("D3 (sigma 值消融)", D3_EXPERIMENTS,
                           epoch=150, patience=1000,
                           folds=["fold_V0", "fold_V1", "fold_V2"])
        all_ok.append(("D3", ok))

        # D4: 结构化增强 vs Uni-Mol 噪声
        # D4-0 无增强 → 复用 D0 基线
        if ok:
            ok = run_group("D4 (结构化增强 vs Uni-Mol 噪声)", D4_EXPERIMENTS,
                           epoch=150, patience=1000,
                           folds=["fold_V0", "fold_V1", "fold_V2"])
        all_ok.append(("D4", ok))

    finally:
        cleanup_symlink()

    duration = datetime.now() - start
    print(f"\n{'='*60}")
    print(f"  总用时: {duration}")
    for g, ok in all_ok:
        print(f"  {g}: {ok}")
    if os.path.exists(LOG_ROOT):
        logs = sorted(f for f in os.listdir(LOG_ROOT) if f.endswith('.log'))
        print(f"\n  日志文件 ({len(logs)}个):")
        for lf in logs:
            fp = os.path.join(LOG_ROOT, lf)
            print(f"    {lf}  ({os.path.getsize(fp)/1024:.1f} KB)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
