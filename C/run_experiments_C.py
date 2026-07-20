#!/usr/bin/env python3
"""
run_experiments_C.py -- C1/C2/C3/C4 四组实验统一调度脚本
===========================================================
按顺序执行所有子实验，通过环境变量传参，不修改 train.py 文件。

用法:
    cd SWA_experiments
    python run_experiments_C.py

机制:
    - 通过 COMET_SWA_* 环境变量覆盖 train.py 中 SWA 参数
    - 日志保存到 ./logs/C1_C2_C3_C4/ 目录，按实验编号命名
    - 原始 train.py 只读不写，永不修改
    - C3 为分析性实验，从 C1/C2 结果中提取，无需单独跑
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
LOG_ROOT = "./logs/C1_C2_C3_C4"


# ==================== 实验配置 ====================

# C1: SWA 启动时机 swa_start 消融
# 默认: lr=1e-4, anneal_epochs=5, anneal_strategy=cos
# C1-0 为基线（不用 SWA），在 C4-0 中复用
C1_EXPERIMENTS = [
    {"id": "C1-1", "name": "start_20",  "swa": True,  "start": 20, "lr": 1e-4, "ae": 5, "as": "cos"},
    {"id": "C1-2", "name": "start_40",  "swa": True,  "start": 40, "lr": 1e-4, "ae": 5, "as": "cos"},
    {"id": "C1-3", "name": "start_60",  "swa": True,  "start": 60, "lr": 1e-4, "ae": 5, "as": "cos"},
    {"id": "C1-4", "name": "start_80",  "swa": True,  "start": 80, "lr": 1e-4, "ae": 5, "as": "cos"},
]

# C2: SWA 学习率 swa_lr 消融
# 默认: start=40, anneal_epochs=5
# C2-0 为基线（不用 SWA），复用 C4-0
C2_EXPERIMENTS = [
    {"id": "C2-1", "name": "lr_1e-3_cos",     "swa": True,  "start": 40, "lr": 1e-3, "ae": 5, "as": "cos"},
    {"id": "C2-2", "name": "lr_1e-4_cos",     "swa": True,  "start": 40, "lr": 1e-4, "ae": 5, "as": "cos"},
    {"id": "C2-3", "name": "lr_1e-5_cos",     "swa": True,  "start": 40, "lr": 1e-5, "ae": 5, "as": "cos"},
    {"id": "C2-4", "name": "lr_1e-4_const",   "swa": True,  "start": 40, "lr": 1e-4, "ae": 5, "as": "constant"},
]

# C3: SWA 对验证集稳定性影响
# 分析性实验，从 C1/C2 的日志中提取验证集 Spearman 波动数据
# 无需单独跑模型

# C4: SWA 与 Sharp Minima 关系
# C4-0: SGD 基线（不用 SWA）
# C4-1: +SWA
# C4-2: +SAM（需要 SAM 优化器支持，标记为可选）
C4_EXPERIMENTS = [
    {"id": "C4-0", "name": "baseline_sgd",   "swa": False, "start": 0,  "lr": 0,   "ae": 0, "as": "cos"},
    {"id": "C4-1", "name": "plus_swa",       "swa": True,  "start": 40, "lr": 1e-4, "ae": 5, "as": "cos"},
    # C4-2 SAM 需要额外安装和配置，如需运行请手动添加
    # {"id": "C4-2", "name": "plus_sam",       "swa": False, "start": 0,  "lr": 0,   "ae": 0, "as": "cos"},
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


def run_exp(exp_id, exp_name, swa, start, lr, ae, aas,
            epoch, patience, folds, log_file):
    """执行单个子实验，通过环境变量传参"""
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"\n{'='*60}")
    print(f"  [{ts}] {exp_id} | {exp_name}")
    print(f"  swa={swa}, start={start}, lr={lr}, anneal={ae}, strategy={aas}")
    print(f"{'='*60}")

    ensure_dir(os.path.dirname(log_file))

    with open(log_file, "w") as log_fh:
        log_fh.write(f"# {exp_id} | {exp_name}\n")
        log_fh.write(f"# swa={swa} start={start} lr={lr} anneal={ae} strategy={aas}\n")
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
            env["COMET_SWA_ENABLED"] = "1" if swa else "0"
            env["COMET_SWA_START"] = str(start)
            env["COMET_SWA_LR"] = str(lr)
            env["COMET_SWA_ANNEAL_EPOCHS"] = str(ae)
            env["COMET_SWA_ANNEAL_STRATEGY"] = str(aas)

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
            exp["swa"], exp["start"], exp["lr"], exp["ae"], exp["as"],
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
    print("  C1/C2/C3/C4 实验调度 (SWA 消融)")
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
        # C4-0: 基线（不用 SWA），作为所有组的共用基线
        ok = run_group("C4-0 (SGD 基线)", [C4_EXPERIMENTS[0]],
                       epoch=150, patience=1000,
                       folds=["fold_V0", "fold_V1", "fold_V2"])
        all_ok.append(("C4-0", ok))

        # C1: SWA 启动时机消融
        if ok:
            ok = run_group("C1 (SWA 启动时机)", C1_EXPERIMENTS,
                           epoch=150, patience=1000,
                           folds=["fold_V0", "fold_V1", "fold_V2"])
        all_ok.append(("C1", ok))

        # C2: SWA 学习率消融
        if ok:
            ok = run_group("C2 (SWA 学习率)", C2_EXPERIMENTS,
                           epoch=150, patience=1000,
                           folds=["fold_V0", "fold_V1", "fold_V2"])
        all_ok.append(("C2", ok))

        # C4-1: +SWA
        if ok:
            ok = run_group("C4-1 (+SWA)", [C4_EXPERIMENTS[1]],
                           epoch=150, patience=1000,
                           folds=["fold_V0", "fold_V1", "fold_V2"])
        all_ok.append(("C4-1", ok))

        # C3: 分析性实验提示
        print(f"\n{'#'*60}")
        print("# C3 (SWA 验证集稳定性影响)")
        print("# 分析性实验：从 C1/C2 的日志中提取验证集 Spearman 波动数据")
        print("# 无需单独跑模型")
        print(f"{'#'*60}")
        all_ok.append(("C3", True))

        # C4-2 已取消（SAM 对比实验不需要）

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
