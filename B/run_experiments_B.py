#!/usr/bin/env python3
"""
run_experiments_B.py -- B1/B2/B3/B4 四组实验统一调度脚本
===========================================================
按顺序执行所有子实验，通过环境变量传参，不修改 train.py 文件。

用法:
    cd HARD_experiments
    python run_experiments_B.py

机制:
    - 通过 COMET_HNM_* 环境变量覆盖 train.py 中 HNM 参数
    - 日志保存到 ./logs/B1_B2_B3_B4/ 目录，按实验编号命名
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
LOG_ROOT = "./logs/B1_B2_B3_B4"


# ==================== 实验配置 ====================

# B1: hard_ratio 消融
# 默认: mining=True, every=5, start=0.10, end=0.30, warmup=20, weight=2.0, scale=1.2
B1_EXPERIMENTS = [
    {"id": "B1-0", "name": "ratio_0pct_baseline",     "mining": False, "me": 5,  "rs": 0.0,  "re": 0.0,  "rw": 0,  "pw": 2.0, "bs": 1.2},
    {"id": "B1-1", "name": "ratio_10pct_fixed",       "mining": True,  "me": 5,  "rs": 0.10, "re": 0.10, "rw": 0,  "pw": 2.0, "bs": 1.2},
    {"id": "B1-2", "name": "ratio_10to30_curriculum", "mining": True,  "me": 5,  "rs": 0.10, "re": 0.30, "rw": 20, "pw": 2.0, "bs": 1.2},
    {"id": "B1-3", "name": "ratio_30pct_fixed",       "mining": True,  "me": 5,  "rs": 0.30, "re": 0.30, "rw": 0,  "pw": 2.0, "bs": 1.2},
    {"id": "B1-4", "name": "ratio_50pct_fixed",       "mining": True,  "me": 5,  "rs": 0.50, "re": 0.50, "rw": 0,  "pw": 2.0, "bs": 1.2},
    {"id": "B1-5", "name": "ratio_10to50_curriculum", "mining": True,  "me": 5,  "rs": 0.10, "re": 0.50, "rw": 20, "pw": 2.0, "bs": 1.2},
]

# B2: mining 频率消融
# 默认: mining=True, start=0.10, end=0.30, warmup=20, weight=2.0, scale=1.2
# 注: 基线(mining=False)与B1-0相同，不复跑，结果复用B1-0
B2_EXPERIMENTS = [
    {"id": "B2-1", "name": "mining_every_1",   "mining": True,  "me": 1,  "rs": 0.10, "re": 0.30, "rw": 20, "pw": 2.0, "bs": 1.2},
    {"id": "B2-2", "name": "mining_every_5",   "mining": True,  "me": 5,  "rs": 0.10, "re": 0.30, "rw": 20, "pw": 2.0, "bs": 1.2},
    {"id": "B2-3", "name": "mining_every_10",  "mining": True,  "me": 10, "rs": 0.10, "re": 0.30, "rw": 20, "pw": 2.0, "bs": 1.2},
    {"id": "B2-4", "name": "mining_every_20",  "mining": True,  "me": 20, "rs": 0.10, "re": 0.30, "rw": 20, "pw": 2.0, "bs": 1.2},
]

# B3: 双重加权机制消融
# 默认: mining=True, every=5, start=0.10, end=0.30, warmup=20
B3_EXPERIMENTS = [
    {"id": "B3-0", "name": "weight_1.0_scale_1.0", "mining": True, "me": 5, "rs": 0.10, "re": 0.30, "rw": 20, "pw": 1.0, "bs": 1.0},
    {"id": "B3-1", "name": "weight_2.0_scale_1.0", "mining": True, "me": 5, "rs": 0.10, "re": 0.30, "rw": 20, "pw": 2.0, "bs": 1.0},
    {"id": "B3-2", "name": "weight_1.0_scale_1.2", "mining": True, "me": 5, "rs": 0.10, "re": 0.30, "rw": 20, "pw": 1.0, "bs": 1.2},
    {"id": "B3-3", "name": "weight_2.0_scale_1.2", "mining": True, "me": 5, "rs": 0.10, "re": 0.30, "rw": 20, "pw": 2.0, "bs": 1.2},
    {"id": "B3-4", "name": "weight_3.0_scale_1.5", "mining": True, "me": 5, "rs": 0.10, "re": 0.30, "rw": 20, "pw": 3.0, "bs": 1.5},
]

# B4: HNM 独立贡献（组件消融）
# 注: 基线(baseline)与B1-0相同，不复跑，结果复用B1-0
B4_EXPERIMENTS = [
    {"id": "B4-1", "name": "plus_global_mining",       "mining": True,  "me": 5, "rs": 0.0,  "re": 0.0,  "rw": 0,  "pw": 1.0, "bs": 1.0},
    {"id": "B4-2", "name": "plus_mixed_sampler",       "mining": True,  "me": 5, "rs": 0.10, "re": 0.10, "rw": 0,  "pw": 1.0, "bs": 1.0},
    {"id": "B4-3", "name": "plus_curriculum",          "mining": True,  "me": 5, "rs": 0.10, "re": 0.30, "rw": 20, "pw": 1.0, "bs": 1.0},
    {"id": "B4-4", "name": "plus_dual_weight_fullhnm", "mining": True,  "me": 5, "rs": 0.10, "re": 0.30, "rw": 20, "pw": 2.0, "bs": 1.2},
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


def run_exp(exp_id, exp_name, mining, me, rs, re, rw, pw, bs,
            epoch, patience, folds, log_file):
    """执行单个子实验，通过环境变量传参"""
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"\n{'='*60}")
    print(f"  [{ts}] {exp_id} | {exp_name}")
    print(f"  mining={mining}, me={me}, ratio={rs}->{re}({rw}), pw={pw}, bs={bs}")
    print(f"{'='*60}")

    ensure_dir(os.path.dirname(log_file))

    with open(log_file, "w") as log_fh:
        log_fh.write(f"# {exp_id} | {exp_name}\n")
        log_fh.write(f"# mining={mining} me={me} ratio={rs}->{re} warmup={rw} pw={pw} bs={bs}\n")
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
            env["COMET_HNM_ENABLED"] = "1" if mining else "0"
            env["COMET_HNM_MINING_EVERY"] = str(me)
            env["COMET_HNM_RATIO_START"] = str(rs)
            env["COMET_HNM_RATIO_END"] = str(re)
            env["COMET_HNM_RATIO_WARMUP"] = str(rw)
            env["COMET_HNM_PAIR_WEIGHT"] = str(pw)
            env["COMET_HNM_BATCH_SCALE"] = str(bs)

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
            exp["mining"], exp["me"], exp["rs"], exp["re"], exp["rw"],
            exp["pw"], exp["bs"],
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
    print("  B1/B2/B3/B4 实验调度 (HNM 消融)")
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
        # B1: hard_ratio 消融
        ok = run_group("B1 (hard_ratio 消融)", B1_EXPERIMENTS,
                       epoch=150, patience=1000,
                       folds=["fold_V0", "fold_V1", "fold_V2"])
        all_ok.append(("B1", ok))

        # B2: mining 频率消融
        if ok:
            ok = run_group("B2 (mining 频率消融)", B2_EXPERIMENTS,
                           epoch=150, patience=1000,
                           folds=["fold_V0", "fold_V1", "fold_V2"])
        all_ok.append(("B2", ok))

        # B3: 双重加权机制消融
        if ok:
            ok = run_group("B3 (双重加权机制消融)", B3_EXPERIMENTS,
                           epoch=150, patience=1000,
                           folds=["fold_V0", "fold_V1", "fold_V2"])
        all_ok.append(("B3", ok))

        # B4: HNM 独立贡献
        if ok:
            ok = run_group("B4 (HNM 独立贡献)", B4_EXPERIMENTS,
                           epoch=150, patience=1000,
                           folds=["fold_V0", "fold_V1", "fold_V2"])
        all_ok.append(("B4", ok))

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
