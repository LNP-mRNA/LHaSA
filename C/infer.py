#!/usr/bin/env python3
"""
infer.py -- 外层推理脚本
============================
- 调用 infer_ensemble.py 进行5次取中位数推理
- 集成日志系统，简洁输出
- 日志文件：./logs/YYYYMMDD/infer_{timestamp}_{subset}.log

用法：
    python infer.py
"""

import subprocess
import os
import sys
import time
import json
import re
from datetime import datetime

# ==================== 日志系统 ====================
from utils.logger import setup_logger, print_header, print_section, print_result, get_log_dir

# ==================== 配置区 ====================
data_path = './'
dict_name = 'dict.txt'
task_num = 1
dropout = 0.1
warmup = 0.06
local_batch_size = 32
only_polar = 0
conf_size = 11
seed = 0

full_dataset_task_schema_path = "task_schemas/in_house_lnp_master_schema_NPratio_AOvolratio.json"
lnp_encoder_attention_heads = 8
lnp_encoder_ffn_embed_dim = 256
lnp_encoder_embed_dim = 256
lnp_encoder_layers = 8
loss_func = 'np_finetune_contrastive'
eval_batch_size = 256

task_name = 'processed_data_dirs/OS_demo_in_house_lnp_data_overall_new_full_without_pbae_NPratios_updated_09222023_npratios_09252023gen_fig3di/fold_V0'

model_dir = './save_demo/save_demo_in_house_ED09262023_fig3dii_fold_V0_lnp_np_finetune_contrastive-bs128-lr0.0001-lnpmodparams8-256-256-8-trainrat1-ep200-pat20-metricvalid_spearmanr_coeff-cagrad0.2-percentnoise0.1-labelmargin0.01-seed1_OS'

# SWA推理支持：优先使用SWA检查点
swa_weight_path = os.path.join(model_dir, 'checkpoint_swa.pt')
if os.path.exists(swa_weight_path):
    weight_path = swa_weight_path
    use_swa_infer = True
    print(f"[SWA] 使用SWA检查点进行推理: {weight_path}")
else:
    weight_path = os.path.join(model_dir, 'checkpoint_best.pt')
    use_swa_infer = False

output_root = './eval_results'
NUM_RUNS = 5

# ==================== 子集列表 ====================
subsets = ['test', 'infer']

# ==================== 主程序 ====================
if __name__ == '__main__':
    if not os.path.exists(weight_path):
        print(f"[ERROR] Model weight not found: {weight_path}")
        sys.exit(1)

    # 获取时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 设置日志（使用第一个子集名作为日志标识）
    subset_tag = subsets[0] if len(subsets) == 1 else '_'.join(subsets)
    log_file = setup_logger("infer", subset_tag, timestamp=timestamp)

    print_header("ENSEMBLE INFERENCE START")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Model: {weight_path}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Subsets: {subsets}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Runs: {NUM_RUNS} (median aggregation)")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Output: {output_root}")

    # 对每个子集调用 infer_ensemble.py
    for subset in subsets:
        ts = datetime.now().strftime("%H:%M:%S")
        print("")
        print(f"[{ts}] {'='*60}")
        print(f"[{ts}] Subset: {subset}")
        print(f"[{ts}] {'='*60}")

        results_dir = os.path.join(output_root, subset)
        os.makedirs(results_dir, exist_ok=True)

        # 构建命令行参数
        cmd = (
            f"python ../unimol/infer_ensemble.py --user-dir ../unimol {data_path} "
            f"--task-name {task_name} --valid-subset {subset} "
            f"--num-workers 8 --ddp-backend=c10d --batch-size {eval_batch_size} "
            f"--task mol_np_finetune --loss {loss_func} --arch np_unimol "
            f"--classification-head-name {task_name} --num-classes {task_num} "
            f"--dict-name {dict_name} --conf-size {conf_size} "
            f"--only-polar {only_polar} "
            f"--path {weight_path} "
            f"--fp16 --fp16-init-scale 4 --fp16-scale-window 256 "
            f"--log-interval 50 --log-format simple "
            f"--results-path {results_dir} "
            f"--lnp-encoder-layers {lnp_encoder_layers} --lnp-encoder-embed-dim {lnp_encoder_embed_dim} "
            f"--lnp-encoder-ffn-embed-dim {lnp_encoder_ffn_embed_dim} --lnp-encoder-attention-heads {lnp_encoder_attention_heads} "
            f"--full-dataset-task-schema-path {full_dataset_task_schema_path} "
            f"--load-full-np-model --concat-datasets"
            f"{' --use-swa' if use_swa_infer else ''}"
        )

        # 运行推理（实时捕获子进程输出到日志）
        t0 = time.time()
        print(f"[{ts}] Running: {cmd[:80]}...")
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in process.stdout:
            print(line, end='')
        process.wait()
        if process.returncode != 0:
            print(f"[WARNING] infer_ensemble.py exited with code {process.returncode}")
        elapsed = time.time() - t0

        ts = datetime.now().strftime("%H:%M:%S")
        print(f"[{ts}] Subset '{subset}' complete | Time: {elapsed:.1f}s")

        # 尝试读取聚合后的 JSON 结果
        fname = os.path.basename(model_dir)
        json_path = os.path.join(results_dir, f"{fname}_{subset}.json")
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    results = json.load(f)

                # 提取关键指标（精确匹配 spearmanr_coeff，排除 p_val）
                b16_sp = None
                dc24_sp = None
                for k, v in results.items():
                    if "B16F10" in k and "spearmanr_coeff" in k:
                        b16_sp = float(v)
                    if "DC24" in k and "spearmanr_coeff" in k:
                        dc24_sp = float(v)

                sep = "=" * 60
                if b16_sp is not None and dc24_sp is not None:
                    # 高亮结果面板
                    print(f"\n{sep}")
                    print(f"🎯 ENSEMBLE RESULTS | Subset: {subset}")
                    print(f"{sep}")
                    print(f"  ⭐ B16F10 Spearman: {b16_sp:.4f}")
                    print(f"  ⭐ DC24   Spearman: {dc24_sp:.4f}")
                    print(f"  📝 JSON: {json_path}")
                    print(f"{sep}\n")
                else:
                    print(f"\n{sep}")
                    print(f"📁 PREDICTIONS SAVED | Subset: {subset}")
                    print(f"  📂 Results: {json_path}")
                    print(f"  💡 (infer subset has no ground truth labels)")
                    print(f"{sep}\n")
            except Exception as e:
                print(f"[{ts}] WARN | Failed to read results: {e}")

    # ==================== 汇总 ====================
    print_header("ENSEMBLE INFERENCE COMPLETE")
    print_result("Subsets processed", ", ".join(subsets))
    print_result("Runs per subset", NUM_RUNS)
    print_result("Aggregation", "median")
    print_result("Output root", output_root)
    print_result("Log file", log_file)
