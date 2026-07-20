#!/usr/bin/env python3 -u
# Copyright (c) DP Techonology, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
COMBINE4: Ensemble Inference Script (MC Dropout)
=================================================
基于 MC Dropout 的多次推理聚合：
1. 执行 NUM_RUNS 次推理（默认 5 次），每次使用不同的 dropout 随机种子
2. 收集每次推理的预测结果
3. 对所有运行的预测结果取中位数 (median) 作为最终预测

支持 SWA 检查点加载，通过 --use-swa 标志启用。

用法：
    python infer_ensemble.py --valid-subset infer --path <model.pt> --results-path <out_dir>
    python infer_ensemble.py --valid-subset infer --path <swa_model.pt> --results-path <out_dir> --use-swa
"""

import logging
import os
import sys
import pickle
import torch
import random
import numpy as np

import json

import importlib
from pyprojroot import here as project_root
sys.path.insert(0, str(project_root()))
importlib.import_module('unimol')

from unimol.core import checkpoint_utils, distributed_utils, options, utils
from unimol.core.logging import progress_bar
from unimol.core import tasks

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("unimol.inference")

# ========== 【集成推理配置】==========
NUM_RUNS = 5  # 默认推理次数（MC Dropout 采样次数）
# ======================================


def set_seed(seed=0):
    """固定所有随机种子，确保 dropout 掩码等随机操作可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def aggregate_median(all_run_predictions):
    """
    对多次推理的结果取中位数聚合。

    参数:
        all_run_predictions: List[dict]，每个元素是一次 run 的 reduced_metrics_dict

    返回:
        aggregated_metrics: dict，中位数聚合后的结果
    """
    # 获取所有 key
    all_keys = set()
    for pred_dict in all_run_predictions:
        all_keys.update(pred_dict.keys())

    aggregated_metrics = {}
    for key in all_keys:
        # 收集所有 run 中该 key 的值
        values_list = []
        for pred_dict in all_run_predictions:
            if key in pred_dict:
                val = pred_dict[key]
                # 将 torch.Tensor 转为 numpy array
                if hasattr(val, 'numpy'):
                    val = val.numpy()
                values_list.append(val)

        if len(values_list) == 0:
            continue

        # 尝试数值型中位数聚合
        try:
            stacked = np.stack(values_list)  # shape: [NUM_RUNS, ...]
            median_val = np.median(stacked, axis=0)
            aggregated_metrics[key] = median_val
        except Exception as e:
            # 非数值型（如字符串、不规则列表等），取第一次的结果
            aggregated_metrics[key] = values_list[0]
            print(f"[WARN] Key '{key}' 无法计算中位数，使用第一次结果: {e}")

    return aggregated_metrics


def main(args):
    # ========== 【推理模式修复】==========
    # 强制 mode='infer'，使数据集构象选择变为确定性的 [:conf_size]
    # 避免 train 模式下的 random.sample 导致每次输入不同
    args.mode = 'infer'
    # ========================================

    # ========== 【COMBINE4 模块开关日志】==========
    # 读取环境变量仅用于日志记录，推理本身不需要切换模块
    use_swa_env = os.environ.get('COMET_USE_SWA', '0') == '1'
    print(f"[COMBINE4 Ensemble Infer] SWA mode (env): {use_swa_env}")
    # ========================================

    assert (
        args.batch_size is not None
    ), "Must specify batch size either with --batch-size"

    use_fp16 = args.fp16
    use_cuda = torch.cuda.is_available() and not args.cpu

    if use_cuda:
        torch.cuda.set_device(args.device_id)

    if args.distributed_world_size > 1:
        data_parallel_world_size = distributed_utils.get_data_parallel_world_size()
        data_parallel_rank = distributed_utils.get_data_parallel_rank()
    else:
        data_parallel_world_size = 1
        data_parallel_rank = 0

    # ========== 【模型加载（只加载一次）】==========
    logger.info("loading model(s) from {}".format(args.path))
    state = checkpoint_utils.load_checkpoint_to_cpu(args.path)
    task = tasks.setup_task(args)
    model = task.build_model(args)

    # ===== SWA: 智能检查点加载 =====
    model_key = None
    if "model" in state:
        model_key = "model"
    elif "model_state_dict" in state:
        model_key = "model_state_dict"
    if model_key is not None:
        if getattr(args, 'use_swa', False) and "swa_state_dict" in state:
            print(f"[SWA] 检测到 SWA 检查点，加载 SWA 平均权重 (n_averaged={state.get('swa_n_averaged', 'unknown')})")
            model_key = "swa_state_dict"
        model.load_state_dict(state[model_key], strict=False)
        print(f"[INFO] 模型权重已加载: {args.path} (key='{model_key}')")
    else:
        raise RuntimeError(f"检查点中未找到模型权重: {args.path}")
    # =================================

    # Move models to GPU
    if use_fp16:
        model.half()
    if use_cuda:
        model.cuda()
    # ==========================================

    logger.info(args)

    # Build loss
    loss = task.build_loss(args)
    loss.eval()
    print("loss: ", loss)

    for subset in args.valid_subset.split(","):
        try:
            if args.concat_datasets:
                task.load_concat_dataset(subset, combine=False, epoch=1)
            else:
                task.load_dataset(subset, combine=False, epoch=1)
            dataset = task.dataset(subset)
            print("dataset len: ", len(dataset))
        except KeyError:
            raise Exception("Cannot find dataset: " + subset)

        if not os.path.exists(args.results_path):
            os.makedirs(args.results_path)
        fname = (args.path).split("/")[-2]
        save_path = os.path.join(args.results_path, fname + "_" + subset + ".out.pkl")
        json_save_path = os.path.join(args.results_path, fname + "_" + subset + ".json")

        # ========== 【多次推理，收集预测】==========
        all_run_predictions = []  # List[dict]，每个元素是一次 run 的 reduced_metrics_dict

        for run_idx in range(NUM_RUNS):
            print(f"\n{'='*60}")
            print(f"Ensemble Run {run_idx+1}/{NUM_RUNS}")
            print(f"{'='*60}")

            # 每次 run 重新创建数据迭代器（因为迭代器只能遍历一次）
            itr = task.get_batch_iterator(
                dataset=dataset,
                batch_size=args.batch_size,
                ignore_invalid_inputs=True,
                required_batch_size_multiple=args.required_batch_size_multiple,
                seed=args.seed,
                num_shards=data_parallel_world_size,
                shard_id=data_parallel_rank,
                num_workers=args.num_workers,
                data_buffer_size=args.data_buffer_size,
            ).next_epoch_itr(shuffle=False)
            progress = progress_bar.progress_bar(
                itr,
                log_format=args.log_format,
                log_interval=args.log_interval,
                prefix=f"valid on '{subset}' subset (run {run_idx+1}/{NUM_RUNS})",
                default_log_format=("tqdm" if not args.no_progress_bar else "simple"),
            )

            log_outputs = []
            for i, sample in enumerate(progress):
                sample = utils.move_to_cuda(sample) if use_cuda else sample
                if len(sample) == 0:
                    continue
                # 每次 batch 前固定种子，确保每次 run 的 dropout 掩码不同
                # 种子设计: args.seed + batch_idx + run_idx * 1000
                # 这样同一 run 内不同 batch 种子不同，不同 run 之间种子也不同
                set_seed(args.seed + i + run_idx * 1000)
                _, _, log_output = task.valid_step(
                    sample, model, loss,
                    test=True,
                    infer=(subset == "infer"),
                    output_cls_rep=args.output_cls_rep,
                )
                progress.log({}, step=i)
                log_outputs.append(log_output)

            # 聚合本次 run 的 metrics
            reduced_metrics_dict = task.reduce_metrics(
                log_outputs, loss, subset, infer=(subset == "infer")
            )
            all_run_predictions.append(reduced_metrics_dict)

            # 【诊断】打印本次 run 的预测信息
            print(f"Run {run_idx+1}/{NUM_RUNS} 完成")
            print("reduced_metrics_dict keys: ", reduced_metrics_dict.keys())
            for k in reduced_metrics_dict:
                if "cls_" in k:
                    print(k, " shape, run {}: ".format(run_idx+1), reduced_metrics_dict[k].shape)
            # 检测预测值中的 NaN/Inf
            for k in reduced_metrics_dict:
                if "_predict" in k:
                    arr = reduced_metrics_dict[k]
                    if hasattr(arr, 'dtype'):
                        nan_count = np.isnan(arr).sum() if 'float' in str(arr.dtype) else 0
                        inf_count = np.isinf(arr).sum() if 'float' in str(arr.dtype) else 0
                        if nan_count > 0 or inf_count > 0:
                            print(f"[WARNING] Run {run_idx+1} {k}: NaN={nan_count}, Inf={inf_count} out of {arr.size}")
        # ==========================================

        # ========== 【中位数聚合】==========
        print(f"\n{'='*60}")
        print(f"聚合 {NUM_RUNS} 次推理结果 (median)")
        print(f"{'='*60}")

        aggregated_metrics = aggregate_median(all_run_predictions)

        # 打印聚合后的关键指标
        for k in aggregated_metrics:
            if "cls_" in k:
                print(k, " shape, aggregated: ", aggregated_metrics[k].shape)
        # 检测聚合后预测值中的 NaN/Inf
        for k in aggregated_metrics:
            if "_predict" in k:
                arr = aggregated_metrics[k]
                if hasattr(arr, 'dtype'):
                    nan_count = np.isnan(arr).sum() if 'float' in str(arr.dtype) else 0
                    inf_count = np.isinf(arr).sum() if 'float' in str(arr.dtype) else 0
                    if nan_count > 0 or inf_count > 0:
                        print(f"[WARNING] Aggregated {k}: NaN={nan_count}, Inf={inf_count} out of {arr.size}")

        cor_only_reduced_metrics_dict = {
            k: aggregated_metrics[k]
            for k in aggregated_metrics
            if ("spearman" in k or "pearson" in k or "accuracy" in k)
        }
        # ==========================================

        # ========== 【保存结果】==========
        pickle.dump(aggregated_metrics, open(save_path, "wb"))
        with open(json_save_path, "w") as outfile:
            json.dump(cor_only_reduced_metrics_dict, outfile, indent=4)

        print(f"聚合完成 | 结果保存: {save_path}")
        print(f"JSON 指标保存: {json_save_path}")
        # ==========================================

        logger.info("Done ensemble inference! ")
    return None


def cli_main():
    parser = options.get_validation_parser()
    options.add_model_args(parser)
    # 添加 SWA 参数支持
    parser.add_argument(
        "--use-swa",
        action="store_true",
        default=False,
        help="使用 SWA (Stochastic Weight Averaging) 平均权重进行推理",
    )
    args = options.parse_args_and_arch(parser)
    # 强制将 mode 设为 infer，禁用噪声增强
    args.mode = 'infer'
    distributed_utils.call_main(args, main)


if __name__ == "__main__":
    cli_main()
