#!/usr/bin/env python3 -u
# Copyright (c) DP Techonology, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
COMBINE4: NP (Nanoparticle) Inference Script
=============================================
Baseline inference script with SWA-aware checkpoint loading.

Features:
- SWA-aware checkpoint loading (auto-detects SWA checkpoints)
- Deterministic inference via set_seed()
- NaN/Inf diagnostics on predictions
- COMBINE4 module switch logging via environment variables

Usage:
    python infer_np.py --valid-subset infer --path <model.pt> --results-path <out_dir>
    python infer_np.py --valid-subset infer --path <swa_model.pt> --results-path <out_dir>
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


def set_seed(seed=0):
    """固定所有随机种子，确保推理结果可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main(args):
    # ========== 【推理模式修复】==========
    # 强制 mode='infer'，使数据集构象选择变为确定性的 [:conf_size]
    # 避免 train 模式下的 random.sample 导致每次输入不同
    args.mode = 'infer'
    # ========================================

    # ========== 【COMBINE4 模块开关日志】==========
    # 读取环境变量仅用于日志记录，推理本身不需要切换模块
    use_swa = os.environ.get('COMET_USE_SWA', '0') == '1'
    print(f"[COMBINE4 Infer] SWA mode: {use_swa}")
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

    # Load model
    logger.info("loading model(s) from {}".format(args.path))
    state = checkpoint_utils.load_checkpoint_to_cpu(args.path)
    task = tasks.setup_task(args)
    model = task.build_model(args)

    # ===== SWA: 智能检查点加载 =====
    # 检查点可能来自：
    # 1. 普通检查点: state["model"] 包含模型权重
    # 2. SWA检查点: state["model_state_dict"] 包含原始权重, state["swa_state_dict"] 包含SWA权重
    # 3. 最佳检查点: state["model"] 或 state["model_state_dict"] 包含最佳权重

    model_key = None
    if "model" in state:
        model_key = "model"
    elif "model_state_dict" in state:
        model_key = "model_state_dict"

    if model_key is not None:
        if getattr(args, 'use_swa', False) and "swa_state_dict" in state:
            print(f"[SWA] 检测到SWA检查点，加载SWA平均权重 (n_averaged={state.get('swa_n_averaged', 'unknown')})")
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
            prefix=f"valid on '{subset}' subset",
            default_log_format=("tqdm" if not args.no_progress_bar else "simple"),
        )
        log_outputs = []
        for i, sample in enumerate(progress):
            sample = utils.move_to_cuda(sample) if use_cuda else sample
            if len(sample) == 0:
                continue
            # 每次 batch 前固定种子，确保 dropout 掩码和任何随机操作确定性
            set_seed(args.seed + i)
            _, _, log_output = task.valid_step(sample, model, loss, test=True, infer=(subset == "infer"), output_cls_rep=args.output_cls_rep)
            progress.log({}, step=i)
            log_outputs.append(log_output)

        reduced_metrics_dict = task.reduce_metrics(log_outputs, loss, subset, infer=(subset == "infer"))
        print("reduced_metrics_dict keys: ", reduced_metrics_dict.keys())
        for k in reduced_metrics_dict:
            if "cls_" in k:
                print(k, " shape, infer_np: ", reduced_metrics_dict[k].shape)
        # 【诊断】检测预测值中的 NaN/Inf
        for k in reduced_metrics_dict:
            if "_predict" in k:
                arr = reduced_metrics_dict[k]
                if hasattr(arr, 'dtype'):
                    nan_count = np.isnan(arr).sum() if 'float' in str(arr.dtype) else 0
                    inf_count = np.isinf(arr).sum() if 'float' in str(arr.dtype) else 0
                    if nan_count > 0 or inf_count > 0:
                        print(f"[WARNING] {k}: NaN={nan_count}, Inf={inf_count} out of {arr.size}")
        cor_only_reduced_metrics_dict = {k: reduced_metrics_dict[k] for k in reduced_metrics_dict if ("spearman" in k or "pearson" in k or "accuracy" in k)}
        pickle.dump(reduced_metrics_dict, open(save_path, "wb"))
        with open(json_save_path, "w") as outfile:
            json.dump(cor_only_reduced_metrics_dict, outfile, indent = 4)
        
        logger.info("Done inference! ")
    return None


def cli_main():
    parser = options.get_validation_parser()
    options.add_model_args(parser)
    args = options.parse_args_and_arch(parser)
    # 强制将 mode 设为 infer，禁用噪声增强
    args.mode = 'infer'
    distributed_utils.call_main(args, main)


if __name__ == "__main__":
    cli_main()
