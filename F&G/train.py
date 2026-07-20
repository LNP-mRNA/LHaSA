#!/usr/bin/env python3
"""
train.py -- COMBINE4 外层训练脚本 (SWA + LOSS + HARD + AUG)
================================================================
- 调用 train_np.py 进行训练，然后调用 infer_np.py 进行推理
- 支持4个独立开关：SWA, LOSS(LambdaRank), HARD(Hard Negative Mining), AUG(Structural Augmentation)
- 支持多数据集切换：fig3dii(双标签), fig3dii_b16f10(B16F10单标签), fig3dii_dc24(DC24单标签)
- 通过环境变量传递开关状态（Uni-Core 的 parse_args_and_arch() 会拒绝未知参数）
- 使用 CONFIG_GRID + itertools.product 展平嵌套循环，避免 Python 20层嵌套限制
- 日志文件：./logs/YYYYMMDD/train_{timestamp}_{exp_name}.log

用法示例：
    ./run.sh combine4 train --use-swa --use-loss --use-hard --use-aug
    ./run.sh combine4 train --use-swa --use-loss --dataset fig3dii_b16f10
    ./run.sh combine4 train --use-hard --dataset fig3dii_dc24
    ./run.sh combine4 train --dataset fig3dii               # 默认，等同不加
    ./run.sh combine4 train                                # 基线（全关，fig3dii数据集）
任意组合共 2^4 = 16 种均可独立工作。
"""

import subprocess
import subprocess as sp
import os
import shutil
import sys
import argparse
from datetime import datetime, time as dt_time
import time
from itertools import product  # 用于展平多层嵌套循环

# ==================== 日志系统 ====================
from utils.logger import setup_logger, print_header, print_section, print_result, get_log_dir


def run_with_log_capture(cmd):
    """运行命令并实时捕获输出到日志（通过print走Tee），避免增加嵌套层级"""
    process = sp.Popen(cmd, shell=True, stdout=sp.PIPE, stderr=sp.STDOUT, text=True, bufsize=1)
    for line in process.stdout:
        print(line, end='')
    process.wait()
    if process.returncode != 0:
        print(f"[WARNING] Command exited with code {process.returncode}")
    return process.returncode


# ==================== 数据集配置表 ====================
# 支持的数据集及其对应路径配置
DATASET_CONFIGS = {
    "fig3dii": {
        "root_task_name": "processed_data_dirs/OS_demo_in_house_lnp_data_overall_new_full_without_pbae_NPratios_updated_09222023_npratios_09252023gen_fig3di",
        "exp_prefix": "demo_in_house_ED09262023_fig3dii",
        "label": "dual",
    },
    "fig3di_b16f10": {
        "root_task_name": "processed_data_dirs/OS_demo_in_house_lnp_data_overall_lance_B16F10_only_09252023gen_fig3di",
        "exp_prefix": "demo_in_house_B16F10_ED09262023_fig3di",
        "label": "B16F10",
        "schema": "task_schemas/in_house_lnp_master_schema_NPratio_B16F10.json",
    },
    "fig3di_dc24": {
        "root_task_name": "processed_data_dirs/OS_demo_in_house_lnp_data_overall_lance_DC24_only_09252023gen_fig3di",
        "exp_prefix": "demo_in_house_DC24_ED09262023_fig3di",
        "label": "DC24",
        "schema": "task_schemas/in_house_lnp_master_schema_NPratio_DC24.json",
    },
}

# ==================== 解析4个独立开关 + 数据集选择 ====================
parser = argparse.ArgumentParser(
    description="COMBINE4 outer training script: SWA + LOSS + HARD + AUG",
    formatter_class=argparse.RawDescriptionHelpFormatter,
)
parser.add_argument(
    "fold", nargs="?", default="fold_V0",
    help="Fold name (default: fold_V0)"
)
parser.add_argument(
    "--use-swa", action="store_true", default=False,
    help="Enable SWA (Stochastic Weight Averaging)"
)
parser.add_argument(
    "--use-loss", action="store_true", default=False,
    help="Enable LambdaRank mixed loss"
)
parser.add_argument(
    "--use-hard", action="store_true", default=False,
    help="Enable Hard Negative Mining"
)
parser.add_argument(
    "--use-aug", action="store_true", default=False,
    help="Enable Structural Augmentation"
)
parser.add_argument(
    "--dataset", type=str, default="fig3dii",
    choices=list(DATASET_CONFIGS.keys()),
    help=f"Dataset to use. Choices: {', '.join(DATASET_CONFIGS.keys())} (default: fig3dii)"
)
args, _ = parser.parse_known_args()

fold = args.fold
use_swa = args.use_swa
use_loss = args.use_loss
use_hard = args.use_hard
use_aug = args.use_aug
dataset_key = args.dataset

# 获取数据集配置
ds_cfg = DATASET_CONFIGS[dataset_key]

# ==================== 全局配置 ====================
data_path = './'
MASTER_PORT = 10086
n_gpu = 1
dict_name = 'dict.txt'
weight_path = '../ckp/mol_pre_no_h_220816.pt'
task_num = 1
local_batch_size = 128
only_polar = 0
conf_size = 11

root_task_name = f"{ds_cfg['root_task_name']}/{fold}"
root_save_dir = './save_demo'
root_tmp_save_dir = './tmp_save_demo'

metric = "valid_spearmanr_coeff"
lr = 1e-5
batch_size = 128
local_batch_size = batch_size
update_freq = batch_size / local_batch_size
warmup = 0.06
dropout = 0.1
loss_sample_dropout = 0.2
epoch = 150

# 自动选择 schema：单标签数据集用单标签 schema，双标签用默认 schema
full_dataset_task_schema_path = ds_cfg.get(
    "schema",
    "task_schemas/in_house_lnp_master_schema_NPratio_AOvolratio.json"
)
save_all_model_weights = True

# ===================== CONFIG_GRID 配置网格 =====================
# 基础超参数 + 4个模块各自的配置列表
# 所有配置项统一放列表中，通过 itertools.product 展平循环
CONFIG_GRID = {
    # ---- 基础模型超参数 ----
    'seed': [1],
    'lnp_encoder_attention_heads': [8],
    'lnp_encoder_ffn_embed_dim': [256],
    'lnp_encoder_embed_dim': [256],
    'lnp_encoder_layers': [8],
    'warmup': [0.06],
    'dropout': [0.1],
    'epoch': [150],
    'lr': [1e-4],
    'batch_size': [128],
    'loss_sample_dropout': [0],
    'loss_func': ['np_finetune_contrastive'],
    'cagrad_c': [0.2],
    'epoch_to_freeze_molecule_encoder': [1000000],
    'subdataset_patience': [-1],
    'contrast_margin_coeff': [0.01],
    'percent_noise': [0.1],
    'percent_noise_type': ['normal_proportionate'],
    'patience': [1000],
    'train_data_ratio': [1],

    # ===================== SWA 配置 =====================
    'use_swa': [use_swa],          # 从命令行开关注入
    'swa_start': [60],              # SWA起始epoch***40
    'swa_lr': [1e-5],               # SWA学习率***1e-4
    'swa_anneal_epochs': [5],       # 退火epoch数
    'swa_anneal_strategy': ['cos'], # 退火策略
    # ===================================================

    # ===================== LOSS (LambdaRank) 配置 =====================
    'use_loss': [use_loss],              # 从命令行开关注入
    'lambdarank_alpha': [0.1],            # 混合比例***0.3
    'grad_clip_norm': [1.0],              # 梯度裁剪
    # ================================================================

    # ===================== HARD (Hard Negative Mining) 配置 =====================
    'use_hard': [use_hard],                     # 从命令行开关注入
    'hnm_top_k': [2000],                         # 困难样本池容量
    'hnm_mining_every_n_epochs': [10],            # 每N个epoch触发mining***5
    'hnm_eta': [1e-6],                           # 困难度分数数值稳定项
    'hnm_label_gap_threshold': [0.15],           # 最小标签差距阈值
    'hnm_hard_ratio_start': [0.10],              # 课程式起始hard比例
    'hnm_hard_ratio_end': [0.30],                # 课程式最终hard比例
    'hnm_hard_ratio_warmup_epochs': [20],        # 过渡轮数
    'hnm_hard_pair_weight': [2.0],               # 困难样本对损失权重乘数
    'hnm_batch_loss_scale': [1.0],               # 困难batch整体损失放大系数***1.2
    # ==========================================================================

    # ===================== AUG (结构化数据增强) 配置 =====================
    'use_aug': [use_aug],                       # 从命令行开关注入
    'structural_aug_scheduler': [True],          # 使用课程式概率调度
    'structural_aug_static_prob': [0.5],         # 静态增强概率
    # ==================================================================
}

# 计算总实验数
total_exp = 1
for v in CONFIG_GRID.values():
    total_exp *= len(v)

# 获取基础时间戳（用于日志文件名）
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# 设置日志（每个fold独立一个日志文件）
modules_tag_all = f"S{int(use_swa)}L{int(use_loss)}H{int(use_hard)}A{int(use_aug)}"
base_exp_name = f"{ds_cfg['exp_prefix']}_{fold}_{modules_tag_all}"
log_file = setup_logger("train", base_exp_name, timestamp=timestamp)

# ===================== 打印启动信息 =====================
print_header("TRAINING START (COMBINE4)")
print(f"[{datetime.now().strftime('%H:%M:%S')}] Total experiments: {total_exp}")
print(f"[{datetime.now().strftime('%H:%M:%S')}] Dataset : {dataset_key} ({ds_cfg['label']})")
print(f"[{datetime.now().strftime('%H:%M:%S')}] Task    : {root_task_name}")
print(f"[{datetime.now().strftime('%H:%M:%S')}] Save dir: {root_save_dir}")
print(f"[{datetime.now().strftime('%H:%M:%S')}] Pretrained weight: {weight_path}")
print(f"[{datetime.now().strftime('%H:%M:%S')}] Fold: {fold}")

# 打印开关状态
swa_status = "ON" if use_swa else "OFF"
loss_status = "ON" if use_loss else "OFF"
hard_status = "ON" if use_hard else "OFF"
aug_status = "ON" if use_aug else "OFF"
print(f"[{datetime.now().strftime('%H:%M:%S')}] SWA : {swa_status}")
print(f"[{datetime.now().strftime('%H:%M:%S')}] LOSS: {loss_status}")
print(f"[{datetime.now().strftime('%H:%M:%S')}] HARD: {hard_status}")
print(f"[{datetime.now().strftime('%H:%M:%S')}] AUG : {aug_status}")

# ===================== 主循环（展平为单层） =====================
exp_idx = 0
skipped = 0
completed = 0

for combo in product(*CONFIG_GRID.values()):
    # 解包配置组合
    c = dict(zip(CONFIG_GRID.keys(), combo))

    # ---- 基础超参数 ----
    seed = c['seed']
    lnp_encoder_attention_heads = c['lnp_encoder_attention_heads']
    lnp_encoder_ffn_embed_dim = c['lnp_encoder_ffn_embed_dim']
    lnp_encoder_embed_dim = c['lnp_encoder_embed_dim']
    lnp_encoder_layers = c['lnp_encoder_layers']
    warmup = c['warmup']
    dropout = c['dropout']
    epoch = c['epoch']
    lr = c['lr']
    batch_size = c['batch_size']
    loss_sample_dropout = c['loss_sample_dropout']
    loss_func = c['loss_func']
    cagrad_c = c['cagrad_c']
    epoch_to_freeze_molecule_encoder = c['epoch_to_freeze_molecule_encoder']
    subdataset_patience = c['subdataset_patience']
    contrast_margin_coeff = c['contrast_margin_coeff']
    percent_noise = c['percent_noise']
    percent_noise_type = c['percent_noise_type']
    patience = c['patience']
    train_data_ratio = c['train_data_ratio']

    # ---- SWA 参数 ----
    swa_start = c['swa_start']
    swa_lr = c['swa_lr']
    swa_anneal_epochs = c['swa_anneal_epochs']
    swa_anneal_strategy = c['swa_anneal_strategy']

    # ---- LOSS 参数 ----
    lambdarank_alpha = c['lambdarank_alpha']
    grad_clip_norm = c['grad_clip_norm']

    # ---- HARD 参数 ----
    hnm_top_k = c['hnm_top_k']
    hnm_mining_every_n_epochs = c['hnm_mining_every_n_epochs']
    hnm_eta = c['hnm_eta']
    hnm_label_gap_threshold = c['hnm_label_gap_threshold']
    hnm_hard_ratio_start = c['hnm_hard_ratio_start']
    hnm_hard_ratio_end = c['hnm_hard_ratio_end']
    hnm_hard_ratio_warmup_epochs = c['hnm_hard_ratio_warmup_epochs']
    hnm_hard_pair_weight = c['hnm_hard_pair_weight']
    hnm_batch_loss_scale = c['hnm_batch_loss_scale']

    # ---- AUG 参数 ----
    structural_aug_scheduler = c['structural_aug_scheduler']
    structural_aug_static_prob = c['structural_aug_static_prob']

    task_name = root_task_name

    # set up batch_size
    local_batch_size = batch_size
    update_freq = batch_size / local_batch_size

    # compensate max_epoch with loss_sample_dropout
    max_epoch = int(epoch // (1 - loss_sample_dropout))

    # ===================== 构建实验标签（缩短防止路径超长）=====================
    # 模块开关编码: S=SWA, L=LOSS, H=HARD, A=AUG
    modules_code = f"S{int(use_swa)}L{int(use_loss)}H{int(use_hard)}A{int(use_aug)}"
    
    exp_name = (
        f"{ds_cfg['exp_prefix']}_{fold}"
        f"_{modules_code}"
        f"_seed{seed}"
    )

    exp_idx += 1
    ts = datetime.now().strftime('%H:%M:%S')

    print(f"\n[{ts}] task_name: {task_name}")

    if save_all_model_weights:
        save_path = 'save_' + exp_name
        save_dir = os.path.join(root_save_dir, save_path)
        tmp_save_dir = os.path.join(root_tmp_save_dir, save_path)
    else:
        save_dir = os.path.join(root_save_dir, task_name)
        tmp_save_dir = os.path.join(root_tmp_save_dir, task_name)

    # tensorboard log path
    log_path = 'log_' + exp_name
    logdir = os.path.join("./logs/tmp/", log_path)

    # infer output path (加时间戳防止重复运行覆盖)
    results_folder = 'infer_' + exp_name + '_' + timestamp
    eval_results_path = os.path.join("./infer_results/", results_folder)
    eval_weight_path = os.path.join(save_dir, 'checkpoint_best.pt')

    # ===================== 跳过已完成的实验 =====================
    if os.path.exists(eval_results_path):
        print(f"[{ts}] [{exp_idx}/{total_exp}] SKIP | Infer output exists: {eval_results_path}")
        skipped += 1
        continue
    elif os.path.exists(logdir):
        print(f"[{ts}] [{exp_idx}/{total_exp}] RERUN | Tensorboard log exists but inference not done. Removing: {logdir}")
        shutil.rmtree(logdir)

    print(f"[{ts}] [{exp_idx}/{total_exp}] NEW  | Exp: {exp_name}")

    if os.path.exists(save_dir) and os.path.isdir(save_dir):
        shutil.rmtree(save_dir)

    if os.path.exists(tmp_save_dir) and os.path.isdir(tmp_save_dir):
        shutil.rmtree(tmp_save_dir)

    print(f"[{ts}] [{exp_idx}/{total_exp}] RUN  | Training: {logdir}")

    # ===================== 设置环境变量（传递给子进程） =====================
    # SWA 开关
    os.environ['COMET_USE_SWA'] = '1' if use_swa else '0'

    # LOSS (LambdaRank) 开关 + 详细参数
    os.environ['COMET_USE_LOSS'] = '1' if use_loss else '0'
    os.environ['COMET_USE_LAMBDARANK'] = '1' if use_loss else '0'
    os.environ['COMET_LAMBDARANK_ALPHA'] = str(lambdarank_alpha)
    os.environ['COMET_GRAD_CLIP_NORM'] = str(grad_clip_norm)
    os.environ['COMET_LAMBDARANK_K'] = '10'

    # HARD 开关
    os.environ['COMET_USE_HARD'] = '1' if use_hard else '0'

    # AUG 开关
    os.environ['COMET_USE_AUG'] = '1' if use_aug else '0'

    # ===================== Training =====================
    t0_train = time.time()

    # 构建结构化增强参数
    structural_aug_args = ""
    if use_aug:
        aug_parts = ["--use-structural-aug"]
        if structural_aug_scheduler:
            aug_parts.append("--structural-aug-scheduler")
        aug_parts.append(f"--structural-aug-static-prob {structural_aug_static_prob}")
        structural_aug_args = " ".join(aug_parts)

    train_cmd = (
        f"python ../COMBINE4_unimol/train_np.py {data_path} "
        f"--task-name {task_name} --user-dir ../COMBINE4_unimol "
        f"--train-subset train --valid-subset valid "
        f"--conf-size {conf_size} "
        f"--num-workers 4 --ddp-backend=c10d "
        f"--dict-name {dict_name} "
        f"--task mol_np_finetune --loss {loss_func} --arch np_unimol "
        f"--classification-head-name {task_name} --num-classes {task_num} "
        f"--optimizer adam --adam-betas '(0.9, 0.99)' --adam-eps 1e-6 --clip-norm 1.0 "
        f"--lr-scheduler polynomial_decay --lr {lr} --warmup-ratio {warmup} "
        f"--max-epoch {max_epoch} --batch-size {local_batch_size} --pooler-dropout {dropout} "
        f"--loss-sample-dropout {loss_sample_dropout} "
        f"--update-freq {update_freq} --seed {seed} "
        f"--fp16 --fp16-init-scale 4 --fp16-scale-window 256 "
        f"--log-interval 100 --log-format simple "
        f"--validate-interval 1 --keep-last-epochs 10 "
        f"--finetune-from-model {weight_path} "
        f"--best-checkpoint-metric {metric} --patience {patience} "
        f"--maximize-best-checkpoint-metric "
        f"--save-dir {save_dir} --tmp-save-dir {tmp_save_dir} --only-polar {only_polar} "
        f"--tensorboard-logdir {logdir} "
        f"--full-dataset-task-schema-path {full_dataset_task_schema_path} "
        f"--multitask-reg --cagrad-c {cagrad_c} "
        f"--epoch-to-freeze-molecule-encoder {epoch_to_freeze_molecule_encoder} "
        f"--concat-datasets "
        f"--train-data-ratio {train_data_ratio} "
        f"--lnp-encoder-layers {lnp_encoder_layers} "
        f"--lnp-encoder-embed-dim {lnp_encoder_embed_dim} "
        f"--lnp-encoder-ffn-embed-dim {lnp_encoder_ffn_embed_dim} "
        f"--lnp-encoder-attention-heads {lnp_encoder_attention_heads} "
        f"--noise-augment-percent --percent-noise {percent_noise} "
        f"--percent-noise-type {percent_noise_type} "
        f"--contrast-margin-coeff {contrast_margin_coeff}"
    )

    # ---- SWA 参数 ----
    if use_swa:
        train_cmd += (
            f" --use-swa --swa-start {swa_start} --swa-lr {swa_lr} "
            f"--swa-anneal-epochs {swa_anneal_epochs} --swa-anneal-strategy {swa_anneal_strategy}"
        )
        print(f"[{ts}] [{exp_idx}/{total_exp}] SWA  | start={swa_start}, lr={swa_lr}, anneal={swa_anneal_epochs}, strategy={swa_anneal_strategy}")

    # ---- LOSS (LambdaRank) 参数 ----
    if use_loss:
        print(f"[{ts}] [{exp_idx}/{total_exp}] LOSS | alpha={lambdarank_alpha}, grad_clip={grad_clip_norm}")

    # ---- HARD (Hard Negative Mining) 参数 ----
    if use_hard:
        train_cmd += (
            f" --hard-negative-mining"
            f" --hnm-top-k {hnm_top_k}"
            f" --hnm-mining-every-n-epochs {hnm_mining_every_n_epochs}"
            f" --hnm-eta {hnm_eta}"
            f" --hnm-label-gap-threshold {hnm_label_gap_threshold}"
            f" --hnm-hard-ratio-start {hnm_hard_ratio_start}"
            f" --hnm-hard-ratio-end {hnm_hard_ratio_end}"
            f" --hnm-hard-ratio-warmup-epochs {hnm_hard_ratio_warmup_epochs}"
            f" --hnm-hard-pair-weight {hnm_hard_pair_weight}"
            f" --hnm-batch-loss-scale {hnm_batch_loss_scale}"
        )
        print(
            f"[{ts}] [{exp_idx}/{total_exp}] HARD | top_k={hnm_top_k}, "
            f"mining_every={hnm_mining_every_n_epochs}, eta={hnm_eta}, "
            f"gap_thresh={hnm_label_gap_threshold}, "
            f"hard_ratio=[{hnm_hard_ratio_start}->{hnm_hard_ratio_end}@{hnm_hard_ratio_warmup_epochs}ep], "
            f"pair_weight={hnm_hard_pair_weight}, batch_scale={hnm_batch_loss_scale}"
        )

    # ---- AUG (Structural Augmentation) 参数 ----
    if use_aug:
        train_cmd += f" {structural_aug_args}"
        print(f"[{ts}] [{exp_idx}/{total_exp}] AUG  | scheduler={structural_aug_scheduler}, static_prob={structural_aug_static_prob}")

    # 实时捕获子进程输出到日志
    run_with_log_capture(train_cmd)
    t_train = time.time() - t0_train

    ts = datetime.now().strftime('%H:%M:%S')
    print(f"[{ts}] [{exp_idx}/{total_exp}] TRAIN DONE | Time: {t_train:.1f}s | Save: {save_dir}")

    # ===================== Inference =====================
    # eval params
    eval_batch_size = 32

    # infer output path (加时间戳防止重复运行覆盖)
    results_folder = 'infer_' + exp_name + '_' + timestamp
    eval_results_path = os.path.join("./infer_results/", results_folder)

    # SWA推理支持：优先使用SWA检查点
    if use_swa and os.path.exists(os.path.join(save_dir, "checkpoint_swa.pt")):
        eval_weight_path = os.path.join(save_dir, "checkpoint_swa.pt")
        print(f"[{ts}] [{exp_idx}/{total_exp}] SWA  | 使用SWA模型进行推理: {eval_weight_path}")
    else:
        eval_weight_path = os.path.join(save_dir, 'checkpoint_best.pt')

    ts = datetime.now().strftime('%H:%M:%S')
    print(f"[{ts}] [{exp_idx}/{total_exp}] INFER | Weight: {eval_weight_path}")

    t0_infer = time.time()
    infer_cmd = (
        f"python ../COMBINE4_unimol/infer_np.py --user-dir ../COMBINE4_unimol {data_path} "
        f"--task-name {task_name} --valid-subset test "
        f"--num-workers 4 --ddp-backend=c10d --batch-size {eval_batch_size} "
        f"--task mol_np_finetune --loss {loss_func} --arch np_unimol "
        f"--classification-head-name {task_name} --num-classes {task_num} "
        f"--dict-name {dict_name} --conf-size {conf_size} "
        f"--only-polar {only_polar} "
        f"--path {eval_weight_path} "
        f"--fp16 --fp16-init-scale 4 --fp16-scale-window 256 "
        f"--log-interval 50 --log-format simple "
        f"--results-path {eval_results_path} "
        f"--lnp-encoder-layers {lnp_encoder_layers} "
        f"--lnp-encoder-embed-dim {lnp_encoder_embed_dim} "
        f"--lnp-encoder-ffn-embed-dim {lnp_encoder_ffn_embed_dim} "
        f"--lnp-encoder-attention-heads {lnp_encoder_attention_heads} "
        f"--full-dataset-task-schema-path {full_dataset_task_schema_path} "
        f"--load-full-np-model --concat-datasets"
    )

    # SWA推理标记
    if use_swa and "checkpoint_swa.pt" in eval_weight_path:
        infer_cmd += " --use-swa"

    # 实时捕获子进程输出到日志
    run_with_log_capture(infer_cmd)
    t_infer = time.time() - t0_infer

    ts = datetime.now().strftime('%H:%M:%S')
    print(f"[{ts}] [{exp_idx}/{total_exp}] INFER DONE | Time: {t_infer:.1f}s | Results: {eval_results_path}")
    completed += 1


# ===================== Summary =====================
ts = datetime.now().strftime('%H:%M:%S')
print_header("TRAINING COMPLETE")
print_result("Dataset", f"{dataset_key} ({ds_cfg['label']})")
print_result("Total experiments", total_exp)
print_result("Completed", completed)
print_result("Skipped", skipped)
print_result("SWA", swa_status)
print_result("LOSS", loss_status)
print_result("HARD", hard_status)
print_result("AUG", aug_status)
print_result("Log file", log_file)
