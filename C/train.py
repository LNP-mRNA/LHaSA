#!/usr/bin/env python3
"""
train.py -- 外层训练脚本
============================
- 调用 train_np.py 进行训练，然后调用 infer_np.py 进行推理
- 集成日志系统，简洁输出
- 日志文件：./logs/YYYYMMDD/train_{timestamp}_{exp_name}.log
"""

import subprocess
import subprocess as sp
import os
import shutil
import sys
from datetime import datetime, time as dt_time
import time
import itertools

# ==================== 日志系统 ====================
from utils.logger import setup_logger, print_header, print_section, print_result, get_log_dir


def run_with_log_capture(cmd):
    """运行命令并实时捕获输出到日志（通过print走Tee），避免增加嵌套层级"""
    process = sp.Popen(cmd, shell=True, stdout=sp.PIPE, stderr=sp.STDOUT, text=True, bufsize=1)
    for line in process.stdout:
        print(line, end="")
    process.wait()
    if process.returncode != 0:
        print(f"[WARNING] Command exited with code {process.returncode}")
    return process.returncode

# ==================== 全局配置 ====================
data_path="./"
MASTER_PORT=10086
n_gpu=1
dict_name="dict.txt"
weight_path="../ckp/mol_pre_no_h_220816.pt"
task_num=1
local_batch_size=128
only_polar=0
conf_size=11

# fold name from command line, default fold_V0
fold = sys.argv[1] if len(sys.argv) > 1 else "fold_V0"
root_task_name = f"processed_data_dirs/OS_demo_in_house_lnp_data_overall_new_full_without_pbae_NPratios_updated_09222023_npratios_09252023gen_fig3di/{fold}"
root_save_dir="./save_demo"
root_tmp_save_dir = "./tmp_save_demo"

metric="valid_spearmanr_coeff"
lr=1e-5
batch_size=128
local_batch_size=batch_size
update_freq=batch_size / local_batch_size
warmup=0.06
dropout=0.1
loss_sample_dropout=0.2
epoch=150###

lnp_encoder_attention_heads_list = [8]
lnp_encoder_ffn_embed_dim_list = [256]
lnp_encoder_embed_dim_list = [256]
lnp_encoder_layers_list = [8]
warmups=[0.06]
dropouts=[0.1]
epoch_list = [150]###
lrs = [1e-4]
batch_sizes = [128]
loss_sample_dropouts = [0]
loss_funcs = ["np_finetune_contrastive"]
full_dataset_task_schema_path = "task_schemas/in_house_lnp_master_schema_NPratio_AOvolratio.json"
patiences = [1000]
subdataset_patiences = [-1]
epoch_to_freeze_molecule_encoder_list = [1000000]
cagrad_cs = [0.2]
percent_noises = [0.1]
contrast_margin_coeffs = [0.01]
percent_noise_types = ["normal_proportionate"]
save_all_model_weights = True
train_data_ratios = [1]
seeds=[1]

# ===================== SWA配置 =====================
use_swa_list = [True]           # SWA开关列表
swa_starts = [40]               # SWA起始epoch列表
swa_lrs = [1e-4]                # SWA学习率列表
swa_anneal_epochs_list = [5]    # 退火epoch列表
swa_anneal_strategies = ["cos"] # 退火策略列表
# =================================================

# ========== 实验调度支持：环境变量覆盖 SWA 参数 ==========
import os as _env_os
if _env_os.environ.get('COMET_EXP_ID'):
    # SWA 开关
    _env_val = _env_os.environ.get('COMET_SWA_ENABLED')
    if _env_val is not None:
        use_swa_list = [_env_val.lower() in ('1', 'true', 'yes')]
    # SWA 起始 epoch
    _env_val = _env_os.environ.get('COMET_SWA_START')
    if _env_val is not None:
        swa_starts = [int(_env_val)]
    # SWA 学习率
    _env_val = _env_os.environ.get('COMET_SWA_LR')
    if _env_val is not None:
        swa_lrs = [float(_env_val)]
    # SWA 退火 epoch
    _env_val = _env_os.environ.get('COMET_SWA_ANNEAL_EPOCHS')
    if _env_val is not None:
        swa_anneal_epochs_list = [int(_env_val)]
    # SWA 退火策略
    _env_val = _env_os.environ.get('COMET_SWA_ANNEAL_STRATEGY')
    if _env_val is not None:
        swa_anneal_strategies = [_env_val]
# 实验标记（用于区分保存目录）
COMET_EXP_TAG = _env_os.environ.get('COMET_EXP_TAG', '')
# ========================================================


# 计算总实验数
total_exp = len(seeds) * len(lnp_encoder_attention_heads_list) * len(lnp_encoder_ffn_embed_dim_list) * \
            len(lnp_encoder_embed_dim_list) * len(lnp_encoder_layers_list) * len(warmups) * len(dropouts) * \
            len(epoch_list) * len(lrs) * len(batch_sizes) * len(loss_sample_dropouts) * len(loss_funcs) * \
            len(cagrad_cs) * len(epoch_to_freeze_molecule_encoder_list) * len(subdataset_patiences) * \
            len(contrast_margin_coeffs) * len(percent_noises) * len(percent_noise_types) * len(patiences) * \
            len(train_data_ratios) * \
            len(use_swa_list) * len(swa_starts) * len(swa_lrs) * \
            len(swa_anneal_epochs_list) * len(swa_anneal_strategies)

# 获取基础时间戳（用于日志文件名）
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# 设置日志（在循环外设置一次，避免重复）
base_exp_name = f"demo_in_house_ED09262023_fig3dii_{fold}"
log_file = setup_logger("train", base_exp_name, timestamp=timestamp)

print_header("TRAINING START")
print(f"[{datetime.now().strftime('%H:%M:%S')}] Total experiments: {total_exp}")
print(f"[{datetime.now().strftime('%H:%M:%S')}] Save dir: {root_save_dir}")
print(f"[{datetime.now().strftime('%H:%M:%S')}] Pretrained weight: {weight_path}")

# ==================== 主循环 ====================
exp_idx = 0
skipped = 0
completed = 0

for (seed, lnp_encoder_attention_heads, lnp_encoder_ffn_embed_dim,
     lnp_encoder_embed_dim, lnp_encoder_layers, warmup, dropout,
     epoch, lr, batch_size, loss_sample_dropout, loss_func,
     cagrad_c, epoch_to_freeze_molecule_encoder, subdataset_patience,
     contrast_margin_coeff, percent_noise, percent_noise_type,
     patience, train_data_ratio, use_swa, swa_start, swa_lr,
     swa_anneal_epochs, swa_anneal_strategy) in itertools.product(
        seeds, lnp_encoder_attention_heads_list, lnp_encoder_ffn_embed_dim_list,
        lnp_encoder_embed_dim_list, lnp_encoder_layers_list, warmups, dropouts,
        epoch_list, lrs, batch_sizes, loss_sample_dropouts, loss_funcs,
        cagrad_cs, epoch_to_freeze_molecule_encoder_list, subdataset_patiences,
        contrast_margin_coeffs, percent_noises, percent_noise_types, patiences,
        train_data_ratios, use_swa_list, swa_starts, swa_lrs,
        swa_anneal_epochs_list, swa_anneal_strategies):
    task_name = root_task_name

    # set up batch_size
    local_batch_size=batch_size
    update_freq=batch_size / local_batch_size

    # compensate max_epoch with loss_sample_dropout
    max_epoch = int(epoch // (1 - loss_sample_dropout))

    # SWA标签
    swa_tag = f"-swa{swa_start}-lr{swa_lr}" if use_swa else ""

    # unique experiment name (identifier)
    _ts = datetime.now().strftime("%H%M%S")
    _fold_short = fold[5:] if fold.startswith("fold_") else fold
    _swa_str = f"-swa{swa_start}-{swa_lr}-{swa_anneal_strategy}" if use_swa else ""
    exp_name = (f"f{_fold_short}_lnp_{loss_func}-bs{batch_size}-lr{lr}-"
                f"lnp{lnp_encoder_layers}-{lnp_encoder_embed_dim}-{lnp_encoder_ffn_embed_dim}-"
                f"{lnp_encoder_attention_heads}-tr{train_data_ratio}-ep{max_epoch}-"
                f"pat{patience}-{metric}-cg{cagrad_c}-pn{percent_noise}-"
                f"lm{contrast_margin_coeff}{_swa_str}-"
                f"s{seed}{COMET_EXP_TAG}-t{_ts}_OS")

    exp_idx += 1
    ts = datetime.now().strftime("%H:%M:%S")

    print(f"\n[{ts}] task_name: {task_name}")

    if save_all_model_weights:
        save_path = "save_" + exp_name
        save_dir = os.path.join(root_save_dir, save_path)
        tmp_save_dir = os.path.join(root_tmp_save_dir, save_path)
    else:
        save_dir = os.path.join(root_save_dir, task_name)
        tmp_save_dir = os.path.join(root_tmp_save_dir, task_name)

    # tensorboard log path
    log_path = "log_" + exp_name
    logdir=os.path.join("./logs/tmp/", log_path)


    # infer output path
    results_folder = "infer_" + exp_name
    eval_results_path = os.path.join("./infer_results/", results_folder)
    eval_weight_path = os.path.join(save_dir, "checkpoint_best.pt")


    # Check if this experiment is already done, if so, skip it
    if os.path.exists(eval_results_path):
        print(f"[{ts}] [{exp_idx}/{total_exp}] SKIP | Infer output exists: {eval_results_path}")
        skipped += 1
        continue
    elif os.path.exists(logdir): # if logdir exists, delete it as the previous exp run is not done yet
        print(f"[{ts}] [{exp_idx}/{total_exp}] RERUN | Tensorboard log exists but inference not done. Removing: {logdir}")
        shutil.rmtree(logdir)

    print(f"[{ts}] [{exp_idx}/{total_exp}] NEW  | Exp: {exp_name}")

    if os.path.exists(save_dir) and os.path.isdir(save_dir):
        shutil.rmtree(save_dir)

    if os.path.exists(tmp_save_dir) and os.path.isdir(tmp_save_dir):
        shutil.rmtree(tmp_save_dir)


    print(f"[{ts}] [{exp_idx}/{total_exp}] RUN  | Training: {logdir}")

    # ==================== Training ====================
    t0_train = time.time()
    train_cmd = (
        f"python ../unimol/train_np.py {data_path} --task-name {task_name} --user-dir ../unimol --train-subset train --valid-subset valid "
        f"--conf-size {conf_size} "
        f"--num-workers 4 --ddp-backend=c10d "
        f"--dict-name {dict_name} "
        f"--task mol_np_finetune --loss {loss_func} --arch np_unimol "
        f"--classification-head-name {task_name} --num-classes {task_num} "
        f"--optimizer adam --adam-betas '(0.9, 0.99)' --adam-eps 1e-6 --clip-norm 1.0 "
        f"--lr-scheduler polynomial_decay --lr {lr} --warmup-ratio {warmup} --max-epoch {max_epoch} --batch-size {local_batch_size} --pooler-dropout {dropout} "
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
        f"--lnp-encoder-layers {lnp_encoder_layers} --lnp-encoder-embed-dim {lnp_encoder_embed_dim} --lnp-encoder-ffn-embed-dim {lnp_encoder_ffn_embed_dim} --lnp-encoder-attention-heads {lnp_encoder_attention_heads} "
        f"--noise-augment-percent --percent-noise {percent_noise} --percent-noise-type {percent_noise_type} "
        f"--contrast-margin-coeff {contrast_margin_coeff}"
    )

    # SWA参数
    if use_swa:
        train_cmd += (
            f" --use-swa --swa-start {swa_start} --swa-lr {swa_lr} "
            f"--swa-anneal-epochs {swa_anneal_epochs} --swa-anneal-strategy {swa_anneal_strategy}"
        )
        print(f"[{ts}] [{exp_idx}/{total_exp}] SWA  | start={swa_start}, lr={swa_lr}, anneal={swa_anneal_epochs}, strategy={swa_anneal_strategy}")

    # 实时捕获子进程输出到日志
    run_with_log_capture(train_cmd)
    t_train = time.time() - t0_train

    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] [{exp_idx}/{total_exp}] TRAIN DONE | Time: {t_train:.1f}s | Save: {save_dir}")

    # eval params
    eval_batch_size = 32

    # tensorboard log path
    results_folder = "infer_" + exp_name
    eval_results_path = os.path.join("./infer_results/", results_folder)

    # SWA推理支持：优先使用SWA检查点
    if use_swa and os.path.exists(os.path.join(save_dir, "checkpoint_swa.pt")):
        eval_weight_path = os.path.join(save_dir, "checkpoint_swa.pt")
        print(f"[{ts}] [{exp_idx}/{total_exp}] SWA  | 使用SWA模型进行推理: {eval_weight_path}")
    else:
        eval_weight_path = os.path.join(save_dir, "checkpoint_best.pt")

    # ==================== Inference ====================
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] [{exp_idx}/{total_exp}] INFER | Weight: {eval_weight_path}")

    t0_infer = time.time()
    infer_cmd = (
        f"python ../unimol/infer_np.py --user-dir ../unimol {data_path} --task-name {task_name} --valid-subset test "
        f"--num-workers 4 --ddp-backend=c10d --batch-size {eval_batch_size} "
        f"--task mol_np_finetune --loss {loss_func} --arch np_unimol "
        f"--classification-head-name {task_name} --num-classes {task_num} "
        f"--dict-name {dict_name} --conf-size {conf_size} "
        f"--only-polar {only_polar} "
        f"--path {eval_weight_path} "
        f"--fp16 --fp16-init-scale 4 --fp16-scale-window 256 "
        f"--log-interval 50 --log-format simple "
        f"--results-path {eval_results_path} "
        f"--lnp-encoder-layers {lnp_encoder_layers} --lnp-encoder-embed-dim {lnp_encoder_embed_dim} --lnp-encoder-ffn-embed-dim {lnp_encoder_ffn_embed_dim} --lnp-encoder-attention-heads {lnp_encoder_attention_heads} "
        f"--full-dataset-task-schema-path {full_dataset_task_schema_path} "
        f"--load-full-np-model --concat-datasets"
    )

    # SWA推理标记
    if use_swa and "checkpoint_swa.pt" in eval_weight_path:
        infer_cmd += " --use-swa"

    # 实时捕获子进程输出到日志
    run_with_log_capture(infer_cmd)
    t_infer = time.time() - t0_infer

    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] [{exp_idx}/{total_exp}] INFER DONE | Time: {t_infer:.1f}s | Results: {eval_results_path}")
    completed += 1


# ==================== Summary ====================
ts = datetime.now().strftime("%H:%M:%S")
print_header("TRAINING COMPLETE")
print_result("Total experiments", total_exp)
print_result("Completed", completed)
print_result("Skipped", skipped)
print_result("Log file", log_file)
