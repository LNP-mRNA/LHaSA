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

# ==================== 日志系统 ====================
from utils.logger import setup_logger, print_header, print_section, print_result, get_log_dir


def run_with_log_capture(cmd):
    """运行命令并实时捕获输出到日志（通过print走Tee），避免增加嵌套层级"""
    process = sp.Popen(cmd, shell=True, stdout=sp.PIPE, stderr=sp.STDOUT, text=True, bufsize=1)
    for line in process.stdout:
        print(line, end='')
    process.wait()
    if process.returncode != 0:
        print(f"[ERROR] Command failed with code {process.returncode}")
        print(f"[ERROR] Failed command: {cmd[:200]}...")
    return process.returncode

# ==================== 全局配置 ====================
data_path='./'
MASTER_PORT=10086
n_gpu=1
dict_name='dict.txt'
weight_path='../ckp/mol_pre_no_h_220816.pt'
task_num=1
local_batch_size=128
only_polar=0
conf_size=11

# fold name from command line, default fold_V0
fold = sys.argv[1] if len(sys.argv) > 1 else 'fold_V0'
root_task_name = f'processed_data_dirs/OS_demo_in_house_lnp_data_overall_new_full_without_pbae_NPratios_updated_09222023_npratios_09252023gen_fig3di/{fold}'
root_save_dir='./save_demo'
root_tmp_save_dir = './tmp_save_demo'

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
loss_funcs = ['np_finetune_contrastive']
full_dataset_task_schema_path = "task_schemas/in_house_lnp_master_schema_NPratio_AOvolratio.json"
patiences = [1000]
subdataset_patiences = [-1]
epoch_to_freeze_molecule_encoder_list = [1000000]
cagrad_cs = [0.2]
percent_noises = [0.1]
contrast_margin_coeffs = [0.01]
percent_noise_types = ['normal_proportionate']
save_all_model_weights = True
train_data_ratios = [1]
seeds=[1]

# ============================================================================
# 结构化数据增强配置（化学先验驱动的配方变异）
# ============================================================================
use_structural_aug = True          # 主开关：是否启用结构化增强
structural_aug_scheduler = True    # 使用课程式概率调度（三段式：0.3→0.5→0.2）
structural_aug_static_prob = 0.5   # 静态增强概率（不使用调度器时生效）

# 三类增强子操作开关（D1消融实验用）
structural_aug_enable_replace = True   # 同类替换
structural_aug_enable_perturb = True   # 比例扰动
structural_aug_enable_dropout = True   # 组分Dropout
# 比例扰动sigma值（D3消融实验用）
structural_aug_sigma = 0.05            # 默认5%

# ========== 实验调度支持：环境变量覆盖增强参数 ==========
import os as _env_os
if _env_os.environ.get('COMET_EXP_ID'):
    # 总开关
    _env_val = _env_os.environ.get('COMET_AUG_ENABLED')
    if _env_val is not None:
        use_structural_aug = _env_val.lower() in ('1', 'true', 'yes')
    # 调度器
    _env_val = _env_os.environ.get('COMET_AUG_SCHEDULER')
    if _env_val is not None:
        structural_aug_scheduler = _env_val.lower() in ('1', 'true', 'yes')
    # 静态概率
    _env_val = _env_os.environ.get('COMET_AUG_STATIC_PROB')
    if _env_val is not None:
        structural_aug_static_prob = float(_env_val)
    # 子操作开关
    _env_val = _env_os.environ.get('COMET_AUG_ENABLE_REPLACE')
    if _env_val is not None:
        structural_aug_enable_replace = _env_val.lower() in ('1', 'true', 'yes')
    _env_val = _env_os.environ.get('COMET_AUG_ENABLE_PERTURB')
    if _env_val is not None:
        structural_aug_enable_perturb = _env_val.lower() in ('1', 'true', 'yes')
    _env_val = _env_os.environ.get('COMET_AUG_ENABLE_DROPOUT')
    if _env_val is not None:
        structural_aug_enable_dropout = _env_val.lower() in ('1', 'true', 'yes')
    # 扰动sigma
    _env_val = _env_os.environ.get('COMET_AUG_SIGMA')
    if _env_val is not None:
        structural_aug_sigma = float(_env_val)
# 实验标记（用于区分保存目录）
COMET_EXP_TAG = _env_os.environ.get('COMET_EXP_TAG', '')
# ========================================================

# 计算总实验数
total_exp = len(seeds) * len(lnp_encoder_attention_heads_list) * len(lnp_encoder_ffn_embed_dim_list) * \
            len(lnp_encoder_embed_dim_list) * len(lnp_encoder_layers_list) * len(warmups) * len(dropouts) * \
            len(epoch_list) * len(lrs) * len(batch_sizes) * len(loss_sample_dropouts) * len(loss_funcs) * \
            len(cagrad_cs) * len(epoch_to_freeze_molecule_encoder_list) * len(subdataset_patiences) * \
            len(contrast_margin_coeffs) * len(percent_noises) * len(percent_noise_types) * len(patiences) * \
            len(train_data_ratios)

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

for seed in seeds:
    for lnp_encoder_attention_heads in lnp_encoder_attention_heads_list:
        for lnp_encoder_ffn_embed_dim in lnp_encoder_ffn_embed_dim_list:
            for lnp_encoder_embed_dim in lnp_encoder_embed_dim_list:
                for lnp_encoder_layers in lnp_encoder_layers_list:
                    for warmup in warmups:
                        for dropout in dropouts:
                            for epoch in epoch_list:
                                for lr in lrs:
                                    for batch_size in batch_sizes:
                                        for loss_sample_dropout in loss_sample_dropouts:
                                            for loss_func in loss_funcs:
                                                for cagrad_c in cagrad_cs:
                                                    for epoch_to_freeze_molecule_encoder in epoch_to_freeze_molecule_encoder_list:
                                                        for subdataset_patience in subdataset_patiences:
                                                                for contrast_margin_coeff in contrast_margin_coeffs:
                                                                    for percent_noise in percent_noises:
                                                                        for percent_noise_type in percent_noise_types:
                                                                            for patience in patiences:
                                                                                for train_data_ratio in train_data_ratios:
                                                                                    task_name = root_task_name

                                                                                    # set up batch_size
                                                                                    local_batch_size=batch_size
                                                                                    update_freq=batch_size / local_batch_size

                                                                                    # compensate max_epoch with loss_sample_dropout
                                                                                    max_epoch = int(epoch // (1 - loss_sample_dropout))
                                                                                    print(f"[DEBUG] epoch={epoch} loss_sample_dropout={loss_sample_dropout} max_epoch={max_epoch}")

                                                                                    # unique experiment name (identifier)
                                                                                    _ts = datetime.now().strftime('%H%M%S')
                                                                                    _fold_short = fold[5:] if fold.startswith('fold_') else fold
                                                                                    _aug_tag = f"-aug{int(use_structural_aug)}"
                                                                                    _sched_tag = f"-sch{int(structural_aug_scheduler)}"
                                                                                    _sigma_tag = f"-sig{structural_aug_sigma}"
                                                                                    _replace_tag = f"-r{int(structural_aug_enable_replace)}"
                                                                                    _perturb_tag = f"-p{int(structural_aug_enable_perturb)}"
                                                                                    _dropout_tag = f"-d{int(structural_aug_enable_dropout)}"
                                                                                    exp_name = (f"f{_fold_short}_lnp_{loss_func}-bs{batch_size}-lr{lr}-"
                                                                                                f"lnp{lnp_encoder_layers}-{lnp_encoder_embed_dim}-{lnp_encoder_ffn_embed_dim}-"
                                                                                                f"{lnp_encoder_attention_heads}-tr{train_data_ratio}-ep{max_epoch}-"
                                                                                                f"pat{patience}-{metric}-cg{cagrad_c}-pn{percent_noise}-"
                                                                                                f"lm{contrast_margin_coeff}{_aug_tag}{_sched_tag}{_sigma_tag}"
                                                                                                f"{_replace_tag}{_perturb_tag}{_dropout_tag}-"
                                                                                                f"s{seed}{COMET_EXP_TAG}-t{_ts}_OS")

                                                                                    exp_idx += 1
                                                                                    ts = datetime.now().strftime("%H:%M:%S")

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
                                                                                    logdir=os.path.join("./logs/tmp/", log_path)


                                                                                    # infer output path
                                                                                    results_folder = 'infer_' + exp_name
                                                                                    eval_results_path = os.path.join("./infer_results/", results_folder)
                                                                                    eval_weight_path = os.path.join(save_dir, 'checkpoint_best.pt')


                                                                                    # Check if this experiment is already done, if so, skip it
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

                                                                                    # ==================== Training ====================
                                                                                    t0_train = time.time()
                                                                                    # 构建结构化增强参数
                                                                                    structural_aug_args = ""
                                                                                    if use_structural_aug:
                                                                                        aug_parts = ["--use-structural-aug"]
                                                                                        if structural_aug_scheduler:
                                                                                            aug_parts.append("--structural-aug-scheduler")
                                                                                        aug_parts.append(f"--structural-aug-static-prob {structural_aug_static_prob}")
                                                                                        # 子操作开关
                                                                                        if structural_aug_enable_replace:
                                                                                            aug_parts.append("--structural-aug-enable-replace")
                                                                                        if structural_aug_enable_perturb:
                                                                                            aug_parts.append("--structural-aug-enable-perturb")
                                                                                        if structural_aug_enable_dropout:
                                                                                            aug_parts.append("--structural-aug-enable-dropout")
                                                                                        # 扰动sigma
                                                                                        aug_parts.append(f"--structural-aug-sigma {structural_aug_sigma}")
                                                                                        structural_aug_args = " ".join(aug_parts)

                                                                                    train_cmd = f"PYTHONPATH=../AUG_unimol:.. python ../AUG_unimol/train_np.py {data_path} --task-name {task_name} --user-dir ../AUG_unimol --train-subset train --valid-subset valid \
                                                                                        --conf-size {conf_size} \
                                                                                        --num-workers 4 --ddp-backend=c10d \
                                                                                        --dict-name {dict_name} \
                                                                                        --task mol_np_finetune --loss {loss_func} --arch np_unimol  \
                                                                                        --classification-head-name {task_name} --num-classes {task_num} \
                                                                                        --optimizer adam --adam-betas '(0.9, 0.99)' --adam-eps 1e-6 --clip-norm 1.0 \
                                                                                        --lr-scheduler polynomial_decay --lr {lr} --warmup-ratio {warmup} --max-epoch {max_epoch} --batch-size {local_batch_size} --pooler-dropout {dropout} \
                                                                                        --loss-sample-dropout {loss_sample_dropout} \
                                                                                        --update-freq {update_freq} --seed {seed} \
                                                                                        --fp16 --fp16-init-scale 4 --fp16-scale-window 256 \
                                                                                        --log-interval 100 --log-format simple \
                                                                                        --validate-interval 1 --keep-last-epochs 10 \
                                                                                        --finetune-from-model {weight_path} \
                                                                                        --best-checkpoint-metric {metric} --patience {patience} \
                                                                                        --maximize-best-checkpoint-metric \
                                                                                        --save-dir {save_dir} --tmp-save-dir {tmp_save_dir} --only-polar {only_polar} \
                                                                                        --tensorboard-logdir {logdir} \
                                                                                        --full-dataset-task-schema-path {full_dataset_task_schema_path} \
                                                                                        --multitask-reg --cagrad-c {cagrad_c} \
                                                                                        --epoch-to-freeze-molecule-encoder {epoch_to_freeze_molecule_encoder} \
                                                                                        --concat-datasets \
                                                                                        --train-data-ratio {train_data_ratio} \
                                                                                        {structural_aug_args} \
                                                                                        --lnp-encoder-layers {lnp_encoder_layers} --lnp-encoder-embed-dim {lnp_encoder_embed_dim} --lnp-encoder-ffn-embed-dim {lnp_encoder_ffn_embed_dim} --lnp-encoder-attention-heads {lnp_encoder_attention_heads} \
                                                                                        --noise-augment-percent --percent-noise {percent_noise} --percent-noise-type {percent_noise_type} \
                                                                                        --contrast-margin-coeff {contrast_margin_coeff}"

                                                                                    # 实时捕获子进程输出到日志
                                                                                    run_with_log_capture(train_cmd)
                                                                                    t_train = time.time() - t0_train

                                                                                    ts = datetime.now().strftime("%H:%M:%S")
                                                                                    print(f"[{ts}] [{exp_idx}/{total_exp}] TRAIN DONE | Time: {t_train:.1f}s | Save: {save_dir}")

                                                                                    # eval params
                                                                                    eval_batch_size = 32

                                                                                    # tensorboard log path
                                                                                    results_folder = 'infer_' + exp_name
                                                                                    eval_results_path = os.path.join("./infer_results/", results_folder)
                                                                                    eval_weight_path = os.path.join(save_dir, 'checkpoint_best.pt')

                                                                                    # ==================== Inference ====================
                                                                                    ts = datetime.now().strftime("%H:%M:%S")
                                                                                    print(f"[{ts}] [{exp_idx}/{total_exp}] INFER | Weight: {eval_weight_path}")

                                                                                    t0_infer = time.time()
                                                                                    infer_cmd = f"PYTHONPATH=../AUG_unimol:.. python ../AUG_unimol/infer_np.py --user-dir ../AUG_unimol {data_path} --task-name {task_name} --valid-subset test \
                                                                                        --num-workers 4 --ddp-backend=c10d --batch-size {eval_batch_size} \
                                                                                        --task mol_np_finetune --loss {loss_func} --arch np_unimol \
                                                                                        --classification-head-name {task_name} --num-classes {task_num} \
                                                                                        --dict-name {dict_name} --conf-size {conf_size} \
                                                                                        --only-polar {only_polar}  \
                                                                                        --path {eval_weight_path}  \
                                                                                        --fp16 --fp16-init-scale 4 --fp16-scale-window 256 \
                                                                                        --log-interval 50 --log-format simple \
                                                                                        --results-path {eval_results_path} \
                                                                                        --lnp-encoder-layers {lnp_encoder_layers} --lnp-encoder-embed-dim {lnp_encoder_embed_dim} --lnp-encoder-ffn-embed-dim {lnp_encoder_ffn_embed_dim} --lnp-encoder-attention-heads {lnp_encoder_attention_heads} \
                                                                                        --full-dataset-task-schema-path {full_dataset_task_schema_path} \
                                                                                        --load-full-np-model --concat-datasets"

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
