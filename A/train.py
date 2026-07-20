#!/usr/bin/env python3
import sys
sys.dont_write_bytecode = True
"""
train.py -- 外层训练脚本
============================
- 调用 train_np.py 进行训练，然后调用 infer_np.py 进行推理
- 集成日志系统，简洁输出
- 日志文件：./logs/YYYYMMDD/train_{timestamp}_{exp_name}.log
- 修复：使用 itertools.product 展平嵌套循环，避免 Python 20层嵌套限制
"""

import subprocess
import subprocess as sp
import os
import shutil
import sys
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
        print(f"[WARNING] Command exited with process.returncode")
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
epoch=150

# ===== LambdaRank 混合损失配置（方案三） =====
# 所有配置项统一放列表中，通过 itertools.product 展平循环
CONFIG_GRID = {
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
    'use_lambdarank': [True],       # 是否启用 LambdaRank 混合损失
    'lambdarank_alpha': [0.3],      # 混合比例: alpha * loss_lambdarank + (1-alpha) * loss_pairwise
    'grad_clip_norm': [1.0],        # 梯度裁剪范数，防止LambdaRank的ΔNDCG权重引发梯度爆炸
    'k': [10],                      # NDCG@K 的 K 值
}

# ========== 实验调度支持：环境变量覆盖 CONFIG_GRID ==========
# 外部脚本可通过 COMET_<KEY>=value 形式传参，无需修改本文件
import os as _env_os
if _env_os.environ.get('COMET_EXP_ID'):
    for _key in CONFIG_GRID.keys():
        _env_val = _env_os.environ.get(f'COMET_{_key.upper()}')
        if _env_val is not None:
            _orig = CONFIG_GRID[_key][0]
            if isinstance(_orig, bool):
                CONFIG_GRID[_key] = [_env_val.lower() in ('1', 'true', 'yes')]
            elif isinstance(_orig, int):
                CONFIG_GRID[_key] = [int(_env_val)]
            elif isinstance(_orig, float):
                CONFIG_GRID[_key] = [float(_env_val)]
            else:
                CONFIG_GRID[_key] = [_env_val]
# 实验标记（用于区分保存目录）
COMET_EXP_TAG = _env_os.environ.get('COMET_EXP_TAG', '')
# =============================================================

full_dataset_task_schema_path = "task_schemas/in_house_lnp_master_schema_NPratio_AOvolratio.json"
save_all_model_weights = True

# 计算总实验数
total_exp = 1
for v in CONFIG_GRID.values():
    total_exp *= len(v)

# 获取基础时间戳（用于日志文件名）
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# 设置日志（在循环外设置一次，避免重复）
base_exp_name = f"f{fold[5:]}_lnp"
log_file = setup_logger("train", base_exp_name, timestamp=timestamp)

print_header("TRAINING START")
print(f"[{datetime.now().strftime('%H:%M:%S')}] Total experiments: {total_exp}")
print(f"[{datetime.now().strftime('%H:%M:%S')}] Save dir: {root_save_dir}")
print(f"[{datetime.now().strftime('%H:%M:%S')}] Pretrained weight: {weight_path}")

# ==================== 主循环（展平为单层，避免 Python 20层嵌套限制） ====================
exp_idx = 0
skipped = 0
completed = 0

for combo in product(*CONFIG_GRID.values()):
    # 解包配置组合
    c = dict(zip(CONFIG_GRID.keys(), combo))

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
    use_lambdarank = c['use_lambdarank']
    lambdarank_alpha = c['lambdarank_alpha']
    grad_clip_norm = c['grad_clip_norm']

    task_name = root_task_name

    # set up batch_size
    local_batch_size=batch_size
    update_freq=batch_size / local_batch_size

    # compensate max_epoch with loss_sample_dropout
    max_epoch = int(epoch // (1 - loss_sample_dropout))

    # unique experiment name (identifier) -- 短名字，避免文件名过长
    lr_flag = f'lr{lr}'
    lambdarank_flag = f'lambdarank{lambdarank_alpha}' if use_lambdarank else 'nolambdarank'
    _fold_short = fold[5:] if fold.startswith('fold_') else fold
    _k_val = c.get('k', 10)
    _ts = datetime.now().strftime("%H%M%S")
    exp_name = (f"f{_fold_short}_lnp_{loss_func}-bs{batch_size}-{lr_flag}-"
                f"lnp{lnp_encoder_layers}-{lnp_encoder_embed_dim}-{lnp_encoder_ffn_embed_dim}-"
                f"{lnp_encoder_attention_heads}-tr{train_data_ratio}-ep{max_epoch}-"
                f"pat{patience}-{metric}-cg{cagrad_c}-pn{percent_noise}-"
                f"lm{contrast_margin_coeff}-{lambdarank_flag}-k{_k_val}-"
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
    elif os.path.exists(logdir): # if logdir exists, delete it as the previous exp run is not done yet
        print(f"[{ts}] [{exp_idx}/{total_exp}] RERUN | Tensorboard log exists but inference not done. Removing: {logdir}")
        shutil.rmtree(logdir)

    print(f"[{ts}] [{exp_idx}/{total_exp}] NEW  | Exp: {exp_name}")

    if os.path.exists(save_dir) and os.path.isdir(save_dir):
        shutil.rmtree(save_dir)

    if os.path.exists(tmp_save_dir) and os.path.isdir(tmp_save_dir):
        shutil.rmtree(tmp_save_dir)

    print(f"[{ts}] [{exp_idx}/{total_exp}] RUN  | Training: {logdir}")

    # ===== 🆕 设置 LambdaRank 环境变量（供 train_np.py 和 contrastive_loss.py 读取）=====
    # 使用环境变量而非命令行参数，因为 Uni-Core 框架的 parse_args_and_arch()
    # 会拦截不认识的参数导致 "unrecognized arguments" 报错。
    # 环境变量在 Python 子进程中自动继承，无需通过命令行传递。
    os.environ['COMET_USE_LAMBDARANK'] = '1' if use_lambdarank else '0'
    os.environ['COMET_LAMBDARANK_ALPHA'] = str(lambdarank_alpha)
    os.environ['COMET_GRAD_CLIP_NORM'] = str(grad_clip_norm)
    os.environ['COMET_LAMBDARANK_K'] = os.environ.get('COMET_LAMBDARANK_K', '10')
    # ===== 环境变量设置结束 =====

    # ==================== Training ====================
    t0_train = time.time()
    train_cmd = f"python ../unimol/train_np.py {data_path} --task-name {task_name} --user-dir ../unimol --train-subset train --valid-subset valid \
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
    infer_cmd = f"python ../unimol/infer_np.py --user-dir ../unimol {data_path} --task-name {task_name} --valid-subset test \
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
