#!/usr/bin/env python3 -u

"""
Train a new model on one or across multiple GPUs.
"""

# Adapted From Uni-Core's unicore_cli/train.py

import argparse
import logging
import math
import os
import sys
from typing import Dict, Optional, Any, List, Tuple, Callable

import numpy as np
import json
import torch

# load unimol as a module
import importlib

from pyprojroot import here as project_root
# print("str(project_root()): ", str(project_root()))

# sys.path.insert(0, "/home/gridsan/achan/experiments/lnp_ml/")
sys.path.insert(0, str(project_root()))
importlib.import_module('unimol')

from unimol.core import (
    checkpoint_utils,
    options,
    tasks,
    utils,
)

from unimol.core.data import iterators
from unimol.core.distributed import utils as distributed_utils
from unimol.core.logging import meters, metrics, progress_bar
from unimol.core.trainer import Trainer
from multiprocessing.pool import ThreadPool

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("unicore_cli.train")


def main(args) -> None:

    utils.import_user_module(args)
    utils.set_jit_fusion_options()

    assert (
        args.batch_size is not None
    ), "Must specify batch size either with --batch-size"
    metrics.reset()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    if distributed_utils.is_master(args):
        checkpoint_utils.verify_checkpoint_directory(args.save_dir)
        checkpoint_utils.verify_checkpoint_directory(args.tmp_save_dir)
        ckp_copy_thread = ThreadPool(processes=1)
    else:
        ckp_copy_thread = None

    # Print args
    logger.info(args)

    # Setup task, e.g., translation, language modeling, etc.
    # print("unicore_cli>train>main, args: ", args)
    task = tasks.setup_task(args)

    # 将结构化增强子操作参数传递给 augmenter
    if hasattr(task, 'augmenter') and task.augmenter is not None:
        if hasattr(args, 'structural_aug_enable_replace') and hasattr(task.augmenter, 'enable_replace'):
            task.augmenter.enable_replace = args.structural_aug_enable_replace
        if hasattr(args, 'structural_aug_enable_perturb') and hasattr(task.augmenter, 'enable_perturb'):
            task.augmenter.enable_perturb = args.structural_aug_enable_perturb
        if hasattr(args, 'structural_aug_enable_dropout') and hasattr(task.augmenter, 'enable_dropout'):
            task.augmenter.enable_dropout = args.structural_aug_enable_dropout
        if hasattr(args, 'structural_aug_sigma') and hasattr(task.augmenter, 'sigma'):
            task.augmenter.sigma = args.structural_aug_sigma
        _r = getattr(task.augmenter, 'enable_replace', '?')
        _p = getattr(task.augmenter, 'enable_perturb', '?')
        _d = getattr(task.augmenter, 'enable_dropout', '?')
        _s = getattr(task.augmenter, 'sigma', '?')
        print(f"[结构化增强] replace={_r} perturb={_p} dropout={_d} sigma={_s}")

    assert args.loss, "Please specify loss to train a model"

    # Build model and loss
    model = task.build_model(args) # relevant arg: --arch
    loss = task.build_loss(args) # relevant arg: --loss

    # # ADD HOOKS TO PRINT GRADIENTS - START - !REMOVE THIS after debugging!
    # def hook_fn(m, i, o):
    #     print(m)
    #     print("------------Input Grad------------")
    #     for grad in i:
    #         try:
    #             print(grad.shape)
    #         except AttributeError: 
    #             print ("None found for Gradient")

    # for module in model.named_modules():
    #     module[1].register_full_backward_hook(hook_fn)
    # ADD HOOKS TO PRINT GRADIENTS - END - !REMOVE THIS after debugging!

    # Load valid dataset (we load training data below, based on the latest checkpoint)
    for valid_sub_split in args.valid_subset.split(","):
        if args.concat_datasets:
            task.load_concat_dataset(valid_sub_split, combine=False, epoch=1)
        else:
            task.load_dataset(valid_sub_split, combine=False, epoch=1)
            

    logger.info(model)
    logger.info("task: {}".format(task.__class__.__name__))
    logger.info("model: {}".format(model.__class__.__name__))
    logger.info("loss: {}".format(loss.__class__.__name__))
    logger.info(
        "num. model params: {:,} (num. trained: {:,})".format(
            sum(getattr(p, "_orig_size", p).numel() for p in model.parameters()),
            sum(getattr(p, "_orig_size", p).numel() for p in model.parameters() if p.requires_grad),
        )
    )

    # Build trainer
    trainer = Trainer(args, task, model, loss)
    logger.info(
        "training on {} devices (GPUs)".format(
            args.distributed_world_size
        )
    )
    logger.info(
        "batch size per device = {}".format(
            args.batch_size,
        )
    )

    # Load the latest checkpoint if one is available and restore the
    # corresponding train iterator
    # train dataset gets loaded here!
    print("before extra_state, epoch_itr = checkpoint_utils.load_checkpoint")
    extra_state, epoch_itr = checkpoint_utils.load_checkpoint(
        args,
        trainer,
        # don't cache epoch iterators for sharded datasets
        disable_iterator_cache=False,
    )
    max_epoch = args.max_epoch or math.inf
    epoch_to_stop = args.epoch_to_stop or math.inf
    lr = trainer.get_lr()
    train_meter = meters.StopwatchMeter()
    train_meter.start()
    all_dropped_datasets = []
    prev_all_dropped_datasets = None
    while epoch_itr.next_epoch_idx <= max_epoch and epoch_itr.next_epoch_idx <= epoch_to_stop:
        # =================================================================
        # 结构化数据增强：更新课程式调度器的当前 epoch
        # =================================================================
        if hasattr(task, 'augmenter') and task.augmenter is not None:
            task.augmenter.set_epoch(epoch_itr.epoch)
            if epoch_itr.epoch % 10 == 0 or epoch_itr.epoch <= 3:
                logger.info(
                    f"[结构化增强] epoch={epoch_itr.epoch}, "
                    f"prob={task.augmenter.get_current_prob():.2f}"
                )

        if lr <= args.stop_min_lr:
            logger.info(
                f"stopping training because current learning rate ({lr}) is smaller "
                "than or equal to minimum learning rate "
                f"(--stop-min-lr={args.stop_min_lr})"
            )
            break

        # train for one epoch
        train_output = train(args, trainer, task, epoch_itr, ckp_copy_thread)
        if len(train_output) == 3:
            valid_losses, should_stop, epoch_dropped_datasets = train_output
            if type(epoch_dropped_datasets) == list and len(epoch_dropped_datasets) > 0:
                for dropped_dataset in epoch_dropped_datasets:
                    if dropped_dataset not in all_dropped_datasets:
                        all_dropped_datasets.append(dropped_dataset)
                print(epoch_dropped_datasets, " added to all_dropped_datasets: ", all_dropped_datasets)
        else:
            valid_losses, should_stop = train_output
        if should_stop:
            break


        if len(train_output) == 3 and len(all_dropped_datasets) > 0:
            if epoch_itr.next_epoch_idx >= args.start_epoch_to_drop_datasets: # check if it is time to drop datasets
                datasets_to_drop = epoch_dropped_datasets
            else:
                print("NOT epoch_itr.next_epoch_idx >= args.start_epoch_to_drop_datasets, not dropping datasets")
                datasets_to_drop = None
            load_dataset_next_epoch = ( datasets_to_drop != None and ( prev_all_dropped_datasets == None or len(all_dropped_datasets) > len(prev_all_dropped_datasets) ) )# NOW Check if all_dropped_datasets changed, if so, load_dataset=True

            epoch_itr = trainer.get_train_iterator(
                epoch_itr.next_epoch_idx,
                # sharded data: get train iterator for next epoch
                load_dataset=(task.has_sharded_data("train") or load_dataset_next_epoch),
                # don't cache epoch iterators for sharded datasets
                disable_iterator_cache=False,
                dropped_datasets=epoch_dropped_datasets,
                # load_dataset=load_dataset_next_epoch
            )

            prev_all_dropped_datasets = all_dropped_datasets.copy()
        else:
            epoch_itr = trainer.get_train_iterator(
                epoch_itr.next_epoch_idx,
                # sharded data: get train iterator for next epoch
                load_dataset=task.has_sharded_data("train"),
                # don't cache epoch iterators for sharded datasets
                disable_iterator_cache=False,
            )
    train_meter.stop()
    if ckp_copy_thread is not None:
        ckp_copy_thread.close()
        ckp_copy_thread.join()
    logger.info("done training in {:.1f} seconds".format(train_meter.sum))

    # ===== 🧪 训练结束后在 TEST 集上评估 =====
    if distributed_utils.is_master(args):
        test_subsets = ["test"]
        # 加载 test 数据集
        for test_sub_split in test_subsets:
            if args.concat_datasets:
                task.load_concat_dataset(test_sub_split, combine=False, epoch=1)
            else:
                task.load_dataset(test_sub_split, combine=False, epoch=1)

        print("\n" + "="*60)
        print("🚀 训练完成，开始在 TEST 数据集上进行最终评估...")
        print("="*60)

        # 在 test 集上评估
        test_losses = validate(args, trainer, task, epoch_itr, test_subsets)

        # 获取各数据集样本量
        print("\n" + "="*60)
        print("📊 数据集样本量统计")
        print("="*60)
        for subset_name in args.valid_subset.split(","):
            try:
                dataset = task.dataset(subset_name)
                size = len(dataset)
                print(f"  📁 {subset_name} (验证集): {size} 样本")
            except Exception:
                print(f"  📁 {subset_name} (验证集): 无法获取样本量")
        for subset_name in test_subsets:
            try:
                dataset = task.dataset(subset_name)
                size = len(dataset)
                print(f"  📁 {subset_name} (测试集): {size} 样本")
            except Exception:
                print(f"  📁 {subset_name} (测试集): 无法获取样本量")
        print("="*60 + "\n")


def should_stop_early(args, valid_loss: float) -> bool:
    # skip check if no validation was done in the current epoch
    if valid_loss is None:
        return False
    if args.patience <= 0:
        return False

    def is_better(a, b):
        return a > b if args.maximize_best_checkpoint_metric else a < b

    prev_best = getattr(should_stop_early, "best", None)

    if prev_best is None or is_better(valid_loss, prev_best):
        should_stop_early.best = valid_loss
        should_stop_early.num_runs = 0
        return False
    else:
        should_stop_early.num_runs += 1
        if should_stop_early.num_runs >= args.patience:
            logger.info(
                "early stop since valid performance hasn't improved for last {} runs".format(
                    args.patience
                )
            )
            return True
        else:
            return False

def should_drop_dataset_early(args, valid_losses: dict, metrics_to_dropped_datasets: dict=None, current_epoch=0) -> bool:
    # skip check if no validation was done in the current epoch
    if len(valid_losses) == 0 or metrics_to_dropped_datasets == None:
        return False
    if args.subdataset_patience <= 0:
        return False
    if current_epoch < args.start_epoch_to_drop_datasets: # not time to drop datasets yet
        return False

    def is_better(a, b):
        return a > b if args.maximize_metrics_that_drop_datasets else a < b
        # return a > b if args.maximize_best_checkpoint_metric else a < b
    
    dropped_datasets = []
    for metric in valid_losses:
        prev_best = getattr(should_drop_dataset_early, "best", None)
        if prev_best is None:
            should_drop_dataset_early.best = {}
            should_drop_dataset_early.num_runs = {}
            prev_best = getattr(should_drop_dataset_early, "best", None)
        if metric not in prev_best:
            should_drop_dataset_early.best[metric] = valid_losses[metric]
            should_drop_dataset_early.num_runs[metric] = 0
            # return False
        elif is_better(valid_losses[metric], prev_best[metric]):
            should_drop_dataset_early.best[metric] = valid_losses[metric]
            should_drop_dataset_early.num_runs[metric] = 0
        else:
            should_drop_dataset_early.num_runs[metric] += 1
            if should_drop_dataset_early.num_runs[metric] >= args.subdataset_patience and (metric in metrics_to_dropped_datasets):
                dropped_dataset = metrics_to_dropped_datasets[metric]
                logger.info(
                    "early drop dataset {} since valid performance ({}) hasn't improved for last {} runs".format(
                        metrics_to_dropped_datasets[metric], metric, args.subdataset_patience
                    )
                )
                dropped_datasets.append(dropped_dataset)
    return dropped_datasets

@metrics.aggregate("train")
def train(
    args, trainer: Trainer, task: tasks.UnicoreTask, epoch_itr, ckp_copy_thread
) -> Tuple[List[Optional[float]], bool]:
    """Train the model for one epoch and return validation losses."""
    # Freeze model subset if it is the epoch to freeze (e.g. molecule encoder)
    if args.epoch_to_freeze_molecule_encoder != None and epoch_itr.epoch >= args.epoch_to_freeze_molecule_encoder:
        # freeze_params(self.mol_model)
        for child in trainer.model.mol_model.children():
            for param in child.parameters():
                param.requires_grad = False

        print("args.epoch_to_freeze_molecule_encoder != None and epoch_itr.epoch >= args.epoch_to_freeze_molecule_encoder")
        print("Complete freezing params, args.epoch_to_freeze_molecule_encoder: ", args.epoch_to_freeze_molecule_encoder)

    # Initialize data iterator
    itr = epoch_itr.next_epoch_itr(
        fix_batches_to_gpus=args.fix_batches_to_gpus,
        shuffle=(epoch_itr.next_epoch_idx > args.curriculum),
    )
    update_freq = (
        args.update_freq[epoch_itr.epoch - 1]
        if epoch_itr.epoch <= len(args.update_freq)
        else args.update_freq[-1]
    )
    itr = iterators.GroupedIterator(itr, update_freq)
    progress = progress_bar.progress_bar(
        itr,
        log_format=args.log_format,
        log_interval=args.log_interval,
        epoch=epoch_itr.epoch,
        tensorboard_logdir=(
            args.tensorboard_logdir
            if distributed_utils.is_master(args)
            else None
        ),
        default_log_format=("tqdm" if not args.no_progress_bar else "simple"),
    )

    trainer.begin_epoch(epoch_itr.epoch)
    metrics.log_scalar("epoch_itr_sz", len(epoch_itr), priority=1500, round=1, weight=0)

    valid_subsets = args.valid_subset.split(",")
    should_stop = False
    num_updates = trainer.get_num_updates()
    logger.info("Start iterating over samples")
    max_update = args.max_update or math.inf

    if args.metrics_to_dropped_datasets == None:
        metrics_to_dropped_datasets = {}
    else:
        with open(os.path.join(args.data, args.metrics_to_dropped_datasets), 'r') as openfile:
            # Reading from json file
            metrics_to_dropped_datasets = json.load(openfile)
        
    for i, samples in enumerate(progress):
        with metrics.aggregate("train_inner"), torch.autograd.profiler.record_function(
            "train_step-%d" % i
        ):  
            log_output = trainer.train_step(samples)

        if log_output is not None:  # not OOM, overflow, ...
            # log mid-epoch stats
            num_updates = trainer.get_num_updates()
            if num_updates % args.log_interval == 0:
                stats = get_training_stats(metrics.get_smoothed_values("train_inner"))
                progress.log(stats, tag="train_inner", step=num_updates)

                # reset mid-epoch stats after each log interval
                # the end-of-epoch stats will still be preserved
                metrics.reset_meters("train_inner")

        end_of_epoch = not itr.has_next()
        # check if we want to (possibly) drop data subset during the epoch
        # valid_losses, should_stop, should_stop_dataset = validate_and_save(
        validate_and_save_output = validate_and_save(
            args, trainer, task, epoch_itr, valid_subsets, end_of_epoch, ckp_copy_thread, metrics_to_dropped_datasets
        )
        if len(validate_and_save_output) == 3:
            valid_losses, should_stop, dropped_datasets = validate_and_save_output
        else:
            valid_losses, should_stop = validate_and_save_output
        # print("train valid_losses: ", valid_losses)

        if should_stop:
            break

    # log end-of-epoch stats
    logger.info("end of epoch {} (average epoch stats below)".format(epoch_itr.epoch))
    stats = get_training_stats(metrics.get_smoothed_values("train"))
    progress.print(stats, tag="train", step=num_updates)

    # reset epoch-level meters
    metrics.reset_meters("train")
    # TODO NOW: Edit this to dynamically change dataset sources (e.g. dropping datasets that are overfitted by model)
    # check if we want to (possibly) drop data subset during the epoch
    # valid_losses, should_stop, should_stop_dataset
    return validate_and_save_output


def validate_and_save(
    args,
    trainer: Trainer,
    task: tasks.UnicoreTask,
    epoch_itr,
    valid_subsets: List[str],
    end_of_epoch: bool,
    ckp_copy_thread,
    metrics_to_dropped_datasets=None,
    # start_dropping_datasets=False
) -> Tuple[List[Optional[float]], bool]:
    # print("validate_and_save")
    num_updates = trainer.get_num_updates()
    max_update = args.max_update or math.inf

    # Stopping conditions (and an additional one based on validation loss later
    # on)
    should_stop = False
    if num_updates >= max_update:
        should_stop = True
        logger.info(
            f"Stopping training due to "
            f"num_updates: {num_updates} >= max_update: {max_update}"
        )

    training_time_hours = trainer.cumulative_training_time() / (60 * 60)
    if (
        args.stop_time_hours > 0
        and training_time_hours > args.stop_time_hours
    ):
        should_stop = True
        logger.info(
            f"Stopping training due to "
            f"cumulative_training_time: {training_time_hours} > "
            f"stop_time_hours: {args.stop_time_hours} hour(s)"
        )

    do_save = (
        (end_of_epoch and epoch_itr.epoch % args.save_interval == 0 and not args.no_epoch_checkpoints)
        or should_stop
        or (
            args.save_interval_updates > 0
            and num_updates > 0
            and num_updates % args.save_interval_updates == 0
            and num_updates >= args.validate_after_updates
        )
    )
    do_validate = (
        (not end_of_epoch and do_save)  # validate during mid-epoch saves
        or (end_of_epoch and epoch_itr.epoch % args.validate_interval == 0 and not args.no_epoch_checkpoints)
        or should_stop
        or (
            args.validate_interval_updates > 0
            and num_updates > 0
            and num_updates % args.validate_interval_updates == 0
        )
    ) and not args.disable_validation

    # Validate
    valid_losses = [None]
    if do_validate:
        if metrics_to_dropped_datasets != None:
            valid_losses, metrics_that_drop_datasets = validate(args, trainer, task, epoch_itr, valid_subsets, metrics_to_dropped_datasets)
        else:
            valid_losses = validate(args, trainer, task, epoch_itr, valid_subsets)

    should_stop |= should_stop_early(args, valid_losses[0])

    # Save checkpoint
    checkpoint_utils.save_checkpoint(
        args, trainer, epoch_itr, valid_losses[0], ckp_copy_thread, do_save=(do_save or should_stop),
    )

    if do_validate and metrics_to_dropped_datasets is not None:
        dropped_datasets = should_drop_dataset_early(args, metrics_that_drop_datasets, metrics_to_dropped_datasets, epoch_itr.epoch)

        return valid_losses, should_stop, dropped_datasets
    else:
        return valid_losses, should_stop


def get_training_stats(stats: Dict[str, Any]) -> Dict[str, Any]:
    stats["wall"] = round(metrics.get_meter("default", "wall").elapsed_time, 0)
    return stats


def validate(
    args,
    trainer: Trainer,
    task: tasks.UnicoreTask,
    epoch_itr,
    subsets: List[str],
    metrics_to_dropped_datasets=None,
) -> List[Optional[float]]:
    """Evaluate the model on the validation set(s) and return the losses."""

    seed = None
    if args.fixed_validation_seed is not None:
        # set fixed seed for every validation
        seed = args.fixed_validation_seed

    with utils.torch_seed(seed):
        trainer.begin_valid_epoch(epoch_itr.epoch)
        valid_losses = []
        for subset in subsets:
            logger.info('begin validation on "{}" subset'.format(subset))

            # Initialize data iterator
            itr = trainer.get_valid_iterator(subset).next_epoch_itr(
                shuffle=False, set_dataset_epoch=False  # use a fixed valid set
            )
            progress = progress_bar.progress_bar(
                itr,
                log_format=args.log_format,
                log_interval=args.log_interval,
                epoch=epoch_itr.epoch,
                prefix=f"valid on '{subset}' subset",
                tensorboard_logdir=(
                    args.tensorboard_logdir
                    if distributed_utils.is_master(args)
                    else None
                ),
                default_log_format=("tqdm" if not args.no_progress_bar else "simple"),
            )

            # create a new root metrics aggregator so validation metrics
            # don't pollute other aggregators (e.g., train meters)
            with metrics.aggregate(new_root=True) as agg:
                logging_outputs = []
                for i, sample in enumerate(progress):
                    if args.max_valid_steps is not None and i > args.max_valid_steps:
                        break
                    inner_logging_outputs = trainer.valid_step(sample)
                    logging_outputs.extend(inner_logging_outputs)
                # print("logging_outputs train_np: ", logging_outputs)
                task.reduce_metrics(logging_outputs, trainer.get_loss(), subset)

            # log validation stats
            stats = get_valid_stats(args, trainer, agg.get_smoothed_values())
            progress.print(stats, tag=subset, step=trainer.get_num_updates())
            # ===== 🎯 关键指标高亮 =====
            # 判断是验证集还是测试集，显示不同标题
            is_test = (subset == "test")
            panel_icon = "🧪" if is_test else "🎯"
            panel_title = "TEST RESULTS" if is_test else "VALIDATION RESULTS"
            subset_label = "测试集" if is_test else "验证集"
            # 获取数据集样本量
            try:
                dataset_size = len(task.dataset(subset))
                size_info = f" (n={dataset_size})"
            except Exception:
                size_info = ""
            print("\n" + "="*60)
            print(f"{panel_icon} {panel_title}{size_info}")
            print(f"   数据集: {subset} | 子任务: {subset_label}")
            print("="*60)
            for key, val in stats.items():
                if "spearman" in key and "coeff" in key:
                    print(f"  ⭐ {key}: {val:.4f}")
                elif "pearson" in key and "coeff" in key:
                    print(f"  📊 {key}: {val:.4f}")
                elif "loss" in key and "in_house" not in key:
                    print(f"  📉 {key}: {val:.4f}")
                elif "mae" in key and "agg" in key:
                    print(f"  📏 {key}: {val:.4f}")
            print("="*60 + "\n")
            print(f"validate stats ({subset}): ", stats)
            if args.best_checkpoint_metric in stats:
                valid_losses.append(stats[args.best_checkpoint_metric])

            # check if averaging of validation loss is needed
            else:
                metric_value_list = []
                metric_name_list = []
                for stat_name in stats:
                    if args.best_checkpoint_metric in stat_name and not math.isnan(stats[stat_name]):
                        metric_value_list.append(stats[stat_name])
                        metric_name_list.append(stat_name)

                if len(metric_value_list) != 0:
                    valid_losses.append(sum(metric_value_list)/len(metric_value_list))
                    logger.info("averaging {} for {}".format(str(metric_name_list), args.best_checkpoint_metric))

                # add early stopping of data subset based on subset validation loss: should_stop_dataset is a dict
                # calculate valid_losses for each data subset

        if metrics_to_dropped_datasets is not None:
            metrics_that_drop_datasets = {}
            for metric in metrics_to_dropped_datasets:
                if metric in stats:
                    metrics_that_drop_datasets[metric] = stats[metric]

            return valid_losses, metrics_that_drop_datasets

        print("validate, valid_losses: ", valid_losses)
        return valid_losses


def get_valid_stats(
    args, trainer: Trainer, stats: Dict[str, Any]
) -> Dict[str, Any]:
    stats["num_updates"] = trainer.get_num_updates()
    # 🎯 高亮最佳指标
    if args.best_checkpoint_metric in stats:
        best_val = stats[args.best_checkpoint_metric]
        best_key = f"best_{args.best_checkpoint_metric}"
        if hasattr(checkpoint_utils.save_checkpoint, "best"):
            prev_best = checkpoint_utils.save_checkpoint.best
            marker = "🆕 NEW BEST" if best_val > prev_best else ""
            print(f"\n🏆 {args.best_checkpoint_metric}: {best_val:.4f} {marker}")
    print("get_valid_stats stats: ", stats)
    if hasattr(checkpoint_utils.save_checkpoint, "best") and args.best_checkpoint_metric in stats:
        key = "best_{0}".format(args.best_checkpoint_metric)
        best_function = max if args.maximize_best_checkpoint_metric else min
        stats[key] = best_function(
            checkpoint_utils.save_checkpoint.best,
            stats[args.best_checkpoint_metric],
        )
    return stats


def _has_arg(parser, name):
    """检查 parser 是否已有指定参数"""
    for action in parser._actions:
        if name in action.option_strings:
            return True
    return False


def add_structural_aug_args(parser):
    """添加结构化增强子操作参数（D实验消融用），跳过已存在的参数"""
    if not _has_arg(parser, "--structural-aug-enable-replace"):
        parser.add_argument(
            "--structural-aug-enable-replace", action="store_true", default=False,
            help="启用同类替换增强操作"
        )
    if not _has_arg(parser, "--structural-aug-enable-perturb"):
        parser.add_argument(
            "--structural-aug-enable-perturb", action="store_true", default=False,
            help="启用比例扰动增强操作"
        )
    if not _has_arg(parser, "--structural-aug-enable-dropout"):
        parser.add_argument(
            "--structural-aug-enable-dropout", action="store_true", default=False,
            help="启用组分Dropout增强操作"
        )
    if not _has_arg(parser, "--structural-aug-sigma"):
        parser.add_argument(
            "--structural-aug-sigma", type=float, default=0.05,
            help="比例扰动的sigma值 (默认: 0.05)"
        )


def cli_main(
    modify_parser: Optional[Callable[[argparse.ArgumentParser], None]] = None
) -> None:
    print("running cli_main")
    parser = options.get_training_parser()
    args = options.parse_args_and_arch(parser, modify_parser=add_structural_aug_args)
    if args.profile:
        with torch.cuda.profiler.profile():
            with torch.autograd.profiler.emit_nvtx():
                distributed_utils.call_main(args, main)
    else:
        distributed_utils.call_main(args, main)


if __name__ == "__main__":
    print("before cli_main")
    cli_main()
    print("after cli_main")
