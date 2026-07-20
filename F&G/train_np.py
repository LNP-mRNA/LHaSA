#!/usr/bin/env python3 -u

"""
Train a new model on one or across multiple GPUs.

COMBINE4: Combined inner training script integrating 4 optional modules:
  - SWA  (Stochastic Weight Averaging)
  - LOSS (LambdaRank mixed loss logging)
  - HARD (Hard Negative Mining with MixedSampler)
  - AUG  (Structural data augmentation epoch scheduling)

Module activation via environment variables:
  COMET_USE_SWA=1  - Enable SWA
  COMET_USE_LOSS=1 - Enable LambdaRank loss logging
  COMET_USE_HARD=1 - Enable Hard Negative Mining
  COMET_USE_AUG=1  - Enable augmentation scheduling
"""

# Adapted From Uni-Core's unicore_cli/train.py

import argparse
import logging
import math
import os
import sys
from typing import Dict, Optional, Any, List, Tuple, Callable

# ===== Ensure this file's directory is in sys.path (for hard_negative_mining import) =====
_TRAIN_NP_DIR = os.path.dirname(os.path.abspath(__file__))
if _TRAIN_NP_DIR not in sys.path:
    sys.path.insert(0, _TRAIN_NP_DIR)
# =========================================================================================

import numpy as np
import json
import torch

# ===== SWA: import SWA utilities =====
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
# =====================================

# Hard Negative Mining
from hard_negative_mining import (
    mine_hard_pairs,
    MixedSampler,
    collate_batch_by_indices,
    should_mine_hard_pairs,
    get_hardness_statistics,
)

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


# =============================================================================
# SWA Helper Functions
# =============================================================================

def _unwrap_optimizer(optimizer):
    """Unwrap FP16Optimizer to get the underlying PyTorch native Optimizer."""
    from torch.optim import Optimizer

    # Already a native Optimizer
    if isinstance(optimizer, Optimizer):
        return optimizer

    # COMET FP16 mode: FP16Optimizer -> UnicoreAdam -> FusedAdam
    # Recursively unwrap common wrapper attributes
    for attr_name in ['fp32_optimizer', '_optimizer', 'optimizer', '_fp32_optim', '_optim']:
        if hasattr(optimizer, attr_name):
            inner = getattr(optimizer, attr_name)
            if inner is not optimizer and inner is not None:
                return _unwrap_optimizer(inner)

    # Final attempt: find Optimizer instance through _mro
    for attr_name in dir(optimizer):
        val = getattr(optimizer, attr_name, None)
        if val is not None and val is not optimizer and isinstance(val, Optimizer):
            return val

    return optimizer


def setup_swa(model, optimizer, args):
    """Initialize SWA model and SWA learning rate scheduler."""
    if not getattr(args, 'use_swa', False):
        return None, None

    # Unwrap FP16Optimizer to get underlying Optimizer (SWALR needs native Optimizer)
    base_optimizer = _unwrap_optimizer(optimizer)
    print(f"[SWA] Optimizer unwrap: {type(optimizer).__name__} -> {type(base_optimizer).__name__}")

    # AveragedModel in fp16 mode needs float() to avoid parameter type mismatch
    swa_model = AveragedModel(model)

    swa_scheduler = SWALR(
        base_optimizer,
        swa_lr=args.swa_lr,
        anneal_epochs=args.swa_anneal_epochs,
        anneal_strategy=args.swa_anneal_strategy,
    )
    print(f"[SWA] Enabled | start={args.swa_start}, lr={args.swa_lr}, anneal={args.swa_anneal_epochs}, strategy={args.swa_anneal_strategy}")
    return swa_model, swa_scheduler


def finalize_swa_model(swa_model, train_loader, save_path, args):
    """After training: update BN statistics + save SWA checkpoint."""
    if swa_model is None:
        return
    
    # Update BN statistics if train_loader is available
    if train_loader is not None:
        print(f"[SWA] Updating BN statistics...")
        try:
            update_bn(train_loader, swa_model, device=next(swa_model.parameters()).device)
            print(f"[SWA] BN statistics updated successfully.")
        except Exception as e:
            print(f"[SWA] Warning: BN update failed ({e}), using existing BN stats.")
    else:
        print(f"[SWA] Warning: train_loader is None, skipping BN update.")
    
    swa_model.train()

    # Save SWA checkpoint
    torch.save({
        "model_state_dict": swa_model.module.state_dict(),
        "swa_state_dict": swa_model.state_dict(),
        "swa_n_averaged": swa_model.n_averaged.item() if hasattr(swa_model.n_averaged, 'item') else swa_model.n_averaged,
    }, save_path)
    print(f"[SWA] Model saved: {save_path}")
    return swa_model


# =============================================================================
# Hard Negative Mining Helper Functions and Classes
# =============================================================================

def _move_sample_to_device(sample, device):
    """Recursively move tensors in a nested structure to the target device."""
    if isinstance(sample, dict):
        return {k: _move_sample_to_device(v, device) for k, v in sample.items()}
    elif isinstance(sample, list):
        return [_move_sample_to_device(v, device) for v in sample]
    elif isinstance(sample, tuple):
        return tuple(_move_sample_to_device(v, device) for v in sample)
    elif torch.is_tensor(sample):
        return sample.to(device, non_blocking=True)
    else:
        return sample


class _HNMBatchIterator:
    """
    Lazy batch iterator powered by MixedSampler.
    Yields [batch_dict] per step to match GroupedIterator(update_freq=1) format.
    Tracks hard-batch flag via args._hnm_current_batch_is_hard.
    """

    def __init__(self, mixed_sampler, dataset, args):
        self.sampler = mixed_sampler
        self.dataset = dataset
        self.args = args
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
        )
        # Cache hard-index set for O(1) look-ups
        hard_idx_list = getattr(mixed_sampler, "hard_indices", None) or []
        self.hard_set = set(hard_idx_list)
        self._len = None
        # Use dataset's collater if available (Uni-MOL datasets have collater)
        self.collater = getattr(dataset, 'collater', None)
        if self.collater is None:
            logger.warning("[HNM] dataset.collater not found, falling back to manual collate")
        self._reset()

    def _reset(self):
        self.sampler_iter = iter(self.sampler)
        self.idx = 0
        self._prefetch()

    def _prefetch(self):
        try:
            self._next_indices = next(self.sampler_iter)
            self._has_next = True
        except StopIteration:
            self._next_indices = None
            self._has_next = False

    def _check_is_hard(self, batch_indices):
        if not self.hard_set or not batch_indices:
            return False
        hard_count = sum(1 for idx in batch_indices if idx in self.hard_set)
        return hard_count > len(batch_indices) * 0.5

    def __iter__(self):
        self._reset()
        return self

    def __next__(self):
        if not self._has_next:
            raise StopIteration
        batch_indices = self._next_indices
        self._prefetch()
        # Mark whether this batch is a hard batch
        is_hard = self._check_is_hard(batch_indices)
        self.args._hnm_current_batch_is_hard = is_hard
        # Build batch: use dataset.collater if available (correct format)
        samples = [self.dataset[idx] for idx in batch_indices]
        if self.collater is not None:
            batch = self.collater(samples)
        else:
            batch = collate_batch_by_indices(self.dataset, batch_indices)
        batch = _move_sample_to_device(batch, self.device)
        self.idx += 1
        return [batch]  # wrap to match GroupedIterator output

    def __len__(self):
        if self._len is None:
            self._len = len(self.sampler)
        return self._len

    def has_next(self):
        return self._has_next


def _create_hnm_iterator(mixed_sampler, dataset, args):
    """Factory: build an _HNMBatchIterator."""
    return _HNMBatchIterator(mixed_sampler, dataset, args)


class _DummyIterator:
    """Dummy iterator used after HNM epoch to make end_of_epoch() return True."""
    def has_next(self):
        return False


# =============================================================================
# Main Training Function
# =============================================================================

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
    task = tasks.setup_task(args)

    assert args.loss, "Please specify loss to train a model"

    # Hard Negative Mining: mixed_sampler will be initialized after train dataset is loaded
    mixed_sampler = None

    # Build model and loss
    model = task.build_model(args)  # relevant arg: --arch
    loss = task.build_loss(args)    # relevant arg: --loss

    # LOSS: log LambdaRank loss config if enabled
    if getattr(args, 'use_loss', False):
        logger.info("LambdaRank mixed loss enabled (see contrastive_loss.py for implementation)")

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

    # ===== SWA: initialization =====
    # Note: must initialize SWA after load_checkpoint, because training data is loaded
    # optimizer and lr_scheduler can be correctly built (total_train_steps computed)
    swa_model = None
    swa_scheduler = None
    if getattr(args, 'use_swa', False):
        swa_model, swa_scheduler = setup_swa(trainer.model, trainer.optimizer, args)
        # Attach swa_model and swa_scheduler to trainer for access in train()
        trainer.swa_model = swa_model
        trainer.swa_scheduler = swa_scheduler
    # ================================

    # ===== Hard Negative Mining: initialize MixedSampler =====
    # Must be after checkpoint_utils.load_checkpoint, because train dataset is loaded
    if getattr(args, 'use_hard', False) and mixed_sampler is None:
        try:
            train_dataset = task.dataset(args.train_subset)
            dataset_size = len(train_dataset)
            mixed_sampler = MixedSampler(
                dataset_size=dataset_size,
                hard_pair_pool=[],
                batch_size=args.batch_size,
                normal_ratio=1.0 - args.hnm_hard_ratio_start,
                hard_ratio=args.hnm_hard_ratio_start,
                curriculum_schedule=True,
                hard_ratio_final=args.hnm_hard_ratio_end,
                warmup_epochs=args.hnm_hard_ratio_warmup_epochs,
                seed=args.seed,
            )
            logger.info(
                "[HNM] Hard Negative Mining enabled: top_k=%d, mining_interval=%d, "
                "hard_ratio=[%.2f -> %.2f] over %d epochs",
                args.hnm_top_k, args.hnm_mining_every_n_epochs,
                args.hnm_hard_ratio_start, args.hnm_hard_ratio_end,
                args.hnm_hard_ratio_warmup_epochs,
            )
        except KeyError as e:
            logger.warning("[HNM] Cannot initialize MixedSampler: %s. Disabling HNM.", e)
            mixed_sampler = None
    # ==========================================================

    max_epoch = args.max_epoch or math.inf
    epoch_to_stop = args.epoch_to_stop or math.inf
    lr = trainer.get_lr()
    train_meter = meters.StopwatchMeter()
    train_meter.start()
    all_dropped_datasets = []
    prev_all_dropped_datasets = None

    while epoch_itr.next_epoch_idx <= max_epoch and epoch_itr.next_epoch_idx <= epoch_to_stop:
        if lr <= args.stop_min_lr:
            logger.info(
                f"stopping training because current learning rate ({lr}) is smaller "
                "than or equal to minimum learning rate "
                f"(--stop-min-lr={args.stop_min_lr})"
            )
            break

        # ===== Hard Negative Mining: periodic mining =====
        if getattr(args, 'use_hard', False) and mixed_sampler is not None:
            current_epoch = epoch_itr.epoch

            # Curriculum scheduling: update hard_ratio for current epoch
            mixed_sampler.step_epoch()

            # Periodically trigger full-set mining
            # (first epoch also mines to initialize)
            if should_mine_hard_pairs(current_epoch, args.hnm_mining_every_n_epochs):
                logger.info("[HNM] Epoch %d: Triggering hard negative mining...", current_epoch)

                # Build dataloader for mining (use larger batch_size for speed)
                mining_dataloader = task.get_batch_iterator(
                    dataset=task.dataset(args.train_subset),
                    batch_size=args.batch_size * 4,  # use larger batch for speed
                    ignore_invalid_inputs=True,
                    required_batch_size_multiple=args.required_batch_size_multiple,
                    seed=args.seed,
                    num_shards=1,  # mining on master process
                    shard_id=0,
                    num_workers=args.num_workers,
                    data_buffer_size=args.data_buffer_size,
                ).next_epoch_itr(shuffle=False)

                # Execute mining
                device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
                new_hard_pairs = mine_hard_pairs(
                    model=trainer.model,
                    dataloader=mining_dataloader,
                    device=device,
                    top_k=args.hnm_top_k,
                    eta=args.hnm_eta,
                    label_gap_threshold=args.hnm_label_gap_threshold,
                )

                # Update sampler's hard pool (full replacement)
                if new_hard_pairs:
                    mixed_sampler.update_hard_pool(new_hard_pairs)
                    stats = get_hardness_statistics(new_hard_pairs)
                    logger.info(
                        "[HNM] Mining complete: %d pairs, %d unique samples, "
                        "hardness=[%.2f, %.2f, %.2f]",
                        stats["count"], stats["unique_samples"],
                        stats["min_hardness"], stats["mean_hardness"], stats["max_hardness"],
                    )

            # Reset hard-batch flag before training starts
            args._hnm_current_batch_is_hard = False
        # ==================================================

        # ===== AUG: update augmentation scheduler =====
        if getattr(args, 'use_aug', False) and hasattr(task, 'augmenter') and task.augmenter is not None:
            task.augmenter.set_epoch(epoch_itr.epoch)
            if epoch_itr.epoch % 10 == 0 or epoch_itr.epoch <= 3:
                logger.info("[AUG] epoch=%d, prob=%.2f", epoch_itr.epoch, task.augmenter.get_current_prob())
        # ==============================================

        # train for one epoch
        train_output = train(args, trainer, task, epoch_itr, ckp_copy_thread, mixed_sampler=mixed_sampler)
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
            if epoch_itr.next_epoch_idx >= args.start_epoch_to_drop_datasets:
                datasets_to_drop = epoch_dropped_datasets
            else:
                print("NOT epoch_itr.next_epoch_idx >= args.start_epoch_to_drop_datasets, not dropping datasets")
                datasets_to_drop = None
            load_dataset_next_epoch = ( datasets_to_drop != None and ( prev_all_dropped_datasets == None or len(all_dropped_datasets) > len(prev_all_dropped_datasets) ) )

            epoch_itr = trainer.get_train_iterator(
                epoch_itr.next_epoch_idx,
                load_dataset=(task.has_sharded_data("train") or load_dataset_next_epoch),
                disable_iterator_cache=False,
                dropped_datasets=epoch_dropped_datasets,
            )

            prev_all_dropped_datasets = all_dropped_datasets.copy()
        else:
            epoch_itr = trainer.get_train_iterator(
                epoch_itr.next_epoch_idx,
                load_dataset=task.has_sharded_data("train"),
                disable_iterator_cache=False,
            )
    train_meter.stop()

    # ===== SWA: finalize after training =====
    if getattr(args, 'use_swa', False) and hasattr(trainer, 'swa_model') and trainer.swa_model is not None:
        try:
            if hasattr(task, 'dataset') and hasattr(args, 'train_subset'):
                train_subset = getattr(args, 'train_subset', 'train')
                swa_save_path = os.path.join(args.save_dir, "checkpoint_swa.pt")
                finalize_swa_model(trainer.swa_model, None, swa_save_path, args)
            else:
                swa_save_path = os.path.join(args.save_dir, "checkpoint_swa.pt")
                torch.save({
                    "model_state_dict": trainer.swa_model.module.state_dict(),
                    "swa_state_dict": trainer.swa_model.state_dict(),
                    "swa_n_averaged": trainer.swa_model.n_averaged.item() if hasattr(trainer.swa_model.n_averaged, 'item') else trainer.swa_model.n_averaged,
                }, swa_save_path)
                print(f"[SWA] Model saved (skipping BN update): {swa_save_path}")
        except Exception as e:
            print(f"[SWA] Finalization error: {e}")
            # Fallback: save SWA weights directly
            swa_save_path = os.path.join(args.save_dir, "checkpoint_swa.pt")
            torch.save({
                "model_state_dict": trainer.swa_model.module.state_dict(),
                "swa_state_dict": trainer.swa_model.state_dict(),
                "swa_n_averaged": trainer.swa_model.n_averaged.item() if hasattr(trainer.swa_model.n_averaged, 'item') else trainer.swa_model.n_averaged,
            }, swa_save_path)
            print(f"[SWA] Model saved (skipping BN update): {swa_save_path}")
    # ========================================

    if ckp_copy_thread is not None:
        ckp_copy_thread.close()
        ckp_copy_thread.join()
    logger.info("done training in {:.1f} seconds".format(train_meter.sum))

    # ===== After training: evaluate on TEST set =====
    if distributed_utils.is_master(args):
        test_subsets = ["test"]
        # Load test dataset
        for test_sub_split in test_subsets:
            if args.concat_datasets:
                task.load_concat_dataset(test_sub_split, combine=False, epoch=1)
            else:
                task.load_dataset(test_sub_split, combine=False, epoch=1)

        print("\n" + "="*60)
        print("Training completed, starting final evaluation on TEST dataset...")
        print("="*60)

        # Evaluate on test set
        test_losses = validate(args, trainer, task, epoch_itr, test_subsets)

        # Dataset sample size statistics
        print("\n" + "="*60)
        print("Dataset Sample Statistics")
        print("="*60)
        for subset_name in args.valid_subset.split(","):
            try:
                dataset = task.dataset(subset_name)
                size = len(dataset)
                print(f"  {subset_name} (validation): {size} samples")
            except Exception:
                print(f"  {subset_name} (validation): unable to get sample size")
        for subset_name in test_subsets:
            try:
                dataset = task.dataset(subset_name)
                size = len(dataset)
                print(f"  {subset_name} (test): {size} samples")
            except Exception:
                print(f"  {subset_name} (test): unable to get sample size")
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


def should_drop_dataset_early(args, valid_losses: dict, metrics_to_dropped_datasets: dict=None, current_epoch=0):
    # skip check if no validation was done in the current epoch
    if len(valid_losses) == 0 or metrics_to_dropped_datasets == None:
        return False
    if args.subdataset_patience <= 0:
        return False
    if current_epoch < args.start_epoch_to_drop_datasets:
        return False

    def is_better(a, b):
        return a > b if args.maximize_metrics_that_drop_datasets else a < b

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


# =============================================================================
# Training Loop Function
# =============================================================================

@metrics.aggregate("train")
def train(
    args, trainer: Trainer, task: tasks.UnicoreTask, epoch_itr, ckp_copy_thread, mixed_sampler=None,
) -> Tuple[List[Optional[float]], bool]:
    """Train the model for one epoch and return validation losses."""
    # Freeze model subset if it is the epoch to freeze (e.g. molecule encoder)
    if args.epoch_to_freeze_molecule_encoder != None and epoch_itr.epoch >= args.epoch_to_freeze_molecule_encoder:
        for child in trainer.model.mol_model.children():
            for param in child.parameters():
                param.requires_grad = False

        print("args.epoch_to_freeze_molecule_encoder != None and epoch_itr.epoch >= args.epoch_to_freeze_molecule_encoder")
        print("Complete freezing params, args.epoch_to_freeze_molecule_encoder: ", args.epoch_to_freeze_molecule_encoder)

    # Determine whether HNM is active for this epoch
    use_hnm = getattr(args, 'use_hard', False) and mixed_sampler is not None

    if use_hnm:
        # HNM: use MixedSampler
        # Still create standard epoch iterator to keep _cur_epoch_itr valid
        # for framework's end_of_epoch() check during checkpoint save
        _ = epoch_itr.next_epoch_itr(
            fix_batches_to_gpus=args.fix_batches_to_gpus,
            shuffle=(epoch_itr.next_epoch_idx > args.curriculum),
        )
        itr = _create_hnm_iterator(mixed_sampler, epoch_itr.dataset, args)
        # Each step already yields [batch], skip GroupedIterator
    else:
        # Standard data iterator
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
            metrics_to_dropped_datasets = json.load(openfile)

    for i, samples in enumerate(progress):
        with metrics.aggregate("train_inner"), torch.autograd.profiler.record_function(
            "train_step-%d" % i
        ):
            log_output = trainer.train_step(samples)

        if log_output is not None:  # not OOM, overflow, ...
            # LOSS: log LambdaRank loss components if enabled
            if getattr(args, 'use_loss', False) and isinstance(log_output, dict):
                if 'loss_pairwise' in log_output and 'loss_lambdarank' in log_output:
                    logger.info("[LambdaRank] loss_total={:.4f}, loss_pairwise={:.4f}, loss_lambdarank={:.4f}".format(
                        log_output.get('loss', 0), log_output['loss_pairwise'], log_output['loss_lambdarank']))

            # log mid-epoch stats
            num_updates = trainer.get_num_updates()
            if num_updates % args.log_interval == 0:
                stats = get_training_stats(metrics.get_smoothed_values("train_inner"))
                progress.log(stats, tag="train_inner", step=num_updates)

                # reset mid-epoch stats after each log interval
                metrics.reset_meters("train_inner")

        end_of_epoch = not itr.has_next()
        validate_and_save_output = validate_and_save(
            args, trainer, task, epoch_itr, valid_subsets, end_of_epoch, ckp_copy_thread, metrics_to_dropped_datasets
        )
        if len(validate_and_save_output) == 3:
            valid_losses, should_stop, dropped_datasets = validate_and_save_output
        else:
            valid_losses, should_stop = validate_and_save_output

        if should_stop:
            break

    # ===== SWA: update at end of each epoch =====
    if getattr(args, 'use_swa', False) and hasattr(trainer, 'swa_model') and trainer.swa_model is not None and hasattr(trainer, 'swa_scheduler') and trainer.swa_scheduler is not None:
        if epoch_itr.epoch >= args.swa_start:
            trainer.swa_model.update_parameters(trainer.model)
            trainer.swa_scheduler.step()
            if epoch_itr.epoch == args.swa_start:
                print(f"[SWA] Starting weight collection from epoch {args.swa_start}")
            if epoch_itr.epoch % 10 == 0 or epoch_itr.epoch == args.swa_start:
                print(f"[SWA] epoch {epoch_itr.epoch} | n_averaged={trainer.swa_model.n_averaged}")
    # =============================================

    # log end-of-epoch stats
    logger.info("end of epoch {} (average epoch stats below)".format(epoch_itr.epoch))
    stats = get_training_stats(metrics.get_smoothed_values("train"))
    progress.print(stats, tag="train", step=num_updates)

    # reset epoch-level meters
    metrics.reset_meters("train")

    # HNM FIX: replace _cur_epoch_itr with dummy so end_of_epoch() returns True
    # (standard iterator was never consumed; HNM iterator handled all training)
    if use_hnm:
        epoch_itr._cur_epoch_itr = _DummyIterator()

    return validate_and_save_output


# =============================================================================
# Validation and Checkpoint Functions
# =============================================================================

def validate_and_save(
    args,
    trainer: Trainer,
    task: tasks.UnicoreTask,
    epoch_itr,
    valid_subsets: List[str],
    end_of_epoch: bool,
    ckp_copy_thread,
    metrics_to_dropped_datasets=None,
) -> Tuple[List[Optional[float]], bool]:
    num_updates = trainer.get_num_updates()
    max_update = args.max_update or math.inf

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
        (not end_of_epoch and do_save)
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
        seed = args.fixed_validation_seed

    with utils.torch_seed(seed):
        trainer.begin_valid_epoch(epoch_itr.epoch)
        valid_losses = []
        for subset in subsets:
            logger.info('begin validation on "{}" subset'.format(subset))

            # Initialize data iterator
            itr = trainer.get_valid_iterator(subset).next_epoch_itr(
                shuffle=False, set_dataset_epoch=False
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
            # don't pollute other aggregators (e.g. train meters)
            with metrics.aggregate(new_root=True) as agg:
                logging_outputs = []
                for i, sample in enumerate(progress):
                    if args.max_valid_steps is not None and i > args.max_valid_steps:
                        break
                    inner_logging_outputs = trainer.valid_step(sample)
                    logging_outputs.extend(inner_logging_outputs)
                task.reduce_metrics(logging_outputs, trainer.get_loss(), subset)

            # log validation stats
            stats = get_valid_stats(args, trainer, agg.get_smoothed_values())
            progress.print(stats, tag=subset, step=trainer.get_num_updates())
            # ===== Key Metrics Highlight =====
            is_test = (subset == "test")
            panel_title = "TEST RESULTS" if is_test else "VALIDATION RESULTS"
            subset_label = "test" if is_test else "validation"
            try:
                dataset_size = len(task.dataset(subset))
                size_info = f" (n={dataset_size})"
            except Exception:
                size_info = ""
            print("\n" + "="*60)
            print(f"{'TEST' if is_test else ''} {panel_title}{size_info}")
            print(f"   Dataset: {subset} | Subtask: {subset_label}")
            print("="*60)
            for key, val in stats.items():
                if "spearman" in key and "coeff" in key:
                    print(f"  * {key}: {val:.4f}")
                elif "pearson" in key and "coeff" in key:
                    print(f"  # {key}: {val:.4f}")
                elif "loss" in key and "in_house" not in key:
                    print(f"  > {key}: {val:.4f}")
                elif "mae" in key and "agg" in key:
                    print(f"  - {key}: {val:.4f}")
            print("="*60 + "\n")
            print(f"validate stats ({subset}): ", stats)
            if args.best_checkpoint_metric in stats:
                valid_losses.append(stats[args.best_checkpoint_metric])
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
    if args.best_checkpoint_metric in stats:
        best_val = stats[args.best_checkpoint_metric]
        best_key = f"best_{args.best_checkpoint_metric}"
        if hasattr(checkpoint_utils.save_checkpoint, "best"):
            prev_best = checkpoint_utils.save_checkpoint.best
            marker = " NEW BEST" if best_val > prev_best else ""
            print(f"\n {args.best_checkpoint_metric}: {best_val:.4f} {marker}")
    print("get_valid_stats stats: ", stats)
    if hasattr(checkpoint_utils.save_checkpoint, "best") and args.best_checkpoint_metric in stats:
        key = "best_{0}".format(args.best_checkpoint_metric)
        best_function = max if args.maximize_best_checkpoint_metric else min
        stats[key] = best_function(
            checkpoint_utils.save_checkpoint.best,
            stats[args.best_checkpoint_metric],
        )
    return stats


# =============================================================================
# CLI Entry Point
# =============================================================================

def cli_main(
    modify_parser: Optional[Callable[[argparse.ArgumentParser], None]] = None
) -> None:
    # ===== COMBINE4: Read module activation switches from environment variables =====
    use_swa = os.environ.get('COMET_USE_SWA', '0') == '1'
    use_loss = os.environ.get('COMET_USE_LOSS', '0') == '1'
    use_hard = os.environ.get('COMET_USE_HARD', '0') == '1'
    use_aug = os.environ.get('COMET_USE_AUG', '0') == '1'

    print(f"[COMBINE4] Active modules: SWA={use_swa}, LOSS={use_loss}, HARD={use_hard}, AUG={use_aug}")

    # LOSS: log LambdaRank config if enabled
    if use_loss:
        print(f"[LOSS] LambdaRank mixed loss enabled: alpha={os.environ.get('COMET_LAMBDARANK_ALPHA', '0.3')}")
    # =============================================================================

    print("running cli_main")
    parser = options.get_training_parser()
    args = options.parse_args_and_arch(parser, modify_parser=modify_parser)

    # Store module switches on args for access throughout training
    args.use_swa = use_swa
    args.use_loss = use_loss
    args.use_hard = use_hard
    args.use_aug = use_aug

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
