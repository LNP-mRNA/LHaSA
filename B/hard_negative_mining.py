#!/usr/bin/env python3
"""
hard_negative_mining.py -- Hard Negative Mining Module
=====================================================
困难负样本挖掘与混合采样，用于Uni-MOL LNP效能预测模型的聚焦训练。
Focuses contrastive training on hard sample pairs to improve ranking performance.

核心组件 (Core Components):
    - mine_hard_pairs(): Full-set vectorized hard negative pair mining
    - MixedSampler: Normal/hard sample mixed batch sampler with curriculum scheduling
    - get_hard_pair_indices(): Extract unique indices from hard pair pool
    - collate_batch_by_indices(): Collate batch samples by given indices

作者: HNM Module
"""

import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict, Any, Iterator
import logging
import math
import random
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# 1. mine_hard_pairs -- 全量向量化困难样本对挖掘
# =============================================================================

def mine_hard_pairs(
    model,
    dataloader,
    device: torch.device,
    top_k: int = 2000,
    eta: float = 1e-6,
    label_gap_threshold: float = 0.15,
    target_key: str = "finetune_target",
) -> List[Tuple[int, int, float]]:
    """
    全训练集困难负样本挖掘（Hard Negative Mining）。
    
    对全训练集做前向传播（无梯度），收集所有预测分数和真实标签，
    向量化计算所有样本对的困难度分数 H_ij = |y_i - y_j| / (|s_i - s_j| + eta)，
    过滤标签差距过小的对，返回Top-K困难对。
    
    Vectorized hard negative pair mining over the full training set.
    Computes hardness score H_ij = |y_i - y_j| / (|s_i - s_j| + eta) for all pairs.
    
    Args:
        model: 训练中的模型，需支持 model(sample) 返回 (logits, cls_representations)
        dataloader: 训练数据加载器，yield 的 sample 格式兼容 Uni-MOL LNP
        device: 计算设备 (torch.device)
        top_k: 选取的困难对数量
        eta: 困难度分数分母平滑项，防止除零
        label_gap_threshold: 标签差距过滤阈值，|y_i - y_j| <= threshold 的对被丢弃
        target_key: target dict 中的键名，用于获取回归目标值
    
    Returns:
        List[Tuple[int, int, float]]: Top-K困难对列表，每项为 (idx_i, idx_j, hardness_score)，
                                     按 hardness_score 降序排列
    
    Notes:
        - 本函数自动处理 fp16 模式（内部转换为 float32 计算困难度，避免精度问题）
        - 兼容多任务场景：当 logits 和 targets 均为 dict 时，使用第一个任务的值
        - 使用 model.eval() + torch.no_grad()，不修改模型状态
    """
    model.eval()
    
    all_scores = []      # 收集所有预测分数
    all_labels = []      # 收集所有真实标签
    
    # -----------------------------------------------------------------------
    # Step 1: 全训练集前向传播，收集预测和标签
    # -----------------------------------------------------------------------
    logger.info(
        "[HNM] Starting hard pair mining: top_k=%d, eta=%.1e, label_gap_threshold=%.3f",
        top_k, eta, label_gap_threshold
    )
    
    with torch.no_grad():
        for batch_idx, sample in enumerate(dataloader):
            # 将sample移动到正确设备
            sample = _move_sample_to_device(sample, device)
            
            # 前向传播: model(sample) -> (logits, cls_representations)
            # 兼容 fp16: 前向传播在模型当前精度下进行
            net_output, _ = model(sample)
            
            # ---- 提取预测分数 (兼容多任务场景) ----
            # net_output 可能是 dict（多任务）或 tensor（单任务）
            if isinstance(net_output, dict):
                # 多任务场景：使用第一个任务的输出
                first_task_name = list(net_output.keys())[0]
                batch_scores = net_output[first_task_name]
            else:
                batch_scores = net_output
            
            # ---- 提取真实标签 ----
            target_data = sample["target"][target_key]
            if isinstance(target_data, dict):
                # 多任务场景：使用与预测对应的任务标签
                if isinstance(net_output, dict):
                    first_task_name = list(net_output.keys())[0]
                    batch_labels = target_data[first_task_name]
                else:
                    # 若预测是单任务但标签是多任务，使用第一个
                    first_task_name = list(target_data.keys())[0]
                    batch_labels = target_data[first_task_name]
            else:
                batch_labels = target_data
            
            # 展平为 [batch_size] 或 [batch_size, 1] -> [batch_size]
            batch_scores = batch_scores.view(-1)
            batch_labels = batch_labels.view(-1)
            
            # 过滤掉标签为 NaN 的样本（多任务场景下部分样本可能没有某些任务的标签）
            valid_mask = ~torch.isnan(batch_labels)
            
            if valid_mask.sum() > 0:
                all_scores.append(batch_scores[valid_mask].float())
                all_labels.append(batch_labels[valid_mask].float())
    
    # -----------------------------------------------------------------------
    # Step 2: 拼接所有batch的结果
    # -----------------------------------------------------------------------
    if len(all_scores) == 0 or len(all_labels) == 0:
        logger.warning("[HNM] No valid samples found for hard mining, returning empty list")
        return []
    
    all_scores = torch.cat(all_scores, dim=0)  # [N]
    all_labels = torch.cat(all_labels, dim=0)  # [N]
    n_samples = all_scores.shape[0]
    
    logger.info("[HNM] Collected %d valid samples for pair mining", n_samples)
    
    if n_samples < 2:
        logger.warning("[HNM] Insufficient samples (%d < 2) for pair mining", n_samples)
        return []
    
    # -----------------------------------------------------------------------
    # Step 3: 向量化计算所有样本对的困难度分数
    # -----------------------------------------------------------------------
    # 使用 float32 进行困难度计算，避免 fp16 精度不足
    all_scores = all_scores.float()
    all_labels = all_labels.float()
    
    # 计算标签差异矩阵: |y_i - y_j|
    # 使用广播: [N, 1] - [1, N] -> [N, N]
    label_diff = torch.abs(all_labels.unsqueeze(1) - all_labels.unsqueeze(0))  # [N, N]
    
    # 计算预测差异矩阵: |s_i - s_j|
    score_diff = torch.abs(all_scores.unsqueeze(1) - all_scores.unsqueeze(0))  # [N, N]
    
    # 计算困难度矩阵: H_ij = |y_i - y_j| / (|s_i - s_j| + eta)
    hardness_matrix = label_diff / (score_diff + eta)  # [N, N]
    
    # -----------------------------------------------------------------------
    # Step 4: 过滤无效对
    # -----------------------------------------------------------------------
    # 创建上三角掩码（排除 i == j 的对，且每对只取一次）
    triu_mask = torch.triu(torch.ones_like(hardness_matrix), diagonal=1).bool()
    
    # 过滤标签差距过小的对: |y_i - y_j| > label_gap_threshold
    gap_mask = label_diff > label_gap_threshold
    
    # 合并掩码
    valid_mask = triu_mask & gap_mask
    
    valid_count = valid_mask.sum().item()
    if valid_count == 0:
        logger.warning(
            "[HNM] No valid pairs after filtering (label_gap_threshold=%.3f), "
            "returning empty list", label_gap_threshold
        )
        return []
    
    logger.info("[HNM] Valid pairs after filtering: %d / %d", valid_count, n_samples * (n_samples - 1) // 2)
    
    # -----------------------------------------------------------------------
    # Step 5: 提取有效对的困难度分数和索引
    # -----------------------------------------------------------------------
    # 获取满足条件的索引对
    valid_indices = torch.nonzero(valid_mask, as_tuple=False)  # [M, 2]
    valid_hardness = hardness_matrix[valid_mask]               # [M]
    
    # -----------------------------------------------------------------------
    # Step 6: 使用 torch.topk 选取 Top-K 困难对
    # -----------------------------------------------------------------------
    k = min(top_k, valid_count)
    
    # topk 返回的是最大的 k 个值
    topk_values, topk_local_indices = torch.topk(valid_hardness, k=k, largest=True, sorted=True)
    
    # 转换为全局样本索引
    topk_global_indices = valid_indices[topk_local_indices]  # [k, 2]
    
    # -----------------------------------------------------------------------
    # Step 7: 组装结果
    # -----------------------------------------------------------------------
    hard_pairs = []
    for i in range(k):
        idx_i = topk_global_indices[i, 0].item()
        idx_j = topk_global_indices[i, 1].item()
        h_score = topk_values[i].item()
        hard_pairs.append((idx_i, idx_j, h_score))
    
    logger.info(
        "[HNM] Mining complete: top hardness=%.4f, min hardness=%.4f, "
        "mean label_gap=%.4f",
        hard_pairs[0][2] if hard_pairs else 0.0,
        hard_pairs[-1][2] if hard_pairs else 0.0,
        torch.mean(label_diff[valid_mask]).item() if valid_count > 0 else 0.0,
    )
    
    return hard_pairs


# =============================================================================
# 2. MixedSampler -- 正常/困难样本混合采样器
# =============================================================================

class MixedSampler:
    """
    混合采样器：从正常样本和困难样本中按比例混合采样。
    Mixed batch sampler: generates batches containing either normal or hard samples.
    
    支持困难比例的课程式调度（Curriculum Scheduling）：
    从初始比例逐步提升到目标比例，让模型在训练初期接触更多简单样本，
    后期逐步增加困难样本比例，实现渐进式学习。
    
    Supports curriculum scheduling for hard_ratio: gradually increases the proportion
    of hard batches from initial to target ratio over training epochs.
    
    Args:
        dataset_size: 训练集总样本数
        hard_pair_pool: 困难样本对池，List[(idx_i, idx_j, hardness_score)]
        batch_size: 每个batch的样本数
        normal_ratio: 初始正常batch比例（1 - hard_ratio）
        hard_ratio: 初始困难batch比例
        curriculum_schedule: 是否启用课程式调度
        hard_ratio_final: 课程式调度的最终困难比例
        warmup_epochs: 课程式调度的warmup轮数（线性提升到hard_ratio_final）
    
    Usage:
        >>> sampler = MixedSampler(len(dataset), hard_pairs, batch_size=32, hard_ratio=0.3)
        >>> sampler.update_hard_pool(new_hard_pairs)  # 动态更新困难池
        >>> for batch_indices in sampler:
        ...     batch = collate_batch_by_indices(dataset, batch_indices)
        ...     # train step...
        >>> # 每个epoch后更新hard_ratio（课程式调度）
        >>> sampler.step_epoch()  # 自动更新当前epoch和hard_ratio
    """
    
    def __init__(
        self,
        dataset_size: int,
        hard_pair_pool: List[Tuple[int, int, float]],
        batch_size: int = 32,
        normal_ratio: float = 0.7,
        hard_ratio: float = 0.3,
        curriculum_schedule: bool = False,
        hard_ratio_final: Optional[float] = None,
        warmup_epochs: int = 5,
        seed: int = 42,
    ):
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.normal_ratio = normal_ratio
        self._hard_ratio_initial = hard_ratio
        self._hard_ratio_current = hard_ratio
        self.curriculum_schedule = curriculum_schedule
        self.hard_ratio_final = hard_ratio_final or min(hard_ratio * 2.0, 0.8)
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0
        self.seed = seed
        
        # 困难样本池
        self.hard_pair_pool: List[Tuple[int, int, float]] = []
        self.hard_indices: List[int] = []
        
        # 随机数生成器（确保可复现）
        self.rng = random.Random(seed)
        
        # 初始化困难池
        if hard_pair_pool:
            self.update_hard_pool(hard_pair_pool)
        
        logger.info(
            "[HNM] MixedSampler initialized: dataset_size=%d, batch_size=%d, "
            "normal_ratio=%.2f, hard_ratio=%.2f, curriculum=%s",
            dataset_size, batch_size, normal_ratio, hard_ratio, curriculum_schedule
        )
        if curriculum_schedule:
            logger.info(
                "[HNM] Curriculum scheduling: warmup_epochs=%d, hard_ratio_final=%.2f",
                warmup_epochs, self.hard_ratio_final
            )
    
    def update_hard_pool(self, hard_pair_pool: List[Tuple[int, int, float]]) -> None:
        """
        动态更新困难样本池。
        Updates the hard negative pair pool (typically called after mining).
        
        Args:
            hard_pair_pool: 新的困难样本对列表，完全替换旧池
        """
        self.hard_pair_pool = hard_pair_pool
        self.hard_indices = get_hard_pair_indices(hard_pair_pool)
        
        logger.info(
            "[HNM] Hard pool updated: %d pairs, %d unique indices",
            len(hard_pair_pool), len(self.hard_indices)
        )
    
    def step_epoch(self) -> None:
        """
        进入下一个epoch。
        如果启用课程式调度，更新当前困难比例。
        Advances to the next epoch; updates hard_ratio if curriculum scheduling is enabled.
        """
        self.current_epoch += 1
        
        if self.curriculum_schedule:
            # 线性插值: hard_ratio_current = initial + (final - initial) * min(epoch / warmup, 1)
            progress = min(self.current_epoch / max(self.warmup_epochs, 1), 1.0)
            self._hard_ratio_current = (
                self._hard_ratio_initial
                + (self.hard_ratio_final - self._hard_ratio_initial) * progress
            )
            logger.info(
                "[HNM] Epoch %d: hard_ratio updated to %.3f (progress=%.2f%%)",
                self.current_epoch, self._hard_ratio_current, progress * 100
            )
    
    def _get_normal_batch(self) -> List[int]:
        """从全数据集中随机采样一个batch的索引（正常batch）。"""
        return self.rng.sample(range(self.dataset_size), min(self.batch_size, self.dataset_size))
    
    def _get_hard_batch(self) -> List[int]:
        """从困难样本涉及的索引中采样一个batch（困难batch）。"""
        if len(self.hard_indices) == 0:
            # 如果困难池为空，回退到正常采样
            logger.debug("[HNM] Hard pool empty, falling back to normal sampling")
            return self._get_normal_batch()
        
        batch_size = min(self.batch_size, len(self.hard_indices))
        
        if batch_size < self.batch_size and len(self.hard_indices) > 0:
            # 困难样本不足时，允许重复采样
            return self.rng.choices(self.hard_indices, k=self.batch_size)
        else:
            return self.rng.sample(self.hard_indices, batch_size)
    
    def __iter__(self) -> Iterator[List[int]]:
        """
        生成混合batch的索引列表。
        Yields batches of indices, where each batch is either normal or hard
        based on the current hard_ratio.
        
        Yields:
            List[int]: 一个batch的样本索引列表
        """
        # 计算本epoch需要的batch数量
        n_total = self.dataset_size
        n_batches = math.ceil(n_total / self.batch_size)
        
        # 决定每个batch的类型（正常或困难）
        # 使用当前hard_ratio决定困难batch的数量
        n_hard_batches = int(n_batches * self._hard_ratio_current)
        n_normal_batches = n_batches - n_hard_batches
        
        # 随机打乱batch类型的顺序
        batch_types = ([True] * n_hard_batches + [False] * n_normal_batches)
        self.rng.shuffle(batch_types)
        
        logger.debug(
            "[HNM] Starting epoch: %d batches total (%d hard, %d normal), hard_ratio=%.3f",
            n_batches, n_hard_batches, n_normal_batches, self._hard_ratio_current
        )
        
        for is_hard in batch_types:
            if is_hard:
                yield self._get_hard_batch()
            else:
                yield self._get_normal_batch()
    
    def __len__(self) -> int:
        """返回每个epoch的batch数量。"""
        return math.ceil(self.dataset_size / self.batch_size)
    
    @property
    def hard_ratio(self) -> float:
        """当前困难batch比例（考虑课程式调度）。"""
        return self._hard_ratio_current


# =============================================================================
# 3. get_hard_pair_indices -- 从困难对池提取唯一索引
# =============================================================================

def get_hard_pair_indices(
    hard_pair_pool: List[Tuple[int, int, float]],
    min_hardness: Optional[float] = None,
) -> List[int]:
    """
    从困难样本对池中提取所有涉及的样本索引（去重）。
    Extracts all unique sample indices involved in the hard pair pool.
    
    Args:
        hard_pair_pool: 困难样本对列表，每项为 (idx_i, idx_j, hardness_score)
        min_hardness: 可选的最小困难度阈值，仅包含 hardness_score >= min_hardness 的对
    
    Returns:
        List[int]: 去重后的样本索引列表
    
    Example:
        >>> pairs = [(0, 5, 3.2), (1, 3, 2.8), (5, 7, 1.5)]
        >>> get_hard_pair_indices(pairs)
        [0, 5, 1, 3, 7]
    """
    if not hard_pair_pool:
        return []
    
    unique_indices = set()
    for idx_i, idx_j, hardness in hard_pair_pool:
        if min_hardness is not None and hardness < min_hardness:
            continue
        unique_indices.add(idx_i)
        unique_indices.add(idx_j)
    
    return sorted(list(unique_indices))


# =============================================================================
# 4. collate_batch_by_indices -- 按索引取batch并堆叠
# =============================================================================

def collate_batch_by_indices(dataset, indices: List[int]) -> Dict[str, Any]:
    """
    根据给定索引列表从数据集中取出样本并拼合成一个batch。
    Fetches samples by indices and collates them into a single batch dict.
    
    兼容Uni-MOL LNP的sample格式，支持多种数据类型的自动堆叠：
    - torch.Tensor: 使用 torch.stack (同shape) 或 torch.cat
    - list: 收集为列表
    - dict: 递归处理每个键
    - int/float: 收集为列表
    
    兼容Dict类型的target（多任务场景）。
    
    Args:
        dataset: 数据集对象，需支持 dataset[idx] 返回sample dict
        indices: 样本索引列表
    
    Returns:
        Dict[str, Any]: 拼合后的batch字典，结构与单个sample一致
    
    Raises:
        ValueError: 当索引列表为空时
        TypeError: 当数据类型不支持堆叠时
    """
    if not indices:
        raise ValueError("indices list is empty")
    
    # 取出所有samples
    samples = [dataset[idx] for idx in indices]
    
    if len(samples) == 0:
        raise ValueError("no valid samples found for the given indices")
    
    # 以第一个sample的keys为基准
    batch = {}
    first_sample = samples[0]
    
    for key in first_sample:
        values = [sample[key] for sample in samples]
        batch[key] = _collate_field(values, key)
    
    return batch


# =============================================================================
# 内部辅助函数 (Internal Helpers)
# =============================================================================

def _collate_field(values: List[Any], field_name: str = "") -> Any:
    """
    对一组字段值进行智能堆叠。
    Smart collate function that handles various data types.
    
    Args:
        values: 同字段的多个值列表
        field_name: 字段名（用于调试日志）
    
    Returns:
        堆叠后的值，类型取决于输入类型
    """
    if not values:
        return values
    
    first_value = values[0]
    
    # ---- Case 1: torch.Tensor ----
    if isinstance(first_value, torch.Tensor):
        # 检查所有tensor的shape是否一致
        shapes = [v.shape for v in values]
        
        if all(s == shapes[0] for s in shapes):
            # 所有shape相同，使用stack在第0维堆叠
            return torch.stack(values, dim=0)
        else:
            # Shape不一致，使用pad_sequence或cat
            # 对于一维tensor（变长序列），使用pad_sequence
            if all(v.dim() == 1 for v in values):
                return torch.nn.utils.rnn.pad_sequence(
                    values, batch_first=True, padding_value=0
                )
            else:
                # 其他情况，尝试cat在第0维
                try:
                    return torch.cat(values, dim=0)
                except RuntimeError:
                    # 如果cat也失败，返回列表
                    logger.debug(
                        "[HNM] Cannot stack/cat tensors for field '%s', returning list",
                        field_name
                    )
                    return values
    
    # ---- Case 2: dict (递归处理，支持多任务target) ----
    elif isinstance(first_value, dict):
        collated_dict = {}
        dict_keys = first_value.keys()
        
        for sub_key in dict_keys:
            sub_values = [v[sub_key] for v in values]
            collated_dict[sub_key] = _collate_field(sub_values, f"{field_name}.{sub_key}")
        
        return collated_dict
    
    # ---- Case 3: list ----
    elif isinstance(first_value, list):
        # 如果内部元素是张量，尝试堆叠
        if first_value and isinstance(first_value[0], torch.Tensor):
            # 对每个位置的list元素进行堆叠
            try:
                max_len = max(len(v) for v in values)
                result = []
                for i in range(max_len):
                    tensors_at_i = []
                    for v in values:
                        if i < len(v):
                            tensors_at_i.append(v[i])
                        else:
                            # 用零填充
                            tensors_at_i.append(torch.zeros_like(v[0]))
                    result.append(torch.stack(tensors_at_i, dim=0))
                return result
            except (RuntimeError, IndexError):
                # 堆叠失败，返回列表的列表
                return values
        else:
            # 非tensor列表，返回列表的列表
            return values
    
    # ---- Case 4: int / float / bool / str 等标量 ----
    elif isinstance(first_value, (int, float, bool, str)):
        return values  # 收集为列表
    
    # ---- Case 5: numpy array ----
    elif isinstance(first_value, np.ndarray):
        try:
            return torch.from_numpy(np.stack(values, axis=0))
        except (ValueError, RuntimeError):
            return values
    
    # ---- Case 6: 其他类型，直接返回列表 ----
    else:
        return values


def _move_sample_to_device(sample: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """
    递归地将sample中的所有张量移动到指定设备。
    Recursively moves all tensors in a sample dict to the given device.
    
    兼容fp16模式：只移动张量，不改变数据类型。
    Compatible with fp16: only moves tensors to device, preserves dtype.
    
    Args:
        sample: 样本字典，可能包含嵌套dict和tensor
        device: 目标设备
    
    Returns:
        设备迁移后的样本字典
    """
    if isinstance(sample, dict):
        return {k: _move_sample_to_device(v, device) for k, v in sample.items()}
    elif isinstance(sample, torch.Tensor):
        return sample.to(device, non_blocking=True)
    elif isinstance(sample, list):
        return [_move_sample_to_device(item, device) for item in sample]
    else:
        return sample


# =============================================================================
# HNM训练循环集成辅助 (Training Loop Integration Helpers)
# =============================================================================

def should_mine_hard_pairs(epoch: int, mining_interval: int = 5) -> bool:
    """
    判断当前epoch是否需要进行困难样本挖掘。
    Determines whether hard negative mining should be performed this epoch.
    
    通常在固定间隔进行挖掘（如每5个epoch），避免过度开销。
    
    Args:
        epoch: 当前epoch编号（从1开始）
        mining_interval: 挖掘间隔epoch数
    
    Returns:
        bool: 是否需要进行挖掘
    """
    return epoch % mining_interval == 0 or epoch == 1


def get_hardness_statistics(hard_pairs: List[Tuple[int, int, float]]) -> Dict[str, float]:
    """
    计算困难对池的统计信息，用于监控。
    Computes statistics of the hard pair pool for monitoring.
    
    Args:
        hard_pairs: 困难样本对列表
    
    Returns:
        Dict: 包含max/mean/min hardness等统计信息
    """
    if not hard_pairs:
        return {"count": 0, "max_hardness": 0.0, "mean_hardness": 0.0, "min_hardness": 0.0}
    
    hardness_scores = [h for _, _, h in hard_pairs]
    unique_indices = set()
    for i, j, _ in hard_pairs:
        unique_indices.add(i)
        unique_indices.add(j)
    
    return {
        "count": len(hard_pairs),
        "unique_samples": len(unique_indices),
        "max_hardness": max(hardness_scores),
        "mean_hardness": sum(hardness_scores) / len(hardness_scores),
        "min_hardness": min(hardness_scores),
    }


def log_hardness_statistics(hard_pairs: List[Tuple[int, int, float]], prefix: str = "[HNM]") -> None:
    """
    记录困难对池的统计信息到日志。
    Logs statistics of the hard pair pool.
    
    Args:
        hard_pairs: 困难样本对列表
        prefix: 日志前缀
    """
    stats = get_hardness_statistics(hard_pairs)
    logger.info(
        "%s Hard Pair Statistics: count=%d, unique_samples=%d, "
        "hardness=[%.4f, %.4f, %.4f] (min/mean/max)",
        prefix, stats["count"], stats["unique_samples"],
        stats["min_hardness"], stats["mean_hardness"], stats["max_hardness"]
    )


# =============================================================================
# 单元测试入口 (Self-test)
# =============================================================================

if __name__ == "__main__":
    # 简单的自测试
    import unittest
    
    class TestHardNegativeMining(unittest.TestCase):
        
        def test_get_hard_pair_indices(self):
            pairs = [(0, 5, 3.2), (1, 3, 2.8), (5, 7, 1.5)]
            indices = get_hard_pair_indices(pairs)
            self.assertEqual(sorted(indices), [0, 1, 3, 5, 7])
        
        def test_get_hard_pair_indices_with_threshold(self):
            pairs = [(0, 5, 3.2), (1, 3, 2.8), (5, 7, 1.5)]
            indices = get_hard_pair_indices(pairs, min_hardness=2.0)
            self.assertEqual(sorted(indices), [0, 1, 3, 5])
        
        def test_get_hard_pair_indices_empty(self):
            self.assertEqual(get_hard_pair_indices([]), [])
        
        def test_collate_tensor(self):
            """测试tensor堆叠"""
            values = [torch.randn(3, 4) for _ in range(5)]
            result = _collate_field(values)
            self.assertEqual(result.shape, (5, 3, 4))
        
        def test_collate_dict(self):
            """测试dict递归堆叠（多任务target场景）"""
            values = [
                {"task_a": torch.tensor(1.0), "task_b": torch.tensor(2.0)},
                {"task_a": torch.tensor(3.0), "task_b": torch.tensor(4.0)},
            ]
            result = _collate_field(values)
            self.assertTrue(isinstance(result, dict))
            self.assertEqual(result["task_a"].shape, (2,))
            self.assertEqual(result["task_b"].tolist(), [2.0, 4.0])
        
        def test_collate_mixed_types(self):
            """测试混合类型"""
            values = [1, 2, 3]
            result = _collate_field(values)
            self.assertEqual(result, [1, 2, 3])
        
        def test_mixed_sampler_length(self):
            """测试MixedSampler长度"""
            sampler = MixedSampler(
                dataset_size=100,
                hard_pair_pool=[(0, 1, 2.0), (2, 3, 1.5)],
                batch_size=32,
            )
            self.assertEqual(len(sampler), math.ceil(100 / 32))
        
        def test_mixed_sampler_curriculum(self):
            """测试课程式调度"""
            sampler = MixedSampler(
                dataset_size=100,
                hard_pair_pool=[(0, 1, 2.0)],
                batch_size=32,
                hard_ratio=0.2,
                curriculum_schedule=True,
                hard_ratio_final=0.6,
                warmup_epochs=5,
            )
            self.assertAlmostEqual(sampler.hard_ratio, 0.2)
            for _ in range(5):
                sampler.step_epoch()
            self.assertAlmostEqual(sampler.hard_ratio, 0.6)
        
        def test_hardness_statistics(self):
            """测试统计信息计算"""
            pairs = [(0, 1, 3.0), (2, 3, 1.0), (4, 5, 2.0)]
            stats = get_hardness_statistics(pairs)
            self.assertEqual(stats["count"], 3)
            self.assertEqual(stats["max_hardness"], 3.0)
            self.assertAlmostEqual(stats["mean_hardness"], 2.0)
            self.assertEqual(stats["min_hardness"], 1.0)
        
        def test_should_mine(self):
            """测试挖掘时机判断"""
            self.assertTrue(should_mine_hard_pairs(1, 5))
            self.assertTrue(should_mine_hard_pairs(5, 5))
            self.assertFalse(should_mine_hard_pairs(2, 5))
            self.assertTrue(should_mine_hard_pairs(10, 5))
    
    # 运行测试
    unittest.main(verbosity=2, exit=False)
