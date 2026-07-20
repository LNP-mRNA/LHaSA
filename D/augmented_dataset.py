#!/usr/bin/env python3
"""
augmented_dataset.py -- 结构化增强数据集包装器
================================================
包装 LMDBDataset，在 __getitem__ 中自动应用配方级结构化增强。
仅训练集启用增强，val/test 保持原样。
"""

import logging
from typing import Any, Dict, Optional

from .formulation_augmentation import FormulationAugmenter

logger = logging.getLogger(__name__)


class FormulationAugmentedDataset:
    """配方增强数据集包装器。

    包装原始数据集（如 LMDBDataset），在 __getitem__ 时自动对配方数据
    应用结构化增强（同类别替换、比例扰动、组分Dropout）。

    Args:
        dataset: 原始数据集（需支持 __getitem__ 和 __len__）
        augmenter: FormulationAugmenter 实例
        split: 数据集划分 ("train"/"valid"/"test"/"infer")
        enable_aug: 是否启用增强（默认 train 启用，其余禁用）
    """

    def __init__(
        self,
        dataset: Any,
        augmenter: Optional[FormulationAugmenter] = None,
        split: str = "train",
        enable_aug: bool = True,
    ):
        self.dataset = dataset
        self.augmenter = augmenter
        self.split = split
        self.enable_aug = enable_aug and (split == "train")

        # 统计信息
        self._total_calls = 0
        self._aug_applied = 0
        self._aug_ops_log: list = []

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """获取样本，如启用了增强则先对配方应用结构化变异。"""
        raw_sample = self.dataset[idx]
        self._total_calls += 1

        # 仅训练集且启用了增强时才处理
        if not self.enable_aug or self.augmenter is None:
            return raw_sample

        # 深拷贝避免修改原始数据
        sample = dict(raw_sample)

        # 检查是否包含配方数据（component_list 字段）
        if "component_list" not in sample:
            return sample

        # 应用结构化增强
        try:
            aug_sample, aug_log = self.augmenter.apply(sample)
            if aug_log.get("applied", False):
                self._aug_applied += 1
                if aug_log.get("operations"):
                    self._aug_ops_log.extend(aug_log["operations"])
            return aug_sample
        except Exception as e:
            # 增强失败时回退到原始样本，避免中断训练
            logger.warning(f"Formulation augmentation failed at idx={idx}: {e}")
            return sample

    def __len__(self) -> int:
        return len(self.dataset)

    def set_epoch(self, epoch: int):
        """更新增强调度器的当前 epoch（应在每轮训练开始时调用）。"""
        if self.augmenter is not None:
            self.augmenter.set_epoch(epoch)

    def get_stats(self) -> Dict[str, Any]:
        """返回增强统计信息。"""
        return {
            "total_calls": self._total_calls,
            "aug_applied": self._aug_applied,
            "aug_ratio": self._aug_applied / max(self._total_calls, 1),
            "recent_ops": self._aug_ops_log[-20:],  # 最近20条操作日志
        }

    def reset_stats(self):
        """重置统计信息。"""
        self._total_calls = 0
        self._aug_applied = 0
        self._aug_ops_log = []

    def __repr__(self) -> str:
        return (
            f"FormulationAugmentedDataset(dataset={self.dataset}, "
            f"split={self.split}, enable_aug={self.enable_aug}, "
            f"augmenter={self.augmenter})"
        )
