#!/usr/bin/env python3
"""
formulation_augmentation.py -- 化学先验驱动的配方结构化数据增强
=====================================================================
基于 COMET 优化方案第4章实现，包含三类操作：
  1. 同类别脂质替换 (Intra-class Lipid Substitution)
  2. 比例范围约束扰动 (Ratio-constrained Perturbation)
  3. 组分 Dropout (Component Dropout)

还包含课程式概率调度器 (AugmentationScheduler)，支持动态调整增强强度。
"""

import copy
import logging
import random
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# 0. 增强操作概率配置（修改这四个数字即可调整增强幅度）
# ============================================================================
# 同类别脂质替换概率：IL/HL/sterol（非PEG类别）
# 值越大，配方中脂质分子被同类替换的概率越高
P_REPLACE_NON_PEG = 0.6      # 建议范围: 0.2 ~ 0.8

# 同类别脂质替换概率：PEG（PEG脂质保守替换，概率较低）
# PEG 是锚定分子，变化太大会影响NP结构稳定性
P_REPLACE_PEG = 0.3          # 建议范围: 0.05 ~ 0.4

# 比例范围约束扰动概率
# 每次增强时，有多大可能对某个组分的摩尔百分比施加高斯扰动
P_PERTURB_RATIO = 0.5        # 建议范围: 0.1 ~ 0.6

# 组分 Dropout 概率
# 每次增强时，有多大可能随机移除一个非PEG组分并重新归一化
P_COMPONENT_DROPOUT = 0.1   # 建议范围: 0.0 ~ 0.15

# ============================================================================
# 1. LANCE 数据集四类脂质的完整替换池
# ============================================================================
# 替换池直接映射至 LANCE 实验设计矩阵
# 实际集成时可通过 dataset 自动生成，此处提供完整的 LANCE 基准池

REPLACEMENT_POOL = {
    # Ionizable Lipid (IL): 7种候选分子
    "IL": [
        "CCCCCCCCCCCCCCCC(=O)OCC(COC(=O)CCCCCCCCCCCCCCC)NC(C)C",  # DLin-MC3-DMA
        "CCCCCCCCCCCCCCCCC(=O)O[C@H](COC(=O)CCCCCCCCCCCCCCC)COP(=O)(O)OCCN(C)C",  # SM-102
        "CCCCCCCCCCCCCCCC(=O)OCC(COC(=O)CCCCCCCCCCCCCCC)N(C)C",  # ALC-0315
        "CCCCCCCCCCCCCCCCCC(=O)OCC(COC(=O)CCCCCCCCCCCCCCC)NC(C)C",  # CKK-E12
        "CCCCCCCCCCCCCC(=O)OCC(COC(=O)CCCCCCCCCCCCC)NC(C)C",  # C12-200
        "CCCCCCCCCCCCCCCC(=O)OCC(COC(=O)CCCCCCCCCCCCCCC)N1CCOCC1",  # L319
        "CCCCCCCCCCCCCCCC(=O)OCC(COC(=O)CCCCCCCCCCCCCCC)N(C)CC",  # KC2
    ],
    # Helper Lipid (HL): 2种候选分子
    "HL": [
        "CCCCCCCCCCCCCCCC(=O)OCC(COP(=O)(O)OCCN)OC(=O)CCCCCCCCCCCCCCC",  # DOPE
        "CCCCCCCCCCCCCCCC(=O)OCC(COP(=O)(O)OCC[N+](C)(C)C)OC(=O)CCCCCCCCCCCCCCC",  # DSPC
    ],
    # Sterol: 3种候选分子
    "sterol": [
        "CC(C)CCCC(C)C1CCC2C3CC=C4CC(O)CCC4(C)C3CCC12C",  # Cholesterol
        "CCCCCCCCCCCCCCCC(=O)O[C@H]1CC[C@@]2(C)C(=CC[C@H]3C4CCCC4(C)CC[C@@H]32)C1",  # DC-cholesterol
        "CC(C)CCCC(C)C1CCC2C3CC=C4C[C@@H](O)CCC4(C)C3CCC12C",  # Beta-sitosterol
    ],
    # PEG Lipid: 2种候选分子
    "PEG": [
        "CCCCCCCCCCCCCCCC(=O)OCCOC(=O)CCCCCCCCCCCCCCC",  # C14-PEG (DMG-PEG2k)
        "CCCCCCCCCCCCCCCCCC(=O)OCCOC(=O)CCCCCCCCCCCCCCCCC",  # C18-PEG (DSG-PEG2k)
    ],
}


# ============================================================================
# 2. 各类别摩尔百分比约束范围（来源于 LANCE 经验分布）
# ============================================================================

RATIO_BOUNDS = {
    "IL": (35.0, 50.0),      # 可离子化脂质 35%-50%
    "HL": (10.0, 40.0),      # 辅助脂质 10%-40%
    "sterol": (10.0, 45.0),  # 固醇 10%-45%
    "PEG": (1.5, 3.0),       # PEG 脂质 1.5%-3%
}


# ============================================================================
# 3. 从数据集自动构建替换池
# ============================================================================

def build_pool_from_dataset(dataset: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """从 LANCE 数据集自动构建替换池，确保 SMILES 与训练数据一致。

    Args:
        dataset: 数据集样本列表，每个样本包含 component_list

    Returns:
        pool: {component_type: [smi1, smi2, ...]} 的替换池字典
    """
    pool = defaultdict(list)
    for sample in dataset:
        component_list = sample.get("component_list", [])
        for component in component_list:
            ctype = component.get("component_type", "")
            smi = component.get("smi", "")
            name = component.get("name", ctype)
            if ctype and smi and smi not in pool[ctype]:
                pool[ctype].append(smi)
    # 只保留有至少2种候选的类别（否则无法替换）
    return {k: v for k, v in pool.items() if len(v) >= 2}


# ============================================================================
# 4. 结构化增强主函数
# ============================================================================

def _get_component_type(name_or_type: str) -> str:
    """从组分名称或类型提取类别前缀（如 'IL-1' -> 'IL'）"""
    name_upper = name_or_type.upper()
    for prefix in REPLACEMENT_POOL.keys():
        if name_upper.startswith(prefix.upper()):
            return prefix
    return name_or_type  # 回退：直接用原始值


def augment_formulation(
    formulation: Dict[str, Any],
    aug_prob: float = 0.5,
    random_state: Optional[int] = None,
    replacement_pool: Optional[Dict[str, List[str]]] = None,
    ratio_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """结构化增强主函数：输入配方 dict，输出增强后配方 + 操作日志。

    兼容两种数据格式：
      - 标准格式: formulation["component_list"][i]["component_type"] + "molar_percentage"
      - COMET格式: formulation["components"][i]["name"] + "percent"

    三类操作：
      1. 同类别脂质替换：在相同功能类别内替换分子
      2. 比例范围约束扰动：对摩尔百分比施加有界扰动
      3. 组分 Dropout：随机移除非 PEG 组分并重新归一化

    操作概率由文件顶部配置参数控制：P_REPLACE_NON_PEG, P_REPLACE_PEG,
    P_PERTURB_RATIO, P_COMPONENT_DROPOUT

    Args:
        formulation: 输入配方字典
        aug_prob: 应用增强的总概率（0~1）
        random_state: 随机种子（None 表示不固定）
        replacement_pool: 自定义替换池（None 使用默认 REPLACEMENT_POOL）
        ratio_bounds: 自定义比例约束（None 使用默认 RATIO_BOUNDS）

    Returns:
        (aug_formulation, aug_log): 增强后配方 + 操作日志字典
    """
    pool = replacement_pool if replacement_pool is not None else REPLACEMENT_POOL
    bounds = ratio_bounds if ratio_bounds is not None else RATIO_BOUNDS

    # 设置随机种子
    if random_state is not None:
        random.seed(random_state)
        np.random.seed(random_state)

    aug_log: Dict[str, Any] = {"applied": False, "operations": []}

    # 以 (1 - aug_prob) 的概率跳过增强
    if random.random() > aug_prob:
        return copy.deepcopy(formulation), aug_log

    aug = copy.deepcopy(formulation)

    # 检测数据格式：支持 "component_list"（标准）和 "components"（COMET）
    if "component_list" in aug:
        comps = aug["component_list"]
        pct_key = "molar_percentage"
        type_key = "component_type"
    elif "components" in aug:
        comps = aug["components"]
        pct_key = "percent"
        type_key = "name"  # COMET 中用 name（如 "IL-1"）存储类别信息
    else:
        return aug, aug_log

    if not comps:
        return aug, aug_log

    aug_log["applied"] = True

    # ====================================================================
    # 操作1: 同类别脂质替换
    # 概率配置: P_REPLACE_NON_PEG (IL/HL/sterol), P_REPLACE_PEG (PEG)
    # ====================================================================
    for c in comps:
        ctype = _get_component_type(c.get(type_key, ""))
        if ctype not in pool:
            continue
        p_replace = P_REPLACE_PEG if ctype == "PEG" else P_REPLACE_NON_PEG
        if random.random() < p_replace:
            curr_smi = c.get("smi", "")
            # 排除当前分子后选择候选
            candidates = [s for s in pool[ctype] if s != curr_smi]
            if candidates:
                new_smi = random.choice(candidates)
                aug_log["operations"].append({
                    "op": "replace",
                    "type": ctype,
                    "old_smi": curr_smi[:30] if curr_smi else "",
                    "new_smi": new_smi[:30],
                })
                c["smi"] = new_smi

    # ====================================================================
    # 操作2: 比例范围约束扰动
    # 概率配置: P_PERTURB_RATIO
    # ====================================================================
    if random.random() < P_PERTURB_RATIO and comps:
        ti = random.randint(0, len(comps) - 1)
        target_type = _get_component_type(comps[ti].get(type_key, ""))
        if target_type in bounds:
            lo, hi = bounds[target_type]
            old_val = comps[ti].get(pct_key, 0.0)
            # 高斯扰动：std = 5% 原值，截断到合法范围
            new_val = float(np.clip(np.random.normal(old_val, 0.05 * old_val), lo, hi))
            delta = new_val - old_val
            comps[ti][pct_key] = new_val

            # 将变化量按权重分摊给其他组分
            others = [i for i in range(len(comps)) if i != ti]
            other_vals = np.array([comps[i].get(pct_key, 0.0) for i in others])
            if other_vals.sum() > 0:
                weights = other_vals / other_vals.sum()
                for idx, w in zip(others, weights):
                    comps[idx][pct_key] = max(
                        0.5, comps[idx][pct_key] - delta * w
                    )
            aug_log["operations"].append({
                "op": "perturb_ratio",
                "type": target_type,
                "old": round(old_val, 2),
                "new": round(new_val, 2),
            })

    # ====================================================================
    # 操作3: 组分 Dropout（仅非 PEG 组分）
    # 概率配置: P_COMPONENT_DROPOUT
    # ====================================================================
    if random.random() < P_COMPONENT_DROPOUT:
        # 选择可 dropout 的候选（非 PEG 且百分比 > 0）
        dr = [
            i for i, c in enumerate(comps)
            if _get_component_type(c.get(type_key, "")) != "PEG" and c.get(pct_key, 0) > 0
        ]
        if len(dr) >= 2:
            di = random.choice(dr)
            dropped_val = comps[di].get(pct_key, 0.0)
            others = [i for i in range(len(comps)) if i != di]
            # 将被移除组分的量平均分配给其他组分
            for idx in others:
                comps[idx][pct_key] += dropped_val / len(others)
            comps[di][pct_key] = 0.0
            aug_log["operations"].append({
                "op": "dropout",
                "type": _get_component_type(comps[di].get(type_key, "")),
                "dropped_value": round(dropped_val, 2),
            })

    # ====================================================================
    # 归一化：确保所有组分百分比总和为 100%
    # ====================================================================
    total = sum(c.get(pct_key, 0.0) for c in comps)
    if total > 0 and abs(total - 100.0) > 1e-3:
        for c in comps:
            c[pct_key] = round(c[pct_key] / total * 100.0, 2)

    # 最终修正：将所有组分的百分比总和误差分配给最大组分
    final_total = sum(c.get(pct_key, 0.0) for c in comps)
    if comps and abs(final_total - 100.0) > 1e-3:
        max_idx = max(range(len(comps)), key=lambda i: comps[i].get(pct_key, 0))
        comps[max_idx][pct_key] += round(100.0 - final_total, 2)

    return aug, aug_log


# ============================================================================
# 5. 课程式概率调度器
# ============================================================================

class AugmentationScheduler:
    """课程式增强概率调度器。

    训练初期以低增强概率让模型学习真实分布基础结构，
    中后期逐步提升迫使模型对变异配方保持鲁棒，
    后期逐渐降低以避免分布偏移干扰收敛。

    三段式概率调度：
      - 初期 (0~33% epochs): 概率 0.3
      - 中期 (33%~66% epochs): 概率 0.5
      - 后期 (66%~100% epochs): 概率 0.2
    """

    def __init__(self, max_epoch: int = 200):
        """
        Args:
            max_epoch: 训练总 epoch 数（用于计算比例）
        """
        self.max_epoch = max_epoch
        self.bounds = [0.33, 0.66]  # 分界点比例
        self.probs = [0.3, 0.5, 0.2]  # 三段概率
        self._current_epoch = 0

    def set_epoch(self, epoch: int):
        """设置当前 epoch（通常在每轮训练开始时调用）。"""
        self._current_epoch = epoch

    def get_prob(self, epoch: Optional[int] = None) -> float:
        """获取当前 epoch 的增强概率。

        Args:
            epoch: 指定 epoch（None 使用内部缓存的当前 epoch）

        Returns:
            增强概率值 (0.0 ~ 1.0)
        """
        if epoch is None:
            epoch = self._current_epoch
        if self.max_epoch <= 0:
            return self.probs[0]
        p = epoch / self.max_epoch
        if p < self.bounds[0]:
            return self.probs[0]
        elif p < self.bounds[1]:
            return self.probs[1]
        else:
            return self.probs[2]

    def __repr__(self) -> str:
        return (
            f"AugmentationScheduler(max_epoch={self.max_epoch}, "
            f"bounds={self.bounds}, probs={self.probs})"
        )


# ============================================================================
# 6. 配方增强包装器（用于集成到数据加载流程）
# ============================================================================

class FormulationAugmenter:
    """配方增强包装器，封装增强逻辑和调度器。

    用法：
        augmenter = FormulationAugmenter(max_epoch=200, use_scheduler=True)
        augmenter.set_epoch(10)
        aug_formulation, log = augmenter.apply(formulation)
    """

    def __init__(
        self,
        max_epoch: int = 200,
        use_scheduler: bool = True,
        static_prob: float = 0.5,
        replacement_pool: Optional[Dict[str, List[str]]] = None,
        ratio_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    ):
        """
        Args:
            max_epoch: 训练总 epoch 数
            use_scheduler: 是否使用课程式调度（False 则使用静态概率）
            static_prob: 静态增强概率（use_scheduler=False 时使用）
            replacement_pool: 自定义替换池
            ratio_bounds: 自定义比例约束
        """
        self.use_scheduler = use_scheduler
        self.static_prob = static_prob
        self.scheduler = AugmentationScheduler(max_epoch=max_epoch) if use_scheduler else None
        self.replacement_pool = replacement_pool
        self.ratio_bounds = ratio_bounds

    def set_epoch(self, epoch: int):
        """设置当前 epoch（用于课程式调度）。"""
        if self.scheduler is not None:
            self.scheduler.set_epoch(epoch)

    def get_current_prob(self) -> float:
        """获取当前增强概率。"""
        if self.use_scheduler and self.scheduler is not None:
            return self.scheduler.get_prob()
        return self.static_prob

    def apply(
        self, formulation: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """对配方应用增强。

        Args:
            formulation: 输入配方字典

        Returns:
            (aug_formulation, aug_log)
        """
        prob = self.get_current_prob()
        return augment_formulation(
            formulation,
            aug_prob=prob,
            replacement_pool=self.replacement_pool,
            ratio_bounds=self.ratio_bounds,
        )

    def __repr__(self) -> str:
        return (
            f"FormulationAugmenter(use_scheduler={self.use_scheduler}, "
            f"static_prob={self.static_prob}, scheduler={self.scheduler})"
        )
