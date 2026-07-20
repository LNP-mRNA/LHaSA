# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import os
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from unimol.core import metrics
from unimol.core.losses import UnicoreLoss, register_loss
from scipy.stats import (
    spearmanr,
    pearsonr
)

from scipy.optimize import minimize, Bounds, minimize_scalar


def _read_lambdarank_config():
    """从环境变量读取 LambdaRank 配置。使用环境变量而非命令行参数，
    因为 Uni-Core 框架的 parse_args_and_arch() 会拦截不认识的参数导致报错。
    环境变量在 Python 子进程中自动继承，无需通过命令行传递。
    """
    use_lambdarank = os.environ.get('COMET_USE_LAMBDARANK', '0') == '1'
    alpha = float(os.environ.get('COMET_LAMBDARANK_ALPHA', '0.7'))
    k = int(os.environ.get('COMET_LAMBDARANK_K', '10'))
    env_val = os.environ.get('COMET_USE_LAMBDARANK', 'NOT_SET')
    print(f"[LAMBDARANK] _read_lambdarank_config: COMET_USE_LAMBDARANK={env_val}, use_lambdarank={use_lambdarank}, alpha={alpha}, k={k}")
    return {
        'use_lambdarank': use_lambdarank,
        'lambdarank_alpha': alpha,
        'lambdarank_k': k,
    }


def lambdarank_loss(pred_scores: torch.Tensor,
                    true_labels: torch.Tensor,
                    k: int = 10,
                    sigma: float = 1.0) -> torch.Tensor:
    """
    LambdaRank Loss的向量化实现。
    基于NDCG变化量(delta_NDCG)对Pairwise BCE Loss进行加权，
    使得模型更关注对排序结果影响更大的文档对。

    参数:
        pred_scores: [batch_size] 模型输出的预测分数
        true_labels: [batch_size] 归一化到[0,1]的真实标签
        k: NDCG截断位置，默认10
        sigma: Sigmoid温度系数，控制梯度锐度
    返回:
        loss: 标量损失值
    """
    batch_size = pred_scores.size(0)
    device = pred_scores.device
    if batch_size < 2:
        return torch.tensor(0.0, device=device, requires_grad=True)

    # 构造上三角矩阵索引，取所有无序对(i,j)且i<j
    row_idx, col_idx = torch.meshgrid(
        torch.arange(batch_size, device=device),
        torch.arange(batch_size, device=device),
        indexing='ij'
    )
    mask = row_idx < col_idx

    # 提取满足i<j条件的预测分数和真实标签
    s_i = pred_scores[row_idx[mask]]
    s_j = pred_scores[col_idx[mask]]
    y_i = true_labels[row_idx[mask]]
    y_j = true_labels[col_idx[mask]]

    # 计算预测分数差异并通过Sigmoid得到排序概率
    score_diff = sigma * (s_i - s_j)
    prob_ij = torch.sigmoid(score_diff)
    # S_ij: 真实标签的偏序关系，y_i>y_j时为1，否则为0
    S_ij = (y_i > y_j).float()

    # 计算每个样本的当前排序位次（1-based）
    _, sorted_indices = torch.sort(pred_scores, descending=True)
    ranks = torch.empty_like(sorted_indices, dtype=torch.float)
    ranks[sorted_indices] = torch.arange(batch_size, device=device, dtype=torch.float)
    rank_i = ranks[row_idx[mask]] + 1
    rank_j = ranks[col_idx[mask]] + 1

    # 计算NDCG增益（gain = 2^relevance - 1）
    gains = torch.pow(2.0, true_labels) - 1.0
    gain_i = gains[row_idx[mask]]
    gain_j = gains[col_idx[mask]]

    # 计算折扣因子（discount = 1/log2(rank+1)）
    discount_i = 1.0 / torch.log2(rank_i + 1.0)
    discount_j = 1.0 / torch.log2(rank_j + 1.0)

    # 计算交换位置带来的NDCG变化量（绝对值）
    # |gain_i * d_j + gain_j * d_i - gain_i * d_i - gain_j * d_j|
    delta_ndcg = torch.abs(
        gain_i * discount_j + gain_j * discount_i
        - gain_i * discount_i - gain_j * discount_j
    )
    # 数值稳定性：裁剪到合理范围防止极端值
    delta_ndcg = torch.clamp(delta_ndcg, min=1e-8, max=10.0)

    # 计算Pairwise BCE Loss
    bce = -(S_ij * torch.log(prob_ij + 1e-8)
            + (1.0 - S_ij) * torch.log(1.0 - prob_ij + 1e-8))
    # 用delta_NDCG加权，重要的排序对贡献更大
    weighted_bce = bce * delta_ndcg
    loss = weighted_bce.mean()
    return loss


def compute_contrastive_loss(contrast_pred, contrast_targets, min_loss_sample=3, mask_similar_contrast_label=True, contrast_margin_coeff=0.0):
    
    if min_loss_sample > contrast_targets.shape[0]:
        raise AssertionError("contrast_targets num should not be smaller than min_loss_sample")

    # Process loss mask:
    predicts = contrast_pred.view(-1, 1).float()
    sample_mask = ~contrast_targets.isnan()

    selected_predicts = predicts.masked_select(sample_mask)
    selected_contrast_targets = contrast_targets.masked_select(sample_mask)

    final_predicts = selected_predicts.view(-1, 1)
    final_contrast_targets = selected_contrast_targets.view(-1, 1)

    targets_diff = final_contrast_targets-final_contrast_targets.transpose(1,0)
    contrast_labels = torch.sign(targets_diff)*0.5 + 0.5
    value_pred_diff = final_predicts-final_predicts.transpose(1,0)

    # add label margin to the contrastive loss
    if contrast_margin_coeff == 0:
        contrastive_preds = F.logsigmoid(value_pred_diff)   
        inverse_preds = F.logsigmoid(-1*value_pred_diff)
    else:
        value_pred_diff_with_margin = value_pred_diff - contrast_margin_coeff*targets_diff
        contrastive_preds = F.logsigmoid(value_pred_diff_with_margin)
        inverse_preds = F.logsigmoid(-1*value_pred_diff_with_margin)


    losses = -contrast_labels*contrastive_preds - (1-contrast_labels)*inverse_preds
    similar_label_mask = (contrast_labels != 0.5).float()
    self_mask = 1-torch.eye(losses.shape[0],device=losses.device)
    if mask_similar_contrast_label:
        # print("similar_label_mask: ", similar_label_mask)
        loss_mask = similar_label_mask
    else:
        loss_mask = self_mask
    
    contrastive_pred_loss = torch.sum(losses*loss_mask)/torch.sum(loss_mask)
    
    return contrastive_pred_loss


def compute_reg_metrics(predicts, targets, groups, split="valid", task_name=None, sample_size=None, metric_dict=None, contrast_margin_coeff=0.0, infer=False, use_lambdarank=False, lambdarank_k=10):
    if metric_dict is None:
        metric_dict = {}

    # print("compute_reg_metrics, task_name: ", task_name, split, "predicts: ", predicts)
    # print("compute_reg_metrics, task_name: ", task_name, split, "targets: ", targets)

    metric_dict[f"{task_name}{split}_predict"] = predicts.view(-1).cpu()
    metric_dict[f"{task_name}{split}_target"] = targets.view(-1).cpu()
    metric_dict[f"{task_name}{split}_groups"] = groups

    if infer:
        return metric_dict

    # print("compute_reg_metrics, task_name: ", task_name, "predicts.shape: ", predicts.shape, "predicts: ", predicts)
    # print("compute_reg_metrics, task_name: ", task_name, "targets.shape: ", targets.shape, "targets: ", targets)
    # print("compute_reg_metrics, task_name: ", task_name, "groups len: ", len(groups), "groups: ", groups)
    df_w_nan = pd.DataFrame(
        {
            "predict": predicts.view(-1).cpu(),
            "target": targets.view(-1).cpu(),
            # "groups": groups,
            # "smi": smi_list,
        }
    )
    # drop samples where target is nan
    # print("len(df_w_nan): ", len(df_w_nan))
    df = df_w_nan.dropna()
    # print("len(df): ", len(df))
    mae = np.abs(df["predict"] - df["target"]).mean()
    mse = ((df["predict"] - df["target"]) ** 2).mean()
    # print("len(df['target]): ", len(df["target"]))
    # print("len(df['predict]): ", len(df["predict"]))
    # print("compute_reg_metrics, task_name: ", task_name, "task_logits: ", df["predict"])
    # print("compute_reg_metrics, task_name: ", task_name, "task_target: ", df["target"])
    
    # check if there is any value in df["target"]
    if len(df["target"]) == 0 and len(df["target"]) == 0:
        print("No valid target and predict values for task_name: ", task_name)
        return metric_dict
    spearmanr_coeff, p_val = spearmanr(df["target"], df["predict"])
    pearsonr_coeff, p_val_pearsonr = pearsonr(df["target"], df["predict"])

    # top-50% accuracy, which measures how accurately the model can predict the top-50% lnps vs the remaining 50% lnps, split according to the lnps' target values
    # add metric_dict[f"{task_name}{split}_top50pctACC"] = top50pctACC
    # print(df["predict"].median(), "df[predict]: ", df["predict"])
    # print(df["target"].median(), "df[target]: ", df["target"])
    # print("df[[target, predict]]: ", df[["target", "predict"]])
    target_predict_df = df[["target", "predict"]]
    target_predict_df['target_top50pct'] = target_predict_df['target'] > target_predict_df['target'].median()
    target_predict_df['predict_top50pct'] = target_predict_df['predict'] > target_predict_df['predict'].median()
    # print(target_predict_df['target_top50pct'].sum(), "df[target_top50pct]: ", target_predict_df['target_top50pct'])
    # print(target_predict_df['predict_top50pct'].sum(), "df[predict_top50pct]: ", target_predict_df['predict_top50pct'])
    target_predict_in_top50pct = (target_predict_df['target_top50pct'] & target_predict_df['predict_top50pct']).sum()
    top50pct_accuracy = target_predict_in_top50pct / target_predict_df['target_top50pct'].sum()
    # print(top50pct_accuracy, "target_predict_in_top50pct: ", target_predict_in_top50pct, "target_predict_df['target_top50pct'].sum(): ", target_predict_df['target_top50pct'].sum())

    # print("spearmanr_coeff: ", spearmanr_coeff)
    # print("p_val: ", p_val)
    # df = df.groupby("groups").mean()
    # df = df.groupby("smi").mean()
    agg_mae = np.abs(df["predict"] - df["target"]).mean()
    agg_mse = ((df["predict"] - df["target"]) ** 2).mean()
    if task_name == None:
        task_name = ""
    else:
        task_name = task_name + "_"

    if sample_size == None:
        sample_size = (~targets.isnan()).sum()

    # Compute contrastive loss here to see if aligns with the loss value from compute loss function 
    contrastive_loss = compute_contrastive_loss(predicts, targets, contrast_margin_coeff=contrast_margin_coeff)

    metric_dict[f"{task_name}{split}_spearmanr_coeff"] = spearmanr_coeff
    metric_dict[f"{task_name}{split}_spearmanr_p_val"] = p_val
    metric_dict[f"{task_name}{split}_pearsonr_coeff"] = pearsonr_coeff
    metric_dict[f"{task_name}{split}_pearsonr_p_val"] = p_val_pearsonr
    metric_dict[f"{task_name}{split}_top50pct_accuracy"] = top50pct_accuracy
    metric_dict[f"{task_name}{split}_mae"] = mae
    metric_dict[f"{task_name}{split}_mse"] = mse
    metric_dict[f"{task_name}{split}_agg_mae"] = agg_mae
    metric_dict[f"{task_name}{split}_agg_mse"] = agg_mse
    metric_dict[f"{task_name}{split}_agg_rmse"] = np.sqrt(agg_mse)
    metric_dict[f"{task_name}{split}_contrastive_L"] = contrastive_loss.cpu()

    metrics.log_scalar(f"{task_name}{split}_spearmanr_coeff", spearmanr_coeff, sample_size, round=3)
    metrics.log_scalar(f"{task_name}{split}_spearmanr_p_val", p_val, sample_size, round=3)
    metrics.log_scalar(f"{task_name}{split}_pearsonr_coeff", pearsonr_coeff, sample_size, round=3)
    metrics.log_scalar(f"{task_name}{split}_pearsonr_p_val", p_val_pearsonr, sample_size, round=3)
    metrics.log_scalar(f"{task_name}{split}_mae", mae, sample_size, round=3)
    metrics.log_scalar(f"{task_name}{split}_mse", mse, sample_size, round=3)
    metrics.log_scalar(f"{task_name}{split}_agg_mae", agg_mae, sample_size, round=3)
    metrics.log_scalar(f"{task_name}{split}_agg_mse", agg_mse, sample_size, round=3)
    metrics.log_scalar(f"{task_name}{split}_contrastive_L", contrastive_loss.cpu(), sample_size, round=3)
    metrics.log_scalar(
        f"{task_name}{split}_agg_rmse", np.sqrt(agg_mse), sample_size, round=4
    )

    # 当启用LambdaRank时，独立计算并记录LambdaRank Loss指标
    if use_lambdarank:
        valid_predicts = predicts[~targets.isnan()].view(-1).float()
        valid_targets = targets[~targets.isnan()].view(-1).float()
        if valid_targets.shape[0] >= 2:
            y_min = valid_targets.min()
            y_max = valid_targets.max()
            y_range = y_max - y_min
            if y_range > 1e-8:
                # 对标签进行min-max归一化到[0,1]，满足LambdaRank输入要求
                normalized_labels = (valid_targets - y_min) / y_range
                normalized_labels = torch.clamp(normalized_labels, 0.0, 1.0)
                loss_lambdarank = lambdarank_loss(
                    valid_predicts,
                    normalized_labels,
                    k=lambdarank_k
                )
                metric_dict[f"{task_name}{split}_lambdarank_L"] = loss_lambdarank.cpu()
                metrics.log_scalar(
                    f"{task_name}{split}_lambdarank_L", loss_lambdarank.cpu(), sample_size, round=3
                )
    
    return metric_dict


def get_depth(dictionary):
    """
    Recursively get the maximum depth of a dictionary
    """
    if not isinstance(dictionary, dict) or not dictionary:
        # If the input is not a dictionary or is an empty dictionary, return 0
        return 0
    else:
        # Otherwise, recursively call the function on each value and add 1 to the maximum depth
        return 1 + max(get_depth(dictionary[key]) for key in dictionary)

@register_loss("np_finetune_contrastive")
class NPFinetuneContrastiveLoss(UnicoreLoss):
    def __init__(self, task):
        super().__init__(task)

    def forward(self, model, sample, reduce=True, target_subclass="finetune_target", task_schema=None, loss_sample_dropout=0, multitask_reg=False, cagrad_c=0.5, contrast_margin_coeff=0.0, infer=False, output_cls_rep=False, **kwargs):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        # ===== 🆕 从环境变量读取 LambdaRank 配置 =====
        # 使用环境变量而非命令行参数/self.args，因为 Uni-Core 框架
        # 的 parse_args_and_arch() 会拦截不认识参数导致 "unrecognized arguments" 报错
        _lrcfg = _read_lambdarank_config()
        use_lambdarank = _lrcfg['use_lambdarank']
        lambdarank_alpha = _lrcfg['lambdarank_alpha']
        lambdarank_k = _lrcfg['lambdarank_k']
        # ===== LambdaRank 配置读取结束 =====

        # print("NPFinetuneContrastiveLoss, task_schema: ", task_schema)
        net_output = model(
            sample,
            # **sample["net_input"],
            features_only=True,
            np_classification_head_name=self.args.classification_head_name, # regression_head_name?
        )
        # net_output = model(
        #     **sample["net_input"],
        #     features_only=True,
        #     classification_head_name=self.args.classification_head_name,
        # )
        reg_output = net_output[0] # self.args.num_classes-dimensional reg_output
        # print("reg_output: ", reg_output)

        # compute dictionary of loss values for diff target_subclass(es)
        # final output loss would be sum of target_subclass losses multiplied by their weights
        if type(task_schema) == dict: 
            # print("if type(task_schema) == dict:")
            loss, loss_dict = self.compute_loss_with_schema(model, reg_output, sample, reduce=reduce, task_schema=task_schema, loss_sample_dropout=loss_sample_dropout, multitask_reg=multitask_reg, cagrad_c=cagrad_c, contrast_margin_coeff=contrast_margin_coeff, use_lambdarank=use_lambdarank, lambdarank_alpha=lambdarank_alpha, lambdarank_k=lambdarank_k)
            # print("loss_dict: ", loss_dict)
            tasks_log_dict = self.get_tasks_log_output(reg_output, sample, loss_dict, task_schema, target_subclass, infer=infer)
            # print("tasks_log_dict: ", tasks_log_dict)
        else:
            # print("A loss = self.compute_loss(model, reg_output, sample, reduce=reduce)")
            loss = self.compute_loss(model, reg_output, sample, reduce=reduce, loss_sample_dropout=loss_sample_dropout, contrast_margin_coeff=contrast_margin_coeff, use_lambdarank=use_lambdarank, lambdarank_alpha=lambdarank_alpha, lambdarank_k=lambdarank_k)
        
        # sample_size = sample["target"][target_subclass].size(0)

        if not self.training:
            if output_cls_rep:
                # get LNP representations from here
                cls_representations = net_output[1] 
                # print("cls_representations: ", cls_representations)
                # print("cls_representations in_house_lnp_B16F10_luc shape: ", cls_representations['in_house_lnp_B16F10_luc'].shape)
                # print("cls_representations in_house_lnp_DC24_luc shape: ", cls_representations['in_house_lnp_DC24_luc'].shape)
                cls_rep_log_dict = {}
                if type(cls_representations) == dict:
                    for cls_rep_name in cls_representations:
                        cls_rep_log_dict["cls_" + cls_rep_name] = cls_representations[cls_rep_name].cpu()
                else:
                    cls_rep_log_dict["cls_representations"] = cls_representations.cpu()
            else:
                cls_rep_log_dict = {}
                
            if type(task_schema) == dict: 
                # print("contrastive_loss type(task_schema) == dict")
                for task_ind, task_name in enumerate(task_schema):
                    if task_ind == 0:
                        sample_size = sample["target"][target_subclass][task_name].size(0)
                        # print("sample_size A: ", sample_size)
                        break
                
                # print("sample[net_input][lnp_ids]: ", sample["net_input"]["lnp_ids"])
                logging_output = {
                    "loss": loss.data,
                    "components": sample["net_input"]["components"],
                    # "smi_name": sample["smi_name"],
                    "sample_size": sample_size,
                    "num_task": len(loss_dict),
                    "conf_size": self.args.conf_size,
                    "bsz": sample_size,
                    **tasks_log_dict,
                    **cls_rep_log_dict
                }
            else:
                # print("contrastive_loss type(task_schema) != dict")
                if self.task.mean and self.task.std:
                    targets_mean = torch.tensor(self.task.mean, device=reg_output.device)
                    targets_std = torch.tensor(self.task.std, device=reg_output.device)
                    reg_output = reg_output * targets_std + targets_mean

                sample_size = sample["target"][target_subclass].size(0)
                logging_output = {
                    "loss": loss.data,
                    "predict": reg_output.view(-1, self.args.num_classes).data,
                    "target": sample["target"][target_subclass]
                    .view(-1, self.args.num_classes)
                    .data,
                    "components": sample["net_input"]["components"],
                    # "smi_name": sample["smi_name"],
                    "sample_size": sample_size,
                    "num_task": self.args.num_classes,
                    "conf_size": self.args.conf_size,
                    "bsz": sample["target"][target_subclass].size(0),
                    **cls_rep_log_dict
                }
        else:
            if type(task_schema) == dict:
                for task_ind, task_name in enumerate(task_schema):
                    if task_ind == 0:
                        sample_size = sample["target"][target_subclass][task_name].size(0)
                        break
                # print("sample_size A: ", sample_size)
                # print("tasks_log_dict: ", tasks_log_dict)
                logging_output = {
                    "loss": loss.data,
                    "sample_size": sample_size,
                    "bsz": sample_size,
                    **tasks_log_dict
                }
            else:
                sample_size = sample["target"][target_subclass].size(0)
                logging_output = {
                    "loss": loss.data,
                    "sample_size": sample_size,
                    "bsz": sample["target"][target_subclass].size(0),
                }
            # ===== LambdaRank 训练日志增强 =====
            # 从loss张量的属性中提取各损失分量（在compute_loss中附加）
            if use_lambdarank and hasattr(loss, 'loss_pairwise'):
                logging_output["loss_pairwise"] = loss.loss_pairwise
                logging_output["loss_lambdarank"] = getattr(loss, 'loss_lambdarank', loss.data)
                logging_output["loss_total"] = loss.data

        # if debug:
        #     print("reg_output: ", reg_output)
        #     for output_name in reg_output:
        #         reg_output_grad = torch.autograd.grad(loss, reg_output[output_name], retain_graph=True, allow_unused=True)
        #         print(output_name, ", reg_output_grad: ", reg_output_grad)
        return loss, sample_size, logging_output


    def compute_loss(self, model, net_output, sample, contrast_targets=None, reduce=True, mask_similar_contrast_label=True, task_schema=None, loss_sample_dropout=0, min_loss_sample=3, contrast_margin_coeff=0.0, use_lambdarank=False, lambdarank_alpha=0.7, lambdarank_k=10):
        # print("contrast_margin_coeff: ", contrast_margin_coeff)
        # print("compute_loss, model.task_schema: ", model.task_schema)
        # print("compute_loss, sample[target]['finetune_target']: ", sample["target"]['finetune_target'])
        # print("compute_loss, contrast_targets: ", contrast_targets)
        # print("compute_loss, sample: ", sample)
        # print("compute_loss, net_output: ", net_output)
        
        if contrast_targets == None:
            # print("contrast_targets == None sample[target]['finetune_target']: ", sample["target"]['finetune_target'])
            contrast_targets = (
                sample["target"]['finetune_target'].view(-1, 1).float()
            )

        if min_loss_sample > contrast_targets.shape[0]:
            raise AssertionError("contrast_targets num should not be smaller than min_loss_sample")

        if loss_sample_dropout > 0:
            loss_sample_num = max(int(contrast_targets.view(-1).shape[0] * (1 - loss_sample_dropout)), min_loss_sample)
            # print("int(contrast_targets.view(-1).shape[0]: ", int(contrast_targets.view(-1).shape[0]))
            # print("loss_sample_num: ", loss_sample_num)

        # Process loss mask:
        predicts = net_output.view(-1, 1).float()
        sample_mask = ~contrast_targets.isnan()
        # print("sample_mask A: ", sample_mask)

        # Stochastic number of samples to use for loss:
        # if loss_sample_dropout > 0:
        #     valid_new_sample_mask = False
        #     while not valid_new_sample_mask:
        #         loss_sample_dropout_mask = torch.rand_like(contrast_targets).ge(loss_sample_dropout)
        #         new_sample_mask = sample_mask & loss_sample_dropout_mask
        #         if new_sample_mask.sum() > 1:
        #             valid_new_sample_mask = True
        #     sample_mask = new_sample_mask
        # print("sample_mask B: ", sample_mask)

        # print("predicts: ", predicts)
        # print("predicts.shape: ", predicts.shape)
        # print("contrast_targets: ", contrast_targets)
        # print("contrast_targets.shape: ", contrast_targets.shape)

        # print(loss_sample_dropout > 0, "predicts: ", predicts)
        selected_predicts = predicts.masked_select(sample_mask)
        # final_predicts = predicts.masked_select(sample_mask).view(-1, 1)
        selected_contrast_targets = contrast_targets.masked_select(sample_mask)

        if loss_sample_dropout > 0:
            shuffled_ind = torch.randperm(selected_predicts.shape[0])
            shuffled_predicts = selected_predicts[shuffled_ind]
            shuffled_contrast_targets = selected_contrast_targets[shuffled_ind]

            final_predicts = shuffled_predicts[:loss_sample_num].view(-1, 1)
            final_contrast_targets = shuffled_contrast_targets[:loss_sample_num].view(-1, 1)
            # print("loss_sample_num A: ", loss_sample_num)
            # print("final_predicts.shape A: ", final_predicts.shape)
            # print("final_contrast_targets.shape A: ", final_contrast_targets.shape)
        else:
            final_predicts = selected_predicts.view(-1, 1)
            final_contrast_targets = selected_contrast_targets.view(-1, 1)
        # final_contrast_targets = contrast_targets.masked_select(sample_mask).view(-1, 1)
        # if loss_sample_dropout > 0:


        # print("final_predicts: ", final_predicts)
        # print("final_predicts.shape: ", final_predicts.shape)
        # print("final_contrast_targets: ", final_contrast_targets)
        # print("final_contrast_targets.shape: ", final_contrast_targets.shape)
        # nan_loss = 1 - targets_diff.isnan().float()

        # print(loss_sample_dropout > 0, "final_predicts: ", final_predicts)
        # print(loss_sample_dropout > 0, "final_contrast_targets: ", final_contrast_targets)
        targets_diff = final_contrast_targets-final_contrast_targets.transpose(1,0)
        contrast_labels = torch.sign(targets_diff)*0.5 + 0.5
        # print(loss_sample_dropout > 0, "contrast_labels: ", contrast_labels)
        # print("net_output.shape: ", net_output.shape)
        # print("predicts.shape: ", predicts.shape)
        value_pred_diff = final_predicts-final_predicts.transpose(1,0)
        # print(loss_sample_dropout > 0, "value_pred_diff: ", value_pred_diff)


        # add label margin to the contrastive loss
        if contrast_margin_coeff == 0:
            contrastive_preds = F.logsigmoid(value_pred_diff)   
            inverse_preds = F.logsigmoid(-1*value_pred_diff)
            # print("value_pred_diff A: ", value_pred_diff)
            # print("targets_diff A: ", targets_diff)
            # print("contrast_labels A: ", contrast_labels)
            # print("contrastive_preds A: ", contrastive_preds)
        else:
            value_pred_diff_with_margin = value_pred_diff - contrast_margin_coeff*targets_diff
            contrastive_preds = F.logsigmoid(value_pred_diff_with_margin) 
            inverse_preds = F.logsigmoid(-1*value_pred_diff_with_margin)
            # print("final_contrast_targets B: ", final_contrast_targets)
            # print("value_pred_diff B: ", value_pred_diff)
            # print("targets_diff B: ", targets_diff)
            # print("value_pred_diff_with_margin B: ", value_pred_diff_with_margin)
            # print("contrast_labels B: ", contrast_labels)
            # print("contrastive_preds B: ", contrastive_preds)
            
        # contrastive_preds = F.logsigmoid(value_pred_diff)   
        # # print(loss_sample_dropout > 0, "contrastive_preds: ", contrastive_preds) # very similar values: ~ -0.6933
        # inverse_preds = F.logsigmoid(-1*value_pred_diff)   
        # print(loss_sample_dropout > 0, "inverse_preds: ", inverse_preds)
        # print("contrast_labels.shape: ", contrast_labels.shape)
        # print("contrastive_preds.shape: ", contrastive_preds.shape)
        # print("inverse_preds.shape: ", inverse_preds.shape)
        losses = -contrast_labels*contrastive_preds - (1-contrast_labels)*inverse_preds
        # print(loss_sample_dropout > 0, "-contrast_labels*contrastive_preds: ", -contrast_labels*contrastive_preds)
        # print(loss_sample_dropout > 0, "- (1-contrast_labels)*inverse_preds: ", - (1-contrast_labels)*inverse_preds)
        # print(loss_sample_dropout > 0, "losses: ", losses)
        similar_label_mask = (contrast_labels != 0.5).float()
        self_mask = 1-torch.eye(losses.shape[0],device=losses.device)
        # print(loss_sample_dropout > 0, "self_mask: ", self_mask)
        if mask_similar_contrast_label:
            # print("similar_label_mask: ", similar_label_mask)
            loss_mask = similar_label_mask
        else:
            loss_mask = self_mask

        # # TODO now Check! Remove loss computation for labels with nan value
        # nan_loss = 1 - targets_diff.isnan().float()
        # print("nan_loss: ", nan_loss)
        # print("(nan_loss == 0).any(): ", (nan_loss == 0).any())
        # loss_mask = loss_mask * nan_loss
        # print("loss_mask: ", loss_mask)
        
        contrastive_pred_loss = torch.sum(losses*loss_mask)/torch.sum(loss_mask)
        # print(loss_sample_dropout > 0, "contrastive_pred_loss: ", contrastive_pred_loss)
        
        # print("GRAD Checkpoint D", torch.autograd.grad(contrastive_pred_loss, predicts, retain_graph=True, allow_unused=True))

        # 原有Pairwise Loss计算完成，以下是LambdaRank混合损失分支
        if use_lambdarank and final_contrast_targets.shape[0] >= 2:
            # 对标签进行min-max归一化到[0,1]，满足LambdaRank输入要求
            y_min = final_contrast_targets.min()
            y_max = final_contrast_targets.max()
            y_range = y_max - y_min
            if y_range > 1e-8:
                normalized_labels = (final_contrast_targets.view(-1) - y_min) / y_range
                # 确保归一化后的标签严格在[0,1]范围内
                normalized_labels = torch.clamp(normalized_labels, 0.0, 1.0)
                # 计算LambdaRank Loss
                loss_lambdarank = lambdarank_loss(
                    final_predicts.view(-1),
                    normalized_labels,
                    k=lambdarank_k
                )
                # 70/30混合：默认lambdarank_alpha=0.7
                loss_total = lambdarank_alpha * loss_lambdarank + (1.0 - lambdarank_alpha) * contrastive_pred_loss
                # 将各损失分量附加到loss张量，供forward中的logging_output提取
                loss_total.loss_pairwise = contrastive_pred_loss.detach()
                loss_total.loss_lambdarank = loss_lambdarank.detach()
                return loss_total

        return contrastive_pred_loss

    def compute_cagrad_loss(self, model, final_loss_dict, task_schema, cagrad_c=0.5):
        def cagrad_exact_task_lambda(grad_vec, num_tasks):
            grads = grad_vec / 100.
            g0 = grads.mean(0)
            GG = grads.mm(grads.t())
            x_start = np.ones(num_tasks)/num_tasks
            bnds = tuple((0,1) for x in x_start)
            cons=({'type':'eq','fun':lambda x:1-sum(x)})
            A = GG.cpu().numpy()
            b = x_start.copy()
            c = (cagrad_c*g0.norm()).cpu().item()
            def objfn(x):
                return (x.reshape(1,num_tasks).dot(A).dot(b.reshape(num_tasks, 1)) + \
                        c * np.sqrt(x.reshape(1,num_tasks).dot(A).dot(x.reshape(num_tasks,1))+1e-8)).sum()
            res = minimize(objfn, x_start, bounds=bnds, constraints=cons)
            w_cpu = res.x
            ww= torch.Tensor(w_cpu).to(grad_vec.device)
            
            task_weights = ww.view(-1, 1)
            gw = (grads * ww.view(-1, 1)).sum(0)
            gw_norm = gw.norm() 
            lmbda = c / (gw_norm+1e-4)
            task_lambdas = lmbda * task_weights

            # gw = (grads * ww.view(-1, 1)).sum(0)
            # gw_norm = gw.norm()
            # lmbda = c / (gw_norm+1e-4)
            # g = (g0 + lmbda * gw) / (1 + lmbda)
            return task_lambdas

        parameters_with_grad = [param for param in model.parameters() if param.requires_grad]

        grad = []
        grad_task_names = []
        
        for task_ind, task_name in enumerate(task_schema):
            # print("parameters_with_grad: ", parameters_with_grad)
            # print(task_name, " task_name for grad compute, loss: ", final_loss_value_in_total_dict[task_name])
            # print(task_name, " loss requires_grad: ", final_loss_value_in_total_dict[task_name].requires_grad)
            # print("parameters_with_grad: ", parameters_with_grad)
            # print("parameters_with_grad[-1]: ", parameters_with_grad[-1])
            # print("parameters_with_grad len: ", len(parameters_with_grad))

            for param in parameters_with_grad:
                if not param.requires_grad:
                    print("Parameter does not require grad: ", param)

            # skip task if loss is not differentiable
            if not final_loss_dict[task_name].requires_grad:
                continue

            loss_grad = torch.autograd.grad(
                        final_loss_dict[task_name],
                        # model.parameters(),
                        parameters_with_grad,
                        retain_graph=True,
                        # retain_graph=(retain_graph or index != num_tasks - 1), # remove graph at the last task
                        allow_unused=True,
                    )
            
            grad.append(
                loss_grad
                # tuple(
                #     _grad.contiguous()
                #     for _grad in torch.autograd.grad(
                #         final_loss_value_in_total_dict[task_name],
                #         # model.parameters(),
                #         parameters_with_grad,
                #         retain_graph=True,
                #         # retain_graph=(retain_graph or index != num_tasks - 1), # remove graph at the last task
                #         allow_unused=True,
                #     )
                # )
            )
            grad_task_names.append(task_name)

        param_grad_list = []
        for param_ind, param_grad in enumerate(zip(*grad)):
            if None in param_grad: # check if there is None in any task's param_grad
                new_param_grad = list(param_grad)
                # print("new_param_grad A: ", new_param_grad)
                non_none_grad = None
                all_none_grad = False
                for task_ind, task_param_grad in enumerate(param_grad):
                    if task_param_grad != None:
                        non_none_grad = task_param_grad
                        break
                    elif task_ind == len(param_grad) - 1: # all and last task are None
                        all_none_grad = True
                
                # skip param if all tasks' param_grad is None
                if all_none_grad:
                    continue

                none_replacement = torch.zeros_like(non_none_grad)

                for i, task_param_grad in enumerate(param_grad):
                    if task_param_grad is None:
                        new_param_grad[i] = none_replacement.contiguous()
                    else:
                        new_param_grad[i] = task_param_grad.contiguous()

                param_grad_list.append(new_param_grad)
            else:
                param_grad_list.append(list(param_grad))

        grad = list(zip(*param_grad_list))  

        grad_vec = torch.cat(
            list(
                map(lambda x: torch.nn.utils.parameters_to_vector(x).unsqueeze(0), grad)
            ),
            dim=0,
        )  # num_tasks x dim
        
        cagrad_task_lambdas = cagrad_exact_task_lambda(grad_vec, len(grad_task_names))
        # print(grad_task_names, " cagrad_task_lambdas: ", cagrad_task_lambdas)

        cagrad_task_loss = 0
        for task_ind, task_lambda in enumerate(cagrad_task_lambdas):
            task_name = grad_task_names[task_ind]
            cagrad_task_loss = cagrad_task_loss + task_lambda * final_loss_dict[task_name]
    
        return cagrad_task_loss
    
    # A new version with multitasking function 
    def compute_loss_with_schema(self, model, net_output, sample, reduce=True, task_schema=None, loss_sample_dropout=0, min_loss_sample=3, multitask_reg=False, cagrad_c=0.5, contrast_margin_coeff=0.0, use_lambdarank=False, lambdarank_alpha=0.7, lambdarank_k=10):
        loss_dict = {}
        final_loss_value_in_total_dict = {}
        total_loss = 0

        # print("compute_loss_with_schema sample['target']['finetune_target']: ", sample["target"]['finetune_target'])
        # print("compute_loss_with_schema sample['target']['finetune_target'] keys: ", sample["target"]['finetune_target'].keys())
        # print("sample net_input dataset_name: ", sample["net_input"]["dataset_name"])
        # for multitasking, group target and model's output by dataset_name
        for task_ind, task_name in enumerate(task_schema):
            # print("compute_loss_with_schema, task_ind:", task_ind, "task_name: ", task_name)
            contrast_targets = (
                sample["target"]['finetune_target'][task_name].view(-1, self.args.num_classes).float()
            )

            # print("contrast_targets: ", contrast_targets)

            # net_output is a dict of net_output with key corresponding to task_name
            task_net_output = net_output[task_name]
            # print("compute_loss_with_schema, task_name: ", task_name, "task_logits: ", task_net_output)
            # print("task_net_output.shape: ", task_net_output.shape)
            # print(task_name, " contrast_targets.shape A: ", contrast_targets.shape, "contrast_targets: ", contrast_targets)

            # check if we need to get subset of preds and targets if loss compute mask is present
            task_loss_mask_name = task_name + "_mask"
            if task_loss_mask_name in sample["target"]['finetune_target']:
                task_loss_mask = sample["target"]['finetune_target'][task_loss_mask_name] # this is generated by MultiDatasetDictLabelDataset's collator function in unimol/core/data/raw_dataset.py
                # print(task_loss_mask_name, " task_loss_mask: ", task_loss_mask)
                task_net_output = task_net_output[task_loss_mask]
                contrast_targets = contrast_targets[task_loss_mask]
                # print(task_loss_mask_name, " task_loss_mask_name in sample, task_loss_mask: ", task_loss_mask)
                # print("task_net_output.shape: ", task_net_output.shape)
                # print("task_name, contrast_targets.shape B: ", contrast_targets.shape)
            
            # print("B pred_loss = self.compute_loss(model, task_net_output, sample, contrast_targets=contrast_targets, reduce=reduce), contrast_targets: ", contrast_targets)
            if min_loss_sample <= contrast_targets.shape[0]:
                # print("if min_loss_sample <= contrast_targets.shape[0]")
                # print("compute_loss_with_schema, contrast_margin_coeff: ", contrast_margin_coeff)
                # 传递LambdaRank参数到compute_loss，每个任务独立计算混合损失
                pred_loss = self.compute_loss(model, task_net_output, sample, contrast_targets=contrast_targets, reduce=reduce, loss_sample_dropout=loss_sample_dropout, min_loss_sample=min_loss_sample, contrast_margin_coeff=contrast_margin_coeff, use_lambdarank=use_lambdarank, lambdarank_alpha=lambdarank_alpha, lambdarank_k=lambdarank_k)
            else:
                # print("else min_loss_sample <= contrast_targets.shape[0]")
                pred_loss = torch.tensor(0, device=contrast_targets.device) # use 0 as a placeholder for loss value and skip task if contrast_targets num is smaller than min_loss_sample to prevent training instability

            loss_dict[task_name] = pred_loss

            task_loss_weight = task_schema[task_name]
            # print("task_name: ", task_name, " task_loss_weight: ", task_loss_weight, " pred_loss: ", pred_loss)
            if task_loss_weight == 0:
                # print("skipping task_name: ", task_name)
                continue
            
            # final loss value in total loss
            final_task_loss = pred_loss * task_loss_weight
            final_loss_value_in_total_dict[task_name] = final_task_loss
            total_loss = total_loss + final_task_loss
        
        # cagrad multitask grad reg to make loss convergence stable for all tasks
        # print("task_schema: ", task_schema)
        # print("final_loss_value_in_total_dict: ", final_loss_value_in_total_dict)
        if self.training and multitask_reg == True and cagrad_c > 0:
            cagrad_task_loss = self.compute_cagrad_loss(model, final_loss_value_in_total_dict, task_schema, cagrad_c)
            total_loss = total_loss + cagrad_task_loss
            loss_dict["cagrad"] = cagrad_task_loss

        # print("loss_dict: ", loss_dict)
        return total_loss, loss_dict

    def get_tasks_log_output(self, net_output, sample, loss_dict, task_schema, target_subclass="finetune_target", infer=False):
        # print("get_tasks_log_output, net_output: ", net_output)
        # tasks_log_dict = {}
        tasks_log_dict = {"lnp_ids": sample['net_input']['lnp_ids']}
        for task_ind, task_name in enumerate(task_schema):
            if task_ind == 0:
                sample_size = sample["target"][target_subclass][task_name].size(0)
            loss_name = "loss_{}".format(task_name)
            predict_name = "predict_{}".format(task_name)
            target_name = "target_{}".format(task_name)
            if task_name in loss_dict:
                loss_value = loss_dict[task_name].data
            else:
                loss_value = 0
            
            # Handle task label mask by removing task-masked samples from predict and target
            task_loss_mask_name = task_name + "_mask"
            if task_loss_mask_name in sample["target"]['finetune_target']:
                task_loss_mask = sample["target"]['finetune_target'][task_loss_mask_name] # this is generated by MultiDatasetDictLabelDataset's collator function in unimol/core/data/raw_dataset.py
                if infer:
                    task_loss_mask.fill_(True) # set all to True for inference

            # print("get_tasks_log_output, sample keys: ", sample.keys())
            # print("get_tasks_log_output, sample[net_input][lnp_ids]: ", sample['net_input']['lnp_ids'])
            # print("get_tasks_log_output, sample: ", sample)
            # print("get_tasks_log_output, task_loss_mask: ", task_loss_mask)
            # print("get_tasks_log_output, net_output[task_name][task_loss_mask].view(-1, self.args.num_classes).data: ", net_output[task_name][task_loss_mask].view(-1, self.args.num_classes).data)
            
            task_log_dict = {
                loss_name: loss_value,
                predict_name: net_output[task_name][task_loss_mask].view(-1, self.args.num_classes).data,
                target_name: sample["target"][target_subclass][task_name]
                .view(-1, self.args.num_classes)
                .data[task_loss_mask],
            }
            # print("get_tasks_log_output, predict_name: ", predict_name, "task_log_dict: ", task_log_dict[predict_name])
            tasks_log_dict = {**tasks_log_dict, **task_log_dict}

        # print("get_tasks_log_output, tasks_log_dict lnp_ids: ", tasks_log_dict["lnp_ids"])
        return tasks_log_dict
    # def compute_loss_old(self, model, net_output, sample, reduce=True):
    #     predicts = net_output.view(-1, self.args.num_classes).float()
    #     targets = (
    #         sample["target"]["finetune_target"].view(-1, self.args.num_classes).float()
    #     )
    #     if self.task.mean and self.task.std:
    #         targets_mean = torch.tensor(self.task.mean, device=targets.device)
    #         targets_std = torch.tensor(self.task.std, device=targets.device)
    #         targets = (targets - targets_mean) / targets_std
    #     loss = F.mse_loss(
    #         predicts,
    #         targets,
    #         reduction="sum" if reduce else "none",
    #     )
    #     return loss
   

    @staticmethod
    def reduce_metrics(logging_outputs, split="valid", contrast_margin_coeff=0.0, infer=False, use_lambdarank=False, lambdarank_k=10) -> None:
        """Aggregate logging outputs from data parallel training."""
        # print("reduce_metrics logging_outputs: ", logging_outputs)
        # if "loss" in logging_outputs[0]:
        #     total_loss_name = "loss"
        # elif "total_loss" in logging_outputs[0]:
        #     total_loss_name = "total_loss"
        
        reduced_metrics_dict = {}

        try:
            lnp_ids = []
            for log in logging_outputs:
                lnp_ids.extend(log.get("lnp_ids", []))
                # print("log.keys(): ", log.keys())
            # print("reduce_metrics, lnp_ids: ", lnp_ids)
            reduced_metrics_dict['lnp_ids'] = lnp_ids
        except:
            print("Cannot extract lnp_ids from logging_outputs!")
            print("logging_outputs[0].keys(): ", logging_outputs[0].keys())

        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        # print("reduce_metrics loss_sum: ", loss_sum)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        # print("logging_outputs: ", logging_outputs)
        # print("log.get(sample_size, None) for log in logging_outputs reduce_metrics: ", [log.get("sample_size", None) for log in logging_outputs])
        # print("sample_size reduce_metrics: ", sample_size)
        # we divide by log(2) to convert the loss from base e to base 2

        # print("loss_sum: ", loss_sum, " sample_size: ", sample_size)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )

        # check for multiple tasks; if so, log each task's loss
        task_loss_list_dict = {}
        task_loss_sum_dict = {}
        task_sample_size_list_dict = {}
        task_sample_size_dict = {}
        for log in logging_outputs:
            for key in log:
                if "loss_" in key:
                    # task_name = key
                    task_name = key.split("loss_")[1]
                    if task_name not in task_loss_list_dict:
                        task_loss_list_dict[task_name] = []
                        task_sample_size_list_dict[task_name] = []

                    task_loss_list_dict[task_name].append(log.get(key, 0))

                    target_name = 'target_' + task_name
                    task_sample_size = (~log[target_name].isnan()).sum() # count number of non-nan values in target for that task
                    task_sample_size_list_dict[task_name].append(task_sample_size)
                    
        task_loss_sum_dict = {task_name: sum(task_loss_list_dict[task_name]) for task_name in task_loss_list_dict}
        task_sample_size_dict = {task_name: sum(task_sample_size_list_dict[task_name]) for task_name in task_sample_size_list_dict}

        # print("task_loss_sum_dict: ", task_loss_sum_dict)
        # print("task_sample_size_dict: ", task_sample_size_dict)
        if len(task_loss_sum_dict) > 0:
            for task_name in task_loss_sum_dict:
                metrics.log_scalar(
                    "loss_" + task_name, task_loss_sum_dict[task_name] / task_sample_size_dict[task_name] / math.log(2), task_sample_size_dict[task_name], round=3
                )

                reduced_metrics_dict["loss_" + task_name] = task_loss_sum_dict[task_name] / task_sample_size_dict[task_name] / math.log(2)


        # print("logging_outputs: ", logging_outputs)
        if "valid" in split or "test" in split or infer:
            # process evaluation metric values with predict_<task_name> and  target_<task_name>
            components_list = [
                str(item) for log in logging_outputs for item in log.get("components")
            ]

            # compile cls_representations
            for key in logging_outputs[0]:
                if 'cls_' in key:
                    cls_rep_list = [log.get(key) for log in logging_outputs]
                    cls_representations = torch.cat(cls_rep_list, dim=0)
                    reduced_metrics_dict[key] = cls_representations.cpu()

            if "predict" in logging_outputs[0]:
                predicts = torch.cat([log.get("predict") for log in logging_outputs], dim=0)
                if predicts.size(-1) == 1:
                    # single label regression task, add aggregate acc and loss score
                    targets = torch.cat(
                        [log.get("target", 0) for log in logging_outputs], dim=0
                    )

                    compute_reg_metrics(predicts, targets, components_list, split, task_name=None, metric_dict=reduced_metrics_dict, contrast_margin_coeff=contrast_margin_coeff, infer=infer, use_lambdarank=use_lambdarank, lambdarank_k=lambdarank_k)

            for key in logging_outputs[0]:
                if "predict" == key:
                    predicts = torch.cat([log.get("predict") for log in logging_outputs], dim=0)
                    if predicts.size(-1) == 1:
                        # single label regression task, add aggregate acc and loss score
                        targets = torch.cat(
                            [log.get("target", 0) for log in logging_outputs], dim=0
                        )

                        compute_reg_metrics(predicts, targets, components_list, split, task_name=None, metric_dict=reduced_metrics_dict, contrast_margin_coeff=contrast_margin_coeff, infer=infer, use_lambdarank=use_lambdarank, lambdarank_k=lambdarank_k)

                elif "predict_" in key:
                    task_name = key.split("predict_")[1]
                    predict_name = 'predict_' + task_name
                    target_name = 'target_' + task_name

                    predicts = torch.cat([log.get(predict_name) for log in logging_outputs], dim=0)
                    if predicts.size(-1) == 1:
                        # single label regression task, add aggregate acc and loss score
                        targets = torch.cat(
                            [log.get(target_name, 0) for log in logging_outputs], dim=0
                        )
                        print("reduce_metrics, task_name: ", task_name)
                        compute_reg_metrics(predicts, targets, components_list, split, task_name=task_name, metric_dict=reduced_metrics_dict, contrast_margin_coeff=contrast_margin_coeff, infer=infer, use_lambdarank=use_lambdarank, lambdarank_k=lambdarank_k)

        return reduced_metrics_dict

    @staticmethod
    def logging_outputs_can_be_summed(is_train) -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return is_train

