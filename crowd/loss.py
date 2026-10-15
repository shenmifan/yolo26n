# -*- coding: utf-8 -*-
"""
DC-YOLO26 损失函数
==================

总损失：
    L = L_det(YOLO26 E2E 双头渐进损失)
      + lambda_den  * L_den    密度图 MSE
      + lambda_cnt  * L_cnt    总计数相对误差（密度积分 vs 人头数）
      + lambda_cons * L_cons   网格级检测-密度计数一致性（仅在可分辨的低密度网格）

L_cons 设计动机：密度分支直接受点标注监督，是可靠的"计数教师"；在人头可分辨的
稀疏/中密度网格中，要求检测路的软检出数与密度积分一致，抑制重复检测与漏检；
高密度网格检测路无法分辨，不施加一致性约束（与推理时 DGF 的信任策略一致）。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.loss import E2ELoss

from .fusion import density_grid_counts_torch, soft_detection_grid_counts
from .density import resize_density_preserve_count_torch


def _arg(model, name: str, default):
    """从模型 crowd_cfg（data.yaml 注入）读取自定义损失权重，缺省用默认值。"""
    cfg = getattr(model, "crowd_cfg", None) or {}
    return cfg.get(name, default)


class CrowdCriterion:
    """检测 + 密度计数联合损失。"""

    def __init__(self, model):
        self.det_loss = E2ELoss(model)
        self.mse = nn.MSELoss(reduction="mean")
        self.lambda_den = float(_arg(model, "den_gain", 1.0))
        self.lambda_cnt = float(_arg(model, "cnt_gain", 0.1))
        self.lambda_cons = float(_arg(model, "cons_gain", 0.5))
        self.grid = int(_arg(model, "grid", 16))
        self.tau_cons = float(_arg(model, "tau_cons", 3.0))

    def update(self):
        """委托给 YOLO26 渐进损失的 epoch 调度（trainer 每 epoch 调用）。"""
        self.det_loss.update()

    # ------------------------------------------------------------------
    def density_loss(self, den_pred: torch.Tensor, gt: torch.Tensor):
        """密度图 MSE + 总计数相对误差。"""
        # 对齐多尺度训练造成的分辨率差异，插值时保持密度积分守恒
        if den_pred.shape[-2:] != gt.shape[-2:]:
            den_pred = resize_density_preserve_count_torch(den_pred, gt.shape[-2:])
        den = den_pred.squeeze(1)  # (B, h, w)

        l_den = self.mse(den, gt)

        n_pred = den.sum(dim=(1, 2))
        n_gt = gt.sum(dim=(1, 2))
        l_cnt = (F.smooth_l1_loss(n_pred, n_gt, reduction="none", beta=1.0) / (n_gt + 1.0)).mean()
        return l_den, l_cnt, den

    def consistency_loss(self, det_preds: dict, gt_density: torch.Tensor) -> torch.Tensor:
        o2m = det_preds["one2many"]
        scores = o2m["scores"].sigmoid()  # (B, nc, A)
        feat_sizes = [tuple(f.shape[-2:]) for f in o2m["feats"]]
        soft = soft_detection_grid_counts(scores, feat_sizes, self.grid)
        # GT 密度是可靠的计数教师：网格积分与可信门限均由 GT 决定
        d_cell = density_grid_counts_torch(gt_density.detach(), self.grid)
        mask = (d_cell <= self.tau_cons).float()
        if mask.sum() < 1.0:
            return soft.sum() * 0.0
        l = (F.smooth_l1_loss(soft, d_cell.detach(), reduction="none", beta=0.5) * mask).sum() / mask.sum()
        return l

    # ------------------------------------------------------------------
    def __call__(self, preds, batch):
        det_preds, density_pred = preds
        # eval 模式下 Detect 输出 (decoded, raw_dict)，E2E 损失与一致性均需要 raw dict
        if isinstance(det_preds, tuple):
            det_preds = det_preds[1]
        det_loss, det_items = self.det_loss(det_preds, batch)
        if density_pred is None:
            zero = det_loss.detach().sum() * 0.0
            return det_loss, dict(det_items, den_loss=zero, cnt_loss=zero, cons_loss=zero)

        gt = batch["density"].to(density_pred.device, dtype=density_pred.dtype)
        l_den, l_cnt, den_aligned = self.density_loss(density_pred, gt)
        l_cons = self.consistency_loss(det_preds, gt) if self.lambda_cons else density_pred.sum() * 0.0

        loss = det_loss + self.lambda_den * l_den + self.lambda_cnt * l_cnt + self.lambda_cons * l_cons
        items = dict(det_items)
        items["den_loss"] = l_den.detach()
        items["cnt_loss"] = l_cnt.detach()
        items["cons_loss"] = l_cons.detach()
        return loss, items
