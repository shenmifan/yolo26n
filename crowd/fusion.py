# -*- coding: utf-8 -*-
"""
Density-Guided Fusion (DGF)
===========================

推理计数策略：把图像划分为 G×G 个空间网格，
- 低密度网格（密度积分 <= tau）：信任检测路（实例计数准确，可定位）；
- 高密度网格（密度积分 > tau）：信任密度路（检测路在严重遮挡下漏检，
  且 YOLO26 one-to-one 头存在每图 300 上限）。

最终计数 = Σ_low N_det(cell) + Σ_high N_den(cell)。
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def density_grid_counts(density: np.ndarray, grid: int = 16) -> np.ndarray:
    """把密度图聚合为 grid×grid 的每格人数（积分）。"""
    h, w = density.shape[:2]
    gh, gw = h // grid, w // grid
    d = density[: gh * grid, : gw * grid]
    cells = d.reshape(grid, gh, grid, gw).sum(axis=(1, 3))
    return cells  # (G, G)


def boxes_grid_counts(
    boxes_xyxy: np.ndarray,
    imgsz: int,
    density_hw: tuple[int, int],
    grid: int = 16,
) -> np.ndarray:
    """把检测框（letterbox 画布坐标，xyxy）按中心落到 G×G 网格计数。"""
    counts = np.zeros((grid, grid), dtype=np.float32)
    if len(boxes_xyxy) == 0:
        return counts
    dh, dw = density_hw
    cx = (boxes_xyxy[:, 0] + boxes_xyxy[:, 2]) / 2.0
    cy = (boxes_xyxy[:, 1] + boxes_xyxy[:, 3]) / 2.0
    # 画布坐标 -> 密度图坐标（密度图分辨率为 imgsz/4）
    sx, sy = dw / float(imgsz), dh / float(imgsz)
    gx = np.clip((cx * sx / (dw / grid)).astype(int), 0, grid - 1)
    gy = np.clip((cy * sy / (dh / grid)).astype(int), 0, grid - 1)
    np.add.at(counts, (gy, gx), 1)
    return counts


def dgf_fuse(
    density: np.ndarray,
    boxes_xyxy: np.ndarray,
    imgsz: int,
    grid: int = 16,
    tau: float = 2.0,
) -> dict:
    """密度引导融合计数。

    Returns:
        dict: count_fused / count_density / count_detect / 每格融合结果。
    """
    d_cell = density_grid_counts(density, grid)
    b_cell = boxes_grid_counts(boxes_xyxy, imgsz, density.shape[:2], grid)
    high = d_cell > tau
    fused = np.where(high, d_cell, b_cell)
    return {
        "count_fused": float(fused.sum()),
        "count_density": float(d_cell.sum()),
        "count_detect": float(b_cell.sum()),
        "density_cells": d_cell,
        "detect_cells": b_cell,
        "high_mask": high,
    }


# ---------------------------------------------------------------------------
# 训练时的可微网格一致性损失
# ---------------------------------------------------------------------------
def soft_detection_grid_counts(
    scores: torch.Tensor,
    feat_sizes: list[tuple[int, int]],
    grid: int = 16,
) -> torch.Tensor:
    """从 one-to-many 分类分数图构造每格软检出数 (B, G, G)。

    Args:
        scores: (B, nc, A)，sigmoid 之前或之后的分类分数均可（调用处先 sigmoid）。
        feat_sizes: 各检测尺度特征图的 (h, w)，顺序与 scores 拼接顺序一致。
    """
    b, nc, _ = scores.shape
    soft = scores.new_zeros(b, grid, grid)
    start = 0
    for h, w in feat_sizes:
        a = h * w
        s = scores[..., start : start + a].view(b, nc, h, w).sum(dim=1, keepdim=True)  # (B,1,h,w)
        # 每格期望检出数 = 格内分数均值 × 格内 anchor 数
        pooled = F.adaptive_avg_pool2d(s, (grid, grid)) * (h * w) / (grid * grid)
        soft = soft + pooled.squeeze(1)
        start += a
    return soft


def density_grid_counts_torch(density: torch.Tensor, grid: int = 16) -> torch.Tensor:
    """密度图聚合为 (B, G, G) 每格人数，任意输入尺寸均可。"""
    h, w = density.shape[-2:]
    # 平均密度 × 每格像素数 = 每格人数
    pooled = F.adaptive_avg_pool2d(density.unsqueeze(1), (grid, grid)).squeeze(1)
    return pooled * (h * w) / (grid * grid)  # (B, G, G)
