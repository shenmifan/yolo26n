# -*- coding: utf-8 -*-
"""
DC-YOLO26 密度图与伪框生成模块
================================

将人群计数数据集的点标注（人头中心点）转换为两种监督信号：

1. 几何自适应高斯密度图（geometry-adaptive Gaussian density map），
   供密度计数分支（DensityHead）监督，密度图积分等于人头总数；
2. 几何自适应伪边界框（pseudo bounding boxes），供 YOLO26 检测分支监督，
   框尺寸由 k 近邻平均距离决定，YOLO 归一化 xywh 格式。

参考：
- Zhang et al., "Single-Image Crowd Counting via Multi-Column Convolutional
  Neural Network" (CVPR 2016)，几何自适应高斯核 sigma = beta * mean(d_kNN)。
- Liu et al. (2019)，点标注到伪框的最近邻距离初始化策略。
"""

from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np
from scipy import ndimage, spatial


# ---------------------------------------------------------------------------
# 点标注读取
# ---------------------------------------------------------------------------
def load_points(mat_path: str, dataset: str = "sht") -> np.ndarray:
    """读取点标注，返回形状为 (N, 2) 的数组，列为 (x, y)（像素坐标，可为小数）。

    Args:
        mat_path: 标注文件路径。
        dataset: 数据格式。
            - "sht":  ShanghaiTech Part A/B，老式 .mat，
                     image_info -> crowd_center -> Nx2。
            - "qnrf": UCF-QNRF，annPoints 为 Nx2（可能是 v7.3/HDF5）。
            - "jhu":  JHU-Crowd++，txt，每行 "x y ..."。
            - "nwpu": NWPU-Crowd，txt，每行 "x y"。
    """
    dataset = dataset.lower()
    if dataset in ("sht", "qnrf"):
        try:
            from scipy.io import loadmat

            mat = loadmat(mat_path)
            if dataset == "sht":
                points = mat["image_info"][0, 0][0, 0][0].astype(np.float64)
            else:
                points = np.asarray(mat["annPoints"], dtype=np.float64)
        except NotImplementedError:
            # v7.3 mat (HDF5)
            import h5py

            with h5py.File(mat_path, "r") as f:
                key = "image_info" if dataset == "sht" else "annPoints"
                points = np.asarray(f[key], dtype=np.float64)
                points = points.T if points.shape[0] == 2 else points
        return points.reshape(-1, 2)

    if dataset in ("jhu", "nwpu", "txt"):
        pts = []
        with open(mat_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                vals = line.replace(",", " ").split()
                if len(vals) < 2:
                    continue
                try:
                    pts.append([float(vals[0]), float(vals[1])])
                except ValueError:
                    continue
        return np.asarray(pts, dtype=np.float64).reshape(-1, 2)

    raise ValueError(f"未知数据集格式: {dataset}")


# ---------------------------------------------------------------------------
# k 近邻距离
# ---------------------------------------------------------------------------
def knn_mean_dist(points: np.ndarray, k: int = 3) -> np.ndarray:
    """每个点到其 k 个最近邻的平均欧氏距离，返回长度 N 的数组。"""
    n = len(points)
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    if n == 1:
        return np.ones(1, dtype=np.float64) * 15.0
    k = min(k, n - 1)
    tree = spatial.KDTree(points)
    dist, _ = tree.query(points, k=k + 1)  # 第 0 列是自身
    dist = np.atleast_2d(dist)[:, 1:]
    return dist.mean(axis=1)


# ---------------------------------------------------------------------------
# 密度图
# ---------------------------------------------------------------------------
def _splat(points: np.ndarray, h: int, w: int) -> np.ndarray:
    """把亚像素点按双线性权重分摊到离散网格，保证冲激总质量为 N。"""
    canvas = np.zeros((h, w), dtype=np.float64)
    x0 = np.floor(points[:, 0]).astype(np.int64)
    y0 = np.floor(points[:, 1]).astype(np.int64)
    # Renormalize the surviving bilinear weights of each border point.
    weights = []
    for ix in (x0, x0 + 1):
        for iy in (y0, y0 + 1):
            wx = 1.0 - np.abs(ix - points[:, 0])
            wy = 1.0 - np.abs(iy - points[:, 1])
            valid = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
            weights.append((ix, iy, np.where(valid, wx * wy, 0.0)))
    total = sum(item[2] for item in weights)
    if np.any(total <= 0):
        raise ValueError("_splat 接收到图像范围外的点")
    for ix, iy, weight in weights:
        valid = weight > 0
        np.add.at(canvas, (iy[valid], ix[valid]), weight[valid] / total[valid])
    return canvas


def audit_points(points: np.ndarray, h: int, w: int):
    """Keep every finite in-image annotation, including duplicate entries."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or h <= 0 or w <= 0:
        raise ValueError("点数组必须为 (N,2)，图像尺寸必须为正")
    finite = np.isfinite(points).all(axis=1)
    inside = finite & (points[:, 0] >= 0) & (points[:, 0] < w) & (points[:, 1] >= 0) & (points[:, 1] < h)
    valid = points[inside]
    return inside, dict(raw_count=len(points), valid_count=len(valid),
                        nonfinite_count=int((~finite).sum()), outside_count=int((finite & ~inside).sum()),
                        duplicate_extra_count=int(finite.sum()) - len(np.unique(points[finite], axis=0)),
                        valid_duplicate_extra_count=len(valid) - len(np.unique(valid, axis=0)))


@lru_cache(maxsize=64)
def _boundary_kernel_mass(length: int, sigma: float):
    """Mass left in the image by a Gaussian centered at each integer pixel."""
    mass = ndimage.gaussian_filter1d(np.ones(length, dtype=np.float64), sigma=sigma,
                                   mode="constant", truncate=4.0)
    mass.flags.writeable = False
    return mass


def _density_with_count(density: np.ndarray, target: float):
    """Correct numerical/resampling mass drift without fabricating spatial support."""
    density = np.asarray(density)
    if not np.isfinite(density).all() or np.any(density < 0):
        raise ValueError("密度图包含非有限值或负值")
    if target == 0:
        return np.zeros(density.shape, dtype=np.float32)
    mass = float(density.sum(dtype=np.float64))
    if mass <= 0 or not np.isfinite(target) or target < 0:
        raise ValueError("非零计数对应零密度质量，无法保持计数")
    return (density.astype(np.float64) * (target / mass)).astype(np.float32)


def _add_unit_gaussian(density, x, y, sigma):
    """Accumulate one locally truncated, unit-mass adaptive Gaussian."""
    h, w = density.shape
    radius = int(4.0 * sigma + 0.5)
    x0, y0 = int(np.floor(x)), int(np.floor(y))
    left, right = max(0, x0 - radius), min(w, x0 + radius + 2)
    top, bottom = max(0, y0 - radius), min(h, y0 + radius + 2)
    impulse = _splat(np.array([[x - left, y - top]]), bottom - top, right - left)
    kernel = ndimage.gaussian_filter(impulse, sigma=sigma, mode="constant", truncate=4.0)
    kernel /= kernel.sum(dtype=np.float64)
    density[top:bottom, left:right] += kernel


def build_density_map(
    points: np.ndarray,
    h: int,
    w: int,
    k: int = 3,
    beta: float = 0.3,
    sigma_min: float = 1.0,
    fixed_sigma: float | None = None,
) -> np.ndarray:
    """生成密度图，积分等于有限、图内标注数；重复标注逐条贡献质量。

    Args:
        points: (N, 2)，列为 (x, y)。
        h, w: 图像高宽。
        k: 近邻数。
        beta: 高斯核 sigma = beta * kNN 平均距离（CSRNet 取 0.3）。
        sigma_min: sigma 下限，防止稀疏点核过小。
        fixed_sigma: 若给定，则使用固定 sigma（Part B 等固定核场景）。
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    inside, _ = audit_points(points, h, w)
    points = points[inside]
    if len(points) == 0:
        return np.zeros((h, w), dtype=np.float32)

    if fixed_sigma is not None:
        if not np.isfinite(fixed_sigma) or fixed_sigma <= 0:
            raise ValueError("fixed_sigma 必须为有限正数")
        canvas = _splat(points, h, w)
        # Each source pixel's kernel is normalized separately. A single global
        # gain would incorrectly transfer missing border mass to interior people.
        normalizer = _boundary_kernel_mass(h, float(fixed_sigma))[:, None] * _boundary_kernel_mass(w, float(fixed_sigma))[None, :]
        density = ndimage.gaussian_filter(canvas / normalizer, sigma=fixed_sigma, mode="constant", truncate=4.0)
        return _density_with_count(density, len(points))

    mean_dist = knn_mean_dist(points, k=k)
    sigmas = np.maximum(beta * mean_dist, sigma_min)

    density = np.zeros((h, w), dtype=np.float64)
    for (x, y), sigma in zip(points, sigmas):
        _add_unit_gaussian(density, x, y, float(sigma))
    return _density_with_count(density, len(points))


# ---------------------------------------------------------------------------
# 伪边界框
# ---------------------------------------------------------------------------
def build_pseudo_boxes(
    points: np.ndarray,
    h: int,
    w: int,
    k: int = 3,
    size_ratio: float = 0.9,
    min_size: float = 6.0,
    max_size: float = 96.0,
    default_size: float = 16.0,
) -> np.ndarray:
    """点标注 -> 正方形伪框，YOLO 归一化格式 (cls=0, cx, cy, bw, bh)。

    框边长 = clip(size_ratio * kNN 平均距离, min_size, max_size)。
    密集场景下相邻人头中心距离近似人头直径，故 size_ratio≈1；
    稀疏场景距离退化为场景间距，用 max_size 截断。
    """
    if len(points) == 0:
        return np.zeros((0, 5), dtype=np.float32)

    mean_dist = knn_mean_dist(points, k=k)
    sizes = np.clip(mean_dist * size_ratio, min_size, max_size)
    sizes = np.where(mean_dist == 15.0, default_size, sizes)

    boxes = np.zeros((len(points), 5), dtype=np.float32)
    boxes[:, 0] = 0  # 单类别：head
    boxes[:, 1] = np.clip(points[:, 0] / w, 0.0, 1.0)
    boxes[:, 2] = np.clip(points[:, 1] / h, 0.0, 1.0)
    boxes[:, 3] = sizes / w
    boxes[:, 4] = sizes / h
    return boxes


# ---------------------------------------------------------------------------
# 密度图几何变换（积分守恒）
# ---------------------------------------------------------------------------
def resize_density_preserve_count(density: np.ndarray, new_h: int, new_w: int) -> np.ndarray:
    """缩放密度图并保持积分（人数）不变。"""
    if density.ndim != 2 or min(new_h, new_w) <= 0:
        raise ValueError("密度图必须为二维，目标尺寸必须为正")
    target = float(density.sum(dtype=np.float64))
    out = cv2.resize(density, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return _density_with_count(out, target)


def resize_density_preserve_count_torch(density, size, mode="area"):
    """Differentiable count-preserving resize for batched training tensors."""
    import torch
    import torch.nn.functional as F

    if density.ndim != 4 or min(size) <= 0:
        raise ValueError("密度张量必须为 BCHW，目标尺寸必须为正")
    source = density.float()
    old_mass = source.double().sum(dim=(-2, -1), keepdim=True)
    kwargs = {"align_corners": False} if mode in {"bilinear", "bicubic"} else {}
    out = F.interpolate(source, size=size, mode=mode, **kwargs)
    new_mass = out.double().sum(dim=(-2, -1), keepdim=True)
    if torch.any((old_mass > 0) & (new_mass <= 0)):
        raise ValueError("缩放丢失所有非零密度质量")
    gain = torch.where(old_mass > 0, old_mass / new_mass.clamp_min(1e-30), torch.zeros_like(old_mass))
    return out * gain.to(out.dtype)


def letterbox_density(
    density: np.ndarray,
    scale: float,
    pad_w: float,
    pad_h: float,
    new_h: int,
    new_w: int,
) -> np.ndarray:
    """对密度图施加与图像相同的 letterbox 变换（缩放 + 填充）。

    Args:
        density: 原图分辨率密度图。
        scale: letterbox 缩放系数（新 = 旧 * scale）。
        pad_w, pad_h: 左/上填充像素。
        new_h, new_w: letterbox 后画布尺寸。
    """
    h, w = density.shape[:2]
    rh, rw = int(round(h * scale)), int(round(w * scale))
    scaled = resize_density_preserve_count(density, rh, rw)
    canvas = np.zeros((new_h, new_w), dtype=np.float32)
    top, left = int(round(pad_h)), int(round(pad_w))
    bottom = min(top + rh, new_h)
    right = min(left + rw, new_w)
    if top < 0 or left < 0 or top + rh > new_h or left + rw > new_w:
        raise ValueError("letterbox 参数会裁剪密度图，不能保证计数守恒")
    canvas[top:bottom, left:right] = scaled[: bottom - top, : right - left]
    return canvas


def density_count(density: np.ndarray) -> float:
    """密度图积分，即预测人数。"""
    return float(density.sum(dtype=np.float64))
