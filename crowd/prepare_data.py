# -*- coding: utf-8 -*-
"""
DC-YOLO26 数据集预处理
======================

把人群计数数据集的点标注转换为 DC-YOLO26 训练所需的 YOLO 目录：

    <out>/
      images/{train,val}/<prefix>_<name>.jpg
      labels/{train,val}/<prefix>_<name>.txt   # 伪框 0 cx cy w h
      density/{train,val}/<prefix>_<name>.npy  # 原图分辨率密度图
      data.yaml

支持的数据集：
  sht_a ShanghaiTech Part A（几何自适应高斯核）
  sht_b ShanghaiTech Part B（固定高斯核，sigma=15）
  qnrf  UCF-QNRF（几何自适应核）
  jhu   JHU-Crowd++（txt 标注，几何自适应核）

用法：
  python -m crowd.prepare_data --dataset sht_a --src /path/to/part_A --dst datasets/sht_a
  python -m crowd.prepare_data --dataset qnrf --src /path/to/UCF-QNRF --dst datasets/qnrf
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml

from .density import build_density_map, build_pseudo_boxes, load_points

try:
    from tqdm import tqdm
except ImportError:  # tqdm 可选
    def tqdm(x, **kwargs):
        return x


# 每种数据集的 (图像子目录, 标注子目录, 标注格式, 标注文件名构造, 固定核 sigma)
SPECS = {
    "sht_a": dict(
        splits={"train": "train_data", "val": "test_data"},
        img_dir="images",
        gt_dir="ground_truth",
        fmt="sht",
        fixed_sigma=None,
        gt_name=lambda stem: f"GT_{stem}.mat",
    ),
    "sht_b": dict(
        splits={"train": "train_data", "val": "test_data"},
        img_dir="images",
        gt_dir="ground_truth",
        fmt="sht",
        fixed_sigma=15.0,
        gt_name=lambda stem: f"GT_{stem}.mat",
    ),
    "qnrf": dict(
        splits={"train": "Train", "val": "Test"},
        img_dir="images",
        gt_dir="annots",
        fmt="qnrf",
        fixed_sigma=None,
        gt_name=lambda stem: f"{stem}.mat",
    ),
    "jhu": dict(
        splits={"train": "train", "val": "val"},
        img_dir="images",
        gt_dir="gt",
        fmt="txt",
        fixed_sigma=None,
        gt_name=lambda stem: f"{stem}.txt",
    ),
}


def _list_images(folder: Path) -> list[Path]:
    files = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
        files.extend(sorted(folder.glob(ext)))
    return files


def prepare_split(
    src_root: Path,
    out_root: Path,
    split: str,
    split_dir: str,
    spec: dict,
    prefix: str,
    beta: float,
    size_ratio: float,
    min_size: float,
    max_size: float,
):
    img_dir = src_root / split_dir / spec["img_dir"]
    gt_dir = src_root / split_dir / spec["gt_dir"]
    if not img_dir.exists():
        raise FileNotFoundError(f"图像目录不存在: {img_dir}")

    out_img = out_root / "images" / split
    out_lbl = out_root / "labels" / split
    out_den = out_root / "density" / split
    for d in (out_img, out_lbl, out_den):
        d.mkdir(parents=True, exist_ok=True)

    records = []
    for img_path in tqdm(_list_images(img_dir), desc=f"{prefix}/{split}"):
        stem = img_path.stem
        gt_path = gt_dir / spec["gt_name"](stem)
        if not gt_path.exists():
            # JHU 等数据集标注名可能大小写不同
            cands = list(gt_dir.glob(f"{stem}.*"))
            if not cands:
                print(f"[跳过] 缺少标注: {gt_path}")
                continue
            gt_path = cands[0]

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[跳过] 图像无法读取: {img_path}")
            continue
        h, w = img.shape[:2]
        points = load_points(str(gt_path), dataset=spec["fmt"])
        if len(points) > 0:
            points = points[(points[:, 0] >= 0) & (points[:, 0] < w) &
                            (points[:, 1] >= 0) & (points[:, 1] < h)]

        density = build_density_map(points, h, w, beta=beta,
                                    fixed_sigma=spec["fixed_sigma"])
        boxes = build_pseudo_boxes(points, h, w, size_ratio=size_ratio,
                                   min_size=min_size, max_size=max_size)

        new_stem = f"{prefix}_{stem}"
        shutil.copy2(img_path, out_img / f"{new_stem}{img_path.suffix.lower()}")
        np.save(out_den / f"{new_stem}.npy", density)
        with open(out_lbl / f"{new_stem}.txt", "w", encoding="utf-8") as f:
            for b in boxes:
                f.write(f"{int(b[0])} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}\n")
        records.append((new_stem, len(points)))

    print(f"[{prefix}/{split}] 图像 {len(records)} 张")
    return records


def main():
    ap = argparse.ArgumentParser(description="DC-YOLO26 人群计数数据预处理")
    ap.add_argument("--dataset", required=True, choices=list(SPECS.keys()))
    ap.add_argument("--src", required=True, help="原始数据集根目录")
    ap.add_argument("--dst", required=True, help="输出目录")
    ap.add_argument("--beta", type=float, default=0.3, help="几何自适应高斯核系数")
    ap.add_argument("--size-ratio", type=float, default=0.9, help="伪框边长 / kNN 距离")
    ap.add_argument("--min-size", type=float, default=6.0, help="伪框最小边长(原图 px)")
    ap.add_argument("--max-size", type=float, default=96.0, help="伪框最大边长(原图 px)")
    args = ap.parse_args()

    spec = SPECS[args.dataset]
    src_root = Path(args.src).resolve()
    out_root = Path(args.dst).resolve()
    prefix = args.dataset.replace("_", "")

    counts = {}
    for split, split_dir in spec["splits"].items():
        records = prepare_split(
            src_root, out_root, split, split_dir, spec, prefix,
            args.beta, args.size_ratio, args.min_size, args.max_size,
        )
        counts[split] = len(records)

    data_cfg = {
        "path": str(out_root).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "nc": 1,
        "names": ["head"],
        "density_dir": "density",
        "density_stride": 4,
    }
    with open(out_root / "data.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(data_cfg, f, allow_unicode=True, sort_keys=False)

    print(f"\n完成: {out_root}")
    print(f"  train={counts.get('train', 0)}  val={counts.get('val', 0)}")


if __name__ == "__main__":
    main()
