# -*- coding: utf-8 -*-
"""
DC-YOLO26 数据集
================

在 YOLODataset 基础上额外加载密度图（由 crowd/prepare_data.py 预生成），
并保证密度图与图像经历完全相同的 letterbox / 水平翻转几何变换，
全过程保持密度积分（人数）守恒。

为保证点-框-密度三者严格对齐，训练几何增强仅使用：
- LetterBox（确定性缩放填充）
- 同步随机水平翻转（CrowdFlip）
- HSV 色彩增强（不影响密度）
不使用 mosaic / mixup / copy-paste / 随机透视等无法与密度图严格同步的增强。
"""

from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import torch
from ultralytics.data.augment import Format, LetterBox, RandomFlip, RandomHSV
from ultralytics.data.augment import Compose
from ultralytics.data.dataset import YOLODataset

from .density import resize_density_preserve_count
from .protocol import data_path, path_key, read_count_manifest


def im2density_path(im_file: str) -> Path:
    """images/split/x.jpg -> density/split/x.npy。"""
    p = Path(im_file)
    parts = ["density" if part == "images" else part for part in p.parts]
    return Path(*parts).with_suffix(".npy")


class CrowdLetterBox(LetterBox):
    """与密度图同步的 LetterBox。"""

    def __init__(self, *args, density_stride: int = 4, **kwargs):
        super().__init__(*args, **kwargs)
        self.density_stride = density_stride

    def apply_image(self, labels: dict, params: dict) -> dict:
        labels = super().apply_image(labels, params)
        d = labels.pop("density_full", None)
        if d is None:
            return labels
        new_h, new_w = params["new_shape"]
        uw, uh = params["new_unpad"]  # (w, h)
        top, left = params["top"], params["left"]
        d = resize_density_preserve_count(d, int(uh), int(uw))
        canvas = np.zeros((new_h, new_w), dtype=np.float32)
        bottom = min(top + int(uh), new_h)
        right = min(left + int(uw), new_w)
        canvas[top:bottom, left:right] = d[: bottom - top, : right - left]
        sh, sw = new_h // self.density_stride, new_w // self.density_stride
        labels["density"] = torch.from_numpy(resize_density_preserve_count(canvas, sh, sw))
        return labels


class CrowdFlip(RandomFlip):
    """水平翻转时同步翻转密度图。"""

    def apply_image(self, labels: dict, params: dict) -> dict:
        labels = super().apply_image(labels, params)
        if params["flip"]:
            labels["img"] = np.ascontiguousarray(labels["img"])  # np.fliplr 返回视图
        return labels

    def apply_depth(self, labels: dict, params: dict) -> dict:
        d = labels.get("density")
        if d is not None and params["flip"] and params["direction"] == "horizontal":
            if isinstance(d, torch.Tensor):
                labels["density"] = torch.flip(d, dims=[-1]).contiguous()
            else:
                labels["density"] = np.ascontiguousarray(np.fliplr(d))
        return labels


class CrowdDataset(YOLODataset):
    """带密度图监督的人群计数检测数据集。"""

    def __init__(self, *args, density_stride: int = 4, **kwargs):
        self.density_stride = density_stride
        self.density_root = (kwargs.get("data") or {}).get("density_root")
        self.data_root = (kwargs.get("data") or {}).get("path")
        self.density_count_lookup = None
        data = kwargs.get("data") or {}
        self.density_count_tolerance = float(data.get("density_count_tolerance", 1e-4))
        if data.get("supervision_manifest"):
            supervision = json.loads(data_path(data, data["supervision_manifest"]).read_text(encoding="utf-8"))
            self.density_count_lookup = {path_key(data_path(data, row["image"])): row["density_count"] for row in supervision["records"]}
        self.count_lookup = read_count_manifest(kwargs.get("data") or {})
        super().__init__(*args, **kwargs)

    def get_image_and_label(self, index: int) -> dict:
        label = super().get_image_and_label(index)
        if self.density_root:
            relative_image = Path(label["im_file"]).resolve().relative_to(Path(self.data_root).resolve() / "images")
            den_path = Path(self.density_root) / relative_image.with_suffix(".npy")
        else:
            den_path = im2density_path(label["im_file"])
        if not den_path.exists():
            raise FileNotFoundError(
                f"缺少密度图 {den_path}，请先运行 python -m crowd.prepare_data 预处理数据集"
            )
        label["density_full"] = np.load(den_path).astype(np.float32)
        if self.density_count_lookup is not None:
            expected = self.density_count_lookup[path_key(label["im_file"])]
            observed = float(label["density_full"].sum(dtype=np.float64))
            if not np.isfinite(label["density_full"]).all() or np.any(label["density_full"] < 0) or abs(observed - expected) > self.density_count_tolerance:
                raise ValueError(f"密度监督计数不守恒：{den_path}，应为 {expected}，实际 {observed}")
        # Use original point counts, not Gaussian mass or filtered pseudo boxes.
        label["gt_count"] = torch.tensor(
            self.count_lookup[path_key(label["im_file"])] if self.count_lookup is not None
            else len(label["cls"]), dtype=torch.int64,
        )
        return label

    def build_transforms(self, hyp=None) -> Compose:
        if self.augment:
            hyp.mosaic = 0.0
            hyp.mixup = 0.0
            hyp.cutmix = 0.0
            hyp.copy_paste = 0.0
            transforms = Compose(
                [
                    CrowdLetterBox(
                        (self.imgsz, self.imgsz),
                        scaleup=True,
                        auto=False,
                        density_stride=self.density_stride,
                    ),
                    CrowdFlip(p=getattr(hyp, "fliplr", 0.5), direction="horizontal"),
                    RandomHSV(
                        hgain=getattr(hyp, "hsv_h", 0.015),
                        sgain=getattr(hyp, "hsv_s", 0.7),
                        vgain=getattr(hyp, "hsv_v", 0.4),
                    ),
                ]
            )
        else:
            transforms = Compose(
                [
                    CrowdLetterBox(
                        (self.imgsz, self.imgsz),
                        scaleup=False,
                        auto=False,
                        density_stride=self.density_stride,
                    )
                ]
            )
        transforms.append(
            Format(
                bbox_format="xywh",
                normalize=True,
                return_mask=False,
                return_keypoint=False,
                return_obb=False,
                batch_idx=True,
                bgr=hyp.bgr if self.augment else 0.0,
            )
        )
        return transforms

    @staticmethod
    def collate_fn(batch: list[dict]) -> dict:
        """在官方 collate 基础上把 density 堆叠为 (B, H/4, W/4)。"""
        new_batch = {}
        batch = [dict(sorted(b.items())) for b in batch]
        keys = batch[0].keys()
        values = list(zip(*[list(b.values()) for b in batch]))
        for i, k in enumerate(keys):
            value = values[i]
            if k in {"img", "text_feats", "semantic_mask", "sem_masks", "depth", "density", "gt_count"}:
                value = torch.stack(value, 0)
            elif k == "visuals":
                value = torch.nn.utils.rnn.pad_sequence(value, batch_first=True)
            if k in {"masks", "keypoints", "bboxes", "cls", "segments", "obb"}:
                value = torch.cat(value, 0)
            new_batch[k] = value
        if "batch_idx" in new_batch:
            new_batch["batch_idx"] = list(new_batch["batch_idx"])
            for i in range(len(new_batch["batch_idx"])):
                new_batch["batch_idx"][i] += i
            new_batch["batch_idx"] = torch.cat(new_batch["batch_idx"], 0)
        return new_batch
