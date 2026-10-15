# -*- coding: utf-8 -*-
"""
DC-YOLO26 验证器
================

以人群计数指标 MAE / MSE 为核心，同时报告三种计数口径：
- detect:  仅检测路（NMS 后框数）
- density: 仅密度路（密度图积分）
- fused:   DGF 密度引导融合（论文主结果）

fitness 取 -MAE_fused，便于训练器按计数精度选择最佳权重。
"""

from __future__ import annotations

import numpy as np
import torch
from ultralytics.data import build_dataloader
from ultralytics.engine.validator import BaseValidator
from ultralytics.utils import LOGGER, colorstr, nms

from .dataset import CrowdDataset
from .fusion import dgf_fuse


def counting_postprocess(det_y, conf, iou, max_det, nc=1, end2end=False):
    """Validate the selected head format before confidence filtering/NMS."""
    if det_y.ndim != 3:
        raise ValueError(f"检测输出应为三维，实际 {tuple(det_y.shape)}")
    if end2end:
        if det_y.shape[-1] != 6:
            raise ValueError(f"one-to-one 输出应为 (B,N,6)，实际 {tuple(det_y.shape)}")
    elif det_y.shape[1] != nc + 4:
        raise ValueError(f"one-to-many 输出应为 (B,{nc + 4},N)，实际 {tuple(det_y.shape)}")
    return nms.non_max_suppression(
        det_y.clone(), conf, iou, nc=nc, multi_label=False,
        agnostic=True, max_det=max_det, end2end=end2end,
        max_time_img=10.0,
    )


class _CountMetrics:
    """最小 metrics 容器，提供 trainer 需要的 keys。"""

    def __init__(self):
        self.keys = [
            "metrics/MAE",
            "metrics/MSE",
            "metrics/MAE_det",
            "metrics/MAE_den",
        ]
        self.results_dict = {}


class CrowdValidator(BaseValidator):
    """人群计数验证器。"""

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None):
        # Detection mAP uses 0.001 by default; crowd counts require a fixed cutoff.
        if isinstance(args, dict):
            if args.get("conf") is None:
                args["conf"] = 0.25
        elif args is not None and getattr(args, "conf", None) is None:
            args.conf = 0.25
        super().__init__(dataloader, save_dir, args, _callbacks)
        if args is None:
            self.args.conf = 0.25
        self.metrics = _CountMetrics()
        self.args.task = "detect"
        self.grid = 16
        self.tau = 2.0

    def _read_crowd_cfg(self):
        data = getattr(self, "data", None) or {}
        self.grid = int(data.get("grid", 16))
        self.tau = float(data.get("dgf_tau", 2.0))

    # ------------------------------------------------------------------
    def preprocess(self, batch):
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device, non_blocking=self.device.type not in {"cpu", "mps"})
        batch["img"] = (batch["img"].half() if self.args.quantize == 16 else batch["img"].float()) / 255
        return batch

    def init_metrics(self, model):
        self.names = model.names
        self.end2end = bool(getattr(model, "end2end", False))
        self._read_crowd_cfg()
        self.density_enabled = getattr(model, "density_head", None) is not None
        self.gt_counts = []
        self.fused_counts = []
        self.det_counts = []
        self.den_counts = []

    def postprocess(self, preds):
        """preds = ((det_decoded, det_raw), density)。"""
        det, density = preds
        det_y = det[0] if isinstance(det, (tuple, list)) else det
        outputs = counting_postprocess(
            det_y, self.args.conf, self.args.iou, self.args.max_det,
            nc=len(self.names), end2end=self.end2end,
        )
        return list(zip(outputs, density if density is not None else [None] * len(outputs)))

    def update_metrics(self, preds, batch):
        imgsz = batch["img"].shape[-1]
        for si, (pred, density) in enumerate(preds):
            boxes = pred[:, :4].detach().cpu().numpy() if len(pred) else np.zeros((0, 4))
            if density is None:
                result = dict(count_fused=len(boxes), count_detect=len(boxes), count_density=float("nan"))
            else:
                den_np = density[0].detach().float().cpu().numpy()
                result = dgf_fuse(den_np, boxes, imgsz, grid=self.grid, tau=self.tau)
            gt_n = int(batch["gt_count"][si].item())
            self.gt_counts.append(gt_n)
            self.fused_counts.append(result["count_fused"])
            self.det_counts.append(result["count_detect"])
            self.den_counts.append(result["count_density"])

    def _mae_mse(self, pred):
        gt = np.asarray(self.gt_counts, dtype=np.float64)
        pr = np.asarray(pred, dtype=np.float64)
        return float(np.mean(np.abs(pr - gt))), float(np.sqrt(np.mean((pr - gt) ** 2)))

    def get_stats(self):
        mae_f, mse_f = self._mae_mse(self.fused_counts)
        mae_d, _ = self._mae_mse(self.det_counts)
        mae_e, _ = self._mae_mse(self.den_counts)
        stats = {
            "metrics/MAE": round(mae_f, 4),
            "metrics/MSE": round(mse_f, 4),
            "metrics/MAE_det": round(mae_d, 4),
            "metrics/MAE_den": round(mae_e, 4),
            "fitness": -mae_f,
        }
        self.metrics.results_dict = stats
        return stats

    def finalize_metrics(self):
        return

    def print_results(self):
        stats = self.metrics.results_dict
        if not self.density_enabled:
            LOGGER.info(f"{'all':>22}{len(self.gt_counts):>11d}  MAE(detect)={stats['metrics/MAE']:.3f}  RMSE(detect)={stats['metrics/MSE']:.3f}")
            return
        LOGGER.info(
            f"{'all':>22}{len(self.gt_counts):>11d}  "
            f"MAE(fused)={stats['metrics/MAE']:.3f}  MSE(fused)={stats['metrics/MSE']:.3f}  "
            f"MAE(det)={stats['metrics/MAE_det']:.3f}  MAE(den)={stats['metrics/MAE_den']:.3f}  "
            f"tau={self.tau:g}"
        )

    def get_desc(self):
        if not getattr(self, "density_enabled", True):
            return f"{'Class':>22}{'Images':>11}  {'MAE(detect)':>12}{'RMSE(detect)':>12}"
        return f"{'Class':>22}{'Images':>11}  {'MAE(fused)':>12}{'MSE(fused)':>12}{'MAE(det)':>10}{'MAE(den)':>10}"

    @property
    def metric_keys(self):
        return self.metrics.keys

    # ------------------------------------------------------------------
    def build_dataset(self, img_path, mode="val", batch=None):
        return CrowdDataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch or self.args.batch,
            augment=False,
            hyp=self.args,
            rect=False,
            cache=None,
            single_cls=True,
            stride=32,
            pad=0.5,
            prefix=colorstr("val: "),
            task="detect",
            data=self.data,
            classes=None,
            fraction=getattr(self.args, "fraction", 1.0),
            density_stride=self.data.get("density_stride", 4),
        )

    def get_dataloader(self, dataset_path, batch_size):
        dataset = self.build_dataset(dataset_path, mode="val", batch=batch_size)
        return build_dataloader(
            dataset, batch=batch_size, workers=self.args.workers, shuffle=False, rank=-1, device=self.device
        )

    def plot_val_samples(self, batch, ni):
        return

    def plot_predictions(self, batch, preds, ni):
        return

    def eval_json(self, stats):
        return stats
