# -*- coding: utf-8 -*-
"""
DC-YOLO26 训练器
================

- 数据集：CrowdDataset（检测伪框 + 密度图）
- 模型：CrowdDetectionModel（YOLO26n P2-P5 + EMA + 密度头）
- 验证：CrowdValidator（MAE / MSE，DGF 融合）
- 多尺度训练时同步插值密度图并保持积分守恒
"""

from __future__ import annotations

from copy import copy
from typing import Any

import torch
from ultralytics.data import build_dataloader
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import LOGGER, RANK
from ultralytics.utils.torch_utils import torch_distributed_zero_first, unwrap_model

from .dataset import CrowdDataset
from .validator import CrowdValidator
from .density import resize_density_preserve_count_torch


class CrowdTrainer(DetectionTrainer):
    """人群计数训练器。"""

    # 从 YOLO26n 官方检测权重迁移 backbone（P2 头与密度头随机初始化）
    PRETRAINED = "yolo26n.pt"
    VARIANT = "full"

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
        gs = max(int(unwrap_model(self.model).stride.max()), 32)
        return CrowdDataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=mode == "train",
            hyp=self.args,
            rect=False,  # 方形 letterbox，保证密度图可按 batch 堆叠
            cache=self.args.cache or None,
            single_cls=self.args.single_cls or False,
            stride=gs,
            pad=0.0 if mode == "train" else 0.5,
            prefix=f"{mode}: ",
            task=self.args.task,
            classes=self.args.classes,
            data=self.data,
            fraction=self.args.fraction if mode == "train" else 1.0,
            density_stride=self.data.get("density_stride", 4),
        )

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        assert mode in {"train", "val"}
        with torch_distributed_zero_first(rank):
            dataset = self.build_dataset(dataset_path, mode, batch_size)
        shuffle = mode == "train"
        return build_dataloader(
            dataset,
            batch=batch_size,
            workers=self.args.workers if mode == "train" else self.args.workers * 2,
            shuffle=shuffle,
            rank=rank,
            drop_last=self.args.compile and mode == "train",
            device=self.device,
        )

    def get_model(self, cfg=None, weights=None, verbose=True):
        from .model import CrowdDetectionModel
        from .experiments import model_config, VARIANTS

        model = CrowdDetectionModel(
            model_config(self.VARIANT, cfg),
            nc=self.data["nc"],
            ch=self.data["channels"],
            density_hidden=int(self.data.get("density_hidden", 32)),
            verbose=verbose and RANK == -1,
            density_enabled=VARIANTS[self.VARIANT]["density"],
        )
        if weights:
            model.load(weights)
        elif self.PRETRAINED:
            from ultralytics.nn.tasks import torch_safe_load

            LOGGER.info(f"迁移官方预训练权重: {self.PRETRAINED}")
            ckpt, _ = torch_safe_load(self.PRETRAINED)
            model.load(ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt)
        return model

    def set_model_attributes(self):
        super().set_model_attributes()
        # 自定义损失/融合超参挂在 model.crowd_cfg（不能放进 args，否则 get_cfg 校验失败）
        if self.model is not None:
            from .experiments import crowd_settings
            self.model.crowd_cfg = crowd_settings(self.data, self.VARIANT)
            self.model.crowd_cfg["training_seed"] = self.args.seed

    def get_validator(self):
        # YOLO26（reg_max=1）检测项为 box/cls/l1，追加密度、计数、一致性三项
        self.loss_names = ("box_loss", "cls_loss", "l1_loss", "den_loss", "cnt_loss", "cons_loss")
        args = copy(self.args)
        args.conf = float(self.data.get("count_conf", 0.25))
        args.iou = float(self.data.get("count_iou", 0.5))
        args.max_det = int(self.data.get("count_max_det", 2000))
        args.nms = True  # explicitly use one-to-many + NMS throughout validation
        return CrowdValidator(self.test_loader, save_dir=self.save_dir, args=args, _callbacks=self.callbacks)

    def preprocess_batch(self, batch):
        batch = super().preprocess_batch(batch)
        if "density" in batch:
            density = batch["density"].unsqueeze(1).float()
            th, tw = batch["img"].shape[-2] // 4, batch["img"].shape[-1] // 4
            if density.shape[-2:] != (th, tw):
                density = resize_density_preserve_count_torch(density, (th, tw))
            batch["density"] = density.squeeze(1)
        return batch

    def progress_string(self):
        return ("\n" + "%11s" * (4 + len(self.loss_names))) % (
            "Epoch",
            "GPU_mem",
            *self.loss_names,
            "Counts",
            "Size",
        )

    @torch.no_grad()
    def _validate_bare(self, model):
        """用裸模型（不经过 AutoBackend）在验证集上计算 MAE/MSE。"""
        validator = self.validator
        validator.training = False
        validator.data = self.data
        validator.device = self.device
        model = model.float().to(self.device).eval()
        if hasattr(model, "end2end"):
            model.end2end = False
        validator.init_metrics(model)
        for batch in self.test_loader:
            batch = validator.preprocess(batch)
            preds = model(batch["img"])
            preds = validator.postprocess(preds)
            validator.update_metrics(preds, batch)
        stats = validator.get_stats()
        validator.print_results()
        return stats

    def final_eval(self):
        """重写：剥离优化器后用裸模型在 best.pt 上复评（绕过 AutoBackend 对 tuple 输出的不兼容）。"""
        from ultralytics.utils.torch_utils import strip_optimizer

        if RANK not in {-1, 0}:
            return
        if self.last.exists():
            ckpt = strip_optimizer(self.last)
        else:
            ckpt = {}
        if self.best.exists():
            strip_optimizer(self.best, updates={"train_results": ckpt.get("train_results")})
            LOGGER.info(f"\nValidating {self.best}...")
            ckpt = torch.load(self.best, map_location=self.device, weights_only=False)
            model = ckpt.get("ema") or ckpt["model"]
            stats = self._validate_bare(model)
            self.metrics = {k: v for k, v in stats.items() if k != "fitness"}
            self.fitness = stats["fitness"]
