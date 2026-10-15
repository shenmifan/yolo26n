# -*- coding: utf-8 -*-
"""
DC-YOLO26 训练入口
==================

示例:
  python train.py --data datasets/sht_a/data.yaml --epochs 300 --batch 16 --imgsz 640 --device 0
  python train.py --data datasets/qnrf/data.yaml --epochs 250 --batch 8 --imgsz 768 --device 0

损失权重等自定义超参可写入 data.yaml（den_gain / cnt_gain / cons_gain /
grid / tau_cons / dgf_tau），训练时自动注入。
"""

from __future__ import annotations

import argparse
from pathlib import Path
import yaml

from crowd.trainer import CrowdTrainer
from crowd.protocol import verify_protocol


def parse_args():
    p = argparse.ArgumentParser(description="DC-YOLO26 crowd counting training")
    p.add_argument("--data", required=True, help="数据集 data.yaml")
    p.add_argument("--model", default="configs/dc-yolo26n.yaml", help="模型配置")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="0", help="cuda 设备号，或 cpu")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0, help="训练随机种子，与数据划分种子独立")
    p.add_argument("--lr0", type=float, default=0.01, help="MuSGD 初始学习率")
    p.add_argument("--optimizer", default="MuSGD", choices=["MuSGD", "SGD", "AdamW"])
    p.add_argument("--pretrained", default="yolo26n.pt", help="检测预训练权重，置空则从头训练")
    p.add_argument("--project", default="runs/crowd")
    p.add_argument("--name", default="dc_yolo26n")
    p.add_argument("--resume", type=str, default="")
    from crowd.experiments import VARIANTS
    p.add_argument("--variant", choices=tuple(VARIANTS), default="full")
    p.add_argument("--patience", type=int, default=100)
    p.add_argument("--cache", action="store_true")
    p.add_argument("--multi-scale", action="store_true", help="启用多尺度训练（密度图同步缩放）")
    return p.parse_args()


def main():
    args = parse_args()
    if not args.resume and (Path(args.project) / args.name / "weights/best.pt").exists():
        raise FileExistsError("此训练目录已有 best.pt；请使用新的 --name，或明确使用 --resume 恢复未完成训练。")
    with open(args.data, encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if data.get("split_protocol"):
        print("Independent dataset splits:", verify_protocol(data))
        if args.resume:
            import torch
            ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
            old_model = ckpt.get("ema") or ckpt.get("model")
            if getattr(old_model, "crowd_cfg", {}).get("protocol_id") != data["protocol_id"]:
                raise ValueError("恢复权重的数据划分不同；请从官方预训练权重开始新的训练。")
            if getattr(old_model, "crowd_cfg", {}).get("variant", "full") != args.variant:
                raise ValueError("恢复权重的实验变体不匹配")
    overrides = dict(
        model=args.model,
        data=args.data,
        epochs=args.epochs,
        patience=args.patience,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        seed=args.seed,
        conf=float(data.get("count_conf", 0.25)),
        iou=float(data.get("count_iou", 0.5)),
        max_det=int(data.get("count_max_det", 2000)),
        nms=True,
        lr0=args.lr0,
        optimizer=args.optimizer,
        project=args.project,
        name=args.name,
        # 为保证密度图严格对齐，关闭无法同步的几何增强
        mosaic=0.0,
        mixup=0.0,
        cutmix=0.0,
        copy_paste=0.0,
        degrees=0.0,
        translate=0.0,
        scale=0.0,
        shear=0.0,
        perspective=0.0,
        close_mosaic=0,
        multi_scale=0.5 if args.multi_scale else 0.0,
        cache=("ram" if args.cache else False),
        task="detect",
        single_cls=True,
        exist_ok=True,
        plots=True,
    )
    if args.resume:
        overrides["resume"] = args.resume
    trainer = CrowdTrainer(overrides=overrides)
    trainer.VARIANT = args.variant
    if args.pretrained and not args.resume:
        trainer.PRETRAINED = args.pretrained
    elif not args.pretrained:
        trainer.PRETRAINED = None
    trainer.train()


if __name__ == "__main__":
    main()
