# -*- coding: utf-8 -*-
"""
DC-YOLO26 单图/文件夹推理与可视化
=================================

输出：检测框（稀疏区）、密度热图（密集区）、DGF 融合计数。

示例:
  python infer.py --weights best.pt --source demo.jpg --imgsz 640
  python infer.py --weights best.pt --source images/ --save-dir runs/infer
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics.data.augment import LetterBox
from ultralytics.utils import nms

from crowd.fusion import dgf_fuse


def load_model(weights: str, device: str):
    ckpt = torch.load(weights, map_location="cpu", weights_only=False)
    model = (ckpt.get("ema") or ckpt["model"]).float().to(device)
    if hasattr(model, "end2end"):
        model.end2end = False
    model.eval()
    return model


def preprocess(img_bgr, imgsz, device):
    lb = LetterBox((imgsz, imgsz), auto=False, scaleup=False)
    canvas = lb(image=img_bgr)
    x = canvas[:, :, ::-1].transpose(2, 0, 1)  # BGR -> RGB, HWC -> CHW
    x = np.ascontiguousarray(x)
    t = torch.from_numpy(x).float().unsqueeze(0).to(device) / 255.0
    return t, canvas, lb


def visualize(canvas, boxes, density, result, imgsz, out_path):
    vis = canvas.copy()
    for x1, y1, x2, y2, conf, _ in boxes:
        cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 200, 0), 1)
    # 密度热图放大到画布尺寸
    den = cv2.resize(density, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
    den_norm = (den / max(den.max(), 1e-6) * 255).astype(np.uint8)
    heat = cv2.applyColorMap(den_norm, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(canvas, 0.55, heat, 0.45, 0)
    panel = np.concatenate([vis, overlay], axis=1)
    txt = (
        f"Fused: {result['count_fused']:.0f}  "
        f"Det: {result['count_detect']:.0f}  Den: {result['count_density']:.0f}"
    )
    cv2.putText(panel, txt, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    cv2.imwrite(str(out_path), panel)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--source", required=True, help="图片路径或文件夹")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--grid", type=int, default=16)
    p.add_argument("--tau", type=float, default=2.0)
    p.add_argument("--max-det", type=int, default=2000)
    p.add_argument("--device", default="0")
    p.add_argument("--save-dir", default="runs/infer")
    args = p.parse_args()

    device = "cpu" if args.device == "cpu" else f"cuda:{args.device}"
    if device != "cpu" and not torch.cuda.is_available():
        device = "cpu"
    model = load_model(args.weights, device)

    src = Path(args.source)
    files = [src] if src.is_file() else sorted(
        q for q in src.iterdir() if q.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for f in files:
            img = cv2.imread(str(f))
            if img is None:
                continue
            x, canvas, _ = preprocess(img, args.imgsz, device)
            det, density = model(x)
            det_y = det[0] if isinstance(det, (tuple, list)) else det
            preds = nms.non_max_suppression(
                det_y, args.conf, args.iou, nc=0, multi_label=True,
                agnostic=True, max_det=args.max_det, end2end=False,
            )[0].cpu().numpy()
            den = density[0, 0].cpu().numpy()
            result = dgf_fuse(den, preds[:, :4] if len(preds) else np.zeros((0, 4)),
                              args.imgsz, grid=args.grid, tau=args.tau)
            print(
                f"{f.name}: fused={result['count_fused']:.1f} "
                f"det={result['count_detect']:.1f} density={result['count_density']:.1f}"
            )
            visualize(canvas, preds, den, result, args.imgsz, save_dir / f"{f.stem}_dc.jpg")

    print(f"\n可视化结果已保存到 {save_dir}")


if __name__ == "__main__":
    main()
