# -*- coding: utf-8 -*-
"""
DC-YOLO26 端到端冒烟测试（无需真实数据集）
=========================================

运行: python tests/test_smoke.py
覆盖：密度图/伪框生成、模型构建、训练前向+损失、推理前向、DGF 融合、
      合成数据集上 CrowdDataset 读取。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_density_and_boxes():
    from crowd.density import build_density_map, build_pseudo_boxes
    rng = np.random.default_rng(0)
    points = rng.uniform(20, 220, size=(120, 2))
    den = build_density_map(points, 256, 256, beta=0.3)
    boxes = build_pseudo_boxes(points, 256, 256)
    assert den.shape == (256, 256)
    assert boxes.shape == (120, 5)
    assert abs(den.sum() - 120) < 1.0, f"密度积分应约等于点数，实际 {den.sum()}"
    assert boxes[:, 1:].min() >= 0 and boxes[:, 1:].max() <= 1
    print(f"[OK] 密度图积分={den.sum():.2f}（真值 120），伪框 {boxes.shape[0]} 个")


def test_model_and_loss():
    from crowd.model import CrowdDetectionModel

    model = CrowdDetectionModel(str(ROOT / "configs" / "dc-yolo26n.yaml"), nc=1, verbose=False)
    model.args = SimpleNamespace(
        box=7.5, cls=0.5, dfl=1.5, l1=1.0, reg_max=1, nc=1,
        den_gain=1.0, cnt_gain=0.1, cons_gain=0.5, grid=16, tau_cons=3.0,
    )
    n_params = sum(p.numel() for p in model.parameters())
    imgsz = 320  # 测试用小尺寸
    model.train()
    x = torch.randn(2, 3, imgsz, imgsz)
    det, density = model(x)
    assert isinstance(det, dict) and "one2many" in det and "one2one" in det
    assert density.shape == (2, 1, imgsz // 4, imgsz // 4), density.shape
    assert float(density.min().detach()) >= 0.0

    # 合成训练 batch：伪框（单类，归一化 xywh）
    bboxes = torch.tensor([[0.02, 0.02, 0.02, 0.02], [0.10, 0.10, 0.02, 0.02],
                           [0.50, 0.50, 0.03, 0.03]], dtype=torch.float32)
    batch_idx = torch.tensor([0, 0, 1])
    cls = torch.zeros((3, 1))
    gt_density = torch.zeros(2, imgsz // 4, imgsz // 4)
    gt_density[0, 2, 2] = 1.0
    gt_density[0, 10, 10] = 1.0
    gt_density[1, 20, 20] = 1.0
    batch = dict(img=x, batch_idx=batch_idx, cls=cls, bboxes=bboxes,
                 density=gt_density, imgsz=torch.tensor([imgsz, imgsz]))
    criterion = model.init_criterion()
    loss, items = criterion((det, density), batch)
    assert torch.isfinite(loss).all(), loss
    loss = loss.sum()
    for k in ("box_loss", "cls_loss", "den_loss", "cnt_loss", "cons_loss"):
        assert k in items, f"缺少损失项 {k}"
    loss.backward()
    grad_head = sum(1 for p in model.density_head.parameters() if p.grad is not None)
    assert grad_head > 0, "密度头没有梯度"
    print(f"[OK] 训练前向+损失: loss={loss.item():.4f}，参数量={n_params/1e6:.3f}M")
    print("     损失项:", {k: round(float(v), 4) for k, v in items.items()})

    # 推理前向
    model.eval()
    with torch.no_grad():
        out, den = model(x)
    assert isinstance(out, tuple) and len(out) == 2
    print(f"[OK] 推理前向: det={out[0].shape}, density={den.shape}")


def test_dgf():
    from crowd.fusion import dgf_fuse
    den = np.zeros((80, 80), dtype=np.float32)
    den[0:20, 0:20] = 5.0 / 400  # 高密度网格（左上）
    boxes = np.array([[400, 400, 420, 420], [500, 500, 520, 520]], dtype=np.float32)
    r = dgf_fuse(den, boxes, imgsz=320, grid=16, tau=2.0)
    assert r["count_density"] > 0
    assert r["count_detect"] == 2.0
    assert r["count_fused"] >= 2.0
    print(f"[OK] DGF: fused={r['count_fused']:.2f} det={r['count_detect']:.1f} den={r['count_density']:.2f}")


def test_dataset():
    """合成一个微型 YOLO 数据集，验证 CrowdDataset 密度图同步。"""
    import cv2
    from crowd.dataset import CrowdDataset
    tmp = ROOT / "tests" / "_tmp_data"
    for split in ("train", "val"):
        (tmp / "images" / split).mkdir(parents=True, exist_ok=True)
        (tmp / "labels" / split).mkdir(parents=True, exist_ok=True)
        (tmp / "density" / split).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1)
    for split, n in (("train", 2), ("val", 2)):
        for i in range(n):
            stem = f"syn_{split}_{i}"
            img = rng.integers(0, 255, (256, 256, 3), dtype=np.uint8)
            cv2.imwrite(str(tmp / "images" / split / f"{stem}.jpg"), img)
            pts = rng.uniform(20, 236, size=(15, 2))
            from crowd.density import build_density_map, build_pseudo_boxes
            den = build_density_map(pts, 256, 256)
            np.save(tmp / "density" / split / f"{stem}.npy", den)
            boxes = build_pseudo_boxes(pts, 256, 256)
            with open(tmp / "labels" / split / f"{stem}.txt", "w") as f:
                for b in boxes:
                    f.write(f"{int(b[0])} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}\n")

    hyp = SimpleNamespace(
        imgsz=320, fliplr=0.5, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        mosaic=0.0, mixup=0.0, cutmix=0.0, copy_paste=0.0, bgr=0.0,
        mask_ratio=0.0, overlap_mask=True, max_gt=0, use_obb=False,
        multi_modal=False, use_segments=False, use_keypoints=False,
        rect=False, single_cls=True, task="detect",
    )
    data = dict(nc=1, names=["head"], density_stride=4)
    ds = CrowdDataset(
        img_path=str(tmp / "images" / "train"), imgsz=320, batch_size=2,
        augment=True, hyp=hyp, rect=False, stride=32, pad=0.0,
        prefix="train: ", task="detect", data=data, density_stride=4,
    )
    sample = ds[0]
    assert sample["img"].shape == (3, 320, 320)
    assert "density" in sample and sample["density"].shape == (80, 80)
    batch = CrowdDataset.collate_fn([ds[0], ds[1]])
    assert batch["density"].shape == (2, 80, 80)
    assert batch["img"].shape == (2, 3, 320, 320)
    print(f"[OK] CrowdDataset: img={tuple(batch['img'].shape)} density={tuple(batch['density'].shape)}")


if __name__ == "__main__":
    test_density_and_boxes()
    test_dgf()
    test_model_and_loss()
    test_dataset()
    print("\n全部冒烟测试通过 ✓")
