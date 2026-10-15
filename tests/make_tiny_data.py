# -*- coding: utf-8 -*-
"""生成微型合成数据集（用于集成测试），输出到 tests/_tmp_data/。"""

import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crowd.density import build_density_map, build_pseudo_boxes  # noqa: E402


def main():
    root = Path(__file__).resolve().parent / "_tmp_data"
    rng = np.random.default_rng(1)
    for split in ("train", "val"):
        n = 4 if split == "train" else 2
        for i in range(n):
            stem = f"syn_{split}_{i}"
            img = rng.integers(0, 255, (256, 256, 3), dtype=np.uint8)
            cv2.imwrite(str(root / "images" / split / f"{stem}.jpg"), img)
            pts = rng.uniform(20, 236, size=(15, 2))
            np.save(root / "density" / split / f"{stem}.npy", build_density_map(pts, 256, 256))
            boxes = build_pseudo_boxes(pts, 256, 256)
            with open(root / "labels" / split / f"{stem}.txt", "w") as f:
                for b in boxes:
                    f.write(f"{int(b[0])} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}\n")

    cfg = {
        "path": str(root).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "nc": 1,
        "names": ["head"],
        "density_dir": "density",
        "density_stride": 4,
        "den_gain": 1.0,
        "cnt_gain": 0.1,
        "cons_gain": 0.5,
        "grid": 16,
        "tau_cons": 3.0,
        "dgf_tau": 2.0,
    }
    with open(root / "data.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print("微型数据集已生成:", root)


if __name__ == "__main__":
    main()
