# -*- coding: utf-8 -*-
"""
DC-YOLO26 模型
==============

CrowdDetectionModel 在 YOLO26 检测模型上挂载密度计数分支：

- 检测路：YOLO26 原生双头（one-to-many + one-to-one），P2-P5 四尺度，
  训练时输出 dict，推理时输出 (decoded, raw)；
- 密度路：DensityHead 取 Detect 输入中的 P2/P3 特征，输出 stride 4 密度图。

训练前向返回 (det_preds, density_pred)；推理前向返回 ((det_decoded, det_raw), density_pred)。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ultralytics.nn.modules.head import Detect
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.torch_utils import initialize_weights

from . import EMA  # noqa: F401  确保 EMA 已注册
from .modules import DensityHead


def _detect_in_channels(detect: Detect) -> list[int]:
    """从 Detect 的 cv2 ModuleList 推断各检测尺度的输入通道数。"""
    chs = []
    for seq in detect.cv2:
        in_ch = None
        for m in seq.modules():
            if isinstance(m, nn.Conv2d):
                in_ch = m.in_channels
                break
        chs.append(in_ch)
    return chs


class CrowdDetectionModel(DetectionModel):
    """带密度计数头的 YOLO26 检测模型。"""

    def __init__(
        self,
        cfg="configs/dc-yolo26n.yaml",
        ch: int = 3,
        nc: int = 1,
        density_hidden: int = 32,
        verbose: bool = False,
        density_enabled: bool = True,
    ):
        self._density_ready = False
        self.density_head = None
        self.density_hidden = density_hidden
        # 父类 __init__ 会用一次 dummy 前向计算 stride，此时密度头尚未挂载，
        # _predict_once 自动回退到原生路径。
        super().__init__(cfg, ch=ch, nc=nc, verbose=verbose)

        detect = self.model[-1]
        assert isinstance(detect, Detect), "配置文件最后一层必须是 Detect"
        chs = _detect_in_channels(detect)
        assert len(chs) >= 2, "DC-YOLO26 需要至少 P2/P3 两个检测尺度"
        self._density_output_scale = int(self.stride[0]) // 4
        # chs 顺序与 YAML Detect 的 from 一致：[P2, P3, P4, P5]
        if density_enabled:
            self.density_head = DensityHead(chs[0], chs[1], hidden=density_hidden)
            initialize_weights(self.density_head)
            self.density_head = self.density_head.to(next(self.parameters()).device)
        self._density_ready = True

    def init_criterion(self):
        from .loss import CrowdCriterion

        return CrowdCriterion(self)

    def _predict_once(self, x, profile=False, embed=None):
        """前向传播，在 Detect 处截获 P2/P3 特征并计算密度图。"""
        # 强制全精度float32，解决类型不匹配报错
        self.model = self.model.float()
        x = x.float()
        if not self._density_ready:
            return super()._predict_once(x, profile=profile, embed=embed)

        y = []  # saved outputs
        for m in self.model:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            if isinstance(m, Detect):
                det_preds = m(x)
                density_pred = self.density_head(x[0], x[1]) if self.density_head is not None else None
                output_scale = getattr(self, "_density_output_scale", 1)
                if density_pred is not None and output_scale != 1:
                    from .density import resize_density_preserve_count_torch
                    # P3/P4 decoder ablation: use the same stride-4 counting canvas.
                    target = (x[0].shape[-2] * output_scale, x[0].shape[-1] * output_scale)
                    density_pred = resize_density_preserve_count_torch(density_pred, target)
                return det_preds, density_pred
            x = m(x)
            y.append(x if m.i in self.save else None)
        return x

    def predict(self, x, profile=False, augment=False, embed=None):
        """返回 (检测输出, 密度图)。"""
        if augment:
            return self._predict_augment(x)
        return self._predict_once(x, profile=profile, embed=embed)
