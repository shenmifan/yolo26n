# -*- coding: utf-8 -*-
"""
DC-YOLO26 自定义模块
====================

- EMA: Efficient Multi-Scale Attention（Ouyang et al., ICASSP 2023），
  纯卷积的轻量通道-空间注意力，几乎不增加参数，适合 nano 级模型。
  采用延迟初始化（首次前向时按输入通道建层），以便直接写进 YOLO YAML
  （ultralytics 的 parse_model 对未知模块不传通道参数）。

- DensityHead: 轻量密度计数分支，融合 P2(stride 4) 与 P3(stride 8) 特征，
  输出单通道密度图（Softplus 保证非负），积分即人数。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.conv import Conv


class EMA(nn.Module):
    """Efficient Multi-Scale Attention.

    Args:
        factor: 通道分组数，要求 channels // factor > 0。
    """

    def __init__(self, factor: int = 8):
        super().__init__()
        self.factor = int(factor)
        self._built = False

    def _build(self, channels: int, device=None, dtype=None):
        g = min(self.factor, channels)
        while channels % g != 0:
            g -= 1
        self.groups = max(g, 1)
        cg = channels // self.groups
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d(1)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(cg, cg)
        self.conv1x1 = nn.Conv2d(cg, cg, kernel_size=1)
        self.conv3x3 = nn.Conv2d(cg, cg, kernel_size=3, padding=1)
        self.to(device=device, dtype=dtype)
        self._built = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._built:
            self._build(x.shape[1], x.device, x.dtype)
        b, c, h, w = x.shape
        g = self.groups
        cg = c // g
        x0 = x.reshape(b * g, cg, h, w)
        x_h = self.pool_h(x0)  # (bg, cg, h, 1)
        x_w = self.pool_w(x0).transpose(-1, -2)  # (bg, cg, w, 1)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.softmax(self.conv1x1(y) + self.conv3x3(y))
        att_h, att_w = torch.split(y, [h, w], dim=2)
        att_w = att_w.transpose(-1, -2)
        z = self.gn(x0 * att_h * att_w)
        z = self.agp(z).reshape(b * g, -1, 1, 1)
        z = z.expand(-1, cg, -1, -1).reshape(b, g, 1, cg, 1).permute(0, 2, 1, 3, 4)
        z = z.reshape(b, g * cg, 1, 1)
        return x * z


class DensityHead(nn.Module):
    """轻量密度计数头。

    Args:
        c_p2: P2(stride 4) 特征通道数。
        c_p3: P3(stride 8) 特征通道数。
        hidden: 融合后的隐藏通道数（nano 模型取 32，控制额外开销）。
    """

    def __init__(self, c_p2: int, c_p3: int, hidden: int = 32):
        super().__init__()
        self.lat2 = Conv(c_p2, hidden, 1)
        self.lat3 = Conv(c_p3, hidden, 1)
        self.fuse = nn.Sequential(Conv(hidden, hidden, 3), Conv(hidden, hidden, 3))
        self.head = nn.Conv2d(hidden, 1, kernel_size=1)
        self.act = nn.Softplus()

    def forward(self, p2: torch.Tensor, p3: torch.Tensor) -> torch.Tensor:
        high = self.lat2(p2)
        low = F.interpolate(self.lat3(p3), size=p2.shape[-2:], mode="bilinear", align_corners=False)
        x = self.fuse(high + low)
        return self.act(self.head(x))  # (B, 1, H/4, W/4)，非负
