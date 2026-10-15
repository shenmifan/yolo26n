# -*- coding: utf-8 -*-
"""DC-YOLO26: 密度校准的 YOLO26n 实时人群计数。

导入本包即完成自定义模块（EMA）向 ultralytics 模型解析器的注册。
"""

from ultralytics.nn import tasks

from .modules import EMA, DensityHead

# 让 parse_model 的 globals() 能在 YAML 中按名字找到 EMA
if not hasattr(tasks, "EMA"):
    tasks.EMA = EMA

__all__ = ["EMA", "DensityHead"]
