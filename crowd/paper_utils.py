"""Shared frozen-protocol data and metadata for the supplemental experiments."""
import json
import os
import platform
import random
import subprocess
import sys
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path

import numpy as np
import torch
import ultralytics
import yaml
from ultralytics.cfg import get_cfg

from .dataset import CrowdDataset
from .protocol import data_path, verify_protocol


def load_data(path):
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    data["path"] = str(Path(data["path"]).resolve())
    verify_protocol(data)
    return data


def make_dataset(data, split, imgsz=640, augment=False):
    cfg = get_cfg(overrides=dict(imgsz=imgsz, task="detect", single_cls=True, rect=False,
                                mosaic=0., mixup=0., cutmix=0., copy_paste=0., plots=False))
    return CrowdDataset(img_path=str(data_path(data, data[split])), imgsz=imgsz,
                        batch_size=1, augment=augment, hyp=cfg, rect=False, single_cls=True,
                        stride=32, pad=0., prefix=f"{split}: ", task="detect", data=data,
                        density_stride=data.get("density_stride", 4))


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_device(value):
    device = torch.device("cpu" if value == "cpu" else f"cuda:{value}")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use --device cpu explicitly")
    return device


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def environment(device=None):
    record = dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
                  cpu=platform.processor(), torch=torch.__version__, ultralytics=ultralytics.__version__,
                  cuda_build=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                  cpu_threads=torch.get_num_threads(),
                  tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
                  tf32_cudnn=torch.backends.cudnn.allow_tf32)
    import cv2
    record.update(numpy=np.__version__, opencv=cv2.__version__)
    try:
        record["torchvision"] = version("torchvision")
    except PackageNotFoundError:
        record["torchvision"] = None
    if device is not None:
        device = torch.device(device)
        record.update(device=str(device), device_name=torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU")
        if device.type == "cuda":
            record["gpu_memory_bytes"] = torch.cuda.get_device_properties(device).total_memory
    try:
        record["nvidia_smi"] = subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"], text=True, timeout=10).strip()
    except (OSError, subprocess.SubprocessError):
        record["nvidia_smi"] = None
    try:
        import psutil
        record["ram_bytes"] = psutil.virtual_memory().total
    except ImportError:
        record["ram_bytes"] = None
    return record
