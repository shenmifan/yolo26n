"""Batch-one FP32 counting used by runtime and qualitative exports."""
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics.data.augment import LetterBox

from .baselines import baseline_input, make_baseline
from .fusion import dgf_fuse
from .protocol import sha256_file
from .validator import counting_postprocess


def prepare_image(image, imgsz):
    """Match YOLODataset.load_image + CrowdLetterBox + Format, excluding disk I/O."""
    h, w = image.shape[:2]
    ratio = imgsz / max(h, w)
    if ratio != 1:
        image = cv2.resize(image, (min(math.ceil(w * ratio), imgsz), min(math.ceil(h * ratio), imgsz)), interpolation=cv2.INTER_LINEAR)
    canvas = LetterBox((imgsz, imgsz), auto=False, scaleup=False)(image=image)
    tensor = torch.from_numpy(np.ascontiguousarray(canvas[..., ::-1].transpose(2, 0, 1))).unsqueeze(0)
    return tensor, canvas


def load_counter(weights, parameters, device):
    if parameters["weights_sha256"] != sha256_file(weights):
        raise ValueError("Weights do not match the frozen validation parameters")
    if parameters["source_split"] != "val" or parameters["diagnostic"]:
        raise ValueError("Use parameters frozen on independent validation data")
    checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
    if "method" in checkpoint:
        method = checkpoint["method"]
        model = make_baseline(method)
        model.load_state_dict(checkpoint["state_dict"])
        protocol_id = checkpoint["protocol_id"]
    else:
        from val import load_model
        model = load_model(weights, device)
        method = getattr(model, "crowd_cfg", {}).get("variant", "full")
        protocol_id = getattr(model, "crowd_cfg", {}).get("protocol_id")
    if protocol_id != parameters["protocol_id"]:
        raise ValueError("Checkpoint protocol differs from frozen parameters")
    return model.float().to(device).eval(), method


def input_tensor(images, method, device):
    images = images.to(device)
    return baseline_input(images, method) if method in {"mcnn", "csrnet"} else images.float() / 255.


def postprocess_count(output, method, settings, index=0, include_map=True):
    if method in {"mcnn", "csrnet"}:
        count = float(output[index].sum().item())
        density = output[index, 0].float().cpu().numpy() if include_map else None
        return dict(density=count), dict(density=density, boxes=np.empty((0, 4)))
    det, density = output
    decoded = det[0] if isinstance(det, (tuple, list)) else det
    boxes = counting_postprocess(decoded[index:index + 1], settings["conf"], settings["iou"], settings["max_det"], nc=1, end2end=False)[0][:, :4].float().cpu().numpy()
    if density is None:
        return dict(detect=float(len(boxes))), dict(density=None, boxes=boxes)
    density = density[index, 0].float().cpu().numpy()
    result = dgf_fuse(density, boxes, settings["imgsz"], settings["grid"], settings["tau"])
    return dict(detect=result["count_detect"], density=result["count_density"], fused=result["count_fused"]), dict(density=density, boxes=boxes, **result)
