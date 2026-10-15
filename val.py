"""Evaluate three counting paths with one checkpoint and one protocol.

Use --split val for calibration, then --split test --params .../parameters.json.
Old checkpoints must use --diagnostic; they were not trained on the new split.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
import ultralytics
from tqdm import tqdm
from ultralytics.cfg import get_cfg
from ultralytics.data import build_dataloader
from ultralytics.utils.checks import check_imgsz

from crowd.dataset import CrowdDataset
from crowd.fusion import dgf_fuse
from crowd.protocol import data_path, sha256_file, verify_protocol
from crowd.validator import CrowdValidator


def load_model(weights, device):
    checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
    model = checkpoint.get("ema") or checkpoint["model"]
    model = model.float().to(device).eval()
    model.end2end = False
    if getattr(model.model[-1], "end2end", True):
        raise RuntimeError("无法切换到 one-to-many 检测头")
    return model


def validate_request(args, data, model):
    if args.sweep_tau and args.split != "val":
        raise ValueError("禁止在测试集扫描阈值；请在独立验证集选择参数后固定测试。")
    if args.sweep_tau and args.diagnostic:
        raise ValueError("旧权重诊断不能用于新验证集调参；请先按新划分重新训练。")
    if data.get("split_protocol"):
        verify_protocol(data)
    elif not args.diagnostic:
        raise ValueError("正式评估需要独立 train/val/test 协议和原始点数清单。")
    trained_protocol = getattr(model, "crowd_cfg", {}).get("protocol_id")
    if not args.diagnostic and trained_protocol != data.get("protocol_id"):
        raise ValueError("权重未按当前独立划分训练。旧权重请加 --diagnostic；正式实验请重新训练。")
    if args.params and args.sweep_tau:
        raise ValueError("--params 固定参数与 --sweep-tau 不可同时使用。")
    if args.sweep_tau and hasattr(model, "density_head") and model.density_head is None:
        raise ValueError("检测基线没有密度分支，不应扫描 DGF 阈值")


def collect(model, data, args, device):
    # Keyword overrides preserve all required Ultralytics defaults.
    cfg = get_cfg(overrides=dict(
        model=args.weights, data=args.data, imgsz=args.imgsz, batch=args.batch,
        device=device, workers=args.workers, conf=args.conf, iou=args.iou,
        max_det=args.max_det, task="detect", single_cls=True, rect=False, plots=False,
        nms=True, quantize=None,
    ))
    dataset = CrowdDataset(
        img_path=str(data_path(data, data[args.split])), imgsz=args.imgsz,
        batch_size=args.batch, augment=False, hyp=cfg, rect=False, single_cls=True,
        stride=32, pad=0.0, prefix=f"{args.split}: ", task="detect", data=data,
        fraction=1.0, density_stride=data.get("density_stride", 4),
    )
    loader = build_dataloader(dataset, batch=args.batch, workers=args.workers,
                              shuffle=False, rank=-1, device=torch.device(device))
    validator = CrowdValidator(loader, save_dir=args.save_dir, args=cfg)
    validator.data, validator.device = data, torch.device(device)
    validator.init_metrics(model)
    records = []
    with torch.inference_mode():
        for batch in tqdm(loader, desc=f"Evaluate {args.split}"):
            batch = validator.preprocess(batch)
            outputs = validator.postprocess(model(batch["img"]))
            height, width = batch["img"].shape[-2:]
            if height != width:
                raise ValueError("当前 DGF 需要正方形 letterbox 画布")
            for index, (pred, density) in enumerate(outputs):
                records.append(dict(
                    image=str(batch["im_file"][index]), gt=int(batch["gt_count"][index].item()),
                    label_count=int((batch["batch_idx"] == index).sum().item()),
                    target_density_sum=float(batch["density"][index].sum().item()),
                    density=density[0].float().cpu().numpy() if density is not None else None,
                    boxes=pred[:, :4].float().cpu().numpy(), imgsz=width,
                ))
    if len(records) != len(dataset):
        raise RuntimeError("评估图片数量与清单不一致")
    return records


def mae_rmse(gt, pred):
    difference = np.asarray(pred, dtype=np.float64) - np.asarray(gt, dtype=np.float64)
    if difference.size == 0 or not np.isfinite(difference).all():
        raise ValueError("计数结果为空或包含 NaN/Inf")
    return dict(MAE=float(np.abs(difference).mean()), RMSE=float(np.sqrt(np.square(difference).mean())),
                bias=float(difference.mean()))


def evaluate(records, grid, tau):
    rows = []
    for record in records:
        result = dgf_fuse(record["density"], record["boxes"], record["imgsz"], grid=grid, tau=tau) if record["density"] is not None else dict(count_detect=len(record["boxes"]))
        row = {key: record[key] for key in ("image", "gt", "label_count", "target_density_sum")}
        for mode, result_key in (("detect", "count_detect"), ("density", "count_density"), ("fused", "count_fused")):
            if result_key not in result:
                continue
            row[mode] = result[result_key]
            row[f"error_{mode}"] = row[mode] - row["gt"]
            row[f"abs_error_{mode}"] = abs(row[f"error_{mode}"])
        rows.append(row)
    metrics = {mode: mae_rmse([row["gt"] for row in rows], [row[mode] for row in rows])
               for mode in ("detect", "density", "fused") if mode in rows[0]}
    return metrics, rows


def write_csv(path, rows):
    with open(path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def create_save_dir(path, split):
    """Keep previous results; repeated tests reserve a fresh numbered directory."""
    path = Path(path)
    candidate, index = path, 2
    while True:
        try:
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        except FileExistsError:
            if split != "test":
                raise FileExistsError(
                    f"校准结果目录已存在：{path}。可使用已有 parameters.json；"
                    "若需要重新校准，请指定新的 --save-dir，并将测试的 --params "
                    "改为本次生成的参数文件。"
                ) from None
            if not candidate.is_dir():
                raise
            candidate = path.with_name(f"{path.name}_{index}")
            index += 1


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--conf", type=float, default=None)
    parser.add_argument("--iou", type=float, default=None)
    parser.add_argument("--max-det", type=int, default=None)
    parser.add_argument("--grid", type=int, default=None)
    parser.add_argument("--tau", type=float, default=None)
    parser.add_argument("--sweep-tau", action="store_true")
    parser.add_argument("--params", help="验证集生成的 parameters.json，固定全部评估参数")
    parser.add_argument("--diagnostic", action="store_true", help="旧权重探索性诊断，不作为正式独立测试")
    parser.add_argument("--save-dir", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.data, encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    data["path"] = str(Path(data["path"]).resolve())
    defaults = dict(imgsz=640, conf=data.get("count_conf", 0.25), iou=data.get("count_iou", 0.5),
                    max_det=data.get("count_max_det", 2000), grid=data.get("grid", 16), tau=data.get("dgf_tau", 2.0))
    weights_hash = sha256_file(args.weights)
    if args.params:
        frozen = json.loads(Path(args.params).read_text(encoding="utf-8"))
        if frozen["weights_sha256"] != weights_hash or frozen["protocol_id"] != data.get("protocol_id"):
            raise ValueError("固定参数文件的权重或数据划分不匹配")
        if not args.diagnostic and (frozen["source_split"] != "val" or frozen["diagnostic"]):
            raise ValueError("正式测试只能使用独立验证集产生的参数文件")
        defaults = frozen["settings"]
        for key in defaults:
            explicit = getattr(args, key)
            if explicit is not None and explicit != defaults[key]:
                raise ValueError(f"--params 已固定 {key}，不能覆盖")
    for key, value in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    args.imgsz = check_imgsz(args.imgsz, stride=32)
    if args.grid <= 0 or args.imgsz % (4 * args.grid) != 0:
        raise ValueError("imgsz 必须能被 4*grid 整除，防止网格聚合丢弃边缘密度")
    device = "cpu" if args.device == "cpu" else f"cuda:{args.device}"
    if device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用；使用 --device cpu 明确切换")
    model = load_model(args.weights, device)
    validate_request(args, data, model)
    requested_dir = args.save_dir or Path("runs/eval") / datetime.now().strftime(f"{args.split}_%Y%m%d_%H%M%S")
    args.save_dir = create_save_dir(requested_dir, args.split)
    if args.save_dir != requested_dir:
        print(f"已有测试结果保留；本次结果目录：{args.save_dir.resolve()}")
    print(f"Purpose: {'exploratory diagnostic' if args.diagnostic else 'independent ' + args.split}")
    print(f"GT: {'original point annotations' if data.get('count_manifest') else 'legacy pseudo-box counts'}")
    print(f"one-to-many + NMS | conf={args.conf} iou={args.iou} max_det={args.max_det} grid={args.grid} tau={args.tau}")
    records = collect(model, data, args, device)
    if args.sweep_tau:
        sweep = []
        for tau in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0):
            metrics, _ = evaluate(records, args.grid, tau)
            sweep.append(dict(tau=tau, **metrics["fused"]))
        write_csv(args.save_dir / "tau_sweep.csv", sweep)
        args.tau = min(sweep, key=lambda row: row["MAE"])["tau"]
    metrics, rows = evaluate(records, args.grid, args.tau)
    settings = {key: getattr(args, key) for key in defaults}
    parameters = dict(settings=settings, source_split=args.split, diagnostic=args.diagnostic,
                      weights_sha256=weights_hash, protocol_id=data.get("protocol_id"))
    summary = dict(**parameters, weights=str(Path(args.weights).resolve()), data=str(Path(args.data).resolve()),
                   images=len(rows), gt_source="original_points" if data.get("count_manifest") else "pseudo_boxes",
                   metrics=metrics, detection_limit_images=sum(row["detect"] >= args.max_det for row in rows),
                   torch=torch.__version__, ultralytics=ultralytics.__version__, device=device,
                   device_name=torch.cuda.get_device_name(torch.device(device)) if device != "cpu" else "CPU")
    summary["variant"] = getattr(model, "crowd_cfg", {}).get("variant", "full")
    summary["training_seed"] = getattr(model, "crowd_cfg", {}).get("training_seed")
    summary["crowd_cfg"] = getattr(model, "crowd_cfg", {})
    summary["evaluation_batch"] = args.batch
    write_csv(args.save_dir / "per_image.csv", rows)
    for name, content in (("summary.json", summary), ("parameters.json", parameters)):
        (args.save_dir / name).write_text(json.dumps(content, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Results: {args.save_dir.resolve()}")


if __name__ == "__main__":
    main()
