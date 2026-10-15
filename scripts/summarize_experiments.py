"""Audit downloaded A/B evaluations and build measured single-seed paper results."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
from pathlib import Path, PurePosixPath

import yaml

ROOT = Path(__file__).resolve().parents[1]
PARTS = {"sht_a": ("A", 60, 182), "sht_b": ("B", 80, 316)}
MODES = ("detect", "density", "fused")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def image_key(value, dataset):
    marker = f"/datasets/{dataset}/"
    normalized = str(value).replace("\\", "/")
    require(marker in normalized, f"Unexpected {dataset} image path: {value}")
    return normalized.split(marker, 1)[1]


def close(left, right):
    return math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-8)


def server_protocol(root, dataset, summary):
    """Reproduce the relocation hashes using frozen originals and the recorded host root."""
    base_dir = root / "datasets" / f"{dataset}_density_v2"
    base = yaml.safe_load((base_dir / "data.yaml").read_text(encoding="utf-8"))
    hashes = dict(base["split_protocol"]["sha256"])
    require(hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest() == base["protocol_id"], "Original protocol ID changed")
    remote_config = PurePosixPath(summary["data"])
    server_root = remote_config.parents[2]
    require(remote_config == server_root / "datasets" / f"{dataset}_density_v2/data_server.yaml", "Unexpected server configuration path")
    dataset_root = server_root / "datasets" / dataset
    source_dir = root / "datasets" / dataset / f"splits_seed{base['split_protocol']['seed']}"
    members = {}
    for split in ("train", "val", "test") + (("excluded",) if base.get("excluded") else ()):
        path = source_dir / f"{split}.txt"
        require(sha256(path) == hashes[split], f"Original {dataset}/{split} manifest changed")
        keys = [image_key(line, dataset) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        require(keys and len(set(keys)) == len(keys), f"Empty/duplicate {split} membership")
        members[split] = set(keys)
        content = "".join((dataset_root / key).as_posix() + "\n" for key in keys)
        hashes[split] = hashlib.sha256(content.encode("utf-8")).hexdigest()
    for index, split in enumerate(members):
        require(all(not members[split] & members[other] for other in list(members)[index + 1:]), "Split overlap")
    counts_path = source_dir / "counts.json"
    require(sha256(counts_path) == hashes["counts"], "Original raw count manifest changed")
    supervision_path = base_dir / "supervision_manifest.json"
    require(sha256(supervision_path) == hashes["supervision"], "Original supervision manifest changed")
    supervision = copy.deepcopy(read_json(supervision_path))
    prefix = str(base["density_root"]).replace("\\", "/").rstrip("/") + "/"
    for row in supervision["records"]:
        density = row["density_path"].replace("\\", "/")
        require(density.casefold().startswith(prefix.casefold()), "Unexpected density provenance path")
        suffix = PurePosixPath(density[len(prefix):])
        require(not suffix.is_absolute() and ".." not in suffix.parts, "Invalid density provenance path")
        row["density_path"] = (server_root / "datasets" / f"{dataset}_density_v2/density" / suffix).as_posix()
    supervision["relocation"] = dict(source_protocol_id=base["protocol_id"], project_root=server_root.as_posix(), image_membership_preserved=True)
    content = (json.dumps(supervision, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    hashes["supervision"] = hashlib.sha256(content).hexdigest()
    derived = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    require(derived == summary["protocol_id"], "Recorded server protocol does not match reconstructed frozen manifests")
    return members, read_json(counts_path)


def audit_dataset(root, dataset, seed):
    part, val_count, test_count = PARTS[dataset]
    name = f"sht{part.lower()}_density_v2_seed{seed}"
    run = root / "runs/crowd" / name
    folders = {split: root / "runs/eval" / f"{name}_{split}" for split in ("val", "test")}
    summaries = {split: read_json(folder / "summary.json") for split, folder in folders.items()}
    parameters = {split: read_json(folder / "parameters.json") for split, folder in folders.items()}
    members, count_rows = server_protocol(root, dataset, summaries["test"])
    counts = {row["image"]: row["gt_count"] for row in count_rows}
    require(len(counts) == len(count_rows), "Duplicate raw count records")
    require(set(counts) == set.union(*members.values()), "Counts and frozen membership differ")
    input_paths = [run / "weights/best.pt", run / "args.yaml", run / "results.csv",
                   root / "datasets" / f"{dataset}_density_v2/data.yaml",
                   root / "datasets" / f"{dataset}_density_v2/supervision_manifest.json"]
    for split, expected_count in (("val", val_count), ("test", test_count)):
        summary, params = summaries[split], parameters[split]
        require(summary["source_split"] == params["source_split"] == split, "Wrong evaluation split")
        require(summary["diagnostic"] is False and params["diagnostic"] is False, "Diagnostic result cannot become a formal result")
        require(summary["gt_source"] == "original_points", "Raw original GT is required")
        for key in ("settings", "weights_sha256", "protocol_id"):
            require(summary[key] == params[key] == parameters["val"][key], f"Frozen evaluation mismatch: {key}")
        for key in ("torch", "ultralytics", "device_name"):
            require(summary[key] == summaries["val"][key], f"Environment mismatch: {key}")
        rows = read_csv(folders[split] / "per_image.csv")
        keys = [image_key(row["image"], dataset) for row in rows]
        require(len(rows) == summary["images"] == expected_count, "Unexpected image count")
        require(len(set(keys)) == len(keys) and set(keys) == members[split], "Evaluation image membership mismatch")
        for key, row in zip(keys, rows):
            require(float(row["gt"]) == counts[key], f"Raw GT mismatch: {key}")
        for mode in MODES:
            errors = [float(row[mode]) - float(row["gt"]) for row in rows]
            require(all(math.isfinite(value) for value in errors), "Nonfinite count prediction")
            for row, error in zip(rows, errors):
                require(close(row[f"error_{mode}"], error) and close(row[f"abs_error_{mode}"], abs(error)), "Per-image error mismatch")
            recomputed = dict(MAE=math.fsum(abs(value) for value in errors) / len(rows),
                              RMSE=math.sqrt(math.fsum(value * value for value in errors) / len(rows)),
                              bias=math.fsum(errors) / len(rows))
            require(all(close(summary["metrics"][mode][key], value) for key, value in recomputed.items()), "Summary does not reproduce per-image metrics")
        input_paths.extend(folders[split] / filename for filename in ("summary.json", "parameters.json", "per_image.csv"))
    require(sha256(run / "weights/best.pt") == summaries["test"]["weights_sha256"], "Downloaded checkpoint hash mismatch")
    sweep_path = folders["val"] / "tau_sweep.csv"
    sweep = read_csv(sweep_path)
    selected = min(sweep, key=lambda row: float(row["MAE"]))
    require(close(selected["tau"], parameters["val"]["settings"]["tau"]), "Tau not selected by validation MAE")
    require(all(close(selected[key], summaries["val"]["metrics"]["fused"][key]) for key in ("MAE", "RMSE", "bias")), "Selected sweep row differs from validation summary")
    training = read_csv(run / "results.csv")
    args = yaml.safe_load((run / "args.yaml").read_text(encoding="utf-8"))
    require(args["seed"] == seed and dataset in args["data"], "Wrong training run")
    require([int(row["epoch"]) for row in training] == list(range(1, len(training) + 1)), "Training epoch sequence changed")
    best_epoch = int(min(training, key=lambda row: float(row["metrics/MAE"]))["epoch"])
    input_paths.extend([sweep_path, root / "datasets" / dataset / "splits_seed42/counts.json"])
    input_paths.extend(root / "datasets" / dataset / "splits_seed42" / f"{split}.txt" for split in members)
    test = summaries["test"]
    return dict(dataset=f"ShanghaiTech-{part}", seed=seed, checked_images=dict(val=val_count, test=test_count),
                training_epochs=len(training), best_epoch_from_csv=best_epoch, settings=test["settings"],
                protocol_id=test["protocol_id"], weights_sha256=test["weights_sha256"], metrics=test["metrics"],
                environment={key: test[key] for key in ("torch", "ultralytics", "device", "device_name")},
                training_settings={key: args[key] for key in ("seed", "epochs", "patience", "batch", "imgsz", "optimizer", "lr0", "lrf", "cos_lr", "warmup_epochs", "momentum", "weight_decay", "amp")},
                checks=["raw GT and exact image membership", "all per-image errors and summary metrics", "validation-selected tau and frozen test settings", "downloaded best.pt SHA-256", "server protocol ID reconstructed from frozen local source manifests and recorded server root"],
                limitations=["Only this single training seed is summarized; no repeated-run mean/std, baseline, architectural ablation or runtime measurement is implied.", "Original server YAML and relocated sidecar files should also be archived; their protocol ID has been independently reconstructed."],
                inputs_sha256={path.relative_to(root).as_posix(): sha256(path) for path in input_paths})


def write_results(reports, output, seed):
    output.mkdir(parents=True, exist_ok=True)
    lines = [f"# A/B seed{seed} 正式测试结果", "", "以下数值来自同一联合模型检查点的三条计数路径；每个数据集独立训练并在独立验证集选取 tau。", "", "| 数据集 | 测试图数 | 路径 | MAE | RMSE | Bias |", "| --- | ---: | --- | ---: | ---: | ---: |"]
    csv_rows = []
    for part, report in reports.items():
        (output / f"{part}_seed{seed}_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        for mode, label in zip(MODES, ("检测", "密度", "融合")):
            value = report["metrics"][mode]
            lines.append(f"| {part} | {report['checked_images']['test']} | {label} | {value['MAE']:.3f} | {value['RMSE']:.3f} | {value['bias']:.3f} |")
            csv_rows.append(dict(dataset=part, seed=seed, images=report["checked_images"]["test"], path=mode, tau=report["settings"]["tau"], **value))
    lines += ["", "| 数据集 | 实际训练轮数 | 训练 CSV 最优轮次 | 验证图数 | 验证选择的 tau |", "| --- | ---: | ---: | ---: | ---: |"]
    for part, report in reports.items():
        lines.append(f"| {part} | {report['training_epochs']} | {report['best_epoch_from_csv']} | {report['checked_images']['val']} | {report['settings']['tau']} |")
    lines += ["", "融合相对同一检查点的密度路径：", ""]
    for part, report in reports.items():
        metrics = report["metrics"]
        lines.append(f"- {part}：MAE 降低 {100 * (1 - metrics['fused']['MAE'] / metrics['density']['MAE']):.2f}%，RMSE 降低 {100 * (1 - metrics['fused']['RMSE'] / metrics['density']['RMSE']):.2f}%。")
    lines += ["", "实际运行环境：", ""]
    for part, report in reports.items():
        env = report["environment"]
        lines.append(f"- {part}：{env['device_name']}；PyTorch {env['torch']}；Ultralytics {env['ultralytics']}。")
    lines += ["", "复现实验应保存并使用实际运行环境。", "", "核对已覆盖原始 GT、完整成员清单、逐图及汇总指标、权重哈希、验证选参和服务器协议 ID 重建。逐项输入 SHA-256 见各 audit.json。", "", "论文尚需补充：其他训练种子的均值/标准差、外部基线、架构和损失消融、速度测试及定性案例。三条输出路径属于同一模型的推理比较，不能替代架构消融。", ""]
    (output / f"AB_seed{seed}_results.md").write_text("\n".join(lines), encoding="utf-8")
    with (output / f"AB_seed{seed}_metrics.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    reports = {PARTS[dataset][0]: audit_dataset(ROOT, dataset, args.seed) for dataset in PARTS}
    output = ROOT / "reports/experiment_results"
    write_results(reports, output, args.seed)
    print(json.dumps({part: {key: report[key] for key in ("training_epochs", "best_epoch_from_csv", "settings", "metrics")} for part, report in reports.items()}, indent=2))
    print(f"Results: {output.resolve()}")


if __name__ == "__main__":
    main()
