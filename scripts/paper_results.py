"""Audit every completed paper run; report sample standard deviation over seeds.

Missing runs stay pending. A single seed has std=None, never a fabricated zero.
"""
import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from crowd.experiments import EXTERNAL, VARIANTS, run_name
from scripts.summarize_experiments import (PARTS, close, image_key, read_csv, read_json,
                                          require, server_protocol, sha256)


def frozen_members(root, dataset, summary):
    path = root / "datasets" / f"{dataset}_density_v2/data_server.yaml"
    # Relocated server sidecars exist on the server; downloaded records are also
    # auditable locally using deterministic reconstruction from frozen originals.
    if path.is_file():
        import yaml
        from crowd.protocol import data_path, verify_protocol
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not Path(data["path"]).is_dir():
            return server_protocol(root, dataset, summary)
        verify_protocol(data)
        require(data["protocol_id"] == summary["protocol_id"], "Wrong frozen server protocol")
        members = {split: {image_key(line, dataset) for line in data_path(data, data[split]).read_text(encoding="utf-8").splitlines() if line.strip()} for split in ("train", "val", "test")}
        expected, counts = server_protocol(root, dataset, summary)
        require(all(members[split] == expected[split] for split in members), "Server membership changed from frozen originals")
        require(read_json(data_path(data, data["count_manifest"])) == counts, "Server original counts changed")
        return members, counts
    return server_protocol(root, dataset, summary)


def audit_run(root, dataset, variant, seed):
    name = run_name(dataset, variant, seed)
    run = root / "runs/crowd" / name
    folders = {split: root / "runs/eval" / f"{name}_{split}" for split in ("val", "test")}
    if not (folders["test"] / "summary.json").is_file():
        return None
    summaries = {split: read_json(folder / "summary.json") for split, folder in folders.items()}
    params = {split: read_json(folder / "parameters.json") for split, folder in folders.items()}
    members, count_rows = frozen_members(root, dataset, summaries["test"])
    counts = {row["image"]: row["gt_count"] for row in count_rows}
    require(len(counts) == len(count_rows), "Duplicate count manifest entries")
    modes = ("density",) if variant in EXTERNAL else (("detect",) if not VARIANTS[variant]["density"] else ("detect", "density", "fused"))
    evidence = [run / "weights/best.pt"]
    for split in ("val", "test"):
        summary, parameter = summaries[split], params[split]
        require(summary["source_split"] == parameter["source_split"] == split, "Wrong split")
        require(summary["diagnostic"] is False and parameter["diagnostic"] is False, "Diagnostic result rejected")
        require(summary["gt_source"] == "original_points", "Raw GT required")
        require(summary.get("variant", "full") == variant, "Wrong variant")
        require(set(summary["metrics"]) == set(modes), "Wrong metric modes")
        require(summary.get("training_seed") in (None, seed), "Wrong recorded seed")
        for key in ("settings", "weights_sha256", "protocol_id"):
            require(summary[key] == parameter[key] == params["val"][key], f"Frozen parameter mismatch: {key}")
        for key in ("torch", "ultralytics", "device_name"):
            require(summary[key] == summaries["val"][key], "Validation/test environment mismatch")
        rows = read_csv(folders[split] / "per_image.csv")
        keys = [image_key(row["image"], dataset) for row in rows]
        expected_count = PARTS[dataset][1 if split == "val" else 2]
        require(len(rows) == summary["images"] == expected_count, "Incomplete evaluation")
        require(len(set(keys)) == len(keys) and set(keys) == members[split], "Wrong image membership")
        for row, key in zip(rows, keys):
            require(float(row["gt"]) == counts[key], "Wrong original count")
        for mode in modes:
            errors = [float(row[mode]) - float(row["gt"]) for row in rows]
            require(all(math.isfinite(error) for error in errors), "Nonfinite prediction")
            expected = dict(MAE=math.fsum(map(abs, errors)) / len(rows),
                            RMSE=math.sqrt(math.fsum(error**2 for error in errors) / len(rows)),
                            bias=math.fsum(errors) / len(rows))
            require(all(close(summary["metrics"][mode][key], value) for key, value in expected.items()), "Metrics do not match per-image predictions")
            require(all(close(row[f"error_{mode}"], error) and close(row[f"abs_error_{mode}"], abs(error)) for row, error in zip(rows, errors)), "Wrong per-image errors")
        evidence.extend(folders[split] / filename for filename in ("summary.json", "parameters.json", "per_image.csv"))
    require(sha256(run / "weights/best.pt") == summaries["test"]["weights_sha256"], "Checkpoint hash mismatch")
    if variant in EXTERNAL:
        config = read_json(run / "baseline_config.json")
        require(config["method"] == variant and config["seed"] == seed, "Wrong baseline training metadata")
        history = read_csv(run / "results.csv")
        require([int(row["epoch"]) for row in history] == list(range(1, len(history) + 1)), "Incomplete training history")
        require(config["protocol_id"] == summaries["test"]["protocol_id"], "Baseline training protocol mismatch")
        evidence.append(run / "baseline_config.json")
        recipe = {key: config[key] for key in ("recipe", "epochs", "patience", "batch", "imgsz", "optimizer", "lr0", "precision", "loss", "initialization")}
    else:
        import yaml
        config = yaml.safe_load((run / "args.yaml").read_text(encoding="utf-8"))
        require(config["seed"] == seed and dataset in config["data"], "Wrong training seed/dataset")
        history = read_csv(run / "results.csv")
        require([int(row["epoch"]) for row in history] == list(range(1, len(history) + 1)), "Incomplete training history")
        recipe = {key: config[key] for key in ("epochs", "patience", "batch", "imgsz", "optimizer", "lr0", "lrf", "cos_lr", "warmup_epochs", "momentum", "weight_decay", "amp")}
        if variant != "full":
            settings = summaries["test"].get("crowd_cfg", {})
            require(settings.get("variant") == variant, "Missing trained variant record")
            require(tuple(settings.get(key) for key in ("den_gain", "cnt_gain", "cons_gain")) == VARIANTS[variant]["gains"], "Wrong trained loss gains")
            require(settings.get("density_enabled") == VARIANTS[variant]["density"], "Wrong trained density-head setting")
        if "fused" in modes:
            sweep_file = folders["val"] / "tau_sweep.csv"
            selected = min(read_csv(sweep_file), key=lambda row: float(row["MAE"]))
            require(close(selected["tau"], params["val"]["settings"]["tau"]), "Tau not selected on validation")
            require(all(close(selected[key], summaries["val"]["metrics"]["fused"][key]) for key in ("MAE", "RMSE", "bias")), "Calibration metrics mismatch")
            evidence.append(sweep_file)
        evidence.append(run / "args.yaml")
    evidence.append(run / "results.csv")
    return dict(dataset=dataset, variant=variant, seed=seed, protocol_id=summaries["test"]["protocol_id"],
                metrics=summaries["test"]["metrics"], settings=params["val"]["settings"], recipe=recipe,
                environment={key: summaries["test"][key] for key in ("torch", "ultralytics", "device_name")},
                weights_sha256=summaries["test"]["weights_sha256"],
                inputs_sha256={path.relative_to(root).as_posix(): sha256(path) for path in evidence})


def aggregate(reports, expected_seeds):
    if not reports:
        return []
    seeds = [report["seed"] for report in reports]
    require(len(seeds) == len(set(seeds)), "Duplicate seed")
    require(set(seeds) <= set(expected_seeds), "Unexpected seed")
    for key in ("dataset", "variant", "protocol_id", "environment", "recipe"):
        require(all(report[key] == reports[0][key] for report in reports), f"Cannot pool different {key}")
    settings = [{key: value for key, value in report["settings"].items() if key != "tau"} for report in reports]
    require(all(value == settings[0] for value in settings), "Cannot pool different inference settings")
    require(len({report["weights_sha256"] for report in reports}) == len(reports), "The same checkpoint was reused for multiple seeds")
    rows = []
    for mode in reports[0]["metrics"]:
        row = dict(dataset=reports[0]["dataset"], variant=reports[0]["variant"], mode=mode,
                   n=len(reports), seeds=",".join(map(str, sorted(seeds))),
                   status="complete" if set(seeds) == set(expected_seeds) else "pending_seeds")
        for metric in ("MAE", "RMSE", "bias"):
            values = [report["metrics"][mode][metric] for report in reports]
            row[f"{metric}_mean"] = statistics.mean(values)
            row[f"{metric}_std"] = statistics.stdev(values) if len(values) >= 2 else None
        rows.append(row)
    return rows


def runtime_reports(root, datasets, variants):
    rows, pending = [], []
    for dataset in datasets:
        for variant in variants:
            name = run_name(dataset, variant, 0)
            folder = root / "runs/benchmark" / name
            path = folder / "summary.json"
            if not path.is_file():
                pending.append(name)
                continue
            record = read_json(path)
            require(record["purpose"] == "runtime_measurement", "Smoke timing cannot become paper timing")
            require(record["batch"] == 1 and record["precision"] == "FP32", "Runtime protocol changed")
            require(record["warmup"] >= 30, "Insufficient runtime warmup")
            require(record["method"] == variant, "Wrong runtime variant")
            eval_params = root / "runs/eval" / f"{name}_val/parameters.json"
            require(record["weights_sha256"] == sha256(root / "runs/crowd" / name / "weights/best.pt"), "Runtime checkpoint mismatch")
            require(record["params_sha256"] == sha256(eval_params), "Runtime parameters mismatch")
            times = read_json(folder / "timings_ms.json")
            for scope in ("network_only", "end_to_end"):
                values = times[scope]
                require(len(values) >= 200 and all(math.isfinite(value) and value > 0 for value in values), "Invalid runtime samples")
                require(len(values) == record["metrics"][scope]["iterations"], "Wrong timing iteration count")
                require(close(record["metrics"][scope]["mean_ms"], statistics.mean(values)), "Wrong timing mean")
                require(close(record["metrics"][scope]["median_ms"], statistics.median(values)), "Wrong timing median")
                require(close(record["metrics"][scope]["serial_images_per_second"], 1000 / statistics.mean(values)), "Wrong throughput")
                ordered = sorted(values)
                position = .95 * (len(values) - 1)
                lower = math.floor(position)
                p95 = ordered[lower] + (ordered[min(lower + 1, len(values) - 1)] - ordered[lower]) * (position - lower)
                require(close(record["metrics"][scope]["p95_ms"], p95), "Wrong timing p95")
            env = record["environment"]
            rows.append(dict(dataset=dataset, variant=variant, device=env["device_name"], torch=env["torch"],
                             ultralytics=env["ultralytics"], parameters=record["parameter_count"],
                             network_median_ms=record["metrics"]["network_only"]["median_ms"],
                             end_to_end_median_ms=record["metrics"]["end_to_end"]["median_ms"],
                             end_to_end_p95_ms=record["metrics"]["end_to_end"]["p95_ms"],
                             serial_images_per_second=record["metrics"]["end_to_end"]["serial_images_per_second"],
                             summary_sha256=sha256(path), samples_sha256=sha256(folder / "timings_ms.json")))
    return rows, pending


def build_report(root=ROOT, output=None, seeds=(0, 1, 2), datasets=("sht_a", "sht_b"), variants=None):
    variants = variants or tuple(VARIANTS) + EXTERNAL
    output = Path(output or root / "reports/paper_experiments")
    audits, pending, pooled, invalid = [], [], [], []
    for dataset in datasets:
        for variant in variants:
            group = []
            for seed in seeds:
                try:
                    report = audit_run(root, dataset, variant, seed)
                except (ValueError, OSError, KeyError) as error:
                    invalid.append(dict(dataset=dataset, variant=variant, seed=seed, error=str(error)))
                    continue
                if report is None:
                    pending.append(dict(dataset=dataset, variant=variant, seed=seed))
                else:
                    group.append(report)
                    audits.append(report)
            if group:
                try:
                    pooled.extend(aggregate(group, seeds))
                except ValueError as error:
                    invalid.append(dict(dataset=dataset, variant=variant, error=str(error)))
    output.mkdir(parents=True, exist_ok=True)
    try:
        runtime, runtime_pending = runtime_reports(root, datasets, variants)
    except (ValueError, KeyError, OSError) as error:
        runtime, runtime_pending = [], []
        invalid.append(dict(stage="runtime", error=str(error)))
    # Pure metadata export: do not import any training framework for aggregation.
    record = dict(expected_seeds=list(seeds), std="sample standard deviation, ddof=1; null for n<2",
                  runs=audits, pending=pending, invalid=invalid, aggregates=pooled,
                  runtime=runtime, runtime_pending=runtime_pending)
    (output / "audit.json").write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    fields = list(pooled[0]) if pooled else ["dataset", "variant", "mode", "n", "seeds", "status", "MAE_mean", "MAE_std", "RMSE_mean", "RMSE_std", "bias_mean", "bias_std"]
    with (output / "mean_std.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(pooled)
    lines = ["# 论文补充实验进度与结果", "", "标准差为训练种子间样本标准差（ddof=1）。只有一个种子时显示待补；各种子的 tau 独立由验证集选择。", "", "| 数据集 | 变体 | 输出 | 种子数 | MAE 均值 ± SD | RMSE 均值 ± SD | 状态 |", "| --- | --- | --- | ---: | --- | --- | --- |"]
    def cell(row, metric):
        deviation = row[f"{metric}_std"]
        return f"{row[f'{metric}_mean']:.3f} ± {deviation:.3f}" if deviation is not None else f"{row[f'{metric}_mean']:.3f}（SD 待补）"
    for row in pooled:
        lines.append(f"| {row['dataset']} | {row['variant']} | {row['mode']} | {row['n']} | {cell(row, 'MAE')} | {cell(row, 'RMSE')} | {row['status']} |")
    lines += ["", f"已通过核验的训练运行：{len(audits)}；尚缺运行：{len(pending)}；核验异常：{len(invalid)}。", "", "缺失清单和逐项输入哈希见 audit.json。MCNN/CSRNet 是统一画布与 density_v2 目标下的架构复现；原论文结果和更强基线需另行标注。", ""]
    if invalid:
        lines += ["核验异常：", ""] + [f"- {item}" for item in invalid] + [""]
    lines += ["## 速度测量", "", "仅接收 runs/benchmark 下通过逐次计时核验的正式输出；本地 smoke 结果不进入此表。FP32、batch=1，预热≥30 次，计时≥200 次；磁盘读取与解码不计入。", ""]
    if runtime:
        lines += ["| 数据集 | 变体 | GPU/CPU | 参数量 | 网络中位数 ms | 全流程中位数 ms | 全流程 P95 ms | 串行图/秒 |", "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |"]
        for row in runtime:
            lines.append(f"| {row['dataset']} | {row['variant']} | {row['device']} | {row['parameters']} | {row['network_median_ms']:.3f} | {row['end_to_end_median_ms']:.3f} | {row['end_to_end_p95_ms']:.3f} | {row['serial_images_per_second']:.2f} |")
        lines += ["", "跨硬件或软件环境的速度记录应分开比较；各行 torch/ultralytics 信息保存在 audit.json。", ""]
    else:
        lines += ["服务器正式速度测量待运行。", ""]
    (output / "results.md").write_text("\n".join(lines), encoding="utf-8")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    record = build_report(output=args.output, seeds=args.seeds)
    print(f"Audited {len(record['runs'])}; pending {len(record['pending'])}; invalid {len(record['invalid'])}")
    if record["invalid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
