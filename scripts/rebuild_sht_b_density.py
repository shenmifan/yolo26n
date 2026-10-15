"""Rebuild SHT-B spatial supervision and audit every saved map/transform.

python scripts/rebuild_sht_b_density.py --data datasets/sht_b/data_seed42.yaml \
    --src ../data/part_B --output datasets/sht_b_density_v2 --workers 4
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from tqdm import tqdm
from ultralytics.utils.instance import Instances

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from crowd.dataset import CrowdFlip, CrowdLetterBox
from crowd.density import audit_points, build_density_map, load_points, resize_density_preserve_count, resize_density_preserve_count_torch
from crowd.protocol import data_path, path_key, sha256_file, verify_protocol


def json_write(path, content):
    path.write_text(json.dumps(content, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def csv_write(path, rows, fields=None):
    with open(path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(dataset="sht_b"):
    part = {"sht_a": "A", "sht_b": "B"}[dataset]
    fixed_sigma = None if part == "A" else 15.0
    parser = argparse.ArgumentParser(description=f"Rebuild and audit ShanghaiTech-{part} density supervision")
    parser.add_argument("--data", default=f"datasets/{dataset}/data_seed42.yaml")
    parser.add_argument("--src", default=f"../data/part_{part}")
    parser.add_argument("--output", default=f"datasets/{dataset}_density_v2")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument("--resume", action="store_true", help="校验已写出的文件并继续中断的同一构建")
    args = parser.parse_args()
    if args.workers < 1 or args.tolerance <= 0:
        parser.error("workers 和 tolerance 必须为正")
    data = yaml.safe_load(Path(args.data).read_text(encoding="utf-8"))
    counts = verify_protocol(data)
    if data["split_protocol"].get("dataset") != f"ShanghaiTech-{part}":
        raise ValueError(f"该入口只能重建 ShanghaiTech-{part}，数据协议不匹配")
    source, output, dataset_root = Path(args.src).resolve(), Path(args.output).resolve(), Path(data["path"]).resolve()
    if output == dataset_root or output.is_relative_to(dataset_root) or output.is_relative_to(source):
        raise ValueError("输出目录必须与原始数据、旧数据目录分离")
    records = json.loads(data_path(data, data["count_manifest"]).read_text(encoding="utf-8"))
    membership = {}
    for split in (("train", "val", "test", "excluded") if data.get("excluded") else ("train", "val", "test")):
        for image in data_path(data, data[split]).read_text(encoding="utf-8").splitlines():
            if image.strip():
                membership[path_key(image)] = split
    if len(membership) != len(records):
        raise ValueError("点数清单与图片划分不一致")
    spec = dict(schema=1, source=source.as_posix(), source_protocol_id=data["protocol_id"],
                fixed_sigma=fixed_sigma, gaussian_truncate=4.0, tolerance=args.tolerance,
                density_code_sha256=sha256_file(ROOT / "crowd/density.py"),
                valid_point_rule="finite and 0<=x<width and 0<=y<height; no coordinate shift",
                outside_point_policy="exclude from spatial target and list individually; retain original evaluation count",
                duplicate_policy="preserve each annotation entry, do not deduplicate",
                normalization="per-source Gaussian boundary-mass correction plus numerical mass correction",
                transform_sizes=[320, 640, 768], numpy=np.__version__, torch=torch.__version__)
    if part == "A":
        spec.update(adaptive_k=3, adaptive_beta=0.3, adaptive_sigma_min=1.0,
                    normalization="each adaptive Gaussian independently normalized at image boundaries plus numerical mass correction")
    spec_path = output / "build_spec.json"
    if output.exists():
        if not args.resume or not spec_path.exists() or json.loads(spec_path.read_text(encoding="utf-8")) != spec:
            raise FileExistsError("输出已存在；同一构建使用 --resume，其他构建使用新目录")
    else:
        output.mkdir(parents=True)
        json_write(spec_path, spec)
    reports = output / "reports"
    reports.mkdir(exist_ok=True)
    cv2.setNumThreads(1)
    torch.set_num_threads(1)

    def process(record):
        image = data_path(data, record["image"])
        official_stem = image.stem.removeprefix(f"sht{part.lower()}_")
        annotation = source / record["source_split"] / "ground_truth" / f"GT_{official_stem}.mat"
        if sha256_file(image) != record["image_sha256"] or sha256_file(annotation) != record["annotation_sha256"]:
            raise ValueError(f"源文件已变化：{image}")
        img = cv2.imread(str(image))
        if img is None:
            raise ValueError(f"图像无法读取：{image}")
        h, w = img.shape[:2]
        points = load_points(str(annotation), "sht")
        inside, audit = audit_points(points, h, w)
        if len(points) != record["gt_count"]:
            raise ValueError(f"原始人数清单不一致：{image}")
        relative = image.relative_to(dataset_root / "images").with_suffix(".npy")
        destination = output / "density" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            density = build_density_map(points, h, w, fixed_sigma=fixed_sigma)
            temporary = destination.with_suffix(".npy.partial")
            with open(temporary, "wb") as handle:
                np.save(handle, density, allow_pickle=False)
            temporary.replace(destination)
        density = np.load(destination, allow_pickle=False)
        if density.shape != (h, w) or density.dtype != np.float32:
            raise ValueError(f"密度图形状/类型不正确：{destination}")
        expected = audit["valid_count"]
        mass = float(density.sum(dtype=np.float64))
        valid_numeric = bool(np.isfinite(density).all() and np.all(density >= 0))
        row = dict(image=record["image"], split=membership[path_key(image)], source_split=record["source_split"],
                   height=h, width=w, raw_point_count=audit["raw_count"], supervision_point_count=expected,
                   outside_point_count=audit["outside_count"], nonfinite_point_count=audit["nonfinite_count"],
                   duplicate_extra_count=audit["duplicate_extra_count"], valid_duplicate_extra_count=audit["valid_duplicate_extra_count"],
                   old_density_sum=record["density_sum"], old_error_vs_raw=record["density_sum"] - audit["raw_count"],
                   old_error_vs_supervision=record["density_sum"] - expected,
                   new_density_sum=mass, new_error_vs_supervision=mass - expected,
                   new_error_vs_raw=mass - audit["raw_count"], finite_nonnegative=valid_numeric)
        irregular = resize_density_preserve_count(density, 257, 389)
        row["irregular_resize_sum"] = float(irregular.sum(dtype=np.float64))
        for size in spec["transform_sizes"]:
            transformed = CrowdLetterBox((size, size), scaleup=False, auto=False, density_stride=4)(
                dict(img=img.copy(), density_full=density))
            target = transformed["density"]
            row[f"letterbox_{size}_stride4_sum"] = float(target.double().sum().item())
            if size == 640:
                transformed["instances"] = Instances(np.zeros((0, 4), dtype=np.float32), bbox_format="xywh", normalized=False)
                flipped = CrowdFlip(p=1.0, direction="horizontal")(transformed)
                row["flip_sum"] = float(flipped["density"].double().sum().item())
                row["flip_spatial_alignment"] = bool(torch.equal(flipped["density"], torch.flip(target, [-1])))
                batched = target[None, None]
                for target_size in (80, 192):
                    scaled = resize_density_preserve_count_torch(batched, (target_size, target_size))
                    row[f"torch_multiscale_{target_size}_sum"] = float(scaled.double().sum().item())
        checked_sums = [value for key, value in row.items() if key in {"new_density_sum", "irregular_resize_sum", "flip_sum"}
                        or key.startswith("letterbox_") or key.startswith("torch_multiscale_")]
        row["max_abs_count_error"] = max(abs(value - expected) for value in checked_sums)
        row["pass_conservation"] = valid_numeric and row["flip_spatial_alignment"] and row["max_abs_count_error"] <= args.tolerance
        row["matches_raw_count"] = abs(mass - audit["raw_count"]) <= args.tolerance
        anomalous = []
        seen = set()
        for index, (x, y) in enumerate(points):
            reasons = []
            finite = bool(np.isfinite([x, y]).all())
            if not finite:
                reasons.append("nonfinite")
            elif not inside[index]:
                reasons.append("outside_image")
            coordinate = (float(x), float(y))
            if finite and coordinate in seen:
                reasons.append("duplicate_extra_entry")
            seen.add(coordinate)
            if reasons:
                anomalous.append(dict(image=record["image"], split=row["split"], annotation_index_0_based=index,
                                      x=float(x), y=float(y), width=w, height=h, reasons=";".join(reasons),
                                      contributes_to_density=bool(inside[index]), contributes_to_raw_gt=True))
        manifest_record = dict(image=record["image"], split=row["split"], raw_count=audit["raw_count"],
                               density_count=expected, density_path=destination.as_posix(),
                               density_sha256=sha256_file(destination), annotation_sha256=record["annotation_sha256"])
        return row, anomalous, manifest_record

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for result in tqdm(pool.map(process, records), total=len(records), desc=f"Rebuild and audit {part} density"):
            results.append(result)
    rows = [item[0] for item in results]
    anomalies = [row for item in results for row in item[1]]
    manifests = [item[2] for item in results]
    csv_write(reports / "per_image_conservation.csv", rows)
    csv_write(reports / "point_anomalies.csv", anomalies,
              ["image", "split", "annotation_index_0_based", "x", "y", "width", "height", "reasons", "contributes_to_density", "contributes_to_raw_gt"])
    summary = dict(images=len(rows), split_counts=counts,
                   total_raw_points=sum(row["raw_point_count"] for row in rows),
                   total_supervision_points=sum(row["supervision_point_count"] for row in rows),
                   total_outside_points=sum(row["outside_point_count"] for row in rows),
                   total_nonfinite_points=sum(row["nonfinite_point_count"] for row in rows),
                   total_duplicate_extra_entries=sum(row["duplicate_extra_count"] for row in rows),
                   conservation_pass_images=sum(row["pass_conservation"] for row in rows),
                   raw_count_match_images=sum(row["matches_raw_count"] for row in rows),
                   max_abs_count_error=max(row["max_abs_count_error"] for row in rows),
                   old_mean_abs_error_vs_raw=float(np.mean([abs(row["old_error_vs_raw"]) for row in rows])),
                   old_mean_abs_error_vs_supervision=float(np.mean([abs(row["old_error_vs_supervision"]) for row in rows])),
                   new_mean_abs_error_vs_supervision=float(np.mean([abs(row["new_error_vs_supervision"]) for row in rows])),
                   new_mean_abs_error_vs_raw=float(np.mean([abs(row["new_error_vs_raw"]) for row in rows])),
                   tolerance=args.tolerance, spec=spec,
                   note="Conservation is checked against spatially valid annotation entries. Outside/nonfinite entries remain in raw evaluation GT and are explicitly reported; duplicate entries are retained.")
    if data.get("excluded"):
        summary["excluded_images"] = sum(row["split"] == "excluded" for row in rows)
    json_write(reports / "summary.json", summary)
    if summary["conservation_pass_images"] != len(rows):
        raise RuntimeError("逐图守恒检查失败，未生成可训练的新配置；详见报告")
    manifest_path = output / "supervision_manifest.json"
    json_write(manifest_path, dict(spec=spec, records=manifests))
    new_data = dict(data)
    new_data["density_root"] = (output / "density").as_posix()
    new_data["density_count_tolerance"] = args.tolerance
    new_data["supervision_manifest"] = manifest_path.as_posix()
    hashes = dict(data["split_protocol"]["sha256"])
    hashes["supervision"] = sha256_file(manifest_path)
    new_data["split_protocol"] = dict(data["split_protocol"], sha256=hashes)
    new_data["protocol_id"] = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    config_path = output / "data.yaml"
    config_path.write_text(yaml.safe_dump(new_data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    verify_protocol(new_data)
    report = f"""{part} 集密度监督重建报告

完整处理 {len(rows)} 张图：训练 {counts['train']}、验证 {counts['val']}、测试 {counts['test']}；沿用 seed={data['split_protocol']['seed']} 的图片划分。
排除候选图片 {summary.get('excluded_images', 0)} 张，其密度图仅供审计，不进入训练、验证或测试清单。
高斯核：{'几何自适应，sigma=max(0.3×三近邻平均距离, 1)' if part == 'A' else '固定 sigma=15'}；每个有效点的边界核独立归一化。

| 检查 | 结果 |
| --- | ---: |
| 官方原始标注条目 | {summary['total_raw_points']} |
| 图内有效监督条目 | {summary['total_supervision_points']} |
| 越界条目 | {summary['total_outside_points']} |
| 非有限坐标条目 | {summary['total_nonfinite_points']} |
| 重复坐标的额外条目（保留） | {summary['total_duplicate_extra_entries']} |
| 生成、缩放、填充、翻转及多尺度守恒通过 | {summary['conservation_pass_images']} / {len(rows)} |
| 全部检查最大人数误差 | {summary['max_abs_count_error']:.8g} |
| 旧密度图对有效监督点的平均绝对偏差 | {summary['old_mean_abs_error_vs_supervision']:.6f} |
| 新密度图对有效监督点的平均绝对偏差 | {summary['new_mean_abs_error_vs_supervision']:.8g} |
| 新密度图对官方原始点数的平均绝对偏差 | {summary['new_mean_abs_error_vs_raw']:.6f} |

每个有效点贡献 1 人，重复条目原样保留。越界或非有限坐标不移到图像边缘，坐标和索引列在 point_anomalies.csv 中；官方评估点数不改写。
{len(rows) - summary['raw_count_match_images']} 张图存在原始计数与可定位监督的差别，需在论文中说明；该差别不计入数值守恒通过条件。

报告检查从磁盘重新加载的 float32 密度图：非整数缩放到 257×389；320/640/768 的真实 CrowdLetterBox 与 stride-4 标签；强制水平翻转及空间对齐；PyTorch 多尺度缩放到 80×80 和 192×192。绝对误差阈值 {args.tolerance} 人。

新训练配置：{config_path.as_posix()}。逐图报告：per_image_conservation.csv。标注异常：point_anomalies.csv。
本次任务完成监督重建和检查；训练指标需重新训练后获得。
"""
    (reports / "REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "spec"}, ensure_ascii=False, indent=2))
    print(f"New data config: {config_path}")


if __name__ == "__main__":
    main()
