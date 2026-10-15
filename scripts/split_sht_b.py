"""Create 320/80/316 lists from existing SHT-B data, preserving all source files.

python scripts/split_sht_b.py --src ../data/part_B --data datasets/sht_b --seed 42
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crowd.density import load_points
from crowd.protocol import sha256_file, verify_protocol


PARTS = {
    "sht_a": dict(part="A", train=300, test=182, prefix="shta_"),
    "sht_b": dict(part="B", train=400, test=316, prefix="shtb_"),
}


def stratified_split(records, seed=42, dataset="sht_b", excluded_count=0):
    """Five equal count strata; reserve 20% of every stratum for validation."""
    part = PARTS[dataset]
    expected = part["train"] - excluded_count
    if len(records) != expected:
        raise ValueError(f"ShanghaiTech-{part['part']} 官方训练集应为 {expected} 张，实际 {len(records)}")
    ordered = sorted(records, key=lambda row: (row["gt_count"], row["image"]))
    rng = random.Random(seed)
    train, val = [], []
    if expected < 5 or excluded_count < 0:
        raise ValueError("有效训练候选池至少包含 5 张图，排除数量不能为负")
    for indices in np.array_split(np.arange(expected), 5):
        group = [ordered[int(i)] for i in indices]
        val_size = round(len(group) * 0.2)
        rng.shuffle(group)
        val.extend(group[:val_size])
        train.extend(group[val_size:])
    return sorted(train, key=lambda r: r["image"]), sorted(val, key=lambda r: r["image"])


def exclude_test_duplicates(train_records, test_records):
    """Preserve every test image; exclude matching training content from selection."""
    test_hashes = {row["image_sha256"]: row["image"] for row in test_records}
    eligible, excluded, seen = [], [], {}
    for row in train_records:
        fingerprint = row["image_sha256"]
        if fingerprint in test_hashes:
            excluded.append(dict(row, exclusion_reason="duplicate_of_official_test", duplicate_of=test_hashes[fingerprint]))
        elif fingerprint in seen:
            excluded.append(dict(row, exclusion_reason="duplicate_of_training_image", duplicate_of=seen[fingerprint]))
        else:
            eligible.append(row)
            seen[fingerprint] = row["image"]
    return eligible, excluded


def write_once(path, text):
    if path.exists() and path.read_text(encoding="utf-8") != text:
        raise FileExistsError(f"拒绝覆盖不同内容的协议文件：{path}。请使用新的 --output-name。")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main(dataset="sht_b"):
    part = PARTS[dataset]
    parser = argparse.ArgumentParser(description=f"Create frozen ShanghaiTech-{part['part']} train/val/test manifests")
    parser.add_argument("--src", default=f"../data/part_{part['part']}")
    parser.add_argument("--data", default=f"datasets/{dataset}")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-name", default=None)
    args = parser.parse_args()
    root, source = Path(args.data).resolve(), Path(args.src).resolve()
    out_name = args.output_name or f"splits_seed{args.seed}"
    if Path(out_name).name != out_name:
        parser.error("--output-name 必须为单个目录名")
    output = root / out_name
    pools = {}
    fingerprints = {}
    for local_split, official_split, expected in (("train", "train_data", part["train"]), ("val", "test_data", part["test"])):
        records = []
        images = sorted((root / "images" / local_split).glob("*.jpg"))
        if len(images) != expected:
            raise ValueError(f"{local_split}: 期望 {expected} 张，实际 {len(images)}")
        for image in images:
            stem = image.stem.removeprefix(part["prefix"])
            annotation = source / official_split / "ground_truth" / f"GT_{stem}.mat"
            original_image = source / official_split / "images" / f"{stem}.jpg"
            image_hash = sha256_file(image)
            if image_hash != sha256_file(original_image):
                raise ValueError(f"预处理图像与原始图像不一致：{image}")
            if image_hash in fingerprints and dataset == "sht_b":
                raise ValueError(f"发现内容完全相同的图片：{image} 与 {fingerprints[image_hash]}")
            fingerprints[image_hash] = str(image)
            points = load_points(str(annotation), "sht")
            density_file = root / "density" / local_split / f"{image.stem}.npy"
            label_file = root / "labels" / local_split / f"{image.stem}.txt"
            density_sum = float(np.load(density_file, mmap_mode="r").sum(dtype=np.float64))
            label_count = sum(bool(line.strip()) for line in label_file.read_text().splitlines())
            records.append(dict(image=image.relative_to(root).as_posix(), gt_count=len(points),
                                label_count=label_count, density_sum=density_sum,
                                source_split=official_split, image_sha256=image_hash,
                                annotation_sha256=sha256_file(annotation)))
        pools[local_split] = records
    eligible, excluded = (exclude_test_duplicates(pools["train"], pools["val"]) if dataset == "sht_a"
                          else (pools["train"], []))
    train, val = stratified_split(eligible, args.seed, dataset=dataset, excluded_count=len(excluded))
    splits = dict(train=train, val=val, test=pools["val"])
    for split, records in splits.items():
        write_once(output / f"{split}.txt", "".join((root / row["image"]).as_posix() + "\n" for row in records))
    excluded_lookup = {row["image"]: row for row in excluded}
    all_records = [excluded_lookup.get(row["image"], row) for row in pools["train"]] + pools["val"]
    if excluded:
        write_once(output / "excluded.txt", "".join((root / row["image"]).as_posix() + "\n" for row in excluded))
    write_once(output / "counts.json", json.dumps(all_records, ensure_ascii=False, indent=2) + "\n")
    hashes = {split: sha256_file(output / f"{split}.txt") for split in splits}
    hashes["counts"] = sha256_file(output / "counts.json")
    if excluded:
        hashes["excluded"] = sha256_file(output / "excluded.txt")
    strategy = (f"5 count strata, {part['train'] * 4 // 25} train + {part['train'] // 25} val per stratum"
                if not excluded else "exclude exact duplicate training content; 5 near-equal count strata, 20% validation per stratum")
    config = dict(path=root.as_posix(), **{split: f"{out_name}/{split}.txt" for split in splits},
                  nc=1, names=["head"], density_dir="density", density_stride=4,
                  count_manifest=f"{out_name}/counts.json", count_conf=0.25,
                  count_iou=0.5, count_max_det=2000, grid=16, dgf_tau=2.0,
                  protocol_id=hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
                  split_protocol=dict(dataset=f"ShanghaiTech-{part['part']}", seed=args.seed,
                                      strategy=strategy,
                                      sha256=hashes))
    if excluded:
        config["excluded"] = f"{out_name}/excluded.txt"
    config_path = root / f"data_seed{args.seed}.yaml" if args.output_name is None else output / "data.yaml"
    write_once(config_path, yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
    verify_protocol(config)
    audit = dict(counts={k: len(v) for k, v in splits.items()},
                 label_count_mismatch_images=sum(r["gt_count"] != r["label_count"] for r in all_records),
                 mean_gt_minus_density=float(np.mean([r["gt_count"] - r["density_sum"] for r in all_records])),
                 max_gt_minus_density=max(r["gt_count"] - r["density_sum"] for r in all_records),
                 note="GT uses every original annotation point. Existing pseudo boxes/density targets are preserved; their count differences are audited, never used as evaluation GT.")
    if dataset == "sht_a":
        audit.update(excluded_images=len(excluded), exclusions=excluded,
                     eligible_official_train_images=len(eligible),
                     exact_image_hash_overlap_after_exclusions=False)
    write_once(output / "audit.json", json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    print(f"配置：{config_path}")


if __name__ == "__main__":
    main()
