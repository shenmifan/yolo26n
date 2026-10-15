"""Dataset manifests and immutable evaluation protocol checks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def path_key(path):
    return str(Path(path).resolve()).replace("\\", "/").casefold()


def data_path(data, value):
    path = Path(value)
    return path if path.is_absolute() else Path(data["path"]) / path


def read_count_manifest(data):
    if not data.get("count_manifest"):
        return None
    records = json.loads(data_path(data, data["count_manifest"]).read_text(encoding="utf-8"))
    return {path_key(data_path(data, row["image"])): row["gt_count"] for row in records}


def verify_protocol(data):
    """Fail if a frozen list/count file changed or images overlap between splits."""
    protocol = data.get("split_protocol")
    if not protocol:
        raise ValueError("缺少独立 train/val/test 划分；先运行对应数据集的划分脚本。")
    splits = {}
    for split in ("train", "val", "test"):
        manifest = data_path(data, data[split])
        if sha256_file(manifest) != protocol["sha256"][split]:
            raise ValueError(f"{split} 清单已改变，请重新生成并记录实验协议。")
        paths = [path_key(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not paths or len(paths) != len(set(paths)):
            raise ValueError(f"{split} 清单为空或包含重复图片。")
        splits[split] = set(paths)
    if any(splits[a] & splits[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise ValueError("train/val/test 出现图片重叠。")
    if data.get("excluded"):
        excluded_manifest = data_path(data, data["excluded"])
        if sha256_file(excluded_manifest) != protocol["sha256"].get("excluded"):
            raise ValueError("排除图片清单已改变。")
        excluded = [path_key(line) for line in excluded_manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(excluded) != len(set(excluded)) or any(set(excluded) & paths for paths in splits.values()):
            raise ValueError("排除图片清单包含重复图片或与实验划分重叠。")
    counts_path = data_path(data, data["count_manifest"])
    if sha256_file(counts_path) != protocol["sha256"]["counts"]:
        raise ValueError("原始点标注人数清单已改变。")
    if data.get("supervision_manifest"):
        manifest = data_path(data, data["supervision_manifest"])
        if sha256_file(manifest) != protocol["sha256"].get("supervision"):
            raise ValueError("密度监督清单已改变。")
    expected_id = hashlib.sha256(json.dumps(protocol["sha256"], sort_keys=True).encode()).hexdigest()
    if expected_id != data["protocol_id"]:
        raise ValueError("数据划分协议 ID 不匹配。")
    return {split: len(paths) for split, paths in splits.items()}
