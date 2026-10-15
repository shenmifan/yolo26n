"""Create host-specific A/B protocols from uploaded, audited density datasets.

Run this once with the server's Python interpreter before training. Original
configs/manifests and image membership are preserved; no raw MAT files needed.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path, PurePosixPath

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from crowd.protocol import sha256_file, verify_protocol


def relocated_path(value, old_root, new_root):
    """Resolve foreign Windows/POSIX paths without asking the host to parse them."""
    value = str(value).replace("\\", "/")
    prefix = str(old_root).replace("\\", "/").rstrip("/") + "/"
    if value.casefold().startswith(prefix.casefold()):
        value = value[len(prefix):]
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts or ":" in value:
        raise ValueError(f"路径不属于预期的数据根目录：{value}")
    result = new_root.joinpath(*relative.parts).resolve()
    if not result.is_relative_to(new_root.resolve()):
        raise ValueError(f"路径越出数据目录：{result}")
    return result


def check_file(path, expected=None):
    if not path.is_file():
        raise FileNotFoundError(f"服务器缺少文件，请先上传：{path}")
    if expected is not None and sha256_file(path) != expected:
        raise ValueError(f"上传文件与原始审计哈希不同：{path}")


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def prepare_dataset(project_root, dataset, check_only=False):
    project_root = Path(project_root).resolve()
    config_path = project_root / "datasets" / f"{dataset}_density_v2" / "data.yaml"
    check_file(config_path)
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    expected_dataset = {"sht_a": "ShanghaiTech-A", "sht_b": "ShanghaiTech-B"}[dataset]
    if data["split_protocol"]["dataset"] != expected_dataset:
        raise ValueError("数据集与协议不匹配")
    old_root = data["path"]
    dataset_root = project_root / "datasets" / dataset
    density_root = config_path.parent / "density"
    old_supervision_root = str(data["density_root"]).replace("\\", "/").rsplit("/", 1)[0]
    hashes = dict(data["split_protocol"]["sha256"])
    expected_id = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    if expected_id != data["protocol_id"]:
        raise ValueError("原始协议 ID 不匹配")
    output = dataset_root / f"splits_seed{data['split_protocol']['seed']}_server"
    payloads = {}
    split_members = {}
    split_names = ["train", "val", "test"] + (["excluded"] if data.get("excluded") else [])
    for split in split_names:
        source = relocated_path(data[split], old_root, dataset_root)
        check_file(source, hashes[split])
        images = [relocated_path(line, old_root, dataset_root)
                  for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not images or len(images) != len(set(images)):
            raise ValueError(f"{split} 清单为空或包含重复图片")
        split_members[split] = set(images)
        payloads[output / f"{split}.txt"] = "".join(image.as_posix() + "\n" for image in images).encode("utf-8")
    for index, split in enumerate(split_names):
        if any(split_members[split] & split_members[other] for other in split_names[index + 1:]):
            raise ValueError("实验及排除清单存在图片重叠")

    counts_path = relocated_path(data["count_manifest"], old_root, dataset_root)
    check_file(counts_path, hashes["counts"])
    records = json.loads(counts_path.read_text(encoding="utf-8"))
    image_keys = set()
    for row in records:
        image = relocated_path(row["image"], old_root, dataset_root)
        check_file(image, row["image_sha256"])
        relative_image = image.relative_to(dataset_root / "images")
        check_file(dataset_root / "labels" / relative_image.with_suffix(".txt"))
        image_keys.add(image)
    selected = {Path(line) for path, content in payloads.items()
                for line in content.decode("utf-8").splitlines()}
    if image_keys != selected or len(image_keys) != len(records):
        raise ValueError("人数清单与实验及排除清单的成员不一致")

    supervision_path = relocated_path(data["supervision_manifest"], old_supervision_root, config_path.parent)
    check_file(supervision_path, hashes["supervision"])
    supervision = json.loads(supervision_path.read_text(encoding="utf-8"))
    supervision_images = set()
    for row in supervision["records"]:
        image = relocated_path(row["image"], old_root, dataset_root)
        density = relocated_path(row["density_path"], data["density_root"], density_root)
        expected_density = density_root / image.relative_to(dataset_root / "images").with_suffix(".npy")
        if density != expected_density.resolve():
            raise ValueError(f"图像与密度图路径不对应：{image}")
        check_file(density, row["density_sha256"])
        row["density_path"] = density.as_posix()
        supervision_images.add(image)
    if supervision_images != image_keys or len(supervision["records"]) != len(records):
        raise ValueError("监督清单与原始人数清单不一致")
    # The original generation spec is provenance; relocation does not regenerate targets.
    supervision["relocation"] = dict(source_protocol_id=data["protocol_id"],
                                     project_root=project_root.as_posix(), image_membership_preserved=True)
    new_supervision = output / "supervision_manifest.json"
    payloads[new_supervision] = json_bytes(supervision)
    new_data = copy.deepcopy(data)
    new_data["path"] = dataset_root.as_posix()
    new_data["density_root"] = density_root.as_posix()
    new_data["supervision_manifest"] = new_supervision.as_posix()
    for split in split_names:
        new_data[split] = (output / f"{split}.txt").relative_to(dataset_root).as_posix()
        hashes[split] = hashlib.sha256(payloads[output / f"{split}.txt"]).hexdigest()
    hashes["supervision"] = hashlib.sha256(payloads[new_supervision]).hexdigest()
    new_data["split_protocol"]["sha256"] = hashes
    new_data["protocol_id"] = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    new_data["relocated_from_protocol_id"] = data["protocol_id"]
    new_config = config_path.parent / "data_server.yaml"
    payloads[new_config] = yaml.safe_dump(new_data, allow_unicode=True, sort_keys=False).encode("utf-8")
    counts = {split: len(split_members[split]) for split in ("train", "val", "test")}
    result = dict(dataset=expected_dataset, counts=counts, config=new_config.as_posix(),
                  protocol_id=new_data["protocol_id"], source_protocol_id=data["protocol_id"], check_only=check_only)
    if check_only:
        return result
    # Reject differing existing outputs before writing any file.
    for path, content in payloads.items():
        if path.exists() and path.read_bytes() != content:
            raise FileExistsError(f"拒绝覆盖不同的服务器协议：{path}")
    for path, content in payloads.items():
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
    if verify_protocol(new_data) != counts:
        raise ValueError("保存后的服务器协议与原始图片数量不一致")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("all", "sht_a", "sht_b"), default="all")
    parser.add_argument("--check-only", action="store_true", help="只核验原始上传数据，不写服务器配置")
    args = parser.parse_args()
    import torch
    print("Python:", sys.executable)
    print("Project directory:", ROOT.as_posix())
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print("GPU memory GiB:", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2))
    datasets = ("sht_a", "sht_b") if args.dataset == "all" else (args.dataset,)
    for dataset in datasets:
        print(json.dumps(prepare_dataset(ROOT, dataset, check_only=args.check_only), ensure_ascii=False, indent=2))
    print("核验完成，未写入任何配置。" if args.check_only else "服务器路径及上传文件已核对；现在可用 data_server.yaml 开始新的训练。")


if __name__ == "__main__":
    main()
