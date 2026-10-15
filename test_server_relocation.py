"""Check foreign-path migration preserves membership and rejects damaged uploads."""

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from crowd.protocol import verify_protocol
from scripts.prepare_server_data import prepare_dataset, relocated_path


def digest(content):
    return hashlib.sha256(content).hexdigest()


class ServerRelocationTests(unittest.TestCase):
    def make_fixture(self, root, old_project="D:/TEST-1/dc-yolo26-crowd"):
        old_dataset = old_project + "/datasets/sht_a"
        old_density = old_project + "/datasets/sht_a_density_v2"
        dataset = root / "datasets/sht_a"
        target = root / "datasets/sht_a_density_v2"
        target.mkdir(parents=True)
        counts, supervision, hashes = [], [], {}
        for index, split in enumerate(("train", "val", "test", "excluded")):
            image = f"images/{'val' if split == 'test' else 'train'}/{split}.jpg"
            density = image.replace("images/", "density/").replace(".jpg", ".npy")
            for path, content in ((dataset / image, f"image-{index}".encode()),
                                  (dataset / image.replace("images/", "labels/").replace(".jpg", ".txt"), b"0 0.5 0.5 0.1 0.1\n"),
                                  (target / density, f"density-{index}".encode())):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
            counts.append(dict(image=image, gt_count=index + 1, image_sha256=digest(f"image-{index}".encode())))
            supervision.append(dict(image=image, split=split, raw_count=index + 1, density_count=index + 1,
                                    density_path=f"{old_density}/{density}", density_sha256=digest(f"density-{index}".encode())))
            content = (old_dataset + "/" + image + "\n").encode()
            path = dataset / f"splits_seed42/{split}.txt"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
            hashes[split] = digest(content)
        content = json.dumps(counts).encode()
        (dataset / "splits_seed42/counts.json").write_bytes(content)
        hashes["counts"] = digest(content)
        content = json.dumps(dict(spec={"source": old_project + "/../data/part_A"}, records=supervision)).encode()
        (target / "supervision_manifest.json").write_bytes(content)
        hashes["supervision"] = digest(content)
        data = dict(path=old_dataset, density_root=old_density + "/density",
                    supervision_manifest=old_density + "/supervision_manifest.json",
                    count_manifest="splits_seed42/counts.json", nc=1, names=["head"],
                    **{split: f"splits_seed42/{split}.txt" for split in ("train", "val", "test", "excluded")},
                    split_protocol=dict(dataset="ShanghaiTech-A", seed=42, sha256=hashes),
                    protocol_id=digest(json.dumps(hashes, sort_keys=True).encode()))
        (target / "data.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")
        return dataset, target, data

    def test_windows_paths_are_relocated_without_changing_original_files(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            dataset, target, old_data = self.make_fixture(root)
            originals = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
            result = prepare_dataset(root, "sht_a")
            data = yaml.safe_load(Path(result["config"]).read_text(encoding="utf-8"))
            self.assertEqual(verify_protocol(data), dict(train=1, val=1, test=1))
            self.assertEqual(data["relocated_from_protocol_id"], old_data["protocol_id"])
            for split in ("train", "val", "test", "excluded"):
                content = (dataset / data[split]).read_text(encoding="utf-8")
                self.assertIn(dataset.as_posix(), content)
                self.assertNotIn("D:/TEST-1", content)
            for path, content in originals.items():
                self.assertEqual(path.read_bytes(), content)
            self.assertEqual(prepare_dataset(root, "sht_a"), result)

    def test_posix_source_paths_are_supported(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.make_fixture(root, old_project="/home/old/project")
            self.assertEqual(prepare_dataset(root, "sht_a")["counts"], dict(train=1, val=1, test=1))

    def test_check_only_validates_without_creating_any_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.make_fixture(root)
            files = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
            result = prepare_dataset(root, "sht_a", check_only=True)
            self.assertTrue(result["check_only"])
            self.assertEqual(result["counts"], dict(train=1, val=1, test=1))
            self.assertEqual({p: p.read_bytes() for p in root.rglob("*") if p.is_file()}, files)

    def test_damaged_density_is_rejected_before_creating_server_config(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, target, _ = self.make_fixture(root)
            (target / "density/train/train.npy").write_bytes(b"damaged-upload")
            with self.assertRaisesRegex(ValueError, "哈希不同"):
                prepare_dataset(root, "sht_a")
            self.assertFalse((target / "data_server.yaml").exists())

    def test_paths_cannot_escape_the_expected_data_root(self):
        for value in ("../outside.jpg", "C:/other/image.jpg", "/other/image.jpg"):
            with self.assertRaises(ValueError):
                relocated_path(value, "D:/old/data", ROOT / "datasets/sht_a")


if __name__ == "__main__":
    unittest.main()
