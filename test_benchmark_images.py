"""Runtime images bind to full frozen paths, including same-named train/test files."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from crowd.protocol import data_path, sha256_file
from scripts.benchmark import verify_benchmark_images


class BenchmarkImageTests(unittest.TestCase):
    def setUp(self):
        (ROOT / ".runtime").mkdir(exist_ok=True)
        self.directory = tempfile.TemporaryDirectory(dir=ROOT / ".runtime")
        self.addCleanup(self.directory.cleanup)
        self.temporary = Path(self.directory.name)
        self.root = self.temporary / "dataset"
        self.root.mkdir()
        self.train = self.root / "images/train/crowd.jpg"
        self.test = self.root / "images/val/crowd.jpg"
        self.write(self.train, b"frozen validation image from the official training pool")
        self.write(self.test, b"a distinct official test image with the same basename")
        self.data = dict(path=str(self.root), count_manifest="counts.json")
        self.records = [dict(image="images/train/crowd.jpg", gt_count=2,
                             image_sha256=sha256_file(self.train)),
                        dict(image="images/val/crowd.jpg", gt_count=5,
                             image_sha256=sha256_file(self.test))]
        self.save_records(self.records)

    @staticmethod
    def write(path, content):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    def save_records(self, records):
        (self.root / "counts.json").write_text(json.dumps(records), encoding="utf-8")

    def test_same_named_train_and_test_images_use_their_own_full_frozen_keys(self):
        actual = verify_benchmark_images(self.data, [str(self.train), str(self.test)])
        self.assertEqual(actual, {record["image"]: record["image_sha256"] for record in self.records})
        self.assertNotEqual(actual["images/train/crowd.jpg"], actual["images/val/crowd.jpg"])

    def test_tampering_is_rejected_even_if_the_other_same_named_hash_matches(self):
        self.train.write_bytes(self.test.read_bytes())
        self.assertEqual(sha256_file(self.train), self.records[1]["image_sha256"])
        with self.assertRaisesRegex(ValueError, "Benchmark image differs from frozen source content.*images/train/crowd.jpg"):
            verify_benchmark_images(self.data, [str(self.train)])

    def test_an_image_missing_from_the_frozen_manifest_is_rejected(self):
        self.save_records([self.records[1]])
        with self.assertRaises(ValueError):
            verify_benchmark_images(self.data, [str(self.train)])

    def test_duplicate_canonical_manifest_paths_are_rejected(self):
        duplicate = dict(self.records[0], image="images/train/../train/crowd.jpg")
        self.save_records([*self.records, duplicate])
        with self.assertRaises(ValueError):
            verify_benchmark_images(self.data, [str(self.train)])

    def test_a_selected_image_outside_the_dataset_root_is_rejected(self):
        outside = self.temporary / "crowd.jpg"
        outside.write_bytes(self.train.read_bytes())
        with self.assertRaises(ValueError):
            verify_benchmark_images(self.data, [str(outside)])

    def test_real_a_and_b_validation_images_match_the_frozen_full_paths(self):
        for dataset in ("sht_a", "sht_b"):
            with self.subTest(dataset=dataset):
                config = ROOT / "datasets" / f"{dataset}_density_v2/data.yaml"
                if not config.is_file():
                    self.skipTest("Local frozen ShanghaiTech datasets are not available")
                data = yaml.safe_load(config.read_text(encoding="utf-8"))
                manifest = data_path(data, data["val"])
                if not manifest.is_file():
                    self.skipTest("Local frozen validation manifest is not available")
                selected = sorted(line for line in manifest.read_text(encoding="utf-8").splitlines()
                                  if line.strip())[:16]
                if not all(Path(path).is_file() for path in selected):
                    self.skipTest("Local frozen validation images are not available")
                actual = verify_benchmark_images(data, selected)
                root = Path(data["path"]).resolve()
                self.assertEqual(len(actual), 16)
                self.assertEqual(set(actual), {Path(path).resolve().relative_to(root).as_posix()
                                              for path in selected})
                self.assertTrue(all(key.startswith("images/train/") for key in actual))
                for path in selected:
                    key = Path(path).resolve().relative_to(root).as_posix()
                    self.assertEqual(actual[key], sha256_file(path))


if __name__ == "__main__":
    unittest.main()
