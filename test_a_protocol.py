"""Part A integration: split integrity, checkpoint guards and real-data gradients."""

import copy
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import yaml
from ultralytics.cfg import get_cfg
from ultralytics.data.utils import check_det_dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from crowd.dataset import CrowdDataset
from crowd.model import CrowdDetectionModel
from crowd.protocol import data_path, path_key, verify_protocol
from scripts.split_sht_a import exclude_test_duplicates, stratified_split
from val import validate_request

torch.set_num_threads(1)


class AProtocolTests(unittest.TestCase):
    def test_split_is_reproducible_and_balances_count_strata(self):
        rows = [dict(image=f"image_{i:03d}.jpg", gt_count=i) for i in range(300)]
        train, val = stratified_split(rows)
        self.assertEqual((len(train), len(val)), (240, 60))
        self.assertEqual((train, val), stratified_split(list(reversed(rows))))
        self.assertNotEqual(val, stratified_split(rows, seed=43)[1])
        self.assertFalse({r["image"] for r in train} & {r["image"] for r in val})
        self.assertEqual([sum(i * 60 <= r["gt_count"] < (i + 1) * 60 for r in val) for i in range(5)], [12] * 5)
        with self.assertRaises(ValueError):
            stratified_split(rows[:-1])

    def test_duplicate_test_content_is_excluded_before_selection(self):
        rows = [dict(image=f"train_{i}.jpg", gt_count=i, image_sha256=str(i)) for i in range(300)]
        tests = [dict(image="official_test.jpg", image_sha256="121")]
        eligible, excluded = exclude_test_duplicates(rows, tests)
        self.assertEqual(len(excluded), 1)
        self.assertEqual(excluded[0]["duplicate_of"], "official_test.jpg")
        train, val = stratified_split(eligible, excluded_count=len(excluded))
        self.assertEqual((len(train), len(val)), (239, 60))
        self.assertFalse({r["image_sha256"] for r in train + val} & {r["image_sha256"] for r in tests})

    def load_data(self):
        config_path = ROOT / "datasets/sht_a_density_v2/data.yaml"
        if not config_path.exists():
            self.skipTest("Prepare the Part A protocol and rebuilt supervision first")
        data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        self.assertEqual(verify_protocol(data), dict(train=239, val=60, test=182))
        return data, config_path

    def test_official_test_is_complete_and_not_used_for_selection(self):
        data, _ = self.load_data()
        records = json.loads(data_path(data, data["count_manifest"]).read_text(encoding="utf-8"))
        records = {path_key(data_path(data, row["image"])): row for row in records}
        for split, expected_source in (("train", "train_data"), ("val", "train_data"), ("test", "test_data")):
            images = data_path(data, data[split]).read_text(encoding="utf-8").splitlines()
            self.assertEqual({records[path_key(p)]["source_split"] for p in images}, {expected_source})
            if split == "test":
                self.assertEqual({Path(p).stem for p in images},
                                 {p.stem for p in (ROOT / "datasets/sht_a/images/val").glob("*.jpg")})
        test_hashes = {r["image_sha256"] for r in records.values() if r["source_split"] == "test_data"}
        selected = [path_key(p) for split in ("train", "val")
                    for p in data_path(data, data[split]).read_text(encoding="utf-8").splitlines()]
        self.assertFalse({records[p]["image_sha256"] for p in selected} & test_hashes)
        self.assertEqual(data_path(data, data["excluded"]).read_text(encoding="utf-8").strip(),
                         (ROOT / "datasets/sht_a/images/train/shta_IMG_121.jpg").as_posix())

    def test_manifest_tampering_and_wrong_checkpoint_are_rejected(self):
        data, _ = self.load_data()
        changed = copy.deepcopy(data)
        changed["split_protocol"]["sha256"]["val"] = "0" * 64
        with self.assertRaises(ValueError):
            verify_protocol(changed)
        changed = copy.deepcopy(data)
        changed["split_protocol"]["sha256"]["excluded"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "排除图片清单已改变"):
            verify_protocol(changed)
        changed = copy.deepcopy(data)
        changed["excluded"] = data["train"]
        changed["split_protocol"]["sha256"]["excluded"] = data["split_protocol"]["sha256"]["train"]
        with self.assertRaisesRegex(ValueError, "与实验划分重叠"):
            verify_protocol(changed)
        args = SimpleNamespace(split="val", sweep_tau=True, diagnostic=False, params=None)
        matching = SimpleNamespace(crowd_cfg={"protocol_id": data["protocol_id"]})
        validate_request(args, data, matching)
        with self.assertRaisesRegex(ValueError, "权重未按"):
            validate_request(args, data, SimpleNamespace(crowd_cfg={"protocol_id": "B-or-old-protocol"}))
        args.split = "test"
        with self.assertRaisesRegex(ValueError, "禁止在测试集"):
            validate_request(args, data, matching)

    def make_dataset(self, augment=False):
        data, config_path = self.load_data()
        parsed = check_det_dataset(str(config_path), autodownload=False)
        self.assertEqual(parsed["split_protocol"]["dataset"], "ShanghaiTech-A")
        cfg = get_cfg(overrides=dict(imgsz=320, single_cls=True, rect=False, task="detect", fliplr=1.0))
        dataset = CrowdDataset(img_path=str(data_path(data, data["train"])), imgsz=320,
                               batch_size=2, augment=augment, hyp=cfg, rect=False,
                               single_cls=True, stride=32, pad=0.0, task="detect", data=data)
        self.assertEqual(len(dataset), 239)
        return dataset, data, cfg

    def test_rebuilt_density_and_original_counts_reach_the_loader(self):
        dataset, data, _ = self.make_dataset()
        sample = dataset[0]
        image_key = path_key(sample["im_file"])
        self.assertEqual(int(sample["gt_count"]), dataset.count_lookup[image_key])
        self.assertLess(abs(float(sample["density"].double().sum()) - dataset.density_count_lookup[image_key]), 1e-4)
        self.assertTrue(Path(dataset.density_root).resolve().is_relative_to(ROOT / "datasets/sht_a_density_v2"))
        with patch("crowd.dataset.np.load", return_value=np.zeros((10, 10), dtype=np.float32)):
            with self.assertRaisesRegex(ValueError, "计数不守恒"):
                dataset.get_image_and_label(0)

    def test_real_a_batch_forward_and_both_branch_gradients(self):
        dataset, data, cfg = self.make_dataset(augment=True)
        batch = dataset.collate_fn([dataset[0], dataset[1]])
        batch["img"] = batch["img"].float() / 255.0
        for image_index in range(2):
            expected = dataset.density_count_lookup[path_key(batch["im_file"][image_index])]
            self.assertLess(abs(float(batch["density"][image_index].double().sum()) - expected), 1e-4)
        model = CrowdDetectionModel(str(ROOT / "configs/dc-yolo26n.yaml"), nc=1, verbose=False)
        model.args = cfg
        model.crowd_cfg = {key: data[key] for key in ("grid", "dgf_tau", "protocol_id")}
        model.train()
        predictions = model(batch["img"])
        loss, items = model.init_criterion()(predictions, batch)
        self.assertTrue(torch.isfinite(loss).all())
        self.assertEqual(len(items), 6)
        loss.sum().backward()
        for branch in (model.density_head, model.model[-1]):
            gradients = [p.grad for p in branch.parameters() if p.grad is not None]
            self.assertTrue(gradients)
            self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
            self.assertGreater(sum(float(g.abs().sum()) for g in gradients), 0)


if __name__ == "__main__":
    unittest.main()
