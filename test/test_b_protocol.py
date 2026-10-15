"""Regression checks for count thresholds, GT integrity and split leakage."""

import copy
import importlib.util
import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml
from ultralytics.cfg import get_cfg

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from crowd.protocol import verify_protocol
from crowd.validator import CrowdValidator, counting_postprocess
from val import create_save_dir, mae_rmse, validate_request

spec = importlib.util.spec_from_file_location("split_sht_b", ROOT / "scripts/split_sht_b.py")
split_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(split_module)


class CountEvaluationTests(unittest.TestCase):
    def test_count_cutoff_removes_background_and_nms_removes_duplicate(self):
        # Seven proposals: two overlapping true detections plus five weak backgrounds.
        decoded = torch.tensor([[[20, 20, 80, 120, 160, 200, 240],
                                 [20, 20, 20, 20, 20, 20, 20],
                                 [10, 10, 10, 10, 10, 10, 10],
                                 [10, 10, 10, 10, 10, 10, 10],
                                 [0.9, 0.8, 0.01, 0.01, 0.01, 0.01, 0.01]]], dtype=torch.float32)
        original = decoded.clone()
        self.assertEqual(len(counting_postprocess(decoded, 0.25, 0.5, 2000)[0]), 1)
        self.assertEqual(len(counting_postprocess(decoded, 0.001, 0.5, 2000)[0]), 6)
        self.assertTrue(torch.equal(decoded, original))
        with self.assertRaises(ValueError):
            counting_postprocess(torch.zeros(1, 300, 6), 0.25, 0.5, 2000)

    def test_original_count_not_density_mass_or_filtered_box_count(self):
        with tempfile.TemporaryDirectory() as directory:
            validator = CrowdValidator(save_dir=Path(directory), args=get_cfg(overrides={"conf": None}))
            self.assertEqual(validator.args.conf, 0.25)
            validator.data = {"grid": 1, "dgf_tau": 0}
            validator.init_metrics(SimpleNamespace(names={0: "head"}, end2end=False))
            # GT=100, mass=80, only 12 pseudo boxes. Correct density MAE is 20.
            batch = dict(img=torch.zeros(1, 3, 64, 64), gt_count=torch.tensor([100]),
                         batch_idx=torch.zeros(12, dtype=torch.long))
            validator.update_metrics([(torch.zeros(0, 6), torch.full((1, 16, 16), 80 / 256))], batch)
            self.assertEqual(validator.get_stats()["metrics/MAE_den"], 20)

    def test_rmse_definition(self):
        metric = mae_rmse([10, 20], [13, 16])
        self.assertEqual(metric["MAE"], 3.5)
        self.assertAlmostEqual(metric["RMSE"], math.sqrt(12.5))

    def test_test_set_cannot_select_tau(self):
        args = SimpleNamespace(split="test", sweep_tau=True, diagnostic=True, params=None)
        with self.assertRaisesRegex(ValueError, "禁止在测试集"):
            validate_request(args, {}, SimpleNamespace())

    def test_repeated_test_preserves_existing_results(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "seed0_test"
            self.assertEqual(create_save_dir(target, "test"), target)
            original = target / "summary.json"
            original.write_text('{"MAE": 104.075}', encoding="utf-8")
            second = create_save_dir(target, "test")
            third = create_save_dir(target, "test")
            self.assertEqual(second, target.with_name("seed0_test_2"))
            self.assertEqual(third, target.with_name("seed0_test_3"))
            self.assertFalse(list(second.iterdir()))
            self.assertEqual(original.read_text(encoding="utf-8"), '{"MAE": 104.075}')

    def test_existing_calibration_keeps_parameter_destination_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "seed0_val"
            create_save_dir(target, "val")
            parameters = target / "parameters.json"
            parameters.write_text('{"tau": 1.0}', encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "--params"):
                create_save_dir(target, "val")
            self.assertEqual(parameters.read_text(encoding="utf-8"), '{"tau": 1.0}')
            self.assertFalse(target.with_name("seed0_val_2").exists())

    def test_seeded_split_is_disjoint_reproducible_and_stratified(self):
        rows = [dict(image=f"image_{i:03d}.jpg", gt_count=i) for i in range(400)]
        train, val = split_module.stratified_split(rows)
        self.assertEqual((len(train), len(val)), (320, 80))
        self.assertEqual((train, val), split_module.stratified_split(rows))
        self.assertFalse({r["image"] for r in train} & {r["image"] for r in val})
        self.assertEqual([sum(i * 80 <= r["gt_count"] < (i + 1) * 80 for r in val) for i in range(5)], [16] * 5)

    def test_prepared_protocol_and_old_checkpoint_guard(self):
        config_path = ROOT / "datasets/sht_b/data_seed42.yaml"
        if not config_path.exists():
            self.skipTest("Run scripts/split_sht_b.py for the real-data check")
        data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        self.assertEqual(verify_protocol(data), {"train": 320, "val": 80, "test": 316})
        changed = copy.deepcopy(data)
        changed["protocol_id"] = "changed"
        with self.assertRaises(ValueError):
            verify_protocol(changed)
        args = SimpleNamespace(split="val", sweep_tau=False, diagnostic=False, params=None)
        with self.assertRaisesRegex(ValueError, "权重未按"):
            validate_request(args, data, SimpleNamespace(crowd_cfg={}))


if __name__ == "__main__":
    unittest.main()
