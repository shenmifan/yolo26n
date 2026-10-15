"""Regression tests for boundary kernels, count-preserving resize and target routing."""

import sys
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
import torch
import yaml
from ultralytics.cfg import get_cfg
from ultralytics.utils.instance import Instances

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from crowd.dataset import CrowdDataset, CrowdFlip, CrowdLetterBox
from crowd.protocol import data_path, verify_protocol
from crowd.density import audit_points, build_density_map, letterbox_density, resize_density_preserve_count, resize_density_preserve_count_torch


class DensityConservationTests(unittest.TestCase):
    def assert_mass(self, density, expected, tolerance=1e-4):
        array = density.detach().cpu().numpy() if isinstance(density, torch.Tensor) else density
        self.assertTrue(np.isfinite(array).all())
        self.assertTrue((array >= 0).all())
        self.assertLessEqual(abs(array.sum(dtype=np.float64) - expected), tolerance)

    def test_subpixel_corners_and_borders_each_have_unit_mass(self):
        for point in ([0, 0], [0.2, 0.7], [127.99, 127.99], [127.8, 60.3], [63.4, 0]):
            with self.subTest(point=point):
                self.assert_mass(build_density_map(np.array([point]), 128, 128, fixed_sigma=15), 1)

    def test_missing_border_mass_is_not_transferred_to_interior_person(self):
        edge = build_density_map(np.array([[0.1, 0.2]]), 128, 128, fixed_sigma=2)
        middle = build_density_map(np.array([[64.4, 64.3]]), 128, 128, fixed_sigma=2)
        combined = build_density_map(np.array([[0.1, 0.2], [64.4, 64.3]]), 128, 128, fixed_sigma=2)
        np.testing.assert_allclose(combined, edge + middle, rtol=1e-6, atol=1e-8)
        self.assert_mass(combined[:16, :16], 1)
        self.assert_mass(combined[48:80, 48:80], 1)

    def test_adaptive_kernel_also_preserves_count(self):
        points = np.array([[0.1, 0.1], [127.99, 127.99], [60.3, 70.4], [60.3, 70.4]])
        self.assert_mass(build_density_map(points, 128, 128), 4)

    def test_outside_points_are_audited_and_duplicates_preserved(self):
        points = np.array([[1, 1], [1, 1], [-1, 3], [31.99, 20], [40, 4], [np.nan, 7]])
        _, audit = audit_points(points, 32, 32)
        self.assertEqual((audit["raw_count"], audit["valid_count"], audit["outside_count"],
                          audit["nonfinite_count"], audit["duplicate_extra_count"]), (6, 3, 2, 1, 1))
        self.assert_mass(build_density_map(points, 32, 32, fixed_sigma=3), 3)

    def test_empty_density_stays_zero_at_every_size(self):
        density = build_density_map(np.empty((0, 2)), 31, 43, fixed_sigma=15)
        self.assert_mass(resize_density_preserve_count(density, 77, 101), 0)
        self.assert_mass(resize_density_preserve_count_torch(torch.from_numpy(density)[None, None], (7, 9)), 0)

    def test_sparse_noninteger_up_and_down_resize(self):
        density = np.zeros((73, 101), dtype=np.float32)
        density[0, 0], density[-1, -1], density[19, 23] = 1, 2, 3
        for h, w in ((257, 389), (19, 27), (640, 640), (11, 223)):
            with self.subTest(size=(h, w)):
                self.assert_mass(resize_density_preserve_count(density, h, w), 6)

    def test_real_letterbox_and_flip_preserve_spatial_alignment(self):
        density = build_density_map(np.array([[2.3, 3.4], [100.9, 70.8]]), 73, 101, fixed_sigma=3)
        image = np.zeros((73, 101, 3), dtype=np.uint8)
        image[:, :20] = 200
        transformed = CrowdLetterBox((320, 320), scaleup=False, density_stride=4)(dict(img=image, density_full=density))
        target, canvas = transformed["density"].clone(), transformed["img"].copy()
        transformed["instances"] = Instances(np.zeros((0, 4), dtype=np.float32), bbox_format="xywh", normalized=False)
        flipped = CrowdFlip(p=1.0, direction="horizontal")(transformed)
        self.assert_mass(target, 2)
        self.assert_mass(flipped["density"], 2)
        self.assertTrue(torch.equal(flipped["density"], torch.flip(target, [-1])))
        np.testing.assert_array_equal(flipped["img"], np.fliplr(canvas))

    def test_torch_resize_preserves_mass_and_backpropagation(self):
        density = torch.zeros(2, 1, 73, 101)
        density[0, 0, 0, 0] = 2
        density[1, 0, -1, -1] = 3
        density.requires_grad_()
        resized = resize_density_preserve_count_torch(density, (19, 27))
        self.assert_mass(resized[0], 2)
        self.assert_mass(resized[1], 3)
        resized.sum().backward()
        self.assertTrue(torch.isfinite(density.grad).all())
        self.assertGreater(float(density.grad.abs().sum()), 0)

    def test_letterbox_refuses_silent_crop(self):
        with self.assertRaisesRegex(ValueError, "裁剪"):
            letterbox_density(np.ones((10, 10), dtype=np.float32), 1, 3, 3, 10, 10)

    def test_rebuilt_targets_are_loaded_and_bad_mass_is_rejected(self):
        config_path = ROOT / "datasets/sht_b_density_v2/data.yaml"
        if not config_path.exists():
            self.skipTest("Build the real B density targets first")
        data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        self.assertEqual(verify_protocol(data), dict(train=320, val=80, test=316))
        config = get_cfg(overrides=dict(imgsz=640, single_cls=True, rect=False, task="detect"))
        dataset = CrowdDataset(img_path=str(data_path(data, data["val"])), imgsz=640,
                               batch_size=2, augment=False, hyp=config, rect=False,
                               single_cls=True, stride=32, pad=0.0, task="detect", data=data)
        sample = dataset[0]
        expected = dataset.density_count_lookup[str(Path(sample["im_file"]).resolve()).replace("\\", "/").casefold()]
        self.assert_mass(sample["density"], expected)
        self.assertTrue(Path(dataset.density_root).resolve().is_relative_to(ROOT / "datasets/sht_b_density_v2"))
        with patch("crowd.dataset.np.load", return_value=np.zeros((768, 1024), dtype=np.float32)):
            with self.assertRaisesRegex(ValueError, "计数不守恒"):
                dataset.get_image_and_label(0)


if __name__ == "__main__":
    unittest.main()
