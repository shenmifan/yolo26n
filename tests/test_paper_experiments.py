"""Meaningful guards for variants, independent baselines, seed pooling and timing."""
import copy
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
import torch
torch.set_num_threads(2)
from crowd.baselines import baseline_input, make_baseline
from crowd.experiments import VARIANTS, crowd_settings, model_config
from crowd.model import CrowdDetectionModel
from crowd.paper_inference import prepare_image
from crowd.paper_utils import load_data, make_dataset
from crowd.protocol import sha256_file
from crowd.validator import CrowdValidator
from scripts.benchmark import latency_summary
from scripts.paper_results import aggregate
from scripts.qualitative import select_cases
from scripts.run_paper_experiments import training_jobs
from val import evaluate


def synthetic_protocol(root):
    import hashlib
    import yaml
    records, hashes = [], {}
    for split in ("train", "val", "test"):
        paths = []
        for index in range(2):
            relative = f"images/{split}/image_{index}.jpg"
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(path), np.full((55, 91, 3), 60 + index * 20, dtype=np.uint8))
            density_path = root / "density" / split / f"image_{index}.npy"
            density_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(density_path, np.full((55, 91), 2 / (55 * 91), dtype=np.float32))
            label_path = root / "labels" / split / f"image_{index}.txt"
            label_path.parent.mkdir(parents=True, exist_ok=True)
            label_path.write_text("0 0.25 0.5 0.05 0.1\n0 0.75 0.5 0.05 0.1\n", encoding="utf-8")
            paths.append(path.as_posix())
            records.append(dict(image=relative, gt_count=2, image_sha256=sha256_file(path)))
        (root / f"{split}.txt").write_text("\n".join(paths) + "\n", encoding="utf-8")
        hashes[split] = sha256_file(root / f"{split}.txt")
    (root / "counts.json").write_text(json.dumps(records), encoding="utf-8")
    hashes["counts"] = sha256_file(root / "counts.json")
    data = dict(path=root.as_posix(), train="train.txt", val="val.txt", test="test.txt", nc=1,
                names=["head"], channels=3, count_manifest="counts.json", density_stride=4,
                grid=16, dgf_tau=2., count_max_det=2000,
                split_protocol=dict(seed=42, sha256=hashes),
                protocol_id=hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest())
    (root / "data.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")
    return root / "data.yaml"


class PaperExperimentTests(unittest.TestCase):
    def test_all_variants_backpropagate_and_have_expected_heads(self):
        image = torch.rand(2, 3, 64, 64)
        batch = dict(img=image, batch_idx=torch.tensor([0, 1]), cls=torch.zeros(2, 1),
                     bboxes=torch.tensor([[.5, .5, .1, .1], [.5, .5, .1, .1]]),
                     density=torch.full((2, 16, 16), 2 / 256))
        for name, spec in VARIANTS.items():
            with self.subTest(variant=name):
                model = CrowdDetectionModel(model_config(name), density_enabled=spec["density"])
                model.args = SimpleNamespace(box=7.5, cls=.5, dfl=1.5, l1=1., reg_max=1, nc=1)
                model.crowd_cfg = crowd_settings({}, name)
                model.train()
                predictions = model(image)
                self.assertEqual(predictions[1] is not None, spec["density"])
                if spec["density"]:
                    self.assertEqual(tuple(predictions[1].shape), (2, 1, 16, 16))
                criterion = model.init_criterion()
                loss, items = criterion(predictions, batch)
                self.assertTrue(torch.isfinite(loss).all())
                loss.sum().backward()
                self.assertTrue(any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in model.model.parameters()))
                if not spec["density"]:
                    self.assertIsNone(model.density_head)
                    self.assertEqual(float(items["den_loss"]), 0.)
                    metrics, rows = evaluate([dict(image="test.jpg", gt=2, label_count=2, target_density_sum=2,
                                                   density=None, boxes=np.empty((0, 4)), imgsz=64)], 16, 2)
                    self.assertEqual(set(metrics), {"detect"})
                else:
                    self.assertTrue(any(parameter.grad is not None for parameter in model.density_head.parameters()))
                    expected = crowd_settings({}, name)
                    self.assertEqual((criterion.lambda_den, criterion.lambda_cnt, criterion.lambda_cons),
                                     tuple(expected.get(key, default) for key, default in zip(("den_gain", "cnt_gain", "cons_gain"), (1., .1, .5))))
                if name in {"no_p2", "det_yolo26"}:
                    self.assertEqual(model.stride.tolist(), [8., 16., 32.])
                if name in {"no_recal", "det_p2", "no_p2", "det_yolo26"}:
                    self.assertFalse(any(type(module).__name__ == "EMA" for module in model.model))

    def test_both_baselines_train_with_conserved_targets(self):
        from crowd.density import resize_density_preserve_count_torch
        for name, stride in (("mcnn", 4), ("csrnet", 8)):
            with self.subTest(method=name):
                model = make_baseline(name)
                output = model(baseline_input(torch.full((1, 3, 64, 64), 127, dtype=torch.uint8), name))
                self.assertEqual(output.shape[-2:], (64 // stride, 64 // stride))
                target = resize_density_preserve_count_torch(torch.full((1, 1, 16, 16), 5 / 256), output.shape[-2:])
                self.assertAlmostEqual(float(target.sum()), 5., places=5)
                (output - target).square().mean().backward()
                self.assertTrue(torch.isfinite(model.output_layer.weight.grad).all())
                self.assertGreater(float(model.output_layer.weight.grad.abs().sum()), 0.)

    def test_csrnet_frontend_mapping_and_wrong_file_guard(self):
        from torchvision.models import VGG16_Weights
        from unittest.mock import patch
        model = make_baseline("csrnet")
        features = {f"features.{key}": torch.full_like(value, .123) for key, value in model.frontend.state_dict().items()}
        with patch.object(VGG16_Weights, "get_state_dict", return_value=features):
            model.load_vgg_frontend()
        self.assertTrue(all(torch.equal(value, features[f"features.{key}"]) for key, value in model.frontend.state_dict().items()))
        with tempfile.TemporaryDirectory(dir=ROOT / ".runtime") as directory:
            wrong = Path(directory) / "vgg16-397923af.pth"
            wrong.write_bytes(b"not the official checkpoint")
            with self.assertRaisesRegex(ValueError, "397923af"):
                model.load_vgg_frontend(wrong)

    def test_timed_preprocessing_matches_real_dataset(self):
        (ROOT / ".runtime").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / ".runtime") as temporary:
            root = Path(temporary)
            data = load_data(synthetic_protocol(root))
            dataset = make_dataset(data, "val", 64)
            for index, path in enumerate(dataset.im_files):
                expected, _ = prepare_image(cv2.imread(path), 64)
                self.assertTrue(torch.equal(expected[0], dataset[index]["img"]))

    def test_sample_std_and_missing_seed_are_explicit(self):
        report = dict(dataset="sht_a", variant="full", protocol_id="frozen", environment={"torch": "same"},
                      recipe={}, settings=dict(tau=1, imgsz=640), metrics=dict(fused=dict(MAE=10., RMSE=20., bias=1.)),
                      seed=0, weights_sha256="first")
        rows = aggregate([report], [0, 1, 2])
        self.assertIsNone(rows[0]["MAE_std"])
        second, third = copy.deepcopy(report), copy.deepcopy(report)
        second.update(seed=1, weights_sha256="second")
        third.update(seed=2, weights_sha256="third")
        second["metrics"]["fused"]["MAE"] = 12.
        third["metrics"]["fused"]["MAE"] = 14.
        third["settings"]["tau"] = 1.5
        rows = aggregate([report, second, third], [0, 1, 2])
        self.assertEqual(rows[0]["MAE_mean"], 12.)
        self.assertEqual(rows[0]["MAE_std"], 2.)
        self.assertEqual(rows[0]["status"], "complete")
        third["environment"]["torch"] = "different"
        with self.assertRaisesRegex(ValueError, "environment"):
            aggregate([report, third], [0, 1, 2])
        with self.assertRaisesRegex(ValueError, "Duplicate seed"):
            aggregate([report, report], [0, 1, 2])

    def test_timing_summary_uses_all_samples(self):
        summary = latency_summary([10., 20., 30.])
        self.assertEqual(summary["median_ms"], 20.)
        self.assertEqual(summary["serial_images_per_second"], 50.)
        self.assertEqual(summary["p95_ms"], 29.)
        with self.assertRaises(ValueError):
            latency_summary([0., float("nan")])

    def test_downloaded_seed0_passes_new_auditor(self):
        from scripts.paper_results import audit_run
        from scripts.summarize_experiments import audit_dataset
        for dataset in ("sht_a", "sht_b"):
            if not (ROOT / "runs/crowd" / ("shta_density_v2_seed0" if dataset == "sht_a" else "shtb_density_v2_seed0")).exists():
                self.skipTest("Downloaded seed0 checkpoint required")
            self.assertEqual(audit_run(ROOT, dataset, "full", 0)["metrics"], audit_dataset(ROOT, dataset, 0)["metrics"])

    def test_runtime_audit_rejects_smoke_and_changed_percentile(self):
        from crowd.experiments import run_name
        from crowd.paper_utils import write_json
        from scripts.paper_results import runtime_reports
        with tempfile.TemporaryDirectory(dir=ROOT / ".runtime") as directory:
            root = Path(directory)
            name = run_name("sht_b", "full", 0)
            weights = root / "runs/crowd" / name / "weights/best.pt"
            weights.parent.mkdir(parents=True)
            weights.write_bytes(b"fixture checkpoint for timing-metadata checks")
            params = root / "runs/eval" / f"{name}_val/parameters.json"
            write_json(params, {})
            samples = [10., 20.] * 100
            summary = dict(purpose="runtime_measurement", batch=1, precision="FP32", warmup=30, method="full",
                           weights_sha256=sha256_file(weights), params_sha256=sha256_file(params), parameter_count=1,
                           environment=dict(device_name="test", torch="test", ultralytics="test"),
                           metrics={scope: latency_summary(samples) for scope in ("network_only", "end_to_end")})
            folder = root / "runs/benchmark" / name
            write_json(folder / "timings_ms.json", dict(network_only=samples, end_to_end=samples))
            write_json(folder / "summary.json", summary)
            self.assertEqual(len(runtime_reports(root, ("sht_b",), ("full",))[0]), 1)
            summary["purpose"] = "smoke"
            write_json(folder / "summary.json", summary)
            with self.assertRaisesRegex(ValueError, "Smoke"):
                runtime_reports(root, ("sht_b",), ("full",))
            summary["purpose"] = "runtime_measurement"
            summary["metrics"]["end_to_end"]["p95_ms"] += 1
            write_json(folder / "summary.json", summary)
            with self.assertRaisesRegex(ValueError, "p95"):
                runtime_reports(root, ("sht_b",), ("full",))

    def test_diagnostic_selection_is_deterministic_and_unique(self):
        rows = [dict(image=f"image_{index:02}.jpg", gt=index * 10, fused=index * 10 + index - 4,
                     abs_error_fused=abs(index - 4), abs_error_density=index * 2) for index in range(10)]
        cases = select_cases(rows)
        self.assertEqual(cases, select_cases(list(reversed(rows))))
        self.assertEqual(len({row["image"] for _, row in cases}), 6)

    def test_qualitative_export_matches_saved_synthetic_counts(self):
        import shutil
        from unittest.mock import patch
        from crowd.paper_inference import input_tensor, postprocess_count
        from crowd.paper_utils import write_json
        from scripts.summarize_experiments import image_key
        from scripts import qualitative
        from val import write_csv
        with tempfile.TemporaryDirectory(dir=ROOT / ".runtime") as directory:
            root = Path(directory)
            data_path = synthetic_protocol(root / "datasets/sht_b")
            data = load_data(data_path)
            dataset = make_dataset(data, "test", 64)
            model = CrowdDetectionModel(model_config("full")).eval()
            model.crowd_cfg = dict(protocol_id=data["protocol_id"], variant="full")
            model.end2end = False
            settings = dict(imgsz=64, conf=.25, iou=.5, max_det=2000, grid=16, tau=2.)
            with torch.inference_mode():
                output = model(input_tensor(torch.stack([dataset[i]["img"] for i in range(2)]), "full", "cpu"))
                rows = []
                for index, path in enumerate(dataset.im_files):
                    predictions, _ = postprocess_count(output, "full", settings, index=index)
                    row = dict(image=path, gt=2, **predictions)
                    for branch in predictions:
                        row[f"abs_error_{branch}"] = abs(row[branch] - 2)
                    rows.append(row)
            weights = root / "model.pt"
            torch.save(dict(model=model), weights)
            params = dict(source_split="val", diagnostic=False, weights_sha256=sha256_file(weights),
                          protocol_id=data["protocol_id"], settings=settings)
            val_params = root / "val/parameters.json"
            write_json(val_params, params)
            write_json(root / "test/summary.json", dict(params, source_split="test", images=2,
                       metrics=dict(detect={}, density={}, fused={}), evaluation_batch=2))
            write_csv(root / "test/per_image.csv", rows)
            members = dict(test={image_key(path, "sht_b") for path in dataset.im_files})
            counts = json.loads((root / "datasets/sht_b/counts.json").read_text())
            argv = ["qualitative.py", "--dataset", "sht_b", "--weights", str(weights), "--params", str(val_params),
                    "--test-dir", str(root / "test"), "--data", str(data_path), "--save-dir", str(root / "render"), "--device", "cpu"]
            with patch.object(sys, "argv", argv), patch.object(qualitative, "frozen_members", return_value=(members, counts)):
                qualitative.main()
            record = json.loads((root / "render/cases.json").read_text())
            self.assertEqual(len(record["cases"]), 2)
            for case in record["cases"]:
                for branch, value in case["counts"].items():
                    self.assertAlmostEqual(value, case["reference_counts"][branch], places=4)
            image = cv2.imread(str(root / "render/sparse.png"))
            self.assertGreater(image.shape[1], 2000)
            # A retained rendering fixture for visual QA, never a paper case.
            shutil.copyfile(root / "render/sparse.png", ROOT / ".runtime/synthetic_qualitative_render.png")

    def test_repeat_plan_has_only_training_seeds_not_new_splits(self):
        jobs = training_jobs("repeats", ("sht_a", "sht_b"), (0, 1, 2))
        self.assertEqual(len(jobs), 6)
        self.assertTrue(all(variant == "full" for _, variant, _ in jobs))


if __name__ == "__main__":
    unittest.main()
