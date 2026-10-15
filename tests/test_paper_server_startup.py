"""Restore missing migrated-server metadata without overwriting frozen evidence."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_server_data import prepare_dataset
from scripts.run_paper_experiments import ensure_server_data
import test_server_relocation as fixtures


class PaperServerStartupTests(unittest.TestCase):
    def setUp(self):
        (ROOT / ".runtime").mkdir(exist_ok=True)
        self.directory = tempfile.TemporaryDirectory(dir=ROOT / ".runtime")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.dataset, self.target, _ = fixtures.ServerRelocationTests().make_fixture(self.root)

    def seed0_records(self, protocol_id):
        for split in ("val", "test"):
            path = self.root / "runs/eval" / f"shta_density_v2_seed0_{split}/summary.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(dict(protocol_id=protocol_id, source_split=split)), encoding="utf-8")
        weights = self.root / "runs/crowd/shta_density_v2_seed0/weights/best.pt"
        weights.parent.mkdir(parents=True, exist_ok=True)
        weights.write_bytes(b"preserved checkpoint fixture")

    def snapshot(self):
        return {path: path.read_bytes() for path in self.root.rglob("*") if path.is_file()}

    def test_missing_config_restores_matching_seed0_protocol_and_preserves_originals(self):
        preview = prepare_dataset(self.root, "sht_a", check_only=True)
        self.seed0_records(preview["protocol_id"])
        originals = self.snapshot()
        result = ensure_server_data(self.root, "sht_a")
        self.assertEqual(result["protocol_id"], preview["protocol_id"])
        self.assertTrue((self.target / "data_server.yaml").is_file())
        for path, content in originals.items():
            self.assertEqual(path.read_bytes(), content)

    def test_different_seed0_protocol_fails_before_writing_any_sidecars(self):
        self.seed0_records("wrong-root-or-protocol")
        before = self.snapshot()
        with self.assertRaisesRegex(ValueError, "seed0 不一致"):
            ensure_server_data(self.root, "sht_a")
        self.assertEqual(self.snapshot(), before)
        self.assertFalse((self.target / "data_server.yaml").exists())

    def test_conflicting_partial_sidecar_is_never_overwritten(self):
        sidecar = self.dataset / "splits_seed42_server/train.txt"
        sidecar.parent.mkdir(parents=True)
        sidecar.write_text("an incompatible migration\n", encoding="utf-8")
        before = self.snapshot()
        with self.assertRaisesRegex(FileExistsError, "拒绝覆盖"):
            ensure_server_data(self.root, "sht_a")
        self.assertEqual(self.snapshot(), before)
        self.assertFalse((self.target / "data_server.yaml").exists())

    def test_existing_config_is_verified_without_preparing_again(self):
        prepare_dataset(self.root, "sht_a")
        before = self.snapshot()
        with patch("scripts.prepare_server_data.prepare_dataset", side_effect=AssertionError("must not regenerate")):
            data = ensure_server_data(self.root, "sht_a")
        self.assertEqual(Path(data["path"]).resolve(), self.dataset.resolve())
        self.assertEqual(self.snapshot(), before)


if __name__ == "__main__":
    unittest.main()
