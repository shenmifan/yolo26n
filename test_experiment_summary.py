"""Downloaded-result regression checks: reject corrupted or mismatched evidence."""

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.summarize_experiments import audit_dataset


class ExperimentSummaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (ROOT / "runs/eval/shtb_density_v2_seed0_test/summary.json").is_file():
            raise unittest.SkipTest("Download the B seed0 results before this integration check")
        cls.original = audit_dataset(ROOT, "sht_b", 0)
        (ROOT / ".runtime").mkdir(exist_ok=True)

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(dir=ROOT / ".runtime")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        for relative in self.original["inputs_sha256"]:
            target = self.root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / relative, target)

    def edit_json(self, relative, change):
        path = self.root / relative
        content = json.loads(path.read_text(encoding="utf-8"))
        change(content)
        path.write_text(json.dumps(content), encoding="utf-8")

    def test_wrong_frozen_test_parameter_is_rejected(self):
        self.edit_json("runs/eval/shtb_density_v2_seed0_test/parameters.json",
                       lambda content: content["settings"].update(tau=0.5))
        with self.assertRaisesRegex(ValueError, "Frozen evaluation mismatch"):
            audit_dataset(self.root, "sht_b", 0)

    def test_old_diagnostic_cannot_be_promoted_to_formal_result(self):
        self.edit_json("runs/eval/shtb_density_v2_seed0_test/summary.json",
                       lambda content: content.update(diagnostic=True))
        with self.assertRaisesRegex(ValueError, "Diagnostic result"):
            audit_dataset(self.root, "sht_b", 0)

    def test_consistently_relabelled_protocol_is_still_rejected(self):
        for split in ("val", "test"):
            for filename in ("summary.json", "parameters.json"):
                self.edit_json(f"runs/eval/shtb_density_v2_seed0_{split}/{filename}",
                               lambda content: content.update(protocol_id="0" * 64))
        with self.assertRaisesRegex(ValueError, "reconstructed frozen manifests"):
            audit_dataset(self.root, "sht_b", 0)

    def test_changed_downloaded_checkpoint_is_rejected(self):
        path = self.root / "runs/crowd/shtb_density_v2_seed0/weights/best.pt"
        with path.open("ab") as handle:
            handle.write(b"corrupt-transfer")
        with self.assertRaisesRegex(ValueError, "checkpoint hash mismatch"):
            audit_dataset(self.root, "sht_b", 0)


if __name__ == "__main__":
    unittest.main()
