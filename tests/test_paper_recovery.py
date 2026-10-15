"""Interrupted paper jobs restart only after preserving all related evidence."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_paper_experiments as runner


class PaperRecoveryTests(unittest.TestCase):
    def setUp(self):
        (ROOT / ".runtime").mkdir(exist_ok=True)
        self.directory = tempfile.TemporaryDirectory(dir=ROOT / ".runtime")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.name = runner.run_name("sht_b", "full", 1)
        self.run = self.root / "runs/crowd" / self.name
        self.val = self.root / "runs/eval" / f"{self.name}_val"
        self.test = self.root / "runs/eval" / f"{self.name}_test"
        self.write(self.run / "weights/best.pt", b"best: interrupted checkpoint")
        self.write(self.run / "weights/last.pt", b"last: interrupted checkpoint")
        self.write(self.run / "results.csv", b"epoch,loss\n1,42\n2,32\n")
        self.write(self.run / "args.yaml", b"seed: 1\nbatch: 4\n")
        self.write(self.val / "parameters.json", b'{"old": "calibration"}\n')
        self.write(self.val / "summary.json", b'{"old": "validation"}\n')
        self.write(self.test / "per_image.csv", b"image,gt\nold.jpg,5\n")
        self.write(self.test / "nested/diagnostic.bin", bytes(range(256)))
        self.other = self.root / "runs/crowd/shta_density_v2_seed2/weights/best.pt"
        self.write(self.other, b"another complete experiment")
        self.other_eval = self.root / "runs/eval/shtb_density_v2_seed0_test/summary.json"
        self.write(self.other_eval, b'{"preserve": "seed0"}\n')

    @staticmethod
    def write(path, content):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    def snapshot(self):
        return {path.relative_to(self.root): path.read_bytes()
                for path in self.root.rglob("*") if path.is_file()}

    def args(self, restart=False):
        return SimpleNamespace(epochs=300, batch=4, workers=4, device="cpu",
                               restart_incomplete=restart, vgg_weights=None)

    def test_archive_preserves_every_file_and_moves_both_evaluation_directories(self):
        before = self.snapshot()
        targets = (self.run, self.val, self.test)
        archive = runner.archive_incomplete_run(self.root, "sht_b", "full", 1)
        self.assertTrue(archive.is_relative_to(self.root / "runs/paper_recovery"))
        for source in targets:
            self.assertFalse(source.exists())
        for relative, content in before.items():
            original = self.root / relative
            if any(original.is_relative_to(source) for source in targets):
                self.assertEqual((archive / relative).read_bytes(), content)
            else:
                self.assertEqual(original.read_bytes(), content)
        record = json.loads((archive / "recovery.json").read_text(encoding="utf-8"))
        self.assertEqual((record["dataset"], record["variant"], record["seed"]), ("sht_b", "full", 1))
        self.assertEqual(record["mode"], "restart_from_pretrained")
        self.assertEqual(record["reason"], "missing training_complete.json")
        expected = {source.relative_to(self.root).as_posix() for source in targets}
        self.assertEqual({item["original"] for item in record["directories"]}, expected)
        self.assertEqual({item["archived"] for item in record["directories"]}, expected)

    def test_existing_completion_ledger_is_never_archived(self):
        self.write(self.run / "training_complete.json", b'{"completed": true}\n')
        before = self.snapshot()
        with self.assertRaises((ValueError, FileExistsError)):
            runner.archive_incomplete_run(self.root, "sht_b", "full", 1)
        self.assertEqual(self.snapshot(), before)

    def test_second_rename_failure_restores_the_first_directory(self):
        before = self.snapshot()
        original_rename = Path.rename
        calls = []

        def fail_second(path, target):
            calls.append((path, target))
            if len(calls) == 2:
                raise OSError("simulated second directory rename failure")
            return original_rename(path, target)

        with patch.object(Path, "rename", autospec=True, side_effect=fail_second):
            with self.assertRaisesRegex(OSError, "simulated second directory"):
                runner.archive_incomplete_run(self.root, "sht_b", "full", 1)
        self.assertGreaterEqual(len(calls), 3, "the first rename must be rolled back")
        self.assertEqual(self.snapshot(), before)
        for source in (self.run, self.val, self.test):
            self.assertTrue(source.is_dir())

    def test_absent_restart_flag_still_rejects_an_untracked_directory(self):
        before = self.snapshot()
        with patch.object(runner, "ROOT", self.root), \
             patch.object(runner, "audit_run", return_value=None), \
             patch.object(runner, "load_data", return_value={"protocol_id": "frozen"}), \
             patch.object(runner, "archive_incomplete_run") as archive, \
             patch.object(runner, "execute") as execute:
            with self.assertRaisesRegex(FileExistsError, "Incomplete or untracked"):
                runner.train_evaluate("sht_b", "full", 1, self.args(), self.root / "source_archive", {})
            archive.assert_not_called()
            execute.assert_not_called()
        self.assertEqual(self.snapshot(), before)

    def test_restart_flag_preserves_old_outputs_before_starting_fresh_training(self):
        before = self.snapshot()
        with patch.object(runner, "ROOT", self.root), \
             patch.object(runner, "audit_run", return_value=None), \
             patch.object(runner, "load_data", return_value={"protocol_id": "frozen"}), \
             patch.object(runner, "execute", side_effect=RuntimeError("fresh training reached")) as execute:
            with self.assertRaisesRegex(RuntimeError, "fresh training reached"):
                runner.train_evaluate("sht_b", "full", 1, self.args(restart=True), self.root / "source_archive", {})
        execute.assert_called_once()
        command = list(map(str, execute.call_args.args[0]))
        self.assertIn(str(self.root / "train.py"), command)
        self.assertNotIn("--resume", command)
        self.assertEqual(command[command.index("--name") + 1], self.name)
        archives = list((self.root / "runs/paper_recovery").glob("*/recovery.json"))
        self.assertEqual(len(archives), 1)
        archive = archives[0].parent
        for source in (self.run, self.val, self.test):
            self.assertFalse(source.exists())
            for relative, content in before.items():
                if (self.root / relative).is_relative_to(source):
                    self.assertEqual((archive / relative).read_bytes(), content)

    def test_audited_completed_job_skips_even_when_restart_is_requested(self):
        before = self.snapshot()
        with patch.object(runner, "ROOT", self.root), \
             patch.object(runner, "audit_run", return_value={"metrics": "verified"}), \
             patch.object(runner, "load_data") as load, \
             patch.object(runner, "archive_incomplete_run") as archive, \
             patch.object(runner, "execute") as execute:
            runner.train_evaluate("sht_b", "full", 1, self.args(restart=True), self.root / "source_archive", {})
            archive.assert_not_called()
            execute.assert_not_called()
            load.assert_not_called()
        self.assertEqual(self.snapshot(), before)

    def test_csrnet_restart_archives_train_and_eval_then_restarts_the_baseline(self):
        name = runner.run_name("sht_b", "csrnet", 0)
        run = self.root / "runs/crowd" / name
        val = self.root / "runs/eval" / f"{name}_val"
        test = self.root / "runs/eval" / f"{name}_test"
        self.write(run / "weights/last.pt", b"CSRNet interrupted at epoch 14")
        self.write(run / "baseline_config.json", b'{"method": "csrnet", "seed": 0}\n')
        self.write(val / "summary.json", b'{"old": "CSRNet validation"}\n')
        self.write(test / "per_image.csv", b"image,gt\nold-csrnet.jpg,5\n")
        before = self.snapshot()
        with patch.object(runner, "ROOT", self.root), \
             patch.object(runner, "audit_run", return_value=None), \
             patch.object(runner, "load_data", return_value={"protocol_id": "frozen"}), \
             patch.object(runner, "execute", side_effect=RuntimeError("fresh baseline reached")) as execute:
            with self.assertRaisesRegex(RuntimeError, "fresh baseline reached"):
                runner.train_evaluate("sht_b", "csrnet", 0, self.args(restart=True), self.root / "source_archive", {})
        execute.assert_called_once()
        command = list(map(str, execute.call_args.args[0]))
        self.assertIn(str(self.root / "scripts/train_baseline.py"), command)
        self.assertNotIn("--resume", command)
        self.assertEqual(command[command.index("--method") + 1], "csrnet")
        self.assertEqual(command[command.index("--save-dir") + 1], str(run))
        archives = list((self.root / "runs/paper_recovery").glob("*/recovery.json"))
        self.assertEqual(len(archives), 1)
        archive = archives[0].parent
        targets = (run, val, test)
        for source in targets:
            self.assertFalse(source.exists())
        for relative, content in before.items():
            original = self.root / relative
            if any(original.is_relative_to(source) for source in targets):
                self.assertEqual((archive / relative).read_bytes(), content)
            else:
                self.assertEqual(original.read_bytes(), content)


if __name__ == "__main__":
    unittest.main()
