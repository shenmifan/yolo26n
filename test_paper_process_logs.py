"""Paper child-process logs retain the evidence needed to diagnose interruptions."""
import contextlib
import io
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_paper_experiments as runner


class PaperProcessLogTests(unittest.TestCase):
    def setUp(self):
        (ROOT / ".runtime").mkdir(exist_ok=True)
        self.directory = tempfile.TemporaryDirectory(dir=ROOT / ".runtime")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def one_log(self):
        logs = list((self.root / "runs/paper_logs").glob("*.log"))
        self.assertEqual(len(logs), 1, "each child invocation must save its own log")
        return logs[0].read_text(encoding="utf-8")

    def test_stdout_and_stderr_chinese_are_preserved_with_success_exit_code(self):
        code = (r"import sys; print('stdout \u4e2d\u6587\u8bad\u7ec3\u7ed3\u679c'); "
                r"print('stderr \u4e2d\u6587\u8bca\u65ad\u4fe1\u606f', file=sys.stderr)")
        command = [sys.executable, "-c", code]
        console = io.StringIO()
        with patch.object(runner, "ROOT", self.root), contextlib.redirect_stdout(console):
            runner.execute(command)
        log = self.one_log()
        self.assertIn("stdout 中文训练结果", log)
        self.assertIn("stderr 中文诊断信息", log)
        self.assertIn(str(self.root), log)
        self.assertIn(sys.executable, log)
        self.assertRegex(log, r"Exit code:\s*0\s*$")
        self.assertIn("stdout 中文训练结果", console.getvalue())
        self.assertIn("stderr 中文诊断信息", console.getvalue())

    def test_nonzero_process_preserves_diagnostics_and_raises_original_exit_code(self):
        code = (r"import sys; print('\u8bad\u7ec3\u4e2d\u65ad\u4e4b\u524d\u7684\u8bb0\u5f55'); "
                r"print('\u8bca\u65ad\uff1a\u6a21\u62df\u5931\u8d25', file=sys.stderr); sys.exit(7)")
        command = [sys.executable, "-c", code]
        with patch.object(runner, "ROOT", self.root), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(subprocess.CalledProcessError) as raised:
                runner.execute(command)
        self.assertEqual(raised.exception.returncode, 7)
        self.assertEqual(list(map(str, raised.exception.cmd)), command)
        log = self.one_log()
        self.assertIn("训练中断之前的记录", log)
        self.assertIn("诊断：模拟失败", log)
        self.assertRegex(log, r"Exit code:\s*7\s*$")


if __name__ == "__main__":
    unittest.main()
