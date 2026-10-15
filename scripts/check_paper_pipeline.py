"""One-epoch synthetic train -> val -> test checks, isolated from paper results."""
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
from test_paper_experiments import synthetic_protocol
from crowd.paper_utils import write_json
from crowd.protocol import sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    workspace = ROOT / ".runtime" / ("paper_pipeline_" + datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid4().hex[:6])
    workspace.mkdir(parents=True, exist_ok=False)
    data = synthetic_protocol(workspace / "data")
    environment = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    commands = []
    for variant in ("det_p2", "no_p2"):
        run = workspace / "train" / variant
        common = ["--data", data, "--batch", 2, "--imgsz", 64, "--device", args.device, "--workers", 0]
        commands.append([sys.executable, ROOT / "train.py", "--variant", variant, "--pretrained", "",
                         "--epochs", 1, "--project", workspace / "train", "--name", variant, *common])
        commands.append([sys.executable, ROOT / "val.py", "--weights", run / "weights/best.pt", "--split", "val",
                         "--save-dir", workspace / "eval" / f"{variant}_val", *(["--sweep-tau"] if variant == "no_p2" else []), *common])
        commands.append([sys.executable, ROOT / "val.py", "--weights", run / "weights/best.pt", "--split", "test",
                         "--params", workspace / "eval" / f"{variant}_val/parameters.json",
                         "--save-dir", workspace / "eval" / f"{variant}_test", *common])
    commands.append([sys.executable, ROOT / "scripts/train_baseline.py", "--method", "mcnn", "--data", data,
                     "--save-dir", workspace / "train/mcnn", "--eval-root", workspace / "eval",
                     "--epochs", 1, "--batch", 2, "--imgsz", 64, "--device", args.device, "--workers", 0])
    for index, command in enumerate(commands):
        log = workspace / f"step_{index + 1}.log"
        print(f"Smoke {index + 1}/{len(commands)}: {Path(command[1]).name}; log={log}", flush=True)
        with log.open("w", encoding="utf-8") as handle:
            subprocess.run([str(item) for item in command], cwd=ROOT, env=environment,
                           stdout=handle, stderr=subprocess.STDOUT, check=True)
    for variant in ("det_p2", "no_p2", "mcnn"):
        val = json.loads((workspace / "eval" / f"{variant}_val/parameters.json").read_text(encoding="utf-8"))
        test = json.loads((workspace / "eval" / f"{variant}_test/summary.json").read_text(encoding="utf-8"))
        assert test["images"] == 2 and test["gt_source"] == "original_points"
        assert val["settings"] == test["settings"]
        assert val["weights_sha256"] == test["weights_sha256"] == sha256_file(workspace / "train" / variant / "weights/best.pt")
        assert set(test["metrics"]) == ({"detect"} if variant == "det_p2" else {"density"} if variant == "mcnn" else {"detect", "density", "fused"})
    write_json(workspace / "smoke_verified.json", dict(purpose="synthetic integration only; no paper measurement", steps=len(commands)))
    print(f"PASS: synthetic training/calibration/testing; {workspace}")


if __name__ == "__main__":
    main()
