"""Sequential paper experiment runner for the existing PyCharm SSH interpreter.

Without --execute, prints a concrete job plan. Completed, audited runs are skipped.
Each new joint model is trained, calibrated on val, then tested with frozen params.
Interrupted directories are preserved. --restart-incomplete archives them before
starting a fresh train/val/test run; it does not resume checkpoint optimization.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from crowd.experiments import EXTERNAL, VARIANTS, model_config, run_name
from crowd.paper_utils import environment, get_device, load_data, write_json
from crowd.protocol import sha256_file
from scripts.paper_results import audit_run, build_report


def training_jobs(stage, datasets, seeds):
    selected = {"repeats": ("full",), "ablations": tuple(key for key in VARIANTS if key != "full"),
                "baselines": EXTERNAL, "all": tuple(VARIANTS) + EXTERNAL}.get(stage, ())
    return [(dataset, variant, seed) for variant in selected for dataset in datasets for seed in seeds]


def ensure_server_data(project_root, dataset):
    """Recover missing host paths without changing originals or archived seed0."""
    project_root = Path(project_root).resolve()
    config = project_root / "datasets" / f"{dataset}_density_v2/data_server.yaml"
    if not config.exists():
        from scripts.prepare_server_data import prepare_dataset

        print(f"Missing {config}; verifying uploaded data and restoring server paths...", flush=True)
        preview = prepare_dataset(project_root, dataset, check_only=True)
        # A migrated server may have the same images at a different absolute root.
        # Do not create a new protocol and silently mix it with the old seed0.
        for split in ("val", "test"):
            record = project_root / "runs/eval" / f"{run_name(dataset, 'full', 0)}_{split}/summary.json"
            if record.is_file():
                original = json.loads(record.read_text(encoding="utf-8"))
                if original["protocol_id"] != preview["protocol_id"]:
                    raise ValueError(
                        f"恢复的数据协议与已保存 seed0 不一致：{record}。"
                        "请核对冻结原始文件，并沿用 seed0 的服务器项目路径；未写入任何服务器配置。"
                    )
        result = prepare_dataset(project_root, dataset)
        if result["protocol_id"] != preview["protocol_id"]:
            raise RuntimeError("恢复期间数据协议发生变化")
        print(f"Server data ready: {config}; counts={result['counts']}; protocol={result['protocol_id']}", flush=True)
    return load_data(config)


def execute(command):
    command = [str(item) for item in command]
    rendered = subprocess.list2cmdline(command)
    log_dir = ROOT / "runs/paper_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / (datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid4().hex[:8] + ".log")
    print(f"\nRunning: {rendered}\nLog: {log_path}", flush=True)
    child_env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    with log_path.open("x", encoding="utf-8", buffering=1) as log:
        log.write(f"Command: {rendered}\nWorking directory: {ROOT}\n")
        with subprocess.Popen(command, cwd=ROOT, env=child_env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                              errors="replace", bufsize=1) as process:
            try:
                for line in process.stdout:
                    log.write(line)
                    print(line, end="", flush=True)
                returncode = process.wait()
            except BaseException:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                log.write(f"\nExecution interrupted. Child exit code: {process.returncode}\n")
                raise
        log.write(f"\nExit code: {returncode}\n")
    if returncode:
        print(f"Job stopped with exit code {returncode}; saved log: {log_path}", flush=True)
        raise subprocess.CalledProcessError(returncode, command)


def archive_source(device):
    destination = ROOT / "runs/paper_manifests" / (datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid4().hex[:8])
    destination.mkdir(parents=True, exist_ok=False)
    sources = sorted(set(ROOT.glob("crowd/*.py")) | set(ROOT.glob("scripts/*.py")) | set(ROOT.glob("configs/*.yaml")) | {ROOT / "train.py", ROOT / "val.py", ROOT / "requirements.txt"})
    hashes = {}
    for source in sources:
        relative = source.relative_to(ROOT)
        target = destination / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        hashes[relative.as_posix()] = sha256_file(source)
    write_json(destination / "source_hashes.json", hashes)
    write_json(destination / "environment.json", environment(device))
    freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    (destination / "pip_freeze.txt").write_text(freeze, encoding="utf-8")
    return destination, hashes


def assert_existing_train(run, data, variant, seed, recipe):
    if not (run / "training_complete.json").is_file():
        raise FileExistsError(f"Incomplete or untracked training directory: {run}. It is preserved. Inspect the interrupted job, or use --restart-incomplete to archive its train/val/test directories and restart from pretrained weights.")
    ledger = json.loads((run / "training_complete.json").read_text(encoding="utf-8"))
    for key, value in dict(protocol_id=data["protocol_id"], variant=variant, seed=seed, recipe=recipe).items():
        if ledger[key] != value:
            raise ValueError(f"Existing training record differs: {key}")
    if sha256_file(run / "weights/best.pt") != ledger["weights_sha256"]:
        raise ValueError("Completed training checkpoint changed")


def archive_incomplete_run(project_root, dataset, variant, seed):
    """Preserve all outputs of one untracked run before an explicit fresh restart."""
    project_root = Path(project_root).resolve()
    name = run_name(dataset, variant, seed)
    run = project_root / "runs/crowd" / name
    if (run / "training_complete.json").exists():
        raise FileExistsError(f"Completed training is preserved; cannot restart: {run}")
    if not run.is_dir():
        raise FileNotFoundError(run)
    sources = [run] + [project_root / "runs/eval" / f"{name}_{split}" for split in ("val", "test")]
    sources = [path for path in sources if path.exists() or path.is_symlink()]
    base = project_root / "runs/paper_recovery"
    # Validate resolved targets before any mkdir or rename, including parent links.
    for path in [base, *sources]:
        if not path.is_relative_to(project_root) or path.resolve() != path:
            raise ValueError(f"Recovery path must stay in the project without directory links: {path}")
    if any(not path.is_dir() for path in sources):
        raise ValueError("Recovery requires directories for train/val/test outputs")
    backup = base / (name + "_" + datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid4().hex[:8])
    backup.mkdir(parents=True, exist_ok=False)
    moved = []
    try:
        for source in sources:
            target = backup / source.relative_to(project_root)
            target.parent.mkdir(parents=True, exist_ok=True)
            source.rename(target)
            moved.append((source, target))
        write_json(backup / "recovery.json", dict(
            dataset=dataset, variant=variant, seed=seed,
            reason="missing training_complete.json", mode="restart_from_pretrained",
            directories=[dict(original=source.relative_to(project_root).as_posix(),
                              archived=target.relative_to(backup).as_posix()) for source, target in moved],
        ))
    except Exception:
        for source, target in reversed(moved):
            target.rename(source)
        raise
    print(f"Archived incomplete train/val/test: {backup}\nRestarting {name} from pretrained initialization.", flush=True)
    return backup


def train_evaluate(dataset, variant, seed, args, archive, source_hashes):
    name = run_name(dataset, variant, seed)
    run = ROOT / "runs/crowd" / name
    if audit_run(ROOT, dataset, variant, seed) is not None:
        print(f"Audited complete; skip {name}", flush=True)
        return
    data_path = ROOT / "datasets" / f"{dataset}_density_v2/data_server.yaml"
    data = load_data(data_path)
    recipe = dict(epochs=args.epochs, batch=args.batch, imgsz=640, workers=args.workers, patience=100)
    common = ["--data", data_path, "--seed", seed, "--epochs", args.epochs, "--patience", 100,
              "--batch", args.batch, "--imgsz", 640, "--device", args.device, "--workers", args.workers]
    if run.exists() and not (run / "training_complete.json").is_file() and getattr(args, "restart_incomplete", False):
        archive_incomplete_run(ROOT, dataset, variant, seed)
    if run.exists():
        assert_existing_train(run, data, variant, seed, recipe)
    else:
        if variant in EXTERNAL:
            command = [sys.executable, ROOT / "scripts/train_baseline.py", "--method", variant,
                       "--save-dir", run, "--eval-root", ROOT / "runs/eval", *common]
            if args.vgg_weights:
                command += ["--vgg-weights", args.vgg_weights]
        else:
            command = [sys.executable, ROOT / "train.py", "--variant", variant,
                       "--pretrained", ROOT / "weights/yolo26n.pt", "--project", ROOT / "runs/crowd", "--name", name, *common]
        execute(command)
        configuration = model_config(variant) if variant in VARIANTS else None
        write_json(run / "experiment.json", dict(dataset=dataset, variant=variant, seed=seed,
                   description=VARIANTS[variant]["description"] if variant in VARIANTS else "Matched architecture reproduction",
                   protocol_id=data["protocol_id"], recipe=recipe, model_config=configuration,
                   source_archive=str(archive), source_hashes=source_hashes, command=list(map(str, command)),
                   pretrained_sha256=sha256_file(ROOT / "weights/yolo26n.pt") if variant in VARIANTS else None))
        write_json(run / "training_complete.json", dict(protocol_id=data["protocol_id"], variant=variant,
                   seed=seed, recipe=recipe, weights_sha256=sha256_file(run / "weights/best.pt")))
    if variant not in EXTERNAL:
        weights = run / "weights/best.pt"
        val_dir = ROOT / "runs/eval" / f"{name}_val"
        test_dir = ROOT / "runs/eval" / f"{name}_test"
        options = ["--weights", weights, "--data", data_path, "--batch", args.batch,
                   "--device", args.device, "--workers", args.workers]
        if not (val_dir / "summary.json").exists():
            execute([sys.executable, ROOT / "val.py", "--split", "val", "--save-dir", val_dir,
                     *(["--sweep-tau"] if VARIANTS[variant]["density"] else []), *options])
        if not (test_dir / "summary.json").exists():
            execute([sys.executable, ROOT / "val.py", "--split", "test", "--save-dir", test_dir,
                     "--params", val_dir / "parameters.json", *options])
    if audit_run(ROOT, dataset, variant, seed) is None:
        raise RuntimeError(f"No complete final evaluation for {name}")
    print(f"Audited train/val/test complete: {name}", flush=True)


def export_extras(stage, datasets, args):
    variants = tuple(VARIANTS) + EXTERNAL if stage == "speed" else ("full",)
    for dataset in datasets:
        for variant in variants:
            name = run_name(dataset, variant, 0)
            if audit_run(ROOT, dataset, variant, 0) is None:
                print(f"Pending checkpoint: {name}; skip {stage}", flush=True)
                continue
            output = ROOT / "runs" / ("benchmark" if stage == "speed" else "qualitative") / name
            marker = "summary.json" if stage == "speed" else "cases.json"
            if (output / marker).is_file():
                saved = json.loads((output / marker).read_text(encoding="utf-8"))
                weights = ROOT / "runs/crowd" / name / "weights/best.pt"
                params = ROOT / "runs/eval" / f"{name}_val/parameters.json"
                if saved["weights_sha256"] != sha256_file(weights) or saved["params_sha256"] != sha256_file(params):
                    raise ValueError(f"Existing {stage} belongs to a different checkpoint/parameters")
                if stage == "speed" and saved["purpose"] != "runtime_measurement":
                    raise ValueError("A smoke timing cannot replace the server runtime measurement")
                print(f"Complete export; skip {output}")
                continue
            common = ["--weights", ROOT / "runs/crowd" / name / "weights/best.pt",
                      "--params", ROOT / "runs/eval" / f"{name}_val/parameters.json",
                      "--data", ROOT / "datasets" / f"{dataset}_density_v2/data_server.yaml",
                      "--device", args.device, "--save-dir", output]
            command = [sys.executable, ROOT / "scripts" / ("benchmark.py" if stage == "speed" else "qualitative.py"), *common]
            if stage == "qualitative":
                command += ["--dataset", dataset, "--test-dir", ROOT / "runs/eval" / f"{name}_test"]
            execute(command)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("repeats", "ablations", "baselines", "speed", "qualitative", "report", "all"), default="repeats")
    parser.add_argument("--dataset", choices=("all", "sht_a", "sht_b"), default="all")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--vgg-weights", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--restart-incomplete", action="store_true", help="Preserve unfinished train/val/test directories in runs/paper_recovery and train again from initialization; completed runs are skipped")
    parser.add_argument("--allow-env-change", action="store_true", help="Explicitly allow a new environment; the aggregator will still reject pooling unlike environments")
    args = parser.parse_args()
    if len(args.seeds) != len(set(args.seeds)) or any(seed < 0 for seed in args.seeds):
        parser.error("Use unique nonnegative training seeds")
    datasets = ("sht_a", "sht_b") if args.dataset == "all" else (args.dataset,)
    jobs = training_jobs(args.stage, datasets, args.seeds)
    pending = []
    for dataset, variant, seed in jobs:
        done = audit_run(ROOT, dataset, variant, seed) is not None
        print(f"{'DONE' if done else 'PENDING'} {run_name(dataset, variant, seed)}")
        if not done:
            pending.append((dataset, variant, seed))
    print(f"New training jobs: {len(pending)}; sequential execution; stage={args.stage}", flush=True)
    if not args.execute:
        print("Plan only. Add --execute to run this plan on the configured server interpreter.")
        return
    if args.stage == "report":
        record = build_report(seeds=args.seeds, datasets=datasets)
        if record["invalid"]:
            raise RuntimeError(record["invalid"])
        print(ROOT / "reports/paper_experiments/results.md")
        return
    device = get_device(args.device)
    measured = environment(device)
    print(f"Python: {sys.executable}", flush=True)
    print(f"Environment: torch={measured['torch']}, ultralytics={measured['ultralytics']}, GPU={measured['device_name']}", flush=True)
    for dataset in datasets:
        reference = ROOT / "runs/eval" / f"{run_name(dataset, 'full', 0)}_test/summary.json"
        if reference.exists() and not args.allow_env_change:
            original = json.loads(reference.read_text(encoding="utf-8"))
            for key in ("torch", "ultralytics", "device_name"):
                if measured[key] != original[key]:
                    raise RuntimeError(f"Environment changed: {key}={measured[key]}, seed0 used {original[key]}. Use the original server environment for a matched repeated-run comparison.")
    for dataset in datasets:
        ensure_server_data(ROOT, dataset)
    archive, source_hashes = archive_source(device)
    for dataset, variant, seed in pending:
        train_evaluate(dataset, variant, seed, args, archive, source_hashes)
    for stage in (("speed", "qualitative") if args.stage == "all" else (args.stage,)):
        if stage in {"speed", "qualitative"}:
            export_extras(stage, datasets, args)
    record = build_report(seeds=(0, 1, 2), datasets=datasets)
    if record["invalid"]:
        raise RuntimeError(record["invalid"])
    print(f"Saved audit and available mean/std: {ROOT / 'reports/paper_experiments'}")


if __name__ == "__main__":
    main()
