"""Plot audited validation sensitivity and frozen test scatters, no rerunning inference."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from crowd.experiments import run_name
from crowd.protocol import sha256_file
from scripts.paper_results import audit_run
from scripts.summarize_experiments import read_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-dir", type=Path, default=ROOT / "paper/SIVP_20261009/figures")
    args = parser.parse_args()
    args.save_dir.mkdir(parents=True, exist_ok=True)
    audits, paths = {}, {}
    for dataset in ("sht_a", "sht_b"):
        audits[dataset] = audit_run(ROOT, dataset, "full", args.seed)
        if audits[dataset] is None:
            raise FileNotFoundError(f"Missing audited {dataset} seed{args.seed} results")
        name = run_name(dataset, "full", args.seed)
        paths[dataset] = {split: ROOT / "runs/eval" / f"{name}_{split}" for split in ("val", "test")}
    figure, axes = plt.subplots(2, 2, figsize=(8.5, 6), constrained_layout=True)
    for row, dataset in enumerate(audits):
        sweep = read_csv(paths[dataset]["val"] / "tau_sweep.csv")
        selected = audits[dataset]["settings"]["tau"]
        for column, metric in enumerate(("MAE", "RMSE")):
            axis = axes[row, column]
            axis.plot([float(item["tau"]) for item in sweep], [float(item[metric]) for item in sweep], "o-", color="#235a86", markersize=4)
            axis.axvline(selected, color="#c45a2a", linestyle="--", label=f"Validation selection: {selected:g}")
            axis.set(xlabel=r"Occupancy threshold $\tau$", ylabel=metric + " (persons)", title=f"ShanghaiTech-{dataset[-1].upper()}, seed{args.seed}")
            axis.grid(alpha=.2)
            axis.legend(fontsize=8)
    for extension in ("png", "svg"):
        figure.savefig(args.save_dir / f"validation_sensitivity_seed{args.seed}.{extension}", dpi=250)
    plt.close(figure)
    figure, axes = plt.subplots(2, 3, figsize=(10.5, 7), constrained_layout=True)
    for row, dataset in enumerate(audits):
        predictions = read_csv(paths[dataset]["test"] / "per_image.csv")
        gt = np.array([float(item["gt"]) for item in predictions])
        for column, mode in enumerate(("detect", "density", "fused")):
            axis = axes[row, column]
            pred = np.array([float(item[mode]) for item in predictions])
            limit = max(gt.max(), pred.max()) * 1.05
            axis.scatter(gt, pred, s=10, alpha=.6, color="#235a86", edgecolors="none")
            axis.plot([0, limit], [0, limit], "--", color="#c45a2a", linewidth=1)
            axis.set(xlim=(0, limit), ylim=(0, limit), xlabel="Original annotation count", ylabel="Predicted count",
                     title=f"Part {dataset[-1].upper()} | {mode} | MAE {audits[dataset]['metrics'][mode]['MAE']:.2f}")
            axis.set_aspect("equal", adjustable="box")
            axis.grid(alpha=.2)
    for extension in ("png", "svg"):
        figure.savefig(args.save_dir / f"test_scatter_seed{args.seed}.{extension}", dpi=250)
    plt.close(figure)
    inputs = [folder / filename for split_folders in paths.values() for split, folder in split_folders.items()
              for filename in (("tau_sweep.csv", "parameters.json") if split == "val" else ("per_image.csv", "summary.json"))]
    record = dict(purpose="Figures from audited measured CSVs; no inference or retuning", seed=args.seed,
                  weights={dataset: audit["weights_sha256"] for dataset, audit in audits.items()},
                  protocols={dataset: audit["protocol_id"] for dataset, audit in audits.items()},
                  input_hashes={path.relative_to(ROOT).as_posix(): sha256_file(path) for path in inputs})
    (args.save_dir / f"figure_audit_seed{args.seed}.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"Measured validation and test figures: {args.save_dir}")


if __name__ == "__main__":
    main()
