"""Export descriptive test cases, verify counts against the frozen test CSV.

Selections are post-test diagnostics, never used for retuning the model or tau.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import torch
from crowd.paper_inference import input_tensor, load_counter, postprocess_count, prepare_image
from crowd.paper_utils import environment, get_device, load_data, make_dataset, write_json
from crowd.protocol import sha256_file
from scripts.paper_results import frozen_members
from scripts.summarize_experiments import image_key, read_csv, read_json, require


def select_cases(rows, mode="fused"):
    """Three GT quantiles plus maximum signed errors and fusion gain, with unique IDs."""
    ordered = sorted(rows, key=lambda row: (float(row["gt"]), row["image"]))
    candidates = []
    for label, fraction in (("sparse", .1), ("median", .5), ("dense", .9)):
        rank = round((len(ordered) - 1) * fraction)
        candidates.append((label, sorted(ordered, key=lambda row: (abs(ordered.index(row) - rank), row["image"]))))
    candidates += [("worst_underestimate", sorted(rows, key=lambda row: (float(row[mode]) - float(row["gt"]), row["image"]))),
                   ("worst_overestimate", sorted(rows, key=lambda row: (-(float(row[mode]) - float(row["gt"])), row["image"])))]
    if mode == "fused":
        candidates.append(("largest_fusion_gain", sorted(rows, key=lambda row: (-(float(row["abs_error_density"]) - float(row["abs_error_fused"])), row["image"]))))
    chosen, used = [], set()
    for label, options in candidates:
        row = next((row for row in options if row["image"] not in used), None)
        if row is not None:
            chosen.append((label, row))
            used.add(row["image"])
    return chosen


def plot_tau(folder, output):
    path = folder / "tau_sweep.csv"
    if not path.is_file():
        return
    rows = read_csv(path)
    selected = read_json(folder / "parameters.json")["settings"]["tau"]
    figure, axes = plt.subplots(1, 2, figsize=(8, 3), constrained_layout=True)
    for axis, metric in zip(axes, ("MAE", "RMSE")):
        axis.plot([float(row["tau"]) for row in rows], [float(row[metric]) for row in rows], marker="o", color="#235a86")
        axis.set(xlabel=r"Validation occupancy threshold $\tau$", ylabel=metric + " (persons)")
        axis.grid(alpha=.25)
        axis.axvline(selected, linestyle="--", color="#c45a2a", label=f"Selected tau={selected:g}")
        axis.legend(fontsize=8)
    figure.savefig(output / "validation_tau.png", dpi=220)
    figure.savefig(output / "validation_tau.svg")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("sht_a", "sht_b"), required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--params", required=True)
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--save-dir", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, help="Use the recorded evaluation batch size; legacy exports used 4")
    args = parser.parse_args()
    data, device = load_data(args.data), get_device(args.device)
    summary = read_json(args.test_dir / "summary.json")
    frozen = read_json(args.params)
    require(summary["source_split"] == "test" and summary["diagnostic"] is False, "Use independent test results")
    for key in ("protocol_id", "weights_sha256", "settings"):
        require(summary[key] == frozen[key], "Test results differ from frozen validation parameters")
    members, counts = frozen_members(ROOT, args.dataset, summary)
    rows = read_csv(args.test_dir / "per_image.csv")
    require(len(rows) == len(members["test"]) == summary["images"], "Incomplete test cases")
    require({image_key(row["image"], args.dataset) for row in rows} == members["test"], "Wrong test membership")
    gt = {row["image"]: row["gt_count"] for row in counts}
    image_hashes = {row["image"]: row["image_sha256"] for row in counts}
    require(all(float(row["gt"]) == gt[image_key(row["image"], args.dataset)] for row in rows), "Wrong original GT")
    model, method = load_counter(args.weights, frozen, device)
    settings = frozen["settings"]
    mode = "fused" if "fused" in summary["metrics"] else next(iter(summary["metrics"]))
    evaluation_batch = summary.get("evaluation_batch", 4) if args.batch is None else args.batch
    require(evaluation_batch > 0, "Batch size must be positive")
    selected = select_cases(rows, mode)
    dataset = make_dataset(data, "test", settings["imgsz"])
    indexes = {image_key(path, args.dataset): index for index, path in enumerate(dataset.im_files)}
    cases, cached_group, output = [], None, None
    with torch.inference_mode():
        for label, row in selected:
            key = image_key(row["image"], args.dataset)
            path = Path(data["path"]) / key
            require(sha256_file(path) == image_hashes[key], "Qualitative image differs from frozen source content")
            image = cv2.imread(str(path))
            if image is None:
                raise FileNotFoundError(path)
            tensor, canvas = prepare_image(image, settings["imgsz"])
            sample = dataset[indexes[key]]
            require(torch.equal(tensor[0], sample["img"]), "Qualitative preprocessing differs from evaluation")
            position = indexes[key]
            start = position // evaluation_batch * evaluation_batch
            if cached_group != start:
                # Use the same cohort shape as the original val.py evaluation:
                # cuDNN convolution arithmetic can differ between batch 1 and 4.
                cohort = torch.stack([dataset[i]["img"] for i in range(start, min(start + evaluation_batch, len(dataset)))])
                output = model(input_tensor(cohort, method, device))
                cached_group = start
            predicted, details = postprocess_count(output, method, settings, index=position - start)
            # Small FP32 differences are tolerated; a different count cannot be
            # silently shown with the old metric label (e.g. after an env change).
            for branch, value in predicted.items():
                tolerance = max(.02, 2e-5 * abs(float(row[branch])))
                require(abs(value - float(row[branch])) <= tolerance, f"Cannot reproduce {key}/{branch}; render with the original server environment")
            cases.append(dict(label=label, image=key, gt=int(row["gt"]), counts=predicted,
                              reference_counts={branch: float(row[branch]) for branch in predicted},
                              abs_error=abs(predicted[mode] - int(row["gt"])), canvas=canvas,
                              details=details, target_density=sample["density"].numpy(), image_sha256=sha256_file(path)))
    args.save_dir.mkdir(parents=True, exist_ok=False)
    den_max = max((float(case["details"]["density"].max()) for case in cases if case["details"]["density"] is not None), default=1.)
    den_min = min(0., min((float(case["details"]["density"].min()) for case in cases if case["details"]["density"] is not None), default=0.))
    # Shared density scale across all selected cases; cell selector uses fixed colors.
    for case in cases:
        panels = 4 if "fused" in case["counts"] else (3 if "detect" in case["counts"] else 2)
        figure, axes = plt.subplots(1, panels, figsize=(3.7 * panels, 3.8), constrained_layout=True)
        axes = np.atleast_1d(axes)
        rgb = case["canvas"][..., ::-1]
        axes[0].imshow(rgb)
        axes[0].set_title(f"{Path(case['image']).stem} | GT={case['gt']}\n{case['label']}", fontsize=10)
        next_panel = 1
        if "detect" in case["counts"]:
            axes[next_panel].imshow(rgb)
            boxes = case["details"]["boxes"]
            axes[next_panel].scatter((boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2, s=3, c="#00ffd0")
            axes[next_panel].set_title(f"Detection={case['counts']['detect']:.2f}\nNMS box centers", fontsize=10)
            next_panel += 1
        if case["details"]["density"] is not None:
            density = case["details"]["density"]
            heat = axes[next_panel].imshow(density, cmap="magma", vmin=den_min, vmax=max(den_max, 1e-8))
            axes[next_panel].set_title(f"Density={case['counts']['density']:.2f}\nShared intensity scale", fontsize=10)
            figure.colorbar(heat, ax=axes[next_panel], shrink=.65, label="persons / density pixel")
            next_panel += 1
        if "fused" in case["counts"]:
            axis = axes[next_panel]
            axis.imshow(rgb)
            selector = np.zeros((settings["grid"], settings["grid"], 4))
            high = case["details"]["high_mask"]
            selector[high] = (.95, .42, .1, .4)
            selector[~high] = (.1, .6, .95, .3)
            axis.imshow(selector, interpolation="nearest", extent=(0, settings["imgsz"], settings["imgsz"], 0))
            axis.set_title(f"DGF={case['counts']['fused']:.2f}, tau={settings['tau']:g}\nOrange: density; blue: detection", fontsize=10)
        for axis in axes:
            axis.axis("off")
        figure.savefig(args.save_dir / f"{case['label']}.png", dpi=200, bbox_inches="tight", pad_inches=.12)
        figure.savefig(args.save_dir / f"{case['label']}.svg", bbox_inches="tight", pad_inches=.12)
        plt.close(figure)
    plot_tau(Path(args.params).parent, args.save_dir)
    metadata = [{key: value for key, value in case.items() if key not in ("canvas", "details", "target_density")} for case in cases]
    write_json(args.save_dir / "cases.json", dict(purpose="post-test descriptive diagnostics; no retuning",
               selection="GT count quantiles 10/50/90%; signed-error extremes; maximum density-to-fusion improvement; unique image IDs",
               weights_sha256=sha256_file(args.weights), params_sha256=sha256_file(args.params),
               test_summary_sha256=sha256_file(args.test_dir / "summary.json"),
               test_csv_sha256=sha256_file(args.test_dir / "per_image.csv"), protocol_id=summary["protocol_id"],
               environment=environment(device), settings=settings, evaluation_batch=evaluation_batch, cases=metadata))
    print(f"Exported {len(cases)} verified cases: {args.save_dir.resolve()}")


if __name__ == "__main__":
    main()
