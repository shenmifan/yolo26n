"""Measure actual batch-one FP32 network and full counting pipeline latency.

End-to-end timing includes resize/letterbox, RGB/tensor conversion, normalization,
host-to-device transfer, inference, decoding/NMS, device-to-host transfer and fusion.
Disk decode, visualization, target-density loading and report export are excluded.
"""
import argparse
import json
import sys
import time
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
import torch
from crowd.paper_inference import input_tensor, load_counter, postprocess_count, prepare_image
from crowd.paper_utils import environment, get_device, load_data, make_dataset, write_json
from crowd.protocol import data_path, sha256_file


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def latency_summary(milliseconds):
    values = np.asarray(milliseconds, dtype=np.float64)
    if len(values) == 0 or not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Invalid timing samples")
    return dict(iterations=len(values), mean_ms=float(values.mean()), median_ms=float(np.median(values)),
                p95_ms=float(np.percentile(values, 95)), serial_images_per_second=float(1000. / values.mean()))


def verify_benchmark_images(data, paths):
    """Check frozen image bytes by dataset-relative path, preserving split identity."""
    root = Path(data["path"]).resolve()
    originals = json.loads(data_path(data, data["count_manifest"]).read_text(encoding="utf-8"))

    def image_key(path):
        resolved = Path(path).resolve()
        if not resolved.is_relative_to(root):
            raise ValueError(f"Benchmark image path is outside the dataset: {path}")
        return resolved.relative_to(root).as_posix()

    expected = {}
    for row in originals:
        key = image_key(data_path(data, row["image"]))
        if key in expected:
            raise ValueError(f"Duplicate frozen benchmark image record: {key}")
        expected[key] = row["image_sha256"]
    verified = {}
    for path in paths:
        key = image_key(path)
        if key not in expected:
            raise ValueError(f"Benchmark image is missing from frozen source manifest: {key}")
        if key in verified:
            raise ValueError(f"Duplicate selected benchmark image: {key}")
        observed = sha256_file(path)
        if observed != expected[key]:
            raise ValueError(f"Benchmark image differs from frozen source content: {key}; expected {expected[key]}, observed {observed}")
        verified[key] = observed
    return verified


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--params", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--save-dir", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--images", type=int, default=16)
    parser.add_argument("--smoke", action="store_true", help="Mark a small local check; do not use it as server paper timing")
    args = parser.parse_args()
    if min(args.warmup, args.iterations, args.images) <= 0:
        parser.error("warmup/iterations/images must be positive")
    if not args.smoke and (args.warmup < 30 or args.iterations < 200):
        parser.error("Formal runtime needs >=30 warmups and >=200 repetitions; use --smoke for a small check")
    device, data = get_device(args.device), load_data(args.data)
    parameters = json.loads(Path(args.params).read_text(encoding="utf-8"))
    settings = parameters["settings"]
    model, method = load_counter(args.weights, parameters, device)
    paths = sorted(line for line in data_path(data, data["val"]).read_text(encoding="utf-8").splitlines() if line.strip())[:args.images]
    image_hashes = verify_benchmark_images(data, paths)
    images = [cv2.imread(path) for path in paths]
    if any(image is None for image in images):
        raise FileNotFoundError("Cannot decode a fixed validation benchmark image")
    # Verify the timed preprocessing against the actual evaluation dataset.
    dataset = make_dataset(data, "val", settings["imgsz"])
    by_file = {str(Path(path).resolve()): index for index, path in enumerate(dataset.im_files)}
    for path, image in zip(paths, images):
        actual = dataset[by_file[str(Path(path).resolve())]]["img"]
        expected, _ = prepare_image(image, settings["imgsz"])
        if not torch.equal(actual, expected[0]):
            raise ValueError("Runtime preprocessing differs from the frozen counting evaluation")
    fixed = [input_tensor(prepare_image(image, settings["imgsz"])[0], method, device) for image in images]
    args.save_dir.mkdir(parents=True, exist_ok=False)
    times = {"network_only": [], "end_to_end": []}
    memory = {}
    with torch.inference_mode():
        for scope in times:
            def execute(index):
                if scope == "network_only":
                    return model(fixed[index % len(fixed)])
                tensor, _ = prepare_image(images[index % len(images)], settings["imgsz"])
                output = model(input_tensor(tensor, method, device))
                counts, _ = postprocess_count(output, method, settings, include_map=False)
                if not all(np.isfinite(value) for value in counts.values()):
                    raise RuntimeError("Nonfinite prediction while benchmarking")
                return counts
            for index in range(args.warmup):
                execute(index)
            synchronize(device)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            for index in range(args.iterations):
                synchronize(device)
                started = time.perf_counter()
                execute(index)
                synchronize(device)
                times[scope].append((time.perf_counter() - started) * 1000.)
            memory[scope] = dict(peak_allocated_bytes=torch.cuda.max_memory_allocated(device), peak_reserved_bytes=torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else None
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    operations = dict(status="unavailable", convention="THOP MACs; FLOPs=2*MACs; postprocessing excluded; actual one-to-many eval path")
    try:
        from thop import profile
        with torch.inference_mode():
            macs, _ = profile(deepcopy(model), inputs=(fixed[0],), verbose=False)
        operations.update(status="measured_estimate", MACs=macs, FLOPs=2 * macs,
                          limitations="THOP counts registered module operations; functional interpolation, Softplus, decode and other functional tensor operations may be omitted")
    except (ImportError, RuntimeError, TypeError, AttributeError) as error:
        operations["reason"] = str(error)
    record = dict(purpose="smoke" if args.smoke else "runtime_measurement", method=method, batch=1,
                  precision="FP32", fused_modules=False, settings=settings, warmup=args.warmup,
                  sample_selection="First sorted frozen-validation paths; cycle through all selected images",
                  image_hashes=image_hashes,
                  weights_sha256=sha256_file(args.weights), params_sha256=sha256_file(args.params),
                  source_protocol_id=parameters["protocol_id"], local_data_protocol_id=data["protocol_id"],
                  environment=environment(device), parameter_count=parameter_count, operations=operations,
                  metrics={scope: latency_summary(values) for scope, values in times.items()}, memory=memory,
                  includes="End-to-end: CPU preprocessing + H2D + model + NMS + D2H + count/fusion",
                  excludes="Disk I/O/decode, target loading, rendering, report writes",
                  model_note="Existing CrowdDetectionModel forces .model.float() inside forward; this cost is retained")
    write_json(args.save_dir / "summary.json", record)
    write_json(args.save_dir / "timings_ms.json", times)
    print(json.dumps(record["metrics"], indent=2))


if __name__ == "__main__":
    main()
