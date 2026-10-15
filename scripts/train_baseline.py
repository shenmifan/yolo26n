"""Train matched-split MCNN/CSRNet and evaluate the frozen best checkpoint.

The common canvas/targets/augmentation differ from the original full-image/crop
recipes. Report these as matched architecture reproductions, not published scores.
"""
import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from torch.utils.data import DataLoader
from crowd.baselines import baseline_input, make_baseline
from crowd.dataset import CrowdDataset
from crowd.density import resize_density_preserve_count_torch
from crowd.paper_utils import environment, get_device, load_data, make_dataset, seed_worker, set_seed, write_json
from crowd.protocol import sha256_file
from val import mae_rmse, write_csv


def predict_rows(model, loader, name, device):
    model.eval()
    rows = []
    with torch.inference_mode():
        for batch in loader:
            density = model(baseline_input(batch["img"].to(device), name))
            counts = density.flatten(1).sum(1).double().cpu().tolist()
            for index, prediction in enumerate(counts):
                gt = int(batch["gt_count"][index])
                rows.append(dict(image=batch["im_file"][index], gt=gt, density=prediction,
                                 error_density=prediction - gt, abs_error_density=abs(prediction - gt)))
    metrics = mae_rmse([row["gt"] for row in rows], [row["density"] for row in rows])
    return rows, metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("mcnn", "csrnet"), required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--save-dir", type=Path, required=True)
    parser.add_argument("--eval-root", type=Path, default=ROOT / "runs/eval")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="0")
    parser.add_argument("--vgg-weights", type=Path, help="Optional local official VGG16 ImageNet state dict; otherwise download with hash check")
    args = parser.parse_args()
    if args.imgsz % 32 or min(args.epochs, args.batch) <= 0 or args.patience < 0:
        parser.error("Use a positive multiple-of-32 canvas, positive epochs/batch and nonnegative patience")
    data, device = load_data(args.data), get_device(args.device)
    # Reject a reused output before any download or training.
    args.save_dir.mkdir(parents=True, exist_ok=False)
    for split in ("val", "test"):
        if (args.eval_root / f"{args.save_dir.name}_{split}").exists():
            raise FileExistsError("Baseline evaluation output already exists")
    set_seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    loaders = {}
    for split in ("train", "val"):
        ds = make_dataset(data, split, args.imgsz, augment=split == "train")
        loaders[split] = DataLoader(ds, batch_size=args.batch, shuffle=split == "train",
                                   num_workers=args.workers, collate_fn=CrowdDataset.collate_fn,
                                   worker_init_fn=seed_worker, generator=generator,
                                   pin_memory=device.type == "cuda")
    model = make_baseline(args.method)
    if args.method == "csrnet":
        model.load_vgg_frontend(args.vgg_weights)
    model.to(device)
    lr = 1e-7 if args.method == "csrnet" else 1e-4
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=.95, weight_decay=5e-4) if args.method == "csrnet" else torch.optim.Adam(model.parameters(), lr=lr)
    settings = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    settings.update(optimizer=type(optimizer).__name__, lr0=lr, precision="FP32",
                    loss="sum squared density error per image, averaged over batch" if args.method == "csrnet" else "mean squared density error",
                    initialization="VGG16 ImageNet frontend; Gaussian backend" if args.method == "csrnet" else "Gaussian std=0.01",
                    protocol_id=data["protocol_id"], recipe="matched_canvas_density_v2_v1",
                    augmentation="CrowdDataset: synchronized horizontal flip + HSV; fixed letterbox")
    write_json(args.save_dir / "baseline_config.json", settings)
    write_json(args.save_dir / "environment.json", environment(device))
    (args.save_dir / "weights").mkdir()
    best, best_epoch, history, started = math.inf, 0, [], time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, images = 0., 0
        for batch in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(baseline_input(batch["img"].to(device), args.method))
            target = batch["density"].unsqueeze(1).float().to(device)
            target = resize_density_preserve_count_torch(target, prediction.shape[-2:])
            difference = (prediction - target).square()
            loss = difference.flatten(1).sum(1).mean() if args.method == "csrnet" else difference.mean()
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite baseline loss; do not silently alter the registered recipe")
            loss.backward()
            optimizer.step()
            total += loss.item() * len(prediction)
            images += len(prediction)
        _, metrics = predict_rows(model, loaders["val"], args.method, device)
        history.append(dict(epoch=epoch, time=time.perf_counter() - started, train_loss=total / images,
                            val_MAE=metrics["MAE"], val_RMSE=metrics["RMSE"]))
        checkpoint = dict(state_dict=model.state_dict(), method=args.method, epoch=epoch,
                          training_seed=args.seed, protocol_id=data["protocol_id"], config=settings)
        torch.save(dict(checkpoint, optimizer=optimizer.state_dict()), args.save_dir / "weights/last.pt")
        if metrics["MAE"] < best:
            best, best_epoch = metrics["MAE"], epoch
            torch.save(checkpoint, args.save_dir / "weights/best.pt")
        write_csv(args.save_dir / "results.csv", history)
        print(f"{args.method} seed{args.seed} epoch {epoch}/{args.epochs}: val MAE={metrics['MAE']:.3f}, best={best:.3f} @ {best_epoch}", flush=True)
        if args.patience and epoch - best_epoch >= args.patience:
            break
    checkpoint = torch.load(args.save_dir / "weights/best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    weight_hash = sha256_file(args.save_dir / "weights/best.pt")
    test_dataset = make_dataset(data, "test", args.imgsz)
    loaders["test"] = DataLoader(test_dataset, batch_size=args.batch, shuffle=False,
                                 num_workers=args.workers, collate_fn=CrowdDataset.collate_fn,
                                 worker_init_fn=seed_worker, pin_memory=device.type == "cuda")
    parameters = dict(settings=dict(imgsz=args.imgsz, normalization="imagenet" if args.method == "csrnet" else "unit_rgb", output="signed_density_integral"),
                      weights_sha256=weight_hash, protocol_id=data["protocol_id"], source_split="val", diagnostic=False)
    for split in ("val", "test"):
        folder = args.eval_root / f"{args.save_dir.name}_{split}"
        folder.mkdir(parents=True, exist_ok=False)
        rows, metrics = predict_rows(model, loaders[split], args.method, device)
        summary = dict(parameters, source_split=split, metrics=dict(density=metrics), images=len(rows),
                       gt_source="original_points", data=str(Path(args.data).resolve()),
                       weights=str((args.save_dir / "weights/best.pt").resolve()), variant=args.method,
                       training_seed=args.seed, torch=torch.__version__, ultralytics=__import__("ultralytics").__version__,
                       device=str(device), device_name=torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU")
        summary["evaluation_batch"] = args.batch
        write_csv(folder / "per_image.csv", rows)
        write_json(folder / "summary.json", summary)
        write_json(folder / "parameters.json", dict(parameters, source_split=split))
    write_json(args.save_dir / "completed.json", dict(best_epoch=best_epoch, epochs=len(history), weights_sha256=weight_hash))


if __name__ == "__main__":
    main()
