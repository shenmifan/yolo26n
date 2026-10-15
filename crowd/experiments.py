"""Predeclared paper variants. The full recipe preserves the completed seed0 model."""
from copy import deepcopy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = {
    "full": dict(description="P2-P5 + current recalibration + density + count + consistency", density=True, gains=(1., .1, .5)),
    "no_recal": dict(description="Full model with the current P2 recalibration replaced by Identity", density=True, gains=(1., .1, .5)),
    "no_p2": dict(description="P3-P5 joint model without P2/recalibration; compare with no_recal to isolate P2", density=True, gains=(1., .1, .5)),
    "det_p2": dict(description="P2-P5 detector, no recalibration, no density head or auxiliary losses", density=False, gains=(0., 0., 0.)),
    "det_yolo26": dict(description="Original P3-P5 YOLO26n detector, no density head", density=False, gains=(0., 0., 0.)),
    "no_count": dict(description="Full architecture with count loss weight zero", density=True, gains=(1., 0., .5)),
    "no_cons": dict(description="Full architecture with consistency loss weight zero", density=True, gains=(1., .1, 0.)),
    "density_mse": dict(description="Full architecture with density MSE only as auxiliary supervision", density=True, gains=(1., 0., 0.)),
}
EXTERNAL = ("mcnn", "csrnet")


def model_config(variant, cfg=None):
    if variant not in VARIANTS:
        raise ValueError(f"Unknown experiment variant: {variant}")
    if variant in {"no_p2", "det_yolo26"}:
        # Copy the installed official architecture: its exact hash is archived by the runner.
        import ultralytics
        source = Path(ultralytics.__file__).parent / "cfg/models/26/yolo26.yaml"
        config = yaml.safe_load(source.read_text(encoding="utf-8"))
        config.update(nc=1, scale="n")
    else:
        config = deepcopy(cfg) if isinstance(cfg, dict) else yaml.safe_load(Path(cfg or ROOT / "configs/dc-yolo26n.yaml").read_text(encoding="utf-8"))
        config["scale"] = "n"
        if variant in {"no_recal", "det_p2"}:
            index = next(i for i, layer in enumerate(config["head"]) if layer[2] == "EMA")
            config["head"][index] = [-1, 1, "nn.Identity", []]
    return config


def crowd_settings(data, variant):
    keys = ("den_gain", "cnt_gain", "cons_gain", "grid", "tau_cons", "dgf_tau", "protocol_id")
    result = {key: data[key] for key in keys if key in data}
    # Preserve full defaults/data overrides for compatibility with existing checkpoints.
    if variant != "full":
        result.update(zip(("den_gain", "cnt_gain", "cons_gain"), VARIANTS[variant]["gains"]))
    result.update(variant=variant, density_enabled=VARIANTS[variant]["density"])
    return result


def run_name(dataset, variant, seed):
    prefix = "shta" if dataset == "sht_a" else "shtb"
    return f"{prefix}_density_v2_seed{seed}" if variant == "full" else f"{prefix}_{variant}_seed{seed}"
