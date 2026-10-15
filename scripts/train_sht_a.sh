#!/usr/bin/env bash
# ShanghaiTech-A: train, select tau on validation, evaluate frozen test parameters.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python}"
TRAIN_SEED="${TRAIN_SEED:-0}"
RUN_NAME="shta_density_v2_seed${TRAIN_SEED}"

"$PYTHON" train.py --data datasets/sht_a_density_v2/data.yaml --pretrained weights/yolo26n.pt \
  --epochs 300 --batch 4 --imgsz 640 --device 0 --workers 0 --seed "$TRAIN_SEED" \
  --project "$PWD/runs/crowd" --name "$RUN_NAME"
"$PYTHON" val.py --weights "runs/crowd/$RUN_NAME/weights/best.pt" \
  --data datasets/sht_a_density_v2/data.yaml --split val --sweep-tau \
  --batch 4 --workers 0 --save-dir "runs/eval/${RUN_NAME}_val"
"$PYTHON" val.py --weights "runs/crowd/$RUN_NAME/weights/best.pt" \
  --data datasets/sht_a_density_v2/data.yaml --split test \
  --params "runs/eval/${RUN_NAME}_val/parameters.json" \
  --batch 4 --workers 0 --save-dir "runs/eval/${RUN_NAME}_test"
