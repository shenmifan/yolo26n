#!/usr/bin/env bash
# UCF-QNRF（极高密度、分辨率大，建议 768，batch 视显存调整）
set -e
python -m crowd.prepare_data --dataset qnrf --src data/UCF-QNRF --dst datasets/qnrf
python train.py --data datasets/qnrf/data.yaml --epochs 250 --batch 8 --imgsz 768 --device 0 \
  --name dc_yolo26n_qnrf --project runs/crowd
python val.py --weights runs/crowd/dc_yolo26n_qnrf/weights/best.pt --data datasets/qnrf/data.yaml \
  --imgsz 768 --sweep-tau
