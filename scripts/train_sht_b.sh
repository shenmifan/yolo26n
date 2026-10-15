#!/usr/bin/env bash
# ShanghaiTech Part B（中低密度，固定高斯核，640）
set -e
python -m crowd.prepare_data --dataset sht_b --src data/ShanghaiTech/part_B --dst datasets/sht_b
python train.py --data datasets/sht_b/data.yaml --epochs 300 --batch 16 --imgsz 640 --device 0 \
  --name dc_yolo26n_shtB --project runs/crowd
python val.py --weights runs/crowd/dc_yolo26n_shtB/weights/best.pt --data datasets/sht_b/data.yaml \
  --imgsz 640 --sweep-tau
