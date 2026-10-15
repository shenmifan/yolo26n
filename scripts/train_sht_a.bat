@echo off
REM ShanghaiTech-A: fixed 239/60/182 protocol. Run preparation first.
setlocal
cd /d "%~dp0.."
set KMP_DUPLICATE_LIB_OK=TRUE
set YOLO_CONFIG_DIR=%CD%\Ultralytics
set MPLCONFIGDIR=%CD%\.runtime\matplotlib
if not defined PYTHON set "PYTHON=python"
if not defined TRAIN_SEED set "TRAIN_SEED=0"
set "RUN_NAME=shta_density_v2_seed%TRAIN_SEED%"

"%PYTHON%" train.py --data datasets/sht_a_density_v2/data.yaml --pretrained weights/yolo26n.pt --epochs 300 --batch 4 --imgsz 640 --device 0 --workers 0 --seed %TRAIN_SEED% --project "%CD%\runs\crowd" --name %RUN_NAME%
if errorlevel 1 exit /b 1
"%PYTHON%" val.py --weights runs/crowd/%RUN_NAME%/weights/best.pt --data datasets/sht_a_density_v2/data.yaml --split val --sweep-tau --batch 4 --workers 0 --save-dir runs/eval/%RUN_NAME%_val
if errorlevel 1 exit /b 1
"%PYTHON%" val.py --weights runs/crowd/%RUN_NAME%/weights/best.pt --data datasets/sht_a_density_v2/data.yaml --split test --params runs/eval/%RUN_NAME%_val/parameters.json --batch 4 --workers 0 --save-dir runs/eval/%RUN_NAME%_test
if errorlevel 1 exit /b 1
endlocal
