# 论文补充实验进度与结果

标准差为训练种子间样本标准差（ddof=1）。只有一个种子时显示待补；各种子的 tau 独立由验证集选择。

| 数据集 | 变体 | 输出 | 种子数 | MAE 均值 ± SD | RMSE 均值 ± SD | 状态 |
| --- | --- | --- | ---: | --- | --- | --- |
| sht_a | full | detect | 3 | 291.040 ± 19.941 | 376.740 ± 27.181 | complete |
| sht_a | full | density | 3 | 100.126 ± 6.481 | 169.662 ± 4.285 | complete |
| sht_a | full | fused | 3 | 96.951 ± 6.362 | 165.844 ± 3.828 | complete |
| sht_a | no_recal | detect | 1 | 240.527（SD 待补） | 301.618（SD 待补） | pending_seeds |
| sht_a | no_recal | density | 1 | 205.364（SD 待补） | 351.597（SD 待补） | pending_seeds |
| sht_a | no_recal | fused | 1 | 149.528（SD 待补） | 213.040（SD 待补） | pending_seeds |
| sht_a | no_p2 | detect | 1 | 185.308（SD 待补） | 258.197（SD 待补） | pending_seeds |
| sht_a | no_p2 | density | 1 | 95.871（SD 待补） | 160.201（SD 待补） | pending_seeds |
| sht_a | no_p2 | fused | 1 | 95.244（SD 待补） | 153.809（SD 待补） | pending_seeds |
| sht_a | det_p2 | detect | 1 | 157.198（SD 待补） | 228.932（SD 待补） | pending_seeds |
| sht_a | det_yolo26 | detect | 1 | 81.841（SD 待补） | 166.528（SD 待补） | pending_seeds |
| sht_a | no_count | detect | 1 | 292.484（SD 待补） | 386.644（SD 待补） | pending_seeds |
| sht_a | no_count | density | 1 | 235.782（SD 待补） | 250.433（SD 待补） | pending_seeds |
| sht_a | no_count | fused | 1 | 93.006（SD 待补） | 143.645（SD 待补） | pending_seeds |
| sht_a | no_cons | detect | 1 | 191.462（SD 待补） | 333.347（SD 待补） | pending_seeds |
| sht_a | no_cons | density | 1 | 83.464（SD 待补） | 129.295（SD 待补） | pending_seeds |
| sht_a | no_cons | fused | 1 | 78.047（SD 待补） | 128.824（SD 待补） | pending_seeds |
| sht_a | density_mse | detect | 1 | 144.225（SD 待补） | 275.762（SD 待补） | pending_seeds |
| sht_a | density_mse | density | 1 | 166.972（SD 待补） | 190.783（SD 待补） | pending_seeds |
| sht_a | density_mse | fused | 1 | 78.168（SD 待补） | 137.228（SD 待补） | pending_seeds |
| sht_a | mcnn | density | 1 | 130.933（SD 待补） | 199.175（SD 待补） | pending_seeds |
| sht_a | csrnet | density | 1 | 84.785（SD 待补） | 132.362（SD 待补） | pending_seeds |
| sht_b | full | detect | 3 | 29.398 ± 3.662 | 49.242 ± 6.611 | complete |
| sht_b | full | density | 3 | 19.456 ± 4.822 | 30.721 ± 10.597 | complete |
| sht_b | full | fused | 3 | 14.471 ± 2.735 | 24.117 ± 4.034 | complete |
| sht_b | no_recal | detect | 1 | 30.367（SD 待补） | 50.533（SD 待补） | pending_seeds |
| sht_b | no_recal | density | 1 | 16.474（SD 待补） | 23.884（SD 待补） | pending_seeds |
| sht_b | no_recal | fused | 1 | 13.117（SD 待补） | 19.309（SD 待补） | pending_seeds |
| sht_b | no_p2 | detect | 1 | 20.326（SD 待补） | 35.964（SD 待补） | pending_seeds |
| sht_b | no_p2 | density | 1 | 12.833（SD 待补） | 19.756（SD 待补） | pending_seeds |
| sht_b | no_p2 | fused | 1 | 11.700（SD 待补） | 18.181（SD 待补） | pending_seeds |
| sht_b | det_p2 | detect | 1 | 18.101（SD 待补） | 32.710（SD 待补） | pending_seeds |
| sht_b | det_yolo26 | detect | 1 | 15.690（SD 待补） | 31.937（SD 待补） | pending_seeds |
| sht_b | no_count | detect | 1 | 27.456（SD 待补） | 44.725（SD 待补） | pending_seeds |
| sht_b | no_count | density | 1 | 242.279（SD 待补） | 243.438（SD 待补） | pending_seeds |
| sht_b | no_count | fused | 1 | 11.438（SD 待补） | 18.958（SD 待补） | pending_seeds |
| sht_b | no_cons | detect | 1 | 24.028（SD 待补） | 46.100（SD 待补） | pending_seeds |
| sht_b | no_cons | density | 1 | 12.236（SD 待补） | 19.595（SD 待补） | pending_seeds |
| sht_b | no_cons | fused | 1 | 11.239（SD 待补） | 18.857（SD 待补） | pending_seeds |
| sht_b | density_mse | detect | 1 | 20.402（SD 待补） | 39.224（SD 待补） | pending_seeds |
| sht_b | density_mse | density | 1 | 182.892（SD 待补） | 184.309（SD 待补） | pending_seeds |
| sht_b | density_mse | fused | 1 | 11.280（SD 待补） | 19.700（SD 待补） | pending_seeds |
| sht_b | mcnn | density | 1 | 20.230（SD 待补） | 29.838（SD 待补） | pending_seeds |
| sht_b | csrnet | density | 1 | 13.590（SD 待补） | 24.490（SD 待补） | pending_seeds |

已通过核验的训练运行：24；尚缺运行：36；核验异常：0。

缺失清单和逐项输入哈希见 audit.json。MCNN/CSRNet 是统一画布与 density_v2 目标下的架构复现；原论文结果和更强基线需另行标注。

## 速度测量

仅接收 runs/benchmark 下通过逐次计时核验的正式输出；本地 smoke 结果不进入此表。FP32、batch=1，预热≥30 次，计时≥200 次；磁盘读取与解码不计入。

| 数据集 | 变体 | GPU/CPU | 参数量 | 网络中位数 ms | 全流程中位数 ms | 全流程 P95 ms | 串行图/秒 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| sht_a | full | NVIDIA GeForce RTX 3080 Ti | 2537945 | 19.404 | 81.377 | 498.349 | 7.59 |
| sht_a | no_recal | NVIDIA GeForce RTX 3080 Ti | 2537769 | 18.970 | 95.279 | 471.432 | 7.35 |
| sht_a | no_p2 | NVIDIA GeForce RTX 3080 Ti | 2558751 | 15.241 | 124.926 | 368.220 | 6.86 |
| sht_a | det_p2 | NVIDIA GeForce RTX 3080 Ti | 2515976 | 18.378 | 209.827 | 619.058 | 4.33 |
| sht_a | det_yolo26 | NVIDIA GeForce RTX 3080 Ti | 2533886 | 14.297 | 163.860 | 410.692 | 4.82 |
| sht_a | no_count | NVIDIA GeForce RTX 3080 Ti | 2537945 | 19.193 | 68.214 | 362.804 | 9.42 |
| sht_a | no_cons | NVIDIA GeForce RTX 3080 Ti | 2537945 | 19.415 | 107.942 | 388.933 | 6.85 |
| sht_a | density_mse | NVIDIA GeForce RTX 3080 Ti | 2537945 | 19.142 | 145.524 | 516.626 | 5.42 |
| sht_a | mcnn | NVIDIA GeForce RTX 3080 Ti | 133705 | 3.276 | 5.185 | 5.331 | 194.54 |
| sht_a | csrnet | NVIDIA GeForce RTX 3080 Ti | 16263489 | 13.590 | 15.515 | 15.676 | 64.56 |
| sht_b | full | NVIDIA GeForce RTX 3080 Ti | 2537945 | 19.140 | 55.696 | 98.760 | 17.28 |
| sht_b | no_recal | NVIDIA GeForce RTX 3080 Ti | 2537769 | 23.591 | 73.065 | 133.577 | 13.13 |
| sht_b | no_p2 | NVIDIA GeForce RTX 3080 Ti | 2558751 | 18.891 | 71.185 | 140.352 | 13.12 |
| sht_b | det_p2 | NVIDIA GeForce RTX 3080 Ti | 2515976 | 23.081 | 77.752 | 174.826 | 11.84 |
| sht_b | det_yolo26 | NVIDIA GeForce RTX 3080 Ti | 2533886 | 14.546 | 56.067 | 101.289 | 16.72 |
| sht_b | no_count | NVIDIA GeForce RTX 3080 Ti | 2537945 | 19.372 | 53.444 | 92.678 | 17.47 |
| sht_b | no_cons | NVIDIA GeForce RTX 3080 Ti | 2537945 | 18.788 | 56.720 | 97.173 | 17.02 |
| sht_b | density_mse | NVIDIA GeForce RTX 3080 Ti | 2537945 | 19.119 | 58.972 | 99.961 | 16.55 |
| sht_b | mcnn | NVIDIA GeForce RTX 3080 Ti | 133705 | 3.265 | 5.368 | 5.402 | 185.78 |
| sht_b | csrnet | NVIDIA GeForce RTX 3080 Ti | 16263489 | 13.621 | 15.695 | 16.009 | 63.58 |

跨硬件或软件环境的速度记录应分开比较；各行 torch/ultralytics 信息保存在 audit.json。
