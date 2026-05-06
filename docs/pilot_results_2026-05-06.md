# Pilot Results 2026-05-06

## Scope

- Pretraining: `cadquery_pilot_300_e20_n4096`, 2 epochs.
- Downstream: `d1_proxy_pilot_300_c5_n8192`, train/val/test = 1200/150/150 cases.
- Evaluation: test split, 4096 points/case, 150 cases.
- Data and checkpoints are generated artifacts and remain outside Git.

## Result Summary

| run | test_relative_l2 | test_rmse | test_max_value_error |
|---|---:|---:|---:|
| baseline_mean_temperature | 0.064578 | 22.417916 | -46.844730 |
| d1_scratch_ep3 | 0.048966 | 16.948986 | -4.245930 |
| d1_pretrained_ep3 | 0.059174 | 20.532413 | -19.260945 |

Relative L2 deltas:

- Scratch vs baseline: 24.18% improvement.
- Pretrained vs baseline: 8.37% improvement.
- Pretrained vs scratch: -20.85% improvement. Negative means pretrained is worse than scratch.

## Pretraining

Final pretraining train MSE: `0.040319`.

| epoch | train_mse | val_relative_l2 | val_rmse | elapsed_sec |
|---:|---:|---:|---:|---:|
| 1 | 0.056550 |  |  | 202.9 |
| 2 | 0.040319 |  |  | 396.4 |

## D1 Scratch

| epoch | train_mse | val_relative_l2 | val_rmse | elapsed_sec |
|---:|---:|---:|---:|---:|
| 1 | 1.046127 | 0.073128 | 25.451961 | 20.9 |
| 2 | 1.002529 | 0.073373 | 25.499487 | 41.7 |
| 3 | 0.847958 | 0.048856 | 16.995173 | 62.3 |

## D1 Pretrained

| epoch | train_mse | val_relative_l2 | val_rmse | elapsed_sec |
|---:|---:|---:|---:|---:|
| 1 | 1.013564 | 0.073137 | 25.455589 | 21.7 |
| 2 | 0.997408 | 0.073080 | 25.395707 | 43.0 |
| 3 | 0.868186 | 0.059351 | 20.690435 | 64.6 |

Pretrained load report:

- loaded tensors: 166
- skipped tensors: 3
- missing tensors: 3
- checkpoint: `outputs/checkpoints/pretrain_pilot_300_e20_n4096_ep2/best_model.pt`

## Interpretation

The pilot confirms the training and evaluation pipeline works, and the scratch Transolver improves over the mean-temperature baseline. The current TDF pretraining does not help this D1 proxy after 3 fine-tuning epochs; it underperforms scratch on both validation and test relative L2.

Most likely reasons:

- The D1 proxy target is simple and directly encoded by source/sink condition features, so scratch learns it quickly.
- The pretraining target has 14 TDF/diffusion channels, while D1 predicts 1 temperature channel; shape-incompatible input/output projection tensors are intentionally skipped.
- Only 2 pretraining epochs and 3 fine-tuning epochs were run, so this is a pipeline pilot, not a final pretraining efficacy result.

## Next Runs

1. Extend D1 fine-tuning to 10-20 epochs for scratch and pretrained with the same split.
2. Add a label-scarcity matrix, for example 50/100/300 train cases, because pretraining should matter most when labels are limited.
3. Move from D1 proxy to a harder D1/D2 target before judging Thermal GeoPT value.
4. For full pretraining candidates, increase geometry diversity and pretraining epochs only after the label-scarcity comparison is in place.
