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

## Strategy Correction

This pilot should be treated as a pipeline validation only. Continuing to optimize this same 1200-case D1 proxy split is not the right main test of GeoPT-style pretraining.

GeoPT's core claim is not that a pretrained model always wins on an easy, fully labeled downstream proxy. Its core claim is data-efficient transfer from large-scale dynamics-lifted geometric self-supervision. Therefore the next gate should test label efficiency and pretraining scale, not just longer fine-tuning on the same split.

Recommended next gate:

1. Build label-scarcity downstream splits: 10/25/50/100 train cases, fixed validation/test.
2. Compare the same Transolver architecture: scratch vs Thermal GeoPT pretrained, with matched point budget and evaluation.
3. Add at least one pretext ablation before scaling: static TDF-only or Brownian/no-boundary-field.
4. Use current `cadquery_pilot_300_e20_n4096` only as a lower-bound pretraining run. Move to 500-2,000 shapes and 5k-40k episodes before drawing conclusions about the method.
5. Do not move to P3/P4 large pretraining unless Thermal GeoPT beats scratch by at least about 10% relative L2 in the 25 or 50 label regime, or shows clearly faster convergence at matched downstream budget.

The current result is therefore not evidence against Thermal GeoPT. It is evidence that this D1 proxy/full-label pilot is too easy and too small to validate GeoPT's intended advantage.
