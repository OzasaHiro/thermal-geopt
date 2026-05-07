# ML Training Setup Audit 2026-05-07

## Verdict

The current P2-to-M3 result is not a valid efficacy test of Thermal GeoPT pretraining.
The downstream training loop can optimize scratch models, but the pretraining and fine-tuning setup violates basic transfer-learning assumptions.

## Findings

- **CRITICAL**: Pretraining has no validation/held-out selection. best_model.pt is selected by training loss, so representation quality and overfitting are not monitored.
- **CRITICAL**: Pretraining input/target normalization is not recorded. Downstream fine-tuning normalizes condition channels from downstream train statistics, while pretraining used raw condition channels.
- **CRITICAL**: Coordinate scales between pretraining and downstream are incompatible. The loaded positional/geometry representation is reused across domains with substantially different coordinate ranges.
- **CRITICAL**: Condition-channel distributions do not match. Even when condition names match, train-set downstream standardization and source/sink feature semantics change the input distribution.
- **MAJOR**: Relative L2 on absolute Kelvin temperature is too forgiving. The denominator is dominated by the 300K offset. Report centered or nondimensional temperature errors as primary metrics.
- **MAJOR**: Pretraining does not improve monotonically and degrades after best epoch. Final train loss 0.3105 is worse than best train loss 0.2387; no validation signal exists to interpret this.
- **MAJOR**: Current pretrained initialization is a harmful prior on M3. Representative best validation relL2: scratch=0.000708, pretrained=0.03751.

## Data Interface

- Pretrain condition names: `['conductivity', 'source_temperature', 'sink_temperature', 'source_patch', 'sink_patch', 'nearest_boundary_distance']`
- Downstream condition names: `['conductivity', 'source_temperature', 'sink_temperature', 'source_patch', 'sink_patch', 'nearest_boundary_distance']`
- Pretrain coordinate mean/std: `[-0.00535, 0.00241, -0.006375]` / `[0.3108, 0.2415, 0.1659]`
- Downstream coordinate mean/std: `[0.04009, 0.04136, 0.009084]` / `[0.02639, 0.02684, 0.006363]`
- Pretrain condition mean/std: `[1.594, 398.7, 292.8, 0.01565, 0.01427, 0.04613]` / `[2.563, 20.1, 8.365, 0.07935, 0.07307, 0.04156]`
- Downstream condition mean/std: `[5.301, 397.4, 295.7, 0.1645, 0.1645, 0.003819]` / `[6.319, 20.39, 9.309, 0.23, 0.23, 0.002749]`

## Pretraining Run

- Epochs logged: `20`
- Has validation metrics: `False`
- Has normalization metadata: `False`
- First train loss: `0.2719453957410529`
- Best train loss: `0.23874868723042308` at epoch `11`
- Final train loss: `0.3105070266019553`

## Representative Downstream Runs

| run | best val relL2 | best val RMSE | final train MSE |
|---|---:|---:|---:|
| scratch | 0.000707967 | 0.242174 | 4.34746849896328e-05 |
| pretrained | 0.0375093 | 12.9263 | 0.14190828692167998 |

## Required Standard Before More Heavy Runs

1. Record and reuse one explicit pretraining input/target normalization contract.
2. Apply the same coordinate convention to pretraining and downstream.
3. Add pretraining validation shards and select checkpoints by validation loss or transfer-proxy validation.
4. Standardize heterogeneous pretraining targets per channel or per target group before weighting losses.
5. Make nondimensional or centered thermal errors primary; keep absolute-Kelvin relative L2 only as a secondary metric.
6. Treat current block D1 as pipeline smoke only, not as the main GeoPT transfer benchmark.
