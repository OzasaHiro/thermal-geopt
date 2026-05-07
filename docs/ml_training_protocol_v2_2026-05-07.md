# ML Training Protocol V2 2026-05-07

## Purpose

This protocol fixes the basic transfer-learning failures found in the M3 P2 review.

The goal is not to rescue the old P2 checkpoint. The goal is to make the next Thermal GeoPT run a valid test of GeoPT-style pretraining:

- shared coordinate and prompt normalization between pretraining and downstream;
- pretraining validation split and checkpoint selection by validation loss;
- normalized multi-target pretraining loss;
- downstream scratch and pretrained runs using the same input normalization contract;
- heat-transfer metrics beyond absolute-Kelvin relative L2.

## Implemented Changes

### Pretraining

`scripts/train_pretrain.py` now supports:

- `--val-fraction`
- `--normalization standardize`
- `--normalization-max-episodes`
- `--target-min-std`
- `--best-metric auto|train_loss|val_loss`

With `--val-fraction > 0`, `best_model.pt` is selected by `val_loss` unless overridden.

The saved `config.json` now contains a `normalization` contract with:

- coordinate mean/std;
- feature/condition mean/std;
- target mean/std;
- backward-compatible aliases: `coordinate_mean`, `coordinate_std`, `x_mean`, `x_std`, `feature_mean`, `feature_std`, `target_mean`, `target_std`.

### Fine-Tuning

`scripts/train_finetune_d1.py` now supports:

- `--normalization-protocol legacy_downstream`
- `--normalization-protocol downstream`
- `--normalization-protocol pretrained`
- `--normalization-protocol none`
- `--normalization-config`

The legacy default preserves old behavior. The valid Thermal GeoPT transfer protocol should use:

```bash
--normalization-protocol pretrained \
--normalization-config outputs/checkpoints/<pretrain-run>/config.json
```

This reuses pretraining coordinate and feature statistics for both scratch and pretrained downstream runs, while still using downstream train labels for target-temperature normalization.

### Evaluation

`scripts/evaluate_d1.py` now reports:

- `centered_relative_l2`
- `normalized_rmse_range`
- `max_temperature_abs_error`
- `hotspot_abs_error`

Existing JSON keys remain available.

## Heavy Run Commands

### 1. R2 P2 Pretraining With Validation And Normalization

```bash
PY=../../.venv/bin/python

$PY scripts/train_pretrain.py \
  --manifest data/pretrain_zarr/cadquery_p2_d1_thermal_2000_e20_n8192/manifest.json \
  --output-dir outputs/checkpoints/pretrain_r2_d1_thermal_dynamics_p2_norm_val_ep20 \
  --epochs 20 \
  --batch-size 1 \
  --point-budget 8192 \
  --max-episodes 0 \
  --val-fraction 0.05 \
  --normalization standardize \
  --normalization-max-episodes 2048 \
  --target-min-std 0.05 \
  --pretext-ablation dynamics_lifted \
  --lr 1e-3 \
  --weight-decay 1e-5 \
  --amp \
  --amp-dtype bfloat16 \
  --device cuda
```

Do not use the old `pretrain_r1_d1_thermal_dynamics_p2_ep20` checkpoint for new transfer claims. It has no validation selection and no normalization contract.

### 2. Aligned M3 Transfer Gate

Run this after the R2 checkpoint exists:

```bash
PY=../../.venv/bin/python \
PRETRAIN_DYNAMICS=outputs/checkpoints/pretrain_r2_d1_thermal_dynamics_p2_norm_val_ep20 \
NORMALIZATION_PROTOCOL=pretrained \
NORMALIZATION_CONFIG=outputs/checkpoints/pretrain_r2_d1_thermal_dynamics_p2_norm_val_ep20/config.json \
RUN_PREFIX=m3_openfoam_p2_r2_normval_oclr \
EPOCHS=100 \
TRAIN_SIZES="10 25 50 100" \
SPLIT_SEEDS="42 43 44" \
TRAIN_SEEDS="42" \
POINT_BUDGET=3072 \
EVAL_POINT_BUDGET=3072 \
GATE_GROUPS="scratch dynamics_lifted" \
FINETUNE_SCHEDULER=onecycle \
PRETRAINED_BACKBONE_LR=3e-4 \
PRETRAINED_HEAD_LR=1e-3 \
FREEZE_PRETRAINED_BACKBONE_EPOCHS=5 \
MAX_GRAD_NORM=1.0 \
MODE=all \
bash scripts/run_m3_openfoam_p2_transfer_gate.sh
```

This is still a block-D1 gate, not the paper-level heat-sink benchmark.

## Interpretation Rule

Treat the current old P2 negative result as invalid for method rejection.

For the R2 aligned protocol:

- Go: dynamics-lifted improves scratch by about 10% at 25 or 50 labels and does not worsen Tmax/hotspot metrics.
- Conditional: improvement appears only at 100 labels or only in centered/nondimensional metrics.
- No-Go for current recipe: no improvement under aligned normalization and validation-selected pretraining.

Even with a No-Go result, the conclusion is about the current R2 target design. It is not a rejection of the broader Thermal GeoPT idea until the downstream task is a nontrivial solver-backed heat-sink D1 benchmark.

## Smoke Tests Already Run

CPU-only smoke tests completed:

- pretraining with `--val-fraction 0.25 --normalization standardize`;
- fine-tuning with `--normalization-protocol pretrained`;
- evaluation with new thermal metrics and coordinate normalization.

