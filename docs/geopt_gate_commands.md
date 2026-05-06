# GeoPT Validation Gate Commands

This gate replaces "keep tuning the full-label pilot" as the next decision point. It tests whether Thermal GeoPT behaves like GeoPT should: better data efficiency under scarce downstream labels.

All commands assume:

```bash
cd /home/hiroaki-ozasa/Desktop/Development/PINN_CFD/EXPERIMENT/Thermal_GeoPT
PY=../../.venv/bin/python
PRETRAIN_MANIFEST=data/pretrain_zarr/cadquery_pilot_300_e20_n4096/manifest.json
D1_MANIFEST=data/downstream_npz/d1_proxy_pilot_300_c5_n8192/manifest.json
BASE_SPLIT=configs/d1_proxy_pilot_300_c5_n8192_split.json
GATE_SPLIT=configs/d1_proxy_pilot_300_c5_n8192_label_scarcity_split.json
```

## 1. Build Label-Scarcity Splits

Already generated locally, but this is the reproducible command:

```bash
$PY scripts/build_label_scarcity_splits.py \
  --base-split $BASE_SPLIT \
  --output $GATE_SPLIT \
  --train-sizes 10 25 50 100 \
  --seed 42
```

The resulting split has fixed `val` and `test`, plus `train_10`, `train_25`, `train_50`, and `train_100`.

Validate split nesting, manifest coverage, and pretraining ablation schemas:

```bash
$PY scripts/validate_gate_setup.py \
  --case-manifest $D1_MANIFEST \
  --split-path $GATE_SPLIT \
  --pretrain-manifest $PRETRAIN_MANIFEST \
  --train-sizes 10 25 50 100
```

## 2. Pretraining Ablations

The previous full pretraining run can be reused as the lower-bound full pretraining checkpoint:

```bash
PRETRAIN_FULL=outputs/checkpoints/pretrain_pilot_300_e20_n4096_ep2
```

Run two ablation pretraining variants from the same Zarr shards:

- `static_tdf_only`: predicts only VDF, distance, and boundary normal channels; all condition prompts are zeroed.
- `no_boundary_field`: keeps the full 14-channel target but zeros the `q_near` boundary-field prompt while preserving input dimensionality.

```bash
PRETRAIN_STATIC=outputs/checkpoints/pretrain_gate_static_tdf_only_ep2

$PY scripts/train_pretrain.py \
  --manifest $PRETRAIN_MANIFEST \
  --output-dir $PRETRAIN_STATIC \
  --pretext-ablation static_tdf_only \
  --epochs 2 \
  --batch-size 1 \
  --point-budget 4096 \
  --max-episodes 0 \
  --amp \
  --amp-dtype bfloat16 \
  --device cuda \
  --seed 42
```

```bash
PRETRAIN_NO_BOUNDARY=outputs/checkpoints/pretrain_gate_no_boundary_field_ep2

$PY scripts/train_pretrain.py \
  --manifest $PRETRAIN_MANIFEST \
  --output-dir $PRETRAIN_NO_BOUNDARY \
  --pretext-ablation no_boundary_field \
  --epochs 2 \
  --batch-size 1 \
  --point-budget 4096 \
  --max-episodes 0 \
  --amp \
  --amp-dtype bfloat16 \
  --device cuda \
  --seed 42
```

Optional full rerun with the same naming convention:

```bash
PRETRAIN_FULL=outputs/checkpoints/pretrain_gate_full_ep2

$PY scripts/train_pretrain.py \
  --manifest $PRETRAIN_MANIFEST \
  --output-dir $PRETRAIN_FULL \
  --pretext-ablation full \
  --epochs 2 \
  --batch-size 1 \
  --point-budget 4096 \
  --max-episodes 0 \
  --amp \
  --amp-dtype bfloat16 \
  --device cuda \
  --seed 42
```

## 3. Downstream Matrix

Run the four groups for each train size:

- `scratch`
- `full`
- `static_tdf_only`
- `no_boundary_field`

Use this shell loop. It is intentionally explicit and writes outputs in the convention expected by `scripts/summarize_label_scarcity_results.py`.

```bash
set -euo pipefail

PRETRAIN_FULL=outputs/checkpoints/pretrain_pilot_300_e20_n4096_ep2
PRETRAIN_STATIC=outputs/checkpoints/pretrain_gate_static_tdf_only_ep2
PRETRAIN_NO_BOUNDARY=outputs/checkpoints/pretrain_gate_no_boundary_field_ep2

for N in 10 25 50 100; do
  $PY scripts/train_finetune_d1.py \
    --case-manifest $D1_MANIFEST \
    --split-path $GATE_SPLIT \
    --train-split train_${N} \
    --val-split val \
    --output-dir outputs/checkpoints/d1_gate_scratch_train${N}_ep20 \
    --epochs 20 \
    --batch-size 1 \
    --point-budget 4096 \
    --eval-point-budget 4096 \
    --max-train-cases 0 \
    --max-val-cases 0 \
    --amp \
    --amp-dtype bfloat16 \
    --device cuda \
    --seed 42

  $PY scripts/train_finetune_d1.py \
    --case-manifest $D1_MANIFEST \
    --split-path $GATE_SPLIT \
    --train-split train_${N} \
    --val-split val \
    --output-dir outputs/checkpoints/d1_gate_full_train${N}_ep20 \
    --pretrained-model-dir $PRETRAIN_FULL \
    --pretrained-checkpoint-file best_model.pt \
    --epochs 20 \
    --batch-size 1 \
    --point-budget 4096 \
    --eval-point-budget 4096 \
    --max-train-cases 0 \
    --max-val-cases 0 \
    --amp \
    --amp-dtype bfloat16 \
    --device cuda \
    --seed 42

  $PY scripts/train_finetune_d1.py \
    --case-manifest $D1_MANIFEST \
    --split-path $GATE_SPLIT \
    --train-split train_${N} \
    --val-split val \
    --output-dir outputs/checkpoints/d1_gate_static_tdf_only_train${N}_ep20 \
    --pretrained-model-dir $PRETRAIN_STATIC \
    --pretrained-checkpoint-file best_model.pt \
    --epochs 20 \
    --batch-size 1 \
    --point-budget 4096 \
    --eval-point-budget 4096 \
    --max-train-cases 0 \
    --max-val-cases 0 \
    --amp \
    --amp-dtype bfloat16 \
    --device cuda \
    --seed 42

  $PY scripts/train_finetune_d1.py \
    --case-manifest $D1_MANIFEST \
    --split-path $GATE_SPLIT \
    --train-split train_${N} \
    --val-split val \
    --output-dir outputs/checkpoints/d1_gate_no_boundary_field_train${N}_ep20 \
    --pretrained-model-dir $PRETRAIN_NO_BOUNDARY \
    --pretrained-checkpoint-file best_model.pt \
    --epochs 20 \
    --batch-size 1 \
    --point-budget 4096 \
    --eval-point-budget 4096 \
    --max-train-cases 0 \
    --max-val-cases 0 \
    --amp \
    --amp-dtype bfloat16 \
    --device cuda \
    --seed 42
done
```

## 4. Test Evaluation

```bash
set -euo pipefail

for N in 10 25 50 100; do
  for GROUP in scratch full static_tdf_only no_boundary_field; do
    $PY scripts/evaluate_d1.py \
      --case-manifest $D1_MANIFEST \
      --split-path $GATE_SPLIT \
      --split test \
      --model-dir outputs/checkpoints/d1_gate_${GROUP}_train${N}_ep20 \
      --checkpoint-file best_model.pt \
      --point-budget 4096 \
      --device cuda \
      --output-json outputs/logs/d1_gate_${GROUP}_train${N}_test_eval.json
  done
done
```

## 5. Summarize

```bash
$PY scripts/summarize_label_scarcity_results.py \
  --train-sizes 10 25 50 100 \
  --groups scratch full static_tdf_only no_boundary_field \
  --expected-case-count 150 \
  --expected-point-budget 4096
```

This writes:

- `outputs/logs/label_scarcity_gate_summary.json`
- `docs/label_scarcity_gate_results.md`

The summary command fails if any expected evaluation JSON is missing or has the wrong case count/point budget. For a partial dry-run report only, add `--allow-missing`.

## Decision Rule

Proceed toward larger P2/P3 pretraining only if Thermal GeoPT improves scratch by about 10% relative L2 at 25 or 50 labels, or reaches the same error with materially fewer downstream epochs.
