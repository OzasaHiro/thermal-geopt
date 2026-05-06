# Pilot Training Commands

These commands start after the artifact checks in `docs/heavy_run_commands.md` pass. They are for pilot validation, not final benchmark runs.

Prerequisites:

```bash
cd /home/hiroaki-ozasa/Desktop/Development/PINN_CFD/EXPERIMENT/Thermal_GeoPT
PY=../../.venv/bin/python
PRETRAIN_MANIFEST=data/pretrain_zarr/cadquery_pilot_300_e20_n4096/manifest.json
D1_MANIFEST=data/downstream_npz/d1_proxy_pilot_300_c5_n8192/manifest.json
D1_SPLIT=configs/d1_proxy_pilot_300_c5_n8192_split.json
```

## 1. Pretrain smoke

Use this first to confirm the training loop, checkpoint writing, and bf16 AMP path.

```bash
PRETRAIN_SMOKE=outputs/checkpoints/pretrain_pilot_smoke_1ep_2k

$PY scripts/train_pretrain.py \
  --manifest $PRETRAIN_MANIFEST \
  --output-dir $PRETRAIN_SMOKE \
  --epochs 1 \
  --batch-size 1 \
  --point-budget 2048 \
  --max-episodes 64 \
  --amp \
  --amp-dtype bfloat16 \
  --device cuda \
  --seed 42
```

## 2. Pretrain pilot

This is the first useful pilot pretraining candidate. Increase `--epochs` after the smoke is stable.

```bash
PRETRAIN_RUN=outputs/checkpoints/pretrain_pilot_300_e20_n4096_ep2

$PY scripts/train_pretrain.py \
  --manifest $PRETRAIN_MANIFEST \
  --output-dir $PRETRAIN_RUN \
  --epochs 2 \
  --batch-size 1 \
  --point-budget 4096 \
  --max-episodes 0 \
  --amp \
  --amp-dtype bfloat16 \
  --device cuda \
  --seed 42
```

Outputs:

- `$PRETRAIN_RUN/config.json`
- `$PRETRAIN_RUN/model.pt`
- `$PRETRAIN_RUN/best_model.pt`
- `$PRETRAIN_RUN/history.json`
- `$PRETRAIN_RUN/metrics.json`

## 3. D1 scratch pilot

Run scratch first. This gives the non-pretrained reference for the same split.

```bash
D1_SCRATCH_RUN=outputs/checkpoints/d1_scratch_pilot_300_c5_n8192_ep3

$PY scripts/train_finetune_d1.py \
  --case-manifest $D1_MANIFEST \
  --split-path $D1_SPLIT \
  --output-dir $D1_SCRATCH_RUN \
  --epochs 3 \
  --batch-size 1 \
  --point-budget 4096 \
  --eval-point-budget 4096 \
  --max-train-cases 0 \
  --max-val-cases 0 \
  --amp \
  --amp-dtype bfloat16 \
  --device cuda \
  --seed 42
```

## 4. D1 pretrained pilot

Then run the same D1 split initialized from the pretraining checkpoint.

```bash
D1_PRETRAINED_RUN=outputs/checkpoints/d1_pretrained_pilot_300_c5_n8192_ep3

$PY scripts/train_finetune_d1.py \
  --case-manifest $D1_MANIFEST \
  --split-path $D1_SPLIT \
  --output-dir $D1_PRETRAINED_RUN \
  --pretrained-model-dir $PRETRAIN_RUN \
  --pretrained-checkpoint-file best_model.pt \
  --epochs 3 \
  --batch-size 1 \
  --point-budget 4096 \
  --eval-point-budget 4096 \
  --max-train-cases 0 \
  --max-val-cases 0 \
  --amp \
  --amp-dtype bfloat16 \
  --device cuda \
  --seed 42
```

Expected load behavior: input/output tensors whose shapes differ between TDF pretraining and D1 temperature prediction are skipped, while shape-compatible Transolver blocks are loaded.

## 5. D1 evaluation

Evaluate the held-out `test` split for scratch and pretrained runs.

```bash
$PY scripts/evaluate_d1.py \
  --case-manifest $D1_MANIFEST \
  --split-path $D1_SPLIT \
  --split test \
  --model-dir $D1_SCRATCH_RUN \
  --checkpoint-file best_model.pt \
  --point-budget 4096 \
  --device cuda \
  --output-json outputs/logs/d1_scratch_pilot_test_eval.json

$PY scripts/evaluate_d1.py \
  --case-manifest $D1_MANIFEST \
  --split-path $D1_SPLIT \
  --split test \
  --model-dir $D1_PRETRAINED_RUN \
  --checkpoint-file best_model.pt \
  --point-budget 4096 \
  --device cuda \
  --output-json outputs/logs/d1_pretrained_pilot_test_eval.json
```

Baseline comparison:

```bash
$PY scripts/evaluate_d1.py \
  --case-manifest $D1_MANIFEST \
  --split-path $D1_SPLIT \
  --split test \
  --baseline mean_temperature \
  --point-budget 4096 \
  --output-json outputs/logs/d1_baseline_pilot_test_eval.json
```
