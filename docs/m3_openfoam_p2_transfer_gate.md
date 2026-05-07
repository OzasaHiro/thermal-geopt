# M3 OpenFOAM P2 Transfer Gate

This is the next downstream experiment after P2 dynamics-lifted pretraining.

It tests whether the P2 checkpoint transfers to solver-backed D1 solid conduction. It is still a block-geometry gate, not the final heat-sink paper experiment. A negative result here should trigger pretext/prompt review before P3 scale-up, not an immediate rejection of Thermal GeoPT.

## Inputs

Required P2 checkpoint:

```bash
outputs/checkpoints/pretrain_r1_d1_thermal_dynamics_p2_ep20/best_model.pt
```

Required solver-backed D1 labels:

```bash
data/downstream_npz/d1_openfoam_block_m3_300/manifest.json
configs/d1_openfoam_block_m3_300_split_seed42.json
```

## 1. Generate 300 OpenFOAM D1 Cases

Run this if the M3 300-case dataset does not exist yet:

```bash
cd /home/hiroaki-ozasa/Desktop/Development/PINN_CFD/EXPERIMENT/Thermal_GeoPT
OVERWRITE=1 bash scripts/run_m3_openfoam_d1_300.sh
```

Default output:

- `data/downstream_raw/d1_openfoam_block_m3_300`
- `data/downstream_npz/d1_openfoam_block_m3_300`
- `configs/d1_openfoam_block_m3_300_split_seed42.json`
- `outputs/logs/d1_openfoam_block_m3_300_baseline_test.json`

The default split is 70/15/15, so for 300 cases it should produce roughly:

- train: 210
- val: 45
- test: 45

## 2. Smoke The Transfer Runner

Use this first to catch checkpoint-loading or split mistakes:

```bash
MODE=all \
TRAIN_SIZES="10" \
SPLIT_SEEDS="42" \
EPOCHS=2 \
POINT_BUDGET=512 \
EVAL_POINT_BUDGET=512 \
EXPECTED_CASE_COUNT=45 \
RUN_PREFIX=m3_openfoam_p2_smoke \
SUMMARY_JSON=outputs/logs/m3_openfoam_p2_smoke_summary.json \
SUMMARY_MD=docs/m3_openfoam_p2_smoke_results.md \
bash scripts/run_m3_openfoam_p2_transfer_gate.sh
```

This smoke is only a path check.

## 3. Run The Initial Gate

First practical gate:

```bash
MODE=all \
TRAIN_SIZES="10 25 50 100" \
SPLIT_SEEDS="42 43 44" \
EPOCHS=50 \
POINT_BUDGET=3072 \
EVAL_POINT_BUDGET=3072 \
EXPECTED_CASE_COUNT=45 \
bash scripts/run_m3_openfoam_p2_transfer_gate.sh
```

If time allows and the first gate is not clearly negative, add `200` labels:

```bash
MODE=all \
TRAIN_SIZES="10 25 50 100 200" \
SPLIT_SEEDS="42 43 44" \
EPOCHS=50 \
POINT_BUDGET=3072 \
EVAL_POINT_BUDGET=3072 \
EXPECTED_CASE_COUNT=45 \
bash scripts/run_m3_openfoam_p2_transfer_gate.sh
```

## Interpretation

Use `best_model.pt` from the P2 pretraining run. The final `model.pt` overtrained after epoch 11 and should not be used for transfer.

The default runner keeps the historical equal-optimizer protocol: scratch and pretrained fine-tuning both use `lr=1e-3` unless overridden. This is useful as an aggressive stress test, but it is not the fairest pretraining-efficacy protocol because it can overwrite useful pretrained backbone weights. See `docs/m3_finetuning_protocol_review_2026-05-07.md` for the full protocol rationale.

For the fairer fine-tuning protocol, rerun with a lower LR on tensors loaded from pretraining and a higher LR on the newly initialized output head:

```bash
MODE=all \
RUN_PREFIX=m3_openfoam_p2_ft_tuned_oclr \
TRAIN_SIZES="10 25 50 100" \
SPLIT_SEEDS="42 43 44" \
EPOCHS=100 \
POINT_BUDGET=3072 \
EVAL_POINT_BUDGET=3072 \
EXPECTED_CASE_COUNT=45 \
SCRATCH_LR=1e-3 \
PRETRAINED_BACKBONE_LR=3e-4 \
PRETRAINED_HEAD_LR=1e-3 \
FREEZE_PRETRAINED_BACKBONE_EPOCHS=5 \
FINETUNE_SCHEDULER=onecycle \
FINETUNE_PCT_START=0.3 \
MAX_GRAD_NORM=1.0 \
bash scripts/run_m3_openfoam_p2_transfer_gate.sh
```

Positive signal:

- `dynamics_lifted` improves scratch by about 10% relative L2 at 25 or 50 labels, or
- reaches similar error with materially fewer labels, and
- max temperature error does not degrade.

Weak or negative signal:

- Do not proceed to P3.
- Review the pretext, especially whether individual Brownian displacement prediction is unnecessarily stochastic.
- Consider a simplified R1b pretraining target using heat-aware deterministic fields and Monte Carlo expectation targets.
