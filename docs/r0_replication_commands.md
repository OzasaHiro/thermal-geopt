# R0 Replication Commands

R0 checks whether the `100 labels + no_boundary_field` signal is reproducible. This is still the old TDF-based pretraining line. Do not mix dynamics-lifted R1 checkpoints into this matrix.

All commands assume:

```bash
cd /home/hiroaki-ozasa/Desktop/Development/PINN_CFD/EXPERIMENT/Thermal_GeoPT
PY=../../.venv/bin/python
```

## Scope

Default R0 settings:

- groups: `scratch`, `no_boundary_field`
- train sizes: `50 75 100 125`
- split seeds: `42 43 44 45 46`
- train seeds: `42`
- eval: fixed `test` split, deterministic point selection

This keeps the run focused on the signal that matters. Add `full` and `static_tdf_only` only after `no_boundary_field` reproduces.

## 1. Downstream Matrix

Run:

```bash
bash scripts/run_r0_replication_matrix.sh
```

Resume behavior:

- existing `best_model.pt` runs are skipped by default
- set `SKIP_EXISTING=0` to overwrite/re-run

Short smoke:

```bash
SPLIT_SEEDS="42" TRAIN_SEEDS="42" TRAIN_SIZES="50" GATE_GROUPS="scratch" \
EPOCHS=1 POINT_BUDGET=128 EVAL_POINT_BUDGET=128 MAX_TRAIN_CASES=2 MAX_VAL_CASES=2 \
bash scripts/run_r0_replication_matrix.sh
```

## 2. Test Evaluation And Summary

Run:

```bash
bash scripts/run_r0_test_eval.sh
```

Short smoke matching the matrix smoke:

```bash
SPLIT_SEEDS="42" TRAIN_SEEDS="42" TRAIN_SIZES="50" GATE_GROUPS="scratch" \
EPOCHS=1 POINT_BUDGET=128 \
SUMMARY_JSON=outputs/logs/r0_replication_smoke_summary.json \
SUMMARY_MD=docs/r0_replication_smoke_results.md \
bash scripts/run_r0_test_eval.sh
```

The summary is written to:

- `outputs/logs/r0_replication_summary.json`
- `docs/r0_replication_results.md`

## Interpretation

Treat the 100-label signal as reproduced only if `no_boundary_field` is consistently positive across split seeds and remains visible at 125 labels. If it appears only for seed 42 or collapses around 75/125 labels, keep the old gate negative and prioritize R1 dynamics-lifted pretraining.
