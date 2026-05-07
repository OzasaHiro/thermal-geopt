# M1 OpenFOAM Initialization Path Check

This check verifies that scratch fine-tuning and existing pilot-pretrained
initialization paths both run on the solver-backed D1 OpenFOAM block pilot.

This is not evidence for or against Thermal GeoPT pretraining effectiveness.
The checkpoints used here are small proxy-era pilot checkpoints, not a
solver-backed, dynamics-lifted, paper-level pretraining run.

## Smoke Setting

```bash
INIT_GROUPS="scratch full static_tdf_only no_boundary_field" \
RUN_PREFIX=d1_openfoam_block_initload_smoke \
EPOCHS=2 MAX_TRAIN_CASES=4 MAX_VAL_CASES=2 \
POINT_BUDGET=256 EVAL_POINT_BUDGET=256 DEVICE=cuda \
bash scripts/run_m1_openfoam_init_compare.sh
```

The runner used the architecture aligned to the existing pilot pretraining
checkpoints:

- `N_HIDDEN=256`
- `N_LAYERS=8`
- `N_HEADS=8`
- `SLICE_NUM=32`

## Path Check Result

| Group | Checkpoint load | Rel L2 | RMSE | MaxT Abs Err | Hotspot Dist |
|---|---|---:|---:|---:|---:|
| scratch | no preload | 0.09001876 | 31.604773 | 55.079323 | 0.077254 |
| full | loaded 166 / skipped 3 / missing 3 | 0.08739066 | 30.660360 | 52.992001 | 0.095387 |
| static_tdf_only | loaded 166 / skipped 3 / missing 3 | 0.09329954 | 32.725460 | 57.058613 | 0.000000 |
| no_boundary_field | loaded 166 / skipped 3 / missing 3 | 0.08007166 | 28.078515 | 45.427124 | 0.043212 |

All four paths completed training and test evaluation.

The metric values above should be read only as smoke-run diagnostics because
the run used 4 train cases, 2 validation cases, 2 epochs, and 256 sampled points.
They are not a pretraining efficacy result.

## Interpretation

The operational question is now answered:

- Scratch path runs on the solver-backed D1 OpenFOAM pilot.
- Pilot-pretrained initialization paths load into the architecture-aligned
  downstream model.
- Existing proxy-era checkpoint transfer is technically functional.

The next meaningful GeoPT experiment still requires a solver-backed,
dynamics-lifted pretraining run with architecture and prompt design aligned to
the D1 downstream task.
