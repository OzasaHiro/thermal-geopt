# M1 OpenFOAM Block Pilot 50 Check

Checked after user-side execution of `scripts/run_m1_openfoam_pilot.sh`.

## Data Integrity

- Manifest: `data/downstream_npz/d1_openfoam_block_pilot_50/manifest.json`
- Case count: 50
- Split: `configs/d1_openfoam_block_pilot_50_split_seed42.json`
- Split counts: train 35 / val 7 / test 8
- NPZ count: 50
- OpenFOAM logs: 50 `blockMesh`, 50 `laplacianFoam`, 50 `writeCellCentres`
- Solver failures: 0
- Final residual range: `2.28853502e-11` to `9.77323418e-11`

Common NPZ shapes:

| Field | Shape |
|---|---:|
| `points` | `[3072, 3]` |
| `conditions` | `[3072, 6]` |
| `temperature` / `T` | `[3072, 1]` |
| `normals` | `[3072, 3]` |
| `region` | `[3072]` |
| `material` | `[3072, 2]` |
| `bc_features` | `[3072, 5]` |

No missing required keys and no non-finite numeric values were found.

## Baseline

Baseline: mean of source/sink temperatures, evaluated on test split with
`point_budget=1024`.

| Metric | Value |
|---|---:|
| Relative L2 | 0.0885817193 |
| RMSE | 31.03620754 |
| Max temperature error | -49.42564392 |
| Max temperature absolute error | 49.42564392 |
| Hotspot distance | 0.0 |

The zero hotspot distance here is not a strong result; this block-only setup has
a monotone source-to-sink field, and the constant baseline can share the first
argmax index under the current downsampling order. Use it only as a plumbing
metric for this minimal block pilot.

## Next Step

Run scratch fine-tuning smoke on the solver-backed pilot:

```bash
bash scripts/run_m1_openfoam_downstream_smoke.sh
```

Codex-side runner smoke was also checked with a lighter setting:

```bash
EPOCHS=1 MAX_TRAIN_CASES=4 MAX_VAL_CASES=2 POINT_BUDGET=512 \
EVAL_POINT_BUDGET=512 N_HIDDEN=64 N_LAYERS=1 N_HEADS=4 \
SLICE_NUM=8 DEVICE=cpu \
OUTPUT_DIR=outputs/checkpoints/d1_openfoam_block_scratch_runner_smoke_codex \
EVAL_JSON=outputs/logs/d1_openfoam_block_scratch_runner_smoke_codex_test.json \
bash scripts/run_m1_openfoam_downstream_smoke.sh
```

It completed training and test evaluation. The test Relative L2 was
`0.0917583980`, which is only a plumbing check because this run used 4 train
cases and 1 epoch.

For a slightly larger scratch pilot:

```bash
MAX_TRAIN_CASES=0 MAX_VAL_CASES=0 EPOCHS=20 DEVICE=cuda \
  OUTPUT_DIR=outputs/checkpoints/d1_openfoam_block_scratch_pilot_ep20 \
  EVAL_JSON=outputs/logs/d1_openfoam_block_scratch_pilot_ep20_test.json \
  bash scripts/run_m1_openfoam_downstream_smoke.sh
```
