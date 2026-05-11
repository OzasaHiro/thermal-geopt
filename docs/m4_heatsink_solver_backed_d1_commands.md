# M4 Heat-Sink Solver-Backed D1 Commands

M4 replaces the simple block-only OpenFOAM D1 gate with geometry-varied
solid-conduction heat-sink cases.  It still uses `laplacianFoam`, but the
meshes are multi-block plate-fin and rectangularized pin-fin solids generated
from the same design family as the pretraining CAD shapes.

This is the next main downstream gate.  The earlier block D1 remains useful as
a pipeline check, not as the final transfer benchmark.

## 1. Generate Heat-Sink D1 Data

```bash
cd /home/hiroaki-ozasa/Desktop/Development/PINN_CFD/EXPERIMENT/Thermal_GeoPT
source /opt/openfoam13/etc/bashrc

OVERWRITE=1 bash scripts/run_m4_openfoam_heatsink_d1.sh
```

Default output:

- `data/downstream_npz/d1_openfoam_heatsink_m4_300/manifest.json`
- `configs/d1_openfoam_heatsink_m4_300_split_seed42.json`
- `outputs/logs/d1_openfoam_heatsink_m4_300_baseline_test.json`

Useful pilot override:

```bash
CASE_COUNT=20 CELLS_X=16 CELLS_Y=16 BASE_CELLS_Z=3 FEATURE_CELLS_Z=8 OVERWRITE=1 \
  bash scripts/run_m4_openfoam_heatsink_d1.sh
```

This writes to `data/downstream_npz/d1_openfoam_heatsink_m4_20/` by default.
For this 20-case pilot, use smaller label-scarcity sizes:

```bash
D1_MANIFEST=data/downstream_npz/d1_openfoam_heatsink_m4_20/manifest.json \
BASE_SPLIT=configs/d1_openfoam_heatsink_m4_20_split_seed42.json \
GATE_SPLIT_PATTERN='outputs/logs/m4_splits/d1_openfoam_heatsink_m4_20_label_scarcity_seed{split_seed}.json' \
RUN_PREFIX=m4_heatsink_r4_vs_original_pilot20 \
SUMMARY_JSON=outputs/logs/m4_heatsink_pilot20_transfer_summary.json \
SUMMARY_MD=docs/m4_heatsink_pilot20_transfer_results.md \
EXPECTED_CASE_COUNT=3 \
TRAIN_SIZES="5 10" SPLIT_SEEDS="42" EPOCHS=50 MODE=all \
  bash scripts/run_m4_heatsink_transfer_gate.sh
```

## 2. Run Three-Way Transfer Gate

This compares:

- `scratch`
- `geopt_transport_lifted`: current Thermal GeoPT R4 checkpoint
- `geopt_original`: original GeoPT pretrained checkpoint

```bash
MODE=all bash scripts/run_m4_heatsink_transfer_gate.sh
```

Default output:

- `outputs/logs/m4_heatsink_transfer_summary.json`
- `docs/m4_heatsink_transfer_results.md`

Recommended quick gate before the full matrix:

```bash
TRAIN_SIZES="25 50" SPLIT_SEEDS="42" EPOCHS=50 MODE=all \
  bash scripts/run_m4_heatsink_transfer_gate.sh
```

If the quick gate is negative for both pretrained groups, inspect learning
curves and normalization before scaling.  If original GeoPT helps but Thermal
GeoPT does not, the Thermal pretraining target is the likely issue.  If neither
helps and scratch is already near-saturated, the downstream heat-sink D1 task
still needs more boundary/geometry diversity before drawing a paper-level
conclusion.
