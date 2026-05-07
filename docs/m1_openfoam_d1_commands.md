# M1 OpenFOAM D1 Commands

This is the first solver-backed D1 path. It uses OpenFOAM Foundation v13
`laplacianFoam` and writes NPZ files that keep the existing training keys:
`points`, `conditions`, and `temperature`.

The M1 smoke geometry is intentionally minimal:

- blockMesh rectangular solid
- fixed-temperature `source` patch
- fixed-temperature `sink` patch
- zero-gradient insulated side patches

This validates the solver-backed data path. It is not yet the paper-scale
plate-fin/pin-fin heat-sink dataset.

## 1. One-Case Smoke

```bash
cd /home/hiroaki-ozasa/Desktop/Development/PINN_CFD/EXPERIMENT/Thermal_GeoPT
source /opt/openfoam13/etc/bashrc
../../.venv/bin/python scripts/generate_d1_openfoam_cases.py \
  --case-count 1 \
  --cells 8 8 8 \
  --raw-dir data/downstream_raw/d1_openfoam_block_smoke \
  --output-dir data/downstream_npz/d1_openfoam_block_smoke \
  --overwrite
```

Inspect the generated manifest and NPZ:

```bash
../../.venv/bin/python scripts/inspect_artifacts.py \
  data/downstream_npz/d1_openfoam_block_smoke/manifest.json
```

## 2. Pilot Dataset

After the smoke passes, generate a small solver-backed pilot:

```bash
source /opt/openfoam13/etc/bashrc
../../.venv/bin/python scripts/generate_d1_openfoam_cases.py \
  --case-count 50 \
  --cells 16 16 12 \
  --raw-dir data/downstream_raw/d1_openfoam_block_pilot_50 \
  --output-dir data/downstream_npz/d1_openfoam_block_pilot_50 \
  --overwrite
```

Then build splits:

```bash
../../.venv/bin/python scripts/build_splits.py \
  --input-manifest data/downstream_npz/d1_openfoam_block_pilot_50/manifest.json \
  --output configs/d1_openfoam_block_pilot_50_split_seed42.json \
  --train-frac 0.7 \
  --val-frac 0.15 \
  --seed 42
```

The same pilot preparation can be run as one script:

```bash
OVERWRITE=1 bash scripts/run_m1_openfoam_pilot.sh
```

Useful overrides:

```bash
CASE_COUNT=10 CELLS_X=8 CELLS_Y=8 CELLS_Z=8 OVERWRITE=1 \
  bash scripts/run_m1_openfoam_pilot.sh
```

The runner sources `/opt/openfoam13/etc/bashrc` internally.

## 3. Downstream Smoke

```bash
../../.venv/bin/python scripts/train_finetune_d1.py \
  --case-manifest data/downstream_npz/d1_openfoam_block_pilot_50/manifest.json \
  --split-path configs/d1_openfoam_block_pilot_50_split_seed42.json \
  --train-split train \
  --val-split val \
  --output-dir outputs/checkpoints/d1_openfoam_block_scratch_smoke \
  --epochs 1 \
  --batch-size 1 \
  --point-budget 1024 \
  --eval-point-budget 1024 \
  --max-train-cases 4 \
  --max-val-cases 2 \
  --n-hidden 128 \
  --n-layers 2 \
  --n-heads 4 \
  --slice-num 16 \
  --device auto
```

Evaluate a simple baseline:

```bash
../../.venv/bin/python scripts/evaluate_d1.py \
  --case-manifest data/downstream_npz/d1_openfoam_block_pilot_50/manifest.json \
  --split-path configs/d1_openfoam_block_pilot_50_split_seed42.json \
  --split test \
  --baseline mean_temperature \
  --output-json outputs/logs/d1_openfoam_block_baseline_test.json \
  --point-budget 1024
```

The evaluation JSON reports Relative L2/RMSE, max-temperature error, absolute
max-temperature error, and hotspot-distance error.

## 4. Scratch Fine-Tune Smoke

After the 50-case pilot has been generated and checked:

```bash
bash scripts/run_m1_openfoam_downstream_smoke.sh
```

The default smoke uses 8 train cases, 4 validation cases, 3 epochs, and 1024
sampled points per case.

For a fuller scratch pilot on the complete 35/7 train/val split:

```bash
MAX_TRAIN_CASES=0 MAX_VAL_CASES=0 EPOCHS=20 DEVICE=cuda \
  OUTPUT_DIR=outputs/checkpoints/d1_openfoam_block_scratch_pilot_ep20 \
  EVAL_JSON=outputs/logs/d1_openfoam_block_scratch_pilot_ep20_test.json \
  bash scripts/run_m1_openfoam_downstream_smoke.sh
```

## Next M1 Expansion

The next implementation step should replace the block-only geometry with
plate-fin/pin-fin cases via STL/snappyHexMesh while preserving the same NPZ
schema. Robin cooling or heat-flux source patches should be added after the
basic solver-backed path is stable.
