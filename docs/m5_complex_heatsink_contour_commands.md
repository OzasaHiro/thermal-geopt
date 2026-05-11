# M5 Complex Heat-Sink D1 and Contour Commands

M5 keeps the M4 solver-backed solid-conduction setup, but adds a more visually
interesting `staggered_pin_fin` family.  This is still `blockMesh` plus
`laplacianFoam`, so it is robust and fast compared with STL/snappyHexMesh CHT.

## 1. Generate Complex Heat-Sink Data

```bash
cd /home/hiroaki-ozasa/Desktop/Development/PINN_CFD/EXPERIMENT/Thermal_GeoPT
source /opt/openfoam13/etc/bashrc

OVERWRITE=1 bash scripts/run_m5_openfoam_complex_heatsink_d1.sh
```

Default output:

- `data/downstream_npz/d1_openfoam_complex_heatsink_m5_300/manifest.json`
- `configs/d1_openfoam_complex_heatsink_m5_300_split_seed42.json`
- `outputs/logs/d1_openfoam_complex_heatsink_m5_300_baseline_test.json`

Short smoke:

```bash
CASE_COUNT=12 CELLS_X=16 CELLS_Y=16 BASE_CELLS_Z=3 FEATURE_CELLS_Z=8 OVERWRITE=1 \
  bash scripts/run_m5_openfoam_complex_heatsink_d1.sh
```

## 2. Draw Ground-Truth Temperature Contours

```bash
../../.venv/bin/python scripts/visualize_d1_contours.py \
  --manifest data/downstream_npz/d1_openfoam_complex_heatsink_m5_300/manifest.json \
  --family staggered_pin_fin \
  --max-cases 3 \
  --output-dir outputs/figures/m5_complex_heatsink_contours
```

The script writes one PNG per selected case with three slices:

- `x-z` side slice near mid `y`
- `y-z` side slice near mid `x`
- `x-y` plan slice near the upper feature region

For report-friendly 3D figures, render OpenFOAM VTK surfaces and an internal
temperature cut plane:

```bash
../../.venv/bin/python scripts/render_d1_surface_temperature.py \
  --manifest data/downstream_npz/d1_openfoam_complex_heatsink_m5_300/manifest.json \
  --family staggered_pin_fin \
  --max-cases 3 \
  --output-dir outputs/figures/m5_complex_heatsink_surface_temperature
```

This figure is usually better for communication than the 2D contour slices:
the exterior surface shows the geometry, while the internal cut plane shows the
hot-base to cooled-exterior conduction path.

The default M5 transfer data uses a fixed-cold exterior.  That is useful as a
strict label-efficiency benchmark, but it can make the fins look too cold in a
surface figure.  For communication figures, generate a separate weak-cooling
case:

```bash
../../.venv/bin/python scripts/generate_d1_openfoam_heatsink_cases.py \
  --case-count 1 \
  --families staggered_pin_fin \
  --cells-x 20 \
  --cells-y 20 \
  --base-cells-z 4 \
  --feature-cells-z 12 \
  --source-temperature-min 520 \
  --source-temperature-max 560 \
  --sink-temperature-min 295 \
  --sink-temperature-max 305 \
  --sink-value-fraction 0.025 \
  --raw-dir data/downstream_raw/d1_openfoam_visual_staggered_pin_weak_cooling \
  --output-dir data/downstream_npz/d1_openfoam_visual_staggered_pin_weak_cooling \
  --overwrite

../../.venv/bin/python scripts/render_d1_surface_temperature.py \
  --manifest data/downstream_npz/d1_openfoam_visual_staggered_pin_weak_cooling/manifest.json \
  --family staggered_pin_fin \
  --max-cases 1 \
  --output-dir outputs/figures/m5_visual_weak_cooling_surface_temperature \
  --camera-zoom 0.9
```

Here `--sink-value-fraction 1.0` is the current fixed-temperature cold exterior,
while smaller values behave like weaker mixed cooling.  Lower values let heat
penetrate farther into the fins and make the heat path easier to see.

## 3. Three-Way Transfer Gate

```bash
TRAIN_SIZES="25 50" SPLIT_SEEDS="42 43 44" EPOCHS=50 MODE=all \
  bash scripts/run_m5_complex_heatsink_transfer_gate.sh
```

This compares:

- `scratch`
- `geopt_transport_lifted`
- `geopt_original`
