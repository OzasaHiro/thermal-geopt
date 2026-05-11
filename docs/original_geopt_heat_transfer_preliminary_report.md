# Preliminary Report: Original GeoPT Transfers to Solver-Backed Heat-Sink Conduction

Date: 2026-05-11
Repository: `OzasaHiro/thermal-geopt`
Status: preliminary technical report, not a peer-reviewed paper

## Summary

This experiment started as an attempt to adapt the GeoPT idea to heat-transfer
surrogate modeling through Thermal GeoPT pretraining.  The most interesting
finding so far is different from the initial plan:

> The original GeoPT pretrained checkpoint, trained for fluid-oriented geometric
> pretraining, improves label efficiency on solver-backed heat-sink solid
> conduction surrogate tasks.

Across two OpenFOAM-backed heat-sink D1 benchmarks, the original GeoPT backbone
consistently outperformed scratch training under 25-label and 50-label regimes.
The current Thermal-specific R4 pretraining did not help; it caused negative
transfer.

This supports a narrower and useful hypothesis:

> GeoPT may learn a transferable geometry-boundary-dynamics prior that is not
> limited to fluid prediction.

## What Was Tested

### Downstream Task

The downstream task is steady solid conduction, solved with OpenFOAM Foundation
v13 `laplacianFoam`.

For each case:

- a 3D heat-sink-like solid geometry is generated;
- the base is the heat source or hot boundary;
- the exterior is the cooling boundary;
- OpenFOAM solves the scalar temperature field `T`;
- the result is converted to NPZ with point coordinates, condition features,
  and cell-centered temperature labels;
- a Transolver backbone predicts temperature from point-wise inputs.

This is solver-backed data.  It is not the earlier synthetic D1 proxy.

### Benchmarks

Two geometry-varied D1 benchmarks were used.

| Benchmark | Cases | Families | Test cases | Purpose |
|---|---:|---|---:|---|
| M4 heat-sink D1 | 300 | `plate_fin`, `pin_fin` | 45 | first solver-backed heat-sink gate |
| M5 complex heat-sink D1 | 300 | `plate_fin`, `pin_fin`, `staggered_pin_fin` | 45 | more complex geometry gate and visualization target |

Both use label-scarcity splits at 25 and 50 training cases, with split seeds
`42`, `43`, and `44`.

### Compared Groups

| Group | Meaning |
|---|---|
| `scratch` | same downstream architecture, random initialization |
| `geopt_original` | original GeoPT pretrained checkpoint, `../GeoPT/checkpoints/GeoPT_8layers.pt` |
| `geopt_transport_lifted` | current Thermal GeoPT R4 transport-lifted pretraining |

The original GeoPT checkpoint loads the shape-compatible backbone tensors into
the D1 Transolver.  Input/output tensors with incompatible shapes are skipped.

## Main Results

### M4 Heat-Sink D1

Test Relative L2:

| Train labels | Scratch | Thermal R4 | Original GeoPT |
|---:|---:|---:|---:|
| 25 | 0.013122 +/- 0.000685 | 0.037595 +/- 0.001770 | 0.011370 +/- 0.001009 |
| 50 | 0.010674 +/- 0.000470 | 0.022637 +/- 0.000570 | 0.008840 +/- 0.000345 |

Paired improvement vs scratch:

| Train labels | Thermal R4 | Original GeoPT |
|---:|---:|---:|
| 25 | -187.1% | +13.4% |
| 50 | -112.4% | +17.2% |

Temperature extrema and hotspot metrics:

| Train labels | Group | maxT abs error [K] | hotspot abs error [K] |
|---:|---|---:|---:|
| 25 | scratch | 2.64 | 2.74 |
| 25 | original GeoPT | 1.98 | 2.00 |
| 50 | scratch | 1.14 | 1.24 |
| 50 | original GeoPT | 0.99 | 1.00 |

### M5 Complex Heat-Sink D1

Test Relative L2:

| Train labels | Scratch | Thermal R4 | Original GeoPT |
|---:|---:|---:|---:|
| 25 | 0.012548 +/- 0.000259 | 0.032063 +/- 0.001846 | 0.011186 +/- 0.000006 |
| 50 | 0.010613 +/- 0.000060 | 0.022536 +/- 0.002766 | 0.008950 +/- 0.000194 |

Paired improvement vs scratch:

| Train labels | Thermal R4 | Original GeoPT |
|---:|---:|---:|
| 25 | -155.5% | +10.8% |
| 50 | -112.3% | +15.7% |

Temperature extrema and hotspot metrics:

| Train labels | Group | maxT abs error [K] | hotspot abs error [K] |
|---:|---|---:|---:|
| 25 | scratch | 2.88 | 2.93 |
| 25 | original GeoPT | 1.44 | 1.44 |
| 50 | scratch | 1.12 | 1.14 |
| 50 | original GeoPT | 0.93 | 0.91 |

## Interpretation

The M4 and M5 results point in the same direction:

- Original GeoPT improves solver-backed heat-sink conduction prediction under
  label scarcity.
- The improvement appears in Relative L2 and also in max-temperature and
  hotspot metrics.
- The current Thermal-specific R4 pretraining is not a useful initialization
  for this downstream task.

This is important because it suggests that GeoPT's transferable component is
not simply a fluid-specific prior.  A plausible interpretation is that the
original checkpoint encodes useful geometry-boundary-dynamics structure:

- spatial organization of 3D point sets;
- boundary-aware representation;
- field propagation under geometry constraints;
- reusable inductive bias for PDE-like surrogate tasks.

The negative Thermal R4 result is also useful.  It shows that making a
heat-themed pretraining target is not enough.  A thermal-specific extension
must preserve the useful GeoPT prior instead of overwriting it with a poorly
aligned synthetic task.

### Original GeoPT vs Thermal R4 Pretraining

The comparison should not be read as a rejection of the Thermal GeoPT concept.
The two pretrained initializations are very different in scale and construction.

Original GeoPT was trained at much larger scale: the paper reports more than
one million pretraining samples from diverse off-the-shelf geometries with
dynamics-lifted geometric self-supervision.  Its target is not thermal
conduction itself; it is a generic geometry-boundary-dynamics prior.

The current Thermal R4 pretraining is a project-local, much smaller
thermal-motivated prototype with narrower geometry families and a more specific
synthetic target.  Its negative transfer therefore says that this particular R4
design and scale are not yet aligned with the downstream task.  It does not
show that thermal-specific GeoPT pretraining is impossible or intrinsically
unhelpful.

The more important signal is the positive transfer from original GeoPT.  The
checkpoint used here was not trained with thermal labels, yet it improves
solver-backed heat conduction.  Combined with the original GeoPT paper's report
that the same lifted geometric pretraining benefits both fluid and solid
mechanics benchmarks, this result strengthens the broader hypothesis that a
single dynamics-lifted geometric pretraining can provide a reusable
initialization across multiple simulation families.

For the next Thermal GeoPT iteration, the goal should be to preserve this
general GeoPT prior while adding thermal relevance.  Plausible directions are
continued pretraining from original GeoPT, lightweight thermal adapters, or a
larger and more diverse thermal-lifted pretraining dataset, always evaluated
against original GeoPT as a strong control.

## Visual Example

For communication, the best current figure is a 3D heat-sink surface plus an
internal temperature cut plane.  The weak-cooling case is separate from the
M4/M5 benchmark data and is intended only as an explanatory visualization.

![Weak-cooling staggered-pin heat-sink surface temperature](assets/figures/m5_weak_cooling_surface_temperature.png)

Generate the visualization case:

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
```

Render the figure:

```bash
../../.venv/bin/python scripts/render_d1_surface_temperature.py \
  --manifest data/downstream_npz/d1_openfoam_visual_staggered_pin_weak_cooling/manifest.json \
  --family staggered_pin_fin \
  --max-cases 1 \
  --output-dir outputs/figures/m5_visual_weak_cooling_surface_temperature \
  --camera-zoom 0.9
```

## Reproduction Commands

### Generate M4 Data

```bash
OVERWRITE=1 CASE_COUNT=300 bash scripts/run_m4_openfoam_heatsink_d1.sh
```

### Run M4 Transfer Gate

```bash
TRAIN_SIZES="25 50" SPLIT_SEEDS="42 43 44" EPOCHS=50 MODE=all \
  bash scripts/run_m4_heatsink_transfer_gate.sh
```

### Generate M5 Data

```bash
OVERWRITE=1 bash scripts/run_m5_openfoam_complex_heatsink_d1.sh
```

### Run M5 Transfer Gate

```bash
TRAIN_SIZES="25 50" SPLIT_SEEDS="42 43 44" EPOCHS=50 MODE=all \
  bash scripts/run_m5_complex_heatsink_transfer_gate.sh
```

## Artifacts

The main output files are:

- `docs/m4_heatsink_transfer_results.md`
- `docs/results/m4_heatsink_transfer_summary.json`
- `docs/m5_complex_heatsink_transfer_results.md`
- `docs/results/m5_complex_heatsink_transfer_summary.json`
- `docs/assets/figures/m5_weak_cooling_surface_temperature.png`

Large artifacts such as OpenFOAM case directories, NPZ datasets, checkpoints,
full logs, and bulk generated figures are ignored by default in Git.  The
repository keeps only selected lightweight result summaries and one
communication figure.

## Limitations

This is not a final paper-level claim.

- Only 3 split seeds were used.
- The downstream task is solid conduction, not conjugate heat transfer.
- The pin-fin geometry is blockMesh-friendly and rectangularized, not a fully
  resolved industrial heat sink.
- The original GeoPT checkpoint is used as an external pretrained control and
  is not redistributed by this report.
- The original GeoPT and Thermal R4 pretraining runs differ substantially in
  data scale, geometry diversity, and pretraining target, so the R4 negative
  result is not a controlled ablation of the Thermal GeoPT concept.
- The Thermal GeoPT R4 negative result applies to the current R4 pretraining
  design, not to every possible thermal-specific GeoPT extension.

## Recommended Public Claim

Use a cautious statement:

> We provide preliminary evidence that the original GeoPT pretrained backbone
> transfers to OpenFOAM-backed heat-sink solid-conduction surrogate modeling
> under label scarcity.  Across two D1 heat-sink benchmarks, original GeoPT
> improves scratch training by about 10-17% Relative L2 and improves hotspot
> metrics.  A first thermal-specific pretraining attempt caused negative
> transfer, suggesting that thermal extensions should preserve the original
> GeoPT geometry-boundary prior.

This is suitable for a GitHub technical report or short arXiv-style note after
the repository is tagged and archived.
