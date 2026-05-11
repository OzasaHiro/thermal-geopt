# Thermal GeoPT

Thermal GeoPT is an experiment project for diffusion-lifted geometric pre-training for heat transfer surrogate models.

## Preliminary Finding

The current strongest result is a cross-physics transfer finding:

> The original GeoPT pretrained checkpoint improves label efficiency on
> OpenFOAM-backed heat-sink solid-conduction surrogate tasks.

Across two solver-backed D1 heat-sink benchmarks, the original GeoPT backbone
outperformed scratch training at 25 and 50 downstream labels.  The current
Thermal-specific R4 pretraining caused negative transfer.

See:

- `docs/original_geopt_heat_transfer_preliminary_report.md`
- `docs/original_geopt_heat_transfer_preliminary_report_ja.md`
- `docs/m4_heatsink_transfer_results.md`
- `docs/m5_complex_heatsink_transfer_results.md`
- `docs/public_release_checklist.md`

The working research direction is based on:

- `thermal_geopt_research_plan_draft.md`
- `thermal_geopt_detailed_research_plan.md`

The initial Thermal GeoPT direction was to avoid a single scalar
`TDF = distance * conductivity` definition and explore multi-channel thermal
diffusion features.  The current strongest result, however, is the original
GeoPT cross-physics transfer finding summarized above.

## Visual Example

![Weak-cooling staggered-pin heat-sink surface temperature](docs/assets/figures/m5_weak_cooling_surface_temperature.png)

Annotated view:

- Left: 3D surface temperature on a staggered-pin heat-sink geometry.
- Center: internal cut plane showing heat propagation from the heated base into the fins.
- Right: top layout of the same heat-sink case, included to make the geometry arrangement readable.
- This weak-cooling case is an explanatory visualization only.  It is separate from the M4/M5 benchmark data used for the quantitative transfer results.

## Original GeoPT Reference

This project is motivated by the lifted geometric pre-training idea proposed in:

Wu, Haixu; Guo, Minghao; Li, Zongyi; Dou, Zhiyang; Long, Mingsheng; He, Kaiming; Matusik, Wojciech.
**GeoPT: Scaling Physics Simulation via Lifted Geometric Pre-Training.** arXiv:2602.20399v1, 2026.
https://arxiv.org/abs/2602.20399

```bibtex
@article{wu2026geopt,
  title = {GeoPT: Scaling Physics Simulation via Lifted Geometric Pre-Training},
  author = {Wu, Haixu and Guo, Minghao and Li, Zongyi and Dou, Zhiyang and Long, Mingsheng and He, Kaiming and Matusik, Wojciech},
  journal = {arXiv preprint arXiv:2602.20399},
  year = {2026}
}
```

## Initial Scope

1. Build a small, reproducible D1 solid-conduction benchmark before CHT.
2. Reuse the GeoPT PoC assets from `../GeoPT` where they are proven useful.
3. Keep Git history focused on code, configs, lightweight reports, and runbooks.
4. Keep raw data, VTP/VTU/STL meshes, NumPy arrays, Zarr shards, checkpoints, and bulk generated figures out of normal Git.

## Repository Layout

```text
configs/        Small YAML configs for smoke and baseline runs.
docs/           Setup notes and asset inventory.
scripts/        Executable experiment entrypoints.
thermal_geopt/  Importable project package.
data/           Local data cache, excluded from Git except placeholders.
outputs/        Local run outputs, excluded from Git except placeholders.
```

## Current Setup Status

- `git`: installed.
- `git-lfs`: installed.
- GitHub remote: `https://github.com/OzasaHiro/thermal-geopt.git`
- Shared Python environment: `../../.venv`.
- Verified package baseline: Python 3.12.3, PyTorch 2.10.0+cu128, CUDA 12.8, PyVista 0.47.1.

See `RUNBOOK.md` and `docs/github_setup.md` for the current handoff.
