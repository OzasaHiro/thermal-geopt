# Thermal GeoPT

[![DOI](https://zenodo.org/badge/1231333716.svg)](https://doi.org/10.5281/zenodo.20130626)

Thermal GeoPT is an experimental heat-transfer adaptation of GeoPT's dynamics-lifted geometric pre-training idea for surrogate modeling.

## Preliminary Finding

The current strongest result is a cross-physics transfer finding:

> In this repository's preliminary OpenFOAM-solved heat-sink conduction tests, a
> Transolver initialized from shape-compatible tensors of the original GeoPT
> checkpoint obtains lower test error than scratch training at 25- and 50-case
> low-data settings.

Across two generated OpenFOAM-solved heat-sink solid-conduction benchmarks, the
partially loaded original GeoPT checkpoint improved mean Relative L2 for each
recorded data-split aggregate at 25 and 50 downstream training cases.
Individual test cases were not uniformly improved.  The current project-local
thermal-specific pretraining prototype, internal ID `R4`, caused negative
transfer.

This does not reject thermal-specific GeoPT pretraining as a concept: the
project-local prototype, internal ID `R4`, is much smaller and narrower than
original GeoPT.  The stronger signal is that original GeoPT appears to provide a
reusable initialization for heat-transfer surrogate modeling.  A boundary-aware
geometry-dynamics representation is a plausible interpretation, but it is not
directly identified by the present experiment.

See:

- `docs/original_geopt_heat_transfer_preliminary_report.md`
- `docs/original_geopt_heat_transfer_preliminary_report_ja.md`
- `docs/m4_heatsink_transfer_results.md`
- `docs/m5_complex_heatsink_transfer_results.md`
- `docs/public_release_checklist.md`

The working research direction is based on:

- `thermal_geopt_research_plan_draft.md`
- `thermal_geopt_detailed_research_plan.md`

The initial Thermal GeoPT direction was to avoid a single scalar thermal
diffusion feature, `TDF = distance * conductivity`, and explore multi-channel
thermal diffusion features.  The current strongest result, however, is the
original GeoPT cross-physics transfer finding summarized above.

The internal dataset label `D1` means this project's first OpenFOAM-solved
solid-conduction dataset family: generated heat-sink-like blockMesh geometries
with steady scalar-conduction labels.  It does not mean a one-dimensional
physical problem.

## Visual Example

![Weak-cooling staggered-pin heat-sink surface temperature](docs/assets/figures/m5_weak_cooling_surface_temperature.png)

Annotated view:

- Left: 3D surface temperature on a staggered-pin heat-sink geometry.
- Center: internal cut plane showing heat propagation from the heated base into the fins.
- Right: top layout of the same heat-sink case, included to make the geometry arrangement readable.
- This weak-cooling case is an explanatory visualization only.  It is separate from the quantitative benchmark datasets, internal IDs `M4` and `M5`.

## Original GeoPT Reference

This project is motivated by the lifted geometric pre-training idea proposed in:

Wu, Haixu; Guo, Minghao; Li, Zongyi; Dou, Zhiyang; Long, Mingsheng; He, Kaiming; Matusik, Wojciech.
**GeoPT: Scaling Physics Simulation via Lifted Geometric Pre-Training.** arXiv:2602.20399v1, 2026.
https://arxiv.org/abs/2602.20399

The original GeoPT paper and checkpoint are external to this repository.  Users
should consult the upstream GeoPT release for checkpoint availability, license
terms, and exact pretrained-model provenance.

```bibtex
@article{wu2026geopt,
  title = {GeoPT: Scaling Physics Simulation via Lifted Geometric Pre-Training},
  author = {Wu, Haixu and Guo, Minghao and Li, Zongyi and Dou, Zhiyang and Long, Mingsheng and He, Kaiming and Matusik, Wojciech},
  journal = {arXiv preprint arXiv:2602.20399},
  year = {2026}
}
```

## Citation

Archived release:

- Version DOI: https://doi.org/10.5281/zenodo.20130627
- Concept DOI: https://doi.org/10.5281/zenodo.20130626

Use `CITATION.cff` for citation metadata.

## License

- Code and scripts are licensed under the MIT License.  See `LICENSE`.
- Reports, documentation, and selected figures are licensed under CC-BY-4.0.
  See `LICENSE-DOCS.md`.
- The original GeoPT paper and checkpoint are external to this repository and
  are not redistributed here.

## Initial Scope

1. Build a small, reproducible OpenFOAM-solved solid-conduction benchmark before CHT.
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
