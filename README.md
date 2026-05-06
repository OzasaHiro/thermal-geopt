# Thermal GeoPT

Thermal GeoPT is an experiment project for diffusion-lifted geometric pre-training for heat transfer surrogate models.

The working research direction is based on:

- `thermal_geopt_research_plan_draft.md`
- `thermal_geopt_detailed_research_plan.md`

The current technical stance is to avoid a single scalar `TDF = distance * conductivity` definition. The stronger path is a multi-channel thermal diffusion feature with Brownian random walks, boundary hitting, and synthetic heat-source/cooling-boundary prompts.

## Initial Scope

1. Build a small, reproducible D1 solid-conduction benchmark before CHT.
2. Reuse the GeoPT PoC assets from `../GeoPT` where they are proven useful.
3. Keep Git history focused on code, configs, lightweight reports, and runbooks.
4. Keep raw data, VTP/VTU/STL meshes, NumPy arrays, Zarr shards, checkpoints, and generated figures out of normal Git.

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
- `gh`: installed, but GitHub authentication is not configured yet.
- Shared Python environment: `../../.venv`.
- Verified package baseline: Python 3.12.3, PyTorch 2.10.0+cu128, CUDA 12.8, PyVista 0.47.1.

See `RUNBOOK.md` and `docs/github_setup.md` for the current handoff.
