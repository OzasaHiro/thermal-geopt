# Experiment Realignment To Detailed Plan 2026-05-06

## Why This Document Exists

The experiment drifted too far toward proxy/pipeline checks. That was useful for finding bugs, but it is not enough for the actual research goal:

> Validate whether GeoPT-style dynamics-lifted pretraining improves heat-transfer surrogate modeling.

This note records the course correction from proxy/pipeline checks toward
solver-generated heat-transfer benchmarks.

## Correction Of Course

The current D1 proxy is not a heat-transfer simulation dataset. It is a deterministic source/sink influence field used for plumbing checks. It must not be treated as the main downstream task for a paper claim.

Going forward:

1. Keep the existing D1 proxy only as smoke/pipeline validation.
2. Freeze R0 as a diagnostic result, not a research result.
3. Move the main downstream task to solver-backed D1 solid conduction.
4. Move the main pretraining line to dynamics-lifted Thermal GeoPT.
5. Prepare D1-aligned P1/P2 dynamics-lifted pretraining data; reserve P3/P4 scale-up until solver-backed D1 shows transfer value.

## Detailed Plan Alignment

The detailed plan defines the paper-level path as:

| Stage | Role | Status |
|---|---|---|
| D1 solid conduction | Main experiment | Not implemented yet |
| D2 simplified CHT | Strong secondary experiment | Not started |
| D3 OOD geometry transfer | Paper-strengthening experiment | Not started |
| Dynamics-lifted pretraining | Core GeoPT transfer claim | Minimal implementation exists; not yet validated |
| Static TDF / no-boundary | Ablation/baseline only | R0 diagnostic done |

## Current State

### What Is Done

- STL/CadQuery geometry generation pipeline exists.
- Pretraining shard generation exists.
- Brownian trajectory, hit mask, and hit step are generated and stored.
- `dynamics_lifted` pretraining target mode exists.
- Label-scarcity runners and summaries exist.
- R0 corrected replication is internally consistent.

### What Is Not Done

- No solver-backed thermal downstream dataset exists.
- Current downstream labels are not FEM/FVM/OpenFOAM results.
- No D1 solid conduction benchmark exists.
- No max temperature / hotspot / energy residual evaluation exists.
- No paper-level A/B/C/D/E experiment matrix exists.

### Environment Finding

OpenFOAM Foundation v13 is available on the target machine once `/opt/openfoam13/etc/bashrc` is sourced.
The following commands have been confirmed by the user:

- `foamVersion`: `OpenFOAM-13`
- `blockMesh`
- `snappyHexMesh`
- `surfaceFeatureExtract`
- `laplacianFoam`
- `chtMultiRegionFoam`

Foundation v13 does not provide `foamListApplications`; use `foamInfo <name>` for solver/application checks instead.
The D1 solver-backed benchmark must therefore target OpenFOAM `laplacianFoam` first. Do not switch to a portable FEM/FVM substitute unless the experiment owner explicitly approves that change.

## Main Experimental Claim

The paper should not claim that static geometry pretraining helps heat transfer.

The intended claim is:

> Diffusion-lifted self-supervision from Brownian trajectories, boundary hitting, and thermal boundary interaction learns a heat-aware geometric prior that improves data efficiency and hotspot reliability for solver-backed heat-transfer surrogate tasks.

## Required Experiment Groups

Minimum paper-level groups:

| Group | Meaning | Required |
|---|---|---|
| A | Thermal GeoPT full: Brownian + TDF + boundary hit + thermal prompt | Yes |
| B | Scratch Transolver | Yes |
| C | Static TDF pretraining | Yes |
| D | VDF/SDF geometry-only pretraining | Yes |
| E | Brownian trajectory without q/h boundary prompt | Yes |
| G | Fluid GeoPT checkpoint transfer | Strongly recommended |

Existing `no_boundary_field` belongs under E-like diagnostic baselines, not as the proposed method.

## Corrected Milestones

### M0: Freeze Proxy Work

Outcome:

- D1 proxy remains available for smoke.
- R0 corrected replication is recorded as diagnostic.
- No further optimization of proxy-only label scarcity results unless it supports debugging.

Decision:

- Completed.

### M1: Build Solver-Backed D1 Solid Conduction

Purpose:

Create the actual main downstream task from the detailed plan.

Equation:

```text
-div(k grad T) = Q in solid domain

Boundary:
  -k grad T dot n = q on heat-source patches
  -k grad T dot n = h(T - T_inf) on cooling patches
   grad T dot n = 0 on insulated patches
```

Implementation options:

1. OpenFOAM Foundation v13 `laplacianFoam` path.
2. Portable FEM/FVM path only if explicitly approved as a contingency, not as the default route.

Minimum output schema:

| Field | Required |
|---|---|
| `points` | Yes |
| `conditions` | Yes |
| `temperature` | Yes |
| `boundary/source/sink masks` | Yes |
| `case_params` | Yes |
| `T_max` | Yes |
| `hotspot_position` | Yes |
| `energy_balance_terms` | If available |

Pilot scale:

- 50 train / 20 test.
- 20k-100k points per case if feasible, with downsampled training.

Main scale:

- 300-800 total cases.
- train sizes: 10 / 25 / 50 / 100 / 200.
- test: 50-100.

### M2: Validate R1 Dynamics-Lifted Pretraining

Purpose:

Test the actual GeoPT-like mechanism, not static proxy initialization.

Pretraining target:

- TDF trajectory.
- Brownian first/final displacement.
- Boundary hit probability.
- Hit/survival step.
- Later: boundary influence from source/sink or q/h fields.

Short-term checkpoint:

```text
outputs/checkpoints/pretrain_r1_d1_thermal_dynamics_p2_ep20
```

This is not paper-scale pretraining. It is the first-result scale used to test transfer direction. Tiny/P1 runs are only schema, data-quality, and throughput checks.

### M3: D1 Solver-Backed Label Efficiency Gate

Purpose:

Run the first meaningful transfer test.

Groups:

- B scratch.
- C static TDF.
- E Brownian/no-boundary.
- A dynamics-lifted Thermal GeoPT.
- Optional G fluid GeoPT transfer.

Train sizes:

```text
10 / 25 / 50 / 100 / 200
```

Metrics:

- Relative L2.
- RMSE/MAE.
- max temperature error.
- hotspot localization error.
- boundary/source/sink region error.
- time-to-target.
- data saving at matched error.

Go condition:

- A improves scratch by at least 10% Relative L2 on D1, or
- A reaches the same error with at least 30% fewer solver-labeled cases, and
- max temperature error does not degrade.

### M4: Pretraining Scale-Up

P1/P2 are part of making M3 meaningful. P3/P4 are the scale-up stages and should happen only after M3 passes.

Scale targets from detailed plan:

| Phase | Shapes | episodes |
|---|---:|---:|
| P1 pilot | 500 | 5,000 |
| P2 first result | 2,000 | 40,000 |
| P3 main | 8,000 | 160,000 |

Do not run P2/P3 just because R0 proxy has a weak signal. Scale-up requires solver-backed D1 evidence.

More precise rule:

- P1: run for quality and throughput checks.
- P2: run for the first solver-backed D1 transfer gate.
- P3/P4: run only after P2 + solver-backed D1 gives a positive signal.

### M5: D2 Simplified CHT

After D1 solver-backed result is positive.

Scope:

- channel + heat sink.
- fixed or low-variety flow first.
- train 20 / 50 / 100, test 20-50.

This is for a stronger workshop/journal paper, not for the first debugging pass.

### M6: Paper Package

Minimum publishable result:

- D1 solver-backed benchmark.
- A/B/C/D/E comparison.
- label-efficiency curve.
- maxT/hotspot reliability.
- ablations proving dynamics-lifted > static geometry.

Stronger paper:

- D1 + D2.
- OOD family split.
- fluid GeoPT checkpoint baseline.
- pretraining scale ablation.

## Immediate Next Tasks

1. Stop using D1 proxy as evidence.
2. Implement or integrate solver-backed D1 solid conduction data generation.
3. Add evaluation for maxT and hotspot.
4. Prepare R1 D1-aligned dynamics-lifted P1/P2 pretraining data while D1 solver work is built.
5. Re-run label-scarcity only on solver-backed D1.

## Bottom Line

The previous proxy work found useful plumbing bugs, but it is not the experiment that can support the paper.

The correct next technical task is not more proxy tuning. It is building D1 solid conduction labels from a solver and then using dynamics-lifted Thermal GeoPT as the proposed method in the A/B/C/D/E experiment matrix.
