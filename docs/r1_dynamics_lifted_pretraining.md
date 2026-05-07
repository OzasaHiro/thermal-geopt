# R1 Dynamics-Lifted Pretraining

R1 restores the part of the GeoPT idea that the first gate did not actually test: large-scale dynamics-lifted self-supervision from Brownian trajectories and boundary interaction. This keeps the experiment aligned with the original purpose: checking whether GeoPT-style pretraining transfers to thermal surrogate learning.

The current 300-shape / 6,000-episode data is only a lower-bound plumbing artifact. It can verify schema, loss calculation, checkpoint loading, and smoke training. It cannot support a claim that Thermal GeoPT is effective or ineffective. Meaningful efficacy evaluation starts at P2-scale data, and a paper-level main result requires P3-scale data if P2 is positive.

## Planned Scale Tiers

| Tier | Shapes | Episodes/shape | Total episodes | Point tier | Role |
|---|---:|---:|---:|---|---|
| P0 debug | 100 | 5 | 500 | tiny/pilot | implementation check only |
| P1 pilot | 500 | 10 | 5,000 | pilot 5,120 | loss/speed/data-quality check |
| P2 first result | 2,000 | 20 | 40,000 | base candidate 8,192-9,216 | first downstream efficacy gate |
| P3 main | 8,000 | 20 | 160,000 | base 9,216+ | paper minimum if P2 is positive |
| P4 expanded | 10k-12k | 50 | 500k-600k | base/base+ | stronger ML-scale result |

Do not interpret P0/P1 as a method validation. They are guards against wasting time on malformed data or unstable training.

## What Changed

`scripts/generate_pretrain_episodes.py` already stores:

- `trajectory`
- `hit_mask`
- `hit_step`

It also supports a D1-aligned thermal condition schema:

```bash
--condition-schema d1_thermal
```

This writes point-wise prompt channels:

- `conductivity`
- `source_temperature`
- `sink_temperature`
- `source_patch`
- `sink_patch`
- `nearest_boundary_distance`

The legacy schema `alpha, conductivity, q_near` remains available for old artifacts and ablations, but it is not the preferred R1 path because it does not match the downstream D1 thermal prompt.

`thermal_geopt.datasets.PretrainZarrDataset` can now use them with:

```bash
--pretext-ablation dynamics_lifted
```

The model target becomes:

- existing TDF channels
- first-step Brownian displacement
- final Brownian displacement
- boundary hit indicator
- normalized hit/survival step

`scripts/train_pretrain.py` keeps a single Transolver output tensor, but logs separate loss components for TDF, trajectory, boundary hit, and hit step. This is intentionally minimal: it changes the self-supervised signal without changing the downstream fine-tuning pipeline.

## Smoke

```bash
cd /home/hiroaki-ozasa/Desktop/Development/PINN_CFD/EXPERIMENT/Thermal_GeoPT
PY=../../.venv/bin/python
```

Generate a tiny D1-schema shard only to verify the schema:

```bash
$PY scripts/generate_pretrain_episodes.py \
  --processed-dir data/meshes_processed/cadquery_pilot_300 \
  --output-dir data/pretrain_zarr/d1_thermal_tiny_smoke \
  --max-shapes 2 \
  --episodes-per-shape 1 \
  --points-per-episode 256 \
  --steps 2 \
  --condition-schema d1_thermal \
  --seed 42 \
  --overwrite
```

Check the generated shard:

```bash
$PY scripts/check_pretrain_readiness.py \
  data/pretrain_zarr/d1_thermal_tiny_smoke/manifest.json
```

Run one short dynamics-lifted training smoke:

```bash
$PY scripts/train_pretrain.py \
  --manifest data/pretrain_zarr/d1_thermal_tiny_smoke/manifest.json \
  --output-dir outputs/checkpoints/pretrain_r1_d1_thermal_dynamics_tiny_smoke \
  --pretext-ablation dynamics_lifted \
  --condition-mode full \
  --epochs 1 \
  --batch-size 1 \
  --point-budget 128 \
  --max-episodes 2 \
  --amp \
  --amp-dtype bfloat16 \
  --device cuda \
  --seed 42
```

## P1 Pilot Data

P1 checks data quality and training stability. It is still not enough to validate GeoPT efficacy.

```bash
$PY scripts/generate_cadquery_shapes.py \
  --output-dir data/meshes_raw/cadquery_p1_700 \
  --num-per-family 100 \
  --seed 42 \
  --overwrite
```

```bash
$PY scripts/preprocess_meshes.py \
  --input-dir data/meshes_raw/cadquery_p1_700 \
  --output-dir data/meshes_processed/cadquery_p1_700 \
  --surface-points 8192 \
  --seed 42 \
  --overwrite
```

```bash
$PY scripts/generate_pretrain_episodes.py \
  --processed-dir data/meshes_processed/cadquery_p1_700 \
  --output-dir data/pretrain_zarr/cadquery_p1_d1_thermal_500_e10_n5120 \
  --max-shapes 500 \
  --selection balanced \
  --episodes-per-shape 10 \
  --points-per-episode 5120 \
  --steps 2 \
  --condition-schema d1_thermal \
  --seed 42 \
  --overwrite
```

```bash
$PY scripts/check_pretrain_readiness.py \
  data/pretrain_zarr/cadquery_p1_d1_thermal_500_e10_n5120/manifest.json \
  --require-phase P1
```

## P2 First-Result Data

P2 is the first scale at which downstream transfer should be treated as meaningful. If P2 does not show label-efficiency improvement on solver-backed D1, do not proceed to P3 until the pretext, prompt, or downstream task is corrected.

```bash
$PY scripts/generate_cadquery_shapes.py \
  --output-dir data/meshes_raw/cadquery_p2_2100 \
  --num-per-family 300 \
  --seed 42 \
  --overwrite
```

```bash
$PY scripts/preprocess_meshes.py \
  --input-dir data/meshes_raw/cadquery_p2_2100 \
  --output-dir data/meshes_processed/cadquery_p2_2100 \
  --surface-points 12288 \
  --seed 42 \
  --overwrite
```

```bash
$PY scripts/generate_pretrain_episodes.py \
  --processed-dir data/meshes_processed/cadquery_p2_2100 \
  --output-dir data/pretrain_zarr/cadquery_p2_d1_thermal_2000_e20_n8192 \
  --max-shapes 2000 \
  --selection balanced \
  --episodes-per-shape 20 \
  --points-per-episode 8192 \
  --steps 2 \
  --condition-schema d1_thermal \
  --seed 42 \
  --overwrite
```

```bash
$PY scripts/check_pretrain_readiness.py \
  data/pretrain_zarr/cadquery_p2_d1_thermal_2000_e20_n8192/manifest.json \
  --require-phase P2
```

## P2 Training Candidate

Use this only after the P2 dataset passes readiness checks:

```bash
$PY scripts/train_pretrain.py \
  --manifest data/pretrain_zarr/cadquery_p2_d1_thermal_2000_e20_n8192/manifest.json \
  --output-dir outputs/checkpoints/pretrain_r1_d1_thermal_dynamics_p2_ep20 \
  --pretext-ablation dynamics_lifted \
  --condition-mode full \
  --epochs 20 \
  --batch-size 1 \
  --point-budget 8192 \
  --max-episodes 0 \
  --amp \
  --amp-dtype bfloat16 \
  --device cuda \
  --seed 42
```

If this is stable and the solver-backed D1 downstream gate is positive, extend the same recipe to P3 rather than drawing a paper-level conclusion from P2 alone.

## P3 Main Data

P3 is the minimum scale for the main paper claim. Do not run this until P2 plus solver-backed D1 gives a positive signal.

```bash
$PY scripts/generate_cadquery_shapes.py \
  --output-dir data/meshes_raw/cadquery_p3_8400 \
  --num-per-family 1200 \
  --seed 42 \
  --overwrite
```

```bash
$PY scripts/preprocess_meshes.py \
  --input-dir data/meshes_raw/cadquery_p3_8400 \
  --output-dir data/meshes_processed/cadquery_p3_8400 \
  --surface-points 16384 \
  --seed 42 \
  --overwrite
```

```bash
$PY scripts/generate_pretrain_episodes.py \
  --processed-dir data/meshes_processed/cadquery_p3_8400 \
  --output-dir data/pretrain_zarr/cadquery_p3_d1_thermal_8000_e20_n9216 \
  --max-shapes 8000 \
  --selection balanced \
  --episodes-per-shape 20 \
  --points-per-episode 9216 \
  --steps 2 \
  --condition-schema d1_thermal \
  --seed 42 \
  --overwrite
```

```bash
$PY scripts/check_pretrain_readiness.py \
  data/pretrain_zarr/cadquery_p3_d1_thermal_8000_e20_n9216/manifest.json \
  --require-phase P3
```

Training template:

```bash
$PY scripts/train_pretrain.py \
  --manifest data/pretrain_zarr/cadquery_p3_d1_thermal_8000_e20_n9216/manifest.json \
  --output-dir outputs/checkpoints/pretrain_r1_d1_thermal_dynamics_p3_ep100 \
  --pretext-ablation dynamics_lifted \
  --condition-mode full \
  --epochs 100 \
  --batch-size 1 \
  --point-budget 9216 \
  --max-episodes 0 \
  --amp \
  --amp-dtype bfloat16 \
  --device cuda \
  --seed 42
```

## Downstream Matrix

After the R1 checkpoint exists, compare it under the solver-backed D1 label-scarcity protocol. This should be treated as an R1 gate, not as a replacement for the corrected R0 proxy result.

Run only the new R1 group:

```bash
GATE_GROUPS="dynamics_lifted_no_boundary" \
RUN_PREFIX=d1_r1 \
PRETRAIN_DYNAMICS=outputs/checkpoints/pretrain_r1_d1_thermal_dynamics_p2_ep20 \
bash scripts/run_r0_replication_matrix.sh
```

Evaluate the new R1 group:

```bash
GATE_GROUPS="dynamics_lifted_no_boundary" \
RUN_PREFIX=d1_r1 \
SUMMARY_JSON=outputs/logs/r1_dynamics_lifted_summary.json \
SUMMARY_MD=docs/r1_dynamics_lifted_results.md \
bash scripts/run_r0_test_eval.sh
```

For final comparison against corrected R0, compare `docs/r0_replication_results.md` with `docs/r1_dynamics_lifted_results.md`. The key question is whether R1 beats scratch, static TDF, and old `no_boundary_field`, especially at 25/50 labels on solver-backed D1.

## Interpretation

R1 should be judged against scratch, static TDF, geometry-only/no-boundary ablations, and eventually the fluid GeoPT checkpoint under the same label-scarcity protocol. It is meaningful only if it improves label efficiency, hotspot reliability, or convergence speed under scarce downstream labels.

Claims by scale:

- P0/P1: schema, data quality, loss stability, throughput only.
- P2: first credible transfer signal, still not a paper-level scale claim.
- P3: minimum scale for the main paper claim if D1 solver-backed results are positive.
- P4: stronger ML-scale evidence and scale-ablation support.
