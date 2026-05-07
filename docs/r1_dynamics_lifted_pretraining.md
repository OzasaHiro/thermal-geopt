# R1 Dynamics-Lifted Pretraining

R1 restores the part of the GeoPT idea that the first gate did not actually test: dynamics-lifted self-supervision from Brownian trajectories and boundary interaction. This keeps the experiment aligned with the original purpose: checking whether GeoPT-style pretraining transfers to thermal surrogate learning.

## What Changed

`scripts/generate_pretrain_episodes.py` already stores:

- `trajectory`
- `hit_mask`
- `hit_step`

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
PRETRAIN_MANIFEST=data/pretrain_zarr/cadquery_pilot_300_e20_n4096/manifest.json
```

```bash
$PY scripts/train_pretrain.py \
  --manifest $PRETRAIN_MANIFEST \
  --output-dir outputs/checkpoints/pretrain_r1_dynamics_lifted_smoke \
  --pretext-ablation dynamics_lifted \
  --condition-mode zero_boundary_field \
  --epochs 1 \
  --batch-size 1 \
  --point-budget 128 \
  --max-episodes 2 \
  --amp \
  --amp-dtype bfloat16 \
  --device cuda \
  --seed 42
```

## Pilot Candidate

Run this only after R0 replication is started or queued:

```bash
$PY scripts/train_pretrain.py \
  --manifest $PRETRAIN_MANIFEST \
  --output-dir outputs/checkpoints/pretrain_r1_dynamics_lifted_no_boundary_ep2 \
  --pretext-ablation dynamics_lifted \
  --condition-mode zero_boundary_field \
  --epochs 2 \
  --batch-size 1 \
  --point-budget 4096 \
  --max-episodes 0 \
  --amp \
  --amp-dtype bfloat16 \
  --device cuda \
  --seed 42
```

Then use it in downstream fine-tuning as a new R1 group, not inside the R0 replication summary.

## Interpretation

R1 should be judged against scratch and the old `no_boundary_field` checkpoint under the same label-scarcity protocol. It is meaningful only if it improves label efficiency or convergence speed under scarce downstream labels. A larger dynamics-lifted pretraining run is justified only after this small R1 checkpoint shows a reproducible transfer signal.
