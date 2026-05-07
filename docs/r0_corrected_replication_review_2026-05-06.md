# R0 Corrected Replication Review 2026-05-06

## Data Integrity

The corrected R0 artifacts are internally consistent.

- `outputs/logs/r0_replication_summary.json` has `integrity_warnings=[]`.
- Checkpoint directories: 40 / expected 40.
- Test evaluation JSON files: 40 / expected 40.
- Split files are distinct for seeds `42, 43, 44, 45, 46`.
- Run configs point to the expected split files, e.g. `...label_scarcity_seed42.json` through `...seed46.json`.

This means the previous split-path issue is resolved. The current R0 result can be interpreted as a real 5-split-seed replication attempt.

## Main Result

`no_boundary_field` versus scratch:

| labels | scratch relL2 | no_boundary relL2 | run improvement | case win rate |
|---:|---:|---:|---:|---:|
| 50 | 0.072140 +/- 0.000196 | 0.071947 +/- 0.000281 | 0.27% +/- 0.14 | 0.633 |
| 75 | 0.071590 +/- 0.000525 | 0.071318 +/- 0.000965 | 0.38% +/- 0.70 | 0.523 |
| 100 | 0.065447 +/- 0.007865 | 0.059261 +/- 0.006004 | 7.92% +/- 15.12 | 0.671 |
| 125 | 0.066516 +/- 0.007392 | 0.054812 +/- 0.006726 | 15.21% +/- 21.62 | 0.819 |

Split-wise improvement:

```text
50 labels:  +0.31, +0.15, +0.25, +0.51, +0.11%
75 labels:  +0.64, -0.48, +1.61, -0.11, +0.25%
100 labels: +22.59, +1.23, +14.25, -19.08, +20.63%
125 labels: +33.99, +23.34, +29.23, -27.39, +16.88%
```

## Interpretation

The corrected R0 gate remains negative for the original GeoPT-style label-efficiency claim.

The 50/75-label regime shows no useful improvement. The 100/125-label regime has a clear positive mean and 4/5 positive splits, but the run-level CI is wide and split45 is strongly negative.

Therefore:

1. `no_boundary_field` is a useful diagnostic baseline.
2. `no_boundary_field` is not a successful Thermal GeoPT result by itself.
3. The current TDF-only/static pretraining line should not be scaled up as the main experiment.

## Split45

Split45 is not a bookkeeping error. Its train subsets are balanced by family, but scratch is unusually strong at 100/125 labels:

```text
train100 split45 scratch relL2=0.049594, no_boundary relL2=0.059055
train125 split45 scratch relL2=0.051573, no_boundary relL2=0.065701
```

This suggests either:

- the split45 training subset is particularly favorable for scratch;
- `no_boundary_field` causes negative transfer for some downstream subsets;
- the D1 proxy has split-dependent shortcuts that can hide pretraining benefits.

Do not remove split45 from summaries. Treat it as a failure case to preserve the rigor of the gate.

## GeoPT Implication

R0 does not test the core GeoPT transfer hypothesis strongly enough. It mainly tests whether a TDF-like geometric initialization, with random boundary field removed, improves downstream label efficiency.

The core hypothesis is still:

> Dynamics-lifted self-supervision from Brownian trajectories and boundary interactions learns a representation that transfers to thermal surrogate tasks.

That is R1, not R0.

## Decision

- Scale current TDF-only pretraining: **No-Go**.
- Claim Thermal GeoPT effectiveness: **Hold**.
- Keep `no_boundary_field` as baseline: **Yes**.
- Proceed to R1 dynamics-lifted pretraining pilot: **Go**.

## Next Plan

1. Train `pretrain_r1_dynamics_lifted_no_boundary_ep2`.
2. Evaluate `dynamics_lifted_no_boundary` under the same split seeds and train sizes.
3. Compare against corrected R0 scratch and old `no_boundary_field`.
4. Require stable low-label improvement before any large-scale pretraining.
5. If R1 only helps at 100/125 and not 50/75, keep it as a candidate but move toward a more physical steady-conduction target before scaling.
