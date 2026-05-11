# M4 Heat-Sink Solver-Backed D1 Transfer Gate Results

This report summarizes geometry-varied solver-backed heat-sink D1 label-scarcity transfer runs comparing scratch, Thermal GeoPT pretraining, and the original GeoPT checkpoint.

- split seeds: `[42, 43, 44]`
- train seeds: `[42]`
- groups: `['scratch', 'geopt_transport_lifted', 'geopt_original']`

## Test Relative L2 Across Runs

| train_size | scratch | geopt_transport_lifted | geopt_original |
|---:|---:|---:|---:|
| 25 | 0.013122 +/- 0.000685 (n=3) | 0.037595 +/- 0.001770 (n=3) | 0.011370 +/- 0.001009 (n=3) |
| 50 | 0.010674 +/- 0.000470 (n=3) | 0.022637 +/- 0.000570 (n=3) | 0.008840 +/- 0.000345 (n=3) |

## Paired Improvement Vs Scratch

| train_size | group | run_improvement_pct | case_improvement_pct | case_win_rate |
|---:|---|---:|---:|---:|
| 25 | geopt_transport_lifted | -187.06 +/- 23.36 (n=3) | -193.24 +/- 23.23 | 0.007 |
| 25 | geopt_original | 13.39 +/- 4.84 (n=3) | 13.11 +/- 2.33 | 0.852 |
| 50 | geopt_transport_lifted | -112.41 +/- 14.35 (n=3) | -117.43 +/- 13.95 | 0.030 |
| 50 | geopt_original | 17.15 +/- 2.65 (n=3) | 17.29 +/- 1.53 | 0.963 |

## Interpretation Rule

A positive M4 signal means a pretrained group improves scratch by about 10% relative L2 at 25 or 50 labels, or reaches similar error with fewer labels, without degrading max-temperature or hotspot metrics. Original GeoPT is a control for general GeoPT geometry-boundary priors; Thermal GeoPT must beat or complement it to support a thermal-specific pretraining claim.
