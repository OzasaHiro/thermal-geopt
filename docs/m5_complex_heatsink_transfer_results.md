# M5 Complex Heat-Sink Solver-Backed D1 Transfer Gate Results

This report summarizes more complex solver-backed heat-sink D1 label-scarcity transfer runs comparing scratch, Thermal GeoPT pretraining, and the original GeoPT checkpoint.

- split seeds: `[42, 43, 44]`
- train seeds: `[42]`
- groups: `['scratch', 'geopt_transport_lifted', 'geopt_original']`

## Test Relative L2 Across Runs

| train_size | scratch | geopt_transport_lifted | geopt_original |
|---:|---:|---:|---:|
| 25 | 0.012548 +/- 0.000259 (n=3) | 0.032063 +/- 0.001846 (n=3) | 0.011186 +/- 0.000006 (n=3) |
| 50 | 0.010613 +/- 0.000060 (n=3) | 0.022536 +/- 0.002766 (n=3) | 0.008950 +/- 0.000194 (n=3) |

## Paired Improvement Vs Scratch

| train_size | group | run_improvement_pct | case_improvement_pct | case_win_rate |
|---:|---|---:|---:|---:|
| 25 | geopt_transport_lifted | -155.52 +/- 13.48 (n=3) | -160.04 +/- 17.34 | 0.000 |
| 25 | geopt_original | 10.83 +/- 1.88 (n=3) | 10.40 +/- 2.36 | 0.844 |
| 50 | geopt_transport_lifted | -112.28 +/- 25.01 (n=3) | -116.57 +/- 12.82 | 0.007 |
| 50 | geopt_original | 15.67 +/- 1.53 (n=3) | 15.61 +/- 2.06 | 0.911 |

## Interpretation Rule

A positive M5 signal means a pretrained group improves scratch by about 10% relative L2 at 25 or 50 labels and does not degrade max-temperature or hotspot metrics. Original GeoPT is the primary cross-physics transfer control.
