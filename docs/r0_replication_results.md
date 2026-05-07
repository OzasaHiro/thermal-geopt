# R0 Replication Results

This report summarizes multi-seed R0 label-scarcity replication runs.

- split seeds: `[42, 43, 44, 45, 46]`
- train seeds: `[42]`
- groups: `['scratch', 'no_boundary_field']`

## Test Relative L2 Across Runs

| train_size | scratch | no_boundary_field |
|---:|---:|---:|
| 50 | 0.072140 +/- 0.000196 (n=5) | 0.071947 +/- 0.000281 (n=5) |
| 75 | 0.071590 +/- 0.000525 (n=5) | 0.071318 +/- 0.000965 (n=5) |
| 100 | 0.065447 +/- 0.007865 (n=5) | 0.059261 +/- 0.006004 (n=5) |
| 125 | 0.066516 +/- 0.007392 (n=5) | 0.054812 +/- 0.006726 (n=5) |

## Paired Improvement Vs Scratch

| train_size | group | run_improvement_pct | case_improvement_pct | case_win_rate |
|---:|---|---:|---:|---:|
| 50 | no_boundary_field | 0.27 +/- 0.14 (n=5) | 0.27 +/- 0.24 | 0.633 |
| 75 | no_boundary_field | 0.38 +/- 0.70 (n=5) | 0.29 +/- 0.24 | 0.523 |
| 100 | no_boundary_field | 7.92 +/- 15.12 (n=5) | 6.35 +/- 1.92 | 0.671 |
| 125 | no_boundary_field | 15.21 +/- 21.62 (n=5) | 14.45 +/- 2.23 | 0.819 |

## Interpretation Rule

Treat the 100-label no-boundary signal as replicated only if the paired run-level improvement is consistently positive, preferably above 10%, across multiple split seeds and remains visible at 125 labels.

If the improvement appears only at one split seed or collapses at 75/125 labels, keep the original gate negative and move the emphasis to R1 dynamics-lifted pretraining redesign.
