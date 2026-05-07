# R0 Replication Results

This report summarizes multi-seed R0 label-scarcity replication runs.

- split seeds: `[42, 43, 44, 45, 46]`
- train seeds: `[42]`
- groups: `['scratch', 'no_boundary_field']`

## Integrity Warnings

- All split seeds used the same split_path for group=no_boundary_field, train_size=100, train_seed=42: outputs/logs/r0_splits/d1_proxy_pilot_300_c5_n8192_label_scarcity_seed{split_seed.json}
- All split seeds used the same split_path for group=no_boundary_field, train_size=125, train_seed=42: outputs/logs/r0_splits/d1_proxy_pilot_300_c5_n8192_label_scarcity_seed{split_seed.json}
- All split seeds used the same split_path for group=no_boundary_field, train_size=50, train_seed=42: outputs/logs/r0_splits/d1_proxy_pilot_300_c5_n8192_label_scarcity_seed{split_seed.json}
- All split seeds used the same split_path for group=no_boundary_field, train_size=75, train_seed=42: outputs/logs/r0_splits/d1_proxy_pilot_300_c5_n8192_label_scarcity_seed{split_seed.json}
- All split seeds used the same split_path for group=scratch, train_size=100, train_seed=42: outputs/logs/r0_splits/d1_proxy_pilot_300_c5_n8192_label_scarcity_seed{split_seed.json}
- All split seeds used the same split_path for group=scratch, train_size=125, train_seed=42: outputs/logs/r0_splits/d1_proxy_pilot_300_c5_n8192_label_scarcity_seed{split_seed.json}
- All split seeds used the same split_path for group=scratch, train_size=50, train_seed=42: outputs/logs/r0_splits/d1_proxy_pilot_300_c5_n8192_label_scarcity_seed{split_seed.json}
- All split seeds used the same split_path for group=scratch, train_size=75, train_seed=42: outputs/logs/r0_splits/d1_proxy_pilot_300_c5_n8192_label_scarcity_seed{split_seed.json}
- Relative L2 is identical across split seeds for group=no_boundary_field, train_size=100, train_seed=42; treat run-level CI as non-informative until split independence is verified.
- Relative L2 is identical across split seeds for group=no_boundary_field, train_size=125, train_seed=42; treat run-level CI as non-informative until split independence is verified.
- Relative L2 is identical across split seeds for group=no_boundary_field, train_size=50, train_seed=42; treat run-level CI as non-informative until split independence is verified.
- Relative L2 is identical across split seeds for group=no_boundary_field, train_size=75, train_seed=42; treat run-level CI as non-informative until split independence is verified.
- Relative L2 is identical across split seeds for group=scratch, train_size=100, train_seed=42; treat run-level CI as non-informative until split independence is verified.
- Relative L2 is identical across split seeds for group=scratch, train_size=125, train_seed=42; treat run-level CI as non-informative until split independence is verified.
- Relative L2 is identical across split seeds for group=scratch, train_size=50, train_seed=42; treat run-level CI as non-informative until split independence is verified.
- Relative L2 is identical across split seeds for group=scratch, train_size=75, train_seed=42; treat run-level CI as non-informative until split independence is verified.
- Unresolved placeholder in split_path for no_boundary_field train_size=100: outputs/logs/r0_splits/d1_proxy_pilot_300_c5_n8192_label_scarcity_seed{split_seed.json}
- Unresolved placeholder in split_path for no_boundary_field train_size=125: outputs/logs/r0_splits/d1_proxy_pilot_300_c5_n8192_label_scarcity_seed{split_seed.json}
- Unresolved placeholder in split_path for no_boundary_field train_size=50: outputs/logs/r0_splits/d1_proxy_pilot_300_c5_n8192_label_scarcity_seed{split_seed.json}
- Unresolved placeholder in split_path for no_boundary_field train_size=75: outputs/logs/r0_splits/d1_proxy_pilot_300_c5_n8192_label_scarcity_seed{split_seed.json}
- Unresolved placeholder in split_path for scratch train_size=100: outputs/logs/r0_splits/d1_proxy_pilot_300_c5_n8192_label_scarcity_seed{split_seed.json}
- Unresolved placeholder in split_path for scratch train_size=125: outputs/logs/r0_splits/d1_proxy_pilot_300_c5_n8192_label_scarcity_seed{split_seed.json}
- Unresolved placeholder in split_path for scratch train_size=50: outputs/logs/r0_splits/d1_proxy_pilot_300_c5_n8192_label_scarcity_seed{split_seed.json}
- Unresolved placeholder in split_path for scratch train_size=75: outputs/logs/r0_splits/d1_proxy_pilot_300_c5_n8192_label_scarcity_seed{split_seed.json}

## Test Relative L2 Across Runs

| train_size | scratch | no_boundary_field |
|---:|---:|---:|
| 50 | 0.072263 +/- 0.000000 (n=5) | 0.072040 +/- 0.000000 (n=5) |
| 75 | 0.071817 +/- 0.000000 (n=5) | 0.071358 +/- 0.000000 (n=5) |
| 100 | 0.069744 +/- 0.000000 (n=5) | 0.053986 +/- 0.000000 (n=5) |
| 125 | 0.070943 +/- 0.000000 (n=5) | 0.046829 +/- 0.000000 (n=5) |

## Paired Improvement Vs Scratch

| train_size | group | run_improvement_pct | case_improvement_pct | case_win_rate |
|---:|---|---:|---:|---:|
| 50 | no_boundary_field | 0.31 +/- 0.00 (n=5) | 0.36 +/- 0.09 | 0.507 |
| 75 | no_boundary_field | 0.64 +/- 0.00 (n=5) | 0.58 +/- 0.12 | 0.673 |
| 100 | no_boundary_field | 22.59 +/- 0.00 (n=5) | 22.03 +/- 1.40 | 0.867 |
| 125 | no_boundary_field | 33.99 +/- 0.00 (n=5) | 33.97 +/- 1.19 | 0.980 |

## Interpretation Rule

Treat the 100-label no-boundary signal as replicated only if the paired run-level improvement is consistently positive, preferably above 10%, across multiple split seeds and remains visible at 125 labels.

If the improvement appears only at one split seed or collapses at 75/125 labels, keep the original gate negative and move the emphasis to R1 dynamics-lifted pretraining redesign.
