# R0 Replication Review 2026-05-06

## Decision

The current R0 output does **not** establish independent multi-split replication.

The apparent five-seed result is invalid as a split-seed replication because the initial R0 runner expanded `{split_seed}` incorrectly. The generated run configs point to the same unresolved split path:

```text
outputs/logs/r0_splits/d1_proxy_pilot_300_c5_n8192_label_scarcity_seed{split_seed.json}
```

Therefore, the zero run-level CI in `docs/r0_replication_results.md` is not statistical confirmation. It is a symptom that the same train split and deterministic training setup were reused under different split labels.

## What The Current Numbers Still Suggest

Within the fixed split/training condition that was actually run, `no_boundary_field` remains a strong candidate at 100 and 125 labels:

| train_size | scratch relL2 | no_boundary relL2 | improvement | case win rate |
|---:|---:|---:|---:|---:|
| 50 | 0.072263 | 0.072040 | 0.31% | 0.507 |
| 75 | 0.071817 | 0.071358 | 0.64% | 0.673 |
| 100 | 0.069744 | 0.053986 | 22.59% | 0.867 |
| 125 | 0.070943 | 0.046829 | 33.99% | 0.980 |

The 50/75-label differences remain too small to count as useful GeoPT-style label efficiency. The 100/125-label differences are large enough to keep `no_boundary_field` as a serious baseline/checkpoint, but they must be re-run with truly distinct split files before being treated as replicated.

## GeoPT Interpretation

This result does not prove that Thermal GeoPT is effective, and it also does not reject the hypothesis.

The robust interpretation is narrower:

- the current random boundary-field prompt is not helping transfer;
- removing `q_near` can help under the present D1 proxy setup;
- the original GeoPT idea, especially dynamics-lifted self-supervision, is still not tested by R0.

The research objective remains GeoPT transfer to thermal surrogates. R0 is therefore a screening step, not the main claim.

## Fix Applied

The R0 scripts now correctly replace `{split_seed}` and default to `RUN_PREFIX=d1_r0_v2`, so the next run will not silently reuse the stale `d1_r0_*` checkpoints. The summarizer also emits integrity warnings for unresolved placeholders and identical split paths.

## Next Action

Re-run R0 after the fix:

```bash
bash scripts/run_r0_replication_matrix.sh
bash scripts/run_r0_test_eval.sh
```

Do not interpret the old `d1_r0_*` summary as independent split replication.

If the corrected R0 keeps a clear 100/125-label advantage across true split seeds, proceed to R1 dynamics-lifted pretraining as the main GeoPT-aligned test. Do not scale current TDF-only pretraining before R1 shows transfer value.
