# Label-Scarcity Gate Results

This report is generated from evaluation JSON files. Missing runs are left blank.

## Test Relative L2

| train_size | scratch | full | static_tdf_only | no_boundary_field |
|---:|---:|---:|---:|---:|
| 10 | 0.072236 | 0.072332 | 0.071995 | 0.071925 |
| 25 | 0.072213 | 0.072294 | 0.072235 | 0.072247 |
| 50 | 0.072263 | 0.072227 | 0.072192 | 0.072040 |
| 100 | 0.069744 | 0.070925 | 0.070556 | 0.053986 |

## Improvement vs Scratch

| train_size | group | relative_l2_improvement_pct |
|---:|---|---:|
| 10 | full | -0.132943 |
| 10 | static_tdf_only | 0.333837 |
| 10 | no_boundary_field | 0.430450 |
| 25 | full | -0.112212 |
| 25 | static_tdf_only | -0.029669 |
| 25 | no_boundary_field | -0.046049 |
| 50 | full | 0.050583 |
| 50 | static_tdf_only | 0.098970 |
| 50 | no_boundary_field | 0.308323 |
| 100 | full | -1.692762 |
| 100 | static_tdf_only | -1.163689 |
| 100 | no_boundary_field | 22.593260 |

## Gate Interpretation

Gate status: not positive under the predefined 25/50-label rule.

No pretrained group improves scratch by 10% relative L2 at 25 or 50 labels.

Best observed test relative L2 is `0.053986` from `no_boundary_field` at 100 labels.

## Pretrained Load

| train_size | group | loaded_tensors | skipped_tensors | missing_tensors | source |
|---:|---|---:|---:|---:|---|
| 10 | full | 166 | 3 | 3 | `outputs/checkpoints/pretrain_pilot_300_e20_n4096_ep2/best_model.pt` |
| 10 | static_tdf_only | 166 | 3 | 3 | `outputs/checkpoints/pretrain_gate_static_tdf_only_ep2/best_model.pt` |
| 10 | no_boundary_field | 166 | 3 | 3 | `outputs/checkpoints/pretrain_gate_no_boundary_field_ep2/best_model.pt` |
| 25 | full | 166 | 3 | 3 | `outputs/checkpoints/pretrain_pilot_300_e20_n4096_ep2/best_model.pt` |
| 25 | static_tdf_only | 166 | 3 | 3 | `outputs/checkpoints/pretrain_gate_static_tdf_only_ep2/best_model.pt` |
| 25 | no_boundary_field | 166 | 3 | 3 | `outputs/checkpoints/pretrain_gate_no_boundary_field_ep2/best_model.pt` |
| 50 | full | 166 | 3 | 3 | `outputs/checkpoints/pretrain_pilot_300_e20_n4096_ep2/best_model.pt` |
| 50 | static_tdf_only | 166 | 3 | 3 | `outputs/checkpoints/pretrain_gate_static_tdf_only_ep2/best_model.pt` |
| 50 | no_boundary_field | 166 | 3 | 3 | `outputs/checkpoints/pretrain_gate_no_boundary_field_ep2/best_model.pt` |
| 100 | full | 166 | 3 | 3 | `outputs/checkpoints/pretrain_pilot_300_e20_n4096_ep2/best_model.pt` |
| 100 | static_tdf_only | 166 | 3 | 3 | `outputs/checkpoints/pretrain_gate_static_tdf_only_ep2/best_model.pt` |
| 100 | no_boundary_field | 166 | 3 | 3 | `outputs/checkpoints/pretrain_gate_no_boundary_field_ep2/best_model.pt` |

## Gate Rule

A positive gate means Thermal GeoPT improves scratch by about 10% relative L2 at 25 or 50 labels, or reaches the same error with materially fewer downstream epochs.
