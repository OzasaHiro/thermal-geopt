# M3 Fine-Tuning Protocol Review 2026-05-07

## Conclusion

The current M3 run uses the same downstream optimizer for scratch and pretrained models:

- `AdamW`
- constant `lr=1e-3`
- `weight_decay=1e-5`
- `epochs=50`
- all parameters updated from the first step

This run is useful as an aggressive same-LR stress test, but it should not be the final judgment of Thermal GeoPT pretraining. Updating all pretrained backbone tensors with the same high LR as scratch can erase useful pretraining, especially in the low-label M3 setting.

The final M3 decision should include a pretrained-safe fine-tuning protocol with lower LR for loaded backbone tensors and higher LR for the newly initialized prediction head.

## Evidence Used

Original GeoPT paper:

- downstream fine-tuning is not just checkpoint loading; it reconfigures the synthetic dynamics prompt into the downstream physics prompt;
- standard downstream optimization uses AdamW and OneCycle-style training over a longer budget;
- data efficiency and convergence speed are part of the claim, not only final error under one fixed optimizer.

Previous `EXPERIMENT/GeoPT` PoC:

- `A_geopt_finetune_official_like_200ep_30k_bf16` used AdamW, `lr=1e-3`, OneCycleLR, fixed sampling, bf16 AMP, 200 epochs;
- OneCycle started around `4e-5`, not at full `1e-3`, which protected the pretrained model early in training;
- an experimental fine-tune-params smoke existed with `backbone_lr=3e-5`, `head_lr=1e-3`, `freeze_backbone_epochs=1`, `max_grad_norm=1.0`;
- the best previous PoC interpretation came from comparing a strong scratch baseline with a balanced pretrained fine-tune, not from one naive LR setting.

ML research review:

- scratch and pretrained should have the same compute budget and validation protocol;
- pretrained models should use discriminative learning rates because most tensors are loaded while the downstream head is newly initialized;
- head-only and staged-unfreeze ablations are useful to separate representation quality from full fine-tuning behavior.

## Recommended Protocols

### Protocol A: Current Same-LR Stress Test

Keep the current run result, but label it correctly.

Purpose:

- checks whether P2 initialization survives an aggressive all-layer `1e-3` fine-tune;
- useful forgetting diagnostic;
- not sufficient for No-Go if pretrained is weak.

Settings:

- scratch: all params `lr=1e-3`, constant
- pretrained: all params `lr=1e-3`, constant
- epochs: 50
- report separately as `same_lr_constant_50ep`

### Protocol B: GeoPT-Faithful Optimizer

This mirrors the previous GeoPT PoC and the paper's fine-tuning style more closely.

Settings:

- scratch: all params `lr=1e-3`
- pretrained: all params `lr=1e-3`
- scheduler: `onecycle`
- `pct_start=0.3`
- max grad norm: `1.0`
- epochs: 100 for gate, 200 for final confirmation if promising

Purpose:

- tests whether the current constant-LR result was just an optimizer artifact;
- gives scratch and pretrained the same optimizer shape.

### Protocol C: Pretrained-Protective Fine-Tuning

This is the recommended main practical protocol for judging whether the P2 checkpoint contains useful transferable information.

Settings:

- scratch: all params `lr=1e-3`
- pretrained loaded backbone: `lr=3e-4` as default, with `1e-4` as conservative backup
- pretrained newly initialized head: `lr=1e-3`
- scheduler: `onecycle`
- `pct_start=0.3`
- freeze loaded backbone for 0 or 5 epochs; use 5 epochs as the default protective run
- max grad norm: `1.0`
- epochs: 100 for gate, 200 only for final confirmation

Rationale:

- the head must learn quickly because the pretraining output head is shape-incompatible and skipped;
- the backbone should adapt slowly enough not to erase the P2 representation;
- OneCycle begins below the configured max LR, matching the previous GeoPT PoC behavior.

## Minimal LR Grid

Do not run a large grid. Use a validation-selected mini-grid.

Scratch:

- `3e-4`
- `1e-3`
- `3e-3`

Pretrained backbone:

- `1e-4`
- `3e-4`

Pretrained head:

- default `1e-3`
- only add `3e-4` if the head is unstable

Freeze:

- `0`
- `5`

Recommended first pass:

1. run Protocol C with `backbone_lr=3e-4`, `head_lr=1e-3`, `freeze=5`;
2. if unstable or worse than same-LR, run `backbone_lr=1e-4`, `head_lr=1e-3`, `freeze=5`;
3. if both are weak, do not proceed to P3; review R1b pretraining targets.

## Commands

Pretrained-protective M3 gate:

```bash
MODE=all \
RUN_PREFIX=m3_openfoam_p2_ft_tuned_oclr \
TRAIN_SIZES="10 25 50 100" \
SPLIT_SEEDS="42 43 44" \
EPOCHS=100 \
POINT_BUDGET=3072 \
EVAL_POINT_BUDGET=3072 \
EXPECTED_CASE_COUNT=45 \
SCRATCH_LR=1e-3 \
PRETRAINED_BACKBONE_LR=3e-4 \
PRETRAINED_HEAD_LR=1e-3 \
FREEZE_PRETRAINED_BACKBONE_EPOCHS=5 \
FINETUNE_SCHEDULER=onecycle \
FINETUNE_PCT_START=0.3 \
MAX_GRAD_NORM=1.0 \
bash scripts/run_m3_openfoam_p2_transfer_gate.sh
```

Conservative backup:

```bash
MODE=all \
RUN_PREFIX=m3_openfoam_p2_ft_tuned_oclr_blr1e4 \
TRAIN_SIZES="25 50 100" \
SPLIT_SEEDS="42 43 44" \
EPOCHS=100 \
POINT_BUDGET=3072 \
EVAL_POINT_BUDGET=3072 \
EXPECTED_CASE_COUNT=45 \
SCRATCH_LR=1e-3 \
PRETRAINED_BACKBONE_LR=1e-4 \
PRETRAINED_HEAD_LR=1e-3 \
FREEZE_PRETRAINED_BACKBONE_EPOCHS=5 \
FINETUNE_SCHEDULER=onecycle \
FINETUNE_PCT_START=0.3 \
MAX_GRAD_NORM=1.0 \
bash scripts/run_m3_openfoam_p2_transfer_gate.sh
```

GeoPT-faithful same-optimizer check:

```bash
MODE=all \
RUN_PREFIX=m3_openfoam_p2_geopt_faithful_oclr \
TRAIN_SIZES="25 50 100" \
SPLIT_SEEDS="42 43 44" \
EPOCHS=100 \
POINT_BUDGET=3072 \
EVAL_POINT_BUDGET=3072 \
EXPECTED_CASE_COUNT=45 \
SCRATCH_LR=1e-3 \
FINETUNE_SCHEDULER=onecycle \
FINETUNE_PCT_START=0.3 \
MAX_GRAD_NORM=1.0 \
bash scripts/run_m3_openfoam_p2_transfer_gate.sh
```

## Interpretation

Use Protocol A only as a diagnostic. If Protocol A is negative but Protocol C is positive, the result should be interpreted as:

> The P2 representation is useful but requires standard fine-tuning practice to avoid overwriting the pretrained backbone.

Use Protocol C as the practical gate for P2.

Go:

- Protocol C improves scratch by about 10% relative L2 at 25 or 50 labels;
- or reaches similar error with materially fewer labels;
- and does not worsen `Tmax` / hotspot metrics.

Conditional Go:

- Protocol C improves at 100 labels but not 25/50;
- or improvement is 5-10% but stable across split seeds;
- run R1b pretraining target ablations before P3.

No-Go for current P2 recipe:

- Protocol C and GeoPT-faithful OneCycle both fail against a tuned scratch baseline;
- head-only/staged-unfreeze do not show representation value;
- max-temperature or hotspot reliability consistently worsens.

This No-Go rejects the current P2 fine-tuning/pretraining recipe, not the broader Thermal GeoPT hypothesis.
