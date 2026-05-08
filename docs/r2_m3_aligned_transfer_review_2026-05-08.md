# R2 M3 Aligned Transfer Review 2026-05-08

## Verdict

The R2 aligned protocol is technically much cleaner than the previous M3 transfer run, but it is still negative for the current Thermal GeoPT pretraining target.

This is a valid No-Go for the current R2 target recipe on the block-D1 gate. It is not a rejection of the broader Thermal GeoPT hypothesis, because the R2 pretext still asks the model to regress individual Brownian displacement samples without providing the Brownian randomness as input.

## Protocol Integrity

R2 pretraining used the intended fixes:

- checkpoint selection metric: `val_loss`
- normalization mode: `standardize`
- pretraining split: shard holdout
- train episodes: `38000`
- validation episodes: `2000`
- validation shards: `100`

The M3 transfer runs also used the intended aligned input contract:

- `normalization_protocol`: `pretrained`
- downstream normalization mode: `pretrained_input_downstream_target`
- source config: `outputs/checkpoints/pretrain_r2_d1_thermal_dynamics_p2_norm_val_ep20/config.json`
- pretrained load: `167` tensors loaded, only final head tensors skipped/missing
- scheduler: `onecycle`
- pretrained backbone LR: `3e-4`
- new head LR: `1e-3`
- freeze loaded backbone: `5` epochs

So the result is not caused by accidentally running scratch, failing to load the checkpoint, or reusing the old downstream-only normalization.

## R2 Pretraining Behavior

Best checkpoint:

- best epoch: `16`
- best validation loss: `2.877079`
- final validation loss: `2.881154`

Loss components show the key issue:

| component | epoch 1 val | best val | epoch 20 val | interpretation |
|---|---:|---:|---:|---|
| TDF | 0.244481 | 0.158205 | 0.159376 | learns clearly |
| trajectory | 1.004386 | 1.004333 | 1.004346 | essentially flat |
| boundary hit | 0.852752 | 0.850116 | 0.851602 | nearly flat |
| hit step | 0.866872 | 0.864425 | 0.865829 | nearly flat |

R2 mostly learns static/TDF-like features. The dynamics-lifted parts do not show meaningful learning.

This supports the earlier concern: individual Brownian displacement regression is not a GeoPT-faithful thermal pretext when the Brownian random increments are not part of the input.

## M3 Transfer Results

Aggregated across split seeds `42/43/44`, train seed `42`.

| train cases | scratch relL2 | R2 relL2 | relL2 change | scratch centered relL2 | R2 centered relL2 | scratch nRMSE range | R2 nRMSE range | case win rate |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 0.007597 | 0.030147 | -296.85% | 0.091014 | 0.363471 | 0.028562 | 0.114066 | 0.007 |
| 25 | 0.001494 | 0.012497 | -736.28% | 0.017846 | 0.152658 | 0.005600 | 0.047908 | 0.000 |
| 50 | 0.000666 | 0.004589 | -589.14% | 0.007808 | 0.055966 | 0.002450 | 0.017563 | 0.000 |
| 100 | 0.000443 | 0.002221 | -401.37% | 0.005190 | 0.026474 | 0.001629 | 0.008308 | 0.000 |

Thermal design metrics also degrade:

| train cases | scratch Tmax abs err | R2 Tmax abs err | scratch hotspot abs err | R2 hotspot abs err |
|---:|---:|---:|---:|---:|
| 10 | 3.523 | 14.375 | 3.523 | 14.672 |
| 25 | 0.620 | 5.873 | 0.629 | 6.064 |
| 50 | 0.194 | 2.088 | 0.207 | 2.058 |
| 100 | 0.105 | 0.881 | 0.119 | 0.867 |

## Comparison To Previous M3

The R2 protocol substantially improves the pretrained downstream result relative to the previous R1/P2 tuned run:

| train cases | old tuned R1/P2 pretrained relL2 | R2 aligned pretrained relL2 |
|---:|---:|---:|
| 10 | 0.066496 | 0.030147 |
| 25 | 0.052694 | 0.012497 |
| 50 | 0.044184 | 0.004589 |
| 100 | 0.034151 | 0.002221 |

This means the normalization and validation fixes mattered. However, scratch is still much stronger:

| train cases | R2 scratch relL2 | R2 pretrained relL2 | pretrained/scratch |
|---:|---:|---:|---:|
| 10 | 0.007597 | 0.030147 | 4.0x worse |
| 25 | 0.001494 | 0.012497 | 8.4x worse |
| 50 | 0.000666 | 0.004589 | 6.9x worse |
| 100 | 0.000443 | 0.002221 | 5.0x worse |

## Interpretation

The aligned interface removes one major confounder from the earlier negative result. The remaining negative signal is therefore stronger evidence that the current R2 pretext is not useful for the M3 block-D1 downstream task.

The likely reasons are:

1. The dynamics target is weak or partly unlearnable.
   - TDF improves, but trajectory/hit losses are nearly flat.
   - Brownian displacement samples are random realizations, while the random increments are not inputs.

2. The downstream block-D1 task is too easy.
   - Scratch reaches about `0.00044` relative L2 and about `0.16 K` RMSE at 100 labels.
   - This leaves little room for a general geometric prior to help.

3. The R2 pretraining target is still not heat-sink surrogate aligned.
   - It does not directly teach source/sink reachability, expected hitting time, source-vs-sink absorption, or thermal resistance-like influence.

## Decision

- R2 current pretext -> M3 block-D1: No-Go.
- Scale current R2 recipe to P3: No-Go.
- Treat old normalization mismatch as resolved enough for diagnosis: Yes.
- Reject Thermal GeoPT broadly: No.

## Next Step

Move to R2b/R3 before more large runs:

1. Remove or downweight individual Brownian displacement regression.
2. Add deterministic/statistical heat-aware lifted targets:
   - source hit probability;
   - sink hit probability;
   - source-vs-sink absorption probability;
   - expected hitting time;
   - survival probability at several diffusion times;
   - heat-kernel/source influence;
   - resistance-like source-sink influence.
3. Keep the R2 normalization/validation protocol.
4. Replace block-D1 as the main gate with a nontrivial solver-backed heat-sink D1 benchmark.

