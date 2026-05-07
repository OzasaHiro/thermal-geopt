# M3 Transfer Result Review 2026-05-07

## Conclusion

Both completed M3 protocols are negative for the current P2 `dynamics_lifted` checkpoint.

This is a valid negative result for the current P2 transfer recipe, but it is not a clean rejection of the Thermal GeoPT hypothesis. The runs reveal two larger problems:

1. the solver-backed block D1 downstream task is too simple for a strong GeoPT label-efficiency test;
2. the pretraining and downstream interfaces are not aligned enough to satisfy the original GeoPT idea.

Do not scale the current P2 recipe to P3.

## Protocols Checked

### Protocol A: same-LR constant 50 epochs

Prefix: `m3_openfoam_p2`

- scratch: AdamW, constant `lr=1e-3`, 50 epochs
- pretrained: AdamW, constant `lr=1e-3`, 50 epochs
- train sizes: 10 / 25 / 50 / 100
- split seeds: 42 / 43 / 44

### Protocol C: pretrained-protective OneCycle 100 epochs

Prefix: `m3_openfoam_p2_ft_tuned_oclr`

- scratch: all params `lr=1e-3`
- pretrained backbone: `lr=3e-4`
- pretrained new head: `lr=1e-3`
- freeze loaded backbone: 5 epochs
- scheduler: OneCycleLR, `pct_start=0.3`
- max grad norm: `1.0`
- train sizes: 10 / 25 / 50 / 100
- split seeds: 42 / 43 / 44

The pretrained load itself worked:

- loaded tensors: 167
- skipped/missing tensors: 2
- skipped tensors: `blocks.7.mlp2.weight`, `blocks.7.mlp2.bias`

This means the run is not accidentally scratch; the backbone was loaded.

## Results

### Same-LR Constant

| train cases | scratch relL2 | P2 dynamics relL2 | relative change | case win rate | Tmax change |
|---:|---:|---:|---:|---:|---:|
| 10 | 0.015377 | 0.066031 | -370.43% | 0.007 | -419.14% |
| 25 | 0.005354 | 0.058482 | -1071.33% | 0.000 | -1080.43% |
| 50 | 0.003674 | 0.054403 | -1463.19% | 0.000 | -1036.99% |
| 100 | 0.002384 | 0.040847 | -1702.91% | 0.000 | -1701.33% |

### Tuned OneCycle

| train cases | scratch relL2 | P2 dynamics relL2 | relative change | case win rate | Tmax change |
|---:|---:|---:|---:|---:|---:|
| 10 | 0.006982 | 0.066496 | -966.27% | 0.000 | -920.90% |
| 25 | 0.001366 | 0.052694 | -3766.11% | 0.000 | -2366.81% |
| 50 | 0.000602 | 0.044184 | -7249.42% | 0.000 | -8590.69% |
| 100 | 0.000314 | 0.034151 | -10799.67% | 0.000 | -11530.08% |

The tuned protocol improves scratch dramatically. It does not rescue the pretrained checkpoint.

For a representative tuned run, `split42 train50`:

- scratch best validation relL2: 0.000708 at epoch 94
- P2 dynamics best validation relL2: 0.037509 at epoch 73
- P2 dynamics train MSE improved from about 1.00 to 0.14, so optimization is not frozen, but it remains far behind scratch.

## Machine Learning Interpretation

The tuned fine-tuning protocol is technically reasonable:

- discriminative LR was applied correctly;
- the new head used a higher LR;
- OneCycleLR and gradient clipping ran;
- the checkpoint load was structurally correct.

Therefore the negative result is not just caused by using the same LR as scratch.

However, there is a critical transfer-interface problem. P2 pretraining used raw condition channels, while downstream fine-tuning normalizes condition channels using train-set statistics before passing them into the loaded model. This changes the input distribution seen by the loaded input projection and backbone.

Observed distributions:

- pretrain points: roughly `[-0.62, 0.62]`
- OpenFOAM D1 points: roughly `[0.00, 0.12]` in `x/y` and `[0.00, 0.03]` in `z`
- pretrain conditions are raw: conductivity, source temperature, sink temperature, patch proximity, distance
- downstream conditions are normalized during fine-tuning

This violates a key transfer assumption: the loaded model should see the same type and scale of prompt/interface that it saw during pretraining.

## GeoPT Interpretation

The original GeoPT idea is not just "load a pretrained model." It requires:

- a shared geometry-dynamics input interface between pretraining and downstream;
- dynamics prompts that can be reconfigured into downstream physics conditions;
- feature trajectories that encode geometry-dynamics coupling.

Current M3 violates this in several ways:

1. Input scaling mismatch: pretrained raw prompts vs downstream normalized prompts.
2. Coordinate scaling mismatch: normalized CAD-like pretraining points vs small physical OpenFOAM block coordinates.
3. Source/sink semantics mismatch: pretraining uses random local source/sink centers; M3 block D1 uses bottom/top Dirichlet faces.
4. Dynamics-target weakness: P2 trajectory loss was almost flat during pretraining, indicating that individual Brownian displacement regression was not a strong learnable signal.

So the current result says:

> The current P2 checkpoint does not transfer to the current OpenFOAM block D1 task under either aggressive or tuned fine-tuning.

It does not yet say:

> GeoPT-style dynamics-lifted pretraining is ineffective for heat-transfer surrogates.

## Heat-Transfer Interpretation

M3 block D1 is solver-backed, but physically too simple:

- rectangular solid block;
- fixed source temperature on one face;
- fixed sink temperature on the opposite face;
- insulated sides;
- mostly smooth steady conduction along the thickness direction.

Scratch learns this mapping extremely quickly. In the tuned protocol, scratch reaches:

- 25 labels: relL2 about 0.0014
- 50 labels: relL2 about 0.0006
- 100 labels: relL2 about 0.0003

This is effectively saturated. A GeoPT-style geometry prior has little room to help because geometry diversity and boundary-condition diversity are too low.

Thermal GeoPT should be judged on heat-sink-like geometries, source/sink patch variation, material/boundary variation, and hotspot-sensitive metrics. A simple block conduction gate is useful for pipeline validation, but it is not a strong paper-level downstream task.

## Decision

- Current P2 -> current M3 block D1: No-Go.
- Scale current P2 recipe to P3: No-Go.
- Reject Thermal GeoPT hypothesis: Hold.
- Continue current block D1 as main benchmark: No-Go.

## Required Corrections Before The Next Transfer Claim

1. Align input normalization.
   - Either fine-tune with the same raw prompt convention used by pretraining, or pretrain with the exact normalization used downstream.
   - Record pretraining input statistics and reuse them explicitly.

2. Align coordinate normalization.
   - Pretraining and downstream should use the same coordinate scale convention.
   - Current OpenFOAM points are much smaller than P2 CAD points.

3. Align source/sink semantics.
   - If downstream uses source/sink faces, pretraining should include face/patch-style source/sink prompts.
   - Random point-center proximity is not enough for the block D1 task.

4. Replace block-only D1 with a real heat-sink D1 gate.
   - plate-fin / pin-fin / channel block geometries;
   - source and cooling patch variation;
   - nontrivial hotspot formation;
   - scratch should not saturate at 25-50 labels.

5. Move from current R1 to R1b before scaling.
   - Downweight or remove individual Brownian displacement regression.
   - Use source/sink reachability, hit probability, expected hitting time, and heat-kernel influence targets.

## Next Recommended Experiment

Do not run another large P2/P3 pretraining yet.

First implement an interface-aligned diagnostic:

- no downstream condition normalization for pretrained-compatible runs, or explicit pretraining-stat normalization;
- coordinate normalization consistent with pretraining;
- same split and train sizes;
- small train sizes 10 / 25 / 50 only;
- compare scratch vs pretrained with the same tuned protocol.

If the aligned diagnostic is still strongly negative, prioritize R1b and a heat-sink D1 downstream task before any further scale-up.
