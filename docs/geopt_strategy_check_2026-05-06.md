# GeoPT Strategy Check 2026-05-06

## Conclusion

The completed Thermal GeoPT pilot validates the local data/training/evaluation pipeline, but it does not yet validate the core GeoPT hypothesis.

Continuing to tune only the current 300-shape pretraining run and 1200-case D1 proxy split is not the right main path. The next experiment should be redesigned around GeoPT's actual technical claim: large-scale dynamics-lifted self-supervision should improve data efficiency and transfer under limited downstream labels.

## GeoPT Features That Matter

GeoPT is not just "use Transolver with a checkpoint." The relevant technical features are:

- Dynamics-lifted geometric pretraining, not static geometry-only pretraining.
- Random dynamics conditions as prompts.
- Self-supervised trajectory targets, originally VDF trajectories.
- Large-scale STL-only pretraining: more than 10,000 geometries and over 1M samples in the original setting.
- Main downstream claim: 20-60% reduction in supervised training data and better scaling with model/data.

For Thermal GeoPT, the matching hypothesis is:

- Brownian/diffusion-lifted thermal prompts should encode boundary reachability, diffusion scale, source/sink influence, and thermal resistance.
- The benefit should appear most clearly when true thermal labels are scarce or downstream geometry is out-of-distribution.

## Why The Current Pilot Is Insufficient

Current pilot:

- Pretraining: 300 CadQuery shapes, 20 episodes/shape, 6,000 episodes, 2 epochs.
- Downstream: D1 proxy, 1,200 train cases, 150 validation, 150 test.
- Target: deterministic source/sink inverse-distance proxy, not FEM/FVM/OpenFOAM.

This is useful for plumbing, but weak for GeoPT evidence:

- 1,200 downstream labels is not label-scarce.
- The proxy target is simple and strongly encoded by the input conditions.
- Scratch can learn this proxy quickly, so pretraining has little room to help.
- The pretraining scale is below the planned first-result scale and far below GeoPT's intended regime.
- The current test does not include static-TDF, no-Brownian, or fluid-GeoPT transfer ablations.

## Revised Gate Before Large Pretraining

Before moving to P3/P4 large pretraining, run a smaller but GeoPT-correct gate:

| Item | Setting |
|---|---|
| pretraining lower bound | current 6k episodes |
| pretraining next candidate | 500-2,000 shapes, 5k-40k episodes |
| downstream train sizes | 10, 25, 50, 100 |
| validation/test | fixed, unchanged across all groups |
| groups | scratch, Thermal GeoPT full, static TDF-only, no-boundary-field/Brownian ablation |
| success threshold | Thermal GeoPT improves relative L2 by about 10% at 25 or 50 labels, or reaches the same error with materially fewer epochs |

Only if this gate is positive should we spend time/storage on P3 main pretraining.

## Immediate Action

The next implementation task is now tracked by `docs/geopt_gate_commands.md`: label-scarcity split generation plus an experiment matrix for:

- scratch vs pretrained
- train sizes 10/25/50/100
- fixed validation/test
- identical architecture and evaluation

This is a better test of GeoPT's original advantage than simply extending the current full-label pilot.
