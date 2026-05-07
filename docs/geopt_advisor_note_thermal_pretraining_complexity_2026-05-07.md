# GeoPT Advisor Note: Thermal Pretraining Complexity

## Short Answer

The current Thermal GeoPT pretraining is more complex and harder than the original GeoPT pretext. That complexity is not automatically justified. It is defensible as a first thermal analogue, but only if we treat P2 as an empirical gate and simplify quickly if transfer is weak.

## Why It Is Harder Than GeoPT

Original GeoPT lifts geometry using dynamics that are part of the synthetic condition. The model is asked to learn geometry-conditioned trajectory structure, not to infer hidden random noise.

The current Thermal GeoPT variant includes Brownian random walk displacement targets. If the actual Brownian noise realization is not provided to the model, then predicting the exact displacement is partly irreducible. The model can learn conditional moments or boundary statistics, but it cannot deterministically recover a sampled random walk path.

This explains the P2 history:

- TDF improved until the best epoch.
- boundary/hit-step improved slightly.
- trajectory loss stayed almost flat.

That is a warning about target design, not necessarily a failure of the GeoPT idea.

## What Is Actually Needed For A GeoPT-Faithful Thermal Test

The essential GeoPT idea is not "predict a complicated stochastic simulator." It is:

1. use lots of unlabeled geometry,
2. lift geometry into a downstream-relevant dynamics space,
3. learn a transferable representation before supervised solver labels.

For heat transfer, the minimal faithful lift is likely:

- diffusion length/time features,
- source/sink reachability,
- boundary hit probability or survival time,
- heat-kernel or resistance-like proximity,
- thermal prompt alignment with downstream conditions.

Predicting individual Brownian displacement is optional and may be the least useful part unless the stochastic condition is represented in the input.

## Recommendation

Proceed with the current P2 checkpoint for the M3 transfer gate because it is already trained and uses the correct D1 thermal prompt schema.

If M3 is weak:

1. do not move to P3,
2. lower or remove the individual trajectory displacement loss,
3. replace it with expectation-style targets such as hit probability, survival time, source/sink influence, or heat-kernel fields,
4. rerun P2 as R1b before any large-scale claim.

If M3 is positive:

Keep the current P2 checkpoint as evidence that the dynamics-lifted signal is useful, but still run an ablation to show whether Brownian displacement itself is necessary.
