# Pretraining Strategy Review 2026-05-07

## Conclusion

Current P2 dynamics-lifted pretraining should be evaluated through the running M3 OpenFOAM D1 transfer gate, but it should not be scaled to P3 as-is.

The current checkpoint is a credible first GeoPT-aligned candidate because it uses 2,000 CAD shapes, 40,000 episodes, D1-aligned thermal prompts, and a Transolver backbone. It is much closer to the original research goal than the earlier proxy/static-TDF line.

However, the current Brownian displacement regression target is weak as a faithful GeoPT analogue. Original GeoPT predicts geometry feature trajectories generated deterministically from an input dynamics field. The current Thermal GeoPT run asks the model to regress individual Brownian delta samples without giving the random increments or seed as input. This makes the trajectory component partly irreducible, and the P2 loss history confirms the issue: `loss_tdf` improves, while `loss_trajectory` is almost flat from epoch 1 to epoch 20.

Therefore:

- Continue the current M3 gate, because the P2 checkpoint already exists and the only honest transfer test is solver-backed downstream evaluation.
- Do not start P3/P4 from the same recipe unless M3 is clearly positive.
- Prepare R1b now as the likely next pretraining revision, especially if M3 is weak, seed-dependent, or worsens `Tmax`.

## What Original GeoPT Requires

The GeoPT paper's core claim is not generic checkpoint initialization. It is large-scale dynamics-lifted geometric pretraining:

- static geometry-only pretraining can cause negative transfer;
- adding synthetic dynamics lifts the task from geometry space into a geometry-dynamics space;
- the same kind of dynamics condition is later reconfigured as the downstream physics prompt;
- the benefit is judged by data efficiency, convergence, and stronger performance than scratch and geometry-only baselines.

For Thermal GeoPT, the corresponding requirement is:

- pretraining must encode heat-relevant geometry-boundary interaction, not just boundary distance;
- downstream must be solver-backed heat transfer, not a synthetic proxy;
- the comparison must include scratch, static geometry/TDF, dynamics-lifted, and preferably fluid-GeoPT transfer;
- the claim must be label efficiency in D1/D2 thermal tasks.

## Current P2 Assessment

Strengths:

- Uses P2 scale: 2,000 shapes, 20 episodes per shape, 40,000 episodes.
- Uses thermal-like shape families including plate fins, pin fins, louver fins, channel blocks, brackets, airfoil extrusions, and annular casings.
- Uses D1-aligned condition channels: `conductivity`, `source_temperature`, `sink_temperature`, `source_patch`, `sink_patch`, `nearest_boundary_distance`.
- Trains the actual downstream backbone, not a detached geometry encoder.
- Connects to M3 solver-backed OpenFOAM D1 evaluation.

Weaknesses:

- Individual Brownian displacement is stochastic but the stochastic increment is not part of the input.
- The trajectory loss is effectively not learning; this suggests the model can only learn a mean or boundary bias, not the sampled trajectory.
- Boundary hit is useful but too generic unless source, sink, and boundary condition type are distinguished.
- TDF still dominates the learnable signal; this risks collapsing back toward static geometry pretraining.
- The current M3 block conduction data is solver-backed, but still a block-geometry gate rather than the final heat-sink benchmark.

## Expert Consensus

GeoPT advisor:

- Current P2 is directionally GeoPT-like, but Brownian displacement regression is not faithful to original GeoPT because the path is not determined by an input dynamics condition.
- M3 should continue as the first solver-backed transfer test.
- If M3 is weak, immediately move to R1b rather than scaling P3.

Heat-transfer advisor:

- Brownian motion is a valid conceptual bridge to diffusion, but heat conduction transfer needs source/sink reachability, hitting probability, expected hitting time, Robin/Neumann/Dirichlet prompt alignment, and thermal resistance-like structure.
- Static TDF and nearest-boundary signals are insufficient for steady conduction because temperature depends on global source-to-sink paths, bottlenecks, cooling area, and boundary conditions.

PM / experiment design:

- Do not interrupt the current M3 run.
- Treat M3 as the formal gate for the existing P2 checkpoint.
- Make the next decision from paired scratch vs P2-pretrained OpenFOAM D1 results, not proxy results.
- Keep the project centered on whether large dynamics-lifted pretraining improves label efficiency.

## M3 Decision Rule

Go:

- `dynamics_lifted` beats scratch by about 10% relative L2 at 25 or 50 labels, or reaches similar error with materially fewer labels.
- `Tmax` error and hotspot metrics do not degrade.
- The win is stable across split seeds.

Conditional Go:

- Improvement appears mainly at 100 labels, or mean gain is 5-10%.
- Case win rate is high but average gain is modest.
- Proceed with R1b ablation before P3.

No-Go for current recipe:

- Solver-backed D1 is equal to or worse than scratch.
- Gains exist only in proxy tasks.
- Trajectory-based pretraining does not outperform static TDF/no-boundary baselines.
- `Tmax` or hotspot reliability gets worse.

No-Go here means the current P2 recipe is rejected, not the Thermal GeoPT hypothesis.

## R1b Revision

R1b should keep the GeoPT idea, but replace individual Brownian displacement regression with deterministic or expectation-style heat-aware targets.

Minimum R1b target set:

- static VDF/TDF as a low-weight auxiliary target;
- boundary hit probability;
- normalized expected survival or hitting time;
- source reachability;
- sink reachability;
- source-vs-sink absorption class;
- heat-kernel source/sink proximity at multiple diffusion scales;
- optional synthetic influence such as `E[q_hit exp(-lambda tau_hit)]`.

Prompt alignment:

- `k`, `alpha`;
- source patch or source distance;
- sink/cooling patch or sink distance;
- source temperature or heat flux;
- cooling coefficient `h`;
- ambient temperature `T_inf`;
- boundary type: source, sink, insulated, Robin-like.

Loss strategy:

- downweight static TDF so it does not dominate;
- normalize target losses by scale or variance;
- use BCE or calibrated classification losses for hit/reachability targets;
- add a pretraining validation split and early stopping, since P2 overtrained after the best epoch.

## Near-Term Plan

1. Let the current M3 OpenFOAM P2 transfer gate finish.
2. Summarize paired scratch vs `dynamics_lifted` results by train size, split seed, relative L2, `Tmax`, hotspot distance, and case win rate.
3. If M3 is clearly positive, add static TDF/no-boundary/fluid-GeoPT baselines before any P3 scale-up.
4. If M3 is weak or negative, implement R1b before running more large pretraining.
5. Do not claim Thermal GeoPT effectiveness until solver-backed D1 shows label-efficiency improvement.

The user's concern is correct: if Thermal GeoPT does not test large-scale dynamics-lifted pretraining on solver-backed heat-transfer labels, it is not a meaningful GeoPT transfer experiment. The current P2/M3 path is acceptable as a gate, but the current Brownian target is not strong enough to be the final pretraining design without further evidence.
