# Isolating particle RK time resolution in the 3D cube

Refining inviscid RK2 alone does not improve the matched cube comparison.
With FVM exchange, GBD diffusion/remapping and accepted renewal held at `0.05`,
five RK2 steps of `0.01` produce almost the same force and velocity errors as
one RK2 step of `0.05`. Both trajectories cover physical time `0.5–1.5`.
All 20 candidate intervals converge in three interface sweeps each.

Relative drag-history RMS error changes from `0.555068%` to `0.555463%`.
Final whole-small-FVM velocity RMS error increases by `0.0313%` relative to
its baseline value. This result does not reproduce the much larger response
to changing the complete [exchange cadence](interface-time-resolution-3d.md),
where final whole-FVM velocity error increased `13.49%`.

## What this comparison changes

The [scoped runner](cube_inviscid_subcycling_3d.py) retains the independent
53,752-cell reference and the small 16,936-cell FVM box approximately
`[-1.5,1.5]^3` around the unit cube. Shared native cells, nominal spacing
`h=0.0625`, FVM `dt=0.01`, laminar viscosity `ν=0.001`, initial state,
particle precision and the qualified auxiliary panel queries are unchanged.

The [RK wrapper](experimental_inviscid_subcycling_3d.py) composes the native
coupled position/vector-circulation integrator using each substep's physical
start time. The native `RK2` tableau is Heun: stages occur at each substep's
start and end, with equal final weights. The wrapper never advances the
canonical solver clock between these smaller inviscid updates.

| Operation over physical time 0.5–1.5 | Baseline | Candidate |
| --- | ---: | ---: |
| FVM steps, each 0.01 | 100 | 100 |
| Accepted coupling intervals, each 0.05 | 20 | 20 |
| Inviscid RK2 calls | 20 | 100 |
| RK stage evaluations | 40 | 200 |
| GBD diffusion/remapping calls, each 0.05 | 20 | 20 |
| Outer stabilization phase calls | 80 | 80 |
| Converged interface intervals | 20 | 20 |
| Logical interface sweeps | 60 | 60 |

The candidate records every actual RK call and stage time, every diffusion
call and each stabilization phase. Its outer clock and particle population
remain unchanged during the inviscid subcycles. No pending zero-time GBD
regeneration occurs in these runs. The interface iteration still reconstructs
each trial from a fixed accepted FVM start and one fixed advected particle
predictor; each interval's first map is independently replayed. Repeated
Picard evaluations are not additional physical renewal intervals.

The selected panel scope is `vpm_boundary_condition`. It includes the body
field in target queries but excludes its velocity/gradient from RK stages,
as documented in the [body-field audit](body-query-and-transport-3d.md).
This experiment preserves that behavior. It isolates inviscid RK time
discretization, not every temporal or spatial approximation in VPM.

## Qualification and independent comparison

The [component tests](../../tests/coupler/test_inviscid_subcycling_heun_3d.py)
use a time-dependent, divergence-free 3D strain with analytic position and
vector-strength solutions. Halving the substep reduces both errors by more
than a factor of `3.7`, consistent with second order. The tests also check
physical Heun stage times, unchanged core radii and bitwise equality between
the native single-precision RK call and the one-substep wrapper. Both tests
pass; the [component record](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/results/interface-precision-frozen-workspace-qualified/studies/coupler_accuracy/results/inviscid-subcycling-component-qualification.json)
retains the earlier failed test that incorrectly assumed midpoint stages.
Correcting that test required no numerical implementation change.

The one-substep wrapper then advances the real cube for three intervals with
the original partitioned algorithm. The [wrapper verification](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/results/interface-precision-frozen-workspace-qualified/studies/coupler_accuracy/results/inviscid-subcycling-wrapper-verification-3d.json)
finds bitwise identical histories, six final comparison arrays, 17 FVM
checkpoint entries, 11 boundary-history entries and 11 numeric VPM datasets.
All 48 profile arrays and 136 entries across eight accepted full/hybrid FVM
checkpoints also agree exactly with the existing unwrapped control.

For the 20-interval candidate, the
[comparison verifier](compare_inviscid_subcycling_3d.py) independently checks:

- Identical initial source fields, profile geometry and full-reference fields
  at every saved profile time; identical reference drag at all 20 endpoints.
- All 20 first-map replays, their 24 recorded state/clock entries, interface
  residuals and accepted FVM/renewal counters.
- Both wall forces at all five checkpoint times, and profiles reconstructed
  from canonical cell velocities using stored affine stencils.
- 140 scalar checks with zero maximum difference, including ten independent
  wall-force reconstructions.
- All 799 original frozen source/input files unchanged; 472 source/artifact
  records identify the comparison's inputs and verification dependencies.

Direct evaluation of the final saved particles/panels differs from the runtime
profile query by `7.474e−7 U∞` vector RMS, with maximum `6.731e−6 U∞`.
The particle kernel cross-check differs by at most `1.665e−16` and the panel
kernel cross-check by `3.469e−17`. This checks evaluation of the saved sources,
not the accuracy of their representation or evolution.

## Measured effect

Velocity entries below are vector RMS errors divided by `U∞`, not percentages.
Near-wake lines cover `1.5 < x ≤ 4`, at `z=0`. The off-axis line has `y=0.75`.

| Measurement | RK dt=0.05 | RK dt=0.01 |
| --- | ---: | ---: |
| Relative drag-history RMS error (%) | 0.555068 | 0.555463 |
| Final hybrid drag coefficient | 1.08856073 | 1.08858715 |
| Final whole-small-FVM velocity error | 0.007663745 | 0.007666145 |
| Final near-body FVM velocity error | 0.001954932 | 0.001955938 |
| Final VPM sampled error inside FVM | 0.031242600 | 0.031243398 |
| Final centreline near-wake velocity error | 0.011018528 | 0.011028573 |
| Final off-axis near-wake velocity error | 0.007721032 | 0.007727320 |

The final independent reference drag coefficient is `1.080395340` in both
cases. The candidate's whole-FVM and near-body velocity RMS errors increase
only `0.0313%` and `0.0515%` relative to the existing errors. Its final particle
population is 28,446, versus 28,441 in the baseline; the evolving discrete
clouds are close, but are not being required to remain identical.

![Fixed-cadence RK comparison](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/results/interface-precision-frozen-workspace-qualified/studies/coupler_accuracy/results/inviscid-subcycling-five-twenty-comparison/comparison.png)

The figure was visually inspected. The two hybrid curves nearly overlap;
the unchanged hybrid/reference discrepancy remains visible. The physical
composite uses FVM inside its box and VPM outside. The numerical profile
metrics separately use matched FVM sampling stencils.

## Consequence for the next experiment

The larger cadence effect is not reproduced by refining inviscid RK alone
at the original exchange interval. This comparison does not mathematically
decompose interacting errors or establish that RK error is negligible at
every later time or configuration.

The [next temporal comparison is now complete](split-time-isolation-3d.md):
advection/stretching and GBD diffusion/remapping both use `0.01`, with FVM
exchange and outer stabilization still at `0.05`. It increases drag-history
RMS error to 0.583542% and whole-FVM velocity error by 1.079% relative to
the original schedule. Its contrast with this RK-only run changes GBD and
splitting together. Comparing it with `0.01` exchange leaves additional
exchange/renewal, boundary-update and outer stabilization effects; they are
not an additive error decomposition. Body transport is being tested separately.

The [complete result record](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/results/interface-precision-frozen-workspace-qualified/studies/coupler_accuracy/results/inviscid-subcycling-five-twenty-comparison/inviscid-subcycling-comparison-3d.json)
retains both unshifted histories, profile metrics, force components, direct
checks and PNG/SVG exports. The requested developed-wake force/profile
agreement remains unmet; this negative experiment narrows the diagnosis
without promoting a new production option.
