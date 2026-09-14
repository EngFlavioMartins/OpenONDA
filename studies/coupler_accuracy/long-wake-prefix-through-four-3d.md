# Verified 3D wake comparison through physical time 4.0

The iterated coupler improves short-run drag, but the longer comparison exposes
growing velocity differences. Across the first 70 exchanges, from physical
time `0.5` to `4.0`, relative drag-history RMS error is **0.759%**. At time
`4.0`, the near-wake vector velocity RMS error is **2.846% of freestream speed
on the centreline** and **2.444% on the off-axis line**. These are verified
observations of a fully three-dimensional flow. They do not meet the requested
near-roundoff agreement.

The parent [longer comparison](long-wake-comparison-3d.md) remains in progress.
This report qualifies a fixed accepted prefix; it does not certify completion
of the parent run or a statistically developed wake. The best short-run
[paired result](panel-derivative-precision-3d.md) reduced relative drag-history
RMS error from `2.082%` to `0.555%` over physical time `0.5–1.5`. There is no
separate 70-exchange uniterated control, so the longer error must not be
presented as a measured improvement over such a control.

## Matched problem and independent verification

Both FVMs use the same native near-body cells, nominal spacing `h=0.0625`,
`dt=0.01` and laminar viscosity `ν=0.001`. The small domain contains 16,936
cells in a box approximately `[-1.5,1.5]^3` around the unit cube; the independent
full FVM contains 53,752 cells. VPM advances and exchanges every `0.05` using
the actual 3D particle solver. Auxiliary body-derivative queries use the
qualified double-precision evaluation; particle fields remain single precision.
After the common initial state, reference data never drive the hybrid.

The [prefix verifier](verify_accepted_wake_prefix_3d.py) captures live reports
without changing the running solver and checks the accepted saved states:

- All 70 intervals converge, using 210 logical interface sweeps. Every first
  endpoint map replays 24 recorded state/clock entries bitwise.
- The first 20 recorded comparison intervals exactly reproduce the completed
  short experiment. This does not assert equality of every intermediate field
  against a separate long control.
- Fifteen profile/checkpoint frames cover `0.5–4.0` at intervals of `0.25`.
  Their saved FVM velocities agree bitwise with the canonical checkpoints.
- Thirty independently reconstructed full/hybrid wall-force vectors reproduce
  the reported drag coefficients exactly. Recomputed fields, profiles and
  residuals pass 480 scalar checks with zero maximum recorded difference.
- All 799 original frozen source/input files remain unchanged. The report
  records hashes for 306 source and result artifacts.

The canonical [verification record](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/results/interface-precision-frozen-workspace-qualified/studies/coupler_accuracy/results/long-wake-prefix-through-four-qualified/long-wake-prefix-verification-3d.json)
contains the full history, force components, profile metrics and scope limits.

## Force and velocity agreement

At `t=4.0`, full-FVM drag coefficient is `0.9889538555`, versus hybrid
`0.9883303298`. Their instantaneous relative difference is only `−0.0631%`,
but the history RMS error is `0.7589%` and its maximum absolute relative
error is `1.4813%`. The endpoint is near a crossing of the force curves;
it cannot stand in for agreement over the trajectory.

The independently reconstructed transverse coefficient differences at this
endpoint are `ΔCy=0.000372188` and `ΔCz=0.000432395`. Relative percentages
would be misleading because the reference transverse forces are near zero.
The pressure contribution to `ΔCd` is `−0.00152545`, partly offset by a
viscous contribution of `+0.000901926`.

![Verified force histories and differences](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/results/interface-precision-frozen-workspace-qualified/studies/coupler_accuracy/results/long-wake-prefix-through-four-figures/forces.png)

The volume-weighted small-FVM velocity RMS error at `t=4.0` is `1.029% U∞`;
within the near-body region `max(|x|,|y|,|z|)<1`, it is `0.818% U∞`.
The line measurements below are vector RMS errors normalized by freestream
speed. They are observations of the 3D solution, not exterior volume norms.
The centreline has `y=z=0`; the off-axis line has `y=0.75, z=0`.

| Region at time 4.0 | Centreline | Off-axis |
| --- | ---: | ---: |
| VPM upstream, `x < −1.5` | 0.515% | 0.582% |
| VPM near wake, `1.5 < x ≤ 4` | 2.846% | 2.444% |
| VPM far wake, `4 < x ≤ 10` | 0.485% | 0.520% |
| Physical composite, matched FVM sampling | 1.982% | 1.322% |

The physical composite uses FVM inside the small box and VPM outside.
The full-reference and small-domain interpolation stencils can differ at
the cut boundary; both raw and matched-stencil comparisons are retained.
The centreline near-wake error grows from `1.102% U∞` at `t=1.5` to
`2.064% U∞` at `t=2.5` and `2.846% U∞` at `t=4.0`.

![Verified streamwise profiles at three times](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/results/interface-precision-frozen-workspace-qualified/studies/coupler_accuracy/results/long-wake-prefix-through-four-figures/profiles.png)

## What the direct evaluation rules out

Saved particle and panel sources are evaluated directly in double precision
at `t=0.5, 1.5, 2.5, 4.0`. At `t=4.0`, the direct result differs from the
runtime VPM query by `2.051e−6 U∞` vector RMS over all fluid profile points,
with maximum vector difference `1.936e−5 U∞`.

On the same centreline near-wake points, the runtime/reference RMS difference
is `0.028460653 U∞`, the direct/reference difference is `0.028460126 U∞`,
and direct/runtime difference is `1.227e−6 U∞`. Direct evaluation therefore
does not remove the observed percent-level wake discrepancy. Its particle
kernel is independently cross-checked at eight targets per snapshot, and
its analytical panel kernel at all profile targets, to roundoff.

This isolates the query approximation and arithmetic for the saved sources.
It does not validate how those particle strengths, positions or body-panel
strengths were generated, nor identify the remaining error's complete cause.
Converged interface residuals likewise do not establish agreement with FVM.

The next controlled comparison should separate VPM time integration from
particle-renewal frequency: the earlier cadence experiment changed both.
Its shorter interval improved drag while worsening velocity, so reducing
the combined step is not yet an evidence-backed general fix.

## Artifacts and post-processing precision

The [velocity-error figure](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/results/interface-precision-frozen-workspace-qualified/studies/coupler_accuracy/results/long-wake-prefix-through-four-figures/velocity-errors.png)
shows the volume, regional line and composite errors through this prefix.
The [figure manifest](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/results/interface-precision-frozen-workspace-qualified/studies/coupler_accuracy/results/long-wake-prefix-through-four-figures/wake-prefix-figures-3d.json)
links PNG and SVG exports and their hashes. All three figures were visually
inspected.

The canonical verifier uses a double-precision composite array so assigning
FVM samples preserves their precision. A superseded first report copied the
single-precision VPM array before assigning FVM samples; correcting that cast
changed composite metrics by at most `2.841e−8 U∞`. Separate FVM/VPM metrics,
force vectors, histories and direct checks were unchanged. The superseded
script and report remain intact for provenance; new analyses use schema `/2`
and `verify_accepted_wake_prefix_3d.py`.
