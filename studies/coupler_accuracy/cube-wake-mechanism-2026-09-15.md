# Cause isolation for the remaining cube wake discrepancy

The investigation establishes a numerical evolution inconsistency in the VPM:
the particle field contains vorticity that produces no velocity, but the
positive-transpose stretching law can convert that component into a velocity
disturbance. This mechanism is present in the saved cube state and can be
reproduced without a body, recirculation, FVM boundary condition, LES model,
tree approximation, or time integrator. It is a stronger finding than attributing
the error to the location of a contour plot's maximum.

It does **not** establish that this mechanism accounts for every part of the
coupled/reference discrepancy. In particular, the complete feedback loop,
different viscous closures, and physical wake instability have not been
quantitatively separated over a fresh full trajectory. No production solver or
tutorial setting was changed in this investigation.

## Inputs and comparisons

The input is the completed, centred-lattice cube trial described in
[the preceding study](cube-wake-drift-2026-09-15.md), with `h=0.06`, Gaussian
radius `0.066`, `dt=0.01`, transposed stretching, GBD, and the original fine
reference. This trial did not enable alignment. Its saved source tree is
identified by its `trial.json`; the historical reader is used for the historical
10.0 backups, without adding compatibility code to the current solver.

All evolution and source fields are fully three dimensional. The original
`z=0` plotting points are used for localization. Operator measurements also
use a 392-point 3D grid closed under reflection in `z`. Its downstream region
contains 196 points with `1.25 <= x/D <= 1.62`, distinct from the plot's
`1.25 <= x/D <= 1.75` strip. Velocity errors are normalized by `U_infinity=1`,
never by a near-zero local reference velocity.

## What grows, and where

At `t=8`, the worst plotted point is approximately `(1.44, -0.18, 0) D`.
Its VPM out-of-plane velocity is `0.35793 U_infinity`, while the reference
value is `0.0000831 U_infinity`. The direct 3D source decomposition gives
`0.32355 U_infinity` from particles with `1.25 <= x_p/D <= 1.65`, versus
`0.00151 U_infinity` from the near-body population with
`max(abs(x_p))/D < 0.75`.

Particles are already `0.01 D` from the cube at `t=1`, when the out-of-plane
RMS in the plotted downstream strip is only `0.000038 U_infinity`; it reaches
`0.10436` at `t=8`. Thus the first arrival near the physical body is not the
observed onset. The much stronger association is with particles entering the
VPM-owned downstream region and the recirculation reaching that region.
Association alone does not prove that recirculation is necessary.

The reference is not assumed to remain symmetric forever. Its 3D reflection
asymmetry in the same downstream probe region reaches `0.00167 U_infinity`
by `t=30`. The corrected VPM already has `0.11723` at `t=8`. Different
perturbation levels or physical instability can affect phase and onset; this
comparison by itself would not prove a numerical defect.

## The decisive control

Let `omega_G` be the Gaussian particle vorticity and `q=curl(u)` its actual
velocity curl. The two are not identical. Biot-Savart removes the longitudinal
(gradient) part of `omega_G`. On the saved 3D downstream probes,
`RMS(omega_G-q)/RMS(omega_G)` is `21.4%` at `t=6` and `15.3%` at `t=8`.
These measurements use an independent double-precision Gaussian sum and its
analytical velocity derivative.

A controlled gradient field `ell=grad(psi)` was added mathematically, where
`psi` is a 3D Gaussian centred at `(1.5,0,0)`, width `0.12 D`, and maximum
unfiltered added vorticity `1 U_infinity/D`. It is truncated at seven Gaussian
standard deviations, entirely outside the cube. Refining its quadrature
spacing leaves the existing simulation mesh, particle cores, and flow unchanged.

| Measurement | Quadrature spacing 0.06 D | Quadrature spacing 0.03 D |
| --- | ---: | ---: |
| Initial induced velocity RMS, U_infinity | 7.14e-8 | 4.44e-15 |
| Velocity-rate RMS in saved cube strain, U_infinity²/D | 0.048292 | 0.048290 |
| Velocity-rate RMS in exact manufactured strain, U_infinity²/D | 0.0101812 | 0.0101810 |
| Manufactured gradient-preserving control rate RMS | 7.36e-6 | 5.42e-14 |

The manufactured background is exactly
`u=(1,0,0)+diag(0.5,-0.25,-0.25)(x-(1.5,0,0))`. It is incompressible,
irrotational, and has no closed recirculation or body. Streamwise velocity
is positive at every perturbation source. No FVM, diffusion, LES, treecode,
or RK algorithm enters this control.

With `J_ij=du_i/dx_j`, the positive-transpose particle law transports this
gradient component with

```text
F_plus(ell) = -(u . grad) ell + J.T ell
            = -grad(u . ell) + 2 J.T ell.
```

The first term has zero Biot-Savart velocity; the second generally does not.
Negative-transpose **covector transport of this gradient-only perturbation**
gives `F_minus(ell)=-grad(u . ell)`, and its velocity rate converges to zero.
This is a diagnostic control, not a proposal to reverse the sign of physical
vortex stretching. The algebraic rate identity closes below `7e-16`; an
independent time finite difference validates the rate kernel to relative
error `5.36e-11`.

The nonzero rate therefore survives quadrature refinement while the initial
velocity and the negative control approach roundoff. This demonstrates an
unphysical degree of freedom becoming dynamically active, rather than an
ordinary velocity perturbation undergoing natural wake instability. The
underlying vorticity-consistency problem and alignment treatments are discussed
by [Pedrizzetti (1992)](https://doi.org/10.1016/0169-5983(92)90011-K) and
[Winckelmans and Leonard (1993)](https://www.sciencedirect.com/science/article/pii/S0021999183712167).

## What the other controls establish

| Suspected cause | Evidence and scope |
| --- | --- |
| Missing body contribution at RK stages | Both actual temporary stages include it. Stage velocity errors against independent direct induction are below 0.001 U_infinity on the selected downstream particles. |
| Stale second-stage state or large local RK error | The second stage uses changed positions and strengths. One full inviscid step versus two half steps differs by about 1e-6 U_infinity RMS. This is a local check, not a bound on accumulated trajectory error. |
| Tree approximation as the primary instantaneous error | Selected actual stage stretching errors are below 0.21%; the manufactured reproduction uses no tree at all. Approximation error can still seed disturbances. |
| Diffusion directly driving the observed asymmetric increment | Native molecular and LES GBD increments reduce reflection asymmetry at both checked times. Different FVM/VPM viscous closures remain a separate accuracy concern. |
| Rapid amplification from repeated transfer alone | Twenty renewals with the actual FVM donor state frozen reduce downstream reflection asymmetry from 0.11723 to 0.11504. Removing pruning gives 0.11525. Neither repeats the rapid growth. |
| Colliding particles or first arrival near the cube | No near-duplicate particles occur in the checked regions; near-wall particles are present well before the large discrepancy. These snapshot checks do not certify every temporary trajectory. |
| FVM mixed-boundary feedback being necessary | Growth persists in a 100-step VPM-only control driven by prescribed reference donors, with no FVM solve or boundary iteration. The manufactured control removes boundaries entirely. |
| Recirculation or a physical surface being necessary for this mechanism | The exact forward-flow strain control reproduces it without either. Recirculation can still expose and amplify the problem in the cube. |

The operator split also isolates the amplification: at `t=6`, downstream
reflection asymmetry rises from `0.05891` to `0.05976` in one native RK step;
the subsequent GBD operation reduces it to `0.05957`. The stretching-only
contribution is dominant in this asymmetric increment. At `t=8`, stretching
again increases the asymmetry while advection and diffusion oppose it.

Transfer still matters. Its represented-field blend has the form
`omega_mix=(1-eta) omega_VPM + eta omega_FVM`. Even with individually
solenoidal inputs, its divergence contains
`grad(eta) . (omega_FVM-omega_VPM)`. Gaussian representation, body masking,
and pruning add further discrete consistency effects. In the frozen test,
renewal slightly reduces velocity asymmetry while increasing the vorticity
divergence diagnostic. Thus a small instantaneous velocity change or good
global circulation conservation does not establish a clean vorticity state.
This identifies a way to supply the problematic component; it does not
assign a percentage of the developed-wake error to each supplying operation.

## The existing alignment setup is not an accuracy qualification

Two controls start from the identical saved `t=6` particle state and receive
the same prescribed reference-donor history through `t=7`. The second enables
the tutorial's existing alignment rate of `10/s` with moment preservation.
Initial probe fields are bitwise identical. Both complete all 100 steps with
the original health gates.

| Final 3D probe error, fraction of U_infinity | Without alignment | With alignment |
| --- | ---: | ---: |
| Downstream region, all velocity components | 0.12968 | 0.15665 |
| Outer wake, all velocity components | 0.10259 | 0.13274 |
| Downstream reflection asymmetry | 0.07289 | 0.07349 |

In this control, alignment makes the velocity comparison **21% worse** in the
downstream region and **29% worse** in the outer wake. A short health-limit
continuation therefore cannot be used as proof that this accuracy issue is
fixed. These controls interpolate the donor targets linearly between the actual
3D reference snapshots at `t=6` and `t=7`; they are not full coupled trajectories
or a temporal convergence study. Their wall times are retained as provenance,
not a benchmark, because other diagnostic processes shared the machine.

The appropriate next work is to control the non-solenoidal component at its
representation/transfer/evolution owners and qualify the resulting method on
both the gradient-nullspace test and the cube. Simply changing the FVM boundary
condition, reducing `dt`, or enabling the tested alignment policy is not a
demonstrated accuracy fix. A fresh complete coupled run is still required
before claiming the developed wake agrees with the reference.

## Reproducible evidence

All new evidence is under
[`results/cube-wake-drift-2026-09-15`](results/cube-wake-drift-2026-09-15):

- `source-localization/localization.json`: full 3D source attribution.
- `operator-audit/audit.json`, `asymmetry-budget.json`, and `alignment-and-overlap.json`: native operator measurements.
- `rk-stage-accuracy/audit.json`: actual stage and half-step checks.
- `frozen-renewal-checked/audit.json` and `asymmetry.json`: transfer-only controls.
- `reference-symmetry-history/history.json`: reference evolution through t=30.
- `reference-driven-controls/{native,alignment}/replay.json`: both completed 100-step controls.
- `nullspace-evolution-checked/probe.json`: saved cube strain test.
- `nullspace-manufactured/probe.json`: exact boundary-free test.

The corresponding `cube_wake_*.py` drivers in this directory expose their
input/output arguments with `--help`. The decisive test needs no simulation
backup or historical source tree:

```bash
python studies/coupler_accuracy/cube_wake_nullspace_probe.py --manufactured-only --output /tmp/openonda-nullspace-check
```

Use a new output directory. Native backups and original tutorial outputs are
read-only inputs to these studies. `mechanism-verification.json` records input
integrity, completed checks, and source/artifact hashes. Large scratch particle
arrays remain local; the committed records identify the measurements and inputs.
