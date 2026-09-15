# Conservative-source evolution in the three-dimensional cube wake

## Scope and decision

This study follows the [cause isolation](cube-transfer-cause-2026-09-15.md).
It tests a conservative correction in the evolving, prescribed-reference VPM
replay. It does not change production solver code or tutorial settings.
Both time-step comparisons are complete. The correction has a repeatable
velocity benefit and a closed source-moment budget, but it is not a complete
or production-qualified fix. Its cost and remaining support/impulse sensitivity
do not justify starting a full coupled production run with this implementation.

The completed 100-step comparison from reduced time 6 to 7 lowers velocity
error by 7.7% at the renewal seam and 12.8% in the outer wake. This is an
improvement, not elimination of the discrepancy. Source circulation is
conserved to floating-point accuracy. The present midpoint implementation has
a substantial cost, assessed separately from the concurrent half-step runs.
At half the time step the reductions are 9.9% and 12.7%, respectively.

## The source being tested

Let `J[i,j] = du_i/dx_j`, with `w` the reconstructed Gaussian particle
vorticity, `q = curl(u)`, and `w-q = grad(psi)`. The prior study identified
unphysical velocity production by transport of this longitudinal component.
The finite-support nodal source failed its circulation budget. Here the source is

```
phi = chi * psi
C_chi = -2 div(phi J^T)
Gamma_dot_i = -2 sum_faces(area * phi_face * J_face^T normal)
```

The scalar potential has units m/s and tends to zero at infinity. It is
evaluated by free-space linear convolution with the sampled Gaussian kernel.
There is no periodic potential substitution or inverse filtering. Each internal
face flux enters its two cells with opposite signs. No extra cell-volume factor
is applied to this integral rate, whose units are m³/s².

The cosine taper `chi` has fixed width 0.18 m and vanishes at the occupied-support
boundary, including holes and the body. The actual remeshed lattice has spacing
0.06 m and common core radius 0.066 m. Off-lattice or duplicate-cell states are
rejected. The operator adds no particles, clips no source, and repairs no moments.
The [spatial qualification](results/cube-wake-cause-2026-09-15/covector-flux-qualification/qualification.json)
predates this evolving test.

## Time integration and comparison controls

One physical step is `S(dt/2) A(dt) S(dt/2)`. Each `S` solves the source ODE by
explicit midpoint with fixed positions, cores, occupied cells and taper. Both
the free-space potential and complete particle-plus-panel Jacobian are
reevaluated from the midpoint strengths. Panel refreshes do not advance the
physical clock. `A` is the existing VPM RK2 evolution, GBD diffusion/remeshing,
and prescribed-reference renewal when due. Geometry is rebuilt after `A`.
The original accepted-state health checks run after the final source substep
and final panel refresh.

The full composition is not claimed to be second order: its native part
contains diffusion splitting and an algebraic renewal. The independent
[source-only temporal test](results/cube-compact-flux-evolution-2026-09-15/source-time-qualification-checked/qualification.json)
uses 512 particles, a changing self-induced Jacobian, a solenoidal affine
background, and an independently integrated DOP853 reference. Its measured
orders are 1.997 and 1.998. Net circulation and the nonzero source impulse are
accounted for to float64 rounding.

All trajectory controls start from the same saved time-6 state, with the same
donor cache, particle resolution, kernel, native RK, diffusion, LES, pruning,
and original health limits. They use the source that generated the saved state
and its strict historical reader as research provenance. They have no
Pedrizzetti alignment. They do not run the current production tutorial or FVM.
All controls use the same 300,000-particle capacity, well above their active
population; this is a storage choice, not a mesh or particle-spacing change.

Time steps are 0.01 and 0.005 s. Renewal remains at the same physical 0.01 s
interval in both, including its physical-time parameter. Donor interpolation
uses the solver clock, not the integer step number. The native fields at 6 and
7 are actual fine-reference states; intervening donor fields are linearly
interpolated. Only the endpoint at 7 supplies a native reference comparison
after evolution. The same 392-point three-dimensional probe set and region
masks are retained from the cause study.

The [protocol](results/cube-compact-flux-evolution-2026-09-15/protocol.json)
records the gates and the bounded run sequence. The new sequential native
10-step pilot reproduces the prior native velocity errors within `1e-8 U_inf`.
The two half-step jobs run concurrently for accuracy. Their wall times must not
be used for a performance ratio.

## Completed comparisons

| Region | Native velocity RMS error / U_inf | Corrected error / U_inf | Reduction |
| --- | ---: | ---: | ---: |
| Renewal seam | 0.129684 | 0.119653 | 7.7% |
| Outer wake | 0.102587 | 0.089438 | 12.8% |

These are equal-weight RMS errors over the predefined probes in each region,
not volume integrals or a whole-domain error norm.

The matching 200-step calculation with `dt=0.005 s` gives:

| Region | Native velocity RMS error / U_inf | Corrected error / U_inf | Reduction |
| --- | ---: | ---: | ---: |
| Renewal seam | 0.127676 | 0.114987 | 9.9% |
| Outer wake | 0.103756 | 0.090572 | 12.7% |

Halving the native time step alone leaves the large discrepancy largely
intact. The source benefit persists in both regions. The absolute benefit
changes by `0.00266 U_inf` at the seam (26.5% of its coarse-step benefit) and
`0.0000352 U_inf` in the outer wake (0.27%). Both changes are smaller than the
benefits, as required by the protocol, but the seam result is not an assertion
of a temporally converged error value. The full composition has no measured
convergence order from these two trajectories.

Corrected/native particle counts at time 7 are 51,296/51,313 for `dt=0.01 s`
and 49,051/49,017 for `dt=0.005 s`. The source itself preserves count in every
substep. See the [complete comparison](results/cube-compact-flux-evolution-2026-09-15/comparison.json).

All 100 and 200 final-state health checks passed in the two corrected runs.
They enforce finite state and
Lagrangian strain increment at most one; the saved configuration does not
enable a separate divergence or misalignment rejection threshold. Passing
health is therefore not a claim that those errors vanished.

There are 200 source substeps and 400 actual source evaluations. The maximum
float64 source net-rate norm is `3.60e-16 m³/s²`; the maximum circulation
storage-closure norm per source substep is `1.25e-8 m³/s`. Source-only
accumulated circulation change from storage rounding is approximately
`(-3.77e-8, 5.67e-8, -3.08e-8) m³/s`.

Source impulse is not constrained to zero. Its integrated value is
`(0.002910, -0.007125, 0.003444) m⁴/s` per unit density; forcing that value to zero
would change the intended source. Native transport/GBD and renewal have
separate ledgers. The final exported circulation and impulse match the final
ledger exactly. See the [source check](results/cube-compact-flux-evolution-2026-09-15/full-step-source-check.json)
and [complete replay](results/cube-compact-flux-evolution-2026-09-15/flux-dt001/replay.json).

The half-step run has 400 source substeps and 800 evaluations. Its maximum
source net rate is `3.91e-16 m³/s²`, maximum storage circulation closure is
`1.23e-8 m³/s`, and maximum storage impulse closure is `5.93e-9 m⁴/s`.
The corresponding full-step impulse closure is `6.12e-9 m⁴/s`.

Closed accounting does not prove force accuracy or time convergence. The
source's integrated streamwise impulse changes from `0.00291` to `0.00669 m⁴/s`
between the two step sizes. The transverse source contributions are more
stable, but the streamwise sensitivity must be resolved before drawing a
force conclusion. Remeshing/pruning and the occupied source support change
with the trajectory; this audit does not assign that sensitivity to one of
them without a separate isolation test.

## Cost and an obvious optimization

The sequential 10-step pilots give roughly 3.9 s/step for native evolution plus
renewal and 9.0 s/step with the source, excluding the first two startup steps.
These are laptop measurements with external load, not controlled full-coupler
benchmarks. Four additional body-complete Jacobian evaluations per physical
step account for much of the source cost. The complete 100-step corrected run
averages 9.21 s/step, of which 4.79 s/step is spent in the source substeps.

A [separate benchmark](results/cube-compact-flux-evolution-2026-09-15/potential-workspace-checked/benchmark.json)
tests caching the free-space kernel transforms, using a sufficient `2N-1`
convolution embedding, and combining the three spectral products before one
inverse FFT. Agreement on the real cloud is within `3.34e-16 m/s`; odd/even
random-grid controls also agree to roundoff. The potential evaluation alone
is approximately 8–10 times faster in the measured batches. This optimization
is not used by the trajectory controls and does not imply an 8–10-fold whole-step
speedup.

A cheaper, first-order Lie source kick could reuse a freshly evaluated
accepted-state Jacobian. It would still need current pre-kick `psi`, correct
cache invalidation, fresh native RK fields after the kick, and its own accuracy
and stability qualification. This is an untested direction, not an enabled fix.

## Limits

The source corrects `grad(chi*psi)`, not all `grad(psi)`. The complement,
taper-compensation effects and Gaussian representation error remain. Changing
the time integrator cannot resolve that spatial limitation. The earlier
occupied-support audit found only about 8% of downstream strength inside the
full-taper plateau at this width.

The restart already contains a disturbed wake. The present test measures
continued evolution from that state, not prevention from startup. No full
coupled trajectory, force comparison, or full-coupler timing has been qualified
by this study.
