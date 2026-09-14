# Bounded VPM leapfrogging comparison

These results use the VPM sampler outputs and recorded run statuses. They do not establish spatial convergence or matched periodic boundaries.

| Run | Status | Last scalar time | Last field time | Wall minutes | Pair tracked until |
|---|---|---:|---:|---:|---:|
| baseline | running | 4.688 | 4.65 | not recorded | 3.6 |

Not plotted (no saved fields or metadata): stretching_viscosity, p_moments.

Pair tracking means two separated maxima on the recorded meridional plane. Its termination is a diagnostic cutoff, not proof of physical breakdown. A wall_time_limit status is a computational budget stop. A created status means the native metadata has not been finalized; it does not establish completion or explain termination. Saved sampler times remain usable independently of that lifecycle status. Completed and budget-limited runs provide lower bounds on numerical survival. A failed status needs its native vpm.log to distinguish numerical failure from output/resource failure; neither is automatically physical breakdown.
Optional within-core saddle grouping: 0.9. The raw maximum counts and grouped-peak coordinate spans are retained in the peak tables.
The trajectory figure ends each run when the two core identities can no longer be followed. Later field maxima remain in the peak tables but do not extend the identified-ring scores.

## LBM radius discrepancies

RMS errors below are percentages of R0 on fixed axial intervals. Compare methods on the same covered interval; unavailable intervals are not extrapolated.

| Run | x/R0 interval | RMS radius error (% R0) |
|---|---|---:|
| baseline | [0.55, 1.5] | 2.182 |
| baseline | [0.55, 2.5] | 5.148 |
| baseline | [0.55, 3.5] | 6.809 |
| baseline | [0.55, 5.5] | unavailable: Both rings must cover the entire requested interval |
| baseline | [0.55, 7.0] | unavailable: Both rings must cover the entire requested interval |

**baseline**: step 1000: not exactly two resolved peaks. Solver status: running.
Tracked axial coverage: core 1: x/R0=4.820, core 2: x/R0=4.280. Both cores reached 7: False.

## VPM leapfrogging timing

Axial-order reversals of two identified field cores, interpolated between saved planes. Brackets show the output cadence. These are not breakdown events. The LBM CSV has no timestamps, so LBM periods and temporal phase errors are unavailable.

| Run | Passage times [s] | Two-passage cycle periods [s] |
|---|---|---|
| baseline | 1.1571 [1.050, 1.200]; 3.2500 [3.150, 3.300] | unavailable |

## Whole-solver timestep sensitivity

Equal-time L2 differences on identical SurfaceSampler grids; these include time integration, diffusion and remapping effects.

| Coarse run → fine run | Physical time | Velocity difference (%) | Vorticity difference (%) |
|---|---:|---:|---:|

No comparable noninitial sampler times are available.

The accompanying lbm_agreement.json records solver settings, field hashes, health diagnostics, cadence sensitivity and the reference hash. Longer survival alone is not evidence of better physics.
