# Bounded VPM leapfrogging comparison

These results use the VPM sampler outputs and recorded run statuses. They do not establish spatial convergence or matched periodic boundaries.

| Run | Status | Last scalar time | Last field time | Wall minutes | Pair tracked until |
|---|---|---:|---:|---:|---:|
| baseline | resolution_lost | 5.779 | 5.779 | not recorded | 3.75 |
| stretching_viscosity | completed | 9 | 9 | not recorded | 3.9 |
| p_moments | running | 5.513 | 5.4 | not recorded | 3.75 |

Pair tracking means two separated maxima on the recorded meridional plane. Its termination is a diagnostic cutoff, not proof of physical breakdown. A wall_time_limit status is a computational budget stop. A created status means the native metadata has not been finalized; it does not establish completion or explain termination. Saved sampler times remain usable independently of that lifecycle status. Completed and budget-limited runs provide lower bounds on numerical survival. A failed status needs its native vpm.log to distinguish numerical failure from output/resource failure; neither is automatically physical breakdown.
Optional within-core saddle grouping: 0.9. The raw maximum counts and grouped-peak coordinate spans are retained in the peak tables.
The trajectory figure ends each run when two coherent field tracks can no longer be followed. Track numbers come from initial axial order, not particle group_id ancestry. Later field maxima remain in the peak tables but do not extend the two-core scores.

## LBM radius discrepancies

RMS errors below are percentages of R0 on fixed axial intervals. Compare methods on the same covered interval; unavailable intervals are not extrapolated.

| Run | x/R0 interval | RMS radius error (% R0) |
|---|---|---:|
| baseline | [0.55, 1.5] | 2.279 |
| baseline | [0.55, 2.5] | 6.939 |
| baseline | [0.55, 3.5] | 10.211 |
| baseline | [0.55, 5.5] | unavailable: Both rings must cover the entire requested interval |
| baseline | [0.55, 7.0] | unavailable: Both rings must cover the entire requested interval |
| stretching_viscosity | [0.55, 1.5] | 2.392 |
| stretching_viscosity | [0.55, 2.5] | 5.982 |
| stretching_viscosity | [0.55, 3.5] | 8.056 |
| stretching_viscosity | [0.55, 5.5] | unavailable: Both rings must cover the entire requested interval |
| stretching_viscosity | [0.55, 7.0] | unavailable: Both rings must cover the entire requested interval |
| p_moments | [0.55, 1.5] | 2.279 |
| p_moments | [0.55, 2.5] | 6.939 |
| p_moments | [0.55, 3.5] | 10.211 |
| p_moments | [0.55, 5.5] | unavailable: Both rings must cover the entire requested interval |
| p_moments | [0.55, 7.0] | unavailable: Both rings must cover the entire requested interval |

**baseline**: step 1040: not exactly two resolved peaks. Solver status: resolution_lost.
Tracked axial coverage: core 1: x/R0=5.140, core 2: x/R0=4.620. Both cores reached 7: False.


**stretching_viscosity**: step 1080: not exactly two resolved peaks. Solver status: completed.
Tracked axial coverage: core 1: x/R0=5.240, core 2: x/R0=4.640. Both cores reached 7: False.


**p_moments**: step 1040: not exactly two resolved peaks. Solver status: running.
Tracked axial coverage: core 1: x/R0=5.140, core 2: x/R0=4.620. Both cores reached 7: False.

## VPM leapfrogging timing

Axial-order reversals of two identified field cores, interpolated between saved planes. Brackets show the output cadence. These are not breakdown events. The LBM CSV has no timestamps, so LBM periods and temporal phase errors are unavailable.

| Run | Passage times [s] | Two-passage cycle periods [s] |
|---|---|---|
| baseline | 1.1531 [1.050, 1.200]; 3.4406 [3.300, 3.450] | unavailable |
| stretching_viscosity | 1.1531 [1.050, 1.200]; 3.2893 [3.150, 3.300] | unavailable |
| p_moments | 1.1531 [1.050, 1.200]; 3.4406 [3.300, 3.450] | unavailable |

## Whole-solver timestep sensitivity

Equal-time L2 differences on identical SurfaceSampler grids; these include time integration, diffusion and remapping effects.

| Coarse run → fine run | Physical time | Velocity difference (%) | Vorticity difference (%) |
|---|---:|---:|---:|

No comparable noninitial sampler times are available.

The accompanying lbm_agreement.json records solver settings, field hashes, health diagnostics, cadence sensitivity and the reference hash. Longer survival alone is not evidence of better physics.
