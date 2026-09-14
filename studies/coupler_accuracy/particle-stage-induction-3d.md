# Particle-stage induction accuracy in the physical 3D cube

A stricter FMM setting improves the isolated stage calculations substantially
but does not improve the completed 20-interval coupled comparison below.
The native setting is retained for subsequent experiments. Accurate saved-source
profile queries do not establish accurate particle transport: this backend
uses direct regularized kernels for arbitrary targets and a separate
hierarchical path for RK stages. The
[saved-stage audit](particle_stage_induction_audit_3d.py) measures that
distinction on the accepted state at physical time `1.5`.

The audit uses the same 28,441 particles and 256 selected fluid particle
positions as the [body-stage probe](body-query-and-transport-3d.md). Both
comparisons exclude the body term and include the same freestream. All
source and target core radii are `0.0625`, so symmetric particle-pair
smoothing agrees with the source-only point-query smoothing here.

| Region, 128 targets each | FMM stage velocity error (% U∞) | Point-query velocity error (% U∞) | FMM gradient relative error (%) | FMM stretching-rate relative error (%) |
| --- | ---: | ---: | ---: | ---: |
| Near body | 0.038289 | 0.000127 | 0.341178 | 0.449078 |
| Near wake | 0.032673 | 0.000031 | 1.448955 | 1.105116 |

Gradient errors use the Frobenius norm; stretching errors use vector RMS.
The near-body region has maximum absolute coordinate below one; the near
wake has `1.5 < x ≤ 4` and both transverse coordinates within `±1.5`.
These are selected points in the real three-dimensional flow, not a 2D model.
The differences concern one stage's rates and do not bound accumulated
trajectory error or identify the full cause of the coupled wake discrepancy.

## Independent checks

The reference directly sums the Gaussian velocity kernel and its analytical
target derivative in double precision, retaining the finite self-gradient.
Velocity agrees with the earlier independent Gaussian evaluator to
`2.220e−16`. The full Jacobian agrees with the host pair-kernel evaluator
to `1.110e−14`. Fourth-order differences of the independent velocity at
eight targets, with steps `2e−5` and `1e−5`, agree with the analytical
Jacobian to `1.104e−11` and `2.497e−11`.

The saved native strength rate also reproduces the native transposed-gradient
contraction within its single-precision accumulation bound. This isolates
the induction approximation from a different contraction convention. It
does not prove that the particle stretching formulation exactly matches
the FVM vorticity equation.

The [audit record](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/results/interface-precision-frozen-workspace-qualified/studies/coupler_accuracy/results/particle-stage-induction-at-one-point-five/particle-stage-induction-audit-3d.json)
contains the direct and native stage arrays, all reported metrics and source
hashes. It reuses the strictly restored state and bitwise stage replay from
the qualified body probe.

## Controlled FMM accuracy adjustment

The [separation probe](particle_stage_separation_probe_3d.py) restores the
same canonical backup in separate processes and changes only the internal
geometric separation factor before Taichi compilation. The fixed Cartesian
expansion order, kernel-tail tolerances, particle precision and stretching
scheme remain unchanged. Stricter variants use separate compilation caches.
The default factor `3` reproduces all saved stage velocities, Jacobians and
strength rates bitwise, across all particles. Every probe's second evaluation
also replays bitwise, with primary source fields and clocks unchanged.

| Geometric separation factor | Near-body rate error (%) | Near-wake rate error (%) | Direct particle interactions, excluding self | Second evaluation plus diagnostic I/O (s) |
| ---: | ---: | ---: | ---: | ---: |
| 3, current default | 0.449078 | 1.105116 | 114,300,804 | 3.27 |
| 4.5 | 0.094015 | 0.356769 | 280,181,054 | 7.47 |
| 6 | 0.007057 | 0.141091 | 495,719,156 | 13.56 |

At factor six, gradient relative errors fall to `0.017542%` near the body
and `0.152564%` in the sampled wake. Velocity errors fall to `0.001415%`
and `0.002996% U∞`. This is a measurable component improvement, with a
substantial cost increase. The timings include state hashing, downloads
and compressed output, and were obtained alongside other jobs; they are
not isolated performance benchmarks.

The [default control](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/results/interface-precision-frozen-workspace-qualified/studies/coupler_accuracy/results/particle-stage-separation-three-control/particle-stage-separation-probe-3d.json),
[factor 4.5](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/results/interface-precision-frozen-workspace-qualified/studies/coupler_accuracy/results/particle-stage-separation-four-point-five/particle-stage-separation-probe-3d.json),
and [factor 6](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/results/interface-precision-frozen-workspace-qualified/studies/coupler_accuracy/results/particle-stage-separation-six/particle-stage-separation-probe-3d.json)
retain their exact fields, interaction counts and source records. All 799
original frozen source/input files remain unchanged in each probe.

## Completed advancing comparison

The [advancing runner](cube_fmm_separation_3d.py) changes only the geometric
separation factor before creating the solver. Its default-factor control
has completed three intervals. Histories, six comparison arrays, 17 FVM
entries, 11 boundary-history entries and 11 numeric VPM datasets reproduce
the existing control bitwise. The profile figure has the same hash as the
previously inspected wrapper-control figure.

The [schedule verifier](verify_fmm_separation_schedule_3d.py) confirms six
native FMM stage evaluations, three GBD calls and twelve outer stabilization
phases. Stage times and particle clocks retain the original schedule. The
control records actual interaction counts; the saved-state probes above
separately quantify their accuracy. All 799 original frozen files remain
unchanged. The complete
[control qualification](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/results/interface-precision-frozen-workspace-qualified/studies/coupler_accuracy/results/fmm-separation-three-three-schedule.json)
links the bitwise comparison and the source records.

A factor-six run has completed 20 intervals, from physical time `0.5` to
`1.5`, using its separate compilation cache. It retains the original
`0.05` RK/GBD/exchange interval, `0.01` FVM step, matched cells, renewal and
panel-update schedules. The body-transport addition is tested separately.
All 20 intervals converge in three sweeps each. The independent
[comparison](compare_fmm_separation_3d.py) passes 141 scalar checks with zero
maximum difference, including ten reconstructed wall-force vectors and the
native volume-weighted sampled-cell VPM error. The independent full-reference
fields and initial source fields agree bitwise with the baseline.

| Measurement, physical time 0.5–1.5 | Native factor 3 | Stricter factor 6 |
| --- | ---: | ---: |
| Drag-history RMS relative error (%) | 0.555068 | 0.555345 |
| Maximum absolute relative drag error (%) | 1.481333 | 1.481706 |
| Final hybrid Cd | 1.088560728 | 1.088573505 |
| Final whole-small-FVM velocity error (% U∞) | 0.766375 | 0.766786 |
| Final near-body FVM velocity error (% U∞) | 0.195493 | 0.195700 |
| Final centreline near-wake velocity error (% U∞) | 1.101853 | 1.105401 |
| Final off-axis near-wake velocity error (% U∞) | 0.772103 | 0.770527 |

The stricter setting does not improve the short coupled comparison. Whole-FVM
velocity error increases `0.05365%` relative to baseline, and near-body error
increases `0.10594%`. The off-axis wake line improves slightly, while drag
and centreline wake errors increase slightly. This measured response does
not explain the remaining short-run discrepancy despite the substantial
saved-stage accuracy improvement. No stricter production default is selected.

![Stricter-FMM comparison](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/results/interface-precision-frozen-workspace-qualified/studies/coupler_accuracy/results/fmm-separation-six-twenty-comparison/comparison.png)

The figure was visually inspected. The
[complete comparison record](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/results/interface-precision-frozen-workspace-qualified/studies/coupler_accuracy/results/fmm-separation-six-twenty-comparison/fmm-separation-comparison-3d.json)
retains 534 source/artifact records, unshifted histories, profiles, direct
checks and PNG/SVG exports. Final direct/query velocity difference is
`7.451e-7 U∞` vector RMS, maximum `5.895e-6 U∞`. The schedule verifier
confirms 40 native FMM stage calls, 20 GBD calls and 80 outer stabilization
phases; all 799 original frozen files remain unchanged. This comparison does
not qualify the effect of stricter FMM evaluation on the later developed wake.

Earlier direct profile checks remain valid checks of query evaluation; they
must not be described as ruling out FMM error in particle evolution.

The separate [stretching-consistency audit](particle-stretching-consistency-3d.md)
also measures a discrepancy between the Gaussian vorticity sum and the curl
of its induced velocity on the same physical state. Improving the FMM stage
evaluator does not itself remove that representation difference.
