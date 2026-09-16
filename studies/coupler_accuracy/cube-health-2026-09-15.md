# Cube wake strain failure — 15 September 2026

## What the supplied run establishes

Both attachments describe the same failure at VPM step 1566, t=15.66 s,
using dt=0.01 s. The interface iteration converged in two sweeps. The VPM
accepted-state strain check then rejected the synchronized particle field.
The message rounded both the measured value and the limit to `1`, obscuring
the crossing. It did not compare the rounded values.

The saved history shows a growing wake disturbance well before rejection:

| Time [s] | Maximum speed / Uinf | Maximum particle vorticity [1/s] | Strain increment |
|---:|---:|---:|---:|
| 10.0 | 1.35 | 29.14 | 0.0802 |
| 13.0 | 1.35 | 29.30 | 0.1064 |
| 14.0 | 1.56 | 43.62 | 0.1259 |
| 14.5 | 2.32 | 76.30 | 0.1757 |
| 15.0 | 3.16 | 112.48 | 0.2212 |
| 15.5 | 7.51 | 550.54 | 0.7800 |

These measurements come from the retained HDF5 fields, not the rendered
figures. Their hashes and full values are in
[saved-state-history.json](results/cube-health-2026-09-15/saved-state-history.json).
Here particle vorticity means `|Gamma|/V`, which is distinct from the curl of
the induced velocity or the Gaussian-smoothed particle field.

After repairing the separate restart defect below, the unchanged four-rank
production replay **reproduced the failure at exactly step 1566**. Its strain
increment was 1.00009305954 at (2.83, -0.53, -1.19) m. Independent float64
direct summation at that failed particle also exceeds the limit, giving
1.00003489382. The Jacobian difference is 1.08e-4 relative. Thus even at the
actual crossing the rejection is not solely an approximation/rounding effect.
See [baseline health](results/cube-health-2026-09-15/baseline-health.jsonl) and
[independent gradient check](results/cube-health-2026-09-15/failure-gradient-check.json).

At t=15.5 the maximum strain is at (2.95, -0.17, -1.37) m, outside the
FVM domain, whose downstream boundary is x=1.5 m. A fresh evaluation of the
saved field gives an infinity strain norm of 77.9986 /s. Independently summing
the analytical Gaussian Biot–Savart Jacobian in float64 gives a relative
Frobenius difference of 6.87e-5 from the treecode at that point. The large
gradient is therefore present in the represented wake; it is not a large
tree-approximation error at the limiting location.

The recorded FVM state at t=15.66 has maximum speed 1.3441 Uinf, Courant
number 0.6943, maximum continuity error 5.35e-10, no nonfinite values, and
pressure between -0.7006 and 0.5466 m²/s². These observations locate the
reported runaway in the VPM wake. They do not establish full coupled accuracy.

## Why reducing dt did not settle the problem

The previous attempt reduced dt from 0.05 to 0.01 s. The instability recurred
at almost the same physical time. The smaller step satisfied the earlier
frozen-state bound, but the strain continued to grow.

The particle field also loses its divergence-free vorticity representation.
The existing Gaussian-neighbour diagnostic, evaluated at 512 probes in the
wake x>1.5 while retaining all source particles, gives weighted
`|div(omega)|/||grad(omega)||` of 0.2044 at t=10, 0.2377 at t=13, 0.2767 at
t=15, and 0.2867 at t=15.5. This is a numerical representation error:
physical vorticity is the curl of velocity and has zero divergence.

The configured stabilization policy originally removed particles outside the
domain only. It did not correct vorticity alignment or divergence. The
existing Pedrizzetti operator rotates particle strength toward the curl of
its induced velocity. Such relaxation is also used with transposed stretching
in [FLOWUnsteady's VPM formulation](https://flow.byu.edu/FLOWUnsteady/theory/rvpm/).
OpenONDA can additionally restore total vector strength, linear impulse and
angular impulse after that rotation; their measured transfers must remain
visible in the trial evidence.

This does not equate the FVM and VPM LES models: current variable-viscosity
GBD uses componentwise `div(nu_eff grad(omega))`, whereas FVM uses the viscous
stress divergence. The prior [3D investigation](cube-3d-findings.md) documents
that distinction. Stabilizing a trajectory alone cannot qualify its force
history, mesh resolution, or LES-model equivalence.

## Confirmed restart defect

The first four-rank checkpoint replay hung before advancing the first FVM
step. A process stack sample found rank zero waiting in `MPI_Scatterv`.
Boundary history is loaded on the VPM owner only, but
`initialize_vpm_boundary_history` formerly made its initialization decision
from each rank's local history. The owner skipped initialization while the
other ranks entered an extra collective boundary evaluation. Subsequent MPI
collectives were therefore out of order.

The decision now comes from the owner and is broadcast before branching.
A real two-rank regression writes a checkpoint, constructs fresh solvers,
loads it, advances the remaining step and verifies the final physical state
on both ranks. The production four-rank replay also advances after this fix.
This defect affects restarting; it did not cause the original fresh-run CFL
failure.

## Historical correction trial and validation

The tested cube setup enabled the existing Pedrizzetti alignment operator from
the first step, with relaxation rate 10/s and global moment restoration.
The timestep, mesh, LES coefficients and health thresholds are unchanged.
This addresses the observed particle-strength/curl inconsistency; it is a
case stabilization change, not a replacement of the stretching or diffusion
operators. It is no longer enabled in the tutorial: a later reference-driven
control found worse seam error, and the clean enlarged-domain initialization
exceeded the unchanged strength-growth acceptance limit.

A controlled four-rank continuation of the same t=15.5 checkpoint passed the
original failure point with strain increment **0.684243 at t=15.66**, compared
with **1.000093** for the unchanged policy. It advanced through **step 1577,
t=15.77**, with strain increment **0.656403** and maximum speed 6.949 Uinf.
Global moment transfers from the correction remained at float32 roundoff
scale (about 1e-7). See [corrected health](results/cube-health-2026-09-15/corrected-health.jsonl)
and [trial metadata](results/cube-health-2026-09-15/corrected-trial.json).

The user requested stopping the extended replay and will perform the fresh
case run. The full t=20 trajectory is therefore **not verified**. Particle
enstrophy and maximum particle vorticity still increased in this short
continuation from the damaged wake, so passing the original CFL crossing
does not establish their long-time boundedness or validate the physical
solution. Starting the correction at t=0 avoids importing that damaged wake.

All **76 focused regression tests passed**: 71 covering coupled backups,
factory construction, cube setup, VPM health, stabilization schedules and
transactional evolution, plus all five real MPI factory/restart tests.
Results are retained in [regression.xml](results/cube-health-2026-09-15/regression.xml)
and [mpi-regression.xml](results/cube-health-2026-09-15/mpi-regression.xml).

For replacement case results, use the existing `allrun.sh` in
`tutorials/coupled_fvm_vpm/02_cube_flow`. It starts from t=0, removes the old
solution/samples/figures, and retains the mesh. The supplied failed baseline
has also been preserved under
`studies/coupler_accuracy/results/cube-wake-drift-2026-09-15/baseline`.

## Diagnostic replay commands

The maintained replay harness requires the current checkpoint schema and current
configuration names. Historical checkpoints used for the measurements above are
retained as evidence; the current solver does not convert them on restart. The
subsequent [wake-mechanism investigation](cube-wake-mechanism-2026-09-15.md) tests
the accuracy limitations of the alignment policy separately from this short
stability result.

The CFL exception now reports nine significant digits, the limiting particle
and position, strain rate, actual dt, and the frozen-state timestep ceiling.
It explains that continued strain growth also requires a resolution and
vorticity-consistency investigation. The limit remains 1.

Use a new output directory for each controlled continuation:

```sh
python studies/coupler_accuracy/cube_health_replay.py \
  --output /private/tmp/cube-baseline --steps 20
python studies/coupler_accuracy/cube_health_replay.py \
  --output /private/tmp/cube-relaxation --steps 60 --relaxation-rate 10
```

The second command applies a relaxation frequency of 10/s, giving a blend of
0.1 per 0.01 s step, and restores global strength and impulses. The harness
loads the original checkpoint strictly before applying this explicit
experimental policy. It records accepted health, correction transfers and
failure fields separately from the user's simulation outputs. A continuation
from an already damaged wake is a stress test, not clean replacement data for
the case.
