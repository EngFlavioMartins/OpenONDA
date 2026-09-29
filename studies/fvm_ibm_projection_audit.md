# FVM mixed-boundary projection audit

The actual default `tutorials/fvm/cylinder_ibm` run diverged by time 0.15546 s despite its existing timestep bound. Its conservative face flux and linear residuals remained small while velocity and loads grew. A 1,536-cell uniform-grid reproduction isolated the issue: **without any immersed body**, uniform flow grew from roundoff to a maximum speed of 2.784 after 40 steps. The largest error was at the outlet/lateral-boundary corner. This established a general far-field projection defect rather than a cylinder-specific force limit problem.

The production correction selects ordinary freestream inflow/outflow branches from the incoming conservative face flux once per momentum/projection solve. Momentum boundary values, implicit diffusion, pressure matrix coefficients, pressure null-space selection, pressure gradients and subsequent velocity/flux corrections all use that same branch. Existing fixed directional masks supplied by the coupled solver take precedence. Outflow predictor ghosts refresh from the newly solved owner velocity. There are no new public controls, hysteresis thresholds, force caps or relaxed health checks.

Before correction, the 1,536-cell IBM reproduction reached only 0.19659 s after 40 steps, with maximum speed 2,079 and marker slip 26.16. With the production correction, the same input reached 0.46512 s, maximum speed 1.46759 and final marker slip 0.0012036. The corresponding uniform-flow regression preserves the uniform velocity within 1e-9.

The **actual default 12,642-cell mesh**, with the existing timestep bound retained, completed a bounded verification to 0.600 s in 52 steps. Maximum speed was 2.54129, maximum face-flux divergence 7.2924e-15/s, final marker slip 0.0014904, and final kinetic energy divided by initial energy 1.06029. This passes substantially beyond the former failure; it is not qualification of the full 60-second production run. The exact history is saved in `fvm_ibm_projection_verification.json` and separately in `fvm_ibm_projection_capped_baseline.json`.

## Reproduction and regressions

From an installed checkout, with the selected Python environment:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python studies/verify_fvm_ibm_projection.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q tests/fvm/test_freestream_projection_stability.py tests/fvm/test_matrix_workspace_boundary_layout.py tests/fvm/test_nonorthogonal_pressure_correction.py tests/fvm/test_fixed_flux_pressure_restart.py tests/fvm/test_mixed_boundary_state.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q tests/fvm/test_application_runtime_mpi.py -k freestream
```

All 14 serial regressions passed, including 40-step uniform and IBM histories, no-slip/divergence/energy checks, accepted-clock and field equality after checkpoint rollback, fixed-mask pressure matrix/null-space selection, nonorthogonal pressure correction, native fixed-flux restart and mixed-field setters. The real two-rank MPI freestream application also passed. Local MPI process startup required the normal host execution permission because the filesystem sandbox blocked its transport initialization. Ruff passes on all changed production/test files.

## Separate unresolved timestep restriction

Removing the tutorial's existing Fourier-number bound was tested explicitly; it is **not qualified**. The actual default mesh using the original requested initial step 0.01 s and maximum 0.03 s reached 0.600 s in 23 steps without velocity or face-continuity divergence, but final marker slip increased to 0.78719 and drag coefficient to 16.446. The complete failed history remains in `fvm_ibm_projection_unclamped_failure.json`. The original bound therefore remains in the tutorial; this audit does not claim that knob was removed.

The current IBM applies direct residual forcing before pressure projection and does not jointly solve its marker constraint with pressure. Bounded diagnostic alternatives—extra alternating marker/pressure corrections, implicit momentum residual forcing and projected marker-mobility solves—did not establish a stable, performant replacement with both correct marker velocity and bounded flow. None of those prototypes were added to production. A future removal of the bound must qualify marker slip and continuity together, with conserved forcing and restart behavior, rather than merely report a finite velocity field.
