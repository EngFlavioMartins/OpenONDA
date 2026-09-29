# Rotor refinement and stability audit (29 September 2026)

The protected `06_rotor_flow` run was inspected without modifying or restarting it. At the latest inspected log entry, step 1191 reached 7.146 s with 339,754 particles; the latest inspected native checkpoint was step 1188, 7.128 s, 335,167 particles. This does not establish stability beyond the reported 7.5 s failure or completion at 10 s.

## What the current model does

The default rotor uses CS diffusion, a 0.006 s step, 600,000 hard particle capacity, and filament refinement every five steps. Filament bisection retains each child's parent core radius while halving strength and volume. It is a fixed-core line-resolution operation, not a reset of cores enlarged by CS. The step-1190 log records 4,182 splits, zero deferred, and zero vector-strength closure error. The checkpoint's maximum core radius was 0.910 m, versus the 0.568 m initial wake core and 0.227 m radial wake spacing. Core-size regularization is available in the solver but the resolved rotor configuration has `regularization_interval_steps=0`; the run has zero regularization events. Thus the user's requested CS radius-control strategy is not active in this default case. Enabling it would introduce global redistribution and require a measured field/conservation/cost comparison; there is no evidence that an arbitrary radius trigger fixes the late run.

The wake-Courant warning is based on the smallest blade chordwise panel (about 0.058 m) and a conservative boundary-speed sum. It is not proof of time-step instability. At the tip, rotation moves about 0.294 m per step: 2.81 degrees, 1.29 radial wake spacings, and 0.52 initial core radii. A smaller step would emit more particles. A matched-step accuracy and particle-growth comparison is required before changing this physical discretization.

The five selected existing tests for hard-cap atomic split, absolute-strength split, core-radius trigger, Gaussian remap field/second-moment accuracy, and strict remap capacity all passed on this source tree:

```
/home/flavio-martins/anaconda3/envs/OpenONDA/bin/python -m pytest -q tests/vpm/test_stabilization_schedules.py tests/vpm/test_gaussian_core_remeshing.py -k 'filament_refinement_fails_before_partial_split_at_hard_capacity or filament_refinement_catches_absolute_strength_after_reference_reset or regularization_can_be_triggered_only_by_core_radius or variable_core_reset_preserves_resolved_gaussian_field_and_second_moments or capacity_does_not_silently_override_the_tail_budget'
```

The remap test checks the resolved Gaussian field to 0.3%, energy and enstrophy transfer to 0.5%, vector strength within the declared tail budget, and Gaussian second moments. Filament bisection preserves strength and moments but its test explicitly finds a nonzero isolated energy change. These small tests qualify individual numerical operations, not a 340,000-particle rotor trajectory.

## Cost and outstanding qualification

At step 1171 (321,390 particles) wall time was 315.50 s and the measured VLM solve was 1.1051 s. The next step was 311.37 s; step 1190 (339,619 particles) was 382.21 s. Detailed timing is disabled in the saved metadata, so the remaining time cannot be assigned quantitatively among induction, CS/LES, and other VPM work. The unsampled step 1171 excludes field sampling and the every-five-step split as the dominant cause of its 315 s wall time; the VLM solve itself accounts for only 0.35%. This points to VPM evolution as the broad bottleneck, without a proven kernel-level attribution.

From 7.128 to 7.5 s requires 62 steps. At the observed 311–382 s/step, that is roughly 5.4–6.6 hours if cost stays constant. From the saved checkpoint to 10.0 s requires 479 steps and substantially more work; the growing particle count makes constant-cost extrapolation optimistic. The protected run should be allowed to provide the first evidence past 7.5 s. Any isolated reproduction must copy the authenticated native checkpoint and preserve all physical and numerical settings and failure gates, then record strength/moment/energy histories, core and count distributions, refinement events, and wall timings. The recent private Treecode workspace allocation change needs an actual native restart-compatibility check before such a source-updated trial is called qualified.

Read-only evidence: `tutorials/vpm/06_rotor_flow/setup.py`, `solution/vpm.log`, `solution/vpm_metadata.json`, `solution/vpm/vpm_001188.h5`, and `samples/rotor/flow_integrals.csv`; implementation and tests in `source/solvers/vpm/stabilization/{filament_refinement,regularization,remeshing}.py`, `tests/vpm/test_stabilization_schedules.py`, and `tests/vpm/test_gaussian_core_remeshing.py`.
