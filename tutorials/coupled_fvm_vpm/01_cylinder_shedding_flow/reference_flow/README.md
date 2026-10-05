# Cylinder FVM reference at $Re=150$

This standalone, body-fitted calculation provides force and wake profiles for the [coupled cylinder](../README.md). It uses the same 1 m diameter, 1 m/s freestream, $\nu=1/150$ m²/s, no-slip cylinder and 0.96 m free-slip span.

The domain is $[-8,24]\times[-10,10]\times[-0.48,0.48]$ m. The selected mesh has 0.04 m wall spacing and 24 spanwise layers. `setup.py` defines near-body and wake refinement, a velocity inlet, zero kinematic-pressure outlet and lateral slip boundaries. See [mesh setup](../../../../docs/fvm.md#mesh-setup) and [boundary conditions](../../../../docs/fvm.md#boundary-conditions).

```bash
./allrun.sh --fresh
./allplot.sh
```

The launcher runs `python setup.py -h 0.04`, with six FVM ranks, fixed 0.008 s steps and a 100 s end time. `--fresh` archives this reference case's existing results under `previous_runs/` and starts from zero. Without it, `./allrun.sh` and `./allcontinue.sh` resume compatible native backups. `./allclean.sh` deletes results.

Startup matches the coupled case: the freestream is $(1,0.1,0)$ m/s through 1 s, then its transverse component follows a quintic smoothstep to zero at 2 s. Both endpoint derivatives vanish, avoiding the abrupt switch's pressure/lift impulse. Boundary values follow the accepted FVM endpoint clock. The same small initial 3D curl disturbance is applied only on a fresh start. During the trigger and taper, the lateral patches admit the prescribed transverse normal velocity with zero tangential normal gradient; afterward they recover the original impermeable slip condition. The no-slip cylinder, spanwise slip planes, nominal $Re=150$, viscosity and force normalization retain their original values.

The schedule is saved in `solution/reference_startup.json` and checked on continuation. Resume before, during or after the taper restores the recorded forcing without repeating the initial disturbance. Exact compatible legacy schedules retain their original abrupt switch on continuation; fresh runs use the smooth taper. Existing results produced without a recognized schedule require `--fresh`. Exclude startup from developed shedding statistics.

A small native CPU box test reduced the startup pressure pulse by 86% with the taper; serial and two-rank native restarts agreed to roundoff. This verifies boundary timing and restart behavior, not full-cylinder force accuracy. Evidence is in `../drag_recovery/smooth_startup_validation/reference/`.

Forces and wake probes are sampled every 0.04 s, profiles every 0.2 s, slices every 0.4 s and volumes every 4 s. Read `samples/forces_history.csv`, `solution/fvm.pvd` and `samples/midspan.pvd`. Run `../allplot.sh` from this directory to compare forces and velocity profiles with the coupled case. This single mesh does not establish grid independence.
