# Cylinder FVM reference at $Re=150$

This standalone, body-fitted calculation provides force and wake profiles for the [coupled cylinder](../README.md). It uses the same 1 m diameter, 1 m/s freestream, $\nu=1/150$ m²/s, no-slip cylinder and 0.96 m free-slip span.

The domain is $[-8,24]\times[-10,10]\times[-0.48,0.48]$ m. The selected mesh has 0.04 m wall spacing and 24 spanwise layers. `setup.py` defines near-body and wake refinement, a velocity inlet, zero kinematic-pressure outlet and lateral slip boundaries. See [mesh setup](../../../../docs/fvm.md#mesh-setup) and [boundary conditions](../../../../docs/fvm.md#boundary-conditions).

```bash
./allrun.sh
./allplot.sh
```

The launcher runs `python setup.py -h 0.04`, with six FVM ranks, fixed 0.008 s steps and a 100 s end time. `./allcontinue.sh` resumes compatible backups; both launchers preserve outputs. `./allclean.sh` deletes results.

Forces and wake probes are sampled every 0.04 s, profiles every 0.2 s, slices every 0.4 s and volumes every 4 s. Read `samples/forces_history.csv`, `solution/fvm.pvd` and `samples/midspan.pvd`. Run `../allplot.sh` from this directory to compare forces and velocity profiles with the coupled case. This single mesh does not establish grid independence.
