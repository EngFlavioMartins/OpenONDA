# Cylinder FVM reference at $Re=150$

This standalone, body-fitted calculation provides force and wake profiles for the [coupled cylinder](../README.md). It uses the same 1 m diameter, 1 m/s freestream, $\nu=1/150$ m²/s, no-slip cylinder and unit span with one FVM cell between periodic `zmin` and `zmax` patches.

The domain is $[-8,24]\times[-10,10]\times[-0.5,0.5]$ m. The selected mesh has 0.04 m wall spacing and a single 1 m-wide spanwise cell. Force coefficients use the unit-span frontal area $Db=1$ m². `setup.py` defines near-body and wake refinement, a velocity inlet, zero kinematic-pressure outlet and lateral slip boundaries. See [mesh setup](../../../../docs/fvm.md#mesh-setup) and [boundary conditions](../../../../docs/fvm.md#boundary-conditions).

```bash
./allrun.sh --fresh
./allplot.sh
```

The launcher runs the local `setup.py` through `openonda.tutorial_runner`. All physical inputs, mesh spacing, rank count, time steps and output intervals are defined in this case's `setup.py`. The current defaults use one FVM process, fixed 0.008 s steps and a 100 s end time. Periodic MPI currently replicates the mesh and solver workspace on each rank; one process avoids that extra memory. `--fresh` archives this reference case's existing results, including the native mesh, under `previous_runs/` and starts from zero. Without it, `./allrun.sh` and `./allcontinue.sh` resume native backups matching the current configuration. `./allclean.sh` deletes results. The earlier multi-layer/slip checkpoint cannot be continued with this two-dimensional setup; use `--fresh` to archive it before the manual rerun.

Startup matches the coupled case: the freestream is $(1,0.1,0)$ m/s through 1 s, then its transverse component follows a quintic smoothstep to zero at 2 s. Both endpoint derivatives vanish. Boundary values follow the accepted FVM endpoint clock. The same small initial divergence-free planar curl disturbance is applied only on a fresh start. During the trigger and taper, the lateral patches impose the prescribed transverse normal velocity with zero tangential normal gradient; afterward they recover the original impermeable slip condition. The no-slip cylinder, spanwise periodic pair, nominal $Re=150$, viscosity and force normalization retain their original values.

The velocity histories are declared as physical boundary inputs and saved in the native checkpoint configuration. Continuation restores the accepted velocity without repeating the initial disturbance. Exclude startup from developed shedding statistics.

Forces and wake probes are sampled every 0.04 s, profiles every 0.2 s, slices every 0.4 s and volumes every 4 s. Read `samples/forces_history.csv`, `solution/fvm.pvd` and `samples/midspan.pvd`. `./allplot.sh` plots this case's native drag and lift history to `figures/forces.png` and `figures/forces.pdf`. Choose one format with `./allplot.sh png` or `./allplot.sh pdf`. Run `../allplot.sh` to compare forces and velocity profiles with the coupled case. This single mesh does not establish grid independence.

## Two-dimensional grid study

The reference grid study runs three meshes sequentially, with in-plane wall spacings 0.09, 0.06 and 0.04 m (refinement ratio 1.5). Every mesh keeps exactly one cell over the same unit span, the reciprocal periodic patches, one FVM process and the same startup, viscosity, time step and force normalization. The finest spacing matches the default coupled case. Only the XY mesh is refined; this is a two-dimensional spatial study.

From the repository root, with the OpenONDA environment active, run:

```bash
python -m tests.support.cylinder.run_parameter_study --kind reference
```

The verification launcher lives under `tests/`. It creates a separate timestamped directory under `../study_results/cylinder/`, with each grid's native solution and samples in its own subdirectory. It leaves the standalone reference outputs intact. Resume an interrupted study with the same directory:

```bash
python -m tests.support.cylinder.run_parameter_study --kind reference --run-dir /path/to/study --resume
```

After all grids reach 100 s, the study reports mean drag, drag/lift RMS, shedding frequency and Richardson/GCI estimates, using the common 40–100 s window and complete-cycle sampling uncertainty. The native grid report also compares the centreline velocity profiles. Force convergence is accepted only when the registered sampling and mesh-convergence criteria pass; temporal and domain independence require separate studies.
