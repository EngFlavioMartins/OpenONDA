# Native FVM–VPM cube flow

This Re=1000 cube case uses OpenONDA's adaptive Cartesian FVM mesher and the
FVM–VPM coupler. It has no external mesher or repository-path requirement.

After installing OpenONDA, run the production case with:

```sh
./allrun.sh
./allplot.sh
```

For a short installation, meshing, panel, FVM, VPM, coupling, and solenoidality
qualification, run:

```sh
OPENONDA_SMOKE=1 ./allrun.sh
```

`OPENONDA_T_END` and `OPENONDA_SURFACE_CELL_SIZE` provide explicit run-length
and resolution overrides for isolated audit runs. `allrun.sh` moves any
existing `samples/` into a timestamped `run_backups/` entry before cleaning.
The preserved baseline hybrid samples also live in `samples_backup/`.

The production case uses four partitioned FVM ranks. Its coupled MPI smoke
test includes the body-wall geometry gathers used during transfer setup.

Production defaults use particle spacing `h=0.03125`, FVM time step `0.005`,
and VPM/coupling interval `0.015` (three FVM substeps). Particle diffusion,
global thresholding, and population control are performed only by VPM GBD.
