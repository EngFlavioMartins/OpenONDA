# Native FVM–VPM cube flow

This Re=1000 cube case uses OpenONDA's adaptive Cartesian FVM mesher and the
FVM–VPM coupler. It has no external mesher or repository-path requirement.

After installing OpenONDA, run the production case with:

```sh
./allrun.sh
./allplot.sh
```

For a short installation, meshing, panel, FVM, VPM, coupling, and conservation
qualification, run:

```sh
OPENONDA_SMOKE=1 ./allrun.sh
```

`OPENONDA_T_END`, `OPENONDA_SPACING`, `OPENONDA_SURFACE_CELL_SIZE`,
`OPENONDA_MAX_PARTICLES`, and `OPENONDA_FVM_CORES` provide explicit study and
resource overrides. The production defaults remain the qualified configuration.
