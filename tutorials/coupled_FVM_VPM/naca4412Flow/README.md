# Native FVM–VPM NACA 4412 flow

This finite-span NACA 4412 case at 10 degrees and Re=1000 uses OpenONDA's
internal Cartesian mesher and immersed-boundary FVM solver. It does not require
OpenFOAM, Gmsh, or a repository path on `PYTHONPATH`.

After installing OpenONDA, run:

```sh
./allrun.sh
./allplot.sh
```

The production horizon is 12 convective time units. A two-step installation
and coupling check is available as:

```sh
OPENONDA_SMOKE=1 ./allrun.sh
```

`OPENONDA_PROCESSING_UNIT=CPU` explicitly selects the CPU when no supported
GPU is available. Generated fields are written below `solution/`, sampling
histories below `samples/`, and plots below `figures/`.
