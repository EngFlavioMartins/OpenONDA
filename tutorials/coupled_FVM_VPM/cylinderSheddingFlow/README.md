# Native FVM–VPM cylinder shedding

This Re=200 case uses OpenONDA's internal Cartesian mesher and immersed-
boundary FVM solver. It does not require OpenFOAM, Gmsh, or a repository path
on `PYTHONPATH`.

After installing OpenONDA, run:

```sh
./allrun.sh
./allplot.sh
```

The production horizon is 60 convective time units. Each 0.05 s VPM coupling
window contains five native-FVM substeps of 0.01 s, keeping the immersed-
boundary solve within its CFL limit. A short installation and coupling check
is available as:

```sh
OPENONDA_SMOKE=1 ./allrun.sh
```

`OPENONDA_PROCESSING_UNIT=CPU` explicitly selects the CPU when no supported
GPU is available. Generated fields are written below `solution/`, sampling
histories below `samples/`, and plots below `figures/`.

`OPENONDA_FVM_DT`, `OPENONDA_VPM_DT`, `OPENONDA_T_END`, `OPENONDA_SPACING`,
and `OPENONDA_MAX_PARTICLES` provide explicit resolution/study overrides. The
FVM time step must divide the VPM coupling window exactly.
