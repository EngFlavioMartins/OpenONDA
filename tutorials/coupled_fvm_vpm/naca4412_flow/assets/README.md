# Native FVM–VPM NACA 4412 flow

This finite-span NACA 4412 case at 10 degrees and Re=1000 uses OpenONDA's
internal Cartesian mesher and immersed-boundary FVM solver. It does not require
an external mesher or a repository path on `PYTHONPATH`.

After installing OpenONDA, run:

```sh
./allrun.sh
./allplot.sh
```

The production horizon is 12 convective time units. The VPM advances in 0.04 s
coupling windows while the native FVM uses four 0.01 s substeps per window to
keep the immersed-boundary transient within its CFL limit. A short installation
and coupling check is available as:

```sh
OPENONDA_SMOKE=1 ./allrun.sh
```

`OPENONDA_COMPUTE_DEVICE=CPU` explicitly selects the CPU when no supported
GPU is available. Generated fields are written below `solution/`, sampling
histories below `samples/`, and plots below `figures/`.

`OPENONDA_FVM_TIME_STEP_SIZE`, `OPENONDA_VPM_TIME_STEP_SIZE`,
`OPENONDA_T_END`, `OPENONDA_SPACING`,
`OPENONDA_IBM_MARKER_RATIO`, and `OPENONDA_MAX_PARTICLES` provide explicit
resolution/study overrides. The default 2.5-cell marker separation avoids an
ill-conditioned direct-forcing quadrature where the thin section meets the
finite-span end caps. The FVM time step must divide the VPM coupling window
exactly.
