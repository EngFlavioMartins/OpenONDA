# Matched parallel FVM-VPM cylinder LES benchmark

This sibling of `cubeFlow` isolates the transition that matters: the attached
startup flow is followed by a rapidly developing three-dimensional vortex
street. Both solutions use the same Re=500 circular cylinder, direct-forcing
IBM, Smagorinsky LES (`Cs=0.12`), uniform `h/D=0.125`, `dt_FVM=0.025`, 2D span,
initial wake seed, and numerical schemes.
The reference meshes the complete wake domain; the hybrid replaces the domain
outside a compact FVM box with VPM particles.

The fully meshed reference spans `x/D=[-4, 8]`, `y/D=[-3.5, 3.5]`; the VPM
retention box extends the particle wake to `x/D=10`. The 2D extrusion is a
deliberate quasi-2D compromise: this benchmark tests the current 3-D particle
representation against its matched FVM result, not an analytic infinite
vortex line.

Run the fully meshed reference once, then the hybrid and plots:

```sh
cd referenceFlow
./allrun.sh
cd ..
./allrun.sh
./allplot.sh
```

The production horizon is 15 convective time units. Forces are sampled every
0.05, while velocity lines and z=0 surfaces are sampled every 0.5. The reference
retains only three raw volumes (t=0, 7.5, 15); all comparisons and plots use
the much smaller sampler outputs.

Diagnostics include:

- Cd and Cl histories, mean drag, lift RMS, and Strouhal number;
- centerline and y=0.75D streamwise profiles;
- cross-wake profiles at x=D and x=2D;
- reference/FVM/VPM velocity fields and normalized errors;
- particle population, cost, vorticity-flux ratio, and angular-impulse drift;
- a machine-readable split between pre- and post-shedding errors in
  `samples/comparison_metrics.json`.

The quantitative comparison uses `tU/D=5` as the start of the shedding window
(and reports mean drag, lift RMS, Strouhal number, and profile RMS errors). This
fixed split avoids falsely detecting the deliberately seeded startup lift as a
developed vortex street.

Both `allrun.sh` files intentionally accept no solver arguments. Edit their
explicit `OPENONDA_*` blocks to create a version-controlled resolution or time
study.

Both recommended runs use four-rank replicated PETSc. Each MPI rank retains
the complete 86,016-cell reference mesh and IBM marker support, while PETSc
solves the momentum and pressure systems collectively. This is intentionally a
small-memory starting case; larger non-IBM cases should continue using the
default partitioned backend.

The four-rank production-resolution timing gate on the development MacBook Pro
measured approximately 3.2--3.6 seconds per mature reference step. The 600-step
reference therefore projects to roughly 35 minutes before machine-dependent
variation, leaving useful margin below the one-hour target.
