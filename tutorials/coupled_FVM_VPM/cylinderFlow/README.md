# Matched FVM-VPM cylinder shedding benchmark

This sibling of `cubeFlow` isolates the transition that matters: the attached
startup flow is followed by a simple, laminar, periodic vortex street. Both
solutions use the same Re=100 circular cylinder, direct-forcing IBM, uniform
`h/D=0.10`, `dt_FVM=0.025`, 2.4D span, initial wake seed, and numerical schemes.
The reference meshes the complete wake domain; the hybrid replaces the domain
outside a compact FVM box with VPM particles.

The matched reference/VPM far field spans `x/D=[-4, 10.4]` and `y/D=[-4, 4]`.
This keeps blockage modest without turning the reference into the dominant
cost. The 2.4D extrusion is a deliberate quasi-2D compromise: this benchmark
tests the current 3-D particle representation against its matched FVM result,
not an analytic infinite vortex line.

Run the fully meshed reference once, then the hybrid and plots:

```sh
cd referenceFlow
./allrun.sh
cd ..
./allrun.sh
./allplot.sh
```

The production horizon is 20 convective time units. Forces are sampled every
0.1, while velocity lines and z=0 surfaces are sampled every 1.0. The reference
retains only three raw volumes (t=0, 10, 20); all comparisons and plots use
the much smaller sampler outputs.

Diagnostics include:

- Cd and Cl histories, mean drag, lift RMS, and Strouhal number;
- centerline and y=0.75D streamwise profiles;
- cross-wake profiles at x=D and x=2D;
- reference/FVM/VPM velocity fields and normalized errors;
- particle population, cost, vorticity-flux ratio, and angular-impulse drift;
- a machine-readable split between pre- and post-shedding errors in
  `samples/comparison_metrics.json`.

The quantitative comparison uses `tU/D=8` as the start of the shedding window
(and reports mean drag, lift RMS, Strouhal number, and profile RMS errors). This
fixed split avoids falsely detecting the deliberately seeded startup lift as a
developed vortex street.

Both `allrun.sh` files intentionally accept no solver arguments. Edit their
explicit `OPENONDA_*` blocks to create a version-controlled resolution or time
study.

The FVM is deliberately run with one rank. The current direct-forcing IBM
operator does not exchange marker support across MPI partitions; using more
than one FVM rank gives empty-support markers during initialization.
