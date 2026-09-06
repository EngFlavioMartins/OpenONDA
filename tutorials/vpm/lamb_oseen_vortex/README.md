# Lamb–Oseen vortex benchmark

This tutorial compares core spreading (CS), a ten-realization Random Walk
Method ensemble (RWM), diffusion velocity (DVH), and Gaussian blob diffusion
(GBD) for an isolated vortex, a counter-rotating dipole, and a co-rotating
merger.

## Run the complete comparison

From any directory, with the OpenONDA environment active:

```bash
/path/to/tutorials/vpm/lamb_oseen_vortex/allrun.sh
```

`allrun.sh` removes previous tutorial-local output, runs every case, and creates
the PNG and PDF figures. It stops if a completed result is unsuitable for the
comparison.

To rebuild the figures from completed samples:

```bash
/path/to/tutorials/vpm/lamb_oseen_vortex/allplot.sh
```

No `PYTHONPATH`, Matplotlib path, Taichi cache path, or repository working
directory needs to be configured. Installed copies can also be managed with
`openonda tutorial create`, `openonda tutorial run`, `openonda tutorial plot`,
and `openonda tutorial clean`.

## Numerical setup

- Particle spacing: `h/a0 = 0.60` (2,077 initial particles for the isolated
  vortex and 3,618 for either pair).
- Particle core radius: `sigma/h = 1.20` for all methods and regeneration.
- Time integration: two-stage, second-order RK2 at the documented timestep.
- CS and RWM induction: exact direct summation.
- DVH and GBD induction: kernel-independent treecode.

FMM remains available in OpenONDA, but is not used for this cross-platform
benchmark because it is unavailable on Metal and its surface evaluation is
currently direct. The treecode works on macOS and Linux and accelerates both
particle stages and sampled fields.

Every method always writes total kinetic energy, measured `dE/dt`, its source,
and the viscous energy rate. Fourier diagnostic transitions are
finite and explicitly labelled. The final check rejects missing or
non-finite energy histories, incomplete RWM ensembles, failed GBD moment
closure, incomplete physical-time coverage, or missing figures.

The statistical definition and uncertainty treatment of RWM are documented in
[assets/references/rwm_statistical_methodology.md](assets/references/rwm_statistical_methodology.md).
