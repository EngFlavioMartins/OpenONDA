# Lamb–Oseen vortex benchmark

This tutorial compares core spreading (CS), an adaptive Random Walk
Method ensemble (RWM), discrete vortex heat diffusion (DVH), and Gaussian blob diffusion
(GBD) for an isolated vortex, a counter-rotating dipole, and a co-rotating
merger.

## Run the complete comparison

From this case directory, with the installed OpenONDA environment active:

```bash
./allrun.sh
./allplot.sh
```

`allrun.sh` lists the simulations only. Use `python setup.py vortex CS` to run
one physical/method variant. `./allclean.sh` explicitly removes generated
outputs before a fresh comparison when desired.

RWM starts with ten independent seeds and adds batches until the maximum
velocity and vorticity relative standard errors are both at most 7.5%, up to
80 seeds. The ensemble's statistical stopping rule is part of the experiment.
Validation is separate: run `python assets/postprocess.py` after plotting.

`allplot.sh` extracts the required fields and produces PNG and PDF figures.
This also creates `mergingRenderT0` and `mergingRenderFinal` in PDF and PNG
formats from the initial conditions and the final GBD particle backup in
`solution/merging_gbd/`. Keep that backup and its sample metadata when
reproducing the sphere views. Their camera and field of view are shared;
colour clipping and strength-based sphere sizes are normalised per frame.
The lower-right label shows `nu t / a_c,0^2`. The CPU renderer requires
Numba and TeX Live with NewPX/Pagella fonts and takes a few minutes.

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
- Stretching: transposed for every viscous method.

In `setup.py`, `COMPUTE_METHOD` selects the backend separately from
`STRETCHING_SCHEME`, which accepts `"direct"`, `"mixed"`, or `"transposed"`.
The factory calls, for example, `vpm.TreecodeInduction(stretching_scheme="transposed")`.
Both choices are recorded by the solver in `solution/<case>/vpm_metadata.json`.
Clean the previous outputs before comparing newly edited configurations.

FMM remains available in OpenONDA, but is not used for this cross-platform
benchmark because it is unavailable on Metal and its surface evaluation is
currently direct. The treecode works on macOS and Linux and accelerates both
particle stages and sampled fields.

Every method writes total kinetic energy, measured `dE/dt`, its source,
and the viscous energy rate. Uniform-core DVH/GBD clouds use a zero-padded
FFT convolution with the unbounded Gaussian Green tensor, including the
far-field energy of an open vortex column. Energy is not taken from a
periodic inverse Laplacian. DVH energy samples span at least one complete
heat-transfer interval (36 steps for vortex/dipole, 30 for merging); surface
fields retain their original cadence. The explicit postprocessor check rejects missing or
non-finite energy histories, incomplete RWM ensembles, failed GBD moment
closure, incomplete physical-time coverage, or missing figures.

The statistical definition and uncertainty treatment of RWM are documented in
[assets/references/rwm_statistical_methodology.md](assets/references/rwm_statistical_methodology.md).
