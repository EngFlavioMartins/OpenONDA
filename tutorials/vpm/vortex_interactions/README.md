# Vortex-ring interactions

This tutorial compares three LES treatments of two equal, coaxial vortex rings
undergoing leapfrogging. Every calculation uses the same initial particle
field, molecular diffusion, LES closure, time integration, and velocity
evaluation. The comparison changes only particle splitting and remeshing.

## Reference setup

The physical initial condition follows the `Re_Gamma = 3000` and
`h0/R0 = 1` case of Cheng, Lou, and Lim, [*Leapfrogging of multiple coaxial
viscous vortex rings*](https://doi.org/10.1063/1.4915890), *Physics of Fluids*
27, 031702 (2015).

| Quantity | Value |
|---|---:|
| Ring radius, `R0` | `1.0 m` |
| Gaussian core radius, `a0` | `0.1 R0` |
| Initial ring separation, `h0` | `1.0 R0` |
| Tube circulation, `Gamma0` | `pi m2/s` |
| Circulation Reynolds number, `Gamma0/nu` | `3000` |
| Particle spacing, `h` | `0.035 R0` |
| Initial particle core radius, `sigma0` | `2h = 0.07 R0` |
| Gaussian tail cut-off | `5%` |
| Initial particles | `8772` per ring, `17544` total |

The two ring centres are at `x/R0 = -0.5` and `+0.5`. The arbitrary axial
origin in the digitized LBM trajectory is shifted by `-2.5 R0` when plotted.

The disturbance used for the LBM comparison is represented explicitly as one
sinusoidal centreline mode:

```text
R(theta) = R0 + 0.05 R0 sin(8 theta)
```

Both rings use the same phase. A phase rotation is immaterial for the
unbounded coaxial configuration. The former random phases and broadband
modes 1--24 are not used.

## Common numerical method

All three cases use:

- coupled RK2 for both particle advection and vortex-strength evolution;
- transposed vortex stretching;
- Gaussian particles and Core Spreading molecular diffusion;
- Smagorinsky LES with `Cs = 0.20`;
- a Barnes--Hut treecode with opening angle `0.30`;
- a solver stability-number limit of `1.0`;
- `Delta t = 20 h^2/Gamma0` and 1200 requested steps.

## Armed cases

| Case directory | Splitting | Remeshing |
|---|---:|---:|
| `leapfrog_les` | no | no |
| `leapfrog_les_splitting` | yes | no |
| `leapfrog_les_splitting_remeshing` | yes | yes |

Particle splitting uses an absolute reference to the original cloud. A
particle is split when

```text
|alpha_p| > 2 max_q |alpha_q(t=0)|.
```

The check runs every step. It does not use a lineage-relative strength
factor, so every case has the same fixed physical threshold.

Remeshing is driven only by particle core growth. Gaussian Core Spreading
obeys

```text
sigma(t)^2 = sigma0^2 + 4 nu t.
```

Therefore `sigma = 2 sigma0` after

```text
t_remesh = 3 sigma0^2/(4 nu),
N_remesh = t_remesh/Delta t = 450 steps.
```

The combined case checks at this 450-step cadence. An accepted remesh restores
the initial particle spacing and core radius, so the same analytic cadence
applies again. Divergence and strength-direction misalignment are disabled as
remeshing triggers; the core-radius criterion is the sole trigger.

## Running and output

`allrun.sh` is deliberately explicit:

```sh
./allrun.sh
```

It removes the previous generated `solution/`, `samples/`, and `figures/`
trees, then calls `interactions_setup.py` once for each case. Particle
backups are written only to `solution/<case>/`; sampled diagnostics are
written only to `samples/<case>/`. No `runs/` hierarchy is created. Numerical
instability stops the run in the solver and is reported in that case's log.

To run one setup directly:

```sh
python -u interactions_setup.py --case leapfrog_les_splitting_remeshing
```

Use `allclean.sh CASE_NAME` to remove one result or `allclean.sh --all` to
remove all generated output. Use `allplot.sh` or `allplot.sh pdf` to regenerate
the comparison figures.

The LBM trajectory digitization and provenance are in `assets/references/`.
