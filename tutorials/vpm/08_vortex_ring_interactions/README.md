# Interacting vortex rings: DNS against LES

Two Gaussian vortex rings at $Re_\Gamma=2\times10^4$, each seeded with a large Widnall disturbance, either collide head-on or leapfrog. Each flow runs twice with the particle numerics of the coupled cases, grid-based diffusion (GBD) with the M4′ kernel: once with molecular viscosity only (DNS) and once with the equilibrium Smagorinsky model and its selective eddy viscosity (LES). The particle spacing resolves the initial cores but not the secondary vortices that the instability creates, so the DNS is under-resolved once the rings break down. The case measures what the sub-grid model then does: where in wavenumber it removes energy, how much, and how long each run lives. See [diffusion and LES](../../../docs/vpm.md#diffusion-and-les).

## Run

From this directory:

```bash
./allrun.sh
./allplot.sh
```

The particles run on a GPU (CUDA, Vulkan or Metal) with `compute_device="auto"`, which never falls back to the CPU; without a GPU the run stops with a message.

`allrun.sh` removes previous results and runs `collision_les`, `collision_dns`, `leapfrog_les` and `leapfrog_dns`. For one run, use `python setup.py collision_les`. `./allcontinue.sh` resumes the saved backups. `./allplot.sh` computes the spectra and velocity-gradient statistics from the backups, then draws the figures and renders the particles.

## Inputs

Edit [setup.py](setup.py). Lengths are in units of $R_0=1$ m and $\Gamma_0=1$ m²/s, so $t\Gamma_0/R_0^2$ is the time in seconds.

| Quantity | Collision | Leapfrogging |
| --- | --- | --- |
| Rings | $+\Gamma_0$ at $x=-1.25R_0$, $-\Gamma_0$ at $x=+1.25R_0$ | $+\Gamma_0$ at $x=\mp0.5R_0$ |
| Gaussian core $a_0$ ($\omega\propto e^{-r^2/a_0^2}$) | $0.1R_0$ | $0.1R_0$ |
| $Re_\Gamma=\Gamma_0/\nu$ | $2\times10^4$ | $2\times10^4$ |
| Disturbance of each centreline | 48 modes, radial and axial, $0.005R_0$ each, random phases | same |
| Spacing $h$; core $\sigma/h$ | $R_0/40$; 1.05 | same |
| Step $\Delta t$; end | 0.0125 s; 30 s | 0.02 s; 40 s |
| Domain | $3\times6.5\times6.5\,R_0^3$ | $7\times4.5\times4.5\,R_0^3$, moving with the pair at $0.26\,\Gamma_0/R_0$ |

The rings are listed as groups 0 and 1. GBD regenerates the particles on its lattice every step and gives each new particle the group that holds most of the strength at its node, so the colours of the renders follow how the rings exchange vorticity.
