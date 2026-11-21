# Interacting vortex rings: DNS against two LES closures

Two Gaussian vortex rings at $Re_\Gamma=2\times10^4$, each seeded with a large Widnall disturbance, either collide head-on or leapfrog. Both flows use the particle numerics of the coupled cases, grid-based diffusion (GBD) with the M4′ kernel. Each flow runs three times:

| Run | Closure |
| --- | --- |
| `dns` | Molecular viscosity only. |
| `les` | Equilibrium Smagorinsky in the GBD Laplacian, with the selective eddy viscosity it switches on, as in the coupled cases. |
| `les_cusp` | Spectral eddy viscosity with the Kraichnan cusp, applied to the GBD lattice in Fourier space ([LES](../../../docs/vpm.md#diffusion-and-les)). |

The particle spacing resolves the initial cores but not the secondary vortices that the instability creates, so the DNS is under-resolved once the rings break down. The case measures what each closure then does: where in wavenumber it removes energy, how much, and how long each run lives.

## Run

From this directory:

```bash
./allrun.sh
./allplot.sh
```

The particles run on a GPU (CUDA, Vulkan or Metal) with `compute_device="auto"`, which never falls back to the CPU; without a GPU the run stops with a message.

`allrun.sh` removes previous results and runs the six cases `<flow>_<closure>`, with flows `collision` and `leapfrog`. For one run, use `python setup.py collision_les_cusp`. `./allcontinue.sh` resumes the saved backups. `./allplot.sh` computes the spectra, the energy budget per wavenumber and the velocity-gradient statistics from the backups, then draws the figures and renders the particles.

## Inputs

Edit [setup.py](setup.py). Lengths are in units of $R_0=1$ m and $\Gamma_0=1$ m²/s, so $t\Gamma_0/R_0^2$ is the time in seconds.

| Quantity | Collision | Leapfrogging |
| --- | --- | --- |
| Rings | $+\Gamma_0$ at $x=-1.25R_0$, $-\Gamma_0$ at $x=+1.25R_0$ | $+\Gamma_0$ at $x=\mp0.5R_0$ |
| Gaussian core $a_0$ ($\omega\propto e^{-r^2/a_0^2}$) | $0.1R_0$ | $0.1R_0$ |
| $Re_\Gamma=\Gamma_0/\nu$ | $2\times10^4$ | $2\times10^4$ |
| Disturbance of each centreline | 48 modes, radial and axial, random phases: $0.005R_0$ up to $m=8$, $0.005R_0(8/m)^2$ above | same |
| Spacing $h$; core $\sigma/h$ | $R_0/40$; 1.05 | same |
| Step $\Delta t$; end | 0.0125 s; 30 s | 0.02 s; 40 s |
| Domain | $3\times6.5\times6.5\,R_0^3$ | $7\times4.5\times4.5\,R_0^3$, moving with the pair at $0.26\,\Gamma_0/R_0$ |

Above $m=8$ every mode bends the centreline as much as $m=8$ does. The radius of curvature stays above $1.7a_0$, and the largest displacement is $0.034R_0$, a third of the core. An equal amplitude up to $m=48$ would bend the centreline more tightly than the core and fold it, and in the DNS the folded cores create kinetic energy.

The rings are listed as groups 0 and 1. GBD regenerates the particles on its lattice every step and gives each new particle the group that holds most of the strength at its node, so the colours of the renders follow how the rings exchange vorticity.

A run stops when the step no longer resolves the velocity gradient ($|\nabla u|\Delta t>1$) or the particles have lost their vortex lines (vorticity-divergence error above 0.5; GBD holds it near 0.13). The stopping time is the run's lifetime.

## Energy budget

`assets/energy_transfer.py` splits the change of the energy spectrum of the particle field,

$$
\frac{\mathrm{d}E(k)}{\mathrm{d}t}=T(k)-D_\mathrm{mol}(k)-D_\mathrm{sgs}(k)-D_\mathrm{sel}(k)-D_\mathrm{num}(k).
$$

- $T(k)$ is the resolved non-linear transfer, which integrates to zero.
- $D_\mathrm{mol}$ and $D_\mathrm{sgs}$ are the energy removed by the molecular viscosity and by the closure, through the solver's own operators.
- $D_\mathrm{sel}$ is the selective eddy viscosity of `les`. The backups do not store it, so it is reconstructed from the resolved field.
- $D_\mathrm{num}$ is the remainder: what the regeneration on the lattice and the thresholding remove.

Each term is a shell density in $k$; its integral over $k$ is the total rate.
