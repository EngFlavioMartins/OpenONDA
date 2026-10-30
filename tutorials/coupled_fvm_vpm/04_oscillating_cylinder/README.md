# Heaving cylinder at $Re=150$

A no-slip cylinder heaves across a uniform stream, $y(t)=A\sin(2\pi f t)$, near its natural shedding frequency. The FVM box and mesh move with the cylinder; one particle layer carries the outer wake. The [reference flow](reference_flow/README.md) solves the same moving-body problem with the FVM alone.

| Quantity | Value |
| --- | --- |
| Diameter; freestream | 1 m; 1 m/s in $+x$ |
| Density; viscosity | 1 kg/m³; $1/150$ m²/s |
| Heave motion | $A=0.2D$; $f=0.18$ Hz |
| FVM domain | $[-2.4,4.4]\times[-2.4,2.4]$ m, relative to the cylinder |
| Cell and particle spacing | 0.04 m |
| FVM step; coupling interval | 0.008 s; 0.04 s |
| Statistics window | $25\le t\le50$ s |

## Physics and numerical model

The mesh, transfer region and particle model follow the [fixed cylinder](../01_cylinder_shedding_flow/README.md). The FVM mesh and particle frame translate with the cylinder. In that frame, the particles see the relative stream $\mathbf{U}_\infty-\mathbf{V}(t)$ while the particle domain and diffusion grid remain fixed.

The motion sets the shedding phase, so fields and profiles are compared at equal heave phase in the inertial frame. Particle output is stored relative to the translating box; the plotting scripts restore the physical position and velocity.

## Run

Run commands from this directory.

```bash
(cd reference_flow && ./allrun.sh)
./allrun.sh
./allplot.sh
```

`./allrun.sh` starts from $t=0$; `./allcontinue.sh` resumes the latest accepted coupled backup.

## Results

| File | Content |
| --- | --- |
| `figures/forces` | Drag, lift amplitude, frequency ratio and lift phase |
| `figures/velocity_profiles` | Mean and phase-averaged transverse wake profiles |
| `figures/velocity_fields`, `figures/vorticity_fields` | Phase-averaged reference, coupled and difference fields |
| `figures/coupling_cost` | Wall time by solver part and particle count |
| `solution/fvm.pvd`, `solution/vpm.pvd` | Moving FVM fields and particle states |
