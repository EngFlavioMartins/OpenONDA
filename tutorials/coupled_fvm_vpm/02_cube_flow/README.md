# Coupled cube flow at $Re=1000$

The FVM resolves separation from a no-slip unit cube and its near wake; vortex particles carry the outer wake. Both solvers use equilibrium Smagorinsky LES. The [reference flow](reference_flow/README.md) is a body-fitted FVM grid study of the same problem.

<p align="center">
  <img src="assets/cube_flow.gif" alt="Coupled FVM and VPM vorticity fields around a cube, with drag coefficient history" width="800">
</p>

| Quantity | Value |
| --- | --- |
| Cube side; freestream | 1 m; 1 m/s in $+x$ |
| Density; viscosity | 1 kg/m³; 0.001 m²/s |
| FVM domain | $[-1.8,5.05]\times[-1.8,1.8]^2$ m |
| VPM domain; outflow fringe | $[-4.5,24]\times[-3,3]^2$ m; $x\in[12,24]$ m |
| Cell and particle spacing | 0.045 m |
| FVM step; coupling interval | 0.01 s; 0.01 s |
| Statistics window | $30\le t\le60$ s |

## Physics and numerical model

`assets/cube.stl` defines the cube for the [body-fitted Cartesian mesh](../../../docs/fvm.md#mesh-setup). Particle boundary data supply normal velocity and the normal gradient of tangential velocity on the outer FVM faces. The FVM then replaces particle vorticity inside the transfer region, with a two-cell buffer and a four-cell blending layer.

The particle wake uses Gaussian cores, RK2 transport, grid-based diffusion and FMM induction. Vorticity fades through the outflow fringe rather than being removed at the downstream face. A tilted vortex behind the cube starts asymmetric shedding in both the coupled and reference cases.

## Run

Run commands from this directory.

```bash
(cd reference_flow && ./allrun.sh)
./allrun.sh
./allplot.sh
```

`./allplot.sh` writes the comparison figures and updates the animation; `./allplot_fields.sh` repeats the field comparison alone. Both sample the saved fine-grid reference onto the coupled mid-plane lattice. `./allrun.sh` starts from $t=0$; `./allcontinue.sh` resumes the latest accepted coupled backup.

## Results

| File | Content |
| --- | --- |
| `figures/forces` | Drag, side force and lift against the reference |
| `figures/velocity_profiles` | Mean streamwise velocity and transverse-velocity fluctuation |
| `figures/velocity_fields`, `figures/vorticity_fields` | Mid-plane reference, coupled and difference fields |
| `figures/coupling_cost` | Wall time by solver part and particle count |
| `assets/cube_flow.gif` | Mid-plane and particle vorticity with drag history |
| `solution/fvm.pvd`, `solution/vpm.pvd` | FVM fields and particle states for ParaView |
