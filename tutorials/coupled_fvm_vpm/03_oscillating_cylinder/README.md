# Heaving cylinder at $Re=150$

A no-slip cylinder heaves across a uniform stream, $y(t)=A\sin(2\pi ft)$, near its natural shedding frequency, and its wake locks onto the motion. The FVM box and its body-fitted mesh move with the cylinder; one layer of vortex particles carries the outer wake. The [reference flow](reference_flow/README.md) solves the same moving-body problem with the FVM alone.

| Quantity | Value |
| --- | --- |
| Diameter $D$; freestream $U_\infty$ | 1 m; 1 m/s in $+x$ |
| Density $\rho$; viscosity $\nu=U_\infty D/Re$ | 1 kg/m³; $1/150$ m²/s |
| Motion | `fvm.HeavingMotion`: $A=0.2D$, $f=0.18$ Hz |
| Span | 0.04 m: one periodic FVM cell; one particle layer between free-slip planes |
| FVM box; transfer region | $[-2.4,4.4]\times[-2.4,2.4]$ m; $[-2.05,4.05]\times[-2.05,2.05]$ m, both relative to the moving cylinder |
| VPM domain; outflow fringe | $[-5,30]\times[-5,5]$ m; $x\in[15,30]$ m, relative to the moving cylinder |
| Cell and particle spacing | 0.04 m |
| FVM step; exchange step | 0.008 s; 0.04 s |
| End time | 50 s |

## Physics and numerics

The mesh, boundary conditions and coupling settings are those of the [fixed cylinder](../01_cylinder_shedding_flow/README.md), with `mesh_motion=fvm.MeshMotion(fvm.HeavingMotion(...))` added to the FVM case. The whole FVM box moves with the cylinder; its `cylinder` wall is a no-slip wall moving with the mesh ([moving meshes](../../../docs/fvm.md#moving-meshes)).

The particles are solved in the frame that translates with the box ([moving bodies](../../../docs/coupling.md#moving-bodies)). A uniform translation of the frame leaves the vorticity equation unchanged, so in this frame the particles see the stream $\mathbf{U}_\infty-\mathbf{V}(t)$ and the box, its walls and the particle diffusion grid stay fixed. The boundary trace the particles impose on the FVM box is converted to the inertial frame, and the FVM vorticity renews the particles exactly as for a fixed body. Particle positions, `domain_bounds` and the particle output are relative to the box at $t=0$; `mesh_translation` in `solution/coupler_diagnostics.jsonl` gives its displacement.

The motion breaks the symmetry, so no startup disturbance is added. The wake locks in within a few periods; statistics are taken from 25 s.

## Run

```bash
(cd reference_flow && ./allrun.sh)
./allrun.sh
./allplot.sh
```

`./allrun.sh` deletes `solution/`, `samples/` and `figures/` and starts at $t=0$; `./allcontinue.sh` resumes from the latest backup. The mesh is kept as `assets/coupled_oscillating_cylinder_mesh.npz`. Install the optional [CUDA Gaussian mesh](../../../docs/installation.md); without it the image sums of `SlipSlabInduction` fall back to a much slower CPU evaluation.

## Results

`figures/forces` compares $C_D$ and $C_L$ with the reference; `./allplot.sh` prints the mean drag, the lift amplitude, the lift frequency relative to the heave frequency and the phase of the lift to the motion over the locked-in wake.
