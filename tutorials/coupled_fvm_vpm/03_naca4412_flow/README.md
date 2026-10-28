# Coupled NACA 4412 wing at $10^\circ$ and $Re=1000$

A finite wing with a NACA 4412 section (4% camber at 40% chord, 12% thickness) and flat end caps is pitched $10^\circ$ nose-up about its mid-chord in a uniform stream along $x$. The FVM resolves the no-slip wing, the tips and the near wake; vortex particles carry the trailing vortices and the outer wake. Both solvers use Smagorinsky LES with $C_s=0.17$. Coefficients use the planform area $cb=5$ m². The [reference flow](reference_flow/README.md) solves the same problem with the FVM alone.

| Quantity | Value |
| --- | --- |
| Chord $c$; span $b$ | 1 m, mid-chord on the origin, $10^\circ$ nose-up; 5 m, tips at $z=\pm2.5$ m |
| Freestream $\mathbf{U}_\infty$ | $(1,0,0)$ m/s |
| Density $\rho$; viscosity $\nu=U_\infty c/Re$ | 1 kg/m³; 0.001 m²/s |
| FVM domain | $[-1.6,2.72]\times[-1.28,1.24]\times[-3.6,3.6]$ m, about 1.3 million cells |
| Transfer region | 8.75 particle spacings inside the outer FVM boundary (the coupler default) |
| VPM domain; outflow fringe | $[-3.2,24]\times[-4,3.2]\times[-4.8,4.8]$ m; $x\in[12,24]$ m |
| Cell and particle spacing | 0.04 m, $c/25$ |
| FVM step; coupling interval | 0.008 s; 0.032 s |
| End time; statistics | 24 s; $t\ge12$ s |

## Physics and numerics

`setup.py` writes the wing surface `assets/naca4412_wing.stl` from the analytical NACA four-digit formulas (`assets/naca_surface.py`): the section is extruded over the span and closed by flat caps, every edge is shared by two triangles and every normal points out of the solid. The [body-fitted Cartesian mesh](../../../docs/fvm.md#mesh-setup) is cut from it with uniform spacing; the wing is the no-slip `wing` patch and all outer FVM faces form the `numericalBoundary`, where the particles supply the normal velocity and the normal gradient of the tangential velocity ([mixed boundary condition](../../../docs/coupling.md#boundary-conditions)). Inside the transfer region, FVM vorticity replaces the particle vorticity in every coupling interval ([vorticity transfer](../../../docs/coupling.md#vorticity-transfer)), with a zero-weight layer to $\delta_0=2h$ and a blending ramp to $\delta_1=6h$ below the transfer surface; up to six FVM solves per coupling interval make the boundary data converge. Particles use RK2, Gaussian cores of radius $1.05h$, grid-based diffusion and FMM induction. See [FVM LES](../../../docs/fvm.md#turbulence-and-les) and [VPM diffusion and LES](../../../docs/vpm.md#diffusion-and-les) for the closures.

The layout follows the coupling rules found with the [cylinder](../01_cylinder_shedding_flow/assets/coupling_findings.md):

- **FVM domain.** The FVM domain follows the rules of the [cylinder error study](../../../docs/reports/2026-10-23-coupling-error-sources.md). The replacement region, where FVM vorticity replaces the particle vorticity without blending, starts the transfer inset plus the blend end depth (14.75 cells, 0.59 m) inside the outer FVM boundary. With a shear-layer thickness of about $0.1c$ at $Re=1000$ (the $3.2c/\sqrt{Re}$ scaling the study confirmed on the cylinder and measured as $0.078$ m on the cube), it keeps at least five thicknesses above the wing, below the wake as it descends in the downwash, and beyond each tip. The stream runs along $x$, so it enters only through the upstream face. In an earlier setup, the chord lay along $x$ and the stream crossed the box at $10^\circ$. There, the FVM velocity on the upstream face drifted away from the particle velocity at the spanwise edges, to $-5.4U_\infty$ by $t=3$ s, and the run failed. It reaches $0.51c$ ahead of the leading edge and holds $1.6c$ of wake behind the trailing edge, which contains the roll-up of the tip vortices.
- **Coupling interval.** The coupling error grows with the number of coupling intervals per convective time, so the particle step is long. Four FVM steps of 0.008 s (Courant number 0.2) give 0.032 s, 31 coupling intervals per convective time; the Lagrangian CFL number stays below 0.8 up to a particle-field strain of 25 s⁻¹. Coupling iterations continue until the boundary data changes by less than $10^{-5}$ (the coupler default).
- **Outflow.** The trailing vortices are steady and reach the downstream face. They fade through a 12 m [outflow fringe](../../../docs/vpm.md#stabilization) instead of being cut, so the downwash they induce at the wing stays close to that of semi-infinite vortices, as in the reference.
- **Statistics.** The impulsive start sheds a starting vortex. After 12 s it is 12 chords, about 2.4 spans, downstream; its induced downwash at the wing is then below 1% of the trailing-vortex downwash, and the wake behind the box has been swept through the fringe. Forces, profiles and fields are averaged over $12\le t\le24$ s.

The wing breaks the symmetry in $y$; the flow stays symmetric about mid-span, which the probes and slices sample.

## Run

```bash
(cd reference_flow && ./allrun.sh)
./allrun.sh
./allplot.sh
```

`./allrun.sh` deletes `solution/`, `samples/` and `figures/` and starts at $t=0$; `./allcontinue.sh` resumes from the latest coupled backup in `solution/coupled/`. The mesh is generated once and kept as `assets/coupled_naca4412_flow_mesh.npz`. The FVM runs on eight MPI processes and the particles on the GPU; run the reference after the coupled case, because together they exceed the memory of a 16 GB laptop.

## Results

| File | Content |
| --- | --- |
| `figures/forces` | $C_D$ and $C_L$ against FVM only; their means and rms fluctuations over $t\ge12$ s are printed |
| `figures/velocity_profiles/transverse_mean`, `spanwise_mean` | Time-mean velocity across the mid-span wake and along the span through both tip vortices (at the freestream line through the trailing edge), at $x/c=2$ (coupled FVM and particles) and $5$ (particles) |
| `figures/velocity_profiles/transverse/t*`, `spanwise/t*` | The same profiles at every sample time of the window shared by both runs, every 0.2 s |
| `figures/fields/<plane>_<quantity>/t*`, `.../mean` | Streamwise velocity and plane-normal vorticity of the reference, of the coupled flow (the FVM inside its domain, dashed, and the particles outside it) and their difference on the mid-span plane `z0` and the cross-flow planes `x2` and `x5`, every 0.2 s and time-mean. `figures/auxiliary/field_differences.csv` lists the rms and maximum differences |
| `figures/coupling_cost` | Wall time per second of flow time of each coupling step by solver part, with the mean of the coupled run and of the reference FVM; the particle count |
| `solution/fvm.pvd`, `solution/vpm.pvd` | FVM fields and particles every 2 s over the window |

Compare force statistics and mean profiles while refining the cell and particle spacing, the coupling interval and the box.
