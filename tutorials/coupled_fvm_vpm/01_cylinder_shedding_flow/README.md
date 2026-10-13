# Cylinder shedding at $Re=150$

A no-slip cylinder sheds a laminar, planar vortex street. The FVM resolves the wall and the near wake; one layer of three-dimensional vortex particles carries the outer wake. The [reference flow](reference_flow/README.md) solves the same problem with the FVM alone.

| Quantity | Value |
| --- | --- |
| Diameter $D$; freestream $U_\infty$ | 1 m; 1 m/s in $+x$ |
| Density $\rho$; viscosity $\nu=U_\infty D/Re$ | 1 kg/m³; $1/150$ m²/s |
| Span | 0.04 m: one periodic FVM cell; one particle layer between free-slip planes |
| FVM box; transfer region | $[-2.4,4.4]\times[-2.4,2.4]$ m; $[-2.05,4.05]\times[-2.05,2.05]$ m |
| VPM domain; outflow fringe | $[-5,30]\times[-5,5]$ m; $x\in[15,30]$ m |
| Cell and particle spacing | 0.04 m |
| FVM step; exchange step | 0.008 s; 0.04 s |
| End time | 100 s |

## Physics and numerics

The [body-fitted mesh](../../../docs/fvm.md#mesh-setup) is cut from `assets/cylinder_long.stl`, which is longer than the span. Its outer faces form the `numericalBoundary`, where the particles supply the normal velocity and the tangential part of its normal derivative ([mixed boundary condition](../../../docs/coupling.md#boundary-conditions)).

The FVM box contains the vortex-formation region, about four diameters downstream, where the shear layers are still thin. Particles smooth vorticity over their core radius, so a box ending inside that region lowers the lift amplitude by several per cent. Inside the transfer region, FVM vorticity replaces the particles at every exchange ([renewal](../../../docs/coupling.md#vorticity-transfer)). Up to six FVM solves per exchange converge the boundary trace. The particles use the Gaussian kernel, RK2 transport, vector stretching and grid-based diffusion. `SlipSlabInduction` reflects the full vector sources at $z=\pm0.02$ m. A single layer with all its images induces the planar velocity field: image rows beyond a few shells are summed in closed form, and the grid-based diffusion also includes every image.

Shed vortices fade smoothly through an [outflow fringe](../../../docs/vpm.md#induction-and-stretching) of about three wake wavelengths. Deleting them abruptly at the domain face would change the circulation left in the domain every half-cycle. The resulting cross-flow at the cylinder weakens the lift and lengthens the shedding period.

A transverse freestream of 0.1 m/s and a weak vortex behind the cylinder break the symmetry. Between 1 s and 2 s the transverse component tapers smoothly to zero. Developed shedding is averaged from 40 s.

## Run

```bash
(cd reference_flow && ./allrun.sh)
./allrun.sh
./allplot.sh
```

`./allrun.sh` deletes `solution/`, `samples/` and `figures/` and starts at $t=0$; `./allcontinue.sh` resumes from the latest backup. The mesh is generated once and kept as `assets/coupled_cylinder_flow_mesh.npz`, with a `.vtu` copy for ParaView. Install the optional [CUDA Gaussian mesh](../../../docs/installation.md): without it, the image sums of `SlipSlabInduction` fall back to a much slower CPU evaluation.

## Results

| File | Content |
| --- | --- |
| `figures/forces` | $C_D$ and $C_L$ against the reference; the mean drag, lift amplitude and Strouhal number of developed shedding are printed |
| `figures/velocity_profiles` | Mean streamwise velocity and transverse velocity fluctuation at $x/D=2$ and $4$ |
| `figures/coupling_cost` | Wall time of each exchange and the particle count |
| `solution/fvm.pvd`, `solution/vpm.pvd` | FVM fields and particles; `assets/paraview_state.py` loads both in ParaView |

## Sensitivity study

Three coupling parameters are varied one at a time about the baseline:

```bash
./allrun_sensitivity_study.sh
./allplot_sensitivity_study.sh
```

| Flag of `setup.py` | Physical meaning | Baseline | Study values |
| --- | --- | --- | --- |
| `--box-margin` | FVM box extent: $m$ diameters upstream and sideways, $m+2$ downstream | 2.4 | 1.0, 1.4, 1.8, 2.2, 2.6 |
| `--fvm-substeps` | FVM steps of 0.008 s per exchange: how long the particle boundary data are held, and the particle time step | 5 | 2, 4, 6, 8 |
| `--transfer-inset` | Cells between the FVM box faces and the transfer-region faces, where FVM vorticity stops replacing the particles | 8.75 | 4, 6, 14, 20 |

Each run solves a copy of this case in `sensitivity/<run>/` up to 55 s, by which time the lift-amplitude error has reached its developed level; launching the study again resumes every run from its latest backup. The runs take a few hours each and share the GPU, so they run one after another. `figures/sensitivity` shows the absolute relative errors of the mean drag, the lift amplitude and the Strouhal number against the reference over developed shedding, $t\geq40$ s, with the 2% target dashed and each baseline dotted; the numbers are printed. `./allclean.sh` keeps `sensitivity/`; delete it by hand to start the study again. The results of the study, and the coupling rules drawn from them, are recorded in [coupling_findings.md](coupling_findings.md).
