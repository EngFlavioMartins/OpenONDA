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

A vortex behind the cylinder breaks the symmetry; the freestream is steady from the start. A transverse startup flow is avoided: the zero-pressure outlet of the reference resists a uniform transverse acceleration, so the two cases would receive different startup impulses and shed with a permanent phase offset. Developed shedding is averaged from 40 s.

## Run

```bash
(cd reference_flow && ./allrun.sh)
./allrun.sh
./allplot.sh
```

`./allrun.sh` deletes `solution/`, `samples/` and `figures/` and starts at $t=0$; `./allcontinue.sh` resumes from the latest coupled backup in `solution/coupled/`. The mesh is generated once and kept as `assets/coupled_cylinder_flow_mesh.npz`, with a `.vtu` copy for ParaView. Install the optional [CUDA Gaussian mesh](../../../docs/installation.md): without it, the image sums of `SlipSlabInduction` fall back to a much slower CPU evaluation.

## Results

| File | Content |
| --- | --- |
| `figures/forces` | $C_D$ and $C_L$ against the reference; the mean drag, lift amplitude and Strouhal number of developed shedding are printed |
| `figures/velocity_profiles` | Mean streamwise velocity and transverse velocity fluctuation at $x/D=2$ and $4$ |
| `figures/velocity_profiles/t*` | Streamwise and transverse velocity at $x/D=2$ and $4$ at every sample time shared by both runs, every 0.2 s |
| `figures/velocity_fields/t*`, `figures/vorticity_fields/t*` | Streamwise velocity and spanwise vorticity of the reference, of the coupled flow (the FVM inside its box, dashed, and the particles outside it) and their difference at every time both runs saved fields, every 4 s; instantaneous, so the shed vortices' phases differ once shedding develops. `figures/auxiliary/field_differences.csv` lists the rms and maximum differences |
| `figures/coupling_cost` | Wall time per second of flow time of each exchange by solver part, with the mean of the coupled run and of the reference FVM; the particle count |
| `solution/fvm.pvd`, `solution/vpm.pvd` | FVM fields and particles; every particle frame, written with each coupled backup, has an FVM frame of the same time. `assets/paraview_state.py` loads both in ParaView |

## Sensitivity study

Coupling, resolution and flow parameters are varied about the baseline in study sections; the [coupling-error plan](../../../docs/plans/2026-10-19-coupling-error-sources.md) states what each one isolates:

```bash
assets/run_sensitivity_study.sh       # every section in turn
assets/run_sensitivity_study.sh s4    # one section: convergence, s4, s1b, s2, s3, domain or s1a
```

| Flag of `setup.py` | Physical meaning | Baseline |
| --- | --- | --- |
| `--box-margin` | FVM box extent: $m$ diameters upstream and sideways, $m+2$ downstream | 2.4 |
| `--fvm-substeps` | FVM steps per exchange $n$; the particle step is $n\,\Delta t_f$ | 5 |
| `--transfer-inset` | Cells between the FVM box faces and the transfer-region faces, where FVM vorticity stops replacing the particles | 8.75 |
| `--cell-size` | FVM cell size $h_f$ near the wall [m]; the FVM step and the particle spacing follow it unless given, so refining it keeps the Courant numbers | 0.04 |
| `--fvm-time-step` | FVM step $\Delta t_f$ [s] | $0.2\,h_f/U_\infty$ |
| `--particle-spacing` | Particle spacing $h_p$ [m]; the span, the FVM cell across it and the particle layer between the slip planes follow it | $h_f$ |
| `--interface-iterations`, `--interface-tolerance` | FVM solves per exchange, repeated until the boundary trace changes by less than the tolerance; an unreachable tolerance such as `1e-30` forces exactly that many | 6, $10^{-5}$ |
| `--reynolds-number` | $Re=U_\infty D/\nu$ | 150 |

`reference_flow/setup.py` takes `--cell-size`, `--time-step`, `--reynolds-number` and `--end-time` for the reference variants; its whole mesh scales with the cell size.

| Section | Runs in `sensitivity/` | Varied |
| --- | --- | --- |
| `convergence` | `box2.4_sub2_inset8.75`, `box2.4_sub2_inset8.75_h{0.08,0.056569,0.028284}`, `reference_h{0.08,0.056569,0.028284}` | cells, steps and particles together by $\sqrt2$ at constant Courant number, 2 FVM steps per exchange |
| `s4` | `box2.4_sub5_inset8.75_k{1,2,3,4,6}` | exactly $K$ interface sweeps per exchange |
| `s1b` | `box2.4_sub{10,20,40}_inset8.75_dtf{0.004,0.002,0.001}`, `reference_dt{0.004,0.002}` | FVM step at a fixed particle step of 0.04 s |
| `s2` | `box2.4_sub{2,4,5,6}_inset8.75_hp0.02` | particle spacing $h_f/2$ at $\Delta t_f=0.008$ s |
| `s3` | `box2.4_sub5_inset8.75_re{100,250}`, `reference_re{100,250}` | Reynolds number, baseline coupling |
| `domain` | `box{1.0,1.4,1.8,2.2,2.6}_sub5_inset8.75`, `box2.4_sub5_inset{4,6,14,20}` | FVM box margin; transfer inset |
| `s1a` | `box2.4_sub{5,2,4,6,8}_inset8.75` | FVM steps per exchange at $\Delta t_f=0.008$ s, with the baseline |

A run name is `box{margin}_sub{n}_inset{cells}`, with a suffix only for each other parameter that differs from the baseline: `_h` cell size, `_dtf` FVM step, `_hp` particle spacing, `_k` forced sweeps, `_re` Reynolds number. Each run solves a copy of this case in `sensitivity/<run>/`, and each reference variant a copy of the reference in `sensitivity/reference_<variant>/`, up to 55 s, by which time the lift-amplitude error has reached its developed level; launching a section again resumes its runs from their latest backups. A run that stops, such as the 8-step run at the particle stability limit, does not stop the others. The runs take a few hours each and share the GPU, so they run one after another.

`./allplot.sh` compares every run that reached developed shedding with the reference of its own Reynolds number, cells and FVM step (`reference_flow/` for the baseline ones), over $t\geq40$ s: the absolute relative errors of the mean drag, the lift amplitude and the Strouhal number (2% target dashed), the rms wake velocity difference at $x/D=2$ and $4$ at equal shedding phase, and the wall time per second of flow time. It draws the figures of the sections that have runs:

| Figure | Content |
| --- | --- |
| `figures/sensitivity` | Box margin, FVM steps per exchange and inset, each about its baseline (dotted), with the reference FVM cost dashed |
| `figures/sensitivity_convergence` | `convergence` errors against the cell size, with first- and second-order slopes: the coupled error, the reference's (FVM discretization) against its zero-cell-size extrapolation, and their difference (coupling); the observed orders are printed and written to `figures/auxiliary/convergence.csv` |
| `figures/sensitivity_exchange_ratio` | `s1b` against $n$ and $C_f=U_\infty\Delta t_f/h_f$: errors against the equal-step and the finest-step reference, and each reference against the finest |
| `figures/sensitivity_particle_spacing` | `s2` and `s1a` against $n$ and $C_p=U_\infty\Delta t_p/h_p$, for both particle spacings |
| `figures/sensitivity_interface_sweeps` | `s4` errors, last-sweep change of the boundary trace and cost against $K$; the tolerance-driven baseline at its mean sweep count |
| `figures/sensitivity_reynolds_number` | `s3` errors at $Re=100$, 150 and 250 |
| `figures/flow_scales` | Wake flow scales of the FVM references (`assets/flow_scales.py`): time-mean and rms streamwise velocity on the centreline at $Re=100$, 150 and 250, with the end of the recirculation region $D/2+L_r$ and the formation point $L_f$ (the $u_\mathrm{rms}$ peak) marked. `figures/auxiliary/flow_scales.csv` also lists, for every reference, the shear-layer vorticity thickness at $x=D/2$, the largest wall strain rate $\lvert\omega_z\rvert/2$, the Strouhal number and the wake half-width at $x/D=2$ and 4 |
| `figures/sensitivity_dimensionless` | `s1a`, `s1b` and `s2` errors against $C_f$, $C_p$ and $\Lambda=\max\lvert S\rvert\,\Delta t_p$, with $\max\lvert S\rvert=19.4$ s$^{-1}$ measured at the wall at $Re=150$ and $h_p=h_f$ |

The numbers are printed and written to `figures/auxiliary/sensitivity_runs.csv` and `figures/auxiliary/sensitivity_references.csv`. `./allclean.sh` keeps `sensitivity/`; delete it by hand to start the study again. The results of the study, and the coupling rules drawn from them, are recorded in [coupling_findings.md](assets/coupling_findings.md).
