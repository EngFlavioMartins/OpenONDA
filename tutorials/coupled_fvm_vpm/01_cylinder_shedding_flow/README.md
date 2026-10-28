# Cylinder shedding at $Re=150$

A no-slip cylinder sheds a laminar, planar vortex street. The FVM resolves the wall and the near wake; one layer of three-dimensional vortex particles carries the outer wake. The [reference flow](reference_flow/README.md) solves the same problem with the FVM alone.

| Quantity | Value |
| --- | --- |
| Diameter $D$; freestream $U_\infty$ | 1 m; 1 m/s in $+x$ |
| Density $\rho$; viscosity $\nu=U_\infty D/Re$ | 1 kg/m³; $1/150$ m²/s |
| Span | 0.04 m: one periodic FVM cell; one particle layer between free-slip planes |
| FVM domain; transfer region | $[-2.4,4.4]\times[-2.4,2.4]$ m; $[-2.05,4.05]\times[-2.05,2.05]$ m |
| VPM domain; outflow fringe | $[-5,30]\times[-5,5]$ m; $x\in[15,30]$ m |
| Cell and particle spacing | 0.04 m |
| FVM step; coupling interval | 0.008 s; 0.04 s |
| End time | 100 s |

## Physics and numerics

The [body-fitted mesh](../../../docs/fvm.md#mesh-setup) is cut from `assets/cylinder_long.stl`, which is longer than the span. Its outer faces form the `numericalBoundary`, where the particles supply the normal velocity and the normal gradient of the tangential velocity ([mixed boundary condition](../../../docs/coupling.md#boundary-conditions)).

The FVM domain contains the vortex-formation region, about four diameters downstream, where the shear layers are still thin. Particles smooth vorticity over their core radius, so a box ending inside that region lowers the lift amplitude by several per cent. Inside the transfer region, FVM vorticity replaces the particle vorticity in every coupling interval ([vorticity transfer](../../../docs/coupling.md#vorticity-transfer)). Up to six FVM solves per coupling interval make the boundary data converge. The particles use the Gaussian kernel, RK2 transport, vector stretching and grid-based diffusion. `SlipSlabInduction` reflects the full vector sources at $z=\pm0.02$ m. A single layer with all its images induces the planar velocity field: image rows beyond a few shells are summed in closed form, and the grid-based diffusion also includes every image.

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
| `figures/velocity_fields/t*`, `figures/vorticity_fields/t*` | Streamwise velocity and spanwise vorticity of the reference, of the coupled flow (the FVM inside its domain, dashed, and the particles outside it) and their difference at every time both runs saved fields, every 4 s; instantaneous, so the shed vortices' phases differ once shedding develops. `figures/auxiliary/field_differences.csv` lists the rms and maximum differences |
| `figures/coupling_cost` | Wall time per second of flow time of each coupling step by solver part, with the mean of the coupled run and of the reference FVM; the particle count |
| `figures/cylinder_shedding.gif` | Four shedding periods from $t=75.6$ s: vorticity magnitude of the FVM mid-plane and of the particles outside the FVM domain, with $C_L$ and $C_D$; the frames use the 0.2 s field output of developed shedding and the animation loops |
| `assets/cylinder_wake` | The same scene on white for print at $t=86$ s, 125 mm wide, with freestream arrows: `cylinder_wake_raw.png` is the text-free render and `cylinder_wake.tex` typesets its labels and colour bar in the thesis fonts (newpx) |
| `solution/fvm.pvd`, `solution/vpm.pvd` | FVM fields and particles; every particle frame, written with each coupled backup, has an FVM frame of the same time. `assets/paraview_state.py` loads both in ParaView |

## Sensitivity study

Coupling, resolution and flow parameters are varied about the baseline in study sections; the [coupling-error plan](../../../docs/plans/2026-10-19-coupling-error-sources.md) states what each one isolates:

```bash
assets/run_sensitivity_study.sh       # every section in turn
assets/run_sensitivity_study.sh s4    # one section: convergence, s4, s1b, s2, s3, domain or s1a
```

| Flag of `setup.py` | Physical meaning | Baseline |
| --- | --- | --- |
| `--box-margin` | FVM domain extent: $m$ diameters upstream and sideways, $m+2$ downstream | 2.4 |
| `--fvm-substeps` | FVM substeps per coupling interval $N$; the particle step is $N\,\Delta t_{\rm FVM}$ | 5 |
| `--transfer-inset` | Cells between the outer FVM boundary and the transfer surface, where FVM vorticity stops replacing the particle vorticity | 8.75 |
| `--cell-size` | FVM cell size $h_{\rm FVM}$ near the wall [m]; the FVM step and the particle spacing follow it unless given, so refining it keeps the Courant numbers | 0.04 |
| `--fvm-time-step` | FVM step $\Delta t_{\rm FVM}$ [s] | $0.2\,h_{\rm FVM}/U_\infty$ |
| `--particle-spacing` | Particle spacing $h_{\rm VPM}$ [m]; the span, the FVM cell across it and the particle layer between the slip planes follow it | $h_{\rm FVM}$ |
| `--coupling-iterations`, `--coupling-tolerance` | FVM solves per coupling interval, repeated until the boundary data changes by less than the tolerance; an unreachable tolerance such as `1e-30` forces exactly that many | 6, $10^{-5}$ |
| `--reynolds-number` | $Re=U_\infty D/\nu$ | 150 |

`reference_flow/setup.py` takes `--cell-size`, `--time-step`, `--reynolds-number` and `--end-time` for the reference variants; its whole mesh scales with the cell size.

| Section | Runs in `sensitivity/` | Varied |
| --- | --- | --- |
| `convergence` | `box2.4_sub2_inset8.75`, `box2.4_sub2_inset8.75_h{0.08,0.056569,0.028284}`, `reference_h{0.08,0.056569,0.028284}` | cells, steps and particles together by $\sqrt2$ at constant Courant number, 2 FVM substeps per coupling interval |
| `s4` | `box2.4_sub5_inset8.75_k{1,2,3,4,6}` | exactly $M$ coupling iterations per coupling interval |
| `s1b` | `box2.4_sub{10,20,40}_inset8.75_dtf{0.004,0.002,0.001}`, `reference_dt{0.004,0.002}` | FVM step at a fixed particle step of 0.04 s |
| `s2` | `box2.4_sub{2,4,5,6}_inset8.75_hp0.02` | particle spacing $h_{\rm FVM}/2$ at $\Delta t_{\rm FVM}=0.008$ s |
| `s3` | `box2.4_sub5_inset8.75_re{100,250}`, `reference_re{100,250}` | Reynolds number, baseline coupling |
| `domain` | `box{1.0,1.4,1.8,2.2,2.6}_sub5_inset8.75`, `box2.4_sub5_inset{4,6,14,20}` | FVM domain margin; transfer inset |
| `s1a` | `box2.4_sub{5,2,4,6,8}_inset8.75` | FVM substeps per coupling interval at $\Delta t_{\rm FVM}=0.008$ s, with the baseline |

A run name is `box{margin}_sub{n}_inset{cells}`, with a suffix only for each other parameter that differs from the baseline: `_xd` downstream outer FVM boundary [D] (default margin + 2), `_h` cell size, `_dtf` FVM step, `_hp` particle spacing, `_k` forced coupling iterations, `_re` Reynolds number. Each run solves a copy of this case in `sensitivity/<run>/`, and each reference variant a copy of the reference in `sensitivity/reference_<variant>/`, up to 55 s, by which time the lift-amplitude error has reached its developed level; launching a section again resumes its runs from their latest backups. A run that stops, such as the 8-step run at the particle stability limit, does not stop the others. The runs take a few hours each and share the GPU, so they run one after another.

`./allplot.sh` compares every run that reached developed shedding with the reference of its own Reynolds number, cells and FVM step (`reference_flow/` for the baseline ones), over $t\geq40$ s: the absolute relative errors of the mean drag, the lift amplitude and the Strouhal number (2% target dashed), the rms wake velocity difference at $x/D=2$ and $4$ at equal shedding phase, and the wall time per second of flow time. It draws the figures of the sections that have runs:

| Figure | Content |
| --- | --- |
| `figures/sensitivity` | Box margin, FVM substeps per coupling interval and inset, each about its baseline (dotted), with the reference FVM cost dashed |
| `figures/error_references_C_X_T` | `assets/plot_error_analysis.py`: reference error against $h/D$ and $U\Delta t/D$ in families C, X and T, against $Q_{\rm ext}$ extrapolated to zero cell size and step, with the least-squares fits and uncertainties $\pm U_{\rm ext}$ of Eça and Hoekstra (J. Comput. Phys. 262, 2014); each family is classed monotone, oscillatory or anomalous and an order is given for monotone data only (`figures/auxiliary/error_orders.csv`, and `error_space_time.csv` for $Q=Q_{\rm ext}+a h^p+b\Delta t^q$ over all references once six have run) |
| `figures/error_decomposition_C_X_T2` | Total coupled error $e=Q-Q_{\rm ext}$, FVM error $e_{\rm FVM}$ of the reference on the same cells and step, and coupling error $e_{\rm coupling}=e-e_{\rm FVM}$, against the refinement variable of families C, X and T2, with fits (`figures/auxiliary/error_decomposition.csv`) |
| `figures/error_drivers_T2_P_E` | $e_{\rm coupling}$ against the particle step $U\Delta t_{\rm VPM}/D$ (with $\Lambda=\max\lvert S\rvert\Delta t_{\rm VPM}$ from the particle-field $\max\lvert S\rvert=19.4$ s$^{-1}$ at $Re=150$, scaled with $\sqrt{Re}$), the coupling intervals per convective time $D/(U\Delta t_{\rm VPM})$ and $N$; the fits over T2, P and E and a model weighing a temporal term against an accumulated per-coupling-interval term are printed (`figures/auxiliary/error_drivers.csv`) |
| `figures/error_indicator` | $\lvert e_{\rm coupling}\rvert$ of every run against the developed means of the boundary-data mismatch and first coupling-iteration residual the coupler logs, with their correlation (`figures/auxiliary/error_indicators.csv`): an in-run indicator that needs no reference |
| `figures/error_box_B` | $e_{\rm coupling}$ of family B against the wake reach $\ell_{\rm wake}/L_f=(x_t-L_f)/L_f$, with $L_f$ at the $u_\mathrm{rms}$ or the $v_\mathrm{rms}$ peak, and the lateral clearance $\ell_{\rm side}/\delta_\omega=(m-w-D/2)/\delta_\omega$, from `figures/auxiliary/flow_scales.csv`; the main effects, interactions and additivity of the F factorial are printed |
| `figures/sensitivity_fvm_substeps` | `s1b` against $N$ and $C_{\rm FVM}=U_\infty\Delta t_{\rm FVM}/h_{\rm FVM}$: errors against the equal-step and the finest-step reference, and each reference against the finest |
| `figures/sensitivity_particle_spacing` | `s2` and `s1a` against $N$ and $C_{\rm VPM}=U_\infty\Delta t_{\rm VPM}/h_{\rm VPM}$, for both particle spacings |
| `figures/sensitivity_coupling_iterations` | `s4` errors, last coupling-iteration residual $r_{g_t}$ and cost against $M$; the tolerance-driven baseline at its mean iteration count |
| `figures/sensitivity_reynolds_number` | `s3` errors at $Re=100$, 150 and 250 |
| `figures/flow_scales` | Wake flow scales of the FVM references (`assets/flow_scales.py`): time-mean and rms streamwise velocity on the centreline at $Re=100$, 150 and 250, with the end of the recirculation region $D/2+L_r$ and the formation point $L_f$ (the $u_\mathrm{rms}$ peak) marked. `figures/auxiliary/flow_scales.csv` also lists, for every reference, the shear-layer vorticity thickness at $x=D/2$, the largest wall strain rate $\lvert\omega_z\rvert/2$, the Strouhal number and the wake half-width at $x/D=2$ and 4 |
| `figures/sensitivity_dimensionless` | `s1a`, `s1b` and `s2` errors against $C_{\rm FVM}$, $C_{\rm VPM}$ and $\Lambda=\max\lvert S\rvert\,\Delta t_{\rm VPM}$, with $\max\lvert S\rvert=19.4$ s$^{-1}$ measured at the wall at $Re=150$ and $h_{\rm VPM}=h_{\rm FVM}$ |

The numbers are printed and written to `figures/auxiliary/sensitivity_runs.csv` and `figures/auxiliary/sensitivity_references.csv`. `./allclean.sh` keeps `sensitivity/`; delete it by hand to start the study again. The results of the study, and the coupling rules drawn from them, are recorded in [coupling_findings.md](assets/coupling_findings.md).
