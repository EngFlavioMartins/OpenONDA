# Cylinder shedding at $Re=150$

A no-slip cylinder sheds a planar vortex street. The FVM resolves the wall and the vortex-formation region; one layer of vortex particles carries the outer wake. The [reference flow](reference_flow/README.md) solves the same problem with the FVM alone. The case is also the calibration study of the coupling interface: how much each interface setting changes the answer, and at what cost.

| Quantity | Value |
| --- | --- |
| Diameter $D$; freestream $U_\infty$ | 1 m; 1 m/s in $+x$ |
| Density $\rho$; viscosity $\nu$ | 1 kg/m³; $1/150$ m²/s |
| FVM domain | $[-2.4,4.4]\times[-2.4,2.4]$ m |
| VPM domain; outflow fringe | $[-5,30]\times[-5,5]$ m; $x\in[15,30]$ m |
| Cell and particle spacing | 0.04 m |
| FVM step; coupling interval | 0.008 s; 0.04 s ($N=5$ substeps) |

## Baseline

The defaults of `setup.py` are the baseline of the study.

| Setting | Value |
| --- | --- |
| Cell and particle spacing | $h=0.04D$ |
| FVM step; coupling interval | 0.008 s; 0.04 s ($N=5$) |
| Coupling iterations | at most 6, to the tolerance $10^{-5}$ from a linear first estimate |
| Transfer inset | 8.75 cells |
| FVM box margin | $2.4D$ upstream and sideways, $4.4D$ downstream |
| Particle integration; vorticity floor | RK2; $10^{-2}$ s$^{-1}$ |

```bash
python setup.py --box-margin 2.4 --transfer-inset 8.75 --fvm-substeps 5 --end-time 100 --name box2.4_sub5_inset8.75
```

## Physics and numerical model

The [body-fitted mesh](../../../docs/fvm.md#mesh-setup) is cut from `assets/cylinder_long.stl`. Its outer faces are the `numericalBoundary`: particles provide normal velocity and the normal gradient of tangential velocity through the [mixed boundary condition](../../../docs/coupling.md#boundary-conditions).

Within the transfer region, FVM vorticity replaces particle vorticity at each coupling time. Below the transfer surface, a zero-weight layer of two cells keeps the particle vorticity, and a blending ramp of four cells raises the blending weight to one at $\delta_1=6$ cells, where the replacement region starts. The particles use Gaussian cores, RK2 transport, vector stretching and grid-based diffusion. Free-slip planes bound the single particle layer and reproduce the planar velocity field.

A compact vortex behind the cylinder starts asymmetric shedding.

## Run

Run commands from this directory.

```bash
(cd reference_flow && ./allrun.sh)   # the FVM-only references
./allrun.sh                          # the 18 coupled runs of the study
./allplot.sh                         # statistics, errors and figures
```

`./allrun.sh` resumes each coupled run from its latest coupled backup. Each run writes to `solution/sensitivity_analysis/<name>`.

## The study

Every run holds the baseline and changes the setting of one group. Apart from the baseline, which runs to $tU_\infty/D=100$, every run ends at 55.

| Group | Runs | What it measures |
| --- | --- | --- |
| Baseline | `box2.4_sub5_inset8.75` | the reference point of every group, and the uncertainty of the statistics window |
| T, coupling interval | $N=1$, 2, 3, 6 | the error added at each transfer, from a transfer every FVM step to $N=6$, near the interval at which the particle-strain limit starts to shorten the interval |
| M, coupling iterations | one pass (`_k1`); tolerance $10^{-3}$ (`_tol1e-3`) and $10^{-4}$ (`_tol1e-4`), linear first estimate; tolerance $10^{-5}$, constant first estimate (`_const`) | the error of an unconverged coupling, against the converged-coupling limit |
| D, FVM box margin | 1.6, 2.0, 3.2, 4.0 $D$ | how far from the body the particles can supply the boundary data |
| I, transfer inset | 4, 14 cells | how close to the outer FVM boundary the FVM vorticity can be handed to the particles |
| S, spatial resolution | $h=0.08$ m at $N=3$ and $0.06$ m at $N=4$, against $h=0.04$ m at $N=6$; particle spacing 0.08 m (`_hp0.08`) | the spatial convergence of the coupling error at the fixed coupling interval 0.048 s. The FVM step, particle spacing and interface widths scale with the cells. The last run has particles twice as coarse as the cells |

The names are `box<margin>_sub<N>_inset<cells>`, followed by one suffix for each setting that differs from the baseline.

### Error

Statistics, over whole shedding periods:
- the mean drag $\bar C_D$;
- the Strouhal number $St$, from upward lift crossings;
- the rms lift $C_{L,\mathrm{rms}}$.

The error of a run is the coupling error $e = Q_\mathrm{FVM\text{-}VPM}/Q_\mathrm{FVM}(h,\Delta t_\mathrm{FVM}) - 1$. It is measured against the FVM-only run with the same cells and FVM step ([reference flow](reference_flow/README.md)), so the FVM's own discretization error cancels. That error is measured separately, as the change between the reference grids.

**Blockage.** The FVM-only domain has slip walls at $y=\pm16D$ (blockage $D/H=1/32$); the coupled domain has none. FVM-only values are corrected with the Allen–Vincenti freestream speed-up $U'/U = 1 + \tfrac14 C_D\,D/H + 0.82\,(D/H)^2$:
- forces by $(U/U')^2$;
- $St$, velocities and vorticity by $U/U'$.

### Statistics window and its uncertainty

Runs are compared over $40\le tU_\infty/D\le55$, the window every run covers. The lift amplitude still grows in this window, so a difference between two runs is resolved only when it exceeds the uncertainty of the window (`assets/window_uncertainty.py`). The uncertainty has two terms:
- $\sigma_w$, the spread of a statistic between 15 s windows of the developed flow, slid in steps of 2.5 s from 55 s to the end of each run that reaches 95 s (the coupled baseline and the FVM-only references);
- $s_d$, the spread between those runs of the drift of the 40–55 s value from the developed mean.

An error is a difference of two runs over the same window, so its standard uncertainty is $\sigma=\sqrt{2\sigma_w^2+s_d^2}$. The figures draw $\pm2\sigma$; an error inside that bar is reported as not resolved. The forces and profiles of the baseline use every whole period from 55 s to the end of its run.

## Results

To be filled from the runs of the current transfer. `./allplot.sh` writes to `figures/`:

| File | Content |
| --- | --- |
| `forces`, `velocity_profiles`, `vorticity_profiles` | baseline against the FVM-only reference |
| `error_overview` | error of every run, with the window uncertainty |
| `error_coupling_interval` | group T: error against the coupling interval |
| `error_coupling_iterations` | group M: error and mean iteration count against the tolerance |
| `error_domain` | groups D and I: error against the box margin and the transfer inset |
| `error_cell_size` | group S: error against the cell size and the particle spacing |
| `interface_measures` | boundary mismatch per side, circulation-budget error, near-wall circulation ratio |
| `cost_accuracy` | wall time per flow time against the error |
| `study_summary.csv` | every statistic, error and uncertainty, one row per run |

## Rules for a user

To be drawn from the results above.
