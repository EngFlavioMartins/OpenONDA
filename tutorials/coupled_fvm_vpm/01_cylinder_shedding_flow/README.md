# Cylinder shedding at $Re=150$

A no-slip cylinder sheds a planar vortex street. The FVM resolves the wall and the vortex-formation region; one layer of vortex particles carries the outer wake. The [reference flow](reference_flow/README.md) solves the same problem with the FVM alone. The case is also the sensitivity study of the coupling interface: which interface settings change the answer, by how much, and at what cost.

| Quantity | Value |
| --- | --- |
| Diameter $D$; freestream $U_\infty$ | 1 m; 1 m/s in $+x$ |
| Density $\rho$; viscosity $\nu$ | 1 kg/m³; $1/150$ m²/s |
| FVM domain | $[-2.4,4.4]\times[-2.4,2.4]$ m |
| VPM domain; outflow fringe | $[-5,30]\times[-5,5]$ m; $x\in[15,30]$ m |
| Cell and particle spacing | 0.04 m |
| FVM step; coupling interval | 0.008 s; 0.04 s ($N=5$ substeps) |

## Recommended setup

| Setting | Value |
| --- | --- |
| Cell and particle spacing | $h=0.04D$ |
| FVM step; coupling interval | 0.008 s; 0.04 s ($N=5$, particle CFL number 0.78) |
| Coupling iterations | to the tolerance $10^{-5}$ |
| Strength limit; transfer inset | $\kappa_\mathrm{max}=1.8$; 8.75 cells |
| FVM box margin | $2.4D$ |
| Particle integration; vorticity floor | RK2; $10^{-2}$ s$^{-1}$ |

```bash
python setup.py --box-margin 2.4 --transfer-inset 8.75 --fvm-substeps 5 --end-time 100 --name box2.4_sub5_inset8.75
```

The forces, profiles and field differences in `figures/` come from this run. The basis for each value is in "Rules for a user" below.

## Physics and numerical model

The [body-fitted mesh](../../../docs/fvm.md#mesh-setup) is cut from `assets/cylinder_long.stl`. Its outer faces are the `numericalBoundary`: particles provide normal velocity and the normal gradient of tangential velocity through the [mixed boundary condition](../../../docs/coupling.md#boundary-conditions).

Within the transfer region, FVM vorticity replaces particle vorticity at each coupling time. Below the transfer surface, a zero-weight layer of two cells keeps the particle vorticity, and a blending ramp of four cells raises the blending weight to one at $\delta_1=6$ cells, where the replacement region starts. The particles use Gaussian cores, RK2 transport, vector stretching and grid-based diffusion. Free-slip planes bound the single particle layer and reproduce the planar velocity field.

A compact vortex behind the cylinder starts asymmetric shedding.

## Run

Run commands from this directory.

```bash
(cd reference_flow && ./allrun.sh)   # four FVM-only grids, h = 0.08, 0.06, 0.04, 0.03 m, to 100 s
./allrun.sh                          # the 23 coupled runs, to 55 s (recommended setup to 100 s)
./allplot.sh                         # statistics and figures
```

Every run imports the frozen solver export of commit `3f771630` (campaign of 2 to 8 November 2026). `assets/plot_study.py` is the only analysis script.

## How the error is measured

- **Statistics**: mean drag $\bar C_D$, lift amplitude $\hat C_L$ (extremes at the vertex of a parabola through three samples) and Strouhal number $St$ (upward lift crossings), over whole shedding periods.
- **Windows**: the comparison between runs uses $40\le tU_\infty/D\le55$, the window every run covers. The forces and profiles of the recommended setup use every whole period from $tU_\infty/D=55$, where the lift amplitude becomes steady, to the end of the run.
- **Resolution**: the lift amplitude still grows between 40 and 55. The baseline's 40–55 values differ from its steady values by 0.09 % ($\bar C_D$), 0.25 % ($\hat C_L$) and 0.05 % ($St$). Smaller differences between runs are reported as unresolved and drawn inside the grey band.
- **Coupling error**: $e = Q_\mathrm{FVM\text{-}VPM}/Q_\mathrm{FVM}(h,\Delta t_\mathrm{FVM}) - 1$, against the FVM run with the same cells and step. The FVM's own discretization error is measured separately, as the change from the finest grid.
- **Blockage**: the FVM domain has slip walls at $y=\pm16D$ (blockage 3.1 %); the coupled domain has none. FVM results are corrected with the Allen–Vincenti freestream speed-up $U'/U = 1 + \tfrac14 C_D\,D/H + 0.82\,(D/H)^2$: forces by $(U/U')^2$ (−2.2 %), $St$, velocities and vorticity by $U/U'$ (−1.1 %). Without the correction the coupled drag reads 2.2 % low against the FVM.

## Results

The forces and profiles come from the recommended setup. The $N=6$ run has a slightly smaller error over 40–55 (lift amplitude −1.12 % against −1.25 %), within the resolution, at a particle CFL number of 0.93, close to the limit of 1 at which the solver stops.

| File | Content |
| --- | --- |
| `figures/forces.png` | Drag and lift of the recommended setup and of the FVM ($h=0.04$ m); dotted lines are the steady means |
| `figures/velocity_profiles.png` | Mean and rms streamwise velocity at $x=2D$ and $4D$, with the FVM-VPM minus FVM difference |
| `figures/vorticity_profiles.png` | Mean and rms vorticity at $x=2D$ and $4D$, with the difference |
| `figures/error_convergence.png` | FVM grid error and coupling error against the cell size |
| `figures/error_coupling_interval.png` | Coupling error and boundary mismatch against the coupling interval |
| `figures/error_sensitivity.png` | Change of each statistic from the baseline, one setting at a time |
| `figures/study_summary.csv` | Every statistic, error, mean coupling-iteration count and boundary mismatch |

### Recommended setup, steady periods

Seven periods, $tU_\infty/D$ from 59 to 98.

| | $\bar C_D$ | $\hat C_L$ | $St$ |
| --- | ---: | ---: | ---: |
| FVM, $h=0.04$ m | 1.3101 | 0.5018 | 0.1829 |
| FVM-VPM | 1.3103 | 0.4958 | 0.1823 |
| Coupling error | +0.01 % | −1.19 % | −0.37 % |
| FVM grid error, $h=0.04$ against $0.03$ m (40–55) | +0.14 % | −1.42 % | −0.25 % |

The coupling adds no measurable drag error. Its lift-amplitude error is about the FVM's own error at this cell size. The FVM lift amplitude and Strouhal number are still changing at $h=0.03$ m, so no grid-converged value is quoted for them.

At $x=2D$, inside the FVM domain, the profiles differ by less than 0.7 % of $U_\infty$ in mean velocity and 0.03 $U_\infty/D$ in vorticity. At $x=4D$ the particle wake has a 1.2 % lower centreline velocity and a nearly unchanged rms; the coupled wake recovers slightly more slowly.

### Effect of each setting (coupling error of $\hat C_L$ unless stated)

| Setting | Levels | Effect | Cost |
| --- | --- | --- | --- |
| Coupling interval $\Delta t_\mathrm{VPM}$ | 0.008–0.048 s ($N=1$–6) | −7.4 % at 0.008 s, −2.6 % at 0.016 s, −1.8 % at 0.024 s, −1.25 % at 0.040 s, −1.1 % at 0.048 s. Drag −2.6 % to +0.1 %, $St$ −3.6 % to −0.3 %. The error decreases as the interval grows | Particle steps per $D/U_\infty$: 125 at $N=1$, 25 at $N=5$ |
| Strength limit $\kappa_\mathrm{max}$ | 1.8 and 3, at $N=1,3,4,6$ | None; at $N=1$ the two differ by 0.3 points | None |
| Coupling iterations $M$ | 1, 2, 4, tolerance $10^{-5}$ | $M=1$: drag +3.2 %, lift +2.3 % (+3.6 points from the tolerance run, of opposite sign). $M\ge2$ matches the tolerance run within the resolution | The tolerance needs 1.7 iterations on average |
| FVM box margin | 1.6–4.0 $D$ | Within ±1.5 %, without a trend. The FVM run is refined only over the 2.4 $D$ box, so wider boxes are compared with coarser FVM cells outside it | Cells grow with the box |
| Transfer inset | 4, 8.75, 14 cells | ±0.2 %, at the resolution | None |
| Particle spacing | $2h$ instead of $h$ | Lift −0.9 %, drag −0.4 % | Fewer particles |
| Particle integration | RK4 instead of RK2 | None | More induction evaluations |
| Vorticity floor | $10^{-3}$ instead of $10^{-2}$ s$^{-1}$ | Lift −0.6 %, drag −0.4 %, $St$ −0.5 % | More particles |
| Cell size at $\Delta t_\mathrm{VPM}=0.048$ s | $h=0.08$, 0.06, 0.04 m | −2.6 %, −3.2 %, −1.1 % | Cells and particles |

### Why the interval matters

A shorter interval gives a larger error, so the error is added at each transfer and is not a time-integration error. Raising $\kappa_\mathrm{max}$ from 1.8 to 3 leaves it unchanged, so the strength limiter does not cause it. The mean FVM boundary-velocity mismatch grows with the transfer rate, from $2.6\times10^{-3}U_\infty$ at $N=6$ to $6.4\times10^{-3}U_\infty$ at $N=1$, together with the force error. The remeshing and thresholding applied at each transfer are the next candidates to test.

## Rules for a user

For laminar periodic shedding from a smooth body at this resolution:

1. Use the longest coupling interval the particles allow, a particle CFL number $\Delta t_\mathrm{VPM}S_\mathrm{max}$ of 0.75 to 0.9. It gives the smallest error and the lowest cost. Below 0.5 the error grows quickly. Refine time with the FVM step inside the interval, not with a shorter interval.
2. Do not run one coupling iteration. Converge to the tolerance $10^{-5}$; a fixed $M=2$ gives the same result here.
3. Keep $\kappa_\mathrm{max}=1.8$, a transfer inset of 8.75 cells and RK2 particle integration. Changing them did not move the result beyond the resolution.
4. A box margin of 1.6 to 2.4 $D$ is enough.
5. Keep the particle spacing equal to the cell size and the vorticity floor at $10^{-2}$ s$^{-1}$.
6. Expect a remaining coupling error near 1 % in lift amplitude and 0.3 % in $St$, about the FVM's error at $h=D/25$. To go lower, refine the FVM first.
7. Before comparing with an FVM solution in a walled domain, correct it for blockage.

## Not measured

The interval and $\kappa_\mathrm{max}$ series stop at a particle CFL number of 0.93 ($N=6$). The remeshing and thresholding at the transfer, the blending depths and the moment-correction limit were not varied. Apart from the baseline, runs end at 55, at the end of the lift-amplitude growth.
