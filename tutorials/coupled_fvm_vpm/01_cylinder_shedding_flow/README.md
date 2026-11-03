# Cylinder shedding at $Re=150$

A no-slip cylinder sheds a planar vortex street. The FVM resolves the wall and the vortex-formation region; one layer of vortex particles carries the outer wake. The [reference flow](reference_flow/README.md) solves the same problem with the FVM alone.

| Quantity | Value |
| --- | --- |
| Diameter $D$; freestream $U_\infty$ | 1 m; 1 m/s in $+x$ |
| Density $\rho$; viscosity $\nu$ | 1 kg/m³; $1/150$ m²/s |
| FVM domain | $[-2.4,4.4]\times[-2.4,2.4]$ m |
| VPM domain; outflow fringe | $[-5,30]\times[-5,5]$ m; $x\in[15,30]$ m |
| Cell and particle spacing | 0.04 m |
| FVM step; coupling interval | 0.008 s; 0.04 s |
| Statistics window | $40\le t\le100$ s |

## Physics and numerical model

The [body-fitted mesh](../../../docs/fvm.md#mesh-setup) is cut from `assets/cylinder_long.stl`. Its outer faces are the `numericalBoundary`: particles provide normal velocity and the normal gradient of tangential velocity through the [mixed boundary condition](../../../docs/coupling.md#boundary-conditions).

Within the transfer region, FVM vorticity replaces particle vorticity at each coupling time. Below the transfer surface, a zero-weight layer of two cells keeps the particle vorticity, and a blending ramp of four cells raises the blending weight to one at $\delta_1=6$ cells, where the replacement region starts. The particles use Gaussian cores, RK2 transport, vector stretching and grid-based diffusion. Free-slip planes bound the single particle layer and reproduce the planar velocity field.

A compact vortex behind the cylinder starts asymmetric shedding. Fields and statistics begin after the startup transient has left the near wake.

## Run

Run commands from this directory.

```bash
(cd reference_flow && ./allrun.sh)
./allrun.sh
./allplot.sh
```

`reference_flow/allrun.sh` simulates the four grids of the FVM-only reference from $t=0$; `reference_flow/allcontinue.sh` resumes them. `./allrun.sh` runs every coupled case of the study that is not commented out, each from its latest backup in `solution/sensitivity_analysis/<name>`. Both import the frozen solver export of commit `3f771630`. `./allplot.sh` makes every figure below.

## Results

| File | Content |
| --- | --- |
| `figures/forces` | Drag and lift against the reference, with mean drag, lift amplitude and Strouhal number |
| `figures/velocity_profiles` | Mean velocity and transverse-velocity fluctuation at $x/D=2$ and $4$ |
| `figures/velocity_fields`, `figures/vorticity_fields` | Reference, coupled and difference fields |
| `figures/coupling_cost` | Wall time by FVM, VPM, coupling and output work; particle count |
| `figures/cylinder_shedding.gif` | Developed vorticity field and force history |
| `figures/reference_forces`, `figures/flow_scales` | Grid independence of the FVM-only reference: forces and wake scales of the four grids |
| `figures/error_*`, `figures/sensitivity*`, `figures/interface_*` | Extrapolated reference ($Q_\mathrm{ext}$), FVM and coupling errors of every coupled run, and the one-at-a-time series |

## Sensitivity study

The sensitivity study varies the FVM domain, transfer region, coupling interval, cell size, particle side, FVM side and coupling iterations, one at a time. Statistics use $40\le t\le55$ s in every run and reference; the lift amplitude takes each extreme at the vertex of a parabola through its three samples, so that runs sampled at different intervals compare alike.

### Run tracker, solver `3f771630` (campaign of 2 to 8 November 2026)

Every run below imports the frozen export of commit `3f771630`; runs of other commits are archived and never compared. Baseline: $h=0.04$ m (cells and particle spacing), $\Delta t_\mathrm{FVM}=0.008$ s, $N=5$ ($\Delta t_\mathrm{VPM}=0.04$ s), $y_\mathrm{box}=2.4$ m, $x_\mathrm{box}=4.4$ m, $L_\mathrm{inset}=8.75h$, $\kappa_\mathrm{max}=1.8$, $\tau_\mathrm{t}=0.05$, tolerance-controlled coupling iterations. Statistics over $40\le t\le55$ s. Status: **ok** = reached its end time; **running**; **queued** = in the driver's order; **postponed** = runs only if time remains before 8 November; **not planned** = not part of this campaign. Groups: D FVM domain, I interface, T coupling interval, V particle side, F FVM side, R FVM-only reference.

Coupled runs: `./allrun.sh`, one after another, each in `solution/sensitivity_analysis/<name>`; finished runs are commented out there. FVM-only references: `reference_flow/allrun.sh`, each in `reference_flow/samples/<name>` and `reference_flow/solution/<name>`. A coupled run is compared with the reference on its cells and FVM step; runs without one (sections `a`, `a_fine`, `p`, `c` and the two Courant runs of `f_side`) get no $e_\mathrm{coupling}$.

| Section | Group | Run directory under `solution/sensitivity_analysis/` | Varied quantity | Status |
| --- | --- | --- | --- | --- |
| anchor | – | `box2.4_sub5_inset8.75` | baseline, to $t=100$ s (sampling floor) | ok |
| `k` | I | `box2.4_sub5_inset8.75_k1` | coupling iterations $M=1$ | ok |
| `k` | I | `box2.4_sub5_inset8.75_k2` | $M=2$ | ok |
| `k` | I | `box2.4_sub5_inset8.75_k4` | $M=4$ | to run (resumes from 31 s) |
| `n_interval` | T | `box2.4_sub1_inset8.75` | $N=1$ ($\Delta t_\mathrm{VPM}=0.008$ s) | to run (resumes from 36 s) |
| `n_interval` | T | `box2.4_sub2_inset8.75` | $N=2$ ($\Delta t_\mathrm{VPM}=0.016$ s) | ok |
| `n_interval` | T | `box2.4_sub3_inset8.75` | $N=3$ ($\Delta t_\mathrm{VPM}=0.024$ s) | ok |
| `n_interval` | T | `box2.4_sub6_inset8.75` | $N=6$ ($\Delta t_\mathrm{VPM}=0.048$ s) | to run |
| `n_interval` | T | `box2.4_sub1_inset8.75_kappa3` | $N=1$, $\kappa_\mathrm{max}=3$ | to run |
| `n_interval` | T | `box2.4_sub3_inset8.75_kappa3` | $N=3$, $\kappa_\mathrm{max}=3$ | to run |
| `n_interval` | T | `box2.4_sub4_inset8.75_kappa3` | $N=4$, $\kappa_\mathrm{max}=3$ | to run |
| `n_interval` | T | `box2.4_sub6_inset8.75_kappa3` | $N=6$, $\kappa_\mathrm{max}=3$ | to run |
| `box_min` | D | `box1.6_sub5_inset8.75` | $y_\mathrm{box}=1.6$ m | ok |
| `box_min` | D | `box2.0_sub5_inset8.75` | $y_\mathrm{box}=2.0$ m | ok |
| `box_conv` | D | `box3.2_sub5_inset8.75` | $y_\mathrm{box}=3.2$ m | ok |
| `box_conv` | D | `box4.0_sub5_inset8.75` | $y_\mathrm{box}=4.0$ m | ok |
| `inset` | I | `box2.4_sub5_inset4` | $L_\mathrm{inset}=4h$ | to run |
| `inset` | I | `box2.4_sub5_inset14` | $L_\mathrm{inset}=14h$ | to run |
| `a` | F | `box2.4_sub8_inset8.75_h0.08_dtf0.004_ic0.04` | cells and particle spacing $h=0.08$ m, $\Delta t_\mathrm{FVM}=0.004$ s, $N=8$, interface depths in 0.04 m cells | to run |
| `a` | F | `box2.4_sub8_inset8.75_h0.06_dtf0.004_ic0.04` | $h=0.06$ m | to run |
| `a` | F | `box2.4_sub8_inset8.75_dtf0.004` | $h=0.04$ m | to run |
| `a_fine` | F | `box2.4_sub8_inset8.75_h0.03_dtf0.004_ic0.04` | $h=0.03$ m | to run |
| `p` | F | `box2.4_sub10_inset8.75_dtf0.004` | $\Delta t_\mathrm{FVM}=0.004$ s at $\Delta t_\mathrm{VPM}=0.04$ s ($N=10$) | to run |
| `c` | F | `box2.4_sub20_inset8.75_dtf0.002` | $\Delta t_\mathrm{FVM}=0.002$ s at $\Delta t_\mathrm{VPM}=0.04$ s ($N=20$) | to run |
| `v_side` | V | `box2.4_sub5_inset8.75_hp0.08` | particle spacing 0.08 m | to run |
| `v_side` | V | `box2.4_sub5_inset8.75_rk4` | RK4 particle integrator | to run |
| `v_side` | V | `box2.4_sub5_inset8.75_vf0.001` | vorticity floor 0.001 | to run |
| `v_side` | V | `box2.4_sub5_inset8.75_vf0.1` | vorticity floor 0.1 | to run |
| `f_side` | F | `box2.4_sub5_inset8.75_oc1` | 1 PIMPLE outer corrector | to run |
| `f_side` | F | `box2.4_sub5_inset8.75_oc3` | 3 PIMPLE outer correctors | to run |
| `f_side` | F | `box2.4_sub2_inset8.75_dtf0.02_co0.5` | Courant controller 0.5, $\Delta t_\mathrm{FVM}\le0.02$ s | to run |
| `f_side` | F | `box2.4_sub2_inset8.75_dtf0.02_co0.9` | Courant controller 0.9 | to run |
| `b1` | I | `box2.4_sub{5,2}_inset8.75_kappa{1,2,3,10}` (8 runs) | $\kappa_\mathrm{max}=1,2,3,10$ at $N=5$ and 2 | postponed |
| `i_transfer` | I | `box2.4_sub5_inset8.75_{tau0.01,tau0.2,mc0.02,mc0.3}`, `box2.4_sub2_inset8.75_tau0.2` | $\tau_\mathrm{t}=0.01, 0.2$; moment cap 0.02, 0.3 | postponed |

Grid independence of the FVM-only reference, at a constant Courant number ($\Delta t = 0.008$ s $h/0.04$ m), $Re=150$:

| Reference under `reference_flow/samples/` | $h$ [m] | $\Delta t$ [s] | Status |
| --- | --- | --- | --- |
| `h0.08_dt0.016` | 0.08 | 0.016 | to run |
| `h0.06_dt0.012` | 0.06 | 0.012 | to run |
| `h0.04_dt0.008` | 0.04 | 0.008 | to run (the coupled baseline's reference) |
| `h0.03_dt0.006` | 0.03 | 0.006 | to run |

`./allplot.sh` writes the study figures and the tables in `figures/auxiliary/`. [coupling_findings.md](assets/coupling_findings.md) records the physical conclusions and the quantities each section measures.
