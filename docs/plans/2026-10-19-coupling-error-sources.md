# Where do the coupled-solution errors come from?

Plan agreed with the supervisor on 2026-10-19 and revised on 2026-10-20. All
studies are runs of case 01, the planar cylinder (its sensitivity study), and
are plotted by the case's `allplot.sh`. Results and the report are due by
2026-10-23.

## Objectives

1. Give users an objective way to measure the spatial and temporal order of
   their own problem with the coupled method, and the uncertainty of a result.
2. Find each source of the coupled error and isolate it, with every parameter
   expressed in flow scales, so that a recommendation does not hold only
   because two errors happen to cancel at one setup.

## Why the first approach was not enough

- **Biased yardstick.** The earlier sensitivity runs were compared with the
  reference on the same cells. That removes the shared FVM error but hides
  that the reference itself is 4% low in lift amplitude at the baseline cells
  ($h=0.04$): the 2% target was met against an unconverged value.
- **Cancelling errors.** At constant Courant number and $N=2$ the coupling
  error in lift amplitude changes sign under refinement (+1.9%, -1.2%,
  -3.0% at $h=0.08$, $0.0566$, $0.04$): at least two contributions of opposite
  sign. An order fitted to such data means nothing.
- **One parameter at a time.** The FVM domain, inset and FVM-substep runs varied
  one parameter about one baseline, so interactions were invisible, and the
  box margin was in diameters and the interface depths in cells, neither of
  which is a scale of the flow.

## Method

### Error measures

Each statistic $Q$ (mean drag, lift amplitude, Strouhal number over
$t\ge40$ s, and the phase-averaged wake velocity at $x/D=2$ and $4$) is
compared with $Q_\mathrm{ext}$, the FVM-only value extrapolated to zero cell
size and step. For a coupled run on cells $h$ with FVM time step
$\Delta t_\mathrm{FVM}$:

- total error $e=Q_\mathrm{coupled}-Q_\mathrm{ext}$;
- FVM error $e_\mathrm{FVM}=Q_\mathrm{FVM}(h,\Delta t_\mathrm{FVM})-Q_\mathrm{ext}$,
  the FVM-only run on the same cells and step;
- coupling error $e_\mathrm{coupling}=Q_\mathrm{coupled}-Q_\mathrm{FVM}(h,\Delta t_\mathrm{FVM})=e-e_\mathrm{FVM}$.

### Order and uncertainty

Least-squares fits after Eça and Hoekstra (2014): $Q_i=Q_\mathrm{ext}+\alpha x_i^p$
with $p$ free, and the fixed-order alternatives, chosen by the standard
deviation of the fit; uncertainty $U_\mathrm{ext}$ with safety factor 1.25 for
well-behaved data and 3 otherwise. Each family is classed as monotone,
oscillatory or anomalous, and an order is reported only for monotone data. For
the FVM-only runs, $Q=Q_\mathrm{ext}+a h^p+b\,\Delta t^q$ over all levels gives
the spatial and temporal orders separately. This procedure is the tool users
apply to their own runs.

### Normalization

| Parameter | Flow-scaled form |
| --- | --- |
| Cells, particle spacing | $h/\delta_\omega$, with $\delta_\omega$ the vorticity thickness of the separated shear layer; $h_\mathrm{VPM}/h_\mathrm{FVM}$ |
| FVM and VPM time steps | $C_\mathrm{FVM}=U\Delta t_\mathrm{FVM}/h_\mathrm{FVM}$, $C_\mathrm{VPM}=U\Delta t_\mathrm{VPM}/h_\mathrm{VPM}$, $U\Delta t_\mathrm{VPM}/D$ |
| Particle stability | Lagrangian CFL number $\Lambda=\Delta t_\mathrm{VPM}\lVert S\rVert_\infty$, with $\lVert S\rVert_\infty$ measured at each $Re$ |
| Coupling frequency | coupling intervals per convective time $D/(U\Delta t_\mathrm{VPM})$ |
| Interface depths | transfer inset $L_\mathrm{inset}$, zero-weight layer $\delta_0$ and blend end depth $\delta_1$, in particle core radii $\sigma$ |
| FVM domain | wake reach $\ell_\mathrm{wake}$ in vortex-formation lengths $L_f$ and lateral clearance $\ell_\mathrm{side}$ in $\delta_\omega$ (measured at each $Re$) |

### In-run indicator

The coupler logs the boundary-data velocity mismatch on the outer FVM boundary
and the coupling-iteration residuals in every coupling interval. Their
developed-flow means are correlated with $|e_\mathrm{coupling}|$; a good
correlation gives users an a-posteriori check of the coupling error that needs
no reference.

## Run families (case 01)

| Family | Fixed | Varied | Isolates |
| --- | --- | --- | --- |
| C, combined | $C_\mathrm{FVM}=0.2$, $N=2$ | $h=0.08$, $0.0566$, $0.04$, $0.0283$ | overall order of the method |
| X, space | $\Delta t_\mathrm{FVM}=0.004$, $N=4$ | $h=0.08$, $0.0566$, $0.04$ | spatial order |
| T2, time at fixed $N$ | $h=0.04$, $N=2$ | $\Delta t_\mathrm{FVM}=0.016$, $0.008$, $0.004$ | temporal order with the coupling frequency rising |
| P, time at fixed $\Delta t_\mathrm{VPM}$ | $h=0.04$, $\Delta t_\mathrm{VPM}=0.04$ | $N=5$, $10$, $20$ | FVM time step alone, coupling frequency fixed |
| E, FVM substeps | $h=0.04$, $\Delta t_\mathrm{FVM}=0.008$ | $N=2$, $4$, $5$, $6$ | VPM time step and coupling frequency together |
| K, coupling iterations | baseline | $M=1$, $2$, $3$, $4$, $6$ | coupling iteration |
| B, FVM domain | $N=5$, inset $8.75\sigma$ | four FVM domains in $L_f$, at $Re=150$ and $250$ | FVM-domain error, flow-scaled |
| F, factorial | $h=0.04$ | FVM domain $\times$ inset $\times$ $N$ at two levels | interactions; additivity of the errors |
| R, Reynolds number | baseline rules | $Re=100$, $250$ | transfer of the rules to other flows |
| S, particle spacing | $\Delta t_\mathrm{FVM}=0.008$ | $h_\mathrm{VPM}=h_\mathrm{FVM}/2$ at $N=2$, $5$ | particle discretization |

### What each family holds fixed

Each family varies one group and holds the others, so its error trend has a
single cause. $h_\mathrm{FVM}=h_\mathrm{VPM}=h$; interface depths $w$ (inset,
blending ramp, zero-weight layer); coupling intervals per convective time
$D/(U\Delta t_\mathrm{VPM})$.

| Family | $h/\delta_\omega$ | $C_\mathrm{FVM}$ | $U\Delta t_\mathrm{FVM}/D$ | $U\Delta t_\mathrm{VPM}/D$, $D/(U\Delta t_\mathrm{VPM})$ | $\Lambda$ | $w/D$ | FVM domain ($\ell_\mathrm{wake}/L_f$, $\ell_\mathrm{side}/\delta_\omega$) | $Re$ | Isolates |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| X | varied | varies with $h$ | fixed | fixed | fixed | **fixed** (depths in 0.04 m cells) | fixed | fixed | spatial error of the coupled solver |
| T2 | fixed | varied | varied | varied ($N=2$) | varied | fixed | fixed | fixed | temporal error of the coupled solver |
| P | fixed | varied | varied | **fixed** | fixed | fixed | fixed | fixed | FVM temporal error inside the coupling |
| E | fixed | fixed | fixed | varied | varied | fixed | fixed | fixed | VPM time step and transfer count together |
| K | fixed | fixed | fixed | fixed | fixed | fixed | fixed | fixed | coupling iteration only |
| B | fixed | fixed | fixed | fixed | per $Re$ | fixed | varied in $L_f$ and $\delta_\omega$ | 150, 250 | FVM-domain error, flow-scaled |
| R | fixed in $D$ | fixed | fixed | fixed | matched ($N=4$ at 250) or not ($N=5$) | fixed | per $Re$ | varied | transfer of the rules to another flow |
| C | varied | fixed | varied | varied ($N=2$) | varied | varies (in cells) | fixed | fixed | what a user sees refining everything together |

The VPM time step and the number of transfers are tied (one transfer per
coupling interval), so E cannot separate them by design; their functional forms
differ (a temporal error grows with $\Delta t_\mathrm{VPM}$, an error added at
each transfer grows with $D/(U\Delta t_\mathrm{VPM})$), so a fit
$e_\mathrm{coupling}=c_0+a\,U\Delta t_\mathrm{VPM}/D+b\,D/(U\Delta t_\mathrm{VPM})$
over E, P and T2 attributes the coupling error to one or the other. Family C
mixes everything and is reported only as the user-level check; the spatial and
temporal orders of the coupled solver come from X and T2.

## Schedule

Three coupled runs share the GPU and one reference runs on the CPU; the cube
run waits until the study ends. Wall-time costs of runs made in parallel are
inflated, so cost conclusions use runs made alone.

| Day | Work |
| --- | --- |
| Tue 20 | K, T2, X, finest C; flow scales; analysis toolkit |
| Wed 21 | R, B (flow-scaled FVM domain); analysis of C, X, T2, K |
| Thu 22 | F, P, S; additivity and indicator analysis |
| Fri 23 | Report: orders and uncertainties, error sources, flow-scaled rules, user procedure |

## Decisions

1. Convergence problem: case 01 itself (2026-10-19); no new tutorial case.
2. Generality: the cylinder at $Re=100$ and $250$; no second geometry this
   week (2026-10-20).
3. The studies run before the cube and the NACA wing (2026-10-20).

## Results

To be filled in as the families complete.
