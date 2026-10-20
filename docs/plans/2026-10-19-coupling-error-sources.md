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
- **Cancelling errors.** At constant Courant number and $n=2$ the coupling
  error in lift amplitude changes sign under refinement (+1.9%, -1.2%,
  -3.0% at $h=0.08$, $0.0566$, $0.04$): at least two contributions of opposite
  sign. An order fitted to such data means nothing.
- **One parameter at a time.** The box, inset and exchange-ratio runs varied
  one parameter about one baseline, so interactions were invisible, and the
  box margin was in diameters and the interface widths in cells, neither of
  which is a scale of the flow.

## Method

### Error measures

Each statistic $\phi$ (mean drag, lift amplitude, Strouhal number over
$t\ge40$ s, and the phase-averaged wake velocity at $x/D=2$ and $4$) is
compared with $\phi_0$, the reference extrapolated to zero cell size and step.
For a coupled run on cells $h$ with FVM step $\Delta t_f$:

- total error $e=\phi_c-\phi_0$;
- FVM error $e_f=\phi_r(h,\Delta t_f)-\phi_0$, the reference on the same cells
  and step;
- coupling error $e_c=\phi_c-\phi_r(h,\Delta t_f)=e-e_f$.

### Order and uncertainty

Least-squares fits after Eça and Hoekstra (2014): $\phi_i=\phi_0+\alpha x_i^p$
with $p$ free, and the fixed-order alternatives, chosen by the standard
deviation of the fit; uncertainty $U$ with safety factor 1.25 for well-behaved
data and 3 otherwise. Each family is classed as monotone, oscillatory or
anomalous, and an order is reported only for monotone data. For the
references, $\phi=\phi_0+a h^p+b\,\Delta t^q$ over all levels gives the
spatial and temporal orders separately. This procedure is the tool users apply
to their own runs.

### Normalization

| Parameter | Flow-scaled form |
| --- | --- |
| Cells, particle spacing | $h/\delta$, with $\delta$ the separated shear-layer thickness; $h_p/h_f$ |
| FVM and particle steps | $C_f=U\Delta t_f/h_f$, $C_p=U\Delta t_p/h_p$, $U\Delta t_p/D$ |
| Particle stability | $\Lambda=\max\lvert S\rvert\,\Delta t_p$, with $\max\lvert S\rvert$ measured at each $Re$ |
| Exchange frequency | exchanges per convective time $D/(U\Delta t_p)$ |
| Interface widths | inset, blend and particle-only band in particle core radii $\sigma$ |
| FVM box | distance from the formation point to the transfer region, in vortex-formation lengths $L_f$ (measured at each $Re$) |

### In-run indicator

The coupler logs the mismatch of the FVM boundary trace and the interface
residuals at every exchange. Their developed-flow means are correlated with
$|e_c|$; a good correlation gives users an a-posteriori check of the coupling
error that needs no reference.

## Run families (case 01)

| Family | Fixed | Varied | Isolates |
| --- | --- | --- | --- |
| C, combined | $C_f=0.2$, $n=2$ | $h=0.08$, $0.0566$, $0.04$, $0.0283$ | overall order of the method |
| X, space | $\Delta t_f=0.004$, $n=4$ | $h=0.08$, $0.0566$, $0.04$ | spatial order |
| T2, time at fixed $n$ | $h=0.04$, $n=2$ | $\Delta t_f=0.016$, $0.008$, $0.004$ | temporal order with the exchange frequency rising |
| P, time at fixed $\Delta t_p$ | $h=0.04$, $\Delta t_p=0.04$ | $n=5$, $10$, $20$ | FVM step alone, exchange frequency fixed |
| E, exchange ratio | $h=0.04$, $\Delta t_f=0.008$ | $n=2$, $4$, $5$, $6$ | particle step and exchange frequency together |
| K, interface sweeps | baseline | $K=1$, $2$, $3$, $4$, $6$ | interface iteration |
| B, box | $n=5$, inset $8.75\sigma$ | four box sizes in $L_f$, at $Re=150$ and $250$ | box error, flow-scaled |
| F, factorial | $h=0.04$ | box $\times$ inset $\times$ $n$ at two levels | interactions; additivity of the errors |
| R, Reynolds number | baseline rules | $Re=100$, $250$ | transfer of the rules to other flows |
| S, particle spacing | $\Delta t_f=0.008$ | $h_p=h_f/2$ at $n=2$, $5$ | particle discretization |

T2, P and E vary the particle step and the exchange frequency independently,
which separates a temporal particle error from an error added at each
transfer.

## Schedule

Three coupled runs share the GPU and one reference runs on the CPU; the cube
run waits until the study ends. Wall-time costs of runs made in parallel are
inflated, so cost conclusions use runs made alone.

| Day | Work |
| --- | --- |
| Tue 20 | K, T2, X, finest C; flow scales; analysis toolkit |
| Wed 21 | R, B (flow-scaled box); analysis of C, X, T2, K |
| Thu 22 | F, P, S; additivity and indicator analysis |
| Fri 23 | Report: orders and uncertainties, error sources, flow-scaled rules, user procedure |

## Decisions

1. Convergence problem: case 01 itself (2026-10-19); no new tutorial case.
2. Generality: the cylinder at $Re=100$ and $250$; no second geometry this
   week (2026-10-20).
3. The studies run before the cube and the NACA wing (2026-10-20).

## Results

To be filled in as the families complete.
