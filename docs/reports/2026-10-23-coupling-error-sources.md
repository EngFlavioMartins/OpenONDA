# Sources of error in the coupled FVM–VPM solver: the planar cylinder at $Re=150$

Terms and symbols follow [terminology](../terminology.md). All conclusions hold for laminar periodic shedding ($Re=100$–$250$, one flow regime); other regimes are tested by the cube and wing tutorials.

Technical report, 2026-10-23, completed 2026-10-25 with every run of the study. Study plan: [2026-10-19-coupling-error-sources](../plans/2026-10-19-coupling-error-sources.md). Case: [tutorials/coupled_fvm_vpm/01_cylinder_shedding_flow](../../tutorials/coupled_fvm_vpm/01_cylinder_shedding_flow/README.md), whose `./allplot.sh` draws every figure and writes every table cited here (figures in `figures/`, tables in `figures/auxiliary/`). Earlier findings: [coupling_findings.md](../../tutorials/coupled_fvm_vpm/01_cylinder_shedding_flow/assets/coupling_findings.md).

All 36 coupled and 12 FVM-only runs were made with the same solver, development commits `3b4c0f23` to `97d65b09`; the later commit only changed default values that the case sets explicitly.

## 1. Summary

The coupled error of the planar cylinder splits into the error of the FVM discretization, which the coupled solver inherits, and a coupling error that the FVM-only solution does not have. Measured against the FVM-only value extrapolated to zero cell size and step, with every parameter in flow scales and one parameter varied at a time:

- **The coupling error is an error added at each transfer.** On the baseline cells it follows the number of coupling intervals per convective time, $L/(U\Delta t_\mathrm{VPM})=D/(U\Delta t_\mathrm{VPM})$, as $e_\mathrm{coupling}=c_0-a\,L/(U\Delta t_\mathrm{VPM})^{1.7\text{–}1.75}$ with $R^2=1.000$ over six runs that vary the particle step and the FVM step independently. Halving the FVM step at a fixed particle step leaves it unchanged (lift amplitude −1.42% at $N=5$, −1.35% at $N=10$).
- **The coupled solver therefore has no temporal order at a fixed coupling interval ratio.** Refining both steps together ($N=2$) makes the coupling error grow: −1.5%, −2.8%, −7.1% in lift amplitude at $L/(U\Delta t_\mathrm{VPM})=31$, 63, 125. The coupled solver converges in time only with the particle step held at its largest stable value and the FVM step refined under it; it then inherits the FVM temporal order.
- **Spatially, the coupled solution converges with the FVM towards a value offset by the transfer error.** At a fixed particle step ($\Delta t_\mathrm{VPM}=0.04$ s, $L/(U\Delta t_\mathrm{VPM})=25$) and a fixed interface, the total lift-amplitude error is −10.4%, −7.7%, −5.2% at $h/D=0.08$, 0.0566, 0.04; the FVM part converges with order 1.5, and the coupling part (+2.2%, −0.9%, −1.35%) changes sign and approaches the −1.4% that $L/(U\Delta t_\mathrm{VPM})=25$ sets on the baseline cells. The Strouhal-number coupling error falls monotonically (+3.1%, +1.0%, +0.35%). Three levels do not give an order for the coupled solver. A fourth level, $h/D=0.0283$, was inadmissible at the same particle step: the particle-field strain rate rose from 19.4 to 25.8 s⁻¹, $\Lambda=1.03$. The largest stable particle step therefore shrinks under refinement and must be chosen for the finest cells.
- **FVM only:** spatial order 1.5–2.0 (lift amplitude) and 1.7 (Strouhal number), temporal order 0.9 (lift amplitude) and 1.7 (Strouhal number), mean drag converged within 0.2–0.6%. For the lift amplitude the spatial error is negative and the temporal error positive, so refining both together hides part of each.
- **Error budget at the baseline** ($h/\delta_\omega=0.15$, $C_\mathrm{FVM}=0.2$, $N=5$, $\Lambda=0.78$, $L/(U\Delta t_\mathrm{VPM})=25$): total $e$ = −0.22% (drag), −4.53% (lift amplitude), −0.90% (Strouhal number), of which the FVM part is −0.14%, −3.11%, −1.22% and the coupling part −0.09%, −1.42%, +0.33%. The coupling part is $c_0$ (−1.0% in lift amplitude, not removed by fewer transfers) plus the transfer part (−0.4% at $L/(U\Delta t_\mathrm{VPM})=25$).
- **FVM domain:** the lateral clearance, in shear-layer thicknesses $\delta_\omega$, is the FVM domain parameter that matters. Within laminar periodic shedding it gives the same effect at $Re=150$ and $250$; other regimes are not covered. Relative to the widest domain, the lift-amplitude coupling error grows by 4.0 / 2.2 / 0.6 points at $\ell_\mathrm{side}/\delta_\omega\approx1.2$ / 1.9 / 2.7 at $Re=150$, and by 4.4 / 2.4 / 1.3 points at 1.3 / 2.0 / 2.75 at $Re=250$; about 5 makes it small at both. The wake reach in formation lengths changes the error by at most about 2 points for $\ell_\mathrm{wake}/L_f\ge0.35$, without a common trend at the two Reynolds numbers. A too-short FVM domain looks more accurate at $Re=150$ (+0.7% against −1.4%) because its error cancels the transfer error.
- **Box and coupling interval errors do not add** (2×2 factorial, downstream face 3.36/4.4 D × $N$=2/5): more coupling intervals move the lift-amplitude coupling error from −1.42% to −2.82% in the long box but from +0.68% to +2.02% in the short one (additivity defect 2.75 points). More coupling intervals increase the magnitude in both and lower the Strouhal number in both. Size the FVM domain first.
- **Transfer inset:** 4, 8.75 and 14 spacings give −1.67%, −1.42%, −1.29% in lift amplitude (drag −0.15%, −0.09%, −0.04%). A smaller rewritten volume reduces the error, so the vorticity transfer contributes to the per-coupling interval error; the effect over this range is a quarter of the $L/(U\Delta t_\mathrm{VPM})$ effect between 25 and 63.
- **Repeatability:** a repeat of the baseline reproduces its coupling errors within 0.01 points, so GPU summation order does not limit any comparison in this report.
- **Coupling iteration:** two coupling iterations per coupling interval, or a boundary-data tolerance of $10^{-5}$ (1.8 coupling iterations on average), keep the iteration error below 0.15 points; one coupling iteration costs +3.0% in drag and +2.1% in lift amplitude.
- **In-run indicator:** the developed mean of the boundary-data velocity mismatch correlates with the lift-amplitude coupling error ($r=0.82$ over 36 runs); every run with a mismatch below $3.5\times10^{-3}U$ has $\lvert e_\mathrm{coupling}\rvert\le2.1\%$ in lift amplitude. It is a weaker indicator for drag and Strouhal number ($r\approx0.4$).
- **Another Reynolds number:** at $Re=250$ with the same rules the coupling error is −0.14% (drag), −1.93% (lift amplitude), +0.53% (Strouhal number) at $N=5$ ($\Lambda=1.0$, stable) and −0.19%, −2.01%, +0.50% at $N=4$ ($\Lambda=0.8$). Matching $\Lambda$ does not match the error of $Re=150$. From $N=5$ to $N=2$ the drag and Strouhal number fall at $Re=100$, 150 and 250 alike (0.3–1.0 points), but the lift amplitude falls by 1.40 points at 150, 0.71 at 250 and not at all at 100 (+0.04): the direction of the trend with $L/(U\Delta t_\mathrm{VPM})$ is the same at the three Reynolds numbers, its size is not. All three lie in one flow regime.

## 2. The question, and why naive refinement misleads

The coupled solver couples a finite-volume (FVM) solution of the near body, inside a box, to one layer of vortex particles between slip planes that carries the outer wake. At every coupling interval the particles supply the FVM boundary condition on the FVM domain faces, and FVM vorticity replaces the particles inside the transfer region. A coupled result therefore carries the discretization errors of both solvers and an error of the coupling itself: of the interface position, of the transfer, of the boundary-data iteration and of the coupling interval frequency. The study sets out to

1. give users an objective procedure, free of a choice of favourable levels, to measure the spatial and temporal order of their own coupled problem and the uncertainty of a result;
2. isolate each source of error, with every parameter expressed in scales of the flow, so that a recommended value does not rest on two errors cancelling at one setup;
3. state values for the coupling parameters in lengths and times of the flow, so that they can be tested in other flows, and the error and the spatial and temporal orders of the coupled solver as a whole, independent of the orders of the FVM and the VPM on their own.

Two features of the earlier sensitivity study made its conclusions unsafe.

**Same-cell comparison.** The coupled runs were compared with an FVM-only run on the same cells and step. That comparison removes the FVM discretization error shared by both runs and measures only the coupling error $e_\mathrm{coupling}$. It does not show how far either run is from the solution of the equations. In the preliminary runs of the plan, the FVM-only lift amplitude on the baseline cells ($h=0.04$ m) lies about 4% below its extrapolated value, so the 2% target had been met against an unconverged value. The present study measures every error against $Q_\mathrm{ext}$, the FVM-only value extrapolated to zero cell size and step, and splits it into an FVM part and a coupling part (section 3.1).

**Cancelling errors.** Refining cells, steps and particles together at constant Courant number is what a user would do first. It mixes errors of opposite sign. For the lift amplitude, the FVM spatial error is negative and the temporal error positive, so their sum can be small, or change sign, on a level where neither is small. In the preliminary runs at $C_\mathrm{FVM}=0.2$ and $N=2$, the coupling error in lift amplitude changes sign under refinement (+1.9%, −1.3%, −2.8% at $h=0.08$, $0.0566$, $0.04$ m, against the extrapolated value): at least two contributions of opposite sign are present, and an order fitted to such data has no meaning. The study therefore refines space and time separately (families X and T2) and keeps family C only as the check a user sees when refining everything together.

A third weakness was that each parameter had been varied about one baseline and expressed in diameters or cells. Neither is a scale of the flow, so a value found at $Re=150$ need not carry over, and interactions between parameters were not seen.

## 3. Method

### 3.1 Case and error measures

The flow is the planar cylinder at $Re=U D/\nu=150$, with $D=1$ m and $U=1$ m/s. In the coupled case the FVM resolves the wall and the near wake in the FVM domain $[-2.4,4.4]\times[-2.4,2.4]$ m, with cells of $h=0.04$ m and a step $\Delta t_\mathrm{FVM}=0.008$ s at the baseline; one layer of vortex particles of spacing $h_\mathrm{VPM}=h$ between slip planes carries the wake to $x=30$ m, through an outflow fringe over $x\in[15,30]$ m. The FVM-only case solves the same problem on a large, fully meshed domain $[-40,60]\times[-60,60]$ m with the same near-body cells; it is called *FVM only* throughout.

Each statistic $Q$ is taken over developed shedding, $40\le t\le55$ s: the mean drag $\bar C_D$, the lift amplitude $\hat C_L=\tfrac12(\max C_L-\min C_L)$, the Strouhal number $St$ from the upward zero crossings of $C_L$, and the phase-averaged wake velocity at $x/D=2$ and $4$. For a coupled run on cells $h$ with FVM step $\Delta t_\mathrm{FVM}$, giving $Q_\mathrm{coupled}$, and the FVM-only run on the same cells and step, giving $Q_\mathrm{FVM}(h,\Delta t_\mathrm{FVM})$:

- total error $e=Q_\mathrm{coupled}-Q_\mathrm{ext}$;
- FVM discretization error $e_\mathrm{FVM}=Q_\mathrm{FVM}(h,\Delta t_\mathrm{FVM})-Q_\mathrm{ext}$;
- coupling error $e_\mathrm{coupling}=Q_\mathrm{coupled}-Q_\mathrm{FVM}(h,\Delta t_\mathrm{FVM})$, so that $e=e_\mathrm{FVM}+e_\mathrm{coupling}$.

All three are given in per cent of $Q_\mathrm{ext}$. The value $Q_\mathrm{ext}$ comes from the space-time fit of section 3.2 over every FVM-only level when that fit determines $Q_\mathrm{ext}$ with a standard error smaller than the smallest family-C uncertainty; otherwise it is the extrapolation of FVM-only family C (four levels, refinement ratio $\sqrt2$, constant Courant number). A statistic that barely changes with the cells or the step, such as the mean drag, leaves the orders of the space-time fit undetermined, and its family-C value is used. The wake velocity has no scalar to extrapolate: its total and FVM errors are rms distances, at equal shedding phase, to the finest FVM-only run, and its coupling error the distance to the FVM-only run on the same cells and step.

### 3.2 Order and uncertainty

The order and the uncertainty follow the least-squares procedure of Eça and Hoekstra (J. Comput. Phys. 262, 2014, 104–130), implemented in `assets/error_analysis.py`. A family is a set of $n_g\ge3$ runs that varies one refinement variable $x$ ($h/D$ or $U\Delta t/D$) with everything else fixed.

**Fits.** Four forms are fitted to the levels $(x_i,Q_i)$:

- free order, $Q_i=Q_\mathrm{ext}+\alpha x_i^p$, with $p$ searched in $[0.05,8]$;
- first order, $Q_i=Q_\mathrm{ext}+\alpha x_i$;
- second order, $Q_i=Q_\mathrm{ext}+\alpha x_i^2$;
- first and second order, $Q_i=Q_\mathrm{ext}+\alpha_1x_i+\alpha_2x_i^2$, only when it has fewer coefficients than levels.

Each form is fitted unweighted and with weights $w_i\propto1/x_i$, which favour the finest levels, and the weighting with the smaller standard deviation $\sigma$ of the fit is kept.

**Convergence class.** From the signs of the successive differences of $Q$ towards $x\to0$:

- *monotone*: all differences of one sign, and the free-order fit converges (its order is not at the lower search bound);
- *oscillatory*: the differences alternate in sign;
- *anomalous*: otherwise, including monotone data whose free-order fit does not converge.

**Choice of fit and safety factor.**

- Monotone data with $0.5\le p<2.1$: the free-order fit, with safety factor $F_s=1.25$.
- Every other case: the fixed-order fit with the smallest $\sigma$, with $F_s=3$. For monotone data with $p\ge2.1$ the first-order fit is excluded.

**Uncertainty.** With the fitted error $\varepsilon_i=Q_i-Q_\mathrm{ext}$ of each level, the deviation $\lvert Q_i-Q_\mathrm{fit}(x_i)\rvert$ and the data range $\Delta_Q=(Q_\mathrm{max}-Q_\mathrm{min})/(n_g-1)$:

$$U_i=F_s\lvert\varepsilon_i\rvert+\sigma+\lvert Q_i-Q_\mathrm{fit}(x_i)\rvert\quad\text{if }\sigma<\Delta_Q,$$

$$U_i=3\,\frac{\sigma}{\Delta_Q}\left(\lvert\varepsilon_i\rvert+\sigma+\lvert Q_i-Q_\mathrm{fit}(x_i)\rvert\right)\quad\text{otherwise,}$$

the second form marking data that are not well behaved.

**When no order is reported.** An observed order is reported only for monotone data. For oscillatory or anomalous data the report gives the class, $Q_\mathrm{ext}$ and $U$ of the chosen fixed-order fit, and no order. A family with fewer than three developed levels gives neither.

**Space and time together.** Over every FVM-only level at $Re=150$, $Q=Q_\mathrm{ext}+a\,h^p+b\,\Delta t^q$ separates the spatial and the temporal order. The orders are first located on a grid $p,q\in[0.5,4]$ with the linear coefficients solved at each pair, then all five parameters are refined by nonlinear least squares; their standard errors follow from the Jacobian and the residual variance. The fit needs at least six levels.

### 3.3 Normalization by flow scales

Every parameter is expressed in a scale of the flow, measured on the FVM-only runs by `assets/flow_scales.py` (`figures/flow_scales`, `figures/auxiliary/flow_scales.csv`):

- vortex-formation length $L_f$, from the cylinder centre to the centreline peak of the rms streamwise velocity: $1.94D$ at $Re=150$ (uncertain by about $\pm0.15D$), $1.22D$ at $Re=250$, $3.0D$ at $Re=100$;
- vorticity thickness of the separated shear layer at the base plane, $\delta_\omega=\Delta u/\max\lvert\partial\bar u/\partial y\rvert\approx3.2D/\sqrt{Re}$;
- the largest strain rate the particle field sees, $\max\lvert S\rvert=19.4$ s$^{-1}$ at $Re=150$, scaling as $\sqrt{Re}$. The FVM wall strain, 12.3 s$^{-1}$ on the baseline cells, is lower and is not what the particles see.

| Parameter | Flow-scaled form | Baseline at $Re=150$ |
| --- | --- | --- |
| FVM cell size, particle spacing | $h/\delta_\omega$; spacing ratio $h_\mathrm{VPM}/h_\mathrm{FVM}$ | $0.15$; $1$ |
| FVM step | Courant number $C_\mathrm{FVM}=U\Delta t_\mathrm{FVM}/h_\mathrm{FVM}$; $U\Delta t_\mathrm{FVM}/D$ | $0.2$; $0.008$ |
| Particle step | $U\Delta t_\mathrm{VPM}/D$, with $\Delta t_\mathrm{VPM}=n\,\Delta t_\mathrm{FVM}$ | $0.04$ ($N=5$) |
| Particle stability | $\Lambda=\max\lvert S\rvert\,\Delta t_\mathrm{VPM}$, stable below 1 | $0.78$ |
| Coupling interval frequency | coupling intervals per convective time $L/(U\Delta t_\mathrm{VPM})=D/(U\Delta t_\mathrm{VPM})$ | $25$ |
| Interface widths | transfer inset, blend end depth, zero-weight layer, in particle core radii $\sigma$ and in $D$ | 8.75, 6 and 2 cells; $0.67D$ together |
| FVM domain, downstream | wake reach $\ell_\mathrm{wake}/L_f=(x_t-L_f)/L_f$, with $x_t$ the downstream transfer face | $x_t=4.05D$, $R\approx1.09$ |
| FVM domain, sides | clearance $\ell_\mathrm{side}/\delta_\omega=(m-w-D/2)/\delta_\omega$, with $m$ the side margin and $w=L_\mathrm{inset}+\delta_1$ the inset plus the blend end depth (the zero-weight layer lies inside the blend) | $\ell_\mathrm{side}\approx5.0$ |
| Coupling iteration | FVM solves per coupling interval $K$, or boundary-data tolerance | up to 6, $10^{-5}$ |

The box is set by two distances, the wake reach $\ell_\mathrm{wake}/L_f$ and the lateral clearance $\ell_\mathrm{side}/\delta_\omega$. Between $Re=100$ and $250$ the formation length falls from $3.0D$ to $1.22D$, while $\delta_\omega$ falls as $Re^{-1/2}$, so one margin in diameters cannot hold both fixed across Reynolds numbers. The case therefore takes the downstream box face separately (`--box-downstream`) from the side margin (`--box-margin`).

### 3.4 Run families and what each holds fixed

All families are runs of case 01 (`assets/run_sensitivity_study.sh`), each up to 55 s. $h_\mathrm{FVM}=h_\mathrm{VPM}=h$ unless stated, $w$ are the interface widths, and $L/(U\Delta t_\mathrm{VPM})=D/(U\Delta t_\mathrm{VPM})$.

| Family | Fixed | Varied | Isolates | Runs |
| --- | --- | --- | --- | --- |
| C, space and time | $C_\mathrm{FVM}=0.2$, $N=2$ | $h=0.08$, $0.0566$, $0.04$ m | what a user sees refining everything together | 3 coupled, 4 FVM only |
| X, space | $\Delta t_\mathrm{FVM}=0.004$ s, $N=4$, interface widths in metres | $h=0.08$, $0.0566$, $0.04$ m | spatial behaviour of the coupled solver | 3 coupled, 4 FVM only |
| T2, time | $h=0.04$ m, $N=2$ | $\Delta t_\mathrm{FVM}=0.016$, $0.008$, $0.004$ s | temporal behaviour of the coupled solver | 3 coupled, 4 FVM only |
| P, FVM step | $h=0.04$ m, $\Delta t_\mathrm{VPM}=0.04$ s | $N=5$, $10$ | FVM temporal error inside the coupling, coupling interval frequency fixed | 2 |
| E, coupling interval ratio | $h=0.04$ m, $\Delta t_\mathrm{FVM}=0.008$ s | $N=2$, $4$, $5$ | particle step and coupling interval frequency together | 3 |
| K, coupling iterations | baseline | $K=1$, $2$, $4$, tolerance-driven | coupling iteration | 4 |
| B, box | $N=5$, inset 8.75 cells | wake reach $\ell_\mathrm{wake}/L_f=0.35$–$2.3$ and lateral clearance $\ell_\mathrm{side}/\delta_\omega=1.2$–$6.1$, at $Re=150$ and $250$ | box error, flow-scaled | 9 |
| F, factorial | $h=0.04$ m, inset 8.75 cells | downstream face $3.36D$ / $4.4D$ $\times$ $N=2$ / $5$ | additivity of the FVM domain and transfer errors | 4 |
| R, Reynolds number | baseline rules | $Re=250$ at $N=5$ and at $N=4$ ($\Lambda$ matched) | transfer of the rules to another flow | 2 |

Which flow-scaled groups each family varies (bold: held fixed by construction, where a naive refinement would change it):

| Family | $h/\delta_\omega$ | $C_\mathrm{FVM}$ | $U\Delta t_\mathrm{FVM}/D$ | $U\Delta t_\mathrm{VPM}/D$, $L/(U\Delta t_\mathrm{VPM})$ | $\Lambda$ | $w/D$ | Box ($\ell_\mathrm{wake}/L_f$, $\ell_\mathrm{side}/\delta_\omega$) | $Re$ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| X | varied | varies with $h$ | fixed | fixed | fixed | **fixed** | fixed | fixed |
| T2 | fixed | varied | varied | varied ($N=2$) | varied | fixed | fixed | fixed |
| P | fixed | varied | varied | **fixed** | fixed | fixed | fixed | fixed |
| E | fixed | fixed | fixed | varied | varied | fixed | fixed | fixed |
| K | fixed | fixed | fixed | fixed | fixed | fixed | fixed | fixed |
| B | fixed | fixed | fixed | fixed | per $Re$ | fixed | varied in $L_f$ and $\delta_\omega$ | 150, 250 |
| R | fixed in $D$ | fixed | fixed | matched ($N=5$) or not ($N=4$) | matched ($N=4$) or not ($N=5$) | fixed | per $Re$ | varied |
| C | varied | fixed | varied | varied ($N=2$) | varied | varies (in cells) | fixed | fixed |

**Space.** In family X the inset, blend and zero-weight layer are counted in cells of 0.04 m (`--interface-cell-size 0.04`) at every level, so the interface stays in place and only the discretization of both solvers is refined. In family C the widths are counted in each level's own cells, as a user refining the whole method would do, and the interface moves towards the FVM domain faces with refinement.

**Time.** The coupler takes one particle step per coupling interval, so the particle step and the number of transfers per convective time cannot be varied apart: $L/(U\Delta t_\mathrm{VPM})=D/(U\Delta t_\mathrm{VPM})$. They are separated by functional form instead. A temporal error of the particle step grows with $\Delta t_\mathrm{VPM}$; an error added at each transfer accumulates with $L/(U\Delta t_\mathrm{VPM})$ and so grows as $\Delta t_\mathrm{VPM}$ falls. The model

$$e_\mathrm{coupling}=c_0+a\,\frac{U\Delta t_\mathrm{VPM}}{D}+b\,L/(U\Delta t_\mathrm{VPM})\;\left(+\,d\,\frac{U\Delta t_\mathrm{FVM}}{D}\right)$$

is fitted by linear least squares over families T2, P and E, all on the baseline cells. P holds $\Delta t_\mathrm{VPM}$ while $\Delta t_\mathrm{FVM}$ changes, E holds $\Delta t_\mathrm{FVM}$ while $\Delta t_\mathrm{VPM}$ changes, and T2 changes both at a fixed ratio. The FVM-step term $d$ is included when the runs leave enough freedom. A coefficient is called significant when it exceeds twice its standard error. Each driver is also fitted alone as $e_\mathrm{coupling}=c_0+a\,x^q$, with $q\in[-3,3]$, to show on which one the coupling error collapses.

**Factorial.** Family F is a full $2^2$ design (the inset factor of the plan was not run). The effect of a set of factors is the mean error where the product of their coded levels ($\pm1$) is $+1$ minus the mean where it is $-1$. The additivity defect of two changes A and B, $e(A,B)-[e(A)+e(B)-e(0)]$, is zero when they add their errors independently.

### 3.5 In-run indicator

At every coupling interval the coupler logs the mismatch between the FVM boundary data and the particle velocity on the FVM domain faces (mean, maximum and on the outflow faces, in units of $U$), and the normal-velocity and tangential-gradient residuals of the first and last coupling iteration (in $U$ and $U/D$). Their means over developed flow, $t\ge40$ s, are correlated with $\lvert e_\mathrm{coupling}\rvert$ over every run (`figures/error_indicator`, `figures/auxiliary/error_indicators.csv`). A good correlation gives users an a-posteriori estimate of the coupling error from their own run, without an FVM-only solution.

## 4. Procedure for a user's own flow

1. **Measure the flow scales.** Run the FVM only on a moderate domain, or a coarse coupled case, to developed flow. Measure the shear-layer vorticity thickness $\delta_\omega$ at separation, the formation length $L_f$ (centreline peak of $u_\mathrm{rms}$; the $v_\mathrm{rms}$ peak is less sensitive to the cells), and the largest strain rate of the particle field, $\max\lvert S\rvert$ (reported by the particle solver's stability check). Do not use the strain rate of the FVM wall cells, which is lower.
2. **Set the interface** (section 6):
   - lateral clearance $\ell_\mathrm{side}/\delta_\omega\ge5$ shear-layer thicknesses between the separated shear layer and the inner edge of the replacement region;
   - downstream transfer face at $x_t\ge L_f(1+R)$ with $\ell_\mathrm{wake}/L_f\ge1$;
   - transfer inset, blend end depth and zero-weight layer of about 9, 6 and 2 particle core radii;
   - the largest stable particle step, $\Lambda=\max\lvert S\rvert\,\Delta t_\mathrm{VPM}\approx0.8$ (stable up to 1.0 here), which keeps the coupling intervals per convective time $L/(U\Delta t_\mathrm{VPM})=D/(U\Delta t_\mathrm{VPM})$ low; the FVM step then follows from its own Courant number, $C_\mathrm{FVM}\le0.2$;
   - two coupling iterations per coupling interval, or a boundary-data tolerance of $10^{-5}$.
3. **Refine space** (family X): three or four levels of cell and particle size, ratio $\sqrt2$, at fixed FVM and particle steps, with the interface widths fixed in metres (`--interface-cell-size`).
4. **Refine time at a fixed particle step** (family P): three or four FVM steps under the particle step of step 2. Do not refine the particle step to show temporal convergence: at a fixed coupling interval ratio the coupling error grows as the particle step falls (section 5.4).
5. **Estimate order and uncertainty.** Apply the least-squares procedure of section 3.2 (`discretization_uncertainty` in `assets/error_analysis.py`) to each statistic of each family. Report an order only for monotone data; report $Q_\mathrm{ext}$ and $U$ in every case.
6. **Read the indicator.** Compare the developed mean of the boundary-data velocity mismatch, logged by the coupler at every coupling interval, with $3.5\times10^{-3}U$. Below it, the lift-amplitude coupling error stayed within 2.1% in every run of this study; above $5.5\times10^{-3}U$ it reached 5–7%. A high value points at the lateral clearance or at too many coupling intervals.
7. **Do not refine everything together** as the only check. Refining cells and steps together at a fixed coupling interval ratio mixes the FVM spatial and temporal errors, which have opposite signs for the lift amplitude, with a coupling error that grows with the number of coupling intervals.

## 5. Results

### 5.1 Flow scales

Figure: `figures/flow_scales`, the centreline mean and rms velocity at $Re=100$, 150 and 250 with the recirculation and formation lengths marked.

| $Re$ | $L_r/D$ | $L_f/D$ ($u_\mathrm{rms}$) | $L_{f,v}/D$ ($v_\mathrm{rms}$) | $\delta_\omega/D$ | $\delta_\omega\sqrt{Re}/D$ | FVM wall $\max\lvert S\rvert$ [s$^{-1}$] | $St$ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 100 | 1.46 | 3.00 | 3.08 | 0.316 | 3.16 | 9.4 | 0.163 |
| 150 | 1.06 | 1.94 | 2.28 | 0.263 | 3.23 | 12.3 | 0.182 |
| 250 | 0.74 | 1.22 | 1.75 | 0.214 | 3.39 | 16.9 | 0.201 |

The shear-layer thickness scales as $D/\sqrt{Re}$ ($\delta_\omega\sqrt{Re}/D=3.2\pm0.1$), and the FVM wall strain as $\approx(U/D)\sqrt{Re}$. The formation length falls by a factor of 2.5 from $Re=100$ to 250. At $Re=150$ the $u_\mathrm{rms}$ peak sits on a plateau and moves from $2.32D$ to $1.81D$ between $h=0.08$ and $0.0283$ m, while the $v_\mathrm{rms}$ peak moves only from $2.35D$ to $2.25D$ and the recirculation length from $1.13D$ to $1.05D$.

### 5.2 FVM-only convergence and $Q_\mathrm{ext}$ (families C, X, T)

Figure: `figures/error_references_C_X_T`.

| Statistic | Family | Class | Order | Fit, $F_s$ | $Q_\mathrm{ext}$ | $U$ at finest level [%] |
| --- | --- | --- | --- | --- | --- | --- |
| $\bar C_D$ | C | oscillatory | — | order 2, 3 | 1.3107 | 0.21 |
| $\bar C_D$ | X | oscillatory | — | order 1, 3 | 1.3113 | 0.55 |
| $\bar C_D$ | T | monotone | 0.26 | orders 1 and 2, 3 | 1.3040 | 0.34 |
| $\hat C_L$ | C | monotone | 2.05 | free, 1.25 | 0.5178 | 2.05 |
| $\hat C_L$ | X | monotone | 1.52 | free, 1.25 | 0.5236 | 3.70 |
| $\hat C_L$ | T | monotone | 0.92 | free, 1.25 | 0.4937 | 0.58 |
| $St$ | C | monotone | 1.75 | free, 1.25 | 0.1837 | 0.82 |
| $St$ | X | monotone | 1.66 | free, 1.25 | 0.1838 | 0.84 |
| $St$ | T | monotone | 1.71 | free, 1.25 | 0.1816 | 0.02 |

Family T extrapolates to zero step on the baseline cells, so its $Q_\mathrm{ext}$ keeps the spatial error of $h=0.04$ m. The space-time fit over the ten FVM-only levels determines only the Strouhal number: $p=1.70\pm0.08$, $q=2.0\pm1.0$, $Q_\mathrm{ext}=0.1838\pm0.0002$. For the mean drag and the lift amplitude its orders are undetermined (standard error of $Q_\mathrm{ext}$ larger than the family-C uncertainty), and $Q_\mathrm{ext}$ is taken from family C: $\bar C_D=1.3107$, $\hat C_L=0.5178$.

On the baseline cells and step, the FVM-only lift amplitude is 3.1% below $Q_\mathrm{ext}$; the spatial part of that error is negative (family X: −12.5%, −6.8%, −3.8% at $h=0.08$, 0.0566, 0.04 m with $\Delta t_\mathrm{FVM}=0.004$ s) and the temporal part positive (family T: the lift amplitude falls by 1.8% from $\Delta t_\mathrm{FVM}=0.008$ s to zero step). The FVM is first order in time for the lift amplitude, although its scheme is second order; the under-relaxed outer correctors are the likely cause, not tested here.

### 5.3 Error decomposition of the coupled solver (families C, X, T2)

Figure: `figures/error_decomposition_C_X_T2`.

| Family | Level | $e$: $\bar C_D$ / $\hat C_L$ / $St$ [%] | $e_\mathrm{FVM}$ | $e_\mathrm{coupling}$ | $e_\mathrm{coupling}$ wake velocity [%] |
| --- | --- | --- | --- | --- | --- |
| X | $h=0.08$ m | +1.28 / −9.57 / −0.24 | −0.47 / −12.53 / −3.69 | +1.76 / +2.97 / +3.46 | 5.27 |
| X | $h=0.0566$ m | −0.70 / −7.99 / −1.41 | −0.24 / −6.76 / −2.06 | −0.46 / −1.23 / +0.65 | 2.86 |
| X | $h=0.04$ m | −0.87 / −6.47 / −1.51 | −0.27 / −3.83 / −1.18 | −0.60 / −2.65 / −0.33 | 1.82 |
| T2 | $\Delta t_\mathrm{FVM}=0.016$ s | −0.11 / −3.18 / −1.11 | +0.06 / −1.70 / −1.34 | −0.17 / −1.47 / +0.23 | 3.17 |
| T2 | $\Delta t_\mathrm{FVM}=0.008$ s | −0.82 / −5.93 / −1.65 | −0.14 / −3.11 / −1.22 | −0.68 / −2.82 / −0.42 | 3.06 |
| T2 | $\Delta t_\mathrm{FVM}=0.004$ s | −2.66 / −10.91 / −3.77 | −0.27 / −3.83 / −1.18 | −2.39 / −7.08 / −2.59 | 5.70 |
| C | $h=0.08$ m | +1.30 / −9.92 / −0.90 | −0.38 / −11.86 / −4.00 | +1.68 / +1.93 / +3.10 | 4.68 |
| C | $h=0.0566$ m | −0.44 / −6.92 / −1.46 | −0.05 / −5.66 / −2.18 | −0.39 / −1.26 / +0.72 | 2.44 |
| C | $h=0.04$ m | −0.82 / −5.93 / −1.65 | −0.14 / −3.11 / −1.22 | −0.68 / −2.82 / −0.42 | 3.06 |

**Space (X).** The total error falls monotonically with $h$ for all three statistics, but more slowly than the FVM part: the extrapolated coupled values are $\bar C_D=1.286$, $\hat C_L=0.500$, $St=0.180$, which lie 1.9%, 4.5% and 2.1% below the FVM-only $Q_\mathrm{ext}$, with uncertainties of 3.6%, 9.3% and 2.2%. The coupling error changes sign between $h=0.08$ and $0.0566$ m and then grows slowly in magnitude, so it does not vanish with the cells. The free-order fits give no usable order (0.1 for the lift amplitude, about 7 for drag and Strouhal number, from three levels without residual freedom). What can be stated is that the coupled solver follows the FVM spatial convergence (order 1.5–1.7) towards an offset of a few per cent, set by the coupling error at the fixed particle step ($L/(U\Delta t_\mathrm{VPM})=63$ in this family).

**Time (T2).** The data are anomalous: the total error grows as the steps are refined, because the coupling error grows (−1.5%, −2.8%, −7.1% in lift amplitude) while the FVM part barely changes. At a fixed coupling interval ratio the coupled solver has no temporal order; section 5.4 identifies the cause.

**Space and time together (C).** The coupling error changes sign under refinement (+1.9%, −1.3%, −2.8% in lift amplitude), and the total error decreases only because the FVM part, which dominates on coarse cells, falls faster than the coupling part grows. A user refining this way would see apparent convergence.

**Space at a fixed particle step (XP).** Family X keeps $N$ fixed, so its particle step and $L/(U\Delta t_\mathrm{VPM})$ are those of $N=4$ at $\Delta t_\mathrm{FVM}=0.004$ s ($L/(U\Delta t_\mathrm{VPM})=63$), where the transfer error is large. Family XP holds $\Delta t_\mathrm{VPM}=0.04$ s ($L/(U\Delta t_\mathrm{VPM})=25$, $N=10$, $\Delta t_\mathrm{FVM}=0.004$ s) and the interface widths in metres:

| $h/D$ | $h/\delta_\omega$ | $e$: $\bar C_D$ / $\hat C_L$ / $St$ [%] | $e_\mathrm{FVM}$ | $e_\mathrm{coupling}$ |
| --- | --- | --- | --- | --- |
| 0.08 | 0.30 | +1.11 / −10.36 / −0.63 | −0.47 / −12.53 / −3.70 | +1.58 / +2.17 / +3.07 |
| 0.0566 | 0.22 | −0.50 / −7.67 / −1.07 | −0.24 / −6.77 / −2.07 | −0.26 / −0.90 / +1.00 |
| 0.04 | 0.15 | −0.34 / −5.18 / −0.83 | −0.28 / −3.83 / −1.18 | −0.06 / −1.35 / +0.35 |
| 0.0283 | 0.11 | inadmissible: $\Lambda=1.03$ at $\Delta t_\mathrm{VPM}=0.04$ s | | |

The total lift-amplitude error falls monotonically and stays dominated by the FVM part. The coupling error of the Strouhal number falls monotonically; that of the lift amplitude changes sign and reaches the value set by $L/(U\Delta t_\mathrm{VPM})=25$ on the baseline cells. The finest level stopped at its first particle step: on $h=0.0283$ m the largest particle-field strain rate is 25.8 s⁻¹ (19.4 s⁻¹ on $h=0.04$ m), so the fixed particle step exceeded the stability limit. A spatial study at a fixed coupling interval step must therefore choose that step for its finest level.

### 5.4 Drivers of the coupling error (families T2, P, E)

Figure: `figures/error_drivers_T2_P_E`, the coupling error against $U\Delta t_\mathrm{VPM}/D$ (with $\Lambda$) and against $L/(U\Delta t_\mathrm{VPM})$, for the six runs on the baseline cells.

| Statistic | $c_0$ [%] | $a$ | exponent $q$ in $e_\mathrm{coupling}=c_0-a\,L/(U\Delta t_\mathrm{VPM})^{q}$ | $R^2$ | transfer part at $L/(U\Delta t_\mathrm{VPM})=25$ (baseline) [%] |
| --- | --- | --- | --- | --- | --- |
| $\bar C_D$ | +0.09 | $7.6\times10^{-4}$ | 1.67 | 1.000 | −0.16 |
| $\hat C_L$ | −1.00 | $1.3\times10^{-3}$ | 1.75 | 1.000 | −0.36 |
| $St$ | +0.54 | $8.5\times10^{-4}$ | 1.70 | 1.000 | −0.20 |

The six runs separate the steps: E changes the particle step at a fixed FVM step, P changes the FVM step at a fixed particle step, and T2 changes both. The coupling error collapses on the particle step alone (or, equivalently, $L/(U\Delta t_\mathrm{VPM})$); fitted on the FVM step alone or on $N$, $R^2$ falls to 0.3. The P pair confirms it directly: halving the FVM step at $\Delta t_\mathrm{VPM}=0.04$ s changes the coupling error by +0.03 (drag), +0.07 (lift amplitude) and +0.02 (Strouhal number) points.

The error grows as the particle step falls, the opposite of a temporal error, and with an exponent of 1.7 in $L/(U\Delta t_\mathrm{VPM})$. It is an error added at each coupling interval (the transfer of FVM vorticity to the particles and the vorticity transfer of the transfer region) and accumulated over the coupling intervals in a convective time. The particle step and the transfer count are tied by design, so this attribution rests on the trend, not on independent variation. The consequence for the settings is direct: the particle step should be the largest that the particle stability allows ($\Lambda\approx0.8$–1.0), and the FVM step is refined under it. $c_0$ is the coupling error that fewer coupling intervals cannot remove; at the baseline it is −1.0% in lift amplitude and comes from the FVM domain and the interface (section 5.6).

### 5.5 Coupling iteration (family K)

Coupling error $e_\mathrm{coupling}$ at the baseline ($h=0.04$ m, $\Delta t_\mathrm{FVM}=0.008$ s, $N=5$), in per cent:

| FVM solves per coupling interval | $\bar C_D$ | $\hat C_L$ | $St$ |
| --- | --- | --- | --- |
| $K=1$ | +2.96 | +2.09 | +0.34 |
| $K=2$ | −0.23 | −1.52 | +0.33 |
| $K=4$ | −0.09 | −1.39 | +0.33 |
| Tolerance $10^{-5}$ (at most 6; mean 1.8) | −0.09 | −1.42 | +0.33 |

With a single solve per coupling interval the FVM boundary data lags the particle velocity, and the drag and lift-amplitude errors take the opposite sign and a larger magnitude. A second coupling iteration removes almost all of this. From $K=2$ to $K=4$ the drag error changes by 0.14 and the lift-amplitude error by 0.13 percentage points, and the tolerance-driven runs agree with $K=4$ to within 0.03 points. The Strouhal number does not depend on $K$. The coupling error that remains at $K\ge2$ is not an error of the coupling iteration. The tolerance-driven runs need 1.8 coupling iterations per coupling interval on average and so cost less than a fixed $K=2$. (The $K=3$ run stopped on a GPU memory shortage and was not repeated; $K=6$ was dropped for time.)

### 5.6 FVM domain (family B) and factorial (family F)

Figure: `figures/error_box_B`, the coupling error against the wake reach $\ell_\mathrm{wake}/L_f$ (with $L_f$ from the $u_\mathrm{rms}$ and the $v_\mathrm{rms}$ peak) and against the lateral clearance $\ell_\mathrm{side}/\delta_\omega$.

| $Re$ | Downstream face $[D]$ | Side margin $[D]$ | $\ell_\mathrm{wake}/L_f$ | $\ell_\mathrm{side}/\delta_\omega$ | $e_\mathrm{coupling}$: $\bar C_D$ / $\hat C_L$ / $St$ [%] |
| --- | --- | --- | --- | --- | --- |
| 150 | 2.97 | 2.4 | 0.35 | 4.97 | +0.49 / +0.69 / +0.66 |
| 150 | 3.36 | 2.4 | 0.55 | 4.97 | +0.20 / +0.68 / +0.62 |
| 150 | 4.4 | 2.4 | 1.08 | 4.97 | −0.09 / −1.42 / +0.33 |
| 150 | 4.4 | 1.8 | 1.08 | 2.70 | −0.09 / −2.00 / +0.37 |
| 150 | 4.4 | 1.6 | 1.08 | 1.94 | −0.48 / −3.57 / +0.51 |
| 150 | 4.4 | 1.4 | 1.08 | 1.18 | −0.44 / −5.37 / +0.71 |
| 250 | 1.99 | 2.4 | 0.35 | 6.11 | +0.19 / −2.14 / +0.83 |
| 250 | 2.24 | 2.4 | 0.55 | 6.11 | +0.23 / −1.60 / +0.85 |
| 250 | 2.9 | 2.4 | 1.10 | 6.11 | +0.57 / −0.64 / +0.81 |
| 250 | 4.4 | 2.4 | 2.33 | 6.11 | −0.14 / −1.93 / +0.53 |
| 250 | 4.4 | 1.68 | 2.33 | 2.75 | −0.57 / −3.25 / +0.66 |
| 250 | 4.4 | 1.52 | 2.33 | 2.01 | −0.81 / −4.36 / +0.73 |
| 250 | 4.4 | 1.36 | 2.33 | 1.26 | −1.50 / −6.34 / +0.74 |

**Lateral clearance.** Bringing the replacement region to within $1.2\,\delta_\omega$ of the separated shear layer changes the lift-amplitude coupling error by −3.95 points at $Re=150$ and −4.41 points at $Re=250$, and the drag error by −0.35 and −1.36 points. The clearance in shear-layer thicknesses collapses the lift-amplitude effect at the two Reynolds numbers; it is the FVM domain parameter that sets the largest error. Both Reynolds numbers lie in laminar periodic shedding, so this shows the right length within that regime only. With four clearances per Reynolds number the excess over the widest box decays as 4.0 / 2.2 / 0.6 points ($Re=150$, $\ell_\mathrm{side}/\delta_\omega=1.2$ / 1.9 / 2.7) and 4.4 / 2.4 / 1.3 points ($Re=250$, $\ell_\mathrm{side}/\delta_\omega=1.3$ / 2.0 / 2.75). A clearance of $\ell_\mathrm{side}\approx5$ makes the lateral error small; $\ell_\mathrm{side}\approx2.7$ costs 0.6–1.3 points.

**Downstream wake reach.** For $\ell_\mathrm{wake}/L_f\ge0.35$ the wake reach changes the lift-amplitude coupling error by at most about 2 points, and not in the same way at the two Reynolds numbers: at $Re=150$ shortening the FVM domain moves the error from −1.42% to +0.7%, at $Re=250$ the error varies between −2.1% and −0.6% without a monotone trend. Neither choice of formation length collapses the two. At $Re=150$ the short box looks better only because its positive error cancels the negative transfer error: a box judged by its total error alone would be chosen too small. No minimum wake reach was found within $0.35\le R\le2.33$; the downstream face should lie beyond the formation point, and the wake reach should not be chosen by the size of the total error.

**Factorial.** Coupling error $e_\mathrm{coupling}$ in lift amplitude (drag, Strouhal number in brackets), in per cent:

| | $N=5$ ($L/(U\Delta t_\mathrm{VPM})=25$) | $N=2$ ($L/(U\Delta t_\mathrm{VPM})=63$) |
| --- | --- | --- |
| Downstream face 4.4 D | −1.42 (−0.09, +0.33) | −2.82 (−0.68, −0.42) |
| Downstream face 3.36 D | +0.68 (+0.20, +0.62) | +2.02 (+0.05, +0.30) |

The additivity defects are 2.75 points in lift amplitude and 0.44 and 0.43 points in drag and Strouhal number. In the short box more coupling intervals push the lift-amplitude error further positive, in the long box further negative: the coupling intervals amplify the error that the FVM domain sets instead of adding their own. In both boxes they lower the Strouhal number. The box must therefore be sized before the coupling interval step is chosen.

### 5.7 Transfer to another Reynolds number (family R)

| $Re$ | $N$ | $\Lambda$ | $L/(U\Delta t_\mathrm{VPM})$ | $\ell_\mathrm{wake}/L_f$ | $\ell_\mathrm{side}/\delta_\omega$ | $e_\mathrm{coupling}$: $\bar C_D$ / $\hat C_L$ / $St$ [%] |
| --- | --- | --- | --- | --- | --- | --- |
| 150 | 5 | 0.78 | 25 | 1.08 | 4.97 | −0.09 / −1.42 / +0.33 |
| 250 | 5 | 1.00 | 25 | 2.33 | 6.11 | −0.14 / −1.93 / +0.53 |
| 250 | 4 | 0.80 | 31 | 2.33 | 6.11 | −0.19 / −2.01 / +0.50 |
| 250 | 2 | 0.40 | 63 | 2.33 | 6.11 | −0.49 / −2.64 / +0.24 |
| 150 | 2 | 0.31 | 63 | 1.08 | 4.97 | −0.68 / −2.82 / −0.42 |
| 100 | 5 | 0.63 | 25 | 0.35 | 4.14 | +0.06 / +0.34 / +0.18 |
| 100 | 2 | 0.25 | 63 | 0.35 | 4.14 | −0.55 / +0.38 / −0.84 |

At $Re=250$ no extrapolated value is available, so these are coupling errors against the FVM-only run of the same Reynolds number, cells and step. The particle step at $\Lambda=1.0$ is still stable. Matching $\Lambda$ ($N=4$) does not reproduce the error of $Re=150$; matching $L/(U\Delta t_\mathrm{VPM})$ ($N=5$) comes slightly closer, and the change from $N=5$ to $N=4$ at $Re=250$ (−0.08 points in lift amplitude) is close to the $L/(U\Delta t_\mathrm{VPM})$ trend fitted at $Re=150$ (−0.17 points). The remaining offset of about −0.5 points in lift amplitude is not explained by $\Lambda$, $L/(U\Delta t_\mathrm{VPM})$, the wake reach or the clearance; at $Re=250$ the FVM domain is larger in flow scales than at $Re=150$. All $Re=250$ errors stay within the 2% target in drag and Strouhal number, and at the limit of it in lift amplitude. From $N=5$ to $N=2$ the drag falls by 0.62, 0.59 and 0.35 points and the Strouhal number by 1.01, 0.75 and 0.29 points at $Re=100$, 150 and 250; the lift amplitude changes by +0.04, −1.40 and −0.71 points. The fitted exponent of section 5.4 holds at $Re=150$ only; the direction of the trend holds at all three.

### 5.8 Transfer inset and repeatability

| Run | $e_\mathrm{coupling}$: $\bar C_D$ / $\hat C_L$ / $St$ [%] |
| --- | --- |
| Inset 4 spacings | −0.15 / −1.67 / +0.28 |
| Inset 8.75 spacings (baseline) | −0.09 / −1.42 / +0.33 |
| Inset 14 spacings | −0.04 / −1.29 / +0.36 |
| Baseline repeated | −0.09 / −1.41 / +0.32 |

A deeper inset rewrites a smaller volume at each coupling interval at an unchanged number of remeshings, and it reduces the drag and lift-amplitude errors monotonically. Vorticity transfer therefore contributes to the error added per coupling interval; the GBD remeshing contribution is not measured separately. Over the range tested the inset changes the lift-amplitude error by 0.38 points, a quarter of the change from $L/(U\Delta t_\mathrm{VPM})=25$ to 63. The default of 8.75 spacings stays; 14 gains 0.13 points at the cost of a smaller unblended core. The repeat agrees with the baseline within 0.01 points in all three statistics, so the run-to-run scatter from the GPU summation order is far below every difference reported here.

### 5.9 In-run indicator

Figure: `figures/error_indicator`, $\lvert e_\mathrm{coupling}\rvert$ against the developed mean of the boundary-data velocity mismatch for all 36 coupled runs.

| Indicator (mean over $t\ge40$ s) | Correlation with $\lvert e_\mathrm{coupling}\rvert$: $\bar C_D$ / $\hat C_L$ / $St$ / wake |
| --- | --- |
| Boundary velocity mismatch, mean | 0.40 / **0.82** / 0.36 / 0.46 |
| Boundary velocity mismatch, maximum | 0.45 / 0.72 / 0.28 / 0.29 |
| Outflow mismatch, mean | 0.37 / 0.38 / 0.29 / 0.21 |
| First-coupling iteration normal residual | −0.14 / −0.09 / −0.13 / −0.13 |
| First-coupling iteration gradient residual | −0.14 / −0.07 / −0.14 / −0.13 |
| Last-coupling iteration gradient residual | 0.07 / −0.03 / −0.09 / −0.12 |

The boundary-data mismatch follows the lift-amplitude coupling error over every family: below $3.5\times10^{-3}U$ all 16 runs have $\lvert e_\mathrm{coupling}\rvert\le2.1\%$, and the five runs above $5\times10^{-3}U$ (side clearances of $1.2$–$2.75\,\delta_\omega$ and $L/(U\Delta t_\mathrm{VPM})=125$) have 3.2–7.1%. The interface-coupling iteration residuals do not indicate the coupling error.

## 6. Recommendations

Values in flow scales, for a bluff body shedding a laminar wake, together with the conditions established in the earlier study: an outflow fringe of at least two wake wavelengths at the particle-domain outflow, an FVM-only comparison with the same unbounded far field, and converged FVM linear solves (momentum tolerance $10^{-9}$).

| Parameter | Flow-scaled form | Recommended value | Evidence |
| --- | --- | --- | --- |
| Side margin | clearance $\ell_\mathrm{side}/\delta_\omega=(m-w-D/2)/\delta_\omega$ | $\ell_\mathrm{side}/\delta_\omega\ge5$; $\ell_\mathrm{side}\approx2.7$ costs 0.6–1.3 points, $\ell_\mathrm{side}\approx2$ about 2.3 and $\ell_\mathrm{side}\approx1.2$ about 4 points of lift amplitude at both $Re$ | B |
| Downstream transfer face | wake reach $\ell_\mathrm{wake}/L_f=(x_t-L_f)/L_f$ | beyond the formation point; $0.35\le R\le2.3$ changed the error by up to 2 points without a trend; never choose it by the total error | B, F |
| Transfer inset, blend end depth, zero-weight layer | in particle core radii $\sigma=h_\mathrm{VPM}$ | 8.75, 6 and 2 (the defaults); the inset changed the lift-amplitude error by 0.38 points between 4 and 14 | inset runs |
| Particle step | $\Lambda=\max\lvert S\rvert\,\Delta t_\mathrm{VPM}$, with the particle-field strain | the largest stable value on the finest cells, $\Lambda\approx0.8$ (1.0 still stable, 1.03 stopped); $\max\lvert S\rvert$ grows by a third per refinement by $\sqrt2$ | E, T2, P, R, XP |
| Coupling intervals per convective time | $L/(U\Delta t_\mathrm{VPM})=D/(U\Delta t_\mathrm{VPM})$ | as low as $\Lambda$ allows; the transfer part of $e_\mathrm{coupling}$ grows as $L/(U\Delta t_\mathrm{VPM})^{1.7}$ (−0.4% at $L/(U\Delta t_\mathrm{VPM})=25$, −1.8% at 63, −6.1% at 125 in lift amplitude) | E, T2, P |
| FVM step | Courant number $C_\mathrm{FVM}$ | $C_\mathrm{FVM}\le0.2$; under a fixed particle step it does not change $e_\mathrm{coupling}$ | P |
| Cell size | $h/\delta_\omega$ | 0.15 gives an FVM error of 3% in lift amplitude and 1.2% in Strouhal number; the coupling error does not fall with $h$ | X, C |
| Coupling iteration | FVM solves per coupling interval, or boundary-data tolerance | tolerance $10^{-5}$ (1.8 coupling iterations on average), or $K=2$ | K |
| Indicator | developed mean boundary velocity mismatch | below $3.5\times10^{-3}U$ | all |

Overall behaviour of the coupled solver:

| Statistic | Space (X): FVM part / coupled total | Time | $e$ at baseline [%] | $U$ of $Q_\mathrm{ext}$ [%] |
| --- | --- | --- | --- | --- |
| $\bar C_D$ | converged within 0.6% / XP: −0.34% at $h=0.04$, coupling part −0.06% | no order at fixed $N$; FVM part order 0.3 | −0.22 | 0.21 |
| $\hat C_L$ | order 1.5 / XP: monotone, −5.2% at $h=0.04$, coupling part → −1.4% | no order at fixed $N$; FVM part order 0.9 | −4.53 | 2.05 |
| $St$ | order 1.7 / XP: coupling part falls monotonically to +0.35% | no order at fixed $N$; FVM part order 1.7 | −0.90 | 0.82 |

The coupled solver inherits the spatial order of the FVM and, at a fixed particle step, its temporal order; it adds a coupling error that does not fall with the cells and that grows with the number of coupling intervals. Its order as a whole is that of the FVM up to this coupling floor, which at the recommended settings is about 1% in lift amplitude ($c_0$) plus 0.4% for the coupling intervals.

## 7. Limitations

- **Three levels per coupled family.** Families X and T2 have three levels each. The free-order fit then has no residual freedom, the combined first- and second-order fit cannot be made, and the uncertainty rests on the safety factor alone. The offsets of family X carry uncertainties of 2–9%.
- **One body, one regime.** Only the circular cylinder in laminar periodic shedding ($Re=100$–$250$) was studied. The rules are expressed in lengths of the wake so that they can be tested elsewhere; the cube and wing tutorials are those tests. Further Reynolds numbers in the same regime would not show generality.
- **Three Reynolds numbers.** The box study covers $Re=150$ and $250$ and the coupling interval study $Re=100$, 150 and 250, all in laminar planar shedding. The $L/(U\Delta t_\mathrm{VPM})$ exponent was fitted at $Re=150$ only.
- **Not run.** The particle-spacing family ($h_\mathrm{VPM}=h_\mathrm{FVM}/2$), $K=3$ and 6, $N=6$, 20 and 40 and the inset factor of the factorial were dropped to finish within the week. The finest XP level was inadmissible at its particle step, and a second repeat stopped on a GPU memory shortage. The blend end depth and zero-weight layer were not varied.
- **Wake velocity against the finest FVM-only run.** The wake-velocity errors are distances to the finest FVM-only run, not to an extrapolated value.
- **Limits of the uncertainty estimate.** The least-squares estimate degrades for shallow orders and noisy data: in a synthetic check with $p=1$ and 0.1% noise, $\pm U$ covered the true error in 67–84% of cases. The statistics are averaged over $40\le t\le55$ s, about 2.7 shedding periods.
- **Particle step and transfer count tied.** The coupler takes one particle step per coupling interval, so the temporal and the per-transfer parts of the coupling error are separated only by their trends, not by independent variation.
- **Cost.** Most runs shared the GPU with another run, so their wall times are not comparable and no cost conclusion is drawn.
- **Repeatability.** One repeat of the baseline agreed within 0.01 points; scatter of other configurations was not measured.
