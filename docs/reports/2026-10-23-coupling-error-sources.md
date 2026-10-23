# Sources of error in the coupled FVM–VPM solver: the planar cylinder at $Re=150$

Technical report, 2026-10-23. Study plan: [2026-10-19-coupling-error-sources](../plans/2026-10-19-coupling-error-sources.md). Case: [tutorials/coupled_fvm_vpm/01_cylinder_shedding_flow](../../tutorials/coupled_fvm_vpm/01_cylinder_shedding_flow/README.md), whose `./allplot.sh` draws every figure and writes every table cited here (figures in `figures/`, tables in `figures/auxiliary/`). Earlier findings: [coupling_findings.md](../../tutorials/coupled_fvm_vpm/01_cylinder_shedding_flow/assets/coupling_findings.md).

Placeholders for results that are still being computed are marked **[RESULT: ...]**.

## 1. Summary

The coupled error of the planar cylinder splits into the error of the FVM discretization, which the coupled solver inherits, and a coupling error that the FVM-only solution does not have. Measured against the FVM-only value extrapolated to zero cell size and step, with every parameter in flow scales and one parameter varied at a time:

- **The coupling error is an error added at each transfer.** On the baseline cells it follows the number of exchanges per convective time, $N_x=D/(U\Delta t_p)$, as $e_c=c_0-a\,N_x^{1.7\text{–}1.75}$ with $R^2=1.000$ over six runs that vary the particle step and the FVM step independently. Halving the FVM step at a fixed particle step leaves it unchanged (lift amplitude −1.42% at $n=5$, −1.35% at $n=10$).
- **The coupled solver therefore has no temporal order at a fixed exchange ratio.** Refining both steps together ($n=2$) makes the coupling error grow: −1.5%, −2.8%, −7.1% in lift amplitude at $N_x=31$, 63, 125. The coupled solver converges in time only with the particle step held at its largest stable value and the FVM step refined under it; it then inherits the FVM temporal order.
- **Spatially, the coupled solution converges with the FVM towards a value offset by the transfer error.** At fixed steps and a fixed interface, the total lift-amplitude error is −9.6%, −8.0%, −6.5% at $h/D=0.08$, 0.0566, 0.04 and extrapolates to −4.5% ($\pm$9%); the FVM part converges with order 1.5 and the coupling part changes sign. Three levels do not give a reliable order for the coupled solver; the data are consistent with the FVM spatial order plus a coupling offset set by $N_x$.
- **FVM only:** spatial order 1.5–2.0 (lift amplitude) and 1.7 (Strouhal number), temporal order 0.9 (lift amplitude) and 1.7 (Strouhal number), mean drag converged within 0.2–0.6%. For the lift amplitude the spatial error is negative and the temporal error positive, so refining both together hides part of each.
- **Error budget at the baseline** ($h/\delta_\omega=0.15$, $C_f=0.2$, $n=5$, $\Lambda=0.78$, $N_x=25$): total $e$ = −0.22% (drag), −4.53% (lift amplitude), −0.90% (Strouhal number), of which the FVM part is −0.14%, −3.11%, −1.22% and the coupling part −0.09%, −1.42%, +0.33%. The coupling part is $c_0$ (−1.0% in lift amplitude, not removed by fewer transfers) plus the transfer part (−0.4% at $N_x=25$).
- **FVM box:** the side clearance, in shear-layer thicknesses $\delta_\omega$, is the box parameter that matters and that carries over between Reynolds numbers: at $c\approx1.2\,\delta_\omega$ the lift-amplitude coupling error grows by 4.0 points at $Re=150$ and 4.4 points at $Re=250$. The downstream reach in formation lengths changes the error by at most about 2 points for $R\ge0.35$, without a common trend at the two Reynolds numbers. A too-short downstream box looks more accurate at $Re=150$ (+0.7% against −1.4%) because its error cancels the transfer error.
- **Interface iteration:** two sweeps per exchange, or a boundary-trace tolerance of $10^{-5}$ (1.8 sweeps on average), keep the iteration error below 0.15 points; one sweep costs +3.0% in drag and +2.1% in lift amplitude.
- **In-run indicator:** the developed mean of the boundary-trace velocity mismatch correlates with the lift-amplitude coupling error ($r=0.84$ over 23 runs); every run with a mismatch below $3.5\times10^{-3}U$ has $\lvert e_c\rvert\le2.1\%$ in lift amplitude. It is a weaker indicator for drag and Strouhal number ($r\approx0.4$).
- **Another Reynolds number:** at $Re=250$ with the same rules the coupling error is −0.14% (drag), −1.93% (lift amplitude), +0.53% (Strouhal number) at $n=5$ ($\Lambda=1.0$, stable) and −0.19%, −2.01%, +0.50% at $n=4$ ($\Lambda=0.8$). Matching $\Lambda$ does not match the error of $Re=150$; matching $N_x$ comes closer, with an offset of about −0.5 points in lift amplitude.

## 2. The question, and why naive refinement misleads

The coupled solver couples a finite-volume (FVM) solution of the near body, inside a box, to one layer of vortex particles between slip planes that carries the outer wake. At every exchange the particles supply the FVM boundary condition on the box faces, and FVM vorticity replaces the particles inside the transfer region. A coupled result therefore carries the discretization errors of both solvers and an error of the coupling itself: of the interface position, of the transfer, of the boundary-trace iteration and of the exchange frequency. The study sets out to

1. give users an objective procedure, free of a choice of favourable levels, to measure the spatial and temporal order of their own coupled problem and the uncertainty of a result;
2. isolate each source of error, with every parameter expressed in scales of the flow, so that a recommended value does not rest on two errors cancelling at one setup;
3. state flow-agnostic values for the coupling-interface parameters, and the error and the spatial and temporal orders of the coupled solver as a whole, independent of the orders of the FVM and the VPM on their own.

Two features of the earlier sensitivity study made its conclusions unsafe.

**Same-cell comparison.** The coupled runs were compared with an FVM-only run on the same cells and step. That comparison removes the FVM discretization error shared by both runs and measures only the coupling error $e_c$. It does not show how far either run is from the solution of the equations. In the preliminary runs of the plan, the FVM-only lift amplitude on the baseline cells ($h=0.04$ m) lies about 4% below its extrapolated value, so the 2% target had been met against an unconverged value. The present study measures every error against $\phi_0$, the FVM-only value extrapolated to zero cell size and step, and splits it into an FVM part and a coupling part (section 3.1).

**Cancelling errors.** Refining cells, steps and particles together at constant Courant number is what a user would do first. It mixes errors of opposite sign. For the lift amplitude, the FVM spatial error is negative and the temporal error positive, so their sum can be small, or change sign, on a level where neither is small. In the preliminary runs at $C_f=0.2$ and $n=2$, the coupling error in lift amplitude changes sign under refinement (+1.9%, −1.3%, −2.8% at $h=0.08$, $0.0566$, $0.04$ m, against the extrapolated value): at least two contributions of opposite sign are present, and an order fitted to such data has no meaning. The study therefore refines space and time separately (families X and T2) and keeps family C only as the check a user sees when refining everything together.

A third weakness was that each parameter had been varied about one baseline and expressed in diameters or cells. Neither is a scale of the flow, so a value found at $Re=150$ need not carry over, and interactions between parameters were not seen.

## 3. Method

### 3.1 Case and error measures

The flow is the planar cylinder at $Re=U D/\nu=150$, with $D=1$ m and $U=1$ m/s. In the coupled case the FVM resolves the wall and the near wake in the box $[-2.4,4.4]\times[-2.4,2.4]$ m, with cells of $h=0.04$ m and a step $\Delta t_f=0.008$ s at the baseline; one layer of vortex particles of spacing $h_p=h$ between slip planes carries the wake to $x=30$ m, through an outflow fringe over $x\in[15,30]$ m. The FVM-only case solves the same problem on a large, fully meshed domain $[-40,60]\times[-60,60]$ m with the same near-body cells; it is called *FVM only* throughout.

Each statistic $\phi$ is taken over developed shedding, $40\le t\le55$ s: the mean drag $\bar C_D$, the lift amplitude $\hat C_L=\tfrac12(\max C_L-\min C_L)$, the Strouhal number $St$ from the upward zero crossings of $C_L$, and the phase-averaged wake velocity at $x/D=2$ and $4$. For a coupled run on cells $h$ with FVM step $\Delta t_f$, giving $\phi_c$, and the FVM-only run on the same cells and step, giving $\phi_r(h,\Delta t_f)$:

- total error $e=\phi_c-\phi_0$;
- FVM discretization error $e_f=\phi_r(h,\Delta t_f)-\phi_0$;
- coupling error $e_c=\phi_c-\phi_r(h,\Delta t_f)$, so that $e=e_f+e_c$.

All three are given in per cent of $\phi_0$. The value $\phi_0$ comes from the space-time fit of section 3.2 over every FVM-only level when that fit determines $\phi_0$ with a standard error smaller than the smallest family-C uncertainty; otherwise it is the extrapolation of FVM-only family C (four levels, refinement ratio $\sqrt2$, constant Courant number). A statistic that barely changes with the cells or the step, such as the mean drag, leaves the orders of the space-time fit undetermined, and its family-C value is used. The wake velocity has no scalar to extrapolate: its total and FVM errors are rms distances, at equal shedding phase, to the finest FVM-only run, and its coupling error the distance to the FVM-only run on the same cells and step.

### 3.2 Order and uncertainty

The order and the uncertainty follow the least-squares procedure of Eça and Hoekstra (J. Comput. Phys. 262, 2014, 104–130), implemented in `assets/error_analysis.py`. A family is a set of $n_g\ge3$ runs that varies one refinement variable $x$ ($h/D$ or $U\Delta t/D$) with everything else fixed.

**Fits.** Four forms are fitted to the levels $(x_i,\phi_i)$:

- free order, $\phi_i=\phi_0+\alpha x_i^p$, with $p$ searched in $[0.05,8]$;
- first order, $\phi_i=\phi_0+\alpha x_i$;
- second order, $\phi_i=\phi_0+\alpha x_i^2$;
- first and second order, $\phi_i=\phi_0+\alpha_1x_i+\alpha_2x_i^2$, only when it has fewer coefficients than levels.

Each form is fitted unweighted and with weights $w_i\propto1/x_i$, which favour the finest levels, and the weighting with the smaller standard deviation $\sigma$ of the fit is kept.

**Convergence class.** From the signs of the successive differences of $\phi$ towards $x\to0$:

- *monotone*: all differences of one sign, and the free-order fit converges (its order is not at the lower search bound);
- *oscillatory*: the differences alternate in sign;
- *anomalous*: otherwise, including monotone data whose free-order fit does not converge.

**Choice of fit and safety factor.**

- Monotone data with $0.5\le p<2.1$: the free-order fit, with safety factor $F_s=1.25$.
- Every other case: the fixed-order fit with the smallest $\sigma$, with $F_s=3$. For monotone data with $p\ge2.1$ the first-order fit is excluded.

**Uncertainty.** With the fitted error $\varepsilon_i=\phi_i-\phi_0$ of each level, the deviation $\lvert\phi_i-\phi_\mathrm{fit}(x_i)\rvert$ and the data range $\Delta_\phi=(\phi_\mathrm{max}-\phi_\mathrm{min})/(n_g-1)$:

$$U_i=F_s\lvert\varepsilon_i\rvert+\sigma+\lvert\phi_i-\phi_\mathrm{fit}(x_i)\rvert\quad\text{if }\sigma<\Delta_\phi,$$

$$U_i=3\,\frac{\sigma}{\Delta_\phi}\left(\lvert\varepsilon_i\rvert+\sigma+\lvert\phi_i-\phi_\mathrm{fit}(x_i)\rvert\right)\quad\text{otherwise,}$$

the second form marking data that are not well behaved.

**When no order is reported.** An observed order is reported only for monotone data. For oscillatory or anomalous data the report gives the class, $\phi_0$ and $U$ of the chosen fixed-order fit, and no order. A family with fewer than three developed levels gives neither.

**Space and time together.** Over every FVM-only level at $Re=150$, $\phi=\phi_0+a\,h^p+b\,\Delta t^q$ separates the spatial and the temporal order. The orders are first located on a grid $p,q\in[0.5,4]$ with the linear coefficients solved at each pair, then all five parameters are refined by nonlinear least squares; their standard errors follow from the Jacobian and the residual variance. The fit needs at least six levels.

### 3.3 Normalization by flow scales

Every parameter is expressed in a scale of the flow, measured on the FVM-only runs by `assets/flow_scales.py` (`figures/flow_scales`, `figures/auxiliary/flow_scales.csv`):

- vortex-formation length $L_f$, from the cylinder centre to the centreline peak of the rms streamwise velocity: $1.94D$ at $Re=150$ (uncertain by about $\pm0.15D$), $1.22D$ at $Re=250$, $3.0D$ at $Re=100$;
- vorticity thickness of the separated shear layer at the base plane, $\delta_\omega=\Delta u/\max\lvert\partial\bar u/\partial y\rvert\approx3.2D/\sqrt{Re}$;
- the largest strain rate the particle field sees, $\max\lvert S\rvert=19.4$ s$^{-1}$ at $Re=150$, scaling as $\sqrt{Re}$. The FVM wall strain, 12.3 s$^{-1}$ on the baseline cells, is lower and is not what the particles see.

| Parameter | Flow-scaled form | Baseline at $Re=150$ |
| --- | --- | --- |
| FVM cell size, particle spacing | $h/\delta_\omega$; spacing ratio $h_p/h_f$ | $0.15$; $1$ |
| FVM step | Courant number $C_f=U\Delta t_f/h_f$; $U\Delta t_f/D$ | $0.2$; $0.008$ |
| Particle step | $U\Delta t_p/D$, with $\Delta t_p=n\,\Delta t_f$ | $0.04$ ($n=5$) |
| Particle stability | $\Lambda=\max\lvert S\rvert\,\Delta t_p$, stable below 1 | $0.78$ |
| Exchange frequency | exchanges per convective time $N_x=D/(U\Delta t_p)$ | $25$ |
| Interface widths | transfer inset, blend width, particle-only band, in particle core radii $\sigma$ and in $D$ | 8.75, 6 and 2 cells; $0.67D$ together |
| FVM box, downstream | reach $R=(x_t-L_f)/L_f$, with $x_t$ the downstream transfer face | $x_t=4.05D$, $R\approx1.09$ |
| FVM box, sides | clearance $c=(m-w-D/2)/\delta_\omega$, with $m$ the side margin and $w=L_T+\delta_{\rm buf}$ the inset plus the blend width (the particle-only band lies inside the blend) | $c\approx5.0$ |
| Interface iteration | FVM solves per exchange $K$, or boundary-trace tolerance | up to 6, $10^{-5}$ |

The box is set by two distances, the downstream reach $R$ and the side clearance $c$. Between $Re=100$ and $250$ the formation length falls from $3.0D$ to $1.22D$, while $\delta_\omega$ falls as $Re^{-1/2}$, so one margin in diameters cannot hold both fixed across Reynolds numbers. The case therefore takes the downstream box face separately (`--box-downstream`) from the side margin (`--box-margin`).

### 3.4 Run families and what each holds fixed

All families are runs of case 01 (`assets/run_sensitivity_study.sh`), each up to 55 s. $h_f=h_p=h$ unless stated, $w$ are the interface widths, and $N_x=D/(U\Delta t_p)$.

| Family | Fixed | Varied | Isolates | Runs |
| --- | --- | --- | --- | --- |
| C, space and time | $C_f=0.2$, $n=2$ | $h=0.08$, $0.0566$, $0.04$ m | what a user sees refining everything together | 3 coupled, 4 FVM only |
| X, space | $\Delta t_f=0.004$ s, $n=4$, interface widths in metres | $h=0.08$, $0.0566$, $0.04$ m | spatial behaviour of the coupled solver | 3 coupled, 4 FVM only |
| T2, time | $h=0.04$ m, $n=2$ | $\Delta t_f=0.016$, $0.008$, $0.004$ s | temporal behaviour of the coupled solver | 3 coupled, 4 FVM only |
| P, FVM step | $h=0.04$ m, $\Delta t_p=0.04$ s | $n=5$, $10$ | FVM temporal error inside the coupling, exchange frequency fixed | 2 |
| E, exchange ratio | $h=0.04$ m, $\Delta t_f=0.008$ s | $n=2$, $4$, $5$ | particle step and exchange frequency together | 3 |
| K, interface sweeps | baseline | $K=1$, $2$, $4$, tolerance-driven | interface iteration | 4 |
| B, box | $n=5$, inset 8.75 cells | downstream reach $R=0.35$–$2.3$ and side clearance $c=1.2$–$6.1$, at $Re=150$ and $250$ | box error, flow-scaled | 9 |
| F, factorial | $h=0.04$ m, inset 8.75 cells | downstream face $3.36D$ / $4.4D$ $\times$ $n=2$ / $5$ | additivity of the box and transfer errors | 4 |
| R, Reynolds number | baseline rules | $Re=250$ at $n=5$ and at $n=4$ ($\Lambda$ matched) | transfer of the rules to another flow | 2 |

Which flow-scaled groups each family varies (bold: held fixed by construction, where a naive refinement would change it):

| Family | $h/\delta_\omega$ | $C_f$ | $U\Delta t_f/D$ | $U\Delta t_p/D$, $N_x$ | $\Lambda$ | $w/D$ | Box ($R$, $c$) | $Re$ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| X | varied | varies with $h$ | fixed | fixed | fixed | **fixed** | fixed | fixed |
| T2 | fixed | varied | varied | varied ($n=2$) | varied | fixed | fixed | fixed |
| P | fixed | varied | varied | **fixed** | fixed | fixed | fixed | fixed |
| E | fixed | fixed | fixed | varied | varied | fixed | fixed | fixed |
| K | fixed | fixed | fixed | fixed | fixed | fixed | fixed | fixed |
| B | fixed | fixed | fixed | fixed | per $Re$ | fixed | varied in $L_f$ and $\delta_\omega$ | 150, 250 |
| R | fixed in $D$ | fixed | fixed | matched ($n=5$) or not ($n=4$) | matched ($n=4$) or not ($n=5$) | fixed | per $Re$ | varied |
| C | varied | fixed | varied | varied ($n=2$) | varied | varies (in cells) | fixed | fixed |

**Space.** In family X the inset, blend and particle-only widths are counted in cells of 0.04 m (`--interface-cell-size 0.04`) at every level, so the interface stays in place and only the discretization of both solvers is refined. In family C the widths are counted in each level's own cells, as a user refining the whole method would do, and the interface moves towards the box faces with refinement.

**Time.** The coupler takes one particle step per exchange, so the particle step and the number of transfers per convective time cannot be varied apart: $N_x=D/(U\Delta t_p)$. They are separated by functional form instead. A temporal error of the particle step grows with $\Delta t_p$; an error added at each transfer accumulates with $N_x$ and so grows as $\Delta t_p$ falls. The model

$$e_c=c_0+a\,\frac{U\Delta t_p}{D}+b\,N_x\;\left(+\,d\,\frac{U\Delta t_f}{D}\right)$$

is fitted by linear least squares over families T2, P and E, all on the baseline cells. P holds $\Delta t_p$ while $\Delta t_f$ changes, E holds $\Delta t_f$ while $\Delta t_p$ changes, and T2 changes both at a fixed ratio. The FVM-step term $d$ is included when the runs leave enough freedom. A coefficient is called significant when it exceeds twice its standard error. Each driver is also fitted alone as $e_c=c_0+a\,x^q$, with $q\in[-3,3]$, to show on which one the coupling error collapses.

**Factorial.** Family F is a full $2^2$ design (the inset factor of the plan was not run). The effect of a set of factors is the mean error where the product of their coded levels ($\pm1$) is $+1$ minus the mean where it is $-1$. The additivity defect of two changes A and B, $e(A,B)-[e(A)+e(B)-e(0)]$, is zero when they add their errors independently.

### 3.5 In-run indicator

At every exchange the coupler logs the mismatch between the FVM boundary trace and the particle velocity on the box faces (mean, maximum and on the outflow faces, in units of $U$), and the normal-velocity and tangential-gradient residuals of the first and last interface sweep (in $U$ and $U/D$). Their means over developed flow, $t\ge40$ s, are correlated with $\lvert e_c\rvert$ over every run (`figures/error_indicator`, `figures/auxiliary/error_indicators.csv`). A good correlation gives users an a-posteriori estimate of the coupling error from their own run, without an FVM-only solution.

## 4. Procedure for a user's own flow

1. **Measure the flow scales.** Run the FVM only on a moderate domain, or a coarse coupled case, to developed flow. Measure the shear-layer vorticity thickness $\delta_\omega$ at separation, the formation length $L_f$ (centreline peak of $u_\mathrm{rms}$; the $v_\mathrm{rms}$ peak is less sensitive to the cells), and the largest strain rate of the particle field, $\max\lvert S\rvert$ (reported by the particle solver's stability check). Do not use the strain rate of the FVM wall cells, which is lower.
2. **Set the interface** (section 6):
   - side clearance $c\ge5$ shear-layer thicknesses between the separated shear layer and the inner edge of the interface bands;
   - downstream transfer face at $x_t\ge L_f(1+R)$ with $R\ge1$;
   - transfer inset, blend width and particle-only band of about 9, 6 and 2 particle core radii;
   - the largest stable particle step, $\Lambda=\max\lvert S\rvert\,\Delta t_p\approx0.8$ (stable up to 1.0 here), which keeps the exchanges per convective time $N_x=D/(U\Delta t_p)$ low; the FVM step then follows from its own Courant number, $C_f\le0.2$;
   - two interface sweeps per exchange, or a boundary-trace tolerance of $10^{-5}$.
3. **Refine space** (family X): three or four levels of cell and particle size, ratio $\sqrt2$, at fixed FVM and particle steps, with the interface widths fixed in metres (`--interface-cell-size`).
4. **Refine time at a fixed particle step** (family P): three or four FVM steps under the particle step of step 2. Do not refine the particle step to show temporal convergence: at a fixed exchange ratio the coupling error grows as the particle step falls (section 5.4).
5. **Estimate order and uncertainty.** Apply the least-squares procedure of section 3.2 (`discretization_uncertainty` in `assets/error_analysis.py`) to each statistic of each family. Report an order only for monotone data; report $\phi_0$ and $U$ in every case.
6. **Read the indicator.** Compare the developed mean of the boundary-trace velocity mismatch, logged by the coupler at every exchange, with $3.5\times10^{-3}U$. Below it, the lift-amplitude coupling error stayed within 2.1% in every run of this study; above $5.5\times10^{-3}U$ it reached 5–7%. A high value points at the side clearance or at too many exchanges.
7. **Do not refine everything together** as the only check. Refining cells and steps together at a fixed exchange ratio mixes the FVM spatial and temporal errors, which have opposite signs for the lift amplitude, with a coupling error that grows with the number of exchanges.

## 5. Results

### 5.1 Flow scales

Figure: `figures/flow_scales`, the centreline mean and rms velocity at $Re=100$, 150 and 250 with the recirculation and formation lengths marked.

| $Re$ | $L_r/D$ | $L_f/D$ ($u_\mathrm{rms}$) | $L_{f,v}/D$ ($v_\mathrm{rms}$) | $\delta_\omega/D$ | $\delta_\omega\sqrt{Re}/D$ | FVM wall $\max\lvert S\rvert$ [s$^{-1}$] | $St$ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 100 | 1.46 | 3.00 | 3.08 | 0.316 | 3.16 | 9.4 | 0.163 |
| 150 | 1.06 | 1.94 | 2.28 | 0.263 | 3.23 | 12.3 | 0.182 |
| 250 | 0.74 | 1.22 | 1.75 | 0.214 | 3.39 | 16.9 | 0.201 |

The shear-layer thickness scales as $D/\sqrt{Re}$ ($\delta_\omega\sqrt{Re}/D=3.2\pm0.1$), and the FVM wall strain as $\approx(U/D)\sqrt{Re}$. The formation length falls by a factor of 2.5 from $Re=100$ to 250. At $Re=150$ the $u_\mathrm{rms}$ peak sits on a plateau and moves from $2.32D$ to $1.81D$ between $h=0.08$ and $0.0283$ m, while the $v_\mathrm{rms}$ peak moves only from $2.35D$ to $2.25D$ and the recirculation length from $1.13D$ to $1.05D$.

### 5.2 FVM-only convergence and $\phi_0$ (families C, X, T)

Figure: `figures/error_references_C_X_T`.

| Statistic | Family | Class | Order | Fit, $F_s$ | $\phi_0$ | $U$ at finest level [%] |
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

Family T extrapolates to zero step on the baseline cells, so its $\phi_0$ keeps the spatial error of $h=0.04$ m. The space-time fit over the ten FVM-only levels determines only the Strouhal number: $p=1.70\pm0.08$, $q=2.0\pm1.0$, $\phi_0=0.1838\pm0.0002$. For the mean drag and the lift amplitude its orders are undetermined (standard error of $\phi_0$ larger than the family-C uncertainty), and $\phi_0$ is taken from family C: $\bar C_D=1.3107$, $\hat C_L=0.5178$.

On the baseline cells and step, the FVM-only lift amplitude is 3.1% below $\phi_0$; the spatial part of that error is negative (family X: −12.5%, −6.8%, −3.8% at $h=0.08$, 0.0566, 0.04 m with $\Delta t_f=0.004$ s) and the temporal part positive (family T: the lift amplitude falls by 1.8% from $\Delta t_f=0.008$ s to zero step). The FVM is first order in time for the lift amplitude, although its scheme is second order; the under-relaxed outer correctors are the likely cause, not tested here.

### 5.3 Error decomposition of the coupled solver (families C, X, T2)

Figure: `figures/error_decomposition_C_X_T2`.

| Family | Level | $e$: $\bar C_D$ / $\hat C_L$ / $St$ [%] | $e_f$ | $e_c$ | $e_c$ wake velocity [%] |
| --- | --- | --- | --- | --- | --- |
| X | $h=0.08$ m | +1.28 / −9.57 / −0.24 | −0.47 / −12.53 / −3.69 | +1.76 / +2.97 / +3.46 | 5.27 |
| X | $h=0.0566$ m | −0.70 / −7.99 / −1.41 | −0.24 / −6.76 / −2.06 | −0.46 / −1.23 / +0.65 | 2.86 |
| X | $h=0.04$ m | −0.87 / −6.47 / −1.51 | −0.27 / −3.83 / −1.18 | −0.60 / −2.65 / −0.33 | 1.82 |
| T2 | $\Delta t_f=0.016$ s | −0.11 / −3.18 / −1.11 | +0.06 / −1.70 / −1.34 | −0.17 / −1.47 / +0.23 | 3.17 |
| T2 | $\Delta t_f=0.008$ s | −0.82 / −5.93 / −1.65 | −0.14 / −3.11 / −1.22 | −0.68 / −2.82 / −0.42 | 3.06 |
| T2 | $\Delta t_f=0.004$ s | −2.66 / −10.91 / −3.77 | −0.27 / −3.83 / −1.18 | −2.39 / −7.08 / −2.59 | 5.70 |
| C | $h=0.08$ m | +1.30 / −9.92 / −0.90 | −0.38 / −11.86 / −4.00 | +1.68 / +1.93 / +3.10 | 4.68 |
| C | $h=0.0566$ m | −0.44 / −6.92 / −1.46 | −0.05 / −5.66 / −2.18 | −0.39 / −1.26 / +0.72 | 2.44 |
| C | $h=0.04$ m | −0.82 / −5.93 / −1.65 | −0.14 / −3.11 / −1.22 | −0.68 / −2.82 / −0.42 | 3.06 |

**Space (X).** The total error falls monotonically with $h$ for all three statistics, but more slowly than the FVM part: the extrapolated coupled values are $\bar C_D=1.286$, $\hat C_L=0.500$, $St=0.180$, which lie 1.9%, 4.5% and 2.1% below the FVM-only $\phi_0$, with uncertainties of 3.6%, 9.3% and 2.2%. The coupling error changes sign between $h=0.08$ and $0.0566$ m and then grows slowly in magnitude, so it does not vanish with the cells. The free-order fits give no usable order (0.1 for the lift amplitude, about 7 for drag and Strouhal number, from three levels without residual freedom). What can be stated is that the coupled solver follows the FVM spatial convergence (order 1.5–1.7) towards an offset of a few per cent, set by the coupling error at the fixed particle step ($N_x=63$ in this family).

**Time (T2).** The data are anomalous: the total error grows as the steps are refined, because the coupling error grows (−1.5%, −2.8%, −7.1% in lift amplitude) while the FVM part barely changes. At a fixed exchange ratio the coupled solver has no temporal order; section 5.4 identifies the cause.

**Space and time together (C).** The coupling error changes sign under refinement (+1.9%, −1.3%, −2.8% in lift amplitude), and the total error decreases only because the FVM part, which dominates on coarse cells, falls faster than the coupling part grows. A user refining this way would see apparent convergence.

### 5.4 Drivers of the coupling error (families T2, P, E)

Figure: `figures/error_drivers_T2_P_E`, the coupling error against $U\Delta t_p/D$ (with $\Lambda$) and against $N_x$, for the six runs on the baseline cells.

| Statistic | $c_0$ [%] | $a$ | exponent $q$ in $e_c=c_0-a\,N_x^{q}$ | $R^2$ | transfer part at $N_x=25$ (baseline) [%] |
| --- | --- | --- | --- | --- | --- |
| $\bar C_D$ | +0.09 | $7.6\times10^{-4}$ | 1.67 | 1.000 | −0.16 |
| $\hat C_L$ | −1.00 | $1.3\times10^{-3}$ | 1.75 | 1.000 | −0.36 |
| $St$ | +0.54 | $8.5\times10^{-4}$ | 1.70 | 1.000 | −0.20 |

The six runs separate the steps: E changes the particle step at a fixed FVM step, P changes the FVM step at a fixed particle step, and T2 changes both. The coupling error collapses on the particle step alone (or, equivalently, $N_x$); fitted on the FVM step alone or on $n$, $R^2$ falls to 0.3. The P pair confirms it directly: halving the FVM step at $\Delta t_p=0.04$ s changes the coupling error by +0.03 (drag), +0.07 (lift amplitude) and +0.02 (Strouhal number) points.

The error grows as the particle step falls, the opposite of a temporal error, and with an exponent of 1.7 in $N_x$. It is an error added at each exchange (the transfer of FVM vorticity to the particles and the renewal of the transfer region) and accumulated over the exchanges in a convective time. The particle step and the transfer count are tied by design, so this attribution rests on the trend, not on independent variation. The consequence for the settings is direct: the particle step should be the largest that the particle stability allows ($\Lambda\approx0.8$–1.0), and the FVM step is refined under it. $c_0$ is the coupling error that fewer exchanges cannot remove; at the baseline it is −1.0% in lift amplitude and comes from the box and the interface (section 5.6).

### 5.5 Interface iteration (family K)

Coupling error $e_c$ at the baseline ($h=0.04$ m, $\Delta t_f=0.008$ s, $n=5$), in per cent:

| FVM solves per exchange | $\bar C_D$ | $\hat C_L$ | $St$ |
| --- | --- | --- | --- |
| $K=1$ | +2.96 | +2.09 | +0.34 |
| $K=2$ | −0.23 | −1.52 | +0.33 |
| $K=4$ | −0.09 | −1.39 | +0.33 |
| Tolerance $10^{-5}$ (at most 6; mean 1.8) | −0.09 | −1.42 | +0.33 |

With a single solve per exchange the FVM boundary trace lags the particle velocity, and the drag and lift-amplitude errors take the opposite sign and a larger magnitude. A second sweep removes almost all of this. From $K=2$ to $K=4$ the drag error changes by 0.14 and the lift-amplitude error by 0.13 percentage points, and the tolerance-driven runs agree with $K=4$ to within 0.03 points. The Strouhal number does not depend on $K$. The coupling error that remains at $K\ge2$ is not an error of the interface iteration. The tolerance-driven runs need 1.8 sweeps per exchange on average and so cost less than a fixed $K=2$. (The $K=3$ run stopped on a GPU memory shortage and was not repeated; $K=6$ was dropped for time.)

### 5.6 FVM box (family B) and factorial (family F)

Figure: `figures/error_box_B`, the coupling error against the reach $R$ (with $L_f$ from the $u_\mathrm{rms}$ and the $v_\mathrm{rms}$ peak) and against the side clearance $c$.

| $Re$ | Downstream face $[D]$ | Side margin $[D]$ | $R$ | $c$ | $e_c$: $\bar C_D$ / $\hat C_L$ / $St$ [%] |
| --- | --- | --- | --- | --- | --- |
| 150 | 2.97 | 2.4 | 0.35 | 4.97 | +0.49 / +0.69 / +0.66 |
| 150 | 3.36 | 2.4 | 0.55 | 4.97 | +0.20 / +0.68 / +0.62 |
| 150 | 4.4 | 2.4 | 1.08 | 4.97 | −0.09 / −1.42 / +0.33 |
| 150 | 4.4 | 1.4 | 1.08 | 1.18 | −0.44 / −5.37 / +0.71 |
| 250 | 1.99 | 2.4 | 0.35 | 6.11 | +0.19 / −2.14 / +0.83 |
| 250 | 2.24 | 2.4 | 0.55 | 6.11 | +0.23 / −1.60 / +0.85 |
| 250 | 2.9 | 2.4 | 1.10 | 6.11 | +0.57 / −0.64 / +0.81 |
| 250 | 4.4 | 2.4 | 2.33 | 6.11 | −0.14 / −1.93 / +0.53 |
| 250 | 4.4 | 1.36 | 2.33 | 1.26 | −1.50 / −6.34 / +0.74 |

**Side clearance.** Bringing the interface bands to within $1.2\,\delta_\omega$ of the separated shear layer changes the lift-amplitude coupling error by −3.95 points at $Re=150$ and −4.41 points at $Re=250$, and the drag error by −0.35 and −1.36 points. The clearance in shear-layer thicknesses collapses the lift-amplitude effect at the two Reynolds numbers; it is the box parameter that sets the largest error and the one that carries over. With only two clearances per Reynolds number, the threshold lies between $1.2$ and $5.0\,\delta_\omega$; $c\ge5.0\,\delta_\omega$ is what was shown to be sufficient.

**Downstream reach.** For $R\ge0.35$ the reach changes the lift-amplitude coupling error by at most about 2 points, and not in the same way at the two Reynolds numbers: at $Re=150$ shortening the box moves the error from −1.42% to +0.7%, at $Re=250$ the error varies between −2.1% and −0.6% without a monotone trend. Neither choice of formation length collapses the two. At $Re=150$ the short box looks better only because its positive error cancels the negative transfer error: a box judged by its total error alone would be chosen too small. $R\ge1$ keeps the effect of the downstream face within the scatter of the other runs at both Reynolds numbers.

**Factorial.** **[RESULT: the 2×2 design of downstream face (3.36D, 4.4D) and n (2, 5); the corner at 3.36D and n = 2 is running.]**

### 5.7 Transfer to another Reynolds number (family R)

| $Re$ | $n$ | $\Lambda$ | $N_x$ | $R$ | $c$ | $e_c$: $\bar C_D$ / $\hat C_L$ / $St$ [%] |
| --- | --- | --- | --- | --- | --- | --- |
| 150 | 5 | 0.78 | 25 | 1.08 | 4.97 | −0.09 / −1.42 / +0.33 |
| 250 | 5 | 1.00 | 25 | 2.33 | 6.11 | −0.14 / −1.93 / +0.53 |
| 250 | 4 | 0.80 | 31 | 2.33 | 6.11 | −0.19 / −2.01 / +0.50 |

At $Re=250$ no extrapolated value is available, so these are coupling errors against the FVM-only run of the same Reynolds number, cells and step. The particle step at $\Lambda=1.0$ is still stable. Matching $\Lambda$ ($n=4$) does not reproduce the error of $Re=150$; matching $N_x$ ($n=5$) comes slightly closer, and the change from $n=5$ to $n=4$ at $Re=250$ (−0.08 points in lift amplitude) is close to the $N_x$ trend fitted at $Re=150$ (−0.17 points). The remaining offset of about −0.5 points in lift amplitude is not explained by $\Lambda$, $N_x$, the reach or the clearance; at $Re=250$ the box is larger in flow scales than at $Re=150$. All $Re=250$ errors stay within the 2% target in drag and Strouhal number, and at the limit of it in lift amplitude.

### 5.8 In-run indicator

Figure: `figures/error_indicator`, $\lvert e_c\rvert$ against the developed mean of the boundary-trace velocity mismatch for all 23 coupled runs.

| Indicator (mean over $t\ge40$ s) | Correlation with $\lvert e_c\rvert$: $\bar C_D$ / $\hat C_L$ / $St$ / wake |
| --- | --- |
| Trace velocity mismatch, mean | 0.39 / **0.84** / 0.40 / 0.45 |
| Trace velocity mismatch, maximum | 0.49 / **0.86** / 0.35 / 0.38 |
| Outflow mismatch, mean | 0.46 / 0.59 / 0.41 / 0.38 |
| First-sweep normal residual | −0.20 / −0.13 / −0.20 / −0.20 |
| First-sweep gradient residual | −0.20 / −0.11 / −0.20 / −0.20 |
| Last-sweep gradient residual | 0.14 / −0.01 / −0.03 / −0.11 |

The boundary-trace mismatch follows the lift-amplitude coupling error over every family: below $3.5\times10^{-3}U$ all runs have $\lvert e_c\rvert\le2.1\%$, between 3.5 and $5\times10^{-3}U$ they have 1.2–3.0%, and the three runs above $5.5\times10^{-3}U$ (side clearance $1.2$–$1.3\,\delta_\omega$ at both Reynolds numbers, and $N_x=125$) have 5.4–7.1%. The interface-sweep residuals do not indicate the coupling error.

## 6. Recommendations

Values in flow scales, for a bluff body shedding a laminar wake, together with the conditions established in the earlier study: an outflow fringe of at least two wake wavelengths at the particle-domain outflow, an FVM-only comparison with the same unbounded far field, and converged FVM linear solves (momentum tolerance $10^{-9}$).

| Parameter | Flow-scaled form | Recommended value | Evidence |
| --- | --- | --- | --- |
| Side margin | clearance $c=(m-w-D/2)/\delta_\omega$ | $c\ge5$; $c\approx1.2$ costs about 4 points of lift amplitude at both $Re$ | B |
| Downstream transfer face | reach $R=(x_t-L_f)/L_f$ | $R\ge1$; $0.35\le R<1$ changes the error by up to 2 points | B, F |
| Transfer inset, blend width, particle-only band | in particle core radii $\sigma=h_p$ | about 9, 6 and 2; the inset changed the error by less than 0.7 points between 4 and 20 in the earlier study | earlier study |
| Particle step | $\Lambda=\max\lvert S\rvert\,\Delta t_p$, with the particle-field strain | the largest stable value, $\Lambda\approx0.8$ (1.0 still stable, 1.24 unstable) | E, T2, P, R |
| Exchanges per convective time | $N_x=D/(U\Delta t_p)$ | as low as $\Lambda$ allows; the transfer part of $e_c$ grows as $N_x^{1.7}$ (−0.4% at $N_x=25$, −1.8% at 63, −6.1% at 125 in lift amplitude) | E, T2, P |
| FVM step | Courant number $C_f$ | $C_f\le0.2$; under a fixed particle step it does not change $e_c$ | P |
| Cell size | $h/\delta_\omega$ | 0.15 gives an FVM error of 3% in lift amplitude and 1.2% in Strouhal number; the coupling error does not fall with $h$ | X, C |
| Interface iteration | FVM solves per exchange, or boundary-trace tolerance | tolerance $10^{-5}$ (1.8 sweeps on average), or $K=2$ | K |
| Indicator | developed mean trace mismatch | below $3.5\times10^{-3}U$ | all |

Overall behaviour of the coupled solver:

| Statistic | Space (X): FVM part / coupled total | Time | $e$ at baseline [%] | $U$ of $\phi_0$ [%] |
| --- | --- | --- | --- | --- |
| $\bar C_D$ | converged within 0.6% / converges to −1.9% offset | no order at fixed $n$; FVM part order 0.3 | −0.22 | 0.21 |
| $\hat C_L$ | order 1.5 / converges to −4.5% offset | no order at fixed $n$; FVM part order 0.9 | −4.53 | 2.05 |
| $St$ | order 1.7 / converges to −2.1% offset | no order at fixed $n$; FVM part order 1.7 | −0.90 | 0.82 |

The coupled solver inherits the spatial order of the FVM and, at a fixed particle step, its temporal order; it adds a coupling error that does not fall with the cells and that grows with the number of exchanges. Its order as a whole is that of the FVM up to this coupling floor, which at the recommended settings is about 1% in lift amplitude ($c_0$) plus 0.4% for the exchanges.

## 7. Limitations

- **Three levels per coupled family.** Families X and T2 have three levels each. The free-order fit then has no residual freedom, the combined first- and second-order fit cannot be made, and the uncertainty rests on the safety factor alone. The offsets of family X carry uncertainties of 2–9%.
- **One body.** Only the circular cylinder was studied. The rules are expressed in flow scales so that they can carry over, but their transfer to other shapes is not shown.
- **Two Reynolds numbers.** The box and transfer studies cover $Re=150$ and $250$ in laminar planar shedding; $Re=100$ has flow scales but no coupled run.
- **Side clearance at two values.** The clearance threshold lies between $1.2$ and $5.0\,\delta_\omega$; only the upper value is shown to be sufficient.
- **Not run.** The particle-spacing family ($h_p=h_f/2$), $K=3$ and 6, $n=6$, 20 and 40, the inset factor of the factorial and the finest constant-Courant level were dropped to finish within the week; the interface widths rest on the one-at-a-time study of 5–12 October on an older code version and a same-cell comparison.
- **Wake velocity against the finest FVM-only run.** The wake-velocity errors are distances to the finest FVM-only run, not to an extrapolated value.
- **Limits of the uncertainty estimate.** The least-squares estimate degrades for shallow orders and noisy data: in a synthetic check with $p=1$ and 0.1% noise, $\pm U$ covered the true error in 67–84% of cases. The statistics are averaged over $40\le t\le55$ s, about 2.7 shedding periods.
- **Particle step and transfer count tied.** The coupler takes one particle step per exchange, so the temporal and the per-transfer parts of the coupling error are separated only by their trends, not by independent variation.
- **Cost.** Most runs shared the GPU with another run, so their wall times are not comparable and no cost conclusion is drawn.
- **Repeatability.** GPU summation order is not deterministic and no repeat runs were made; differences below about 0.1 percentage points are not resolved.
