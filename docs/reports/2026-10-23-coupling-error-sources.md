# Sources of error in the coupled FVM–VPM solver: the planar cylinder at $Re=150$

Technical report, 2026-10-23. Study plan: [2026-10-19-coupling-error-sources](../plans/2026-10-19-coupling-error-sources.md). Case: [tutorials/coupled_fvm_vpm/01_cylinder_shedding_flow](../../tutorials/coupled_fvm_vpm/01_cylinder_shedding_flow/README.md), whose `./allplot.sh` draws every figure and writes every table cited here (figures in `figures/`, tables in `figures/auxiliary/`). Earlier findings: [coupling_findings.md](../../tutorials/coupled_fvm_vpm/01_cylinder_shedding_flow/assets/coupling_findings.md).

Placeholders for results that are still being computed are marked **[RESULT: ...]**.

## 1. Summary

- **Spatial order of the coupled solver** (family X, interface fixed in metres): mean drag **[RESULT: observed order $p$ or class, $U$ at $h/D=0.04$]**; lift amplitude **[RESULT: $p$ or class, $U$]**; Strouhal number **[RESULT: $p$ or class, $U$]**.
- **Temporal order of the coupled solver** (family T2, $n=2$): mean drag **[RESULT: $q$ or class, $U$ at $U\Delta t_f/D=0.008$]**; lift amplitude **[RESULT: $q$ or class, $U$]**; Strouhal number **[RESULT: $q$ or class, $U$]**.
- **FVM-only orders and $\phi_0$** (space-time fit over the FVM-only levels): **[RESULT: $p$, $q$ with standard errors; $\phi_0$ of $\bar C_D$, $\hat C_L$, $St$ and its origin, space-time fit or family C]**.
- **Error budget at the baseline** ($h=0.04$ m, $\Delta t_f=0.008$ s, $n=5$): total $e$ **[RESULT: per statistic, % of $\phi_0$]**, of which FVM discretization $e_f$ **[RESULT]** and coupling $e_c$ **[RESULT]**.
- **What drives the coupling error** (fit over T2, P and E): **[RESULT: sign and significance of the temporal term $a$ and the per-transfer term $b$; which dominates at the baseline]**.
- **FVM box** (family B, $Re=150$ and $250$): **[RESULT: reach $R$ and clearance $c$ beyond which $e_c$ stops changing, and whether the two Reynolds numbers collapse on them]**.
- **Additivity** (family F): **[RESULT: largest interaction effect relative to the main effects; additivity defect]**.
- **Interface iteration** (family K, established): two sweeps per exchange, or a boundary-trace tolerance of $10^{-5}$, reduce the interface-iteration part of $e_c$ below 0.15 percentage points; a single sweep raises the drag error to +2.96% and the lift-amplitude error to +2.09%.
- **In-run indicator**: **[RESULT: indicator with the highest correlation with $\lvert e_c\rvert$, correlation coefficient, and threshold value that corresponds to the 2% target]**.
- **Transfer to other Reynolds numbers** (family R, $Re=100$ and $250$): **[RESULT: $e_c$ at the flow-scaled rules against the 2% target]**.

## 2. The question, and why naive refinement misleads

The coupled solver couples a finite-volume (FVM) solution of the near body, inside a box, to one layer of vortex particles between slip planes that carries the outer wake. At every exchange the particles supply the FVM boundary condition on the box faces, and FVM vorticity replaces the particles inside the transfer region. A coupled result therefore carries the discretization errors of both solvers and an error of the coupling itself: of the interface position, of the transfer, of the boundary-trace iteration and of the exchange frequency. The study sets out to

1. give users an objective procedure, free of a choice of favourable levels, to measure the spatial and temporal order of their own coupled problem and the uncertainty of a result;
2. isolate each source of error, with every parameter expressed in scales of the flow, so that a recommended value does not rest on two errors cancelling at one setup;
3. state flow-agnostic values for the coupling-interface parameters, and the error and the spatial and temporal orders of the coupled solver as a whole, independent of the orders of the FVM and the VPM on their own.

Two features of the earlier sensitivity study made its conclusions unsafe.

**Same-cell comparison.** The coupled runs were compared with an FVM-only run on the same cells and step. That comparison removes the FVM discretization error shared by both runs and measures only the coupling error $e_c$. It does not show how far either run is from the solution of the equations. In the preliminary runs of the plan, the FVM-only lift amplitude on the baseline cells ($h=0.04$ m) lies about 4% below its extrapolated value, so the 2% target had been met against an unconverged value. The present study measures every error against $\phi_0$, the FVM-only value extrapolated to zero cell size and step, and splits it into an FVM part and a coupling part (section 3.1).

**Cancelling errors.** Refining cells, steps and particles together at constant Courant number is what a user would do first. It mixes errors of opposite sign. For the lift amplitude, the FVM spatial error is negative and the temporal error positive, so their sum can be small, or change sign, on a level where neither is small. In the preliminary runs at $C_f=0.2$ and $n=2$, the coupling error in lift amplitude changes sign under refinement (+1.9%, −1.2%, −3.0% at $h=0.08$, $0.0566$, $0.04$ m): at least two contributions of opposite sign are present, and an order fitted to such data has no meaning. **[RESULT: confirm or update these values from the final family C.]** The study therefore refines space and time separately (families X and T2) and keeps family C only as the check a user sees when refining everything together.

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
| FVM box, sides | clearance $c=(m-w-D/2)/\delta_\omega$, with $m$ the side margin and $w$ the three interface widths | $c\approx4.7$ |
| Interface iteration | FVM solves per exchange $K$, or boundary-trace tolerance | up to 6, $10^{-5}$ |

The box is set by two distances, the downstream reach $R$ and the side clearance $c$. Between $Re=100$ and $250$ the formation length falls from $3.0D$ to $1.22D$, while $\delta_\omega$ falls as $Re^{-1/2}$, so one margin in diameters cannot hold both fixed across Reynolds numbers. The case therefore takes the downstream box face separately (`--box-downstream`) from the side margin (`--box-margin`).

### 3.4 Run families and what each holds fixed

All families are runs of case 01 (`assets/run_sensitivity_study.sh`), each up to 55 s. $h_f=h_p=h$ unless stated, $w$ are the interface widths, and $N_x=D/(U\Delta t_p)$.

| Family | Fixed | Varied | Isolates |
| --- | --- | --- | --- |
| C, space and time | $C_f=0.2$, $n=2$ | $h=0.08$, $0.0566$, $0.04$, $0.0283$ m | what a user sees refining everything together |
| X, space | $\Delta t_f=0.004$ s, $n=4$, interface widths in metres | $h=0.08$, $0.0566$, $0.04$ m | spatial order of the coupled solver |
| T2, time | $h=0.04$ m, $n=2$ | $\Delta t_f=0.016$, $0.008$, $0.004$ s | temporal order of the coupled solver |
| P, FVM step | $h=0.04$ m, $\Delta t_p=0.04$ s | $n=5$, $10$, $20$ | FVM temporal error inside the coupling, exchange frequency fixed |
| E, exchange ratio | $h=0.04$ m, $\Delta t_f=0.008$ s | $n=2$, $4$, $5$, $6$ | particle step and exchange frequency together |
| K, interface sweeps | baseline | $K=1$, $2$, $3$, $4$, $6$, tolerance-driven | interface iteration |
| B, box | $n=5$, inset 8.75 cells | four boxes in $R$ and $c$, at $Re=150$ and $250$ | box error, flow-scaled |
| F, factorial | $h=0.04$ m | downstream face $3.36D$ / $4.4D$ $\times$ inset 4 / 14 cells $\times$ $n=2$ / $5$ | interactions and additivity of the errors |
| R, Reynolds number | baseline rules | $Re=100$, $250$ | transfer of the rules to other flows |
| S, particle spacing | $\Delta t_f=0.008$ s | $h_p=h_f/2$ at $n=2$, $5$ | particle discretization |

Which flow-scaled groups each family varies (bold: held fixed by construction, where a naive refinement would change it):

| Family | $h/\delta_\omega$ | $C_f$ | $U\Delta t_f/D$ | $U\Delta t_p/D$, $N_x$ | $\Lambda$ | $w/D$ | Box ($R$, $c$) | $Re$ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| X | varied | varies with $h$ | fixed | fixed | fixed | **fixed** | fixed | fixed |
| T2 | fixed | varied | varied | varied ($n=2$) | varied | fixed | fixed | fixed |
| P | fixed | varied | varied | **fixed** | fixed | fixed | fixed | fixed |
| E | fixed | fixed | fixed | varied | varied | fixed | fixed | fixed |
| K | fixed | fixed | fixed | fixed | fixed | fixed | fixed | fixed |
| B | fixed | fixed | fixed | fixed | per $Re$ | fixed | varied in $L_f$ and $\delta_\omega$ | 150, 250 |
| R | fixed in $D$ | fixed | fixed | fixed | matched ($n=4$ at 250) or not ($n=5$) | fixed | per $Re$ | varied |
| C | varied | fixed | varied | varied ($n=2$) | varied | varies (in cells) | fixed | fixed |

**Space.** In family X the inset, blend and particle-only widths are counted in cells of 0.04 m (`--interface-cell-size 0.04`) at every level, so the interface stays in place and only the discretization of both solvers is refined. In family C the widths are counted in each level's own cells, as a user refining the whole method would do, and the interface moves towards the box faces with refinement.

**Time.** The coupler takes one particle step per exchange, so the particle step and the number of transfers per convective time cannot be varied apart: $N_x=D/(U\Delta t_p)$. They are separated by functional form instead. A temporal error of the particle step grows with $\Delta t_p$; an error added at each transfer accumulates with $N_x$ and so grows as $\Delta t_p$ falls. The model

$$e_c=c_0+a\,\frac{U\Delta t_p}{D}+b\,N_x\;\left(+\,d\,\frac{U\Delta t_f}{D}\right)$$

is fitted by linear least squares over families T2, P and E, all on the baseline cells. P holds $\Delta t_p$ while $\Delta t_f$ changes, E holds $\Delta t_f$ while $\Delta t_p$ changes, and T2 changes both at a fixed ratio. The FVM-step term $d$ is included when the runs leave enough freedom. A coefficient is called significant when it exceeds twice its standard error. Each driver is also fitted alone as $e_c=c_0+a\,x^q$, with $q\in[-3,3]$, to show on which one the coupling error collapses.

**Factorial.** Family F is a full $2^3$ design. The effect of a set of factors is the mean error where the product of their coded levels ($\pm1$) is $+1$ minus the mean where it is $-1$. The additivity defect of two changes A and B, $e(A,B)-[e(A)+e(B)-e(0)]$, is zero when they add their errors independently.

### 3.5 In-run indicator

At every exchange the coupler logs the mismatch between the FVM boundary trace and the particle velocity on the box faces (mean, maximum and on the outflow faces, in units of $U$), and the normal-velocity and tangential-gradient residuals of the first and last interface sweep (in $U$ and $U/D$). Their means over developed flow, $t\ge40$ s, are correlated with $\lvert e_c\rvert$ over every run (`figures/error_indicator`, `figures/auxiliary/error_indicators.csv`). A good correlation gives users an a-posteriori estimate of the coupling error from their own run, without an FVM-only solution.

## 4. Procedure for a user's own flow

1. **Measure the flow scales.** Run the FVM only on a moderate domain, or a coarse coupled case, to developed flow. Measure the formation length $L_f$ (centreline peak of $u_\mathrm{rms}$), the shear-layer vorticity thickness $\delta_\omega$ at separation, and the largest strain rate the particle field sees, $\max\lvert S\rvert$ (the particle solver's stability check reports it as the ceiling of the particle step). Use the strain rate of the particle field, not that of the FVM wall cells.
2. **Set the interface from the recommendations** (section 6):
   - downstream transfer face at $x_t=L_f(1+R)$ with $R\ge$ **[RESULT: recommended reach]**;
   - side clearance $c\ge$ **[RESULT: recommended clearance]** shear-layer thicknesses from the inner edge of the interface bands;
   - transfer inset, blend width and particle-only band of **[RESULT: values in $\sigma$]**;
   - particle step from $\Lambda=\max\lvert S\rvert\,\Delta t_p\le$ **[RESULT: recommended $\Lambda$]**, which also sets $N_x$;
   - two interface sweeps per exchange, or a boundary-trace tolerance of $10^{-5}$.
3. **Refine space** (family X): three or four levels of cell and particle size, ratio $\sqrt2$, at fixed FVM and particle steps, with the interface widths fixed in metres.
4. **Refine time** (family T): three or four levels of the FVM step at fixed cells and a fixed number of FVM steps per exchange, so that the particle step follows.
5. **Estimate order and uncertainty.** Apply the least-squares procedure of section 3.2 (`discretization_uncertainty` in `assets/error_analysis.py`) to each statistic of each family. Report an order only for monotone data; report $\phi_0$ and $U$ in every case. If a statistic is oscillatory or anomalous in both families, refine further or lengthen the averaging window before drawing conclusions.
6. **Read the indicator.** Compare the developed mean of **[RESULT: chosen indicator]** in the production run with **[RESULT: threshold]**. Above it, the coupling error is likely to exceed 2%; revisit step 2.
7. **Do not refine everything together** as the only check. A family C of the user's own flow shows the combined behaviour, but cancelling spatial and temporal errors can make it look converged when it is not.

## 5. Results

### 5.1 Flow scales

Figure: `figures/flow_scales`. **[RESULT: figure of centreline $\bar u$ and $u_\mathrm{rms}$ at $Re=100$, 150, 250 with $D/2+L_r$ and $L_f$ marked.]**

| $Re$ | $L_r/D$ | $L_f/D$ ($u_\mathrm{rms}$) | $L_{f,v}/D$ ($v_\mathrm{rms}$) | $\delta_\omega/D$ | $\delta_\omega\sqrt{Re}/D$ | wall $\max\lvert S\rvert$ [s$^{-1}$] | $St$ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 100 | **[RESULT]** | 3.0 | **[RESULT]** | **[RESULT]** | **[RESULT]** | **[RESULT]** | **[RESULT]** |
| 150 | **[RESULT]** | 1.94 | **[RESULT]** | **[RESULT]** | **[RESULT]** | 12.3 | **[RESULT]** |
| 250 | **[RESULT]** | 1.22 | **[RESULT]** | **[RESULT]** | **[RESULT]** | **[RESULT]** | **[RESULT]** |

**[RESULT: one paragraph on the scaling of $L_f$ and $\delta_\omega$ with $Re$ and the sensitivity of $L_f$ to the cells.]**

### 5.2 FVM-only convergence and $\phi_0$ (families C, X, T)

Figure: `figures/error_references_C_X_T`. **[RESULT: FVM-only error against $h/D$ and $U\Delta t/D$ with fits and $\pm U$.]**

| Statistic | Family | Class | Order | Fit, $F_s$ | $\phi_0$ | $U$ at baseline [%] |
| --- | --- | --- | --- | --- | --- | --- |
| $\bar C_D$ | C / X / T | **[RESULT]** | **[RESULT]** | **[RESULT]** | **[RESULT]** | **[RESULT]** |
| $\hat C_L$ | C / X / T | **[RESULT]** | **[RESULT]** | **[RESULT]** | **[RESULT]** | **[RESULT]** |
| $St$ | C / X / T | **[RESULT]** | **[RESULT]** | **[RESULT]** | **[RESULT]** | **[RESULT]** |

Space-time fit (`figures/auxiliary/error_space_time.csv`): **[RESULT: $p\pm$, $q\pm$, $\phi_0\pm$, rms residual and freedom per statistic; which $\phi_0$ is used]**.

**[RESULT: paragraph confirming the signs of the FVM spatial and temporal errors in lift amplitude (spatial negative, temporal positive) and their sizes on the baseline cells and step.]**

### 5.3 Error decomposition of the coupled solver (families C, X, T2)

Figure: `figures/error_decomposition_C_X_T2`. **[RESULT: $e$, $e_f$, $e_c$ against the refinement variable of C, X and T2, with fits.]**

| Statistic | Family | $e$: class, order, $U$ | $e_f$ at finest level [%] | $e_c$: class, order or trend |
| --- | --- | --- | --- | --- |
| $\bar C_D$ | X | **[RESULT]** | **[RESULT]** | **[RESULT]** |
| $\bar C_D$ | T2 | **[RESULT]** | **[RESULT]** | **[RESULT]** |
| $\hat C_L$ | X | **[RESULT]** | **[RESULT]** | **[RESULT]** |
| $\hat C_L$ | T2 | **[RESULT]** | **[RESULT]** | **[RESULT]** |
| $St$ | X | **[RESULT]** | **[RESULT]** | **[RESULT]** |
| $St$ | T2 | **[RESULT]** | **[RESULT]** | **[RESULT]** |
| Wake velocity | X / T2 | **[RESULT]** | **[RESULT]** | **[RESULT]** |

Family C, for comparison: **[RESULT: $e$ and $e_c$ per level and class; whether the sign change of the preliminary runs persists]**.

**[RESULT: paragraph stating the spatial and temporal orders of the coupled solver as a whole, and whether $e_c$ vanishes, stays constant or grows with refinement.]**

### 5.4 Drivers of the coupling error (families T2, P, E)

Figure: `figures/error_drivers_T2_P_E`. **[RESULT: $e_c$ against $U\Delta t_p/D$, $N_x$ and $n$.]** The sensitivity figures `figures/sensitivity_exchange_ratio` (P) and `figures/sensitivity_dimensionless` ($C_f$, $C_p$, $\Lambda$) show the same runs.

| Statistic | $c_0$ [%] | $a$ ($U\Delta t_p/D$) | $b$ ($N_x$) | $d$ ($U\Delta t_f/D$) | $R^2$ | Dominant term at baseline |
| --- | --- | --- | --- | --- | --- | --- |
| $\bar C_D$ | **[RESULT]** | **[RESULT]** | **[RESULT]** | **[RESULT]** | **[RESULT]** | **[RESULT]** |
| $\hat C_L$ | **[RESULT]** | **[RESULT]** | **[RESULT]** | **[RESULT]** | **[RESULT]** | **[RESULT]** |
| $St$ | **[RESULT]** | **[RESULT]** | **[RESULT]** | **[RESULT]** | **[RESULT]** | **[RESULT]** |

**[RESULT: paragraph on whether the coupling error is a temporal error of the particle step or an error accumulated per transfer, and the resulting choice of $\Lambda$ and $N_x$.]**

### 5.5 Interface iteration (family K)

Figure: `figures/sensitivity_interface_sweeps`. **[RESULT: errors, last-sweep change of the boundary trace and cost against $K$.]**

Coupling error $e_c$ at the baseline ($h=0.04$ m, $\Delta t_f=0.008$ s, $n=5$), in per cent:

| FVM solves per exchange | $\bar C_D$ | $\hat C_L$ | $St$ |
| --- | --- | --- | --- |
| $K=1$ | +2.96 | +2.09 | +0.34 |
| $K=2$ | −0.23 | −1.52 | +0.33 |
| $K=3$ | **[RESULT]** | **[RESULT]** | **[RESULT]** |
| $K=4$ | −0.09 | −1.39 | +0.33 |
| $K=6$ | **[RESULT]** | **[RESULT]** | **[RESULT]** |
| Tolerance $10^{-5}$ (at most 6; mean **[RESULT: mean sweeps]**) | −0.09 | −1.42 | +0.33 |

With a single solve per exchange the FVM boundary trace lags the particle velocity, and the drag and lift-amplitude errors take the opposite sign and a larger magnitude: +2.96% and +2.09%. A second sweep removes almost all of this. From $K=2$ to $K=4$ the drag error changes by 0.14 and the lift-amplitude error by 0.13 percentage points, and the tolerance-driven runs agree with $K=4$ to within 0.03 points. The Strouhal number does not depend on $K$. The coupling error that remains at $K\ge2$, about −1.4% in lift amplitude and +0.33% in Strouhal number, is therefore not an error of the interface iteration; sections 5.4 and 5.6 attribute it. Two sweeps per exchange, or a boundary-trace tolerance of $10^{-5}$, are sufficient. **[RESULT: wall time per second of flow time at $K=2$ against the tolerance-driven runs, from runs made alone.]**

### 5.6 FVM box (family B) and factorial (family F)

Figure: `figures/error_box_B`. **[RESULT: $e_c$ against $R$ (with $L_f$ from $u_\mathrm{rms}$ and from $v_\mathrm{rms}$) and against $c$, at $Re=150$ and $250$.]**

| $Re$ | Downstream face $[D]$ | Side margin $[D]$ | $R$ | $c$ | $e_c$: $\bar C_D$ / $\hat C_L$ / $St$ [%] |
| --- | --- | --- | --- | --- | --- |
| 150 | **[RESULT]** | **[RESULT]** | **[RESULT]** | **[RESULT]** | **[RESULT]** |
| 250 | **[RESULT]** | **[RESULT]** | **[RESULT]** | **[RESULT]** | **[RESULT]** |

**[RESULT: paragraph on whether $R$ and $c$ collapse the box error at both Reynolds numbers, and which formation length ($u_\mathrm{rms}$ or $v_\mathrm{rms}$ peak) collapses it better.]**

Factorial $2^3$ (downstream face, inset, $n$), lift-amplitude coupling error:

| Effect | Value [percentage points] |
| --- | --- |
| Downstream face | **[RESULT]** |
| Inset | **[RESULT]** |
| $n$ | **[RESULT]** |
| Two-factor interactions | **[RESULT]** |
| Three-factor interaction | **[RESULT]** |
| Largest additivity defect | **[RESULT]** |

**[RESULT: paragraph on whether the errors of the box, the inset and the exchange ratio add, so that each can be set independently.]**

### 5.7 Particle spacing (family S)

Figure: `figures/sensitivity_particle_spacing`. **[RESULT: $e_c$ at $h_p=h_f/2$ against $h_p=h_f$, at $n=2$ and $5$, against $n$ and $C_p$.]**

| $h_p/h_f$ | $n$ | $e_c$: $\bar C_D$ / $\hat C_L$ / $St$ [%] |
| --- | --- | --- |
| 1 | 2, 5 | **[RESULT]** |
| 1/2 | 2, 5 | **[RESULT]** |

**[RESULT: paragraph on the size of the particle discretization error relative to the other coupling errors.]**

### 5.8 Transfer to other Reynolds numbers (family R)

Figure: `figures/sensitivity_reynolds_number`. **[RESULT: errors at $Re=100$, 150 and 250.]**

| $Re$ | $n$ | $\Lambda$ | $R$ | $c$ | $e_c$: $\bar C_D$ / $\hat C_L$ / $St$ [%] |
| --- | --- | --- | --- | --- | --- |
| 100 | 5 | **[RESULT]** | **[RESULT]** | **[RESULT]** | **[RESULT]** |
| 250 | 4 ($\Lambda$ matched) | **[RESULT]** | **[RESULT]** | **[RESULT]** | **[RESULT]** |
| 250 | 5 | **[RESULT]** | **[RESULT]** | **[RESULT]** | **[RESULT]** |

At other Reynolds numbers no extrapolated value is available, so these errors are coupling errors against the FVM-only run of the same $Re$, cells and step. **[RESULT: paragraph on whether the flow-scaled rules hold the error within the 2% target at both Reynolds numbers.]**

### 5.9 In-run indicator

Figure: `figures/error_indicator`. **[RESULT: $\lvert e_c\rvert$ against the developed means of the trace mismatch and the first-sweep gradient residual.]**

| Indicator | Correlation with $\lvert e_c\rvert$: $\bar C_D$ / $\hat C_L$ / $St$ |
| --- | --- |
| Trace velocity mismatch, mean | **[RESULT]** |
| Trace velocity mismatch, maximum | **[RESULT]** |
| Outflow mismatch, mean | **[RESULT]** |
| First-sweep normal residual | **[RESULT]** |
| First-sweep gradient residual | **[RESULT]** |
| Last-sweep gradient residual | **[RESULT]** |

**[RESULT: paragraph on whether any indicator separates runs within and outside the 2% target, and its threshold.]**

## 6. Recommendations

Values in flow scales, for a bluff body shedding a laminar wake. They hold together with the conditions established in the earlier study: an outflow fringe of at least two wake wavelengths at the particle-domain outflow, an FVM-only comparison with the same unbounded far field, and converged FVM linear solves (momentum tolerance $10^{-9}$).

| Parameter | Flow-scaled form | Recommended value | Evidence |
| --- | --- | --- | --- |
| Downstream transfer face | reach $R=(x_t-L_f)/L_f$ | **[RESULT]** | B, F |
| Side margin | clearance $c=(m-w-D/2)/\delta_\omega$ | **[RESULT]** | B |
| Transfer inset | in $\sigma$ | **[RESULT]** | F; earlier study |
| Blend width | in $\sigma$ | **[RESULT]** | earlier study |
| Particle-only band | in $\sigma$ | **[RESULT]** | earlier study |
| Particle step | $\Lambda=\max\lvert S\rvert\,\Delta t_p$; $U\Delta t_p/D$ | **[RESULT]** | E, T2, P |
| Exchanges per convective time | $N_x=D/(U\Delta t_p)$ | **[RESULT]** | E, T2, P |
| FVM steps per exchange | $n=\Delta t_p/\Delta t_f$, given $C_f$ | **[RESULT]** | P, E |
| Cell size | $h/\delta_\omega$ | **[RESULT]** | X |
| Particle spacing | $h_p/h_f$ | **[RESULT]** | S |
| Interface iteration | FVM solves per exchange, or boundary-trace tolerance | $K=2$, or tolerance $10^{-5}$ | K |

Overall error and orders of the coupled solver at the recommended values:

| Statistic | Spatial order (X) | Temporal order (T2) | $e$ at baseline [%] | $U$ [%] |
| --- | --- | --- | --- | --- |
| $\bar C_D$ | **[RESULT]** | **[RESULT]** | **[RESULT]** | **[RESULT]** |
| $\hat C_L$ | **[RESULT]** | **[RESULT]** | **[RESULT]** | **[RESULT]** |
| $St$ | **[RESULT]** | **[RESULT]** | **[RESULT]** | **[RESULT]** |

## 7. Limitations

- **Three levels per coupled family.** Families X and T2 have three levels each. The free-order fit then has no residual freedom ($\sigma=0$), and the combined first- and second-order fit cannot be made, so the uncertainty rests on the safety factor alone and the convergence class on two differences. The FVM-only families have four levels.
- **One body.** Only the circular cylinder was studied. The flow-scaled rules are expressed so that they can carry over, but their transfer to other shapes (the cube, the wing) is not shown here.
- **Two Reynolds numbers.** The box and transfer studies cover $Re=150$ and $250$ (with $Re=100$ in family R only), all in laminar planar shedding.
- **Interface widths from the earlier study.** The recommended inset, blend and particle-only widths rest on the one-at-a-time study of 5–12 October on an older code version and a same-cell comparison; only the inset is varied again here, in two levels of family F.
- **Wake velocity against the finest FVM-only run.** The wake-velocity errors are distances to the finest FVM-only run, not to an extrapolated value, so their total and FVM parts include the remaining discretization error of that run.
- **Limits of the uncertainty estimate.** The least-squares estimate degrades for shallow orders and noisy data. In a synthetic check with $p=1$ and 0.1% noise, the fraction of cases in which $\pm U$ covered the true error dropped to 0.67–0.84. The statistics are averaged over $40\le t\le55$ s, about 2.7 shedding periods, while the lift amplitude is still rising slowly; that sampling noise enters every family.
- **Particle step and transfer count tied.** The coupler takes one particle step per exchange, so the temporal and the per-transfer parts of the coupling error are separated only through the assumed functional forms of section 3.4, not by independent variation.
- **Repeatability.** GPU summation order is not deterministic and no repeat runs were made; differences below about 0.1 percentage points between runs are not resolved.
