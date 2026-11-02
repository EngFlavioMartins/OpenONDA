# Cylinder coupling sensitivity study — assessment and redesign, 2026-10-31

Scope: `tutorials/coupled_fvm_vpm/01_cylinder_shedding_flow` on the laptop checkout (36 coupled runs and 11 FVM-only references under `sensitivity/`, tables in `figures/auxiliary/`, `assets/coupling_findings.md`, `assets/error_analysis.py`). Goal as stated: let a user find how to reduce coupling-interface errors from a sensitivity analysis of the interface parameters, with the interface sensitivity separated from the FVM's and the VPM's own errors.

The design is sound in outline — a two-part decomposition $e = e_\mathrm{FVM} + e_\mathrm{coupling}$ against an extrapolated FVM-only value, one-at-a-time sweeps, and per-transfer indicators. Five things keep the present runs from answering the question. Each is checked against the stored data.

## 1. What the stored runs do and do not show

### 1.1 The statistics window sits inside the transient and has a noise floor comparable to many reported effects

The statistics (`assets/plot_forces.py`) are the mean drag, the lift amplitude as $\tfrac12(\max C_L-\min C_L)$, and $St$ from the mean of the upward lift crossings, over $40\le t\le 55$ s: 2.7 periods, so two complete crossing intervals. From the 100 s reference (`reference_flow/samples/forces_history.csv`), sliding the 15 s window:

| window [s] | $\bar C_D$ | $\hat C_L$ (ptp) | $\hat C_L$ (mean of per-period ptp) | $St$ |
|---|---|---|---|---|
| 35–50 | 1.30897 | 0.50097 | 0.49791 | 0.18148 |
| 40–55 | 1.30889 | 0.50172 | 0.50081 | 0.18154 |
| 45–60 | 1.30923 | 0.50231 | 0.50201 | 0.18157 |
| 55–70 | 1.31208 | 0.50276 | 0.50268 | 0.18157 |
| 70–85 | 1.30920 | 0.50274 | 0.50271 | 0.18155 |
| 80–95 | 1.31135 | 0.50272 | 0.50264 | 0.18154 |

The lift amplitude is still rising at 40–55 s (+0.2 % to saturation at ≈55 s; once saturated it is steady to 0.01 %), and the mean drag scatters by 0.2 % peak to peak between windows of the developed flow (σ = 0.07 %). The same drift is in the coupled runs (baseline $\hat C_L$ 0.4934 → 0.4944 between 40–50 and 45–55 s; $N=2$: 0.4864 → 0.4871). The run-to-run repeat (`_rep1`) agrees to 0.001 points, but that is GPU reproducibility, not sampling error. So the resolution of the present measurement is about

- $\bar C_D$: ±0.1–0.2 points (window scatter), $\hat C_L$: ±0.2 points (transient drift, partly cancelling between runs), $St$: ±0.01 points,

and the following reported differences are at or below it: transfer inset 4 / 8.75 / 14 cells (−1.67 / −1.42 / −1.29), $k=2$ vs $k=4$ vs tolerance (0.15), halving $\Delta t_\mathrm{FVM}$ at a fixed interval (0.07), $Re$ dependence of the inset. They should be reported as "no resolvable effect", not as trends. The amplitude as a global peak-to-peak also makes one extreme period decide the number; the per-period mean (last column above) is steadier.

### 1.2 The "coupling error" contains the reference's mesh, not only the interface

$e_\mathrm{coupling} = (Q_\mathrm{coupled} - Q_\mathrm{FVM})/Q_\mathrm{ext}$ uses an FVM-only reference whose cells coarsen to $2h$ beyond $x = 1.5$ m and to $4h$ beyond the near wake, while the coupled FVM keeps $h$ cells up to $x = 4.4$ m, and whose far field is slip walls at $\pm 60D$ and an inlet at $-40D$ rather than free space. The difference therefore mixes three things: the interface, the near-wake resolution difference of the two FVM meshes, and the far-field model. Section `g` (`--fine-wake-end 4.4`) removes the second; it has not been run, so every $e_\mathrm{coupling}$ number is provisional. The third is small at $\pm 60D$ but is not zero for $\bar C_D$ (blockage at $\pm 10D$ was 3 %).

### 1.3 The interval series fits one variable under three names

`error_drivers.csv` lists "particle step", "Lagrangian CFL number", "coupling intervals per convective time" and "particle Courant number" with identical $R^2 = 0.99977$ and identical exponent. They are the same series rescaled, because every interval run changes $\Delta t_\mathrm{VPM}$ alone. The statement "error grows as $N^{1.7}$" is therefore equally "error grows as $\Delta t_\mathrm{VPM}^{-1.7}$", and the data support it directly: at $N=2$ the lift-amplitude coupling error is −7.08 % ($\Delta t_\mathrm{VPM}=0.008$), −2.82 % (0.016), −1.47 % (0.032), and all runs at 0.04 s give −1.35 to −1.42 % whatever the FVM step. **Shorter coupling intervals give larger error.** That is not a truncation error of any discretisation; it is an error injected per transfer, growing faster than the transfer count (exponent 1.7 > 1), i.e. the per-transfer error itself grows as the interval shrinks. The strength-ratio data point at the mechanism: $\kappa$ sits at $\kappa_\mathrm{max} = 1.8$ in 82 % of the $N=2$ transfers and 100 % at $N=2$, $\Delta t_\mathrm{FVM}=0.004$, in 0 % of the baseline's. When the interval is short the physical change of the FVM vorticity between transfers is small, the weighted mismatch $\Gamma_\mathrm{mis}$ is dominated by remeshing and representation noise, and the limiter and thresholding act on noise. Section `b1` tests exactly this and is the single most informative run not yet made.

A second confound in the same series: the number of GBD steps per shed vortex also scales with $1/\Delta t_\mathrm{VPM}$. The 28 October check (shed vortices agree within 2 % between $N=5$ and $N=2$) argues against GBD; a $\kappa_\mathrm{max}$ sweep settles it.

### 1.4 The baseline FVM domain is not converged in the parameter with the largest effect

Lateral-clearance series at $Re=150$, $e_\mathrm{coupling}(\hat C_L)$: −5.37 (margin 1.4), −3.57 (1.6), −2.00 (1.8), −1.42 (2.4). The increments are 1.8, 1.6, 0.6 points: still moving at 2.4 D. The baseline that every other sweep is built on may carry 0.5–1 point of domain error, and the 2×2 factorial already showed that domain and interval errors interact (defect 2.75 points). Every interface sensitivity measured at margin 2.4 is then a sensitivity *at that domain error*, which is what a user will not have. Two more margins (3.2, 4.0) at $N=5$ cost CPU only (the FVM is serial on the slab) and fix the baseline.

### 1.5 The coarse levels are outside the admissible range, so the order fits are not meaningful

At $h=0.08$ m the median $\kappa$ is 1.92–2.08, above $\kappa_\mathrm{max}=1.8$: the limiter is not even holding, and the sign of $e_\mathrm{coupling}$ flips between 0.08 and 0.0566 m. Three-level fits through such points give "orders" of 3–8 with uncertainties of 4–7 % (`error_orders.csv`, rows 9–20). These should not be reported as orders. Spatial order of the coupled solver needs levels that are all admissible (section `a`, which keeps $\Lambda = 0.83$ on the finest cells) and ideally a fourth level.

### 1.6 Two of the three questions have no runs

The brief asks for the sensitivity of the method to the FVM's and to the VPM's own parameters, at a fixed interface. The study varies the FVM cell size and step (confounded with the particle spacing and $\Delta t_\mathrm{VPM}$ in families C and T2; isolated in X and P), and the particle spacing only through section `f`, which does not fit on the GPU. Nothing varies: the VPM integrator, the GBD removal floor, the particle spacing *coarser* than the cells (cheap and informative), the FVM corrector count, or the FVM Courant target now that the controller exists. Without those, "the interface is the dominant error" cannot be claimed, only "the interface error is smaller than the FVM's spatial error on these cells".

## 2. Redesign

Principle: every sweep changes one quantity, holds the others at a *converged* baseline, and is read against a resolution floor measured on the same statistics. Runs are grouped by what they isolate, which is also how a user will look them up.

### 2.0 Measurement (before any new sweep)

1. **Noise floor.** `assets/window_uncertainty.py` (added) slides the statistics window over every run that reaches ≥ 70 s and reports the spread of $\bar C_D$, $\hat C_L$ and $St$; use the 100 s runs of section `h` for the coupled floor. Report every difference in the study with that floor as its error bar; call anything below $2\sigma$ unresolved.
2. **Window.** Move the statistics window to $60\le t\le 100$ s for runs that go to 100 s (seven to eight periods, after amplitude saturation). Keep 40–55 s only for comparing with stored runs, and mark it as such.
3. **Amplitude.** Add the per-period mean amplitude next to the global peak-to-peak; the two should agree within the floor.
4. **Reference.** Run `g` (fine wake to $x=4.4$ m) at $\Delta t = 0.008$ and 0.004 s. Redefine $e_\mathrm{coupling}$ against this reference; the present numbers move by the near-wake mesh difference, which is then quantified once rather than hidden.

### 2.1 Domain (D): converge the baseline

`box_conv`: margins 3.2 and 4.0 D (downstream 5.2 and 6.0 D), $N=5$, $Re=150$. Expect the lateral series −5.4, −3.6, −2.0, −1.4, … to flatten. The converged margin becomes the baseline of 2.2–2.4 (or the sensitivities there are shown to be unchanged from margin 2.4; either way the claim is then grounded).

### 2.2 Interface (I): one parameter at a time, FVM and VPM fixed

All at $h=0.04$ m, $\Delta t_\mathrm{FVM}=0.008$ s, $\Delta t_\mathrm{VPM}=0.04$ s, coupling iterations to tolerance.

| Parameter | Levels | Isolates | Status |
|---|---|---|---|
| $\kappa_\mathrm{max}$ | 1, 1.8, 2, 3, 10 | the strength limiter; the gain $\gamma=\min(\kappa_\mathrm{max}-1,1)$ is full from 2, so 2 / 3 / 10 separate gain from bound | `b1` (1, 3) + new 2, 10 |
| $\tau_\mathrm{transfer}$ | 0.01 (= GBD floor, the minimum), 0.05, 0.2 | smooth thresholding at the transfer | new (`_tau`) |
| moment-correction cap | 0.02, 0.08, 0.3 | the impulse restoration | new (`_mc`) |
| $\delta_0$, $\delta_1$ | (0, 2, 4), (3, 6, 10) | blending depth | `e` |
| inset | 4, 8.75, 14 | stored; re-read against the floor | — |
| iterations / tolerance | 1, 2, 4, tol | stored | — |

### 2.3 Interval (T): the per-transfer error

`n_interval`: $N = 1, 2, 3, 4, 5, 6$ at $\Delta t_\mathrm{FVM} = 0.008$ s (1, 3, 6 new; 6 is the largest admissible, $\Lambda=0.93$), each at $\kappa_\mathrm{max} = 1.8$ **and** 3. If the two curves coincide the limiter is not the mechanism and the next candidates are the remeshing (M4′) at every transfer and the previous-correction first estimate; if the $\kappa_\mathrm{max}=3$ curve is flat the mechanism is identified and the user rule becomes "do not shorten the interval below the one the strain admits, or raise $\kappa_\mathrm{max}$". Plot against $\Delta t_\mathrm{VPM}$ with $\Delta t_\mathrm{FVM}$ as the marker so the collapse of 1.3 is visible rather than inferred.

### 2.4 VPM side (V): particle parameters at a fixed FVM and interface

All at the baseline FVM and $\Delta t_\mathrm{VPM}=0.04$ s.

| Parameter | Levels | Note |
|---|---|---|
| particle spacing $h_p$ | 0.08 (new, cheap), 0.04, 0.02 (`f`, 12 GB) | coarsening gives the first derivative without the big GPU |
| integrator | RK2 (baseline), RK4 | RK2 at $\Lambda=0.78$ is the suspect |
| vorticity floor | 0.001, 0.01, 0.1 s⁻¹ | GBD removal of weak particles |

### 2.5 FVM side (F): mesh and solver at a fixed interface and VPM

| Parameter | Levels | Note |
|---|---|---|
| cells at fixed $\Delta t_\mathrm{FVM}=0.004$, $\Delta t_\mathrm{VPM}=0.04$ | 0.08, 0.0566, 0.04 (stored, XP) + 0.0283 (new, needs ~6 GB) | interface widths fixed in 0.04 m cells |
| outer correctors | 1, 2, 3 | PIMPLE convergence per substep |
| Courant control | $Co_\mathrm{max} = 0.5$, 0.9 with $\Delta t_\mathrm{VPM}\le 0.04$ s | what a user gets with the controller on |

The Courant runs are not for isolation — the coupler scales the interval with the CFL step, so $\Delta t_\mathrm{VPM}$ varies — but they are what a user runs, and they show whether the fixed-step rules carry over. They use `--max-courant` with the FVM step argument as the initial/maximum step.

### 2.6 Order (only after I, T, V are known to be monotone)

`a` (four cell sizes at $\Delta t_\mathrm{VPM}=0.032$ s) and `c` ($N=20$, 40 at 0.04 s). Report an order only where the three or four levels are admissible and the convergence class is monotone; otherwise report the levels.

### Priority and cost

1. `h` (two 100 s runs) and `g` (two references) — measurement. CPU + one GPU run at a time.
2. `b1` extended to $\kappa_\mathrm{max}=10$, and `n_interval` — the mechanism of the dominant error. Seven GPU runs.
3. `box_conv` — the baseline. Two runs, mostly CPU.
4. `i_transfer` ($\tau$, moment cap), `v_side`, `f_side` — the two unanswered questions. About ten runs; the $h_p=0.08$ run is the cheapest in the study.
5. `e`, `a`, `c`, `f`.

Every section is in `assets/run_sensitivity_study.sh` under these names and resumes from backups as before.

## 3. Interpretation framework for the write-up

- One table per isolated group (D, I, T, V, F) with $e_\mathrm{coupling}$ per statistic ± the noise floor, and the per-transfer indicators of the run (κ median, limit fraction, mean boundary mismatch, removed-strength fraction) in the same row. A parameter is "sensitive" when its range moves the error by more than $2\sigma$ and the indicators move with it.
- A normalised sensitivity $S_i = \Delta e / \Delta \log p_i$ per parameter (units: points per factor of two) ranks the groups in one chart. The ranking, not the absolute numbers, is what a user transfers to another body.
- Keep the two-term decomposition, but against the fine-wake reference, and state what remains in $e_\mathrm{coupling}$ (far-field model, span treatment).
- Report the interval dependence as $e(\Delta t_\mathrm{VPM})$ at two $\kappa_\mathrm{max}$, with the stored $\Delta t_\mathrm{FVM}$ variants as markers; do not fit an exponent to six points of one variable.
- Rules for users then follow from the ranked groups: domain first (D), then interval and limiter together (T), then the interface details (I) only if they exceed the floor.

## 4. Changes made to the case (2026-10-31, not committed to git)

- `setup.py`: new inputs `--transfer-removal-threshold`, `--moment-correction-fraction`, `--vpm-integrator {rk2,ssprk3,rk4}`, `--vorticity-floor`, `--max-courant`, `--fvm-outer-correctors`. With `--max-courant` the FVM step argument is the initial and maximum step and `fvm.MaximumCourantTimeStep` controls the step; the VPM `time_step_size` remains the maximum coupling interval.
- `assets/run_sensitivity_study.sh`: sections `box_conv`, `i_transfer`, `n_interval`, `v_side`, `f_side`; `b1` extended with $\kappa_\mathrm{max}=2$ and 10. `assets/error_analysis.py`: the baseline test accepts the integrator name. Run-name suffixes `_tau`, `_mc`, `_rk`, `_vf`, `_co`, `_oc`, parsed by `assets/plot_sensitivity.py`.
- `assets/window_uncertainty.py`: the sampling floor of the statistics from sliding windows, written to `figures/auxiliary/window_uncertainty.csv`.

Not changed: the statistics window and the amplitude definition in `assets/plot_forces.py` (a change there alters every stored table; do it together with the move to 100 s runs).
