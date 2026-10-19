# Coupling findings: cylinder shedding at $Re=150$

Record of what limited and what improved the accuracy of the coupled FVM–VPM cylinder. It covers the investigations of 5–12 October 2026. The quantitative sensitivity results (section 3) come from one code version and one reference. The earlier fixes (section 2) were measured on older code versions, so their magnitudes are indicative.

## 1. Case, reference and error measure

- **Flow:**
  - Laminar planar shedding, $Re=U_\infty D/\nu=150$, $D=1$ m, $U_\infty=1$ m/s.
  - Cell and particle spacing $h=0.04$ m ($D/25$).
  - FVM step $\Delta t_\mathrm{FVM}=0.008$ s.
  - Planar flow: one periodic FVM cell and one particle layer between free-slip planes one spacing apart.
- **Reference:** FVM-only, body-fitted, same near-wall resolution. Domain $[-40,60]\times[-60,60]$ m with inlet, outlet and slip walls. Run on development 45da734c, 100 s.
- **Error measure:** relative error of three statistics over developed shedding, $40\le t\le55$ s, in both the coupled run and the reference (`assets/plot_forces.shedding_statistics`):
  - the mean drag $\bar C_D$;
  - the lift amplitude $\hat C_L=\tfrac12(\max C_L-\min C_L)$;
  - the Strouhal number $St$, from the mean spacing of upward zero crossings of $C_L$.

  Target: 2%.

| Reference, $40\le t\le55$ s | $\bar C_D$ | $\hat C_L$ | $St$ |
| --- | --- | --- | --- |
| This reference | 1.310 | 0.501 | 0.1816 |
| Commonly cited 2-D values at $Re=150$ | ≈1.32–1.34 | ≈0.52 | ≈0.183 |

The reference lies 1–4% below the literature. This is consistent with a resolution of 25 cells per diameter. The coupled runs are compared with this reference, at the same resolution, so the comparison isolates the coupling error.

## 2. Corrections that removed systematic coupling errors

Listed in the order they were found. Each one removed an error larger than the 2% target.

### 2.1 Abrupt particle removal at the particle-domain outflow

- **Symptom (early code):** $\hat C_L$ −21%, shedding period +3%, $\bar C_D$ −6.6%.
- **Cause:** particles were deleted when they crossed the outflow face at $x=15$ m.
  - Once the wake reached that face ($t\approx18$ s), the circulation left in the domain swung by ±1.4 m²/s every half-cycle; physically it should stay near zero.
  - The missing tail of the street induced a phase-locked cross-flow of 1.5–3% $U_\infty$ at the cylinder.
- **Fix:** a smooth outflow fringe, `StabilizationConfig.bounded_domain(..., outflow_fringe_depth=...)`, at least two wake wavelengths deep. The tutorial uses $x\in[15,30]$ m.
  - In a model street, a sharp cut gives a 1.5% velocity error at the body and a 15 m fringe 0.02%.
- **Guideline:** let shed vortices fade out; never delete them abruptly downstream.

### 2.2 Confined reference

- **Symptom:** a coupled–reference drag offset of about +3%.
- **Cause:** the earlier reference had slip walls at $y=\pm10D$ and its inlet at $x=-8D$, a 5% blockage. It raised the velocity at the coupling boundary by 1.6% compared with free space. A free-space particle wake cannot reproduce that.
- **Fix:** widen the reference to $[-40,60]\times[-60,60]$ m.
- **Guideline:** compare against a reference with the same far-field conditions as the particle domain: unbounded flow.

### 2.3 Planar flow from three-dimensional particles

One particle layer between free-slip planes one spacing apart, together with all its mirror images, induces the planar flow:
- image rows beyond 15 shells are summed in closed form;
- the grid-based diffusion and the renewal include every image.

On a 2-D Gaussian vortex, the induced velocity matches the planar solution to below $10^{-5}$. The particles keep their three-dimensional physics, including stretching and the full vector kernel.

### 2.4 FVM box ending inside the vortex-formation region

- **Symptom (earlier code):** box $[-1.6,2.4]\times[-1.6,1.6]$ m gave $\bar C_D$ −0.3% and $St$ +1%, but $\hat C_L$ −6.7%.
- **Cause:** the overlap was self-consistent: particle and FVM velocities at $x=2$ agreed within 0.2%. The bias came from particle cores, of radius $\sigma=h$, smoothing shear layers that were still thin when they crossed into the particle region.
- **Fix:** box $[-2.4,4.4]\times[-2.4,2.4]$ m gave $\bar C_D$ −0.07%, $\hat C_L$ −0.93% and $St$ +0.69% (over $40\le t\le60$ s, old solver).
- Quantified by the box-margin study in section 3.1.

### 2.5 Linear-solver tolerance of the FVM momentum equation

- **Symptom:** the earlier reference and every coupled FVM region carried a solver-tolerance error of the same size as the coupling effects under study.
- **Measurement:** compared with a converged solution (momentum tolerance $10^{-9}$, two preconditioners agreeing to $2\times10^{-5}$), the loose setting (absolute $10^{-6}$, relative 0.05) gave:
  - $\bar C_D$ −0.41%;
  - $\hat C_L$ −2.62%;
  - $St$ −0.69%.
- **Fix:** the FVM now defaults to a momentum tolerance of $10^{-9}$, with a DILU preconditioner (development 9ac20fd8 and 45da734c). It is also faster: 0.9 s instead of 1.5 s per reference step.
- **Guideline:** before attributing a difference of a few per cent to the coupling, converge the linear solves of both the reference and the coupled FVM.

## 3. Sensitivity of the coupled forces to the coupling parameters

**Code:** development dfa6b54f, plus a GBD robustness fix (66964084) that only acts on a degenerate pruning group; the runs that reached it are noted in section 4.

**Baseline:**
- box margin $m=2.4$, giving the FVM box $(-m,\,m+2,\,-m,\,m)\,D$;
- 5 FVM steps per exchange (particle step 0.04 s);
- transfer inset of 8.75 cells.

Each parameter is varied with the other two at their baselines. The errors are signed, coupled against reference.

| Run | $\bar C_D$ | $\hat C_L$ | $St$ |
| --- | --- | --- | --- |
| **Baseline** | −0.18% | −1.33% | +0.29% |
| Box margin 1.0 | +8.49% | +3.19% | +9.59% |
| Box margin 1.4 | −0.66% | −4.75% | +0.98% |
| Box margin 1.8 | −0.27% | −1.35% | +0.51% |
| Box margin 2.2 | −0.19% | −1.26% | +0.36% |
| Box margin 2.6 | −0.23% | −1.74% | +0.22% |
| 2 FVM steps per exchange | −0.74% | −2.76% | −0.46% |
| 4 FVM steps per exchange | −0.23% | −1.44% | +0.22% |
| 6 FVM steps per exchange | −0.11% | −1.20% | +0.39% |
| 8 FVM steps per exchange | — | — | — |
| Transfer inset 4 cells | −0.25% | −1.61% | +0.24% |
| Transfer inset 6 cells | −0.23% | −1.49% | +0.26% |
| Transfer inset 14 cells | −0.14% | −1.21% | +0.32% |
| Transfer inset 20 cells | −0.07% | −0.93% | +0.39% |

The 8-step run is not admissible: the particle stability check stops it at its first step (section 3.2). `figures/sensitivity` plots the absolute values of these errors; the box margin 1.0 point therefore lies below the margin 1.4 point although its regime is worse (section 3.1).

The lift amplitude is always the most sensitive statistic. The mean drag stays within 0.75% in every admissible run.

### 3.1 FVM box size, and the vortex-formation length

The coupling geometry defines a **pure-FVM core**: the region where the FVM vorticity is used without blending. Its faces lie inside the box faces by the transfer inset plus the particle-only band (2 cells) plus the blend width (6 cells), 16.75 cells or $0.67D$ at the baseline settings.

| Box margin | Pure-FVM core half-width $y/D$ | Core ends at $x/D$ | $\hat C_L$ error |
| --- | --- | --- | --- |
| 1.0 | 0.33 (inside the cylinder) | 2.33 | +3.19% ($\bar C_D$ +8.5%, $St$ +9.6%) |
| 1.4 | 0.73 | 2.73 | −4.75% |
| 1.8 | 1.13 | 3.13 | −1.35% |
| 2.2 | 1.53 | 3.53 | −1.26% |
| 2.4 | 1.73 | 3.73 | −1.33% |
| 2.6 | 1.93 | 3.93 | −1.74% |

The error plateaus once the margin reaches $1.8D$, and rises sharply below it. On the plateau the lift error lies between 1.26% and 1.74%. Margins 1.8, 2.2 and 2.4 agree within 0.1 percentage points. Margin 2.6 is about 0.4 points worse, and every one of its lift peaks in the window is lower by that amount, so the difference is systematic rather than sampling scatter. With the inset fixed, a larger box also enlarges the region renewed from the FVM at every exchange, further into the formed wake. That is consistent with the renewal-smoothing interpretation of sections 3.2 and 3.3, which would make an excessively large box mildly counterproductive. It is an observation, not an isolated cause.

At margin 1.0 the flow enters a different regime: all three statistics are overpredicted, $\bar C_D$ +8.5% and $St$ +9.6%. The coupling boundary lies only $0.5D$ from the wall, and the pure-FVM core lies inside the cylinder, so part of the wall vorticity is blended into the particle representation. A simultaneous rise of drag and shedding frequency is the signature of a confined, blocked flow. This suggests the boundary condition taken from the particles misrepresents the near-field disturbance of the body. This interpretation has not been isolated. **Guideline:** the box must also keep the coupling boundary out of the body's near field, so that the pure-FVM core contains the whole wall with margin. Margins below about $1.4D$ are unusable here.

**Which wake length scale sets the threshold.** The time-averaged FVM fields were probed on the wake centreline: 16 snapshots, every 4 s over $40\le t\le100$ s for the reference and over $40\le t\le55$ s for the coupled runs.

| Run | End of mean reverse flow, $\bar u=0$, $x/D$ | Peak of centreline $v_\mathrm{rms}$, $x/D$ (value) |
| --- | --- | --- |
| Reference | 1.57 | 2.72 (0.578) |
| Box margin 1.4 | 1.58 | 2.28 (0.571) |
| Box margin 1.8 | 1.56 | 2.44 (0.587) |
| Box margin 2.2 | 1.55 | 2.40 (0.592) |
| Box margin 2.4 | 1.55 | 2.40 (0.594) |

- **The mean recirculation length does not set the threshold.** The reverse flow ends at $x\approx1.57D$, which is $1.07D$ behind the rear surface. Every box reproduces this, including the box 1.4 run that fails. Its pure-FVM core already extends more than one diameter past the end of the reverse flow.
- **The vortex-formation region does.** This is where the shear layers roll up and the shed vortex reaches its peak strength, marked by the peak of the transverse velocity fluctuation on the centreline.
  - With margin 1.4, the pure-FVM core ends at $2.73D$, inside that region. The vortex completes its formation in the blended and particle zones.
  - The fluctuation peak moves upstream to $2.28D$ and weakens, and the lift amplitude falls by 4.75%.
  - From margin 1.8 upward, the core extends past the formation region, and the formation peak settles at the same place (about $2.4D$) in every run.
- The earlier result of section 2.4 points the same way. With the lateral margin fixed at $2.4D$, moving the downstream face from $2.4D$ to $4.4D$ moved the core end from $1.73D$ (already past the end of the reverse flow) to $3.73D$. The lift error fell from 6.7% to about 1%.

**Guideline:** extend the pure-FVM core downstream beyond the vortex-formation length, the peak of centreline $v_\mathrm{rms}$. Here that is $\approx2.4$–$2.7D$ from the cylinder centre, about $1.7\times$ the mean recirculation length. The end of the reverse flow is a necessary but not sufficient bound.

**Caveats:**
- The centreline $v_\mathrm{rms}$ is flat within a few per cent between $x=2D$ and $3D$. With 16 snapshots, the peak location is uncertain to about $\pm0.3D$.
- In this study the margin enlarges the box downstream and sideways together; for margin 1.4, the core half-width is also only $0.73D$. A run with a narrow but long box would separate the two effects, for example a lateral margin of 1.4 with the downstream face at $4.4D$.

### 3.2 Exchange step: FVM steps per particle step

| FVM steps per exchange | Particle step [s] | $\hat C_L$ error |
| --- | --- | --- |
| 2 | 0.016 | −2.76% |
| 4 | 0.032 | −1.44% |
| 5 | 0.040 | −1.33% |
| 6 | 0.048 | −1.20% |
| 8 | 0.064 | not admissible |

- **More frequent exchanges are less accurate.** The likely cause, not yet isolated: every exchange renews the particles in the transfer region from the FVM vorticity, interpolating with the M4′ kernel. The smoothing this adds accumulates with the number of exchanges per shedding period, and it weakens the shed vortices. A test would compare the transfer-region circulation, or the peak vorticity of a shed vortex, at equal flow times between the 2- and 5-step runs.
- **The particle step has a stability ceiling at the wall.** With 8 FVM steps per exchange, the particle step stops at its first step: particles adjacent to the cylinder see strain rates $|S|\approx19.4$ s$^{-1}$, which makes the Lagrangian CFL number $|S|\,\Delta t=1.24>1$. The solver reports a ceiling of $\Delta t_\mathrm{VPM}\le0.052$ s $\approx D/(19U_\infty)$.
- **The gain continues up to the limit.** The 6-step run ($\Delta t_\mathrm{VPM}=0.048$ s, 8% below the ceiling) ran stably to the end and is the most accurate: $\bar C_D$ −0.11%, $\hat C_L$ −1.20%. The Strouhal error drifts slowly the other way, +0.22% → +0.39% from 4 to 6 steps, a small trade-off.
- **Guideline:** exchange as rarely as the particle stability limit allows, $\Delta t_\mathrm{VPM}<1/\max|S|$ at the wall, with a margin of about 10%. Here that is 6 FVM steps of 0.008 s. Very frequent exchanges, 2 steps, push the lift error above the 2% target.

### 3.3 Transfer-region inset

The inset is the number of cells between the FVM box faces and the transfer-region faces. Particles in this band are never renewed from the FVM; they evolve as free particles and supply the FVM boundary condition.

| Inset [cells] | Pure-FVM core: half-width, downstream end ($x/D$) | $\bar C_D$ | $\hat C_L$ | $St$ |
| --- | --- | --- | --- | --- |
| 4 | 1.92, 3.92 | −0.25% | −1.61% | +0.24% |
| 6 | 1.84, 3.84 | −0.23% | −1.49% | +0.26% |
| 8.75 | 1.73, 3.73 | −0.18% | −1.33% | +0.29% |
| 14 | 1.52, 3.52 | −0.14% | −1.21% | +0.32% |
| 20 | 1.28, 3.28 | −0.07% | −0.93% | +0.39% |

- **A deeper inset is monotonically more accurate.** From 4 to 20 cells the lift error falls from 1.61% to 0.93%, and the drag error from 0.25% to 0.07%. That holds even though the pure-FVM core shrinks; at 20 cells it still ends past the vortex-formation length (section 3.1).
- **Interpretation:** a deeper inset places the boundary data in a wider band of particles that evolve freely instead of being re-interpolated from the FVM at every exchange. This trend and the exchange-frequency trend (section 3.2) point the same way: the less often, and the less widely, particles are renewed by M4′ interpolation, the more accurate the lift. Both are consistent with renewal adding numerical smoothing, but neither isolates it. The Strouhal error rises slightly with the inset (+0.24% → +0.39%), as with fewer exchanges.
- **Guideline:** use a deep inset, about 15–20 cells here. Combine it with the box rule, so the pure-FVM core, now further inside the box, still extends past the vortex-formation length. A deeper inset therefore needs a correspondingly larger box.

## 4. Numerical and practical notes

- **Observation times on exchanges.** Iterated coupling accepts only output and sampling times that fall on exchanges. For FVM-step counts other than 5, the case snaps its observation intervals to whole exchanges (`on_exchanges` in `setup.py`).
- **GBD pruning closure.** One run (box margin 1.4, $t=37.5$ s) stopped once in the grid-based diffusion: a nearly degenerate group of retained particles could close its moments in float64 but not in float32 storage. Resumed from its backup, the run passed the same point. Since 66964084 such a group keeps its original particles, so the run continues and moments stay exact. That fallback has not fired in any later run.
- **Cost:**
  - About 3 h per run to $t=55$ s on an RTX 3060 Laptop GPU (CUDA image sums), alone on the GPU. Particle counts reach about 60k.
  - The particle image field grows to about 3 GB of GPU memory per run, so on a 6 GB GPU the runs execute one at a time.
  - Without free GPU memory the image sums fall back to the CPU, about 50 times slower.
- **Precision of the statistics.**
  - The window $40\le t\le55$ s holds about 2.7 shedding periods.
  - The lift amplitude is still growing slowly there: the reference peaks rise from 0.4992 to 0.5000 within the window and average 0.5025 over $55\le t\le100$ s. Absolute amplitudes are therefore not fully converged at 55 s. Every run is measured over the same window, so the comparison is consistent.
  - Runs with physically equivalent settings (box margins 1.8, 2.2, 2.4) agree to about 0.1 percentage points in lift. Differences of 0.3 points or more between runs are resolved.
  - GPU summation order is not deterministic, so repeated runs differ at round-off level. Repeat runs were not made.
- **Reproduce:** run `assets/run_sensitivity_study.sh`, then `./allplot.sh`, which writes `figures/sensitivity` and prints the error table.

## 5. Guidelines for the coupled solver

From this case, for a bluff body shedding a laminar wake at cell size $h=D/25$:

1. **Box size, lower bound.** Keep the coupling boundary out of the body's near field, and the whole wall inside the pure-FVM core. With the boundary $0.5D$ from the wall (margin 1.0), drag and Strouhal number rise by 8–10%, as in a blocked flow.
2. **Box size, accuracy.** Extend the pure-FVM core downstream past the vortex-formation length, the peak of centreline $v_\mathrm{rms}$, here $\approx2.4$–$2.7D$ from the centre. The end of the mean reverse flow ($1.57D$) is not sufficient.
   - A core that ends inside the formation region pulls the formation upstream and lowers the lift amplitude: −4.75% at margin 1.4.
   - Beyond the bound the error plateaus. An excessively large box brings no gain and was slightly worse at margin 2.6.
3. **Exchange step.** Exchange as rarely as the particle stability limit allows: $\Delta t_\mathrm{VPM}$ about 10% below $1/\max|S|$ at the wall, here 6 FVM steps of 0.008 s. Frequent exchanges (2 steps) cost accuracy (lift −2.76%). Steps beyond the limit are rejected (8 steps).
4. **Transfer inset.** Use a deep inset, 15–20 cells, and enlarge the box so that rule 2 still holds. Lift error falls from 1.61% at 4 cells to 0.93% at 20 cells.
5. **Reference and solver.** Compare against a reference with the same far-field conditions (section 2.2). Converge the FVM linear solves before attributing per-cent differences to the coupling (section 2.5).

All admissible runs that satisfy rules 1 and 2 reproduce the reference within the 2% target in every statistic. The best combination tested, an inset of 20 cells, gives $\bar C_D$ −0.07%, $\hat C_L$ −0.93% and $St$ +0.39%. Rules 3 and 4 were varied one at a time; their combination, for example 6 steps with a 20-cell inset in a correspondingly larger box, has not yet been run.

## 6. Corroboration in three dimensions: the cube at $Re=1000$

The earlier coupled cube setup (`../02_cube_flow`, before October 2026) had its FVM box faces only $0.485D$ from the cube faces, with a transfer inset under one cell. That is the near-field regime of the cylinder's smallest box (section 3.1).
- Compared with the FVM reference over $10\le t\le22$ s, its mean drag was **7.9% too high**. The cylinder box with its boundary $0.5D$ from the wall gave +8.5%.
- The geometry, Reynolds number and LES closure all differ, so the matching overprediction supports the near-field rule independently.
- Caveats: these runs used an earlier code version over a short window, and the reference itself was not developed (next point).

The same comparison exposed a second requirement: **break the symmetry of the start-up**.
- The impulsively started cube reference kept a perfectly symmetric wake on every grid. Side force and lift stayed below $10^{-4}$ in coefficient for the whole 30 s.
- Its drag was still drifting, from 1.05 to 0.93, between $10$–$20$ s and $20$–$30$ s.
- The Strouhal number reported by its grid study, 0.066 on all three grids, was therefore a sampling artefact, not shedding.
- The cylinder and the cube avoid this with a vortex placed behind the body in the initial field, the same in the coupled case and its reference. A transverse start-up flow is not used: the zero-pressure outlet of the reference resists a uniform transverse acceleration, so the two cases received different start-up impulses and shed with a permanent phase offset. Both run to 100 s, with statistics taken over $t\ge40$ s.

## 7. Planned studies

The studies that attribute the remaining coupled-solution error to its sources (exchange ratio at a fixed particle step, particle resolution, interface sweeps, other Reynolds numbers) are planned in [the coupling-error plan](../../../../docs/plans/2026-10-19-coupling-error-sources.md) and run as sections of `assets/run_sensitivity_study.sh`.
