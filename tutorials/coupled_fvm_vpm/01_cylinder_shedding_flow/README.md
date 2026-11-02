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

`./allrun.sh` removes `solution/`, `samples/` and `figures/` before simulating from $t=0$. `./allcontinue.sh` resumes the latest accepted coupled backup. The mesh remains in `assets/` for reuse.

## Results

| File | Content |
| --- | --- |
| `figures/forces` | Drag and lift against the reference, with mean drag, lift amplitude and Strouhal number |
| `figures/velocity_profiles` | Mean velocity and transverse-velocity fluctuation at $x/D=2$ and $4$ |
| `figures/velocity_fields`, `figures/vorticity_fields` | Reference, coupled and difference fields |
| `figures/coupling_cost` | Wall time by FVM, VPM, coupling and output work; particle count |
| `figures/cylinder_shedding.gif` | Developed vorticity field and force history |

## Sensitivity study

The sensitivity study varies the FVM domain, transfer region, particle interval, particle spacing, coupling iterations and Reynolds number. It uses independent copies under `sensitivity/`; ordinary cleanup leaves that directory intact.

```bash
assets/run_sensitivity_study.sh             # every section, one run after another
assets/run_sensitivity_study.sh b1 d        # the named sections
assets/run_sensitivity_study.sh --list      # section names
./allplot.sh
```

The runs are stored under their parameter names, so launching a section again resumes its runs and skips those that reached the end time. Before each coupled run the script waits until enough GPU memory is free (3.5 GB by default, `MIN_FREE_GPU_MIB`), so it does not start next to another GPU job such as the cube or wing tutorial. Statistics use $40\le t\le55$ s in every run.

### Run tracker, solver `3f771630` (campaign of 2 to 8 November 2026)

Every run below imports the frozen export of commit `3f771630`; runs of other commits are archived and never compared. Baseline: $h=0.04$ m (cells and particle spacing), $\Delta t_\mathrm{FVM}=0.008$ s, $N=5$ ($\Delta t_\mathrm{VPM}=0.04$ s), $y_\mathrm{box}=2.4$ m, $x_\mathrm{box}=4.4$ m, $L_\mathrm{inset}=8.75h$, $\kappa_\mathrm{max}=1.8$, $\tau_\mathrm{t}=0.05$, tolerance-controlled coupling iterations. Statistics over $40\le t\le55$ s. Status: **ok** = reached its end time; **running**; **queued** = in the driver's order; **postponed** = runs only if time remains before 8 November; **not planned** = not part of this campaign. Groups: D FVM domain, I interface, T coupling interval, V particle side, F FVM side, R FVM-only reference.

Queue order: `k`, `n_interval`, `inset`, `a`, `p`, `c`, `g` (CPU, in parallel from the start), `v_side`, `f_side`, then `b1`, `i_transfer`.

| Section | Group | Run directory under `sensitivity/` | Varied quantity | Status |
| --- | --- | --- | --- | --- |
| anchor | – | `box2.4_sub5_inset8.75` | baseline, to $t=100$ s (sampling floor) | ok |
| `k` | I | `box2.4_sub5_inset8.75_k1` | coupling iterations $M=1$ | running |
| `k` | I | `box2.4_sub5_inset8.75_k2` | $M=2$ | running |
| `k` | I | `box2.4_sub5_inset8.75_k4` | $M=4$ | running |
| `n_interval` | T | `box2.4_sub1_inset8.75` | $N=1$ ($\Delta t_\mathrm{VPM}=0.008$ s) | queued (resumes from 36.0 s) |
| `n_interval` | T | `box2.4_sub2_inset8.75` | $N=2$ ($\Delta t_\mathrm{VPM}=0.016$ s) | ok |
| `n_interval` | T | `box2.4_sub3_inset8.75` | $N=3$ ($\Delta t_\mathrm{VPM}=0.024$ s) | queued (resumes from 54.8 s) |
| `n_interval` | T | `box2.4_sub6_inset8.75` | $N=6$ ($\Delta t_\mathrm{VPM}=0.048$ s) | queued |
| `n_interval` | T | `box2.4_sub1_inset8.75_kappa3` | $N=1$, $\kappa_\mathrm{max}=3$ | queued |
| `n_interval` | T | `box2.4_sub3_inset8.75_kappa3` | $N=3$, $\kappa_\mathrm{max}=3$ | queued |
| `n_interval` | T | `box2.4_sub4_inset8.75_kappa3` | $N=4$, $\kappa_\mathrm{max}=3$ | queued |
| `n_interval` | T | `box2.4_sub6_inset8.75_kappa3` | $N=6$, $\kappa_\mathrm{max}=3$ | queued |
| `box_min` | D | `box1.6_sub5_inset8.75` | $y_\mathrm{box}=1.6$ m | ok |
| `box_min` | D | `box2.0_sub5_inset8.75` | $y_\mathrm{box}=2.0$ m | ok |
| `box_conv` | D | `box3.2_sub5_inset8.75` | $y_\mathrm{box}=3.2$ m | ok |
| `box_conv` | D | `box4.0_sub5_inset8.75` | $y_\mathrm{box}=4.0$ m | ok |
| `inset` | I | `box2.4_sub5_inset4` | $L_\mathrm{inset}=4h$ | queued |
| `inset` | I | `box2.4_sub5_inset14` | $L_\mathrm{inset}=14h$ | queued |
| `a` | F | `box2.4_sub8_inset8.75_h0.08_dtf0.004_ic0.04` | FVM cells $h_\mathrm{FVM}=0.08$ m, particles 0.04 m, $\Delta t_\mathrm{FVM}=0.004$ s, $N=8$ | queued |
| `a` | F | `box2.4_sub8_inset8.75_h0.056569_dtf0.004_ic0.04` | $h_\mathrm{FVM}=0.0566$ m | queued |
| `a` | F | `box2.4_sub8_inset8.75_dtf0.004` | $h_\mathrm{FVM}=0.04$ m | queued |
| `a` | F | `box2.4_sub8_inset8.75_h0.028284_dtf0.004_ic0.04` | $h_\mathrm{FVM}=0.0283$ m (`RUN_FINE=1`, about 5.8 GB of GPU memory) | queued (once the cube leaves the GPU) |
| `p` | F | `box2.4_sub10_inset8.75_dtf0.004` | $\Delta t_\mathrm{FVM}=0.004$ s at $\Delta t_\mathrm{VPM}=0.04$ s ($N=10$) | queued |
| `c` | F | `box2.4_sub20_inset8.75_dtf0.002` | $\Delta t_\mathrm{FVM}=0.002$ s at $\Delta t_\mathrm{VPM}=0.04$ s ($N=20$) | queued |
| `c` | F | `box2.4_sub40_inset8.75_dtf0.001`, `reference_dt0.001` | $\Delta t_\mathrm{FVM}=0.001$ s ($N=40$) | not planned (8 times the FVM steps) |
| `v_side` | V | `box2.4_sub5_inset8.75_hp0.08` | particle spacing 0.08 m | queued |
| `v_side` | V | `box2.4_sub5_inset8.75_rk4` | RK4 particle integrator | queued |
| `v_side` | V | `box2.4_sub5_inset8.75_vf0.001` | vorticity floor 0.001 | queued |
| `v_side` | V | `box2.4_sub5_inset8.75_vf0.1` | vorticity floor 0.1 | queued |
| `f_side` | F | `box2.4_sub5_inset8.75_oc1` | 1 PIMPLE outer corrector | queued |
| `f_side` | F | `box2.4_sub5_inset8.75_oc3` | 3 PIMPLE outer correctors | queued |
| `f_side` | F | `box2.4_sub2_inset8.75_dtf0.02_co0.5` | Courant controller 0.5, $\Delta t_\mathrm{FVM}\le0.02$ s | queued |
| `f_side` | F | `box2.4_sub2_inset8.75_dtf0.02_co0.9` | Courant controller 0.9 | queued |
| `b1` | I | `box2.4_sub{5,2}_inset8.75_kappa{1,2,3,10}` (8 runs) | $\kappa_\mathrm{max}=1,2,3,10$ at $N=5$ and 2 | postponed |
| `i_transfer` | I | `box2.4_sub5_inset8.75_{tau0.01,tau0.2,mc0.02,mc0.3}`, `box2.4_sub2_inset8.75_tau0.2` | $\tau_\mathrm{t}=0.01, 0.2$; moment cap 0.02, 0.3 | postponed (two $\tau_\mathrm{t}=0.01$ runs at $y_\mathrm{box}=3.2$, 4.0 m if the wide-box lift deficit survives `g`) |
| `g` | R | `reference_fw4.4` | $h$ cells over the $y_\mathrm{box}=2.4$ m domain | ok |
| `g` | R | `reference_dt0.004_fw4.4` | same, $\Delta t_\mathrm{FVM}=0.004$ s | ok |
| `g` | R | `reference_fw5.2_fb3.2` | $h$ cells over the $y_\mathrm{box}=3.2$ m domain | ok |
| `g` | R | `reference_fw4.0_fb2.0` | $h$ cells over the $y_\mathrm{box}=2.0$ m domain | running |
| `g` | R | `reference_fw3.6_fb1.6` | $h$ cells over the $y_\mathrm{box}=1.6$ m domain | running |
| `g` | R | `reference_fw6.0_fb4.0` | $h$ cells over the $y_\mathrm{box}=4.0$ m domain | running |
| $Q_\mathrm{ext}$ | R | `reference_h0.08`, `reference_h0.056569`, `reference_flow`, `reference_h0.028284` | $h=0.08$ to 0.0283 m at $\Delta t_\mathrm{FVM}=0.008$ s | ok |
| $Q_\mathrm{ext}$ | R | `reference_h{0.08,0.056569,0.028284}_dt0.004`, `reference_dt0.004` | $h=0.08$ to 0.0283 m at $\Delta t_\mathrm{FVM}=0.004$ s | ok |
| $Q_\mathrm{ext}$ | R | `reference_dt0.016`, `reference_flow`, `reference_dt0.004`, `reference_dt0.002` | $\Delta t_\mathrm{FVM}=0.016$ to 0.002 s at $h=0.04$ m | ok |
| `r` | R | `reference_re100`, `reference_re250` | $Re=100$, 250 | ok (coupled $Re$ runs not planned) |
| – | – | sections `x`, `c_ref`, `t2`, `e`, `e_old`, `d`, `b`, `f_fact`, `f` | earlier-solver series, blending depths, particle spacing $h/2$ (12 GB) | not planned |

`./allplot.sh` writes the study figures and the tables in `figures/auxiliary/`. [coupling_findings.md](assets/coupling_findings.md) records the physical conclusions and the quantities each section measures.
