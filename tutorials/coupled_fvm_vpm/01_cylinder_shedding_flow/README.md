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

| Section | Varied quantity | Status |
| --- | --- | --- |
| `baseline` | Baseline and a repeat | run |
| `k` | Coupling iterations per coupling interval | run |
| `c_ref` | Cells, steps and particles together at constant Courant number ($N=2$), with the FVM-only levels that give $Q_\mathrm{ext}$ | run |
| `x` | Cells at $\Delta t_\mathrm{FVM}=0.004$ s, $N=4$ and $N=10$, interface fixed in 0.04 m cells | run |
| `t2`, `p`, `e_old` | Both steps at $N=2$; FVM step at a fixed coupling interval; FVM substeps | run |
| `b`, `f_fact` | Lateral clearance and wake reach at $Re=150$ and 250; 2×2 factorial | run |
| `r`, `inset` | Reynolds number; transfer inset | run |
| `b1` | Strength-ratio limit $\kappa_\mathrm{max}=1$ and 3 at $N=2$ and 5 | to run |
| `d` | $N=3$ and 6 | to run |
| `a` | Four cell sizes at $\Delta t_\mathrm{VPM}=0.032$ s (spatial order) | to run |
| `c` | $N=20$, 40 at $\Delta t_\mathrm{VPM}=0.04$ s (temporal order) | to run |
| `g` | FVM-only reference with $h$ cells over the coupled FVM domain | to run |
| `e` | Blending depths $\delta_0$, $\delta_1$ | to run |
| `h` | Baseline and $N=2$ continued to 100 s | to run |
| `f` | Particle spacing $h/2$ (needs about 12 GB of GPU memory) | to run |

`./allplot.sh` writes the study figures and the tables in `figures/auxiliary/`. [coupling_findings.md](assets/coupling_findings.md) records the physical conclusions and the quantities each section measures.
