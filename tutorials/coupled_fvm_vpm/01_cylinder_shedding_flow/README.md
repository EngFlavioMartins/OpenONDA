# Cylinder shedding at $Re=150$

A no-slip cylinder sheds vorticity into a laminar wake. FVM resolves the wall and near wake; VPM transports the outer wake. The FVM mesh has one cell across a unit span, $z∈[-0.5,0.5]$ m, with periodic `zmin`/`zmax` patches. VPM carries one particle layer at $z=0$ using infinite-span Gaussian filaments. Force coefficients use frontal area $Db=1$ m².

| Quantity | Default |
| --- | --- |
| Diameter $D$; speed $U_\infty$ | 1 m; 1 m/s in $+x$ |
| Density $\rho$; viscosity $\nu=U_\infty D/Re$ | 1 kg/m³; $1/150$ m²/s |
| FVM box | $[-1.6,2.4]\times[-1.6,1.6]\times[-0.5,0.5]$ m |
| Transfer region | $[-1.25,2.05]\times[-1.25,1.25]\times[-0.5,0.5]$ m |
| VPM domain | $[-5,15]\times[-5,5]\times[-0.5,0.5]$ m |
| In-plane FVM/particle spacing | 0.04 m |
| Spanwise resolution | One FVM cell of width 1 m; one VPM layer at $z=0$ |
| FVM/exchange step | 0.008 s / 0.04 s |
| End time | 100 s |
| Startup freestream | $(1,0.1,0)$ m/s through 1 s; smooth taper to $(1,0,0)$ by 2 s |

## Models and mesh

The [body-fitted mesh](../../../docs/fvm.md#mesh-setup) is generated from `assets/cylinder_long.stl`, whose $1.2$ m length keeps both caps outside the resolved span. The cylinder is no-slip; `zmin` and `zmax` form a reciprocal periodic pair; the remaining box faces form `numericalBoundary`. VPM uses RK2, infinite-span Gaussian filaments with core radius equal to the in-plane spacing, and planar GBD diffusion. Stored particle volume is $h^2b$ and axial vortex strength is circulation times $b$. No spanwise velocity, vortex stretching or SGS closure is included.

`PlanarInduction` supplies the span-invariant velocity and its gradient at the FVM boundary, without a truncated set of periodic images. [Mixed vorticity boundaries](../../../docs/coupling.md#boundary-conditions) and [buffered M4-prime renewal](../../../docs/coupling.md#vorticity-transfer) use a $6h$ blend width, a $2h$ VPM-only band and up to six Picard sweeps at the existing $10^{-5}$ tolerances. Iteration stops early when both residuals meet tolerance; an exchange that reaches the cap without passing is flagged unconverged. Converged exchanges enable the safeguarded predictor; successful predictions can finish in one sweep, while a rejected prediction adds one trial sweep before Picard iteration. Physical inputs belong to `setup.py`; `assets/` contains the initial disturbance and scientific plots.

The downstream extension places unit FVM blending weight through $x/D=1.81$, beyond the saved reference's mean recirculation closure near $1.55$. At the default $h=0.04$, the requested box is exact. Other spacings can expand the box to Cartesian cell planes; for example, $h=0.064$ resolves the downstream edge at $2.432$. Check the recorded mesh bounds before interpreting a grid study.

A small divergence-free initial disturbance lies entirely in the $xy$ plane and is identical in both cases. The transverse freestream remains at $0.1$ m/s through 1 s, then follows a quintic smoothstep to zero at 2 s. Its first and second time derivatives vanish at both taper endpoints. Both solvers use the accepted endpoint velocity, with continuous coupling boundary history across the taper. Nominal $U_\infty=1$ m/s continues to define viscosity and force normalization. Exclude startup from developed shedding statistics.

The velocity ramp is declared as a physical input and saved in the native checkpoint configuration. Continuation restores the current velocity without repeating the initial disturbance. Compare developed amplitude and frequency against a reference using the same startup schedule.

The reference case uses the same single periodic FVM layer, unit-span force normalization, startup velocity, taper interval and initial disturbance. This is a strictly two-dimensional model; it does not resolve three-dimensional wake instabilities.

One layer replaces the previous 24 FVM layers at the same in-plane mesh resolution. VPM also retains one particle layer. This reduces cell and particle counts; the wall-time saving still depends on induction, pressure solving and output costs. The planar induction backend evaluates source/target pairs directly, so a 24-fold runtime gain is not assumed.

## Run and compare

From this directory in an [installed environment](../../../docs/installation.md):

```bash
(cd reference_flow && ./allrun.sh --fresh)
./allrun.sh --fresh
./allplot.sh
```

The [standalone FVM reference](reference_flow/README.md) is needed for force/profile comparison, but the coupled solver runs independently. Both cases use the same smooth startup. `./allrun.sh --fresh` archives previous coupled outputs, including the native mesh, under `previous_runs/` and starts at zero. It leaves `reference_flow/`, `drag_recovery/`, assets and study results untouched. Stop an active run before requesting a fresh one. The earlier multi-layer/slip checkpoint is a different model and cannot be continued with this setup; use `--fresh` to preserve it and start the planar case.

Without `--fresh`, `./allrun.sh` preserves outputs and resumes a compatible backup; `./allcontinue.sh` does the same. `./allcontinue.sh --max-coupling-steps 25` performs up to 25 further accepted exchanges and saves a checkpoint. `./allclean.sh` deletes generated results.

The launcher runs the local setup through the generic case runner, which holds a case lock throughout MPI execution. Runtime dependencies are supplied by the [installation](../../../docs/installation.md); the launcher contains no device or toolkit configuration.

`./allplot.sh [both|png|pdf]` writes `velocity_profiles_t*.png` or `.pdf` for coincident saved states. Each figure compares reference flow, coupled FVM and coupled VPM, with the FVM domain and transfer region shaded. At $x/D=4$, outside the FVM box, only reference and VPM curves exist. The time appears in the filename; profiles are compared without time interpolation. Missing coupled FVM transverse probes are reconstructed in memory from native saved fields using the same affine, 12-centre method as the FVM line sampler. The shared reader resolves MPI cell partitions and fluid-domain coverage. Saved times, physical intervals, and errors are recorded in `figures/auxiliary/velocity_profile_errors.json`.

`figures/coupling_diagnostics.*` has two panels: full recorded wall cost per FVM step on a logarithmic scale (exchange costs divided by the recorded FVM substeps), and total particle population. The four exclusive cost categories are VPM, FVM, coupling (boundary traces and transfer), and sampling and output (particle state checks, samplers, backups, reporting and coupling control and waiting). They sum to the recorded total. Renewal conservation checks remain in `solution/coupler_diagnostics.jsonl`; they are no longer a figure panel.

`figures/reference_forces.*` compares the available force histories without shifting their clocks or hiding samples. `figures/auxiliary/reference_force_errors.json` records the common interval and quantitative errors.

Forces and wake probes are sampled every 0.04 s, profiles every 0.2 s, and slices every 0.4 s. Coupled backups and matching FVM/VPM volumes are saved every 1 s, at initialization, and at a requested stop. Read `samples/forces_history.csv`, the plots under `figures/`, and `solution/coupler_diagnostics.jsonl`. Open `solution/fvm.pvd` and `solution/vpm.pvd` in ParaView to compare both solvers at the same saved times. Short transients are insufficient for shedding frequency or phase agreement. Refine the mesh, particle spacing and exchange step before drawing accuracy conclusions.
