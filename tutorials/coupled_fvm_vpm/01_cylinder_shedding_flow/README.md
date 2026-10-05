# Cylinder shedding at $Re=150$

A no-slip cylinder sheds vorticity into a laminar wake. FVM resolves the wall and near wake; VPM transports the outer wake. Two free-slip spanwise planes model a cylinder section with resolved span $b=0.96$ m. Force coefficients use frontal area $Db=0.96$ m².

| Quantity | Default |
| --- | --- |
| Diameter $D$; speed $U_\infty$ | 1 m; 1 m/s in $+x$ |
| Density $\rho$; viscosity $\nu=U_\infty D/Re$ | 1 kg/m³; $1/150$ m²/s |
| FVM box | $[-1.6,2.4]\times[-1.6,1.6]\times[-0.48,0.48]$ m |
| Transfer region | $[-1.25,2.05]\times[-1.25,1.25]\times[-0.48,0.48]$ m |
| VPM domain | $[-5,15]\times[-5,5]\times[-0.48,0.48]$ m |
| FVM/particle spacing | 0.04 m; 24 spanwise layers |
| FVM/exchange step | 0.008 s / 0.04 s |
| End time | 100 s |
| Startup freestream | $(1,0.1,0)$ m/s through 1 s; smooth taper to $(1,0,0)$ by 2 s |

## Models and mesh

The [body-fitted mesh](../../../docs/fvm.md#mesh-setup) is generated from `assets/cylinder_long.stl`, whose $1.2$ m length keeps both caps outside the resolved span. The cylinder is no-slip; `zmin` and `zmax` are slip; the remaining box faces form `numericalBoundary`. VPM uses RK2, Gaussian particles with core radius equal to spacing, and GBD diffusion. No SGS closure is applied.

[Slab induction](../../../docs/coupling.md#free-slip-span) enforces the same physical slip planes in VPM. [Mixed vorticity boundaries](../../../docs/coupling.md#boundary-conditions) and [buffered M4-prime renewal](../../../docs/coupling.md#vorticity-transfer) use a $6h$ blend width, a $2h$ VPM-only band and up to three Picard sweeps at the existing $10^{-5}$ tolerances. Converged exchanges enable the safeguarded predictor; successful predictions can finish in one sweep, while a rejected prediction adds one trial sweep before Picard iteration. All case inputs belong to this tutorial's `setup.py`; its `assets/` contain local construction and execution helpers.

The downstream extension places full FVM authority through $x/D=1.81$, beyond the saved reference's mean recirculation closure near $1.55$. The prepared mesh has 185,784 cells, up from 143,040 (29.9%). At the default $h=0.04$, the requested box is exact. Other spacings can expand the box to Cartesian cell planes; for example, $h=0.064$ resolves the downstream edge at $2.432$. Check the recorded mesh bounds before interpreting a grid study.

The small initial 3D curl disturbance is retained. The transverse freestream remains at $0.1$ m/s through 1 s, then follows a quintic smoothstep to zero at 2 s. Its first and second time derivatives vanish at both taper endpoints. Both solvers use the accepted endpoint velocity, with continuous coupling boundary history across the taper. Nominal $U_\infty=1$ m/s continues to define viscosity and force normalization. The earlier abrupt policy produced an artificial pressure/lift impulse ($C_L=-7.58$ at 2.04 s in the original startup check); the smooth policy avoids that instantaneous velocity jump. Exclude startup from developed shedding statistics.

The schedule is recorded in `solution/cylinder_startup.json`; continuation requires the exact current policy and restores it without repeating the trigger. Keep the sidecar beside a copied native backup. The trigger changes onset and phase, so compare developed amplitude and frequency against a reference using the same schedule.

The reference case uses the same startup velocity, taper interval and initial disturbance.

## Run and compare

From this directory in an [installed environment](../../../docs/installation.md):

```bash
(cd reference_flow && ./allrun.sh --fresh)
./allrun.sh --fresh
./allplot.sh
```

The [standalone FVM reference](reference_flow/README.md) is needed for force/profile comparison, but the coupled solver runs independently. Existing reference results can be plotted, but an older reference without the startup trigger does not provide a matching transient comparison. Fresh runs of both cases use the same smooth startup. `./allrun.sh --fresh` archives previous coupled outputs, including the native mesh, under `previous_runs/` and starts at zero. It leaves `reference_flow/`, `drag_recovery/`, assets and study results untouched. Stop an active run before requesting a fresh one. The previous compact-box checkpoints are incompatible with this revised setup.

Without `--fresh`, `./allrun.sh` preserves outputs and resumes a compatible backup; `./allcontinue.sh` does the same. `./allcontinue.sh --max-coupling-steps 25` stops after 25 accepted exchanges in total, including across the trigger switch, and saves a checkpoint. `./allclean.sh` deletes generated results.

The launcher runs the local setup through the generic case runner, which holds a case lock throughout MPI execution. Runtime dependencies are supplied by the [installation](../../../docs/installation.md); the launcher contains no device or toolkit configuration.

The numerical corrections have regression coverage. A four-rank run on the prepared mesh passed native checkpoint restarts before, at, and after the original abrupt 2 s switch, reaching 2.12 s with 53 consecutive force samples. Its checkpoint and data audit is saved in `drag_recovery/production_startup_smoke/validation.json`; it predates the smooth taper. The first startup exchange exhausted its four-sweep allowance; all subsequent exchanges met the interface tolerances. Recovery of the developed reference drag oscillation (about $\Delta C_D=0.0542$ over 80–100 s) still requires the completed clean run; startup force peaks are not evidence of a matching limit cycle.

The smooth taper was compared against the abrupt policy on the same 30,964-cell planar cylinder mesh with four FVM layers and two CPU MPI ranks. Over 1–2.24 s, the largest $|C_L|$ fell from 7.669 to 0.500, a 93.5% reduction. At 2.04 s, $C_L$ was 0.113 instead of the abrupt impulse of −7.669. Native restarts before, during and after the taper reproduced the next lift sample within $4.6\times10^{-8}$. The common hold cost was 14.24 s per exchange for both policies. Full production 3D startup and developed shedding remain separate checks. Results and the comparison figure are in `drag_recovery/smooth_startup_validation/validation.json` and `comparison.png`.

The production run's separate GBD failure after 25 s was reproduced from its valid native checkpoint. A weak disconnected outer-wake component retained only 120 of 3,024 active diffusion nodes, requiring an unsafe moment correction despite adequate matrix rank. GBD now retains that component's original donors when they fit the declared capacity, preserving all existing moment and strength-correction limits. The reproduced repair added 2,904 nodes (0.39%) and passed a four-rank GPU exchange with a complete native checkpoint audit. Evidence is in `drag_recovery/gbd_crash_recovery/qualification.json`; production continues from the saved state with its original startup policy.

The investigation identified two numerical defects. Renewal blended physical FVM vorticity directly with Gaussian particle coefficients, changing even a perfectly matching represented field. It now corrects the physical mismatch and excludes solid-interior targets before the inverse correction. Repeated boundary updates also erased the native `fixedFluxPressure` gradient history before the momentum predictor; this history is now retained. In the controlled rotating-flow test, retaining it reduced pressure RMS error from $4.67\times10^{-4}$ to $1.36\times10^{-5}$ with the same pressure-correction budget. These tests establish the defects independently of the pending cylinder amplitude comparison.

Geometry remains a plausible contributor: the old downstream boundary at $x/D=1.6$ intersects instantaneous reference backflow, while its full FVM-authority region ended at $1.01$. Extending the wake box reduces that exposure. The pressure linear solve itself converged in the old run; a small algebraic residual did not rule out the boundary-history error. Saved diagnostic evidence is under `drag_recovery/baseline/`.

`./allplot.sh [both|png|pdf]` writes `velocity_profiles_t*.png` or `.pdf` for coincident saved states. Each figure compares reference flow, coupled FVM and coupled VPM, with the FVM domain and transfer region shaded. At $x/D=4$, outside the FVM box, only reference and VPM curves exist. The time appears in the filename; profiles are compared without time interpolation. Missing coupled FVM transverse probes are reconstructed from native saved fields using the same affine, 12-centre method as the FVM line sampler. MPI ownership and fluid-domain coverage are checked; derived samples and source provenance are cached under `figures/auxiliary/velocity_profile_cache/`. Details and errors are in `figures/auxiliary/velocity_profile_errors.json`.

`figures/coupling_diagnostics.*` has two panels: full recorded wall cost per FVM step on a logarithmic scale (exchange costs divided by the recorded FVM substeps), and total particle population. The four exclusive cost categories are VPM, FVM, coupling (boundary traces and transfer), and sampling and output (health checks, samplers, backups, reporting and orchestration/waiting). They sum to the recorded total. Renewal conservation checks remain in `solution/coupler_diagnostics.jsonl`; they are no longer a figure panel.

`figures/reference_forces.*` compares the available force histories without shifting their clocks or hiding samples. `figures/auxiliary/reference_force_errors.json` records the common interval and quantitative errors.

Forces and wake probes are sampled every 0.04 s, profiles every 0.2 s, and slices every 0.4 s. Coupled backups and matching FVM/VPM volumes are saved every 1 s, at initialization, and at a requested stop. Read `samples/forces_history.csv`, the plots under `figures/`, and `solution/coupler_diagnostics.jsonl`. Open `solution/fvm.pvd` and `solution/vpm.pvd` in ParaView to compare both solvers at the same saved times. Short transients are insufficient for shedding frequency or phase agreement. Refine the mesh, particle spacing and exchange step before drawing accuracy conclusions.
