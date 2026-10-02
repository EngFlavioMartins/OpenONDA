# Cylinder shedding at $Re=150$

A no-slip cylinder sheds vorticity into a laminar wake. FVM resolves the wall and near wake; VPM transports the outer wake. Two free-slip spanwise planes model a cylinder section with resolved span $b=0.96$ m. Force coefficients use frontal area $Db=0.96$ m².

| Quantity | Default |
| --- | --- |
| Diameter $D$; speed $U_\infty$ | 1 m; 1 m/s in $+x$ |
| Density $\rho$; viscosity $\nu=U_\infty D/Re$ | 1 kg/m³; $1/150$ m²/s |
| FVM box | $[-1.6,1.6]^2\times[-0.48,0.48]$ m |
| Transfer region | $[-1.25,1.25]^2\times[-0.48,0.48]$ m |
| VPM domain | $[-5,15]\times[-5,5]\times[-0.48,0.48]$ m |
| FVM/particle spacing | 0.04 m; 24 spanwise layers |
| FVM/exchange step | 0.008 s / 0.04 s |
| End time | 100 s |

## Models and mesh

The [body-fitted mesh](../../../docs/fvm.md#mesh-setup) is generated from `assets/cylinder_long.stl`. The cylinder is no-slip; `zmin` and `zmax` are slip; the remaining box faces form `numericalBoundary`. VPM uses RK2, Gaussian particles with core radius equal to spacing, and GBD diffusion. No SGS closure is applied.

[Slab induction](../../../docs/coupling.md#free-slip-span) enforces the same physical slip planes in VPM. [Mixed vorticity boundaries](../../../docs/coupling.md#boundary-conditions) and [buffered M4-prime renewal](../../../docs/coupling.md#vorticity-transfer) use a $6h$ blend width, a $2h$ VPM-only band and up to three interface sweeps. Edit the physical, mesh and coupling constants in `setup.py`; both solvers start with the same small velocity perturbation as the reference.

## Run and compare

From this directory in an [installed environment](../../../docs/installation.md):

```bash
(cd reference_flow && ./allrun.sh)
./allrun.sh
./allplot.sh
```

The [standalone FVM reference](reference_flow/README.md) is needed for force/profile comparison, but the coupled solver runs independently. Both launchers preserve outputs and resume compatible backups; `./allcontinue.sh` also resumes. `./allcontinue.sh --max-coupling-steps 25` stops after 25 accepted exchanges and saves a checkpoint. `./allclean.sh` deletes generated results.

Forces and wake probes are sampled every 0.04 s, profiles every 0.2 s, and slices every 0.4 s. Coupled backups and matching FVM/VPM volumes are saved every 1 s, at initialization, and at a requested stop. Read `samples/forces_history.csv`, the plots under `figures/`, and `solution/coupler_diagnostics.jsonl`. Open `solution/fvm.pvd` and `solution/vpm.pvd` in ParaView to compare both solvers at the same saved times. Short transients are insufficient for shedding frequency or phase agreement. Refine the mesh, particle spacing and exchange step before drawing accuracy conclusions.
