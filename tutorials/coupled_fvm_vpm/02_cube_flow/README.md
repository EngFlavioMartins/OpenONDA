# Coupled cube flow at $Re=1000$

FVM resolves separation from the no-slip faces of a unit cube; VPM carries the outer wake. Both solvers use equilibrium Smagorinsky LES with $C_k=0.094$ and $C_e=1.048$. Force coefficients use frontal area 1 m².

| Quantity | Default |
| --- | --- |
| Cube | $[-0.5,0.5]^3$ m |
| Freestream $\mathbf{U}_\infty$; density $\rho$ | $(1,0,0)$ m/s; 1 kg/m³ |
| Molecular viscosity $\nu$ | 0.001 m²/s |
| FVM box | $[-1.485,1.485]^3$ m |
| Transfer region | $[-1.45,1.45]^3$ m |
| VPM domain | $[-4.5,12]\times[-3,3]^2$ m |
| FVM/particle spacing | 0.045 m |
| FVM/exchange step | 0.01 s / 0.05 s |
| End time | 30 s |

## Models and mesh

The [body-fitted Cartesian mesh](../../../docs/fvm.md#mesh-setup) is generated from `assets/cube.stl`, with uniform spacing and cache `constant/mesh.npz`. The cube is no-slip; all outer FVM faces belong to `numericalBoundary`.

See [FVM LES](../../../docs/fvm.md#turbulence-and-les) and [VPM diffusion and LES](../../../docs/vpm.md#diffusion-and-les) for the closures. Particles use RK2, Gaussian cores of radius $1.05h$, GBD diffusion and free-space FMM induction. [Mixed vorticity boundaries](../../../docs/coupling.md#boundary-conditions) and [buffered M4-prime renewal](../../../docs/coupling.md#vorticity-transfer) use a $6h$ blend width, a $2h$ VPM-only band and up to three interface sweeps. Edit constants and configurations in `setup.py` to change the problem.

## Run and compare

From this directory in an [installed environment](../../../docs/installation.md):

```bash
./allrun.sh
./allplot.sh
```

`./allrun.sh`, `./allcontinue.sh` and `python setup.py` preserve outputs and resume compatible coupled backups. `./allclean.sh` deletes generated results and the mesh cache. See [continuation](../../../docs/continuation.md) before changing a saved configuration.

Forces and profiles are sampled every 0.05 s; retained fields and coupled backups every 0.25 s. Read histories under `samples/`, plots under `figures/` and convergence in `solution/coupler_diagnostics.jsonl`. ParaView opens `solution/fvm.pvd` and `solution/vpm.pvd`.

Run the [three-grid FVM reference](reference_flow/README.md) for comparison. `./allplot.sh` uses a completed reference and common saved physical times. Compare force statistics and wake profiles while refining FVM spacing, particle spacing and exchange step; matching LES coefficients alone does not establish coupled accuracy.
