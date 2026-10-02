# NACA 4412 at $10^\circ$ and $Re=1000$

This finite-span airfoil uses an immersed-boundary FVM near field and a VPM wake. The NACA 4412 section has 4% maximum camber at 40% chord and 12% thickness. It is extruded with end caps over a 5 m span; the freestream direction sets the $10^\circ$ incidence.

| Quantity | Default |
| --- | --- |
| Chord $c$; span $b$ | 1 m; 5 m |
| Freestream $\mathbf{U}_\infty$ | $(\cos10^\circ,\sin10^\circ,0)$ m/s |
| Density $\rho$; viscosity $\nu$ | 1 kg/m³; 0.001 m²/s |
| FVM box | $[-1.2,1.4]\times[-0.8,0.8]\times[-3.3,3.3]$ m |
| Transfer region | $[-1.12,1.32]\times[-0.72,0.72]\times[-3.2,3.2]$ m |
| Blend / VPM-owned band | 0.24 m / 0.08 m |
| VPM domain | $[-2.5,10]\times[-2,2]\times[-4,4]$ m |
| FVM/particle spacing | 0.04 m |
| Immersed-marker spacing | $2.5h=0.10$ m |
| FVM/exchange step | 0.01 s / 0.04 s |
| End time | 12 s, or $12c/U_\infty$ |

## Models and mesh

`setup.py` generates the section analytically with `naca4_vertices` and creates a capped `ImmersedBody.extruded_polygon_z` on a [Cartesian mesh](../../../docs/fvm.md#mesh-setup). Direct forcing imposes no-slip on the airfoil; all FVM box faces form `numericalBoundary`. The default marker separation avoids an ill-conditioned quadrature near the thin section and end caps.

FVM uses [Smagorinsky LES](../../../docs/fvm.md#turbulence-and-les) with $C_s=0.17$. VPM uses the same coefficient, RK2, [GBD diffusion and LES](../../../docs/vpm.md#diffusion-and-les), Gaussian particles and free-space treecode induction. The [coupler](../../../docs/coupling.md#vorticity-transfer) uses buffered M4-prime renewal and supplies [normal velocity and tangential normal derivative](../../../docs/coupling.md#boundary-conditions). Edit the geometry, physical constants and mesh spacing in `setup.py` for other cases.

GBD replaces the former core-spreading discretization of the same viscous term. Start a fresh run when moving from that configuration; its backups use a different numerical model.

## Run and inspect

From this directory in an [installed environment](../../../docs/installation.md):

```bash
python setup.py
./allplot.sh
```

`python setup.py` and `./allcontinue.sh` preserve outputs and resume compatible backups. **`./allrun.sh` cleans generated results before running a fresh case.**

IBM forces are sampled every FVM step (0.01 s); profiles, fields and coupled backups every 0.8 s. Read `samples/ibm_forces_history.csv` and the wind-axis lift/drag plots under `figures/`. Coefficients use area $cb=5$ m². ParaView opens `solution/fvm.pvd` and `solution/vpm.pvd`. Check force and no-slip-error histories, then refine grid and marker spacing before interpreting aerodynamic coefficients.
