# Laminar NACA 0012 flow

This case measures pressure and forces on a symmetric NACA 0012 surface. See [FVM units](../../../docs/fvm.md#physical-model-and-units), [STL mesh setup](../../../docs/fvm.md#mesh-setup), and [external-flow boundaries](../../../docs/fvm.md#boundary-conditions). For a turbulent airfoil example, use [the coupled NACA 4412 case](../../coupled_fvm_vpm/03_naca4412_flow/README.md).

From the repository root:

```bash
cd tutorials/fvm/airfoil_flow
./allrun.sh
./allplot.sh
```

`./allrun.sh` clears previous results; `./allcontinue.sh` resumes them. Edit [setup.py](setup.py) to change:

| Parameter | Default and meaning |
| --- | --- |
| `CHORD` | $c=1$ m |
| `DEPTH` | STL extrusion depth $0.8$ m |
| `FREESTREAM_VELOCITY` | $U_\infty=1$ m/s |
| `DENSITY` | $\rho=1$ kg/m³ |
| `REYNOLDS_NUMBER` | $Re_c=1000$; $\nu=U_\infty c/Re_c=0.001$ m²/s |
| `ANGLE_OF_ATTACK_DEGREES` | Inlet-flow angle, default $0^\circ$ |
| `FINAL_TIME` | $25$ s |

The script generates the STL and meshes the box $[-5,15]\times[-5,5]\times[-0.5,0.5]$ m. Background spacing is $1$ m, the airfoil patch target is $0.03125$ m, and a near-airfoil box uses a $0.125$ m target. The airfoil is no-slip, the outlet fixes $p/\rho=0$, and lateral boundaries are freestream.

The supplied setup assigns `empty` to the outer spanwise faces while meshing a finite-depth body. When adapting it, use a single spanwise cell for a 2D model, or replace `empty` with physical spanwise conditions for a 3D model; the supplied span treatment needs review before using its loads as a physical reference.

## Inspect pressure and forces

`figures/airfoil_surface_cp.png` plots

$$
C_p=\frac{p-p_\infty}{\tfrac12\rho U_\infty^2}
=\frac{q-q_\infty}{\tfrac12U_\infty^2}.
$$

At zero angle, inspect upper/lower pressure symmetry and lift near zero. `figures/airfoil_forces.png` and `figures/airfoil_velocity.png` show force evolution and the flow. Refine the surface and wake and reduce the time step before comparing loads between angles.
