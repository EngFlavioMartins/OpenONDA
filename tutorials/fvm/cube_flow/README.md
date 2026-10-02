# Square-cylinder vortex shedding

This body-fitted 2D case models laminar flow around a square cylinder. See [FVM equations and units](../../../docs/fvm.md#physical-model-and-units), [mesh setup](../../../docs/fvm.md#mesh-setup), and [boundary conditions](../../../docs/fvm.md#boundary-conditions). For a 3D LES case, use [coupled cube flow](../../coupled_fvm_vpm/02_cube_flow/README.md).

From the repository root:

```bash
cd tutorials/fvm/cube_flow
./allrun.sh
./allplot.sh
```

`./allrun.sh` clears previous results; `./allcontinue.sh` resumes them. Edit [setup.py](setup.py) to change:

| Parameter | Default and meaning |
| --- | --- |
| `SIDE` | Square side $D=1$ m |
| `FREESTREAM_VELOCITY` | $U_\infty=1$ m/s |
| `DENSITY` | $\rho=1$ kg/m³ |
| `REYNOLDS_NUMBER` | $Re_D=100$; $\nu=U_\infty D/Re_D=0.01$ m²/s |
| `SPACING` | Near-body/wake spacing $0.0625$ m |
| `FINAL_TIME` | $120$ s |

The cylinder is centred at the origin in a domain $-10\le x/D\le25$, $-10\le y/D\le10$. The core is uniform and the far field is stretched. The inlet is fixed velocity, the outlet fixes $p/\rho=0$, the lateral boundaries are slip, and the cylinder is no-slip. The single spanwise cell has `empty` faces.

A transverse initial velocity $0.05U_\infty$ seeds asymmetric shedding. The initial time step is $0.02$ s and the adaptive maximum is $0.05$ s.

## Inspect shedding

`figures/forces_cube.png` shows force histories, and the vorticity/wake figures show alternating vortices. Calculate $St=fD/U_\infty$ from the settled lift oscillations; coefficients use reference area $Db$, with extrusion thickness $b=$ `SPACING`.

Discard the startup transient when averaging drag or measuring shedding frequency. Reduce `SPACING` and the time-step limits together to check convergence; move the lateral boundaries farther out to assess the default 5% blockage.
