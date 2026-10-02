# Circular-cylinder flow with immersed forcing

This 2D laminar case enforces no-slip through immersed-boundary markers on a Cartesian mesh. See [the FVM model](../../../docs/fvm.md#physical-model-and-units), [mesh setup](../../../docs/fvm.md#mesh-setup), and [boundary conditions](../../../docs/fvm.md#boundary-conditions).

From the repository root:

```bash
cd tutorials/fvm/cylinder_ibm
./allrun.sh
./allplot.sh
```

`./allrun.sh` clears previous results; `./allcontinue.sh` resumes them. Parameters are at the top of [setup.py](setup.py).

## Physical setup

| Parameter | Default and meaning |
| --- | --- |
| `DIAMETER` | $D=1$ m, cylinder centred at $(0,0)$ |
| `FREESTREAM_VELOCITY` | $U_\infty=1$ m/s |
| `DENSITY` | $\rho=1$ kg/m³ |
| `REYNOLDS_NUMBER` | $Re_D=30$; $\nu=U_\infty D/Re_D$ |
| `SPACING` | Core grid spacing $h=D/16=0.0625$ m |
| `MARKER_ALPHA` | Marker spacing divided by $h$, default 1 |
| `FINAL_TIME` | $60$ s |

The mesh spans $[-8,16]\times[-8,8]$ m, with uniform core spacing and stretched far-field cells. It contains no cylinder wall; a ring of stationary markers supplies forcing. The inlet is fixed velocity, the outlet fixes $p/\rho=0$, the upper/lower boundaries are freestream, and the single-cell span is `empty`.

The run caps both the Courant number and the forcing Fourier number:

$$
Fo=\frac{\nu\Delta t}{h^2}\le0.1.
$$

With the default parameters, the initial step is $0.01$ s and the maximum allowed by this cap is $0.01171875$ s. Refining $h$ requires a smaller step.

## Inspect the wake and forces

`figures/forces_cylinder.png` shows drag, lift and marker slip; `figures/wake_centreline.png` shows wake recovery. At $Re_D=30$, look for a steady symmetric wake and small marker velocity relative to $U_\infty$. Compare drag and recirculation length under mesh refinement; the diffuse forcing region makes these sensitive to $h$.

For an unsteady variant, change `REYNOLDS_NUMBER` to 100, clear results with `./allrun.sh`, and inspect alternating lift and shedding frequency. Force coefficients use $A_\mathrm{ref}=Db$, where $b=h$ is the extrusion thickness.
