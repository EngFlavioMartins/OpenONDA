# Laminar flat-plate boundary layer

This 2D case resolves boundary-layer growth and compares velocity profiles and wall friction with Blasius theory. See [FVM units](../../../docs/fvm.md#physical-model-and-units), [wall meshes](../../../docs/fvm.md#mesh-setup), and [boundary conditions](../../../docs/fvm.md#boundary-conditions).

From the repository root:

```bash
cd tutorials/fvm/boundary_layer
./allrun.sh
./allplot.sh
```

`./allrun.sh` clears previous results; `./allcontinue.sh` resumes them. Edit [setup.py](setup.py) to change:

| Parameter | Default and meaning |
| --- | --- |
| `PLATE_LENGTH` | $L=1$ m |
| `FREESTREAM_VELOCITY` | $U_\infty=1$ m/s |
| `DENSITY` | $\rho=1$ kg/m³ |
| `REYNOLDS_NUMBER` | $Re_L=10^4$; $\nu=U_\infty L/Re_L=10^{-4}$ m²/s |
| `N_PLATE` | 72 streamwise cells along the plate |
| `DOMAIN_HEIGHT` | $0.35$ m |
| `WALL_CELL_HEIGHT` | First wall-normal cell height $0.0015$ m |
| `WALL_STRETCHING` | Consecutive wall-normal cell-height ratio $1.12$ |
| `FINAL_TIME` | $8$ s |

The inlet is uniform. The lower boundary is slip upstream of $x=0$ and no-slip along the plate, so boundary-layer growth begins at the leading edge. The top is slip, the outlet fixes $p/\rho=0$, and the one-cell span has `empty` boundaries. The initial velocity is $(U_\infty,0,0)$.

## Compare with Blasius theory

At $x/L=0.25,0.5,0.75$, compare profiles using

$$
\eta=y\sqrt{\frac{U_\infty}{\nu x}},
\qquad \frac{u}{U_\infty}=f'(\eta),
\qquad
C_f(x)=\frac{\tau_w}{\tfrac12\rho U_\infty^2}
=\frac{0.664}{\sqrt{Re_x}},
\qquad Re_x=\frac{U_\infty x}{\nu}.
$$

`figures/blasius_profiles.png` and `figures/skin_friction.png` show the comparisons. Reduce the first-cell height, increase `N_PLATE`, and reduce `TIME_STEP_SIZE` to distinguish wall-resolution error from time error. If the layer approaches the top boundary, increase `DOMAIN_HEIGHT`.
