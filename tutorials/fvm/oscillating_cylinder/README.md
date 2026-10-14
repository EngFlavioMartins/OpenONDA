# Oscillating cylinder

A circular cylinder heaves across a uniform stream at $Re_D=100$. The whole body-fitted mesh moves with the cylinder: a prescribed rigid motion, solved in arbitrary Lagrangian–Eulerian form. See [moving meshes](../../../docs/fvm.md#moving-meshes), [boundary conditions](../../../docs/fvm.md#boundary-conditions) and [FVM equations and units](../../../docs/fvm.md#physical-model-and-units).

From the repository root:

```bash
cd tutorials/fvm/oscillating_cylinder
./allrun.sh
./allplot.sh
```

`./allrun.sh` clears previous results; `./allcontinue.sh` resumes them. Edit [setup.py](setup.py) to change:

| Parameter | Default and meaning |
| --- | --- |
| `DIAMETER` | $D=1$ m |
| `FREESTREAM_VELOCITY` | $U_\infty=1$ m/s |
| `REYNOLDS_NUMBER` | $Re_D=100$; $\nu=U_\infty D/Re_D=0.01$ m²/s |
| `AMPLITUDE` | Heave amplitude $A=0.2D$ |
| `FREQUENCY_RATIO` | Forcing over natural shedding frequency, $f/f_0=1$ |
| `NATURAL_STROUHAL_NUMBER` | $St_0=f_0D/U_\infty=0.165$ for the fixed cylinder |
| `TIME_STEP_SIZE` | $0.01$ s |
| `FINAL_TIME` | $120$ s, about 20 forcing periods |

The cylinder centre follows $y(t)=A\sin(2\pi ft)$, set by `fvm.HeavingMotion` inside `fvm.MeshMotion`. The O-grid has 128 cells around the cylinder and 96 radially, from $0.005D$ at the wall to a far-field circle of diameter $50D$, and one spanwise cell with `empty` faces. The whole grid, far field included, moves with the cylinder.

On the moving mesh the `cylinder` wall is a no-slip wall moving with the mesh. The upstream half of the far-field circle prescribes the stream; the downstream half fixes $p/\rho=0$ and lets the stream enter on reverse flow. These values are absolute-frame (inertial) quantities: the distant stream is the same uniform flow whether or not the boundary moves through it. Velocity, pressure and vorticity in `solution/` are also absolute-frame, on the moved mesh.

## Inspect lock-in

`figures/oscillating_cylinder_forces.png` shows drag, lift with the normalized cylinder position, and the lift spectrum. Forcing near the natural shedding frequency locks the wake onto the motion: lift oscillates at the forcing frequency with a fixed phase to the motion, and drag at twice that frequency. `figures/oscillating_cylinder_vorticity.png` shows the vortex street of the final frame around the displaced cylinder.

`./allplot.sh` prints the mean drag, the r.m.s. lift and the lift frequency over the second half of the run. A lift-to-forcing frequency ratio of one indicates lock-in; the default run gives a ratio of 1.000, mean $C_D=1.55$ and r.m.s. $C_L=0.29$. Change `FREQUENCY_RATIO` to move the forcing away from $f_0$ and find where the wake stops following the cylinder.

The force coefficients use the frontal area $D\times$ `DEPTH`. They include the inertial force of the fluid accelerated by the cylinder. Refine the grid and the time step together before quoting the coefficients.
