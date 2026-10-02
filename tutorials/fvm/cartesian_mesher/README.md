# Flow around an STL body

A short 3D laminar case demonstrates [Cartesian mesh setup](../../../docs/fvm.md#mesh-setup) and [wall boundary conditions](../../../docs/fvm.md#boundary-conditions). The supplied STL is a $0.5\times1.0\times0.5$ m rectangular body centred at the origin.

From the repository root:

```bash
cd tutorials/fvm/cartesian_mesher
./allrun.sh
```

The run advances twenty steps of $\Delta t=0.01$ s. Open `solution/fvm.pvd` to inspect velocity and pressure. `./allcontinue.sh` resumes existing results; `./allrun.sh` clears them.

## Physical setup

`create_fvm_setup()` in [setup.py](setup.py) sets $\rho=1$ kg/m³, $\nu=0.01$ m²/s and inlet velocity $(1,0,0)$ m/s. The body is no-slip, the outlet fixes $p/\rho=0$, and the four lateral faces are slip boundaries. The initial velocity equals the inlet velocity.

`create_mesher()` sets:

| Control | Value |
| --- | --- |
| Domain $(x_\min,x_\max,y_\min,y_\max,z_\min,z_\max)$ | $(-2,3,-1.5,1.5,-1,1)$ m |
| `max_cell_size` | $0.5$ m |
| `boundary_cell_size` and body patch target | $0.25$ m |
| Wake refinement bounds | $(-1.5,2,-0.75,0.75,-0.5,0.5)$ m |
| Wake box target | $0.25$ m |

The box target produces nominal $0.125$ m cells because box refinement uses a strict upper bound; the patch target produces $0.25$ m cells. Overlapping refinement and mesh balancing can add finer cells.

To use another geometry, replace `assets/object.stl` with a closed STL in metres, keep its patch name `body`, and adjust the surrounding box and near-body/wake targets. This $0.2$ s run demonstrates setup; extend `end_time` and refine the mesh before interpreting steady loads or a developed wake.
