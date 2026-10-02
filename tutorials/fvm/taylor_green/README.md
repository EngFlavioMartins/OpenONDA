# Taylor–Green vortex decay

This periodic 2D case checks viscous decay and numerical dissipation. See the [FVM equations and units](../../../docs/fvm.md#physical-model-and-units), [periodic boundaries](../../../docs/fvm.md#boundary-conditions), and [time schemes](../../../docs/fvm.md#time-and-discretisation).

From the repository root:

```bash
cd tutorials/fvm/taylor_green
./allrun.sh
./allplot.sh
```

`./allrun.sh` clears previous results; `./allcontinue.sh` resumes them. Parameters are at the top of [setup.py](setup.py).

## Physical setup

The domain is $[0,2\pi]^2$ m with $24\times24$ cells, reciprocal cyclic pairs in $x$ and $y$, and one spanwise cell with `empty` faces. Density is $1$ kg/m³ and $\nu=0.1$ m²/s. The nominal time step is $0.005$ s, with final time $0.05$ s.

For amplitude $U_0=1$ m/s and wave number $k=1$ m⁻¹, the exact velocity is

$$
u=U_0e^{-2\nu k^2t}\sin(kx)\cos(ky),
\qquad
v=-U_0e^{-2\nu k^2t}\cos(kx)\sin(ky),
\qquad w=0.
$$

Kinetic energy and enstrophy both decay as $e^{-4\nu k^2t}$ relative to their initial values. `CONVECTION_SCHEME="central"` and `TIME_SCHEME="backward"` use central convection and BDF2 after startup.

## Compare with the solution

`solution/history.csv` contains numerical and analytic energy/enstrophy, velocity error, continuity and Courant number; `figures/taylor_green_decay.png` plots the decay. Increase `NUMBER_OF_CELLS` and reduce `TIME_STEP_SIZE` to assess convergence. Change `CONVECTION_SCHEME` to `"upwind"` to observe additional numerical dissipation.
