# taylor_green — periodic 2D decaying vortex

This case runs the incompressible PIMPLE solver on the Taylor–Green vortex in
the periodic domain `[0, 2π]²`. The one-cell-thick mesh uses translational
cyclic pairs in `x` and `y` and empty boundaries in `z`.

The analytic solution is

```text
u = exp(-2 nu t) sin(x) cos(y)
v = -exp(-2 nu t) cos(x) sin(y)
KE(t) = KE(0) exp(-4 nu t)
```

Run the case and generate the validation plot with:

```bash
./allrun.sh
```

The run writes numerical and analytic energy and enstrophy decay, velocity
error, continuity, and CFL to `solution/history.csv`. The default central scheme
is appropriate for this smooth vortex; `--scheme upwind` is available to
demonstrate numerical dissipation.
