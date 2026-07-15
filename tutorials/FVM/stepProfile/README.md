# Backward-facing-step flow

This tutorial solves laminar incompressible flow through a sudden 2:1 expansion
with the native FVM PIMPLE solver. The upstream channel occupies
`1 <= y/h <= 2`; at `x/h = 0` the lower wall drops to `y/h = 0`, creating a
vertical step and a downstream recirculation region.

The default case uses `Re_h = U_b h / nu = 100`, a parabolic inlet, no-slip
solid walls, a fixed-pressure outlet, and one empty cell in the spanwise
direction. It writes the final cell fields and a near-wall reattachment
estimate to CSV. The estimate is a tutorial diagnostic, not a frozen
experimental validation threshold; production qualification still requires a
mesh-convergence study and an independent reference dataset.

Run the case and plots with:

```bash
./allrun.sh
```

For a quick smoke run:

```bash
python stepProfile_setup.py --end-time 0.1 --n-upstream 4 \
    --n-downstream 12 --n-height 4 --linear-solver spsolve
```
