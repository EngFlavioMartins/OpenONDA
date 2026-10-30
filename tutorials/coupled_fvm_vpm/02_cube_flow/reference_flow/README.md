# Cube FVM reference at $Re=1000$

This body-fitted grid study supplies the reference for the [coupled cube](../README.md). The no-slip cube, freestream and equilibrium Smagorinsky LES closure match the coupled case.

The domain is $[-6.48,12.96]\times[-6.48,6.48]^2$ m. The mesh uses the requested spacing $h$ over the coupled FVM region, then coarsens downstream and away from the cube. Time stepping limits the Courant number to 0.5.

## Run

Run commands from this directory.

```bash
./allrun.sh
./allplot.sh
```

The launcher runs three grid levels:

| Case | Cell size $h$ |
| --- | ---: |
| `grid_h010125` | 0.10125 m |
| `grid_h00675` | 0.0675 m |
| `grid_h0045` | 0.045 m |

Use `python setup.py --name grid_h0045 -h 0.045` for one grid. The coupled field plot resamples the completed fine-grid fields onto the coupled mid-plane lattice.

Force histories, profiles and mid-plane slices are written below `samples/<case>/`. `../allplot.sh` compares the completed reference with the coupled solution.
