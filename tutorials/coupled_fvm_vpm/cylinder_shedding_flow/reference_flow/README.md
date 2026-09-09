# Cylinder reference flow

This directory contains the complete Re=150 reference-flow case. The only
geometric input is `assets/cylinder_long.stl`; `setup.py` declares the domain,
boundary types, mesh sizes, refinement boxes, samplers, and solver settings.

Run the complete four-grid study with:

```bash
./allrun.sh
```

The script runs `coarse`, `medium`, `fine`, and `very_fine` in sequence.
Each name selects its own output directories. Run `./allclean.sh` explicitly
when you want to remove old results. To run one grid directly:

```bash
python setup.py --name coarse --dx 0.125
```

Use the Python environment in which OpenONDA was installed.

`--dx` is the target cell size at the cylinder in units of D. The
background is 8 times that size. The near-body and near-wake boxes use 2 times
that size, and the wider wake box uses 4 times that size:

| Region | Bounds in x/D and y/D |
|---|---|
| Near body | `[-1, 2] × [-1, 1]` |
| Near wake | `[0, 6] × [-1, 1]` |
| Wake | `[0, 12] × [-1.5, 1.5]` |

The mesher builds a cfMesh-style Cartesian source mesh from the declarative
objects in `create_solver()`, takes an interior section, and extrudes it across
the one-diameter span. This matches the intended quasi-two-dimensional flow and
keeps the cylinder surface conformal. No generated STL, mesh dictionary, campaign
configuration, native-case directory, or preprocessing script is required.

After the fine resolution completes, OpenONDA writes `grid_study.json`,
`grid_study.csv`, `grid_study.md`, and `grid_study.png` under `solution/`. The
report includes force statistics, Strouhal number, Richardson/GCI estimates,
and the centreline-profile change over the common final-half time window.

`allclean.sh` removes the generated `solution/` and `samples/` directories.
