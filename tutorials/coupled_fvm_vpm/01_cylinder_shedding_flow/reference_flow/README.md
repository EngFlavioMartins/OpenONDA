# Cylinder reference flow

This directory contains the complete Re=150 reference-flow case. The only
geometric input is `assets/cylinder_long.stl`; `setup.py` declares the domain,
boundary types, mesh sizes, refinement boxes, samplers, and solver settings.

Run the complete five-grid study with:

```bash
./allrun.sh
```

The script runs `very_coarse`, `coarse`, `medium`, `fine`, and `very_fine` in
sequence at target `h/D = 0.060, 0.050, 0.040, 0.030, 0.020`. Each name
selects its own output directories. Run `./allclean.sh` explicitly when you
want to remove old results. To run one grid directly:

```bash
python setup.py --name coarse --dx 0.050
```

Use the Python environment in which OpenONDA was installed.

`--dx` is the target cell size at the cylinder in units of D. The
background is 8 times that size. The near-body and near-wake boxes use 3 times
that size, and the wider wake box uses 6 times that size:

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

After each completed run, OpenONDA registers the case below
`samples/<name>/grid_run.json` and refreshes the generic `grid_study.*` summary
once at least three grids are available. After the campaign, run the qualified
case-local postprocessor:

```bash
python -u postprocess_grid_study.py
```

It discovers every completed registered grid, reads the saved force histories
and centreline samples over one common final-half time window, and writes only
derived `grid_convergence.{json,csv,md}` reports and thesis-style PNG/PDF
figures below `solution/`. Use `--statistics-start` and `--statistics-end` to
repeat the analysis for a fixed physical-time window, or `--format png` to
write PNG figures only. The report includes force statistics, a screened
Strouhal estimate, finest-pair changes, Richardson/GCI diagnostics when their
assumptions hold, and centreline-profile comparisons. It never modifies the
native mesh, field, backup, or sample files.

`allclean.sh` removes the generated `solution/` and `samples/` directories.
