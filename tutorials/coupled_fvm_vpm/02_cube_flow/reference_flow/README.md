# Cube reference flow

This is the body-fitted Re=1000 cube reference case. The STL, domain, boundary
types, refinement regions, LES model, samplers, and solver settings are declared
in `create_solver()` in `setup.py`.

Run the non-destructive five-grid study with:

```bash
./allrun.sh
```

The script runs `very_coarse`, `coarse`, `medium`, `fine`, and `very_fine` in
sequence at target `h/D = 0.22, 0.20, 0.18, 0.16, 0.14`. Fields and meshes
are written below `solution/<name>/`; samples are written below
`samples/<name>/`. It never invokes `allclean.sh`, never deletes outputs, and
does not run postprocessing automatically. To run one case directly, use for
example:

```bash
python -u setup.py --name coarse --dx 0.20
```

The outer domain is `[-7.5,15] x [-7.5,7.5] x [-7.5,7.5]`. The background
cell-size target is `12h`; the near-body region
`[-1.5,2.5] x [-1.5,1.5] x [-1.5,1.5]` uses `3h`, the downstream wake
`[0,8] x [-2,2] x [-2,2]` uses `6h`, and the cube patch uses `h`. The native
mesher applies these as upper-size controls.

Each completed case updates the generic `grid_study.json`, `grid_study.csv`,
`grid_study.md`, and `grid_study.png` under `solution/`. After the campaign
finishes, run `postprocess_grid_study.py`. It discovers every completed grid
case with `samples/<name>/grid_run.json`, compares their force statistics over
one common final-half time window, and writes the more detailed
`grid_convergence.{json,csv,md,png}`, `grid_convergence_by_cells.png`, and
`grid_convergence_profiles.png` under `solution/`. The profile figure states
when no common line data are available.

Only completed cases with both `grid_run.json` and `forces_history.csv` are
put on the convergence axes. The plots use the requested cube-patch `h/D` and
the actual registered fluid-cell count. A legacy force history without grid
metadata is listed as excluded rather than being assigned a guessed cell size
or count.

Run the postprocessor again after adding a case or choose a different common
statistics window without changing simulation outputs:

```bash
python -u postprocess_grid_study.py --statistics-start 10
```

Differences are reported relative to the finest available grid; they are not
claimed to be exact discretization errors. Richardson/GCI estimates are only
reported when the three finest distinct levels are monotone and use matching
refinement ratios.

`allclean.sh` removes generated solutions, samples, and grid-study outputs, so
invoke it only when you intentionally want to discard a campaign.
