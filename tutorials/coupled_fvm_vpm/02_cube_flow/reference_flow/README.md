# Cube reference flow

This is the body-fitted Re=1000 cube reference case. The STL, domain, boundary
types, refinement regions, LES model, samplers, and solver settings are declared
in `create_solver()` in `setup.py`.

Run the four-grid study with:

```bash
./allrun.sh
```

The script runs `coarse`, `medium`, `fine`, and `dense` in
sequence at target `h/D = 0.10, 0.08, 0.06, 0.04`. Fields and meshes
are written below `solution/<name>/`; samples are written below
`samples/<name>/`. It never invokes `allclean.sh`, never deletes outputs, and
does not run postprocessing automatically. To run one case directly, use for
example:

```bash
python -u setup.py --name fine --dx 0.06
```

The launcher uses the active OpenONDA installation. When working on source,
install the checkout once with `python -m pip install -e .` from the repository
root so edits apply from any case directory. MPI launch, thread limits and
rank-owned output are handled by the FVM factory using the configured cores.

Adaptive timesteps now divide the interval to the next output/sample/backup
time into CFL-limited steps. This prevents an almost complete interval from
being followed by a tiny remainder step, which caused pressure and drag
spikes in the earlier fine run. The Courant target, BDF2 scheme and output
times are unchanged. See the [diagnosis](../../../../studies/coupler_accuracy/cube-drift-and-drag-2026-09-14.md).

The outer domain is `[-6.5,13] x [-6.5,6.5] x [-6.5,6.5]`. The background
cell-size target is `12h`; the near-body region
`[-1.5,3] x [-1.5,1.5] x [-1.5,1.5]` uses `h`, the downstream wake
`[-2,8] x [-2,2] x [-2,2]` uses `2h`, and the cube patch uses `h`.
The native mesher applies these as upper-size controls. The saved fine mesh
has nominal cube-adjacent Cartesian spacing `0.045 m` for requested
`h=0.06 m`, matching the coupled FVM's nominal local spacing. Surface fitting
and wrapper cells still give different shapes and volumes in the two meshes.

New mesh and time-step VTU/PVTU output includes `cell_volume` (m³),
`cell_equivalent_size` (cube root of volume, m), `cell_size` (nominal octree
edge, m), and `refinement_level` (absolute octree level). In ParaView, inspect
the **Cell Data** association. Existing time-step files are unchanged; open
their separate `mesh.vtu` for the sizes/volumes already saved, or apply
ParaView's **Cell Size** filter to compute geometric volumes.

The previously saved `very_coarse` and `coarse` cases used different domains
and background sizes. Their names alone do not establish a consistent grid
study. See the [cell-size investigation](../../../../docs/verification/mesher_cell_sizes.md)
for the recorded settings and diagnosis.

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
