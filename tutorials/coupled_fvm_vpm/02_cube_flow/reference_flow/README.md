# Cube reference flow

This is the body-fitted Re=1000 cube reference case. The STL, domain, boundary
types, refinement regions, LES model, samplers, and solver settings are declared
in `create_solver()` in `setup.py`.

Run the four-grid study with:

```bash
./allrun.sh
```

The script runs `very_coarse`, `coarse`, `medium`, and `fine` in
sequence at target `h/D = 0.12, 0.10, 0.08, 0.06`. Fields and meshes
are written below `solution/<name>/`; samples are written below
`samples/<name>/`. It never invokes `allclean.sh`, never deletes outputs, and
does not run postprocessing automatically. To run one case directly, use for
example:

```bash
python -u setup.py --name fine --dx 0.06
```

To add the optional denser level without rerunning the other grids:

```bash
python -u setup.py --name dense --dx 0.04
```

The new case will be discovered automatically by the postprocessor after it
completes. Its configured end time is 30 s, as for the other grids; adding cells
alone does not extend the averaging record.

The launcher uses the active OpenONDA installation. When working on source,
install the checkout once with `python -m pip install -e .` from the repository
root so edits apply from any case directory. MPI launch, thread limits and
rank-owned output are handled by the FVM factory using the configured cores.

Adaptive timesteps now divide the interval to the next output/sample/backup
time into CFL-limited steps. This prevents an almost complete interval from
being followed by a tiny remainder step, which caused pressure and drag
spikes in the earlier fine run. Full solution fields and restart backups are
saved every 0.5 s; force samples remain every 0.05 s and line samples every
0.25 s. See the [diagnosis](../../../../studies/coupler_accuracy/cube-drift-and-drag-2026-09-14.md).

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

An older campaign used different domains and background sizes. The four runs
currently saved through 30 s have matching domain bounds, recorded physical
and solver settings, and proportional mesh-size controls. The postprocessor
checks these original records instead of inferring compatibility from case
names or the current setup. See the [cell-size investigation](../../../../docs/verification/mesher_cell_sizes.md)
for the earlier diagnosis.

Each completed case updates the generic `grid_study.json`, `grid_study.csv`,
`grid_study.md`, and `grid_study.png` under `solution/`. After the campaign
finishes, run `postprocess_grid_study.py`. Use its `grid_convergence.*` reports
for the qualified frequency and drift checks; the automatic `grid_study.*`
files are only the generic quick summary. It discovers every completed grid
case with `samples/<name>/grid_run.json` and compares their force statistics over
one common final-half time window. It prints the grid table and assessment in
the terminal, and writes `grid_convergence.{json,csv,md}` under `solution/`.
The command defaults to PNG and PDF exports of:

- `grid_convergence`: mean drag and drag/lift/side-force fluctuation RMS versus target spacing.
- `grid_convergence_by_cells`: the same statistics versus global fluid-cell count.
- `grid_convergence_histories`: drag and lift during the chosen averaging window, exposing remaining drift.
- `grid_convergence_profiles`: mean streamwise velocity on the centreline and off-axis wake line.

Figures use the shared thesis theme, boxed axes without a background grid,
12.5 cm width, and symmetric margins. The report includes pressure/viscous drag
components, changes between half-window means, four-block means, whole-line
and wake-only profile differences, and the recorded sample-time-step range.
Fluctuation RMS measures variation about the mean; it is not a confidence
interval for that mean. The sampled time-step range does not cover steps at
which forces were not written.

Strouhal numbers use `St = f D/U` with the saved force-sampler scales. The
postprocessor withholds a frequency unless the record contains at least five
candidate cycles, eight samples per cycle, limited drift, a concentrated
spectral peak, and consistent frequencies in both window halves. Every
rejection has a reason. These are practical screens, not proof of spectral
convergence. A default final-half window does not establish stationarity.

Only completed cases with both `grid_run.json` and `forces_history.csv` are
put on the convergence axes. The plots use the requested cube-patch `h/D` and
the actual registered fluid-cell count. A legacy force history without grid
metadata is listed as excluded rather than being assigned a guessed cell size
or count.

Run the postprocessor again after adding a case or choose a different common
statistics window without changing simulation outputs:

```bash
python -u postprocess_grid_study.py
python -u postprocess_grid_study.py --statistics-start 20 --statistics-end 30 --output-dir solution/window_20_30
```

Use `--format png` or `--format pdf` to export only one format. `--output-dir`
changes only derived report placement; native metadata is read from the
`solution/` directory beside the selected `--samples-root`, or from an explicit
`--solution-root`. No simulation outputs are modified. Input hashes in the
JSON identify the histories, metadata and postprocessor used.

Differences are reported relative to the finest available grid; they are not
claimed to be exact discretization errors. Richardson/GCI estimates are only
reported when the three finest distinct levels are monotone and use matching
refinement ratios. Incompatible saved settings and drag half-window drift
above the documented 1% diagnostic screen also suppress those estimates.
An optional `--tolerance 0.01` records which finest-pair changes fall below 1%;
it does not certify the run as grid-independent.

For the present 15–30 s window, mean Cd is 1.00498, 0.999316, 0.971000 and
0.965156 from very coarse to fine. Although medium/fine mean Cd differs by
only 0.606%, fine-grid half-window drag drift is 11.9%, drag fluctuation RMS
changes by 33.4%, and wake-profile differences are approximately 6% and 10%.
All force-derived Strouhal estimates are unresolved. These records therefore
do not yet establish fine-grid independence. A dense run is a useful spatial
comparison, but longer stationary records and a separate time-step check are
needed before making that claim.

`allclean.sh` removes generated solutions, samples, and grid-study outputs, so
invoke it only when you intentionally want to discard a campaign.
