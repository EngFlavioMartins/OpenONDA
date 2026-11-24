# Coupled FVM–VPM cases: run status and data folders

Status on Tue 24 Nov 2026, 11:00 CET. The supervising session updates the check boxes as runs end.

- `[x]` complete: the data may be plotted.
- `[ ]` not complete: do not use for results. A running case has partial data that will change.

All paths are relative to this folder. On the workstation it is
`~/Projects/OpenONDA/tutorials/coupled_fvm_vpm/`; on the MacBook it is
`~/OpenONDA_tutorials/coupled_fvm_vpm/` (Syncthing, workstation sends, Mac receives).

## What reaches the MacBook

| Data | On the Mac |
|---|---|
| `samples/` of every case, reference and study run (forces, profiles, slices) | yes |
| `figures/` of every case (PNG and CSV summaries) | yes |
| `figure_data/` (data files of the paper and thesis figure scripts) | yes |
| Run metadata and coupler logs inside `solution/` | yes |
| Particle frames of the baseline cylinder run (`.../box2.4_sub5_inset8.75/solution/vpm/`) | yes |
| All other raw output in `solution/` (FVM fields, particle frames, backups) | no, workstation only |

A figure that needs raw cube or wing particle frames must be drawn on the workstation, or its
data extracted there into `figure_data/` first.

## 01 Cylinder (calibration study of the interface)

Folder `01_cylinder_shedding_flow/`. Every coupled run is in
`01_cylinder_shedding_flow/solution/sensitivity_analysis/<run>/`, with its sampled data in
`<run>/samples/` (`forces_history.csv`, `fvm_transverse_x2.csv`, `vpm_transverse_x2.csv`,
`vpm_transverse_x4.csv`). Run names: `box` = FVM box margin, `sub` = FVM steps per coupling
interval (N), `inset` = transfer inset.

| Done | Group | Variation | `<run>` | End time |
|---|---|---|---|---|
| [x] | baseline | N = 5, margin 2.4, inset 8.75 | `box2.4_sub5_inset8.75` | 100 s |
| [x] | T, interval | N = 1 | `box2.4_sub1_inset8.75` | 55 s |
| [x] | T | N = 2 | `box2.4_sub2_inset8.75` | 55 s |
| [x] | T | N = 3 | `box2.4_sub3_inset8.75` | 55 s |
| [x] | T | N = 6 | `box2.4_sub6_inset8.75` | 55 s |
| [x] | M, transfer iteration | one pass | `box2.4_sub5_inset8.75_k1` | 55 s |
| [x] | M | tolerance 1e-3 | `box2.4_sub5_inset8.75_tol1e-3` | 55 s |
| [x] | M | tolerance 1e-4 | `box2.4_sub5_inset8.75_tol1e-4` | 55 s |
| [x] | M | constant 1e-5 | `box2.4_sub5_inset8.75_const` | 55 s |
| [x] | D, box margin | margin 1.6 | `box1.6_sub5_inset8.75` | 55 s |
| [x] | D | margin 2.0 | `box2.0_sub5_inset8.75` | 55 s |
| [x] | D | margin 3.2 | `box3.2_sub5_inset8.75` | 55 s |
| [x] | D | margin 4.0 | `box4.0_sub5_inset8.75` | 55 s |
| [x] | I, transfer inset | inset 4 | `box2.4_sub5_inset4` | 55 s |
| [x] | I | inset 14 | `box2.4_sub5_inset14` | 55 s |
| [x] | S, FVM cell size | cell 0.08, interval 0.048 s | `box2.4_sub3_inset8.75_h0.08` | 55 s |
| [x] | S | cell 0.06, interval 0.048 s | `box2.4_sub4_inset8.75_h0.06` | 55 s |
| [x] | P, particle spacing | spacing 0.08 | `box2.4_sub5_inset8.75_hp0.08` | 55 s |

Fully meshed references, in `01_cylinder_shedding_flow/reference_flow/samples/<name>/`:

| Done | Reference | `<name>` |
|---|---|---|
| [x] | cell 0.04, the baseline reference | `h0.04_dt0.008` |
| [x] | cell 0.03, 0.06, 0.08 (mesh convergence) | `h0.03_dt0.006`, `h0.06_dt0.012`, `h0.08_dt0.016` |
| [x] | wide far field, for margin 3.2 | `h0.04_dt0.008_fb3.2` |
| [x] | wide far field, for margin 4.0 | `h0.04_dt0.008_fb4.0` |

- Study table: `python -m assets.study_status` from the case folder (workstation), or
  `01_cylinder_shedding_flow/figures/study_summary.csv`.
- Figures: `01_cylinder_shedding_flow/figures/`, redrawn by `allplot.sh`. `[x]` Redrawn on 19 Nov,
  after all 18 runs had ended (the last on 16 Nov).
- `01_cylinder_shedding_flow/solution/coupled`, `fvm` and the logs directly in
  `01_cylinder_shedding_flow/solution/` date from October and are not results. Use only
  `solution/sensitivity_analysis/`.

## 02 Cube

Folder `02_cube_flow/`.

| Done | Run | Data | State |
|---|---|---|---|
| [x] | Coupled final (particle domain to 18 m, 6 m fringe) | `02_cube_flow/samples/`, `02_cube_flow/solution/` | stopped on 19 Nov at t = 98.5 s (forces) by the user's decision; last backup t = 98 s; window 30–98 |
| [x] | Fully meshed reference | `02_cube_flow/reference_flow/samples/grid_h0045_dt0.02/` | complete, t = 130 s; only 30–98 s is used |
| [x] | Figures | `02_cube_flow/figures/` | redrawn on 19 Nov from the final run |

- The earlier coupled cube run (24 m domain, stopped at t = 48 s) is not a result. It is outside
  this folder, in `~/openonda_runs/old_data_2026-11-15_cube_fringe12/` on the workstation.
- `[x]` The configuration figure of the paper (`case_configurations`, redrawn on 19 Nov) draws the
  domains from the `setup.py` parameters and no particles, so it no longer uses that earlier run.
- Other reference folders, used for the FVM error of the paper:
  - `grid_h0045` has the reference cells (343,974) with a step four times shorter, 0.005 s, to
    t = 60 s (5 Nov).
  - `[x]` `grid_h00675_dt0.005` is the next coarser grid (106,399 cells, 0.005 s) to t = 60 s, run
    on 23–24 Nov on 37d7d7e3, the solver of the reference (the FVM of `grid_h0045` and of the
    coupled run). Over 30–60 s its mean drag is 4.1 % above that of `grid_h0045`; the paper uses
    this value.
  - `grid_h00675` is the same grid to t = 100 s, run on 17 Oct with an older solver version
    (+6.1 % over 30–60 s). It is no longer used.
  - The `cell_count` of `grid_run.json` (385,776 and 1,287,704) is not the number of cells in a
    parallel run; `solution/<name>/fvm_metadata.json` gives the cells.
  - `grid_h010125` is an early mesh that stopped at t = 0.16 s, and is not used.

## 03 Wing (NACA 4412)

Folder `03_naca4412_flow/`.

| Done | Run | Data | State |
|---|---|---|---|
| [x] | Coupled final | `03_naca4412_flow/samples/`, `03_naca4412_flow/solution/` | complete, t = 48 s (22 Nov 13:35); restarts at t = 24 and 38.88 checked: forces, profiles and slices continuous |
| [x] | Fully meshed reference | `03_naca4412_flow/reference_flow/samples/` | complete, t = 60 s; only 36–48 s is used |
| [x] | Figures | `03_naca4412_flow/figures/` (`forces.png`, `errors_by_window.png`, `velocity_profiles/`, `starting_vortex/`, `fields/`, `comparison_summary.csv`) | redrawn 22 Nov from the final, window 36–48 s; `fields/t012.00`–`t024.00.png` are the snapshots of the old 12–24 window |

Coupled samples: `forces_history.csv`; slices `fvm_slice_x2`, `fvm_slice_z0`, `vpm_slice_x2`,
`vpm_slice_x5`, `vpm_slice_z0` (181 frames each, t = 12 to 48 s); profiles `*_spanwise_*.csv`,
`*_transverse_*.csv`; `starting_vortex/`.

- Why the extension: the load oscillation (period about 1.5 c/U) is still growing at t = 24 s in both
  runs. Data up to t = 24 s is final and may be plotted (start-up, starting vortex, mean loads).
- The reference is periodic from t = 36 s (cycle mean of C_L constant to 0.0005, amplitude 0.0055–0.0058),
  so both runs are compared over 36–48 s, eight periods.
- Over 36–48 s: Strouhal number 0.664 in both runs (periods within 0.04 %), lift in phase
  (correlation 0.995), mean lift +0.13 % and mean drag −0.14 %, lift amplitude 0.0063 against
  0.0057 (+11 %). The reference amplitude stays at about 0.0058 up to t = 60 s.
- `[x]` Paper and thesis (chapter 5, also chapters 1 and 6) text: wing numbers moved to the 36–48 s window
  and the load oscillation added (22 Nov).

## 04 Oscillating cylinder, 05 Pitching airfoil, 06 Rotor

| Done | Case | Folder | State |
|---|---|---|---|
| [ ] | 04 | `04_oscillating_cylinder/` | not run: no data |
| [ ] | 05 | `05_pitching_airfoil/` | not run: no data |
| [ ] | 06 | `06_rotor_flow/` | not run: no data |

## Data files of the paper and thesis figures

`figure_data/paper/` and `figure_data/thesis/` mirror the `data/` folders beside the figure
scripts of the two repositories; see `figure_data/README.md`.

| Done | Figure data | State |
|---|---|---|
| [x] | Cylinder baseline (paper Figures 1, 4, 6, 7; thesis cylinder wake t = 10, 20, 40, 100) | from the final baseline run |
| [x] | Wing comparison (paper `wing_comparison`, `wing_wake`, `wake_profiles`; thesis `naca4412_wing`) | from the final wing run, window 36–48 (paper redrawn 22 Nov) |
| [x] | Paper `cylinder_study` | re-extracted on 17 Nov from all 18 runs (paper repository, `figures/common/cylinder_study/`); the copy in `figure_data/` is older |
| [x] | Configurations (paper `case_configurations`) | domains from the `setup.py` parameters, no particles (19 Nov) |
| [x] | Cube comparison | paper `force_histories`, `force_spectra`, `wake_profiles` and the cube numbers, from the final run over 30–98 s |
| [x] | Paper `force_histories`, `force_spectra` (three cases together) | final: cylinder, cube window 30–98, wing window 36–48 (redrawn 22 Nov) |
| [x] | Paper Figure 7 in 3D (`figures/flow_fields/render/`) | samples `render_t<T>/` (FVM volume, particles, `render.json`) by `export_render_samples.py`: cylinder t = 100, cube t = 98, wing t = 48 (22 Nov); panels drawn by `render_flow_fields_vtk.py`, stacked by `stack_panels.py` |
