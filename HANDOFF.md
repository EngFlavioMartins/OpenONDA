# OpenONDA continuation handoff, 2 November 2026

This file hands the project from the 16 GB laptop (i7-12700H, RTX 3060 Laptop 6 GB) to the
132 GB machine. It lists the objectives in the order the author wants them done, the exact
state of each, what lives only on the laptop, and how an agent on the large machine can run
many of these in parallel. Everything that exists in git at commit time is on `origin`; the
data of the runs (`sensitivity/`, `solution/`, `samples/`, `figures/auxiliary` excepted) is
not in git and is listed in section 2.

Author of every commit: Flavio Martins <flavio.martins@outlook.com>. Every commit message ends
with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`. Work on branch `development`.

## 0. Ground rules (apply to every objective)

1. **No fabricated or estimated numbers.** Every number in a README, a figure, a note or a
   report comes from a file produced by a run; cite the file. A number copied from memory or a
   chat is not a result.
2. **One solver version per comparison.** Never mix runs of different solver commits in one
   table, figure or series. Each study run records its import path in `solver.txt`; the
   coupled-cylinder study imports a frozen export of commit `3f771630` (section 3.4).
3. **Do not relax a tolerance, limit, check or the halt rule to make a run pass. Do not change
   solver defaults** inside a campaign. A defect found during a campaign is fixed in the
   repository, and the campaign restarts on the fixed commit with a new frozen export.
4. A statement is verified only after a command that could have shown it false.
5. Report failures plainly: what was run, what stopped, where the log is. A run that did not
   happen is "not run", never "pending" without a reason.
6. **Never silently fall back to the CPU.** A coupled or particle run that is meant for the GPU
   and finds none must stop with a message; a CPU run is a declared choice
   (`--compute-device cpu` where a case offers it), recorded in the case's README.
7. Tutorial conventions (checked by `tests/tutorials`): every case has `allclean.sh`,
   `allrun.sh` (clean, then `python setup.py`), `allcontinue.sh`, `allplot.sh`
   (`python -m assets.plot_<x>`), `setup.py`, `assets/`, `README.md`; launchers begin with
   `#!/bin/bash`, `set -e`, a blank line, `cd "$(dirname "$0")"`; tutorial `.py` files use no
   try/raise/assert/decorators, no `.exists()/.is_file()/.is_dir()`, no `getattr/hasattr`;
   ruff line length 100; no case-specific solver code; no result archiving in the case.
8. Figures of fields use the README gallery style (white background, PBR+IBL shaders, TikZ text
   overlay, 12.5 cm at 300 dpi, time in the file name). 2D plots: one legend entry per data type,
   "FVM only" for the FVM-only reference, no `x10^n` axis offsets, left and right margins fitted
   to the y labels.
9. Zero legacy: delete what is not the latest, working and tested (unused public API, knobs that
   never passed, write-only diagnostics). Removal of a physics model is the author's decision.
10. Studies are sections of an existing case's sensitivity study, never new case directories.
11. Tests run one file per process: `python tests/run_isolated.py` (Taichi, CUDA and the FTZ
    state are process-global). Pre-existing failures at handoff are listed in section 8.

## 1. Objectives, in order

1. Complete the coupling-interface error and sensitivity analysis of the planar cylinder
   (`tutorials/coupled_fvm_vpm/01_cylinder_shedding_flow`), section 3.
2. Re-run every tutorial case: VPM only, VLM+VPM and FVM only (`tutorials/vpm`,
   `tutorials/fvm`), section 4.
3. Re-run the cube flow (`02_cube_flow`) and the wing (`03_naca4412_flow`) coupled cases with
   their FVM-only references, section 5.
4. Fix and test the FVM moving mesh and the coupled moving mesh: `tutorials/fvm/oscillating_cylinder`,
   `04_oscillating_cylinder` (elastically mounted cylinder, two degrees of freedom),
   `05_pitching_airfoil`, `06_rotor_flow` (meshed blades), section 6.

Section 7 gives a lane plan for running these in parallel on the large machine.

## 2. State at handoff

### 2.1 Git

| Branch | Head at handoff | Content |
| --- | --- | --- |
| `development` | this commit | Solver with the mixed coupling condition on every face (`3f771630`, 1 Nov), rigid moving mesh (translation and rotation), study scripts and matched references (`c19cd8a2`), the author's "Candidate release" (`5c9a29aa`, 2 Nov 11:07), and the **uncommitted work of the moving-mesh session committed as is** in this commit (section 6.2). |
| `cube-optimisation` | pushed with this handoff | Worktree `../OpenONDA-cube` on the laptop, base `f2c71c6c`: selective-eddy-viscosity constant 0.5, strain-cap check excluding the transfer region, and commit `3f771630` applied as a working-tree patch. Committed as is with this handoff (section 5.2). Not merged: the `3f771630` part duplicates `development`. |
| `moving-mesh-port` | `57d60ac7` | Ancestor of `development`; nothing to merge. |
| `integration`, `conventions-alignment`, `release-candidate`, `codex/cfd-nomenclature` | pushed with this handoff | Older branches, all merged into `development` before 2 Nov as far as the author's notes record; keep for history only. |
| `main` | `0c0b3bc3` | Public release branch. Do not force-push or touch without the author. |

Laptop-only worktrees (`git worktree list` there): `../OpenONDA-cube`, `../OpenONDA-port`,
`../OpenONDA-devcheck` (detached, bisection), and several `.claude/worktrees/agent-*`. None holds
work that is not on a pushed branch after this handoff.

### 2.2 Data that is NOT in git

Everything under `**/sensitivity/`, `**/solution/`, `**/samples/` and `.claude/` is ignored.
Sizes on the laptop at handoff:

| Path (relative to the repository) | Size | What |
| --- | --- | --- |
| `tutorials/coupled_fvm_vpm/01_cylinder_shedding_flow/sensitivity/` | 39 GB | all study runs and references (section 3.2) |
| `tutorials/coupled_fvm_vpm/01_cylinder_shedding_flow/reference_flow/` | 1.6 GB | standard FVM-only reference to 100 s |
| `tutorials/coupled_fvm_vpm/02_cube_flow/` | 13 GB | cube coupled run to 33 s **with the defective 613b5898 solver** and its reference (section 5) |
| `tutorials/coupled_fvm_vpm/03_naca4412_flow/` | 2.4 GB | wing attempts, invalid (section 5) |
| `~/openonda_runs/study_2026-11/` | | frozen solver exports `solver_3f771630`, `solver_613b5898`; `launch_study.sh`, `launch_references.sh`, `resource_log.sh` |
| `~/openonda_runs/bisect/`, `~/openonda_runs/cases3d/` | | regression bisection of 1 Nov; the 3D-case watchdog of October (stale) |
| `.claude/cube_screens/` | | cube screen queue (`gate.sh`, `queue1.sh`, `screen.sh`), waits for the GPU |

To continue the cylinder study with the laptop's runs instead of re-running them, copy them
before the study restarts there (the laptop keeps running them until told otherwise; the
`COMPLETE.sh` driver resumes any copied run from its latest backup):

```bash
rsync -a --info=progress2 <laptop>:Projects/OpenONDA/tutorials/coupled_fvm_vpm/01_cylinder_shedding_flow/sensitivity/ tutorials/coupled_fvm_vpm/01_cylinder_shedding_flow/sensitivity/
```

```bash
rsync -a <laptop>:Projects/OpenONDA/tutorials/coupled_fvm_vpm/01_cylinder_shedding_flow/reference_flow/samples/ tutorials/coupled_fvm_vpm/01_cylinder_shedding_flow/reference_flow/samples/
```

The `figures/auxiliary/*.csv` tables of case 01 are in git and hold every statistic the analysis
printed at handoff, so the analysis scripts can be checked against them before any new run.

### 2.3 Processes still running on the laptop at handoff (2 Nov, 14:00)

- `openonda-cylinder-study-3f771630` (systemd user unit): two GPU slots, running
  `box2.4_sub1_inset8.75` (35 s of 55) and `box2.4_sub3_inset8.75` (52.6 s of 55), then the rest
  of section `n_interval`, then `i_transfer inset k b1 p v_side f_side`.
- `openonda-cylinder-references`: the matched references `reference_fw4.0_fb2.0`,
  `reference_fw3.6_fb1.6`, `reference_fw6.0_fb4.0`, waiting for 7000 MiB of available RAM.
- `openonda-cube-screens`: gate script waiting for the study to free the GPU.
- The moving-mesh session: a short CPU check of case 04 (`--reduced-velocity 6 --end-time 0.4
  --compute-device cpu`, 20k cells), state unknown at commit time.

## 3. Objective 1: coupling-interface error and sensitivity analysis

Case: `tutorials/coupled_fvm_vpm/01_cylinder_shedding_flow`. Planar cylinder, Re = 150, FVM box
around the body, one slip-slab particle layer outside it. Purpose for the paper
(`~/Nextcloud/Research/Hybrid_Solver_Paper1`): split the error of each statistic (mean drag,
lift amplitude, Strouhal number) into the FVM discretisation error and the coupling error,
attribute the coupling error to the groups D (FVM domain), I (interface), T (coupling interval),
V (particle side), F (FVM side), and give tuning rules for speed and accuracy: the smallest FVM
box that couples stably, the largest coupling interval, the interface parameters.

### 3.1 Scripts and files

| File | Role |
| --- | --- |
| `COMPLETE.sh` | Unattended driver: `--slots N` GPU workers walk the section list, references on the CPU; resumes from backups; halts after three quick failures (`sensitivity/HALTED`); `--dry-run` lists stored, pending and archived runs; `--sections "..."` restricts and orders. Log `sensitivity/COMPLETE.log`. |
| `assets/run_sensitivity_study.sh` | Section definitions (`SECTIONS=(...)`, `if want <name>`), the `run` and `reference` functions, GPU and RAM guards (`MIN_FREE_GPU_MIB` 3200 with slots, `MIN_FREE_RAM_MIB` 4500 coupled, `REFERENCE_RAM_MIB` 7000). Do not lower the guards on a 16 GB machine; on the 132 GB machine the RAM guards are irrelevant but harmless. |
| `setup.py` | The coupled case; every study parameter is a command-line option (see the `run` lines in the section file). |
| `reference_flow/setup.py` | FVM-only reference; `--fine-wake-end X --fine-box-margin M` refines h cells over the coupled FVM domain `(-M, X) x (-M, M)` so that a coupled run has a reference with its own cells. |
| `assets/interface_sensitivity.py` | The analysis: period-aligned statistics over 40 to 55 s, sampling floor, e_FVM and e_coupling per run, matched or standard reference per series, series ranges against 2 sigma, tables `figures/auxiliary/interface_*.csv`. Run `python -m assets.interface_sensitivity` from the case directory. |
| `allplot_sensitivity.sh` | All study figures (`figures/interface_group_*.{pdf,png}`, `interface_ranking`, `sensitivity*`, `error_*`). |
| `assets/error_analysis.py`, `assets/plot_error_analysis.py` | Q_ext by the Eca-Hoekstra family fits and the space-time fit; error decomposition tables. |
| `assets/window_uncertainty.py`, `assets/study_status.py`, `assets/strength_ratio_history.py` | Window floor, run inventory, kappa history. |
| `assets/coupling_findings.md` | The findings note. **Still the October text (solver `3b4c0f23` to `97d65b09`); must be rewritten from `3f771630` data only.** |
| `handoff_2026-11-01/` | The author's original brief (`HANDOFF.md`) and the redesign note of 31 Oct. Contains a `setup.py` copy that breaks the collection of `tests/tutorials/test_continuation.py`; move or exclude it (author's call). |

### 3.2 Runs stored at handoff (all solver `3f771630` unless stated; times from `samples/forces_history.csv`)

Coupled, baseline FVM step 0.008 s, h = 0.0566 m, particle spacing 0.04 m, inset 8.75 cells:

| Run directory | Margin [D] | N (FVM steps per coupling) | Reached |
| --- | --- | --- | --- |
| `box2.4_sub5_inset8.75` | 2.4 | 5 | 100 s (floor run) |
| `box2.4_sub2_inset8.75` | 2.4 | 2 | 55 s |
| `box1.6_sub5_inset8.75`, `box2.0_sub5_inset8.75`, `box3.2_sub5_inset8.75`, `box4.0_sub5_inset8.75` | 1.6, 2.0, 3.2, 4.0 | 5 | 55 s |
| `box2.4_sub1_inset8.75`, `box2.4_sub3_inset8.75` | 2.4 | 1, 3 | running (35 s, 52.6 s) |

FVM-only references at 55 s (`sensitivity/reference_*` and `reference_flow/` to 100 s):
`reference_flow` (h 0.0566, dt 0.008, standard mesh), `reference_dt0.016`, `reference_dt0.004`,
`reference_dt0.002`, `reference_h0.08`, `reference_h0.08_dt0.004`, `reference_h0.056569`,
`reference_h0.056569_dt0.004`, `reference_h0.028284`, `reference_h0.028284_dt0.004`,
`reference_re100`, `reference_re250`, `reference_fw4.4` (h cells to x = 4.4 D over the baseline
box), `reference_dt0.004_fw4.4`, `reference_fw5.2_fb3.2` (h cells over the 3.2 D box),
`check_reference_613b5898` (reproduction check of the old solver, equal to `reference_flow` to
1e-6 relative). Archived, do not compare: `sensitivity/archive_pre_2026-10-31/`,
`sensitivity/archive_613b5898/`.

### 3.3 Results at handoff (from `figures/auxiliary/interface_runs.csv`, `interface_series.csv`, `interface_floor.csv`, written 2 Nov 11:35)

Statistics over whole shedding periods in 40 to 55 s; errors in percent of Q_ext.

Sampling floor (two runs to 100 s): sigma of a difference 0.041 % drag, 0.021 % lift amplitude,
0.032 % Strouhal; a series effect counts only above 2 sigma = 0.081 / 0.042 / 0.063 %.

Coupling error against the **standard** reference `reference_flow`, N = 5:

| Margin [D] | Mean drag | Lift amplitude | Strouhal | Boundary mismatch [U] | Removed strength | Cost [s per simulated s, RTX 3060] |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.6 | -0.330 | -2.041 | +0.697 | 4.9e-3 | 0.17 % | 210 |
| 2.0 | -0.121 | -1.375 | +0.474 | 3.6e-3 | 0.26 % | 220 |
| 2.4 | -0.082 | -1.316 | +0.354 | 2.8e-3 | 0.31 % | 252 |
| 3.2 | -0.282 | -2.781 | +0.134 | 2.0e-3 | 0.38 % | 263 |
| 4.0 | -0.336 | -2.014 | +0.169 | 1.7e-3 | 0.45 % | 259 |

Against the matched references: box 2.4 D vs `reference_fw4.4`: -0.144 / -0.857 / -0.053;
box 3.2 D vs `reference_fw5.2_fb3.2`: -0.399 / -2.419 / -0.311. The fine-box references differ
from the standard one by +0.062 / -0.459 / +0.407 (margin 2.4) and +0.117 / -0.363 / +0.445
(margin 3.2), so the lift deficit of the wider boxes is coupling error, not FVM discretisation.
The lift amplitude per 5 s bin from 30 s on differs between boxes by a constant factor (box 3.2
vs 2.4: -1.5 % in every bin): a steady difference, not a transient.

Coupling interval at kappa_max 1.8: N = 2 vs `reference_fw4.4` -0.675 / -2.168 / -0.800 with
kappa median 1.78; N = 5 -0.144 / -0.857 / -0.053 with kappa 1.48. Series ranges (all above
2 sigma): D margin 0.253 / 1.466 / 0.563 points; T interval 0.531 / 1.311 / 0.746 points.
FVM discretisation error of the h = 0.0566 reference itself (printed by the analysis on 2 Nov
02:25): drag +0.74, lift -3.57, Strouhal -0.82 %, so at N = 5 the coupling error is well below
the FVM error.

Standing conclusions, to be confirmed by the remaining sections: every box from 1.6 D up couples
stably (kappa 1.48, no transfer at the strength limit); 2.0 and 2.4 D are equal within the floor
and 2.0 D is 13 % cheaper; wider boxes lower the boundary mismatch but worsen lift, which is not
explained; the removed-strength fraction is the one diagnostic that grows with the box, and the
`i_transfer` section (removal threshold 0.01 and 0.2 against 0.05) tests it.

### 3.4 How the study runs (recipe for the large machine)

```bash
mkdir -p ~/openonda_runs/study_2026-11 && git archive 3f771630 openonda | tar -x -C ~/openonda_runs/study_2026-11/solver_3f771630
```

The export needs the compiled `_fenv*.so` of the FVM next to its source (build the package once
in the environment, then copy the `.so` files from the installed `openonda/` into the export, as
on the laptop). Then, from the case directory, with the `OpenONDA` conda environment and
`CUDA_PATH` pointing at the CUDA root:

```bash
STUDY_SOLVER=~/openonda_runs/study_2026-11/solver_3f771630 bash COMPLETE.sh --slots <GPU slots> --sections "h box_conv box_min n_interval i_transfer inset k b1 p v_side f_side g"
```

One coupled run holds about 2.8 GB of GPU memory and one CPU core for the FVM; slots = floor(GPU
memory / 3.2 GB) per card (the driver pins slot workers to six-core groups: adapt the `taskset`
ranges in `COMPLETE.sh` to the machine). References (`g` and section `p`'s FVM-only runs) are
CPU-only and independent: with 132 GB every reference can run at once (the largest mesher,
`fw6.0_fb4.0`, is estimated at 1.5 times the 4.7 GB of `fw5.2_fb3.2`). Run the driver as a
systemd user unit with `OOMPolicy=continue` as the laptop launchers do, or at least under
`nohup` with the log in `sensitivity/COMPLETE.log`.

With the finest runs (`--with-fine`): h = 0.0283 m and particle spacing 0.02 m need about 12 GB
of GPU memory each; on a card that has it, run them (section `h` with `RUN_FINE=1`, section
`f`). They were never run on the laptop.

### 3.5 Remaining runs (each coupled run about 4 to 4.5 h on the RTX 3060 to 55 s)

| Section | Runs | What it isolates |
| --- | --- | --- |
| `n_interval` | N = 1, 3, 6 at kappa_max 1.8 (two running); N = 1, 3, 4, 6 at kappa_max 3 | coupling interval with and without the strength limit |
| `i_transfer` | tau 0.01, 0.2; moment cap 0.02, 0.3 at N = 5; tau 0.2 at N = 2 | transfer threshold and conservation |
| `inset` | inset 4 and 14 cells | transfer-band position |
| `k` | coupling iterations 1, 2, 4 | interface iteration |
| `b1` | kappa_max 1, 2, 3, 10 at N = 5 and N = 2 | strength limit |
| `p` | N = 10, FVM step 0.004 | FVM step at fixed coupling interval |
| `v_side` | particle spacing 0.08, RK4, vorticity floor 0.001 and 0.1 | particle side |
| `f_side` | outer correctors 1 and 3; Courant controller 0.5 and 0.9 at FVM step 0.02 | FVM side |
| `g` | matched references fb2.0, fb1.6, fb4.0 (CPU) | own-cell references for the box series |
| not run | `e_old`, `e`, `d`, `a`, phase 2, `f`, the finest `h` runs | listed in the report as not run unless the large machine runs them |

### 3.6 Deliverables of objective 1

1. All sections above at 55 s on solver `3f771630`; the two anchor runs to 100 s exist.
2. `python -m assets.interface_sensitivity` and `bash allplot_sensitivity.sh` regenerated; every
   series uses one reference type (the matched references once all members have one).
3. `assets/coupling_findings.md` rewritten from the new tables only: error measure, floor, e_FVM
   and e_coupling per statistic, the group attribution D/I/T/V/F with the resolved ranges, the
   calibration of kappa_max, tau_transfer, moment cap, inset, iterations, interval; the tuning
   rules (minimum stable box, largest interval, parameter values) each tied to the figure and
   table that supports it; the list of what was not run and why.
4. The case `README.md` results section updated from the same tables.
5. An explanation, or an explicit open question, for the wider-box lift deficit before the paper
   claims a box recommendation.

## 4. Objective 2: re-run every tutorial case

The FVM fixes of October (PISO H/A per corrector, DILU default, momentum tolerance 1e-9,
freestream far-field pressure split, slip stress, cyclic LUST) changed the FVM results, so the
figures and README numbers of every FVM-only and coupled tutorial made before them are stale.
Each case is run with its own `allrun.sh`, then `allplot.sh`, and its README numbers replaced
from the files the run wrote. Every case is independent of the others, so all of them can run at
once within the GPU memory.

| Group | Cases (`tutorials/...`) | Device | Notes |
| --- | --- | --- | --- |
| FVM only | `fvm/taylor_green`, `fvm/step_profile`, `fvm/boundary_layer`, `fvm/cartesian_mesher`, `fvm/cube_flow`, `fvm/cylinder_ibm`, `fvm/oscillating_cylinder`, `fvm/airfoil_flow` | CPU (MPI, 4 ranks each as set up) | `airfoil_flow` took about 30 h on the laptop; `oscillating_cylinder` is the FVM-only moving mesh (objective 4). |
| VPM only | `vpm/01_lamb_oseen_vortex`, `vpm/02_vortex_ring`, `vpm/03_vortex_interactions` | GPU (Taichi CUDA) | Lamb-Oseen compares with Cerretelli and Williamson digitised curves in `assets/references`; the merger initial condition uses a0/b0 = 0.125. |
| VLM+VPM | `vpm/04_flat_plate`, `vpm/05_delta_wing`, `vpm/06_rotor_flow`, `vpm/07_quadcopter` | GPU | `06_rotor_flow` is the twin of the coupled rotor of objective 4; its power and blade forces are the comparison baseline. |

Done means: `allclean.sh && allrun.sh && allplot.sh` ran through from a clean directory, the
figures in `figures/` are regenerated, the README numbers come from `samples/` or
`figures/auxiliary/`, `python tests/run_isolated.py tests/tutorials` passes for the case, and
the field renders follow the gallery style. GPU memory per VPM case is modest (the Lamb-Oseen
column needs `processing_unit="CUDA"` for the particle field; the Vulkan backend leaks buffers).

## 5. Objective 3: cube flow and wing coupled cases

### 5.1 Cube (`tutorials/coupled_fvm_vpm/02_cube_flow`)

State: the stored coupled run reached 33 s **with solver 613b5898**, whose direction-aware
coupling condition lowered the cylinder drag by 4 %; every number from it and from its replays is
invalid for comparison. Earlier October runs stopped at t = 40.6 s on the particle-state limit
Lambda > 1 (far-wake strain) at a particle step of 0.04 s; the coupled backup format 16 lets a run
continue with another number of FVM substeps when the backup's step divides by it.

To do, from zero on the current `development` solver:

1. `(cd reference_flow && ./allrun.sh)`: FVM-only reference, CPU, to its end time.
2. `./allrun.sh`: coupled run on the GPU (about 0.9 M particles, 3.7 to 5.7 GB of GPU memory with
   the October FMM memory fixes). If the Lambda stop recurs, the author's October decision was to
   continue from the backup with N = 2 (particle step 0.02 s); record it in the README. Never
   lower the limit.
3. `./allplot.sh` and `./allplot_fields.sh`; README numbers from `samples/`.
4. The `cube-optimisation` branch (section 2.1) carries the selective eddy viscosity constant 0.5
   and the strain-cap exclusion of the transfer region, plus the screen queue
   (`.claude/cube_screens`, laptop only: three 20 s screens, N = 5 with 6 and 1 interface
   iterations and with Courant 0.5). Decide with the author whether to merge it before the cube
   run; if merged, the cylinder study is not affected because it imports the frozen export.

### 5.2 Wing (`tutorials/coupled_fvm_vpm/03_naca4412_flow`)

State: the oblique-freestream setup blew up at the upstream spanwise edges (mixed condition with
tangential inflow); the case was moved to the freestream frame (stream along x, section pitched
10 degrees about mid-chord, box (-1.6, 2.72) x (-1.28, 1.24) x (-3.6, 3.6)), whose FVM-only box
did not drift. The stored coupled run reached 4.92 s and is from before `3f771630`; no reference
run exists (`reference_flow/` has no `samples/`).

To do: reference (CPU) and coupled run (GPU) from zero to the end times in the two `setup.py`,
`allplot.sh`, README from files. Stability of the mixed condition on the wing is unverified
since `3f771630`; if the run stops on `ParticleStateError` or the boundary-flux check, report it
with the log, do not relax the check.

## 6. Objective 4: moving mesh, FVM only and coupled

### 6.1 FVM only

`tutorials/fvm/oscillating_cylinder`: rigid whole-mesh ALE motion (`openonda/fvm/motion`,
kinematics in `openonda/core/kinematics.py`). Tests: `tests/fvm/test_mesh_motion.py`. Re-run the
case (objective 2) and confirm the tests pass on the current `development`.

### 6.2 Coupled, work of the moving-mesh session committed as is

A separate session worked on the three coupled moving cases in this checkout; its uncommitted
state at handoff is part of this commit and is **untested as a whole**. Its own report at
13:40: the short CPU check of case 04 (`--reduced-velocity 6 --end-time 0.4 --compute-device
cpu`) had meshed 20k cells and was in its first coupling step. Files it changed or added:

- Solver: `openonda/fvm/motion/elastic.py` (new, elastic mount: spring-damper driven by the wall
  forces), `openonda/fvm/sampling/motion.py` (new), and edits in `openonda/fvm/__init__.py`,
  `fvm/config/output.py`, `fvm/core/solver.py`, `fvm/io/backup.py`, `fvm/motion/__init__.py`,
  `fvm/motion/mesh_motion.py`, `fvm/motion/rigid.py`, `fvm/sampling/{__init__,executor,forces}.py`;
  tests `tests/fvm/test_mesh_motion.py`, `tests/coupler/test_moving_mesh_coupling.py`; docs
  `docs/fvm.md`, `docs/coupling.md`, `docs/tutorials.md`.
- Case 04 rewritten as the paper's case: `tutorials/coupled_fvm_vpm/two_cylinders.pdf` is Martins
  and Avila (2019), **one** rigid cylinder elastically mounted with **two degrees of freedom** (x
  and y), m* = 2.4, zeta = 0.5 to 10 %, SST k-omega RANS. The session targets the "Low Re,
  zeta = 0.5 %" response contour at reduced velocities 4, 6 and 8 (Re = 250 V*, C_A = 1), one
  `allrun.sh` line per V*, with the paper's contour digitised from its figure 12 or table 5 unless
  the author supplies the raw response data. New assets: `response.py`, `plot_response.py`,
  `plot_histories.py`, `render_vorticity.py`; deleted the heave-only scripts.
  **Open question for the author:** "two elastically mounted cylinder" was read as the paper's
  two-degree-of-freedom single cylinder; if two cylinders (tandem or side by side) were meant,
  the spacing and arrangement are needed.
- Case 05 `05_pitching_airfoil`: unchanged by the session (NACA 0012, +-10 degrees about the quarter
  chord, f = 0.2 Hz, Re = 1000, h = 0.025, FVM step 0.005 s, N = 5, end 30 s, statistics 15 to
  30 s). A smoke run to 0.5 s was clean; the coupled GPU run and the reference were never run.
- Case 06 `06_rotor_flow` (new, untested): the three-bladed turbine of `tutorials/vpm/06_rotor_flow`
  with meshed NACA 0012 blades generated from `vpm/06_rotor_flow/assets/blade.json`
  (`assets/blade_surface.py`), R = 6 m, U = 7 m/s, tip-speed ratio 7, FVM disk
  (-3, 3) x (-9, 9) x (-9, 9) m rotating with the rotor, cells 0.2 m at the interface and 0.05 m at
  the blades, FVM step 0.001 s, coupling 0.006 s, end 15 s; assets for blade forces, rotor
  performance and wake renders. Purpose: compare power and blade forces with the VLM+VPM rotor,
  same geometry and operating point, outputs with the same names and units.

To do, in order:

1. Run `python tests/run_isolated.py tests/fvm/test_mesh_motion.py tests/coupler/test_moving_mesh_coupling.py`
   and the tutorial tests for cases 04 to 06; fix what the session left broken. At handoff three
   tests in `test_moving_mesh_coupling.py` failed already on `3f771630` (`CoupledSolver.n_steps`
   removed, `EverySteps` schedules rejected since `613b5898`).
2. Case 04: short CPU check of each V*, then the three coupled GPU runs (25 natural periods each,
   25 V* s) and the FVM-only references; `allplot.sh`; overlay on the paper's contour; vorticity
   renders of FVM and particles at the turning points.
3. Case 05: reference and coupled run to 30 s; `allplot.sh`; README.
4. Case 06: mesh check first (blade surface closed at root and tip, cell counts, memory), a short
   CPU check, then the coupled GPU run to 15 s and the VLM+VPM twin re-run (objective 2) on the
   same commit; a comparison table of power and blade forces in the case README, both from
   `samples/`.
5. `docs/fvm.md` and `docs/coupling.md` sections on moving meshes checked against the code.

## 7. Lane plan for the 132 GB machine

Lanes are independent and can start together; within a lane the order matters. GPU lanes share
the card(s): one coupled cylinder run about 2.8 GB, the cube 3.7 to 5.7 GB, VPM tutorials less.

| Lane | Device | Content | Order inside the lane |
| --- | --- | --- | --- |
| A1 | CPU | matched references `g` (three meshers at once), section `p` reference if missing | start first; the box series switches to matched references when all exist |
| A2 | GPU slots | cylinder study sections `n_interval i_transfer inset k b1 p v_side f_side` | as listed; `COMPLETE.sh --slots N` does the scheduling |
| A3 | CPU | analysis and figures after each finished section; findings note at the end | `python -m assets.interface_sensitivity`, `bash allplot_sensitivity.sh` |
| B1 | CPU | all FVM-only tutorials, each with its own MPI ranks | all at once; `airfoil_flow` longest |
| B2 | GPU | VPM and VLM+VPM tutorials | two or three at a time |
| C1 | CPU | cube reference, wing reference | both at once |
| C2 | GPU | cube coupled, then wing coupled | cube first (largest) |
| D1 | CPU | moving-mesh tests, case 04 CPU checks, case 05 and 06 references and mesh checks | tests first |
| D2 | GPU | case 04 three runs, case 05, case 06 | after D1's checks pass |

Rules for the agent: one solver commit for all runs of one comparison (note the commit in each
README); never two meshers of the same case at once; record for every run the command, the
commit, the device and the wall time in the case README or `figures/auxiliary`; commit the
scripts and figures, never the data; push `development` at the end of each lane.

## 8. Known failures and open decisions at handoff

- `tests/tutorials/test_tutorial_style.py` fails on `.is_file()` / `.exists()` in case 01's
  `assets/window_uncertainty.py`, `assets/study_status.py`, `assets/plot_sensitivity.py` (study
  tooling against the tutorial rule; author's call: exempt the study assets or rewrite them).
- `tests/tutorials/test_continuation.py` collection breaks on `01_cylinder_shedding_flow/handoff_2026-11-01/setup.py`.
- `tests/coupler/test_moving_mesh_coupling.py`: three failures since `613b5898` (section 6.2).
- The moving-mesh session's work is committed untested as a whole (section 6.2).
- `cube-optimisation`: merge or drop (section 5.1).
- Wider-box lift deficit of the cylinder study (section 3.3).
- Two cylinders versus two degrees of freedom (section 6.2).
- Mesher stall at h = 0.0566 with slab half-depths of 6 and 8 cells (7 works); root cause open.

## 9. Environment

```bash
source install.sh
```

creates the `OpenONDA` conda environment and runs the install checks; `docs/installation.md`
covers CUDA. Set `CUDA_PATH` to the CUDA root in every shell that runs particles (without it the
slab image sums fall back to the CPU, about 15 times slower, which rule 6 forbids silently).
Use `MPLBACKEND=Agg` for unattended plotting. Tests: `python tests/run_isolated.py [paths]`.
