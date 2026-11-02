# Handoff: complete the coupled-cylinder sensitivity study (OpenONDA)

You are taking over a numerical study that must produce publication-grade, evidence-based results. Read this file completely before running anything. Every statement below is marked **[verified]** (checked by running code or reading data, with the evidence named) or **[not verified]**. Keep that discipline in everything you report.

## 1. Goal

For the hybrid FVM–VPM solver OpenONDA, on the planar cylinder at Re = 150 (`tutorials/coupled_fvm_vpm/01_cylinder_shedding_flow`), produce:

1. **Isolated error sources.** Split the error of each statistic (mean drag $\bar C_D$, lift amplitude $\hat C_L$, Strouhal number $St$) into the FVM-only discretisation error $e_\mathrm{FVM}$ and the coupling error $e_\mathrm{coupling}$. Then attribute $e_\mathrm{coupling}$ to the parameter groups: FVM domain (D), interface (I), coupling interval (T), particle side (V), FVM side (F).
2. **A full one-at-a-time sensitivity analysis.** Every comparison is reported with an error bar from the measured sampling floor. A parameter counts as having an effect only if it moves the error by more than 2σ of that floor.
3. **A calibrated coupling interface.** Recommended values for the interface parameters (κ_max, transfer removal threshold, moment-correction cap, inset, blending depths, coupling interval/iterations), each justified by the runs above.
4. **Figures and tables**, regenerated from the stored runs by the scripts in `assets/`.

Context documents, in the claude.ai project "OpenONDA" or as copies in this folder:
- `cylinder_sensitivity_redesign_2026-10-31.md`: why the old study was insufficient, and the redesigned matrix. **Read sections 1–3.**
- `assets/coupling_findings.md` (in the case): results of the 19–25 October study, made with the **older** solver version.

## 2. Machine and code

- Laptop `flavio-work-laptop`, Linux, one RTX 3060 Laptop (6 GB; about 5.78 GB free when idle), conda env `OpenONDA` (Python 3.11, taichi 1.7.4).
- Repository: `~/Projects/OpenONDA`, branch `development`. Commits are made in the user's name: `Flavio Martins <flavio.martins@outlook.com>`. Keep everything on `development`; create no other branches.
- Case: `~/Projects/OpenONDA/tutorials/coupled_fvm_vpm/01_cylinder_shedding_flow`. Run every command from there.
- **[verified]** The `openonda/` source on the laptop matches GitHub `development` at `613b589` (all 287 `.py` files equal in size; none newer than that commit).

## 3. Current state (1 November 2026, 10:30)

### 3.1 Stored data
- `sensitivity/`: 36 coupled runs to t = 55 s and 11 FVM-only references from the October study, made **before** the interface fixes of 31 October. These were the direction-aware coupling boundary condition and the coupling-grade box mesh; see `naca4412_spurious_vorticity_2026-10-30.md`. **These coupled runs must not be compared with new runs.** The FVM-only references remain valid: the fixes do not touch a mesh without coupling patches.
- Runs on 31 Oct – 1 Nov: references `reference_fw4.4` (and possibly `reference_dt0.004_fw4.4`) ran. Check each one's `run.log` and that its `samples/forces_history.csv` reaches 55 s. `box2.4_sub8_inset8.75_h0.08_dtf0.004_ic0.04` finished, but with the slab mesh from before the fix in 3.2, so it must be rerun.
- **[verified]** About 45 new runs stopped within 2 s on 1 November (`sensitivity/COMPLETE.log`, their `run.log`s):
  - Most hit `ValueError: A surface lies 0.04 m from the exact box side zmin; coupling meshes need at least 8 background cells`.
  - The two continuations of section `h` hit `numerical configuration mismatch at coupler.transfer_inset_spacings` instead: an old-version backup refused by the new code.
- A `COMPLETE.sh` process may still be running on the laptop. It waits forever for 5800 MiB of GPU memory, more than the card has. **Stop it first:** `pkill -f COMPLETE.sh; pkill -f run_sensitivity_study.sh`, then confirm with `nvidia-smi` that no `python setup.py` is left on the GPU.

### 3.2 Fixes prepared in `handoff_2026-11-01/` (next to this file) and not yet applied to the case

| File | Change | Status |
|---|---|---|
| `setup.py` | Meshing slab half-depth `min(16, floor(0.5/h))·h`, so the long cylinder (STL z ∈ [−0.6, 0.6] m) crosses the z faces. Crossed faces are not exact coupling sides and need no clearance. Also carries the new study inputs of 31 Oct (`--transfer-removal-threshold`, `--moment-correction-fraction`, `--vpm-integrator`, `--vorticity-floor`, `--max-courant`, `--fvm-outer-correctors`). | **[verified]** All 71 coupled variants listed in `assets/run_sensitivity_study.sh` build through `setup.py`'s own argument handling up to `CoupledSolver` with no error; for every one, z sides are crossed and x/y sides exact. **[verified]** The baseline meshed (406 s) and ran 5 coupled intervals to t = 0.2 s on CPU, exit 0. It wrote backup `coupled_000005` and 6 force rows; C_D at t = 0.2 s = 1.633, which is a startup transient and not a result. |
| `assets/run_sensitivity_study.sh` | A run directory without a coupled backup receives fresh copies of `setup.py`/`assets`; the old `cp --update=none` kept the broken setup in failed directories. Three runs in a row that stop within 600 s halt the study (exit 3, `sensitivity/HALTED`). The h = 0.0283 m level of section `a` runs only with `RUN_FINE=1`. Section `g` adds `reference dt0.004_fw4.4`. | **[verified]** With a fake solver that fails immediately: halts after 3 failures, across sections. |
| `COMPLETE.sh` | Archives coupled runs from older solver or mesher versions to `sensitivity/archive_pre_2026-10-31/` (moved, never deleted). Runs phase 1 (new design), then phase 2 (October families rerun in this version), then `window_uncertainty.py` and `allplot_sensitivity.sh`. `--phase`, `--with-fine` and `--dry-run` select what runs. | **[verified]** Archive and halt logic, with the fake solver. **[not verified]** Its date-based archive rule on the real `sensitivity/` tree: run `--dry-run` and check the list by hand. |
| `assets/plot_sensitivity.py`, `assets/error_analysis.py`, `assets/window_uncertainty.py` | Already applied to the case on 31 Oct. They parse the new run-name suffixes (`_tau _mc _rk4 _vf _co _oc`). `window_uncertainty.py` measures the sampling floor. | **[verified]** The suffix regex was tested on 15 names; `window_uncertainty.py` ran on the 100 s reference. **[not verified]** `allplot_sensitivity.sh` end to end on the new run set. |

**[not verified]**, and you must check each of these before the long campaign:
1. Every new flag at once in one short run, with the Courant controller included: `--fvm-substeps 2 --fvm-time-step 0.02 --max-courant 0.5 --vpm-integrator rk4 --transfer-removal-threshold 0.2 --moment-correction-fraction 0.3 --vorticity-floor 0.001 --fvm-outer-correctors 3 --end-time 0.16`. This test was interrupted.
2. That each flag actually reaches the solver. Grep `solution/coupler_metadata.json` and `solution/vpm_metadata.json` for the value set.
3. The plotting chain on a mix of old references and new runs.

## 4. Procedure

### Step 0: Apply and smoke-test (about 1 h; required before anything long)
1. Stop running study processes (3.1).
2. Copy `handoff_2026-11-01/setup.py`, `handoff_2026-11-01/COMPLETE.sh` and `handoff_2026-11-01/assets/run_sensitivity_study.sh` into the case. Run `chmod +x COMPLETE.sh assets/run_sensitivity_study.sh`.
3. Run `bash COMPLETE.sh --dry-run` and check the archive list. It should contain every coupled run whose forces predate 31 Oct 12:00, plus `box2.4_sub8_inset8.75_h0.08_dtf0.004_ic0.04`, and no reference.
4. Smoke tests on the **GPU**, each in a scratch directory outside `sensitivity/` (for example `/tmp/smoke_*`, holding a copy of `setup.py` and `assets/`):
   - baseline: `python setup.py --end-time 0.4`
   - every-flag run from 3.2 item 1
   - one variant per cell size: `--cell-size 0.08 --fvm-time-step 0.004 --interface-cell-size 0.04 --fvm-substeps 8 --end-time 0.1`, and the same with `--cell-size 0.056569`
   - `--particle-spacing 0.08 --end-time 0.4`
   - `--box-margin 4.0 --end-time 0.2`
   Each must exit 0, write `samples/forces_history.csv`, and log a boundary mismatch well below the 0.2 U∞ warning. Record the wall time per coupling interval, which gives the campaign estimate. Fix any failure at its root cause and rerun all smoke tests.
5. Commit the applied changes on `development` under the user's name.

### Step 1: Campaign
- Start `nohup bash COMPLETE.sh > sensitivity/COMPLETE.nohup 2>&1 &`. Expected cost: a coupled run at h = 0.04 m takes on the order of 3 h to t = 55 s at N = 5, longer at small N. Phase 1 has about 45 runs and phase 2 about 30, so plan for **one to two weeks** of GPU time. Measure this from the smoke tests and the first runs; do not assume it.
- Monitor at least every 12 h:
  - `sensitivity/COMPLETE.log` and `sensitivity/HALTED`
  - every new run's `run.log` tail
  - per run: `solution/coupler_diagnostics.jsonl`, checking `fvm_boundary_data.mean_velocity_mismatch` (expect a few 10⁻³ U∞), `coupling_iteration.converged`, `transfer.strength_ratio`, and that particle counts are not growing without bound.
- A run that stops is diagnosed from its log, fixed at the cause, and resumed. Never delete a run with backups to "start clean" without recording why.
- A physically wrong but running case still counts as a failure; catch it in the diagnostics. Examples: a mismatch growing exponentially, or C_D outside 1.2–1.45 after t = 20 s.

### Step 2: Analysis (only from stored files)
1. Run `python -m assets.window_uncertainty` on the 100 s runs of section `h` and the 100 s reference. This gives σ per statistic (expected magnitude from the reference: C_D σ ≈ 0.07 %, St ≈ 0.01 %; the lift amplitude drifts +0.2 % from the 40–55 s window to saturation).
2. Compute e_coupling against the fine-wake reference `reference_fw4.4` (same Δt), and e_FVM against the extrapolated FVM-only value. The extrapolation uses the existing references; `assets/error_analysis.py` does it.
3. For each group D, I, T, V, F: a table of e_coupling ± floor per statistic, with the per-transfer indicators in the same row (κ median, fraction of transfers at κ_max, mean boundary mismatch, removed-strength fraction). Then a normalised sensitivity Δe per factor of two of the parameter, giving the ranking.
4. Mechanism test for the interval error (section `n_interval`): e_coupling against Δt_VPM at κ_max = 1.8 and 3. If the curves separate, the strength limiter is the mechanism. If they coincide, it is not, and the study must say so.
5. Calibration: for each interface parameter, the recommended value and the run evidence for it. When the data do not resolve an effect, write "no effect resolved above the floor" — that is a valid finding.
6. Report an order of convergence only when all levels are admissible (κ median ≤ κ_max, Λ < 1) and the convergence is monotone; otherwise report the level values.
7. Run `bash allplot_sensitivity.sh`. Fix the plotting scripts where the new run set breaks them; do not hand-edit figures.

### Step 3: Deliverables
- `figures/` and `figures/auxiliary/*.csv`, regenerated.
- `assets/coupling_findings.md`, rewritten from the new data only. Every number in it must be traceable to a CSV row and run directory. List the runs that failed and why.
- A results summary for the user, with the main findings, the calibrated interface and the open limitations, each claim pointing to its evidence.

## 5. Rules
1. **No fabricated or estimated numbers.** Every number comes from a file you read in this campaign; cite the file. If a run has not finished, its result does not exist.
2. **One solver version per comparison.** Never mix archived and new coupled runs in one table or figure.
3. **Do not relax** a tolerance, limit, check or the halt rule to make runs pass. Do not change solver defaults; in particular `max_coupling_iterations`, tolerances, `max_strength_ratio` and `boundary_mismatch_limit` change only as the parameters under study.
4. A statement is "verified" only after a command or data read that could have shown it false.
5. Report failures plainly: what failed, the log line, the cause, the fix.
6. The finest runs need about 6 GB (h = 0.0283 m) or 12 GB (h_VPM = 0.02 m, section `f`) of GPU memory and do not fit this card. Either run them on a larger GPU or state them as not run. Do not let them fall back silently to CPU image sums, which are about 50× slower.

## 6. Files in `handoff_2026-11-01/`
- `HANDOFF.md`: this file
- `setup.py`, `COMPLETE.sh`, `assets/run_sensitivity_study.sh`: fixed versions, to be applied in step 0
- `cylinder_sensitivity_redesign_2026-10-31.md`: the redesign, with its evidence
