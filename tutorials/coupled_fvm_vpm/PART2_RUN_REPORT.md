# Part II run report

Status: in progress. This file is filled in as the runs finish; every number in it is copied from a named field of a delivered file.

## Acceptance tests of Task A

`tests/coupler/test_multi_domain_acceptance.py` (local `tests/` repository), each check followed by its planted fault. Runs: short coupled cylinder flows of 0.4 s (ten coupling intervals) with the particle solver on CUDA, from exports of the frozen commits named below.

Round-off, measured before any acceptance run: two identical one-domain case 01 runs on CUDA differ by 5.65e-7 of the largest force and 9.98e-8 of the largest strength, with equal positions and particle counts (939). The tolerances are ten times that floor, rounded up to a power of ten: force 1e-5, strength 1e-6, position 1e-6 particle spacings. Distant bodies: the force histories of each of two cylinders 200 D apart and of the single cylinder differ by at most 2e-3 of the largest single-cylinder force (stated on 18 Nov, before any run).

| Check | Measured difference | Outcome | Planted fault | Fault's difference | Fault caught |
| --- | --- | --- | --- | --- | --- |
| One domain through the new code path = single-domain coupler (commit 1cdea5cf) | force 5.576e-7, position 0, strength 9.977e-8 | passed | boundary data +1 % | force 9.513e-3, particle count differs | yes |
| Two domains in both orders | force 1.237e-6, position 0, strength 1.003e-7 | passed | transfers one after the other with boundary data refreshed in between | force 7.039e-2, particle count differs | yes |
| Two cylinders 200 D apart = single cylinder | force 5.525e-4 of the largest single-cylinder force (tolerance 2e-3); 1,890 particles against 945 for the single cylinder | passed | cylinders 3 D apart | force 5.203e-1 | yes |
| Continued from a two-domain backup = uninterrupted | force 1.998e-6, position 0, strength 1.244e-7 | passed | backup without the boundary-data correction of one domain | force 5.545e-4, strength 4.201e-6 | yes |

The first check ran from export 336f854f, the order and restart checks from 1c627a9a (identical code; it adds the case 07 cylinder surface files) on CUDA, the distant-bodies checks from 46dba6d9 (identical coupler and particle solver) with the particle solver on the CPU while the GPU was reserved for another run. The side-by-side runs use FVM boxes of 1.5 D, not the 1.2 D first stated: at 1.2 D the coupler's boundary-mismatch check stopped the single cylinder at t = 0.04 s (velocity RMS difference 0.126 of the freestream, limit 0.1); cylinders 3 D apart keep their remeshing regions four spacings apart up to 1.63 D. They hold at most 200,000 particles (about a thousand are present in these runs). The first attempt of the two-domain checks failed before any flow because the case 07 surface was not in the repository; a later attempt of the distant-bodies checks failed because the one-cylinder run used the multi-domain output layout, a fault of the test driver. Inside the test, the 200 D run on the CPU hit the test's 6 h subprocess timeout after 8 of its 10 intervals (about 41 min per interval); it was rerun outside the test with the same driver, export and specification and no timeout (5 h 48 min), and compared with the test's own difference function.

## Acceptance tests of Task B

| Requirement | Outcome |
| --- | --- |
| Case 06 mesh: positive volumes, no cell centre inside a blade | met at d4b6d67b: 797,383 cells, smallest volume 1.33e-8 m³, no fluid-cell centre inside a blade; max non-orthogonality 68.8°, max skewness 0.711, max aspect ratio 3.34 |
| Pitched tip slab meshes, with a regression test | met: `tests/fvm/test_planar_feature_constraints.py` (sharp edges on the wall; fails on the base code with 5 centres inside) and `tests/fvm/test_cfmesh_region_topology.py` (no notch faces between the base and the sides; fails on the base code with `assert [29680] == []`) |
| Meshes of cases 01–05 unchanged, or differences explained | 18a47cbd: cases 02, 04, 05 and the case 01 coupled mesh identical; the case 01 references keep their topology with vertex shifts of 1.9e-7 to 6.8e-5 m, and case 03 and its reference change at the wing tips (up to 13.6 mm), where the cap rims are now captured. d4b6d67b: cases 01, 03 coupled, 04, 05 and 07 identical to 336f854f array by array (02, the 03 reference and the 01 and 04 references not compared) |
| `python setup.py --end-time 0.03` of case 06 completes | met (CPU particle solver): exit 0 in 192.5 min; largest FVM Courant number 15.8 in the first steps, 3.50 at 0.03 s; two coupler warnings, one moment-correction fallback and a net-strength balance of 1.27e-3 against its 1e-3 warning level |
| Mesh tests | 100 passed at 18a47cbd, 104 passed at d4b6d67b |

## Runs

Coupled runs:

| Run | Commit | Flow time | Intervals | Window | Cycles | Wall time | Exit | Peak GPU memory | Record |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S1 (case 05) | 82eaf968 | 50 s | 2,000 | 20–50 s, the last six pitch periods of 5.0 s | 6 | from t = 0, mesh from the cache of its second pre-run; 124,884 s (34.7 h), 62.4 s per interval | 0 | 3,272 MiB | `S1_coupled/run_record.json`; run status "completed"; 35,124 particles at 50 s; warnings: the solid-wall exclusion moved one or two stage particles 1,812 times and accepted particles 7 times, by at most 3.224e-03 m (an eighth of the 0.025 m spacing), and the particle-set net strength changed by more than 1.0e-03 of its L1 norm in 39 intervals between steps 98 and 2,000, by at most 3.668e-03 |

The S2 B1 pre-run starts once S1's cores and enough GPU memory are free; the other runs wait for it.

S1 against its FVM-only reference, from `05_pitching_airfoil/analysis/S1/compare_S1_vs_reference_flow.json` in the 82eaf968 export, coupled minus reference with the combined standard error over the six cycles:

- mean drag coefficient 0.1465 against 0.1484, −0.0019 ± 0.0003 (−1.3 %); second drag harmonic 0.0420 against 0.0410, +0.0010 ± 0.0002;
- first lift harmonic 0.578 against 0.567, +0.011 ± 0.002 (+2.0 %), phase 33.9° against 33.8°;
- first pitching-moment harmonic 0.0724 against 0.0722, +0.0002 ± 0.0002, phase 109.5° against 110.5°;
- largest rms of the spanwise vorticity on the wake line `x2` 1.770 against 1.763, on `x4` 0.821 against 0.674 (+22 %); mean streamwise velocity on `x2` 0.982 against 0.993.

The force, moment and work columns in newtons differ by a factor of about 40 because the coupled mesh spans one 0.025 m cell and the reference 1 m; only the coefficients compare. Over the window the particle circulation residual is at most 5.3e-03 of the particle L1 norm (median 2.0e-03), and the residual of particles and FVM together at most 2.6e-02 (median 1.8e-02), from `summary.json`. Per second of flow time the coupled run took 2,494 s against 1,225 s for the reference; the state checks and samplers took 63,646 s of the coupled run's 124,687 s.

FVM-only references, each continued from its 2 % pre-run:

| Run | Commit | Flow time | Steps | Wall time | Exit | Peak GPU memory | Record |
| --- | --- | --- | --- | --- | --- | --- | --- |
| S1 reference (case 05) | 336f854f | 50 s | 10,000 | 70 min meshing, 31.5 min pre-run (200 steps), then 59,400 s (16.5 h) for 9,800 steps, 6.06 s per step | 0 | 0 MiB | `S1_reference/run_record.json`, rebuilt from its GPU log and run log after the wrapper failed at the end; "RUN COMPLETED" at 50 s; largest Courant number 0.53 in the log, no warnings |
| S4a reference (case 07) | d4b6d67b | 200 s | 25,000 | 49 min meshing, 48.7 min pre-run (500 steps), then 113,532 s (31.5 h) for 24,500 steps, 4.63 s per step | 0 | 0 MiB | `S4a_reference/run_record.json`; largest Courant number 0.38 in the log, no warnings |
| S2 B1 reference (case 04, Re 6400, V* 7.68) | 1c627a9a | 192 s | 76,800 | 66 min meshing, 81.4 min pre-run (1,536 steps), then 259,201 s (72.0 h) for 75,264 steps, 3.44 s per step | 0 | 0 MiB | `S2_B1_reference/run_record.json`; run status "completed" at step 76,800, t = 192.0 s; largest Courant number 1.86 in the log, no warnings |
| S2 B2 reference (case 04, Re 8300, V* 9.96) | d4b6d67b | 249 s | 99,600 | 61 min meshing, 185.3 min pre-run (1,992 steps), then 335,541 s (93.2 h) for 97,608 steps, 3.44 s per step | 0 | 0 MiB | `S2_B2_reference/run_record.json`; run status "completed", last field `fvm_099600.vtu`; largest Courant number 1.92 in the log, no warnings |

## Cost estimates

| Run | Commit | First 2 % | Wall time of the 2 % | Extrapolated wall time | Particles at 2 % |
| --- | --- | --- | --- | --- | --- |
| S1 reference (case 05, FVM only) | 336f854f | 1 s of 50 s | 70 min meshing (69,085 cells), then 200 steps in 31.5 min (9.45 s per step) | 26 h for 10,000 steps | none |
| S1 coupled (case 05) | 1c627a9a | stopped by the GPU guard at 0.275 s of 1 s | mesh from cache; 11 intervals, mean 51.9 s, last ten 37.0 s per 0.025 s interval | not extrapolated (stopped); about 21 h at 37 s per interval for 2,000 intervals, before particle growth | 524 at 0.275 s; 1,414 MiB GPU |
| S1 coupled (case 05), second pre-run | 82eaf968 | 1 s of 50 s | 17 min meshing (16,039 cells), then 40 intervals in 1,252 s: the first 183 s, then a median of 20.8 s and a mean of 27.4 s per 0.025 s interval (last ten 23.2 s) | 13–15 h for 2,000 intervals at 23–27 s, before particle growth | 1,125 at 1 s, about 800 more per second over the last 0.5 s; 1,476 MiB GPU |
| S2 B1 coupled (case 04, Re 6400, V* 7.68) | 82eaf968 | 3.84 s of 192 s | mesh from the cache stored beforehand (22,731 cells); 96 intervals in 6,159 s: the first 286 s, then a median of 51.9 s and a mean of 62.7 s per 0.04 s interval (last ten 72.6 s) | about 97 h for 4,800 intervals at 72.6 s, before particle growth | 2,619 at 3.84 s; 1,024 MiB GPU; the solid-wall exclusion moved particles by up to 9.777e-03 m, 98 % of the quarter spacing (0.01 m) at which the guard stops a run |
| S2 B1 reference (case 04, FVM only) | 1c627a9a | 3.84 s of 192 s | 66 min meshing (43,651 cells), then 1,536 steps in 81.4 min (3.18 s per step) | 68 h for 76,800 steps | none |
| S2 B2 reference (case 04, FVM only) | d4b6d67b | 4.98 s of 249 s | 61 min meshing (43,651 cells), then 1,992 steps in 185.3 min (5.58 s per step) | 154 h for 99,600 steps | none |
| S4a reference (case 07, FVM only) | d4b6d67b | 4 s of 200 s | 49 min meshing (69,448 cells), then 500 steps in 48.7 min (5.84 s per step) | 41 h for 25,000 steps | none |

## Deviations

- The particle solver runs on CUDA, not Vulkan: only CUDA has the fast slip-slab image queries (0.026 s per boundary evaluation against about 60 s on Vulkan or the CPU).
- The coupled simulations come from commit 82eaf968, which writes about 500 field frames per run in cases 04, 05 and 07 for videos (an output-only change asked for by the author) and includes another session's change to the particle diffusion (cb8dfa5c, a narrower viscosity search), made after the acceptance runs. The VLM rotor runs from the same commit.
- Case 06 (S3) runs the files of 82eaf968 with its `setup.py` and `README.md` from bd458a3d, an output-only change the author agreed to on 23 Nov: fields every 0.018 s over the last 9 s (500 frames, about twelve revolutions) instead of every 0.042 s over the last two revolutions (36 frames), and backups every 0.12 s instead of every 0.024 s, since the coupler keeps every backup. Later commits change the particle solver, so S3 does not run from bd458a3d itself. S3 and the VLM rotor run from an export on the 4 TB drive (`solver_82eaf968_c06_bd458a3d`), whose other files are those of 82eaf968 (checked with `tar -d` against `git archive 82eaf968`).
- The FVM-only references ran from earlier commits with the same FVM code, meshes identical array by array and identical case files: the S1 reference from 336f854f, the S2 B1 reference from 1c627a9a, the S2 B2 and S4a references from d4b6d67b.
- Case 06: the author agreed to finer trailing-edge cells. Boxes around the trailing edge alone leave the octree surface non-manifold, so the whole blade is refined from 0.025 m to 0.0125 m, with 0.00625 m boxes along the flat base from 1.5 m radius.
- Output-only changes to cases 04, 05, 06 and the VLM rotor to record what the analysis needs (motion, wake lines with vorticity, particle invariants every interval); case 05 runs 50 s as this file asks.
- Machine sharing: another session's coupled wing run held CPUs 0–3 and 12–15 and up to 7 GB of the GPU until 22 Nov, and ring runs of another tutorial had the GPU from then until 23 Nov 10:24. Part II runs use cores 4–11 and 16–23, one particle solver at a time, at most about 4 GB of GPU memory. A run from 82eaf968 starts only when 2,700 MiB stay free after its own share, and stops itself when the card has less than 1,500 MiB free (1,000 MiB for the earlier runs).
- The coupled runs whose particle frame moves start from t = 0 instead of continuing from their pre-runs: S1 and S3, where the mesh rotates, and S4b, where one of two domains heaves. At 82eaf968 a coupled backup of such a run cannot be loaded (S1, under stopped runs). The particle walls are placed at the starting pose when the coupler is built, and nothing places them again at the backup pose before the particle backup recomputes the velocities through the solid-wall guard. Loading the FVM backup first would not change this. A test confirms it on the coupler code of 82eaf968, which 895cc69c leaves unchanged: the restart of a body pitching about an upstream pivot stops while loading with `WallCorrectionTooLargeError` (3.83e-2 m against a limit of 3.125e-2 m). Development fixes this in 1b1bc56f, after the frozen commit, and the campaign keeps 82eaf968. A single translating domain, as in case 04 (S2), keeps the particle frame on the mesh and is not affected, so S2 continues from its pre-runs; the fixed cylinders of S4a and of the restart acceptance check are not affected either. A run of the first kind stopped by the GPU guard cannot resume, and is reported as stopped.
- S4b runs at 82eaf968 with two limits of its heaving domain, which moves in the inertial particle frame. Its vorticity transfer reads the inertial FVM velocity, so the walls of the heaving cylinder are not at rest in the transfer; 1b1bc56f makes it read the velocity relative to the mesh. Its transfer lattice leaves the particle diffusion grid as the mesh heaves, which costs one more M4′ remeshing per interval; development keeps this too. Its budgets and diagnostics are read with both in mind.
- The FVM meshes of cases 04, 06 and 07 were built before their coupled runs with `premesh.py` in the campaign folder, on CPU cores beside the running jobs. It calls `build_or_load_mesh` with the case's mesh and mesh path and stores the result with `save_native_mesh`, as the FVM solver does on rank 0 at its start, so a run finds the mesh by its meshing input hash ("mesh cache hit" in its log) and its wall time holds no meshing. The stored meshes: case 04 22,731 cells, case 07 19,964 cells per domain, case 06 797,383 cells.
- The acceptance tests live in the local `tests/` repository, which has no remote.

## Stopped runs

| Run | Commit | Stopped at | Message |
| --- | --- | --- | --- |
| S1 pre-run (case 05, first 2 %) | 1c627a9a | t = 0.275 s of 1 s, 11 intervals, 524 particles | GPU guard: "GPU free 951 MiB below 1000 MiB: stopping the run" (the coupled wing run of another session held 7,758 MiB) |
| S1 continued from the 1.0 s backup of its second pre-run | 82eaf968 | while loading the backup, before the first interval (41 s, exit 1) | `WallCorrectionTooLargeError`: "VPM wall correction exceeds a quarter of particle spacing; maximum=4.205220e-02 m, spacing=2.500000e-02 m. Reduce the time step." |

## Analysis checks

`part2_analysis/checks.py`, run before any analysis: all three passed (synthetic signal recovered within 0.17 %, a 2 % change reported at 17.7 combined standard errors; a run compared with itself gives zero differences; a force equal to the body velocity is 90.0° ahead of the displacement and does positive work). Rerun on 21 Nov after the changes below, with the same outcome; to be rerun at the commit that analyses the runs.

End-to-end tests on real files, before any coupled run: each finished or partial FVM-only reference served as both sides of its run (S1, S2 B1 to 60 s, S4a; S3 on the VLM rotor output of an earlier, stopped run), with coupler records of earlier short coupled runs re-timed across the window. Every comparison gives zero differences (190, 215, 346 and 353 statistics), and S4b recovers a synthetic heave of 0.2 D at phase 0. These tests found and fixed, in the analysis only:

- the circulation budget and the coupling iterations read the record layout of a synthetic test file, not the coupler's; on real records the budget would have been empty and every interval counted as one iteration;
- the strength removed by the outflow fringe was not added back to the particle circulation;
- a window of the last N periods lost one period when the end time is not a whole number of periods from t = 0 (S3: five revolutions instead of six), and a window of shedding or response cycles ran to the end of the data instead of the end time;
- S3 had no comparison force columns (the VLM rotor writes `vlm_surface_forces.csv`) and listed its wake planes as lines; S4b lacked the Strouhal numbers and the phase between the lift histories;
- the manifest lacked the run's commit, command line, configuration, mesh cell count and hash, hardware, peak GPU memory, wall time, end state and warnings, and the cost lacked the wall time per unit of flow time, the cell count, the mean particle count and the comparison runs.

A spectral peak carries its frequency resolution, not a standard error: the inverse of the segment length, a quarter of the window, which for case 07 is 0.042 Hz, about a quarter of the shedding frequency. Frequencies with a standard error from the cycle scatter come from the mean period of the cycles: the Strouhal number of each cylinder from its lift up-crossings and, for S2, the cross-flow frequency ratio. S2 statistics with per-cycle values also carry a block-bootstrap 95 % interval over cycles.
