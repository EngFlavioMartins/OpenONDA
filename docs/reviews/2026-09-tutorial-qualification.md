# Tutorial qualification — 2026-09-08

The complete tutorial collection is **not yet qualified**. Across all 20 catalog
entries and 66 run scenarios, 52 bounded runs passed, 10 failed numerical gates,
and four did not finish meshing within their execution limits. No commit was
made: the requested condition was successful tutorial verification first.

All existing work remains in the working tree. The earlier
[cleanup report](2026-09-tutorial-cleanup.md) contains every `allrun.sh` and
`allclean.sh` in full.

## Data protection and execution scope

Every integration, plot, and cleanup check used disposable cases below
`/tmp/openonda-tutorial-qualification-snaju9ml`. Original case results were never
used as output destinations. SHA-256, size, and timestamp checks cover all
**19,515 pre-existing files, totaling 9,007,557,488 bytes**. All simulation
outputs remained unchanged and no file disappeared. The final scan found one
changed macOS directory-metadata file,
`tutorials/coupled_fvm_vpm/cylinder_shedding_flow/reference_flow/solution/.DS_Store`;
the other 19,514 files retained their original hashes and timestamps. The
earlier preservation scan had also matched that directory-metadata file.

The installer was exercised against a source snapshot assembled from Git's
tracked and non-ignored files. Normal installations into disposable virtual
environments were verified outside the checkout with isolated Python imports.
These environments shared the machine's installed numerical dependencies;
they were not editable OpenONDA installations. The existing machine environment
was also refreshed using `python install.py`.

This is an integration qualification, not completion of the long research
campaigns. Except for the complete Taylor–Green and uniform-flow examples, the
probes requested two accepted VPM/FVM steps or two coupling steps. They kept the
authored geometry, mesh resolution, particle capacity, physical inputs,
integrators, time-step rules, and health limits. Diagnostic and backup output
was requested every step to exercise the consumers within the short run.
FVM probes used one worker; VPM probes used CPU except the six dense
stabilization variants, which used their authored `AUTO` → Metal backend.
An additional moving-plate check also passed on Metal. RWM used four realizations
per physical case without the full convergence campaign.

Passing exit codes were checked against solver metadata, including completion
status and accepted step count. This matters because a VPM health stop can
return normally. Obsolete harness errors were replaced by corrected reruns in
the results below.

## All 20 tutorials

| Tutorial | Run result | Plot pipeline |
| --- | --- | --- |
| `fvm/taylor_green` | Full 10-step run passed | Passed |
| `fvm/cube_flow` | Two steps passed | Passed |
| `fvm/boundary_layer` | Two steps passed | Passed |
| `fvm/cylinder_ibm` | Two steps passed | Passed |
| `fvm/step_profile` | Two steps passed | Passed |
| `fvm/cartesian_mesher` | Authored mesh build passed | No `allplot.sh` |
| `fvm/airfoil_flow` | Mesh build exceeded 300 seconds | Not qualified |
| `vpm/delta_wing` | Two steps passed on CPU | Passed |
| `vpm/quadcopter` | Two steps passed on CPU | Passed |
| `vpm/rotor_flow` | First step failed CFL limit | Not qualified |
| `vpm/flat_plate` | All 10 moving angles and static 0°/8° passed; eight static angles failed | Passed using available diagnostics; no completed polar claim |
| `vpm/lamb_oseen_vortex` | All 12 physics/diffusion combinations passed | Passed, including merger snapshots |
| `vpm/vortex_ring` | All four stretching/LES variants passed | Passed, including ParaView scenes and LaTeX figures |
| `vpm/vortex_interactions` | All six `setup.py` variants and all three `setup_les.py` variants passed | Passed; long-time LBM scores unavailable |
| `coupled_fvm_vpm/uniform_flow` | Unchanged `allrun.sh` passed: six FVM and two VPM steps | No `allplot.sh` |
| `coupled_fvm_vpm/naca4412_flow` | Initial transfer failed divergence gate | Not qualified |
| `coupled_fvm_vpm/cube_flow` | Mesh build exceeded 300 seconds | Not qualified |
| `coupled_fvm_vpm/cube_flow/reference_flow` | Coarse, medium, and fine grids passed | No `allplot.sh` |
| `coupled_fvm_vpm/cylinder_shedding_flow` | Corrected probe exceeded 300 seconds during meshing | Not qualified |
| `coupled_fvm_vpm/cylinder_shedding_flow/reference_flow` | Coarse, medium, and fine passed; very-fine mesh exceeded 900 seconds | No `allplot.sh` |

All **20 cleaners passed** when launched by absolute path from an unrelated
working directory. They removed disposable solution files and retained every
copied template file and every unrelated sentinel. Eleven of the 16 plotting
pipelines completed on newly generated output. The other five lack a qualified
fresh solution in this audit.

The interaction plot pipeline produced 31 PNG/PDF artifacts. Its LBM scoring
command correctly reported that the short trajectories could not support the
requested radius-versus-distance intervals. These runs also do not qualify
stabilization events scheduled after the two-step horizon.

The Taylor–Green history is byte-for-byte identical to the previous full
Taylor–Green verification. At step 10, velocity L2 error is
`1.638436274500633e-4`, relative kinetic-energy error is
`7.037134666406301e-5`, and maximum continuity error is
`9.542330803076996e-15`.

## Numerical failures that remain

| Scenario | Failure at the authored time step |
| --- | --- |
| Rotor | CFL approximately `26.4 > 1` at step 1, `dt=0.006` |
| Static plate −5° / +5° | CFL approximately `1.02 > 1` at step 1 |
| Static plate −2° / +2° | CFL approximately `1.32 > 1` at step 1 |
| Static plate −10° / +10° | CFL approximately `1.87 > 1` at step 2 |
| Static plate +12° | CFL approximately `1.90 > 1` at step 2 |
| Static plate +15° | CFL approximately `1.09 > 1` at step 2 |
| Coupled NACA 4412 | Gaussian-vorticity transfer divergence `0.2249094 > 0.08` before advancement |

A separate rotor diagnosis retained the same initial physical configuration
and compared direct induction with FMM for one CPU step. Both failed the same
gate: direct CFL `26.4443035`, FMM CFL `26.4295959`. This does not establish a
remedy, but rules out treating the failure as merely a Vulkan/FMM launch issue.
No time step, physical definition, or acceptance limit was relaxed to obtain
these results.

## Defects repaired during verification

- Required Cartesian-mesher and cylinder-reference STL inputs existed locally
  but were excluded by Git. They are now included. Six small native cfMesh
  regression fixtures were likewise ignored; their hashes match their existing
  provenance document, and the ignore rule now includes them.
- Delta-wing and quadcopter FMM tutorials now select CPU, matching the portable
  FMM choice already used by the rotor and coupled cube on macOS.
- The ring scene generator imports its JSON reader and passes explicit sampled
  colors to ParaView. The rendered particles and LaTeX color bar share those
  colors without depending on ParaView's renamed presets.
- The interaction plot entrypoints establish their local package before
  relative imports, so plain `python assets/name.py` works in copied cases.
- Closing a manually advanced VPM solver records its accepted state. The
  unchanged uniform-flow example now records VPM step 2 at `t=0.3`, rather than
  leaving its startup metadata at step zero.
- Metadata and restart fingerprints encode valid infinite configuration bounds
  as JSON strings. The splitting case retains its infinite relative threshold,
  runs, writes checkpoints, and restores compatible checkpoints. A changed
  finite threshold still fails restart compatibility validation.

No tutorial input deck gained validation gates or metadata-writing code.

## Regression evidence

The clean-copy full-suite run completed with **669 passing tests and one
environment-sensitive launcher-test failure**. That test assumed the caller
had no Taichi cache override, while the audit deliberately supplied one. It now
isolates and checks both the default cache and an explicit override.

After the subsequent fixes:

- All 21 installed-tutorial tests passed, including copied direct plot commands.
- All 31 VPM output-contract tests passed, including manual-close state and
  Python/NumPy infinite thresholds.
- All 24 existing backup tests passed; the additional infinite-threshold
  save/restore/mismatch regression passed after correcting its call to the
  documented string-path API.
- The six native fixture files are present in the clean-copy suite. No test was
  skipped. Full-suite and focused rechecks together cover the 676 tests now in
  the collection; these are separate invocations, not a claim of one fresh
  676-test run after every edit.
- Ruff and `git diff --check` passed. All 449 package/tutorial Python files were
  syntax-checked. Both isolated and machine installations passed the native
  install verifier.

The remaining upstream Taichi locale-deprecation warning does not affect the
test outcomes. ParaView also reported an unavailable optional OpenVKL device;
the ring scene renderer nevertheless completed and its PNG output was inspected.

## Evidence and pending commit

The disposable audit directory retains `qualified-results.json`, per-scenario
logs, `cleanup-results.json`, plot results, installer logs, full and focused
pytest logs, and the before/after result-file hashes. The final scenario list
contains only the authoritative reruns, while the earlier failure logs remain
available for diagnosis.

The pending combined commit would record the installer, minimal direct tutorial
launchers and physics-focused setups, solver-owned output records, documentation
and API cleanup, required reference inputs, and their regression checks. It is
withheld because ten numerical scenarios still fail and four remain unqualified.
