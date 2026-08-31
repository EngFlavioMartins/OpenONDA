# Independent Tutorial Refactoring Certification

## A. Executive conclusion

**NOT CERTIFIED**

The local tutorial implementation is now structurally consistent and its known
API breakages were corrected. All 18 maintained case directories contain the
four standard entrypoints, all shell scripts parse, private VPM imports are
gone, FVM cases use the public factory/lifecycle, and an isolated installed-API
FVM--VPM run including backup/restore passes. The complete 18-case
clean/run/plot/clean matrix was not executed, however. Formal certification is
therefore withheld.

Audited implementation commit:
`d725bef788a95fca71f2d39d126427c74cd14fa7` on Linux CPU.

## B. Complete scope inventory

`Structure` means `setup.py`, `allrun.sh`, `allplot.sh`, and `allclean.sh` are
present and shell syntax passes. `CLI` lists deliberate experiment selectors,
not routine framework parameters.

| Tutorial | System | Structure | CLI | Runtime evidence |
| --- | --- | --- | --- | --- |
| `coupled_fvm_vpm/cube_flow/reference_flow` | FVM LES reference | PASS | none | setup/import tests and installed FVM smoke; full named run not executed |
| `coupled_fvm_vpm/cube_flow` | coupled FVM--VPM cube | PASS | none | stale `VPMCase.output` preflight fixed; setup tests pass; full run not executed |
| `coupled_fvm_vpm/cylinder_shedding_flow/reference_flow` | four-grid FVM cylinder study | PASS | `--dx`, `--case-name` (study selectors) | setup/tool tests pass; a concurrent very-coarse run completed, but its generated files were not used as final audit evidence |
| `coupled_fvm_vpm/cylinder_shedding_flow` | coupled cylinder shedding | PASS | none | setup/import tests pass; production run not executed by this audit |
| `coupled_fvm_vpm/naca4412_flow` | coupled finite-span airfoil | PASS | none | setup/import and native coupled smoke; named production run not executed |
| `fvm/airfoil_flow` | FVM airfoil | PASS | none | public factory/run path imports; named run not executed |
| `fvm/boundary_layer` | FVM flat-plate boundary layer | PASS | none | public factory/run path imports; named run not executed |
| `fvm/cube_flow` | FVM cube | PASS | none | public factory/run path imports; named run not executed |
| `fvm/cylinder_ibm` | FVM immersed cylinder | PASS | none | public factory/run path imports; named run not executed |
| `fvm/step_profile` | FVM inlet-profile step | PASS | none | public factory/run path imports; named run not executed |
| `fvm/taylor_green` | FVM Taylor--Green | PASS | none | setup and numerical component tests; named run not executed |
| `vpm/delta_wing` | VPM--VLM delta wing | PASS | none | public setup import; named run not executed |
| `vpm/flat_plate` | VPM--VLM flat plate | PASS | comparison/angle selectors | public setup import; named run not executed |
| `vpm/lamb_oseen_vortex` | VPM viscous schemes | PASS | circulation, scheme, case selectors | RWM/statistical and qualification components pass; full matrix not executed |
| `vpm/quadcopter` | VPM--VLM multicopter | PASS | none | public setup import; named run not executed |
| `vpm/rotor_flow` | VPM--VLM rotor | PASS | none | public setup import; named run not executed |
| `vpm/vortex_interactions` | four leapfrogging-ring stabilization variants | PASS | `--case` (four model variants) | five construction/configuration tests pass; 1,200-step production variants not executed by audit |
| `vpm/vortex_ring` | isolated vortex ring | PASS | declared comparison selector | stale plot targets fixed; named run/plot not executed from clean state |

## C. Requirement audit

| Requirement | Status | Evidence |
| --- | --- | --- |
| Standard root structure | PASS for source files | 18/18 have all four entrypoints; every `.sh` passes `bash -n`; vortex-interaction cleanup includes its generated root manifest. |
| Physics-first `setup.py` | PASS locally | Physical/numerical constants, geometry, models, samplers, construction, and integration remain visible in each setup. |
| Public API only | PASS locally | No tutorial setup imports `source.solvers.vpm`; VPM objects come from `openonda.vpm`; FVM construction uses `create_fvm_solver`. |
| One construction/lifecycle pattern | PASS locally | FVM cases use `FVMSetup` plus `create_fvm_solver` and `solver.run()` except explicit diagnostic/study control where the case requires it. VPM cases use `VPMCase` plus `VPMSolver.run()`. |
| No hidden numerical environment configuration | PASS | Cube-reference physics/mesh/time controls are explicit constants; VPM compute device is typed configuration. |
| Minimal CLI surface | PASS with justified selectors | Remaining arguments select published comparison cases, circulation/schemes, angle, or grid resolution. No tutorial exposes dt/backend/MPI/logging as routine CLI knobs. |
| Minimal run/plot/clean scripts | PARTIAL | Most scripts are short. Coupled cube still has a substantial reference preflight and allows `OPENONDA_PYTHON`/`OPENONDA_CUBE_REFERENCE`; this is execution plumbing, not hidden solver physics, but exceeds the preferred minimal shell. |
| Output ownership | PASS locally | FVM uses immutable `RunSchedule`; VPM uses framework `Samplers`/`Backup`; cylinder reference uses solver-owned visualization/backup cadence and `solver.run()`. |
| Plot entrypoints synchronized | PASS statically | Vortex-ring nonexistent `.p` call and duplicate invocations were removed; vortex-interaction plot paths match their sampler outputs. |
| Clean safety/idempotence | PARTIAL | Scripts use explicit case-local generated paths, and vortex-interaction cleanup removes its generated manifest. Full repeated cleanup was not run for all cases. |
| Installed public workflow | PASS | `python scripts/validate_native_tutorials.py --compute-device CPU` runs isolated FVM--VPM coupling, output, coupled backup, restore, and artifact checks. |
| Complete named lifecycle matrix | FAIL | The audit did not execute every long-running named case and plot from a disposable clean copy. |
| Cross-platform/backend tutorial matrix | FAIL | Only Linux CPU was available. |

## D. Representative information flow

### FVM tutorial

```text
visible physical/numerical constants
→ FVMSetup + typed subconfiguration + mesh object
→ create_fvm_solver
→ solver.run (accepted steps and RunSchedule events)
→ solution/ and samples/
→ assets plotting/postprocessing
```

### VPM tutorial

```text
visible analytical flow/distribution and numerical constants
→ VPMCase(Numerics, initial_conditions, Backup, Samplers, RunPlan)
→ VPMSolver(case).run()
→ accepted-step OutputManager dispatch
→ solution/<case>, samples/<case>, run_manifest.json
→ assets plotting/postprocessing
```

### Coupled tutorial

```text
visible FVMSetup + VPMCase + CouplerSetup
→ public FVM/VPM construction
→ create_coupler(...).run()
→ accepted post-transfer sampling + coupled checkpoint manifest
→ case-specific comparison/plots under assets/
```

No tutorial callback, sampler, logger, or plot helper owns authoritative solver
state.

## E. Numerical and behavioral evidence

| Evidence | Result |
| --- | --- |
| Stable complete suite | 416 passed, 0 failed/errors/skipped in 479.43 s |
| Vortex-interaction tutorial tests | 5/5 pass, including exact supported splitting/remeshing case configuration |
| Native installed FVM--VPM workflow | PASS on CPU, including 6 FVM substeps, 2 VPM steps, scheduled sample, coupled backup, and restore |
| Coupler interpolation | affine max error `1.11e-16`; refinement orders `2.23279`, `2.05562` |
| FVM manufactured LSQ gradient | minimum observed order `1.96660` |
| VPM Gaussian/treecode | velocity/gradient relative L2 below `1.11e-7`; treecode order `3.06202` |
| VPM restart/reproducibility | deterministic schemes and counter-based RWM fresh-process/restart tests pass |

These component and smoke results support use of the cylinder and
vortex-interaction entrypoints. They do not constitute completed scientific
results for the requested grid study or 1,200-step leapfrogging runs.

## F. Immediate run guidance for priority cases

The coupled cylinder entrypoint runs one coupled case:

```bash
cd tutorials/coupled_fvm_vpm/cylinder_shedding_flow
./allrun.sh
```

The four-resolution grid study is a different entrypoint:

```bash
cd tutorials/coupled_fvm_vpm/cylinder_shedding_flow/reference_flow
./allrun.sh
```

`tests/tutorials/test_vortex_interactions.py` is a fast regression test and
does not generate leapfrogging results. Run the baseline or all variants with:

```bash
cd tutorials/vpm/vortex_interactions
python -u setup.py --case leapfrog_les
# or
./allrun.sh
```

The baseline-only command avoids spending time on splitting/remeshing variants
when the immediate requirement is the main leapfrogging trajectory.

## G. Remaining findings

### Blockers to formal certification

1. All 18 clean/run/plot/clean lifecycles and advertised variants have not been
   executed and archived.
2. Linux/macOS, f32/f64, rank-count, direct/accelerated, and maintained GPU
   equivalence is not established.

### Non-blocking for the two priority runs, but still cleanup debt

- Coupled cube `allrun.sh` still contains more reference/preflight shell
  infrastructure than the preferred tutorial philosophy.

## H. Changes made

- Replaced private tutorial imports with public `openonda` facades.
- Migrated FVM tutorials to factory construction and framework `run()`.
- Made compute/time/output controls explicit typed configuration.
- Corrected coupled cube stale configuration access and diagnostic backup
  replacement.
- Corrected vortex-ring plotting targets and duplicate calls.
- Updated vortex-interaction plotting/setup behavior and its regression tests.
- Replaced the removed comparison with supported splitting-only,
  remeshing-only, and combined stabilization variants plus terminal-run checks.
- Added installed-API validation and canonical API/static gates.

The implementation and stale generated-baseline cleanup are committed as
`d725bef788a95fca71f2d39d126427c74cd14fa7`.

## I. Final statement

The tutorial source is committed and the two priority entrypoints are ready to
run, but the tutorial project is not formally certified because the required
full lifecycle/platform matrix is incomplete.
