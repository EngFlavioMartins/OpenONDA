# Solver audit verification

This report records verification performed on 15 September 2026. Its test
counts and case limits describe that inspection, rather than a qualification
of subsequent source changes.

## Numerical and lifecycle contracts

| Area | Current behavior | Regression evidence |
| --- | --- | --- |
| VLM dense solve | Publish a matrix and its LU factorization together after successful factorization. A singular system raises without changing the circulation or poisoning a retry. | `tests/vpm/test_vlm_qualification.py` |
| Particle diagnostics | Both pair-sum loops use active population. Device accumulators, result fields and downloads honor f32/f64 accumulation precision. | `tests/vpm/test_particle_diagnostics.py`: analytical Gaussian values after removal, replacement and an empty cloud |
| Sampling schedules | Reject nonfinite times, invalid ranges and booleans during construction. Final output uses the explicit final-only schedule contract. | `tests/vpm/test_sampling_schedules.py`, `tests/vpm/test_output_contracts.py` |
| Restart | Require the current saved schema and explicit numerical settings. Reject mismatches before state mutation; preserve current physical and output identities. | `tests/vpm/test_backup_storage.py`, `test_vlm_coupled_restart.py`, `test_vlm_field_identity.py`, `tests/coupler/test_coupled_backup.py` |
| Field sampling | Query the solver-owned field directly. Samplers do not project velocities onto a separately reconstructed body surface. | `tests/vpm/test_output_contracts.py`, `tests/tutorials/test_vortex_core_sections.py` |
| Reporting | Attached boundary solvers use the VPM logging owner. FVM timers keep nested phases independent between owners. | `tests/vpm/test_logging_cadence.py`, `tests/fvm/test_logging.py` |
| Public configuration | One spelling, `n_nonorthogonal_correctors`, governs pressure corrections. Exports expose implemented configuration. | FVM nonorthogonal-pressure, time-step and restart tests |
| Documentation | Per-density energy, helicity and enstrophy have units m⁵/s², m⁴/s² and m³/s². Linear-solver docs describe actual host transfers and residual checks. | Signature, implementation and dimensional review |

VLM surface exports and loading CSVs consistently name the dimensionless
circulation estimate `circulation_pressure_jump_proxy`. Dimensional normal loads
remain distinct. Saved-state visualization reads recorded fields without
advancing the solver or fabricating missing measurements.

The vortex-interactions setup declares its four supported methods directly.
Plotters use those exact names and read only the requested run. PNG is the default
through the plotting helpers and launchers; PDF remains an explicit option.
Shared thesis colours, dimensions and margin fitting remain the presentation
owner. Source data is preserved in place.

## Verification scope

The audit followed selected paths across the solver/coupler source inventory.
It is not a line-by-line qualification of every method or a proof that every
long-running case will finish. Numerical regressions execute real CPU kernels;
restart tests include field/identity rejection and current-schema round trips.
Copied tutorial checks exercise installed imports outside the checkout.

- The focused numerical/schedule/LU checks passed (13 checks).
- The broader VPM suite passed all 163 checks across diagnostics, schedules,
  VLM qualification, coupled restart, backup storage, output contracts, logging,
  DVH and kernel argument validation. Changes made while it ran were also
  exercised by the focused current-output tests below.
- Active-cloud diagnostics passed with f32 and f64 on CPU and f32 on Metal.
  Metal required execution outside the sandbox to access the shader compiler.
  The test asserts the actual backend, so a CPU fallback cannot pass it.
- The current output/identity checks passed (15 checks); the surface-backup test
  fixture was updated for the current particle revision contract and passed.
- The FVM/coupler/tutorial-output batch passed its 94 current checks. Its one
  superseded configuration-adapter expectation was replaced by a strict
  current-configuration test, which passed in the subsequent coupler run.
- The copied vortex-interactions startup and related tutorial tests passed
  (11 checks). The shared shell-launcher test passed for default PNG and explicit
  PDF arguments, including failure propagation.
- Solver/coupler Ruff checks, byte compilation and changed-file whitespace checks
  passed.
- Isolated Python imports outside the checkout resolve to this working tree;
  the installed OpenONDA environment uses these repairs without reinstallation.
- Repository Pyrefly checking reports 165 errors on unchanged source lines;
  the complete type-check scope is not green.
- At that inspection, the collection-wide tutorial-style check failed on validation/control
  flow in the separately maintained rotor setup. That setup was not rewritten
  as part of these repairs.

## Case readiness limits

Delta-wing's moving-surface observer avoids heap arrays inside its native
parallel collision loop. The coordinated investigation found matching event
selections and differences no greater than 2.22e-16 on captured inputs. Its checks
cover 550 frozen-input calls, 20 observer/collision tests, a 40-step CPU/f32/FMM
run and a 12-step explicit-bounds run. The original heap abort was intermittent
and was not reproduced by the short old-code replays. This is bounded repair
evidence, not full-horizon validation.

The vortex-interactions baseline continuation reaches step 1584, t = 5.94 s,
with full-cloud divergence 0.1201552202 above its configured maximum 0.12.
The run correctly preserves its final native checkpoint, ordered terminal
samples and failure reason. Its numerical settings match the current baseline.
This remains a numerical-resolution limitation; it must not be reported as a
completed or qualified 2400-step campaign. Health thresholds are unchanged.

A read-only frozen-state probe compared passive, group-preserving redistribution
at 0.05 m and 0.04 m grid spacing, with 0.05 m cores and the unchanged 0.003 tail
budget. The resulting populations were 157340 and 306545, and full-cloud divergence
was 0.1166576 and 0.1169660 respectively. These are candidate representations
before the solver's moment/transfer acceptance gates, not accepted continuation
states. They do not establish a converged remedy or justify changing the default.
