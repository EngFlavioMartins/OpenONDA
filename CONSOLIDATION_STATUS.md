# OpenONDA Consolidation — Status & Remaining Work

Branch `development` · audited and repaired on top of `a74c351` ("first step of naming update done").
Scope: finish the solver nomenclature/API consolidation, repair only genuine migration defects, verify VPM + FVM + Coupler.

---

## 1. Verification matrix (this tree)

| Gate | Command | Result |
|---|---|---|
| Byte-compile | `python -m compileall -q source tests tutorials openonda scripts` | OK |
| Lint | `ruff check source tests tutorials scripts openonda` | OK |
| Install sanity | `python -m openonda.verify_install` | OK (native FVM solve healthy, taichi 1.7.4/arm64) |
| Collection | `pytest --collect-only tests/vpm tests/coupler` | 1098 collected, 0 errors |
| VPM gate | `pytest -q tests/vpm -m "not gpu and not slow"` | **980 passed, 13 skipped, 0 failed** |
| Coupler gate | `pytest -q tests/coupler -m "not mpi and not slow"` | **101 passed, 0 failed** |
| Checkpoint focus | `pytest -q tests/vpm/test_filament_refinement_checkpoint.py` | 4 passed |
| Cube parity | `pytest -q tests/coupler/test_cube_benchmark_parity.py -m "not mpi and not slow"` | 18 passed |
| FVM gate | `pytest -q tests/fvm -m "(unit or verification) and not slow and not mpi"` | **388 passed, 0 failed** |

Skips are environmental only: OpenVSP Python API absent (5) and RWM ensemble GPU-only (8). No numerical test was weakened or skipped by these changes.

---

## 2. Repairs applied in this pass

### Functional migration defects (behavior restored to pre-rename intent)
1. `source/solvers/VPM/core/evolution.py:459` — `_coupled_stable_time_step_size` read stale attr `characteristic_distance`; now reads `particle_spacing`. The displacement substep limiter had been silently disabled in coupled runs.
2. `source/solvers/VPM/coupling/stepper.py:51` — shed-particle viscosity read stale attr `viscosity`; now reads `kinematic_viscosity`. Wake particles were silently falling back to hardcoded ν = 1e-2.
3. `source/solvers/VPM/core/solver.py` — removed `self.case_dir = final_setup.checkpoint_directory`. It overwrote the resolved case dir, which would have sent VPM samples/CSV exports to `<case>/solution/samples` instead of `<case>/samples` (canonical location used by FVM sampling and all tutorials/scripts).

### Structural artifact
4. `source/solvers/VPM/config/setup.py` — collapsed accidental triple-stacked `@classmethod` (`from_dict`) and triple-stacked `@staticmethod` (`_stabilization_from_dict`) decorators.

### Stale names in user-facing messages / serialized output
5. `source/solvers/VPM/physics/base.py` — docstring + 2 error messages: `max_targets` → `max_evaluation_points`; paired assertion updated in `tests/vpm/test_audit_completion.py`.
6. `source/solvers/VPM/runtime/backend.py` — error messages say `compute_device=...` instead of `processing_unit=...`.
7. `source/solvers/VPM/physics/diffusion/grid.py` — warnings/errors say `domain_bounds` instead of `vpm_domain_bounds`.
8. `source/coupler/config/types.py` — validation message and `to_dict()` keys renamed: `coupler_backup_period` → `checkpoint_interval_steps`, `transfer_diagnostic_interval` → `transfer_diagnostic_interval_steps`.
9. `source/coupler/reporting.py` — run-metadata key `coupler_backup_period` → `checkpoint_interval_steps`.

### Docstrings only
10. `core/solver.py` (`<backup_directory>/samples/...` → `<case_dir>/samples/...`), `physics/pressure.py` (`particles_kernel` → `particle_kernel`), `io/logging.py` (`characteristic_distance`/`viscosity` → `particle_spacing`/`kinematic_viscosity`).

---

## 3. What must still be fixed / decided (checklist)

- [ ] **CI status after push (runs 32461867590 / follow-up on `74d4bab`).**
  - Fixed by this work: lint/ruff, VPM kernel smoke, 5 of 6 wheel-import jobs, and the FVM↔VPM coupler job (stale `format_version == 4` assertion in `tests/coupler/test_fvm_vpm_smoke.py` — the slow smoke test only runs in CI, which does not filter `slow`; fixed in commit `74d4bab`).
  - Still failing (both pre-existing, identical failures on `ccb9fb4` before this work):
    - **FVM correctness job** (`ubuntu`): runs the exact command that passes locally 388/388 — environment-specific failure on the runner. Needs authenticated log access (`gh auth login` or `GH_TOKEN`) to download job logs / `fvm-fast.xml` artifact and diagnose.
    - **Wheel import (macos-15-intel)**: fails at "Install wheel in an isolated environment" — packaging/infra issue on the intel runner, untouched by source changes.
- [ ] **Scientific certification failures (deferred by design — do NOT fix in this phase).** Not exercised by the fast gates above; live in the slow/certification suites:
  - Periodic ABC flow convergence order
  - TGV central/upwind schemes
  - TGV refinement convergence order
  - WALE/TGV decay timing
  - IBM cylinder slip condition
  - IBM square force/wake
  Run the full certification suite in the scientific-debugging phase; thresholds must not be weakened.
- [ ] **MPI end-to-end smoke test.** The new inactive-rank ownership path (`_InactiveVPMSolver` placeholder + coupler `_openonda_inactive_rank`) is unit-tested but a real multi-rank coupled run (e.g., cube under `mpirun -np 2`) should be executed once before release.
- [ ] **GPU suites.** Metal/CUDA paths skipped on this machine; run `tests/vpm -m gpu` and treecode-GPU checks on target hardware.
- [ ] **Legacy read-alias policy.** Deliberately kept as archival read tolerance: `config/state.py` SolverState `AliasChoices` legacy names; `diagnostics/offline.py` `number_of_particles` fallbacks. Decide if/when to drop.
- [ ] **VLM vocabulary.** The VLM subsystem still uses its own `logging_frequency`-style names (internally consistent). Decide whether it adopts the `*_interval_steps` convention in a later pass.
- [ ] **Run-metadata shorthand.** `reporting.py` still writes `"nu"`, `"rho"`, `"dt"`, `"t_end"` keys — left intentionally as an on-disk convention; revisit if post-processing migrates.
- [ ] **Breaking-change notes for users** (document in release notes):
  - Old checkpoints are not restartable: FVM format v3→v4, VPM 3.1→4.0, Coupler v4→v5 (legacy compat readers removed). Archival files remain readable by offline tools only.
  - VPM samples now strictly at `<case_dir>/samples` (no longer `<case>/solution/samples`).
  - Renamed user-facing options in setup files: `cap_absolute_fraction`, `processing_unit` → `compute_device`, `max_targets` → `max_evaluation_points`, `nu` → `kinematic_viscosity`, `tol` → `tolerance`, relaxation/output-interval renames, etc. External user scripts using old kwargs will fail loudly (by design).
- [ ] **CI watch.** After pushing, confirm GitHub Actions passes on origin/development.

## 4. How to re-verify

```bash
python -m compileall -q source tests tutorials openonda scripts
ruff check source tests tutorials scripts openonda
python -m openonda.verify_install
python -m pytest -q tests/vpm -m "not gpu and not slow"
python -m pytest -q tests/coupler -m "not mpi and not slow"
python -m pytest -q tests/fvm -m "(unit or verification) and not slow and not mpi"
```

---

## 3. Post-review waves (items 1–13 complete)

Commits (fast-forward, all hooks green): `e79b1d7` consolidation → `74d4bab` CI assertion fix → `e6e8e68` canonical naming/metadata/property ownership → `9b836e6` alias removal → `d90fab1` VLM serialization guard → `eb589df` VLM absorption fixes → `6806e66` descriptive VLM/Panel public API → `1aad28a` audit sweep → `865325c` MPI coupled smoke test.

- Item 3: `FVMSetup.load()` canonical-only; `SolverState`/`ParticlesState` alias-free.
- Item 4: panel/VLM particle-field fixes; root-caused winding bug in `is_point_in_quad` (collisions never fired); added `tests/vpm/test_vlm_particle_absorption.py`.
- Item 5: stepper no longer clobbers panel density; exact-ν shedding (0.0 inviscid / configured viscous / hard error otherwise); VLM field renamed `viscosity` → `kinematic_viscosity`.
- Item 6: `V_inf→freestream_velocity`, `V_external→external_velocity`, `V_wake(_field)→wake_velocity`, `U_ref→reference_velocity`; legacy positional `advance()` convention and silent `**kwargs` removed; stale `logging_frequency` getattr chains fixed (they silently ignored configured intervals).
- Item 7/8/10: coupler vortex-strength naming; canonical `run_metadata.json` keys (+ archived readers); `max_iterations`, `angular_speed`, `adjust_time_step` renames.
- Item 9: `VPMSetup.to_dict()` rejects VLM-coupled setups instead of writing `"vlm": null`.
- Item 11 audit: remaining hits classified — Taichi kernel names, `grad_u` locals, VTK contract keys (`VelocityGradient`), OpenFOAM names (`Cs` etc.), offline archive readers (`grid_independence_cs.py` fallbacks) intentionally left.
- Item 13: `tests/coupler/test_fvm_vpm_mpi_smoke.py` — real `mpirun -np 2` coupled run via public API (`ComputeConfig.petsc_replicated()`); passes on both ranks locally. Note: coupler init is lazy; substep assertions belong after `run()`.
- Deferred by design: serialized `pruned_circulation_*` diagnostic JSON keys (on-disk contract).

CI on `development`: only two pre-existing failures (ubuntu FVM-runner env — same command passes locally 388/388; macos-15-intel wheel install). Diagnosis needs authenticated `gh` access.

**Next phase (items 14–20, FVM scientific debugging) is blocked pending user review of this consolidation diff.**
