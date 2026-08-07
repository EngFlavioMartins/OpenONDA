# VPM Subsystem Audit — Stage A / A2

**Scope:** `source/solvers/VPM/` (40,108 LOC, 99 Python modules) and tightly coupled subsystems
(VLM, panel/BEM, samplers, backup, profiling, Taichi backend).
**Branch:** `development` · **Started** 2026-08-07 @ `7d7f135` · **Updated** 2026-08-07 (Stage A2)
**Status:** Stage A, Stage A2 (Phases 1–5) and Stage D complete. Stage C partial (A-1 outstanding).
Remaining blockers are listed under [technical debt](#remaining-technical-debt).

Every finding is backed by a code location, a reproduction, or a measurement taken on this
machine (Linux, 14 GB RAM, Taichi 1.7.4 / LLVM 15, CPU + Vulkan backends; **no CUDA device**).
Anything I could not verify is labelled **UNVERIFIED** and carries no code change.

---

## Status board

| Phase | Objective | State |
|---|---|---|
| 1 | Test reliability | Trees merged; backend-skip bug fixed; **full-suite validation runs still outstanding** |
| 2 | Safe numerical guards | **Done** (zero-span guard, RWM seed, kernel/treecode validation retained) |
| 3 | Winckelmans `256/45` | **Done — traced, and constant changed to the derived 4ν on the author's instruction** |
| 4 | Truthful precision contract | **Done** (f64 labelled experimental; f64+TREECODE rejected) |
| 5a | backup / sampler audit | **Done (static)** — R-1 withdrawn, R-2 fixed; equivalence *run* still owed |
| 5b | pressure / hierarchical / interpolation | **Done** — 2 dead modules deleted, 3rd copy of the π-bug fixed |
| 5c | panel / BEM | **Done** — 3 findings; NEUMANN force path **fixed** |
| 5d | OpenVSP | **Done (static)** — 3 orientation/symmetry findings; runtime unverified (API absent) |
| C | Architecture | **Partial** — A-2, A-3 done; **A-1 not started** |
| D | Performance | **Done for P-1…P-6** — measured; per-stage profile now exists |

---

## Phase 3 — the `256/45` core-spreading constant (headline result)

**The constant traces to a real, citable source: your own paper.**

> F. A. C. Martins, A. van Zuijlen, C. J. Simão Ferreira,
> *"Toward Meshless Turbulent Flow Simulation: LES-Integrated Vortex Particle Method"*,
> [arXiv:2601.06942](https://arxiv.org/abs/2601.06942), submitted 11 January 2026.

That paper states, with **exactly OpenONDA's kernel and σ convention**:

- kernel: `ζ(ρ) = (15/8π) · 1/(ρ²+1)^(7/2)`, `ρ ≡ |x − x_p|/σ` — identical to `kernels/winckelmans.py`
- update: `(σ_p)²(t) = (σ_p)²(t−Δt) + C_ν (ν + ν_T,p) Δt`, with `C_ν = 256/45`
- stated justification: *"C_ν = 256/45 is the viscous diffusion constant determined by the
  authors based on how the second moment of the vorticity distribution evolves under viscous
  diffusion"*, and *"derived from the analytical solution of the diffusion equation for a
  Gaussian distribution"*

**What this settles:**

1. The code comment "The reference VPM formulation uses…" is **misleading** — it reads as a
   pointer to Winckelmans & Leonard 1993. W&L 1993 does not contain this constant, and neither
   does Alvarez & Ning 2022 (FLOWVPM, `docs/literature/alvarez2022.pdf`), which I extracted and
   searched: it cites only Rossi for core spreading. The comment should cite arXiv:2601.06942.
2. The convention question the review raised is **closed**: the paper and the code use the same
   ζ and the same σ. There is no scaling mismatch to explain the number.

**What this does not settle — and why it still needs your decision:**

I cannot reproduce `256/45` from the stated criterion. For a self-similar blob
`ω = (Γ/σ³) ζ(r/σ)`, second-moment matching gives `⟨r²⟩ = σ² m₂` and `d⟨r²⟩/dt = 6ν`, hence
`dσ²/dt = 6ν/m₂`. Computed on this machine:

| kernel | m₂ (numeric) | m₂ (exact) | `dσ²/dt` from 2nd moment |
|---|---|---|---|
| Gaussian `π^-3/2 e^{-ρ²}` | 1.50000 | 3/2 | **4ν** |
| Winckelmans `15/(8π)(1+ρ²)^{-7/2}` | 1.50000 | 3/2 | **4ν** |

Both kernels have `m₂ = 3/2` — **by design**; that is the property that makes the W&L kernel a
second-order match to the Gaussian. Second-moment matching therefore *cannot distinguish them*,
and gives 4ν for both. For the whole algebraic family `ζ ∝ (1+ρ²)^{-p}` the closed form is

```
m₂ = (3/2)/(p − 5/2)        dσ²/dt = 6ν/m₂ = 4(p − 5/2) ν
```

verified numerically at p = 3, 3.5, 4, 4.5, 5 (→ 2, 4, 6, 8, 10 ν). Reaching `256/45 ≈ 5.6889`
would require `p = 5/2 + 64/45 ≈ 3.922`, which is not a kernel order anyone uses.

Three other standard criteria also miss it, for the W&L kernel at p = 7/2:

| criterion | Gaussian | Winckelmans |
|---|---|---|
| second moment | 4ν | **4ν** |
| enstrophy-dissipation match | 4ν | 9.625ν |
| origin-curvature match (`−2ν ζ″(0)/ζ(0)`) | 4ν | 14ν |
| L²/Galerkin projection of the heat operator | 4ν | 11ν |
| **declared in code and paper** | 4ν ✓ | **5.689ν** |

All four criteria agree on 4ν for the Gaussian, as they must, which is evidence the machinery is
right and the disagreement is specific to the algebraic kernel.

### Resolution

The author confirmed: **`256/45` was hand-calibrated, not derived.** That closes the
investigation — the value was never the output of the second-moment argument, so the defect was
the *comment* presenting it as a reference formulation, not the number's provenance.

On the author's instruction the constant was **changed to the derived value**:

```
C = 6/m2 = 4        (kernels/winckelmans.py:diffusivity_constant_)
```

**This is a result-changing numerical modification.** Quantified:

| | old (256/45) | new (4) |
|---|---|---|
| C | 5.688889 | 4.000000 |
| dσ²/dt rate | — | **−29.7 %** |
| σ(t) at large t | — | **×0.8385 (−16.1 %)** |

For σ₀ = 0.1, ν = 1e-3: σ differs by −5.5 % at t = 1, −13.5 % at t = 10, −15.8 % at t = 100.
**Any previously published `WINCKELMANS` + core-spreading result will not reproduce.** Two
tutorials (`rotorFlow`, `quadCopter`) select that kernel. If the calibration was tuned against
specific data, restoring `256/45` is a one-line change — but it must then be labelled a
calibration, because it is not what the second-moment argument gives.

Pinned by `test_core_spreading_constant_follows_from_the_second_moment` and
`test_declared_diffusivity_constants_match_the_derivation`.

Note independently: core spreading is an *exact* self-similar solution of the heat equation only
for the Gaussian. For any algebraic kernel it is a model, because the diffused algebraic blob is
not an algebraic blob — so a calibrated constant is a legitimate choice there, provided it is
labelled as one.

---

## Phase 1 — test reliability

### Done

- **`tests/VPM/` merged into `tests/vpm/`** (13 files + one `boundary_elements/vlm/geometry/`
  subtree, via `git mv`). No basename collisions existed, and `tests/VPM/` had no `conftest.py`,
  so no fixture rebinding was involved. Collection after the merge: **45 files, 1074 tests**.
  The stale `tests/VPM` reference in `REFERENCES.md` was updated. This removes the macOS
  case-insensitivity hazard (D-1) — the two directories collapsed into one on any Darwin
  checkout, and `conftest.py` explicitly supports Darwin via `BACKENDS = ["CPU","METAL"]`.

- **Backend-unavailability is now a skip, not a failure (new finding T-1).** The
  `solver_for_backend` fixture *intended* to skip a missing backend, but the check sat **after**
  `Solver(...)`, and an explicit GPU request is strict — `initialize_taichi_backend` raises
  rather than falling back. So on a machine without CUDA every CUDA-parametrised test **failed**.
  Added a session-cached `_backend_available()` probe and an autouse fixture that skips before
  construction. Verified: `test_evaluation_integrals.py` now reports
  `SKIPPED … CUDA backend unavailable on this machine` instead of
  `RuntimeError: Requested Taichi backend CUDA failed to initialise`.

### Correction to the Stage-A report

The Stage-A report claimed an **order-dependent test failure** (old D-4), citing
`test_coupled_time_integration.py` failing in a full run and passing alone, plus
`test_single_blob.py` / `test_two_particles.py` failing together.

**That claim was unsound and is withdrawn.** Both observations were collected while I had a full
pytest suite running in the background *and* a memory-profiling script allocating ~1 GB. Re-running
the exact same commands on an idle machine passes: `test_two_particles + test_single_blob` → 64
passed; `+ test_biot_savart` under the original `-k` filter → 66 passed. The failures were my own
resource contention, not test-order leakage.

I am not asserting the suite *is* order-independent — only that I have no evidence it isn't. That
is what the validation runs below are for.

### Not done — the validation runs

The required evidence is a clean sequence:

```bash
pytest tests/vpm            # x3 consecutive, identical results
pytest tests/vpm --random-order
```

Run 1 was started on an idle machine and was still executing at ~7 % after a long interval; 1074
tests × Taichi JIT per solver construction makes a full pass expensive, and it did not finish
within this session. **Phase 1 is therefore not complete.** Also note `pytest-random-order` /
`pytest-randomly` is **not installed** — the shuffled run needs one added to the dev dependencies.

---

## Phase 2 — safe numerical guards (done)

| Item | Change |
|---|---|
| N-7 zero-span VLM panel | `vlm/solver/kernels.py` — degenerate TE panels (`span_mag < 1e-12`) are now skipped before `span_vec / span_mag` can emit NaN strengths into the VPM |
| N-6 RWM seed | `VPMSetup.random_seed` (default 42) → `initialize_taichi_backend(random_seed=…)` → `ti.init`; serialized in `to_dict()`. RWM ensembles are now possible; the docstring says so |
| N-1 kernel × treecode | Retained; `TREECODE_SUPPORTED_KERNELS` in `config/constants.py` is now the single source of truth read by both `VPMSetup` validation and `TaichiTreecode.set_kernel_type` |
| N-2 direct-vs-tree consistency | Retained + extended (below) |

---

## Phase 4 — precision contract (done)

Rather than implement f64, the contract was made truthful:

- **`precision='f64'` + `VelocityConfig.treecode(...)` is now rejected at config time.** The
  treecode is f32 end-to-end — fields, multipoles, host transfer buffers — so that pairing
  delivered f32 accuracy under an f64 label. The only solver-level f64 test in the suite
  (`test_coupled_time_integration.py`) already uses `VelocityConfig.direct()`, so nothing broke.
- The `precision` docstring now states plainly that f32 is **the supported production precision**
  and that f64 is **EXPERIMENTAL and not end-to-end**, enumerating the three paths that stay f32
  (treecode; gradient/energy/helicity accumulators in `kernels_common.py`; the A&S 7.1.26 erf).

Full f64 support remains deliberately unimplemented — it touches tree storage, kernel generation,
reductions and special functions, and deserves its own scoped project with its own tests.

---

## Phase 5 — completing Stage A

### 5b — duplicate / dead evaluation paths (done)

| ID | Sev | Finding |
|---|---|---|
| **N-2c** | **HIGH** | **A third copy of the Gaussian `q` kernel**, in `physics/pressure.py:_q_kernel`, carried **the identical factor-of-π error** (`4/(3√(π³))` instead of `4/(3√π)`) in its small-ρ branch. This is direct evidence the defect propagated by copy-paste. **Fixed.** Impact is small (f64 NumPy, branch only for ρ<1e-4) but it is the same defect in a third place. |
| **A-6** | MEDIUM | `physics/hierarchical.py` (241 lines) is **entirely dead** — *nothing imports the module*. It is a superseded cluster-based Barnes–Hut split; the `*_hierarchical` methods in `physics/base.py` despite their name use the LBVH treecode. Recommend deletion. |
| **A-7** | MEDIUM | `numerics/interpolation.py` (532 lines, `M4Interpolation`) is **entirely dead** — no importers, no references. The live M4′ implementation is `_m4_prime` inside `numerics/divergence_relaxation.py`. Recommend deletion after confirming no external consumer. |
| — | — | `physics/pressure.py` is **live** (used by `solver.py:1636`, tested by `tests/vpm/test_pressure.py`). |

### 5a — backup / restart

The backup restores `flow_time`, `time_step`, `time_step_size`, filament lineage, the
divergence-relaxation reference moments, and the filament-refinement cumulative transfers.

| ID | Sev | Finding | Status |
|---|---|---|---|
| ~~R-1~~ | — | I claimed the relaxation cumulative gate counters were not persisted. **Wrong — withdrawn.** All 53 keys of `_divergence_relaxation_diagnostics`, including the 11 `relaxation_cumulative_*` gate counters, *are* written by the generic `for name, value in …items(): solver_group.attrs[name] = value` loop (`backup.py:223-228`) and restored by the matching loop (`:667-672`). My grep for the literal key in `backup.py` found nothing because the keys are never literal there. | **WITHDRAWN** |
| **R-2** | MEDIUM | `_dvh_fire_counter` was **not** persisted. DVH fires every `_dvh_substeps` steps off this counter, so a restart resumed at phase 0 and the viscous update landed on different steps than the uninterrupted run. | **FIXED** — saved as `dvh_fire_counter`, restored if present |
| **R-3** | LOW | The energy history behind `dE/dt` is not persisted, so the dissipation-rate diagnostic is wrong for the first logging events after a restart. | OPEN |
| **R-4** | LOW | Taichi's RNG state is not (and cannot easily be) persisted, so an RWM restart draws a different Brownian sequence. Inherent — document, don't fix. | OPEN (document) |

**Still owed:** the restart-equivalence *run* (save at step N, resume, compare against an
uninterrupted run to tolerance), as a standing test.

### 5c — panel / BEM (partial)

| ID | Sev | Finding |
|---|---|---|
| **B-1** | **HIGH** | `panels/solver/influence.py:build_AIC_matrix` is **dead** (no callers) **and actively misleading**: its docstring says *"AIC matrix for Neumann BC"* while it assembles a **doublet** AIC with a **0.5 self-term** — that is the *Dirichlet* potential self-term (the constant-doublet potential jump), not a Neumann normal-velocity self-influence. Anyone wiring it up would get a wrong Neumann formulation. This is consistent with the project's existing note that "NEUMANN is broken". Recommend deletion. |
| — | — | **Verified correct:** the live paths are `build_AIC_matrix_dirichlet` (doublets, DIRICHLET) and `build_source_AIC_matrix` (sources, NEUMANN). A source panel's self-induced normal velocity *is* ±1/2, so the 0.5 diagonal is right there. NEUMANN additionally removes the mean source strength after the solve (`values -= Σ(values·area)/Σarea`), the standard closed-body compatibility fix. |
| **B-2** | MEDIUM | `PanelSolver.compute_induced_velocity` chooses between **two independent implementations** of the same Biot–Savart integral on a size heuristic (`n_panels >= 1000 or n_panels*len(points) >= 100_000`): a Taichi kernel and an inline Python closure. I checked them line by line — the maths and the `-coeff` sign agree — but they use **different epsilons** (`PANEL_EPSILON = 1e-14` vs a hardcoded `1e-12`), so results are not identical across the switch. Collapse onto one implementation. |

| **B-3** | **CRITICAL** | **Panel forces are invalid under `bc_type='NEUMANN'`, on a live path.** `compute_forces` (both BERNOULLI and KUTTA_JOUKOWSKI branches) reads `lattice.strengths` — the *doublet* field. A NEUMANN solve writes `lattice.source_strengths` and leaves `strengths` at its initial zeros; the two fields are never synced. So the reported load omits the body's entire singularity contribution. This runs **every step**: `Solver._advance_panel` → `PanelSolver.advance` → step 6 `compute_loads` → `compute_forces`. `bc_type='NEUMANN'` is used by `tutorials/coupled_FVM_VPM/cubeFlow/cubeFlow_setup.py:153` and asserted by `tests/coupler/test_cube_benchmark_parity.py`. **Mitigated, not fixed:** a one-time `logger.warning` now fires, because the correct fix is to port `compute_forces` onto the source-doublet surface velocity that `compute_postprocess` already uses — which also changes DIRICHLET forces and so needs its own verification. | **WARNS — needs a real fix** |
| — | — | **Verified correct:** `compute_postprocess` *does* use `compute_surface_velocities_with_sources` with both fields, and for DIRICHLET derives `source_strengths = −n·V_inf` first — the standard Hess–Smith source-doublet split. `Cp = 1 − (V/V∞)²` is guarded by `PANEL_EPSILON` against V∞ = 0 (though Cp is physically meaningless in hover). |
| — | LOW | `compute_loads` is a pure forwarding wrapper over `compute_forces`. |

**Conclusion: the panel/BEM subsystem is DIRICHLET-only in practice.** Two of the three NEUMANN
defects are now removed or made loud; the force path still needs a real port.

### 5d — OpenVSP (static only — the Python API is absent on this machine)

| ID | Sev | Finding |
|---|---|---|
| **V-1** | MEDIUM | `_apply_coordinate_transform` accepts any 3×3/4×4 matrix and never checks it is a proper rotation (det = +1). A reflection (det = −1) silently mirrors the geometry and **flips every panel normal**, inverting the lift sign with no diagnostic. Add a determinant check. |
| **V-2** | MEDIUM | `_validate_segment_orientation` does not validate orientation. It checks area > 0 and non-coincident LE/TE vertices — both degeneracy tests. A mis-wound a→b→c→d segment yields an inverted normal and inverted lift, and this validator passes it. (Compare the known "WingSegment a→b = LE/span" trap.) |
| **V-3** | MEDIUM | `OpenVSPImportConfig.symmetry` defaults to `"from_openvsp"`, which `_symmetry_to_int` maps to **0 — i.e. no symmetry**, identical to `"none"`. The name implies the setting is read from the VSP model. A half-model with XZ symmetry therefore imports unmirrored and produces half the lift, silently. Either implement it or rename the default. |

**Unverified:** all runtime behaviour of the importer — the OpenVSP Python API is unavailable here
and its 6 tests skip.

---

## Updated issue inventory

Severity: **CRITICAL** (wrong results or crash on a documented path) · **HIGH** ·
**MEDIUM** · **LOW**.

### Numerical correctness & physical model

| ID | Sev | Issue | Status |
|---|---|---|---|
| N-1 | CRITICAL | `HIGH_ORDER_GAUSSIAN` (documented "recommended production kernel") and `SUPER_GAUSSIAN` crashed at the first velocity evaluation with the **default** velocity method `TREECODE`; config accepted the combination. Reproduced: 2 of 8 kernel×method pairs raised `ValueError`. | **FIXED + TESTED** |
| N-2 | HIGH | Treecode carried a divergent second copy of the Gaussian `q`; its small-ρ branch was a factor of **π** too small (`4/(3√(π³))` vs `4/(3√π)`; ratio verified = 3.14159265). | **FIXED + TESTED** |
| **N-2b** | **HIGH** | **Discovered while writing the test for N-2.** The small-ρ crossover sat at **ρ = 1e-4**, three decades *inside* the region where `erf(ρ) − (2/√π)ρe^{−ρ²}` has already cancelled to nothing in f32. Measured relative error of the closed form: **5.6e-2 at ρ=1e-2, 4.6 at ρ=1e-3, 7.1e+4 at ρ=1e-4**, and it returns **negative** q just above the branch (−4.7e-09 where q = +7.5e-13). Replaced with a 3-term series and crossover 0.2 (measured: series ≤ 4e-6 for ρ ≤ 0.2; closed form ~2e-5 at 0.2 and improving). Coefficient and crossover now live in `config/constants.py`, shared by both kernel copies. | **FIXED + TESTED** |
| **N-2c** | HIGH | Third copy of the same kernel in `physics/pressure.py` with the same π error. | **FIXED** |
| N-3 | HIGH | f64 advertised but not delivered end-to-end (treecode f32; gradient/energy accumulators f32; A&S erf 1.39e-7 measured). | **CONTRACT FIXED** — f64+treecode rejected, docstring truthful; full f64 deferred |
| N-4 | HIGH | Winckelmans core-spreading `256/45` — hand-calibrated, presented as a reference formulation. | **RESOLVED + CHANGED to 4ν, TESTED** (result-changing: −29.7 % in dσ²/dt) |
| N-5 | MEDIUM | `DEFAULT_CUTOFF_RADIUS_FACTOR = 100` applied in vorticity/energy/gradient kernels but not velocity/stretching; at ρ=100 both kernels are ≲1e-14 so it never prunes — a per-pair branch that costs work and prunes nothing, inconsistently. | OPEN |
| N-6 | MEDIUM | RWM seed hardcoded. | **FIXED + TESTED** |
| N-7 | LOW | Zero-span VLM TE panel → NaN strengths. | **FIXED** |

**Verified correct** (independently recomputed this session):

- Both production kernels normalize to **1.0**; **m₂ = 3/2** for both, matching
  `angular_impulse_correction_constant_ = 1.5`.
- **`HIGH_ORDER_GAUSSIAN` and `SUPER_GAUSSIAN` have m₂ = 0 exactly** (their `2.5 − ρ²` polynomial
  cancels the second moment), and both correctly declare `0.0`. The super-Gaussian docstring
  records that this was previously 1.875 "interpolated between Gaussian and Winckelmans" and why
  that was wrong — that earlier fix is confirmed correct.
- Stretching DIRECT/TRANSPOSED/MIXED match Winckelmans & Leonard 1993.
- Smagorinsky: `C_k = (C_s²√C_e)^{2/3}` inverts `C_s² = C_k√(C_k/C_e)`; C_k=0.094, C_e=1.048 → C_s = 0.1678 as documented. `|S| = √(2S_ijS_ij)` standard.
- **VLM → VPM circulation transfer satisfies Kelvin's theorem** — the shed streamwise circulation
  telescopes to exactly zero; now pinned by regression tests.

### Architecture

| ID | Sev | Issue | Status |
|---|---|---|---|
| A-1 | HIGH | `_apply_divergence_relaxation` (527 lines) and `_apply_filament_refinement` (315 lines) inlined in `Solver`, ~90 % acceptance-gate auditing, doing 2× FFT + 2× `discretization_health` + full GPU→CPU transfers per firing, and raising to kill the run. | OPEN — **Stage C** |
| A-2 | HIGH | Constructing a `Solver` rebinds process-global `sys.stdout`/`stderr`. | **FIXED** — `log_mode` now defaults to `'tee'`, so the log file is still written but the caller's console output survives; `'file'` keeps the old redirect and documents that it captures the whole process |
| A-3 | MEDIUM | `Solver` god class: 3,717 lines, ~130 methods, `getattr(self, …)` reads against its own attributes. | **PARTLY FIXED** — `_body_induced_fn`, `_stretch_dt_warned`, `_discretization_health` declared in `__init__` and read directly (8 → 4 sites). The remaining 4 are the `_filament_reference_*` `hasattr` sentinel, which A-1 restructures |
| A-4 | MEDIUM | `_GridDiffusionMixin`: 1,600-line mixin into two classes, `_impl`/public wrapper pairs. | OPEN |
| A-5 | LOW | `ParticlesLES` near-pure forwarding wrapper. | OPEN |
| **A-6** | MEDIUM | `physics/hierarchical.py` (241 lines) dead — no importers. | **FIXED** (deleted) |
| **A-7** | MEDIUM | `numerics/interpolation.py` (532 lines) dead — no importers. | **FIXED** (deleted) |
| **B-1** | HIGH | Dead + misleading Neumann doublet AIC (`build_AIC_matrix`). | **FIXED** (deleted) |
| **B-2** | MEDIUM | Two size-switched implementations of the panel Biot–Savart integral, differing epsilons. | OPEN |
| **B-3** | CRITICAL | NEUMANN panel forces read the never-filled doublet field. | **FIXED** — `compute_forces` and `compute_postprocess` now share one `_update_surface_velocities` helper built on the source-doublet representation, so the two paths cannot disagree and NEUMANN carries its solved source strengths. `v_inf_mag` also floored to avoid a hover blow-up. 7 panel tests pass |
| **R-2** | MEDIUM | `_dvh_fire_counter` not persisted → DVH restart phase-shifted. | **FIXED** |
| **V-1/2/3** | MEDIUM | OpenVSP: unchecked transform determinant; orientation validator that doesn't check orientation; `symmetry="from_openvsp"` silently means "none". | OPEN |

### Stage D — performance (all measured on this machine, CPU backend)

The prerequisite the review insisted on now exists:
`scripts/benchmarks/benchmark_vpm_step.py` is a small seeded deterministic case
that reports the `RuntimeProfiler` breakdown per stage plus RSS.

**Per-stage profile (N = 4000, treecode velocity θ=0.5, RK3, ms/step):**

| stage | direct stretching (default) | treecode stretching |
|---|---|---|
| **Stretching** | **652.55** | **220.11** |
| Velocity + gradients | 82.60 | 78.45 |
| Advection | 73.43 | 75.48 |
| Viscous diffusion | 0.69 | 0.25 |
| everything else | < 0.05 each | < 0.05 each |
| **total step** | **809.9** | **374.9** |

**Stretching is the dominant cost — 81 % of the step at N = 4000** — because the
default `StretchingConfig.use_treecode=False` evaluates ∇u·Γ with a direct O(N²)
pair sum while the velocity already uses the O(N log N) treecode. Switching it
gives **2.96× on the stage and 2.16× on the whole step**. I did **not** flip the
default: it changes results at the Barnes–Hut tolerance and so needs its own
accuracy verification. **This is the single largest remaining performance lever.**

**Applied optimizations (before → after):**

| ID | Change | Measured effect |
|---|---|---|
| P-1 | Treecode sized to actual N, doubling from a 8192 floor, instead of a fixed `MAX_PARTICLES` allocation. Doubling keeps regrows O(log N), which matters because Taichi never frees a field. | 100-particle run: **988 → 615 MB** |
| P-2 | Dipole/quadrupole node moments allocated only at the order that uses them (one-element stub otherwise); every write is order-guarded; `set_multipole_order` refuses to raise past the allocation. | 100-particle run: **1127 → 988 MB** (−139 MB, ≈ the predicted 144) |
| **P-1 + P-2 combined** | | **1126.9 → 614.6 MB, −45 %** |
| P-3 | `_zero_temp_fields` writes `[0:N]` via a kernel instead of `.fill(0)` over the full `max_particles` capacity (9 vec3 fields = 54 MB at the 500k default). Dead calls in CS/RWM already removed. | removes up to 54 MB of writes per call |
| P-5 | Freestream passed to the treecode as a device field (`background_field=`) instead of `bg[None]` → numpy → device. | removes one host round-trip **per RK stage** |
| P-6 | LES min/max reduction hoisted to a top-level loop (Taichi only parallelises the outermost `for`), with seeding split into its own kernel. | 0.446 → 0.207 ms at N = 100k (2.2×) |

Verified after the changes: 121 CPU tests across advection, stretching,
diffusion, two-particle, treecode and the audit regressions — **all passing**;
65 treecode-specific tests passing.

### Performance / memory (Stage A baseline, for reference)

| ID | Sev | Issue |
|---|---|---|
| P-1 | HIGH | *(now fixed — see Stage D)* **1.13 GB RSS for a 100-particle simulation** (import 359 → `Solver()` 465 → +100 particles 562 → first step **1127 MB**). Treecode sized for `MAX_PARTICLES = 500,000` / 1,000,000 nodes regardless of N. |
| P-2 | HIGH | *(now fixed)* Order-2/3 multipole fields (`node_circ_dipole` + `node_circ_quad` ≈ 144 MB at 1 M nodes) allocated at the default `multipole_order=1` and never touched. |
| P-3 | MEDIUM | `_zero_temp_fields()` fills 9 vec3 fields at full capacity (54 MB at default), not at active N. **Dead calls in CS/RWM removed.** |
| P-4 | MEDIUM | `strain_rate` — a persistent 3×3 field (36 B/particle f32, 24 % of the per-particle float budget) strictly derived from `velocity_gradient`, with one physics consumer (Smagorinsky) that immediately reduces it to the scalar \|S\|. |
| P-5 | LOW | `velocity_self` reads the background-velocity field to host **every RK stage**; hardcodes `np.float32`. |
| P-6 | LOW | LES stats reduction nested in `if N > 0:` so Taichi serialises it — measured 0.446 ms vs 0.207 ms at N=100 k. Two host scalar reads per step for logging. |

Per-particle budget: **152 B/particle (f32) / 292 B (f64)**; the fixed treecode allocation
dominates every case below ~1 M particles.

### Testing / hygiene

| ID | Sev | Issue | Status |
|---|---|---|---|
| D-1 | CRITICAL (portability) | `tests/vpm` + `tests/VPM` both tracked; collapse on macOS. | **FIXED** (merged) |
| D-2 | HIGH | Dead, Kelvin-violating `vlm/kernels/wake_shedding.py` (197 lines). | **FIXED** (deleted) |
| D-3 | MEDIUM | `stabilization/` stale `__pycache__` only. | **FIXED** |
| ~~D-4~~ | — | ~~Order-dependent test failure~~ | **WITHDRAWN** — was my own resource contention (see Phase 1) |
| **T-1** | HIGH | Unavailable GPU backends **failed** instead of skipping, because the fixture's skip check sat after the strict `Solver()` construction. | **FIXED** |
| **T-2** | MEDIUM | No shuffle plugin installed, so the required `--random-order` validation cannot run. | OPEN (add dev dep) |
| D-5/6/7/8 | LOW | Dead `FilteredParticles`, dead `compute_eddy_viscosity`, broken doc reference, import order. | **FIXED** |

---

## Test status

- Suite after merge: **45 files, 1074 tests** collected cleanly.
- `tests/vpm/test_audit_2026_08_regressions.py`: **34 tests, all passing** — kernel/treecode
  compatibility, treecode-vs-analytic `q` across six decades, series continuity and positivity,
  Kelvin telescoping, RWM seed, precision contract, and the four kernels' normalization / second
  moment / core-spreading constants.
- Affected-path batch (audit regressions + `test_kernels_math` + `test_treecode_lbvh` +
  `test_diffusion`): **81 passed, 0 failed**.
- `test_two_particles` + `test_single_blob` + `test_biot_savart`: **66 passed** on an idle machine.
- `ruff check` clean across everything I touched. (4 pre-existing errors remain in
  `config/backend.py` and `vlm/kernels/biot_savart.py`, which the author is editing — untouched.)
- `pytest-random-order` added to both dev-dependency groups in `pyproject.toml`; **needs
  installing** before the shuffled run can be executed.
- **Outstanding:** the 3× consecutive + shuffled full-suite validation (Phase 1 exit criterion).
  A full pass is ~1 h+ on this machine (1074 tests × Taichi JIT per solver construction) and did
  not complete within the session.

---

## Remaining technical debt

Nothing here is hidden or downgraded.

**Blocking a clean baseline**

1. **Phase 1 validation runs not executed.** The suite has not been *shown* stable across 3
   consecutive + 1 shuffled full run. `pytest-random-order` is declared in `pyproject.toml` but
   **not installed**, and a full pass is ~1 h+ here (1074 tests × Taichi JIT per solver
   construction). Everything below rests on targeted subsets instead.
2. **Winckelmans + core spreading results changed by −29.7 % in dσ²/dt.** Any prior result on
   that path must be regenerated, and the calibration question reopened if the derived value
   degrades agreement with whatever `256/45` was fitted to.

**A-1 — the one Stage C item not attempted**

3. `_apply_divergence_relaxation` (527 lines) and `_apply_filament_refinement` (315 lines) remain
   inlined in `Solver`, still ~90 % acceptance-gate auditing that does 2× FFT, 2×
   `discretization_health` and full GPU→CPU transfers per firing, and still `raise` to kill the
   run on a gate violation. Extracting them is mechanical but touches the hot path and the
   `DivergenceRelaxationError` contract that tests depend on; it is the natural first task of a
   dedicated Stage C pass, alongside the `_filament_reference_*` `hasattr` sentinel that keeps
   the last 4 `getattr(self, …)` reads alive, and `_GridDiffusionMixin` (A-4).

**Known gaps, not blocking**

4. **Largest remaining performance lever: treecode stretching.** Measured 2.96× on the stage and
   2.16× on the step at N = 4000. Not enabled by default because it changes results at the
   Barnes–Hut tolerance and needs its own accuracy verification.
5. **Restart equivalence never executed** — R-2 fixed, but the save→resume→compare test is owed.
   R-3 (energy history) and R-4 (RNG state) remain.
6. **P-4** `strain_rate` is still a persistent 3×3 field derived from `velocity_gradient` with one
   physics consumer that reduces it to a scalar; removing it needs the sampler/backup contract
   settled first.
7. **f64 remains partial** — the contract is now honest; the implementation is unchanged.
8. **B-2** duplicate panel Biot–Savart implementations with differing epsilons.
9. **V-1/V-2/V-3** OpenVSP orientation, winding and symmetry-default hazards; no runtime
   verification possible without the API.
10. **N-5** cutoff policy (a per-pair branch that never prunes, applied inconsistently).
11. **GPU numbers are unmeasured.** Every performance figure here is CPU-backend; this machine has
    no CUDA device. The memory wins should transfer directly, the timings should be re-taken.

---

## Classification

### **READY WITH DOCUMENTED LIMITATIONS**

Upgraded from the previous draft, where `bc_type='NEUMANN'` panel loads were **NOT READY**. That
defect (B-3) is now fixed: `compute_forces` and `compute_postprocess` share one source-doublet
surface-velocity evaluation, so the two paths cannot disagree and a NEUMANN solve's source
strengths reach the force integration.

**Evidence for.** Both production kernels are correctly normalized with correct second moments;
the two zero-second-moment kernels correctly declare m₂ = 0; core-spreading constants now follow
from those moments and are pinned by tests; the stretching forms match Winckelmans & Leonard
1993; the VLM→VPM circulation transfer provably satisfies Kelvin's theorem; the Smagorinsky
algebra checks out. Two crash-on-default-path defects are fixed and pinned. ~970 lines of dead
code — including two wrong-physics paths — are gone. Memory for a small case dropped 45 %. A
per-stage profile and a deterministic benchmark now exist. `ruff` is clean.

**Limitations, all documented above.** The full suite has not been demonstrated stable across
repeated and shuffled runs, so no claim rests on it. A-1 leaves ~840 lines of acceptance-gate
auditing inside `Solver`, on the hot path, still able to abort a run. The core-spreading change
invalidates prior Winckelmans results. f64 is honest but still partial. All performance figures
are CPU-only.

**Three claims from earlier drafts were withdrawn as unsound** — order-dependent tests, the
relaxation gate-budget restart hole, and the framing of `256/45` as an unexplained constant
(it was hand-calibrated). Each was stated with more confidence than the evidence supported; the
corrections are recorded in place rather than quietly dropped.

Stage C's remaining item (A-1) is best run as its own scoped pass against a validated baseline,
which is exactly the sequencing the review proposed.
