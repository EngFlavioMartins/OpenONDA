# VPM coupled-induction architecture certification

## Scope

This report certifies the implementation structure introduced by the coupled
induction refactor. It does not claim a full-repository release certification:
the repository-wide test and lint commands still include unrelated FVM and
historical study work.

Starting commit: `60fdb79b31c9a4a488972f579ba4815797bc02e8`

Final implementation commit: `8d5283f0` (`cleanup(vpm): finalize the coupled induction architecture`)

## Architecture result

The production information path is:

```text
VPMSolver.advance
  → EvolutionStepper.advance
    → RungeKutta.advance
      → StageRHS.evaluate
        → InductionMethod.evaluate_stage
```

The integrated state is `(position, vortex_strength)`. One common temporary
stage state is passed to the selected induction method, which writes velocity
and the canonical vortex-strength rate into preallocated output fields. The
optional velocity gradient is auxiliary work and is not used as a replacement
for the conservative strength equation.

`DirectInduction`, `TreecodeInduction`, and `FMMInduction` implement the same
stage boundary. RK2, SSPRK3, and RK4 are tableaus consumed by one generic
coupled RK engine. Diffusion remains outside that engine as symmetric operator
splitting. No fractional integration mode, separate advection/stretching
integrator, duplicate VPM setup, or old acceleration package remains.

## Public API result

`Numerics` is the only numerical construction carrier. The public VPM API
exposes `VPMCase`, `Numerics`, the coupled RK tableaus, the three induction
methods, and the physical/output configuration objects needed to build a case.
The removed advection, stretching, velocity, and private VPM setup objects are
not compatibility aliases.

## Commit sequence

```text
ed2adaa5 test(vpm): freeze the canonical coupled induction equations
5323308c refactor(vpm): introduce the stage induction contract
8d4482fb refactor(vpm): unify coupled Runge-Kutta evolution
bb1718c9 refactor(vpm): centralize particle stage-rate evaluation
11997d99 refactor(vpm): remove fractional advection-stretching evolution
6d34ace9 refactor(vpm): simplify VPM integration and induction configuration
b3aa6e2c refactor(vpm): isolate direct and treecode induction methods
d93aec68 refactor(vpm): define a shared radial vortex-kernel contract
910ca87f feat(vpm): implement the regularized vortex FMM hierarchy
c5f52b13 refactor(vpm): remove legacy integration configuration
a12e66d9 refactor(vpm): route derived fields through the stage contract
bf456e19 fix(vpm): preserve configured precision in coupled RK steps
80230501 fix(vpm): complete declarative configuration migration
8d5283f0 cleanup(vpm): finalize the coupled induction architecture
```

## Files added and removed

Added implementation areas include:

- `source/solvers/vpm/physics/induction/` for the common contract and methods;
- `source/solvers/vpm/physics/induction/fmm/` for the hierarchy;
- `source/solvers/vpm/kernels/base.py` for the radial-kernel contract;
- `source/solvers/vpm/numerics/rk_tableaux.py` and `runge_kutta.py`;
- `docs/vpm_induction_operator.md` and
  `docs/vpm_induction_qualification.md`.

Deleted implementation areas include the separate advection/stretching and
velocity configuration modules and the obsolete `source/solvers/vpm/acceleration/`
package. Its active LBVH implementation now lives at
`physics/induction/treecode/lbvh.py`.

The source-only diff from the starting commit is `+1,391/-3,019` lines
excluding the FMM directory (net `-1,628`), and `+1,812/-3,019` including FMM
(net `-1,207`). The larger repository diff also includes permanent tests,
qualification records, documentation, and public/tutorial API migrations.

## Qualification evidence

The focused induction qualification passes:

```text
tests/vpm/test_induction_operator.py
tests/vpm/test_vortex_kernel_contract.py
tests/vpm/test_fmm_hierarchy.py
tests/vpm/test_core_numerical_qualification.py
```

The complete implementation-scoped suites pass with 166 VPM tests and 164
coupler tests. The maintained individual lamb--Oseen, storage/output, and
tutorial-schema test files also pass. VPM source Ruff checks and test
collection pass.

The FMM qualification covers deterministic hierarchy construction, expansion
helpers, shared-kernel near interactions, tolerance trends, and a hierarchical
strength-rate path with zero direct-fallback count. Its far-field velocity and
gradient use second-order singular Biot--Savart source moments, while near
interactions use the shared regularized kernel. FMM remains explicit opt-in
because this reference evaluator is host-oriented; no 14,080- or
70,200-particle performance certification was performed for this path.

## Known validation limits

- A full repository `pytest -q` run completed collection but retained 11
  order-sensitive or unrelated failures outside the implementation-scoped VPM
  and coupler suites; the individual affected files pass in isolation.
- Repository-wide Ruff remains blocked by 485 pre-existing style/unused-import
  findings concentrated in the historical VPM study scripts and tests.
- The final worktree intentionally retains the user's unrelated FVM/tutorial
  edits unstaged.
