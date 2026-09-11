# OpenONDA: VLM–VPM wake–surface interaction implementation plan

## Assignment

Improve the treatment of an incoming particle wake interacting with another lifting surface, including two delta wings in tandem. The objective is a demonstrably better coupled flow solution, not a collision rule that hides penetration.

Repository: `EngFlavioMartins/OpenONDA`, branch `development`.

Review baseline: `927076d6f9fd9bed0b5dd38789011cd3260bb3bc`.

This plan is based on a static inspection of the baseline source and selected tests. The reviewer has not executed the solver or its test suite and has not changed the repository. Begin by reproducing the baseline. If development has advanced, inspect the intervening diff and adapt the implementation rather than resetting someone else's work.

### Intended scope

Deliver improved and qualified **inviscid/attached lifting-surface interaction** within VLM–VPM, with explicit diagnostics identifying under-resolved or unsupported surface encounters. Do not represent particle exclusion as a model of a viscous boundary layer, wall-vorticity generation, or separated delta-wing leading-edge vortices. Those require a separately justified extension or near-body Eulerian coupling.

Do not simultaneously undertake an FVM, mesher, LES, or repository-wide refactor. Changes should follow the dependency chain below and remain reviewable.

## 1. Baseline evidence and what must be preserved

Paths in the following table are relative to `source/solvers/vpm/` unless otherwise stated. References identify source locations at the pinned commit, not claims that their tests have passed.

| Source | Observed behavior | Consequence for this task |
|---|---|---|
| `boundary_elements/vlm/solver/influence.py`, `compute_aerodynamic_influence_coefficient_matrix` | One matrix includes all panels. In coupled mode, internal horseshoe legs extend to the strip trailing edge. | Preserve global multi-surface interaction and existing wake ownership. Do not introduce independent, uncoupled wing solves. |
| `boundary_elements/vlm/kernels/biot_savart.py`, `regularized_segment_velocity_and_gradient` | The Rosenhead finite-segment velocity and its analytic target Jacobian share one implementation. | Do not replace this with inconsistent velocity and stretching kernels. |
| `boundary_elements/vlm/solver/influence.py`, target and stage kernels | Collocation/probe induction uses `VLM_EPSILON`; particle stage induction uses `max(particle_core_radius, VLM_EPSILON)`. | The boundary solve and particle transport do not sample the same regularization scale. Quantify the effect before selecting a replacement policy. |
| `physics/stage_rhs.py`, `VLMStageContribution`; `boundary_elements/vlm/solver/vlm_solver.py`, stage methods | Particle locations and prescribed surface geometry are evaluated at stage time, but solved circulation remains lagged. | Add a stage-state boundary response without repeating accepted-step mutations. |
| `boundary_elements/vlm/solver/vlm_solver.py`, `solve` | Newborn-wake augmentation is conditional on both `coupled` and `save_old`. | `solve(save_old=False, coupled=True)` is not automatically an equivalent coupled stage solve. |
| `boundary_elements/vlm/solver/vlm_solver.py`, `advance_coupled` | Moves accepted geometry, solves circulation, inserts a completed wake row, and refreshes forces. | Never call this method at every RK stage. Extract separate solve and commit operations. |
| `boundary_elements/vlm/solver/vlm_solver.py`, `particle_transport_step`; `influence.py`, exchange kernels | A stage-weighted bound/free vector-strength exchange is accumulated and consumed by wake emission. | Preserve and re-derive this accounting when changing the numerical model. Algebraic balance alone does not validate wall physics. |
| `coupling/stepper.py`, `advance_vlm` | `_release_wake_particles=False` returns before the boundary solve. | Trace callers and separate boundary-update cadence from wake-particle insertion cadence. |
| `boundary_elements/vlm/kernels/collision.py` | Tests current-position proximity to a finite quad, not swept trajectories; uses an unsigned projection distance and f32 point arguments. | Replace contact handling with precision-aware, non-mutating observation of finite-surface events. |
| `boundary_elements/vlm/solver/vlm_solver.py`, `absorb_particles` | An explicit method deletes particles near panels. | Do not use this as impingement physics. No invocation was observed in the inspected normal VLM evolution/coupling path; establish all actual call sites first. |
| `core/evolution.py` | Particle RK and split diffusion/stabilization precede the accepted VLM update; regeneration can replace the particle population. | Observe advective crossings before topology changes, and classify diffusion/regeneration separately. |
| `boundary_elements/vlm/config.py`, `ForceConfig` | Unsteady pressure loads already exist and default to disabled. | Enable the existing term for interaction studies; do not implement it twice. |

Selected tests inspected include `tests/vpm/test_vlm_stage_fields.py`, `test_vlm_bound_exchange.py`, and the opening qualification tests in `test_vlm_qualification.py`. They include stretching-mode checks, independent line-integral/Jacobian checks, RK exchange accounting, and failed-stage behavior. These are useful numerical tests, not evidence that wake impingement is already qualified.

The current VPM documentation also distinguishes symmetric particle-pair core radii from source-radius arbitrary-target evaluation. The total boundary/transport comparison must therefore include the **free-wake operator**, not only bound-filament regularization.

## 2. Non-negotiable implementation constraints

- No default particle deletion, reflection, clipping, force saturation, or normal-velocity clamping to make an encounter appear successful. No automatic transfer of an incoming particle's strength to a wing.
- Preserve the selected stretching formulation. Do not silently change DIRECT, TRANSPOSED, or MIXED while modifying surface interaction.
- Keep one owner for accepted circulation history, geometry history, wake insertion, particle state, sampling, and restart. Temporary stage solutions must not become accepted state merely because a velocity query was made.
- Do not double-count bound induction using both the old filaments and a new sheet representing the same circulation. Do not add downstream horseshoe legs on top of the VPM-owned wake.
- Do not claim an RK4 coupled scheme solely because the particle integrator uses RK4. Boundary response, near-wake construction, shedding, motion, and force time differences must be assessed separately.
- Keep new controls declarative and small. Do not retain several almost-identical production coupling pipelines, tutorial-only numerical patches, or compatibility wrappers without a concrete need.

## 3. PR 1: establish a reproducible interaction baseline and observer-only diagnostics

### 3.1 Reproduce the baseline

Record the actual SHA, numerical precision, device, induction backend, kernel, stretching scheme, time step, surface resolution, wake-emission spacing, core radii, viscosity, and relevant stabilization settings. Inventory all call sites of `absorb_particles`, `detect_surface_collisions_kernel`, `_release_wake_particles`, and `_release_interval`.

Run the existing focused tests before algorithm changes. For example, from a correctly installed checkout:

```sh
python -m pytest tests/vpm/test_vlm_*.py \
    tests/vpm/test_coupled_runge_kutta.py \
    tests/vpm/test_stage_rhs.py \
    tests/vpm/test_evolution_transaction.py \
    tests/vpm/test_boundary_element_state_refresh.py
```

Use the repository's declared installation and test environment; do not invent dependency versions or extras. Report pre-existing failures separately. Record actual commands and results, including skipped or unavailable backends.

Begin with a deterministic inviscid case using direct induction and f64 on CPU. Use circulation relaxation of one and disable force smoothing to expose rather than filter numerical errors. Disable optional topology-changing stabilization for the first diagnostic comparison. Later repeat with the supported production settings.

### 3.2 Measure boundary leakage independently

Extend the existing VLM diagnostics and sampling infrastructure, rather than creating a second logger. At surface probes define

`r_k = (u_total(x_k, t) - u_surface(x_k, t)) dot n_k`.

Report at least

`R1 = sum(w_k * abs(r_k)) / (U_ref * sum(w_k))`

and

`Rinf = max(abs(r_k)) / U_ref`.

Use a documented nonzero reference speed for moving or nominally quiescent cases. Keep algebraic linear-system residual, collocation physical residual, and independent off-collocation physical residual as separate quantities.

Probe a denser set than the solve points and sample both sides where a trace limit is needed. Include all surfaces and the free wake. Compare the point/probe reconstruction with the actual particle-transport reconstruction at representative particle radii. Label each diagnostic by its operator and filter scale; never compare unlike fields under one residual name.

Define an edge/tip band before the study. Report it separately from the smooth interior because sharp-edge behavior can dominate a maximum. Do not silently remove problematic probes or widen the excluded band until a test passes.

Diagnostics must use the matching accepted or temporary stage state. They must not overwrite circulation, consume RK weights, change exchange ledgers, insert particles, or modify the accepted clock.

### 3.3 Detect finite-surface events without mutating particles

Implement a reusable finite-surface observer in or next to `kernels/collision.py`. Separate the geometry operation from any policy deciding whether to warn or stop.

For planar projection use a signed distance:

```python
signed_distance = (position - panel_origin).dot(unit_normal)
projected_position = position - signed_distance * unit_normal
```

The existing unsigned formula is geometrically incorrect below the plane. However, for an exactly planar quad the subsequent cross-dot inside test can be insensitive to that normal offset. Do not present this small correction as the main cure for wake crossing.

Replace hardcoded f32 geometry arguments with the active precision. Use scale-aware geometric tolerances. Test both panel windings, rotation/translation, finite edges, nearly grazing motion, endpoints on the surface, and degenerate/warped geometry. Triangulate valid quads consistently; handle a valid triangular apex explicitly or reject a degenerate panel with a useful error.

Distinguish these events:

1. A transported centre intersects the finite wing interior.
2. A trajectory goes around an edge and changes side without intersection.
3. A particle core overlaps the surface while its centre stays outside.
4. A newly emitted particle begins at a permitted shedding location.
5. Diffusion or regeneration creates or relocates support across a surface.

Do not connect raw RK stage positions as though they were successive physical trajectory samples. RK4 contains equal-time stages, and SSPRK3 stage times need not be monotone. Use accepted substep segments or a verified dense-output trajectory, with a documented temporal approximation. Include moving surface geometry rather than testing against only its final position.

Collect particle identity/provenance according to the existing identity contract, receiving surface/panel, time interval, event type, crossing position, strength magnitude, and local core/mesh scales. Do not mistake a group ID for a unique particle ID. Observe advection before regeneration changes population identity.

Keep diagnostics in existing scheduled CSV/VTK output and health-reporting mechanisms. A strict scientific mode may raise on unresolved penetration. A warning mode must retain and report the event, not modify the solution. Remove an unused absorption API and imports after call-site review; any deliberately retained experimental sink must be unmistakably nonphysical and report removed strength and impulse.

**Acceptance:** observer-only runs reproduce baseline numerical state; true crossings with endpoints on opposite sides and outside the proximity tolerance are detected; edge bypasses are not classified as penetration; diagnostic probes do not mutate exchange/history; no new contact-induced sink exists.

## 4. PR 2: define and test the boundary-field/transport contract

Primary files: `boundary_elements/vlm/solver/influence.py`, `boundary_elements/vlm/kernels/biot_savart.py`, `boundary_elements/vlm/solver/vlm_solver.py`, `physics/stage_rhs.py`, and the relevant target-evaluation interface under `physics/induction/`.

The immediate issue is not two unrelated segment kernels: the existing code shares the Rosenhead formula but evaluates it at different core radii. Preserve the analytic velocity/Jacobian consistency while making that distinction explicit.

### 4.1 Separate representation, evaluation, and regularization

Introduce one internal owner/interface for the bound-surface field. Its contract must identify geometry, circulation coefficients, source representation, regularization, velocity, Jacobian, and boundary traces. Refactor existing kernels behind that contract before adding an alternative physical model.

Name and document three distinct scales: a numerical singularity safeguard, any bound-source regularization width, and the free-particle core radius. Do not use one parameter as an undocumented substitute for all three.

Perform controlled comparisons showing how normal transport and particle trajectories change with target core radius, surface resolution, and near-surface distance. Include the free-wake contribution: the current particle-pair and arbitrary-target smoothing rules also differ.

### 4.2 Select a defensible boundary formulation

Preferred direction for a single impermeable transport field: a source-defined surface representation and an explicitly consistent reconstructed velocity field, with the boundary condition and its Jacobian evaluated under that contract.

If target-volume filtering is retained, document the filtered boundary condition and the distinction between filtered particle-centre transport and material trajectories. A single circulation vector generally cannot be assumed to enforce pointwise no-through for every arbitrary target radius simultaneously. Quantify convergence to the intended boundary condition, or provide a justified boundary-aware filtering formulation.

Do not make all particle targets use an almost singular epsilon core merely to force nominal field equality. Do not fix the matrix using an arbitrary mean particle radius. Do not apply a local velocity projection without deriving and verifying the corresponding total velocity field and Jacobian.

Ensure matrix assembly, RHS construction, newborn-row influence, stage field evaluation, force-target evaluation, and diagnostics use the intended representation consistently. Where a force model requires a particular self-term exclusion or trace, document and test it rather than applying an accidental exception.

Maintain existing far-field behavior and source orientation under refactoring. Any material change of numerical model must be explicit in configuration and restart identity.

**Acceptance:** independent quadrature verifies each retained segment/operator; finite differences verify the returned Jacobian at fixed model parameters; the boundary and transport residual relationship is documented and tested; a smooth, resolved encounter improves without force filtering or particle correction. A small collocation residual alone is not acceptance.

## 5. PR 3: separate pure stage boundary response from accepted wake emission

Primary files: `physics/stage_rhs.py`, `boundary_elements/vlm/solver/vlm_solver.py`, `coupling/stepper.py`, `core/evolution.py`, `numerics/runge_kutta.py`, and `boundary_elements/vlm/coupling/kinematics.py` as required.

### 5.1 Extract the operations without changing the baseline method

Split responsibilities into temporary geometry evaluation, boundary-system assembly/solve, temporary field evaluation, and accepted-step history/emission/force commit. Method names are an implementation choice; do not add a large public API for internal scheduling.

A stage solve must receive stage positions, vector strengths, radii, count, time, and a revision/identity for that state. Do not read the old accepted particle object or an acceleration structure built from it. Equal-time RK stages can have different particle states. A cache keyed only by stage time is therefore insufficient for circulation response.

Temporary geometry must include collocation points, normals, body velocities, corners, and bound-source geometry, not only the corners and filament points needed for lagged field evaluation.

### 5.2 Implement stage-responsive coupling with explicit near-wake ownership

Solve all interacting surfaces together from the actual stage incident field. Store temporary circulation separately from accepted circulation and its old/cumulative histories.

Before implementing stage response, write down the chosen temporal discretization of the near wake. The baseline adds a completed newborn row implicitly to the accepted circulation solve. A stage algorithm must consistently account for the partial/newborn row and its circulation dependence, or use a clearly defined coupled substep formulation with a verified order of accuracy.

Do not omit newborn influence accidentally because `save_old=False` bypasses its augmentation. Do not add a virtual row to both RHS and matrix, or count an already emitted row again. Keep the native row-influence/emission equivalence verified against actual particle sources.

Do not call `advance_coupled` from a stage evaluation. Permanent particle insertion and accepted history updates occur once per accepted interval under the chosen formulation. Newborn elements must not receive a second full birth-interval advection step.

Decouple the frequency of circulation response from wake insertion. Disabling or batching emission must not silently freeze the wing response. Preserve the correctly accumulated release interval and history when emission is deferred. If a release policy is incompatible with the new formulation, reject it clearly rather than silently skipping the physics.

Assess near-wake endpoint transport as well as bound circulation: the current one-velocity row-offset construction can limit temporal accuracy independently of RK. Improve that construction where required by the selected coupled method.

### 5.3 Preserve exchange bookkeeping and observational purity

The existing `VLMStageContribution.integration_step` advances an iterator over RK weights. Additional diagnostics, predictor iterations, or repeated stage solves must not consume extra weights. Make stage identity/weight explicit where needed rather than relying on the number of arbitrary field queries.

Keep exchange contributions local to trial work and publish only the actual accepted quadrature. A rejected predictor or step must not double-count reaction. A field-only query has zero exchange weight and cannot alter histories.

Preserve the selected stretching contraction. Re-derive the bound/free bookkeeping for any new surface representation. Test the vector-valued exchange, including components not parallel to a trailing-edge closing segment. Do not replace that exchange with an unexplained global circulation repair.

### 5.4 Loads, initialization, and restart

Initialize a physically consistent bound response before transporting a nonempty incoming wake. Keep starting circulation and initial shedding explicitly defined.

Use `ForceConfig.kutta_joukowski(unsteady=True)` for transient qualification. The existing pressure-time term uses a backward difference: distinguish that load discretization from the order of particle transport. Do not silently claim high-order loads; improve accepted-time differencing only with a corresponding startup, variable-step, and restart contract.

Update restart fingerprints and any newly persistent accepted histories. Rebuild temporary stage workspaces after restart. Preserve uninterrupted/restarted equivalence.

**Acceptance:** actual stage incident changes alter stage circulation; two equal-time but different states do not share a stale solution; no stage query emits particles or shifts accepted history; newborn-row induction matches emission; release cadence does not suppress boundary response; exchange tests and restart equivalence pass. Demonstrate at least second-order behavior for a smooth boundary-response/trajectory verification problem and separately report the measured order of the full shed-wake and force discretizations.

## 6. PR 4: improve the surface representation only if spatial leakage remains material

Primary locations: the new internal field interface; `boundary_elements/vlm/solver/influence.py`, `solver/lattice.py`, `solver/mesh.py`, `solver/diagnostics.py`, `solver/restart.py`, and `config.py`.

First determine whether the existing chordwise-resolved lattice converges adequately after the preceding corrections. OpenONDA already has chordwise horseshoes and is not simply a one-line wing model.

If unacceptable transport leakage remains after temporal refinement, implement a distributed bound-surface model behind the same field interface. This is the recommended escalation within VLM–VPM, not an instruction to invent a particle repulsion layer.

Use a documented surface-vorticity or potential-jump basis and verify its signs, units, normal orientation, edge behavior, and mapping from the existing solved circulation. Derive how internal trailing vorticity and free-wake attachment are represented. Merely adding arbitrary sheet particles to existing filaments is not acceptable.

Where quadrature sources are used for bound vorticity, keep them owned by the moving surface, not by the advected/free particle population or its remesher. Evaluate both velocity and Jacobian from the same sheet representation. Include that representation in the boundary solve, rather than replacing only particle advection while leaving the solve unchanged.

When changing representation without changing the intended model, preserve the relevant circulation and strength/first-moment quantities to the documented accuracy. When changing the model, report rather than hide its effects on these quantities and on far-field induction. Preserving total vector strength does not by itself preserve impulse.

FLOWUnsteady's actuator-surface model is a relevant design precedent: it spreads bound vorticity to reduce cross-centreline flow under strong wake interaction. Its documented objective is to minimize that flux, not prove exact solid-wall impermeability. Do not copy its smoothing constants without a resolution study, and do not label a distributed actuator surface as a resolved no-slip wall.

**Acceptance:** the new model reduces independently measured leakage at comparable resolution/cost and improves convergence of trajectory and loads, with no duplicated bound field and no artificial particle correction. Promote it to the normal interaction configuration only after qualification. If a particular core-on-surface encounter remains outside the demonstrated validity range, report that limit and flag the case rather than silently making it pass.

## 7. PR 5: add safe temporal refinement, not collision repair

Use the finite-surface observer to distinguish temporal under-resolution from spatial/model leakage. A smaller step should reduce trajectory-integration error; it cannot repair a velocity field that points through the wing.

Initially support convergence runs with smaller prescribed time steps and strict diagnostics. Add automatic retry/substepping only after a real rollback or trial-state boundary is implemented and tested. Delaying the accepted clock is not equivalent to undoing all mutated physical state.

A retried interval must handle particle fields, surface circulation/geometry histories, exchange ledgers, wake buffers, IDs, diffusion state/RNG state, and scheduled outputs consistently. Alternatively, keep trial work confined to an inviscid interval before irreversible operators, and document that scope.

Prefer a globally synchronized near-encounter substep first. Do not independently subcycle selected particles while ignoring their mutual interaction and the wing response. Re-solve the boundary at the appropriate substep/stage state.

A bounded retry budget must terminate with a diagnostic error when leakage persists. No infinite time-step collapse, silent particle removal, or relocation is allowed. Do not interpret RWM displacements or grid-remeshing relocation as deterministically integrated material paths. Until there is a justified surface-aware diffusion treatment, identify unsupported viscous impingement configurations explicitly.

**Acceptance:** retried and equivalently refined runs agree within expected numerical tolerance; rejected attempts do not emit particles or samples, alter budgets, or advance accepted time; persistent spatial/model leakage produces a useful failure rather than a fabricated flow field.

## 8. PR 6: qualification, tutorials, integration, and cleanup

### 8.1 Focused scientific qualification

Use four related cases rather than many scenario-specific assertions:

| Case | Purpose | Required observations |
|---|---|---|
| Smooth prescribed incident field / simple stationary and moving plate | Isolate boundary operator and temporal response | Collocation/off-grid residual, stage response, frame invariance, independent reference where available |
| A resolved vortex packet or ring passing near and interacting with one wing | Isolate wake distortion and surface response without an upstream shedding error | Vortex path, strength distribution, velocity field, transient loads, leakage, finite-surface events |
| Two tandem delta wings, with an offset case and a stronger-interaction case | Demonstrate the requested application | Both wings' loads and wake response; upstream/downstream provenance; no disappearance or clipping of incoming wake |
| Restart and supported-backend repeat of a selected encounter | Establish reproducibility and integration | Accepted histories, forces, particle state, counts, budgets, and diagnostic continuity |

Use independent data appropriate to the model: an analytic/manufactured reference where possible, a separately verified inviscid solver for inviscid comparisons, or resolved/experimental data for clearly identified physical validation. Agreement with a finer run of the same code is convergence evidence, not independent physical validation.

Test a genuinely two-way effect, not only upstream wake loading the downstream wing: the receiving wing's response must alter the incoming wake. Preserve the existing all-surface matrix interaction.

### 8.2 Separate refinement effects

Refine time, chordwise/spanwise surface discretization, particle spacing/core overlap, and any new sheet quadrature separately before a combined refinement.

Changing the baseline time step also changes near-wake row length and can change emitted core radii. A pure temporal study must hold wake representation/emission resolution appropriately controlled, or explicitly identify the study as a coupled temporal/spatial refinement. Compare the same physical vortex core, not increasingly diffuse vortices mislabeled as increasing accuracy.

Report trajectory displacement, lift/moment histories, peak and integrated load, leakage statistics, event counts, vector-strength budgets, and wake/bound/total impulse under the conventions already used by the solver. Include domain truncation, diffusion, regeneration, and external-force effects when applicable. Do not equate particle-only impulse with a complete solid-body force balance without the required derivation.

Specify acceptance tolerances before promoting the algorithm. Base algebraic tolerances on precision/conditioning and scientific tolerances on the reference uncertainty and intended use. Do not set every tolerance to machine epsilon or loosen thresholds after looking at failures. Require a predefined resolved benchmark with no unexplained interior crossing events, plus converging loads and trajectories. No universal guarantee for every geometry/core/viscosity combination is implied.

### 8.3 Test organization

Extend existing tests where they already own a contract. Suggested new files, only where no existing test is a suitable owner:

- `tests/vpm/test_vlm_surface_interaction.py`: finite geometry events, observational purity, off-grid leakage, radius dependence.
- `tests/vpm/test_vlm_stage_boundary_solve.py`: actual stage state, history ownership, virtual/newborn wake, repeated queries and release cadence.

Keep heavy tandem-wing convergence studies in a reproducible tutorial/qualification workflow with a small deterministic smoke test. Reuse existing exchange, precision, qualification, unsteady-force, and restart tests. Do not replace numerical checks with tests that merely assert method names or exact implementation structure.

Add f64 CPU references and f32/backend comparisons within each backend's supported configurations. Report unavailable GPU verification honestly. Profile boundary assembly, solves, near-surface evaluation, and observation after correctness is established.

Cache fixed geometry/source operators only when valid. The complete coupled matrix includes newborn-row contributions that can change even for a stationary wing; geometry immobility alone does not make that entire matrix constant.

### 8.4 Tutorial and public configuration

Use one physics-focused tutorial, reusing an existing relevant case if available. A new case should use the established structure:

```text
<interaction-case>/
    allrun.sh
    allclean.sh
    allplot.sh
    setup.py
    assets/
```

Keep `setup.py` declarative and physics-oriented: geometry, placements/motion, incident wake, numerics, sampling, and finite run plan. Put plotters, reference data, and auxiliary studies in `assets/`. Keep shell scripts short and hard-coded; no MPI orchestration, path injection, elaborate argument parsers, or numerical implementation in user-facing files.

Use existing output scheduling and health-policy ownership. A small immutable surface-interaction configuration may expose the selected representation and diagnostic/failure policy. Proposed options are new API design, not currently existing names. Defaults must not silently change physical models.

Verify the installed tutorial from a directory outside the repository, without `PYTHONPATH` or shell startup modifications.

### 8.5 Documentation and cleanup

Update `docs/vpm.md` and relevant API docstrings with the selected boundary model, field/filter interpretation, stage/accepted-state sequence, wake ownership, diagnostic meanings, supported configurations, convergence procedure, and physical limitations.

Remove temporary implementation notes, phase files, dead experimental branches, duplicate loggers, and generated artifacts introduced by this task. Do not delete unrelated Markdown, established reference documentation, user data, or valid numerical tests. Keep one concise permanent numerical-method/qualification description in the appropriate existing documentation location.

## 9. Definition of completion

The agent's final report must contain:

1. Baseline and final commit identifiers, exact changed-file list, and the selected numerical formulation.
2. A concise account of what was preserved, what changed physically/numerically, and why.
3. Exact executed test and tutorial commands, actual pass/fail/skip results, and available/unavailable backend coverage.
4. Before/after and refinement tables for leakage, trajectory, loads, surface events, budgets, and runtime, with inspectable CSV/VTK/reference outputs.
5. Proof that diagnostics and rejected trial work do not alter accepted history, exchange accounting, wake insertion, or scheduled output.
6. Restart and out-of-repository installed-tutorial verification.
7. An explicit validity statement: qualified inviscid interaction configurations, unresolved configurations, and the distinction from viscous wall/leading-edge separation physics.
8. A cleanup summary restricted to files introduced or made obsolete by this work.

Do not certify completion using only a clean animation, a small linear-solver residual, the absence of a crash, or a passing pre-existing test suite. Do not claim tests that were not executed. If a physical validation gate fails, preserve the evidence and state the limitation; do not conceal it with particle manipulation.

## Source references

The source evidence above is pinned to the review commit. Useful permanent entry points:

- [VLM solver and stage/accepted-state operations](https://github.com/EngFlavioMartins/OpenONDA/blob/927076d6f9fd9bed0b5dd38789011cd3260bb3bc/source/solvers/vpm/boundary_elements/vlm/solver/vlm_solver.py)
- [Bound-field, boundary matrix, and exchange kernels](https://github.com/EngFlavioMartins/OpenONDA/blob/927076d6f9fd9bed0b5dd38789011cd3260bb3bc/source/solvers/vpm/boundary_elements/vlm/solver/influence.py)
- [Central stage provider](https://github.com/EngFlavioMartins/OpenONDA/blob/927076d6f9fd9bed0b5dd38789011cd3260bb3bc/source/solvers/vpm/physics/stage_rhs.py)
- [Existing collision kernel](https://github.com/EngFlavioMartins/OpenONDA/blob/927076d6f9fd9bed0b5dd38789011cd3260bb3bc/source/solvers/vpm/boundary_elements/vlm/kernels/collision.py)
- [Accepted coupling orchestrator](https://github.com/EngFlavioMartins/OpenONDA/blob/927076d6f9fd9bed0b5dd38789011cd3260bb3bc/source/solvers/vpm/coupling/stepper.py)
- [Evolution order and split diffusion](https://github.com/EngFlavioMartins/OpenONDA/blob/927076d6f9fd9bed0b5dd38789011cd3260bb3bc/source/solvers/vpm/core/evolution.py)
- [Existing independent bound-exchange tests](https://github.com/EngFlavioMartins/OpenONDA/blob/927076d6f9fd9bed0b5dd38789011cd3260bb3bc/tests/vpm/test_vlm_bound_exchange.py)
- [VPM numerical contract and filter conventions](https://github.com/EngFlavioMartins/OpenONDA/blob/927076d6f9fd9bed0b5dd38789011cd3260bb3bc/docs/vpm.md)
- [FLOWUnsteady actuator-surface design precedent](https://flow.byu.edu/FLOWUnsteady/examples/blownwing-asm/)
