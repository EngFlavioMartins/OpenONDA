# VLM–VPM interaction: completion plan for Worker 1

> **Superseded by the user's narrowed request. Do not execute the campaign below.**
> The current task is only to decide whether crossing particles should be deleted or retained, based on physical justification. Retain particles under normal transport by default: contact deletion introduces an unmodeled vorticity sink and generally changes impulse. Existing evidence does not demonstrate that deletion improves physical accuracy. Establishing a case-specific accuracy advantage would require a matched independent reference, which the current campaign does not supply. This is a stopping conclusion, not an instruction to build another solver or qualification campaign. The earlier plan is retained below only as history.

## Objective and meaning of one pass

Complete the original plan in `/Users/flaviomartins/.codex/attachments/30e21763-ba1c-4a09-a12a-a8447116e5e1/pasted-text.txt`, including its acceptance criteria and section 9. This document supplements that plan; it does not replace or reduce its scope.

One pass means **one assignment containing implementation, internal repair iterations, qualification, and final review**. It does not mean one simulation or a promise that the present physical model will pass. Continue through failed numerical gates by diagnosing and fixing their causes. Only an actual external prerequisite may require a blocker report. Neither a time-consuming study nor a failed physical gate is completion.

Do not modify particles to hide penetration. Preserve stretching choice, global multi-surface coupling, exchange accounting, and existing unrelated work. Preserve the already deleted original repository plan. Keep lagged and responsive results explicitly distinguished.

## Starting evidence: what is and is not established

The reviewed evidence is in `tutorials/vpm/08_surface_interaction/studies/`:

- The seven-case campaign used two baseline steps, or four half-size steps: all physical runs ended at 0.02 s. The incoming ring centres remained at x = −0.398 to −0.311 m; the first wing begins at x = 0. Consequently the campaign did not exercise the requested encounter.
- Complete-field transport leakage was approximately R1 = 0.04860 and Rinf = 0.15343, almost identical between lagged and responsive runs. Surface refinement did not establish a decreasing transport leakage error.
- Halving the timestep changed the front-wing recorded peak lift from 9.8058 to 18.5701. Investigate initialization and the unsteady-load difference before interpreting this as interaction behavior.
- Worker logs show 74 disjoint relevant regression tests passing, plus installed-tutorial checks. Saved two-step restart position, strength, and bound-circulation arrays agree exactly. Preserve this progress, but it does not establish long-encounter accuracy or full restart/history equivalence.

The companion `vlm_vpm_completion_contract.json` defines the completion gates and proposed engineering accuracy targets. These targets are not universal physical constants. Freeze the benchmark definitions, references, normalizations, exclusions, and targets before collecting acceptance results. An inadequate target requires an explicit contract revision and a new evidence campaign, not retrospective relabeling of a failure.

## 1. Make premature completion mechanically impossible

**Owner:** `tutorials/vpm/08_surface_interaction/assets/run_qualification_studies.py`, a new `assets/validate_qualification.py`, and focused validator tests under `tests/tutorials/`.

1. Preserve the existing campaign as an immutable, labeled short-run baseline. Do not overwrite it with the next campaign. Inventory the current worktree, relevant source hashes, dependency versions, device, exact commands, and source changes since the original pinned commit. Baseline and final evidence must identify the actual source contents, not only the unchanged HEAD of a dirty tree.
2. Implement a validator before further numerical work. It must recompute gate metrics from CSV/HDF5 data and actual accepted clocks. It must not trust `smoke=false`, requested step counts, a solver exit code, nonzero telemetry, or a pre-written `pass=true`.
3. Write `acceptance.json` with every required gate, measured value, threshold, units, evidence paths/hashes, and status. Missing evidence, nonfinite required values, insufficient time coverage, mismatched sources/contracts, and unresolved failures must return a nonzero exit status. A skipped mandatory gate is not a pass.
4. Separate `smoke` from `qualify`. Smoke runs cannot issue a completion certificate. The qualification command accepts a frozen campaign specification with physical endpoints and mesh/time levels; it cannot silently use `--full --steps 2` as qualification.
5. Add negative tests using the existing two-step dataset or small equivalent fixtures: reject an encounter that has not occurred, absent refinement levels/reference, changed input hashes, omitted budgets, nonfinite data, and failed restart. These tests verify the validator's decisions rather than its implementation structure.
6. Maintain a checklist mapping every original-plan acceptance paragraph to a gate and evidence path. A conditional item may be marked inapplicable only with the numerical evidence demonstrating its condition is false. Do not mark mandatory reference, encounter, or convergence gates inapplicable.

**Exit gate G0:** the present short-run evidence is correctly rejected as incomplete; genuine small positive and negative validator fixtures behave correctly.

## 2. Close numerical-contract defects before expensive studies

**Owner:** `solver/vlm_solver.py`, `solver/field.py`, `physics/stage_rhs.py`, `numerics/runge_kutta.py`, associated influence/emission kernels and existing focused tests.

1. Derive one consistent temporary newborn-row operator with its circulation-dependent matrix **and** old-circulation/accepted-exchange RHS contribution. The accepted `_near_wake_particle_influence` returns a matrix and `old_velocity`, while `_near_wake_stage_influence` currently returns only a matrix. Resolve that difference against the selected discretization; do not treat a nonzero matrix norm as verification.
2. Verify the virtual row's field wherever it belongs: boundary solve, particle velocity, Jacobian/stretching, and diagnostics. Its appearance in the boundary matrix alone is insufficient. Compare the virtual contribution with independently assembled actual particle sources at several stage fractions, including zero and one, with nonzero old circulation and all exchange-vector components. Check that the completed row is not counted again after insertion.
3. Include moving trailing-edge history and correct relative endpoint transport. The current stage offsets use incident velocity times elapsed time; determine and demonstrate the order of that approximation. Derive and test startup, elapsed stage time, deferred emission, and restart behavior. No newborn element receives a second full birth-interval advection.
4. Give diagnostic queries an explicit zero publication weight. `_weight_for_stage` currently allows `stage_index=None` to consume a sequential legacy weight when strength rates are enabled. Remove that ambiguity in the production contract and test a diagnostic inserted before/between real stages. For iterative stage evaluation, publish the final accepted contribution, not simply the first query carrying an index.
5. Use actual nonempty stage particle fields in end-to-end tests. Cover equal-time/different-state stages, changed radii/count, rejected trials, accepted history and output purity, release cadence, initialization, and DIRECT/TRANSPOSED/MIXED preservation. Mocks are additional isolation tests, not the only stage-response evidence.
6. Independently verify bound and free-wake velocity/Jacobian evaluations at fixed parameters. Make the actual finite-target transport rule and diagnostic reference agree at multiple fixed radii. Keep absolute radii fixed across comparisons; q25/q50/q75 of different populations alone do not define the same operator.

**Exit gates G1–G2:** independently evaluated row/field equivalence, Jacobian agreement, and accepted-state/exchange purity pass; no unexplained stage-state dependency remains.

## 3. Resolve initialization and establish temporal accuracy

1. Declare a physically consistent initial bound/free wake state for a nonempty incoming ring. Use either a documented steady preconditioning run or a smooth physical startup with identical dimensional ramp duration across refinements. Do not suppress the unsteady term or reset circulation history to conceal a startup impulse.
2. Record raw total loads, steady and unsteady components, startup impulse, and later encounter increments separately. Freeze an interaction window using geometry/physics before results are compared. Retain the full startup history even when it lies outside that window.
3. Use `circulation_relaxation=1` and explicitly disable force smoothing for acceptance runs. Enable the existing unsteady-force term. Keep viscosity, remeshing, and optional stabilization disabled for the initial inviscid reference campaign.
4. Establish a smooth independently known boundary-response/trajectory reference. The reference must exercise the production stage solve and varying incident field; it must not be generated by the same discrete production operator being tested. Select and document an analytic/manufactured reference or a separately verified inviscid discretization before running the acceptance sweep. Quantify its uncertainty. Existing filament quadrature is valuable component verification but does not replace this coupled reference.
5. Run h, h/2, h/4, h/8 to the same physical endpoint on fixed spatial data, with sufficient resolution to separate temporal error from spatial error. Demonstrate observed order of at least approximately two for smooth boundary response and trajectories (contract threshold 1.8 on the finest usable ratios), reporting the norm and reference error. When an error reaches its independently measured numerical floor, document that floor rather than inventing an order from roundoff.
6. Measure full shed-wake and force order separately. Control physical wake release spacing and core scales while changing integration step, or label the sweep as coupled temporal/spatial refinement and supply a separate clean temporal experiment. A shrinking birth row/core is not a pure temporal study.

**Exit gate G3:** the smooth temporal reference passes; startup and encounter loads are distinguishable; full-scheme order and any remaining first-order component are honestly quantified.

## 4. Define a campaign that reaches the physical event

**Owner:** existing tutorial `setup.py` as a declarative case definition; study, reference, and analysis implementation in `assets/`.

Run these configurations with fixed definitions, preserving upstream/receiving-wing provenance:

| Case | Required purpose and endpoint |
|---|---|
| Smooth stationary and moving plate | Independent boundary, trajectory and frame checks from section 3 |
| Ring/packet interacting with one wing | Complete approach, closest encounter, and departure; isolates upstream shedding errors |
| Offset tandem wings | Requested mild interaction; track both loads, upstream wake, and incoming-ring deformation |
| Stronger tandem encounter | Receiving-surface interior interaction; exposes the existing representation's validity limit |
| Two-way controls | Same incoming state with receiving wing removed, and a documented one-way/frozen-response control; compare against full coupling |
| Restart and precision repeats | Restart during the active encounter; f64 CPU reference and supported f32/backend comparisons |

Use actual positions to validate encounter coverage. Define upstream, receiving, and downstream planes from the frozen geometry. Follow identifiable ring/source groups and control volumes; require recorded approach, minimum separation, and departure, not merely a centroid at a requested time. For an outboard bypass, prove the intended near-edge encounter occurred. The stronger case must not be replaced by an easier bypass.

For the current geometry, U∞ = 4 m/s, ring centre x = −0.45 m, and rear-wing chord extending approximately to x = 1.2 m, a **0.6 s pilot horizon** is a reasonable planning estimate, not an acceptance criterion. At h = 0.005 s this is 120 steps, not two. Extend the horizon when actual trajectories or load recovery show the event is unfinished. The validated departure plane should exceed the rear trailing edge by at least one reference chord; document a finite post-encounter load window.

Select the coarse timestep from the solver stability limit and measured displacement relative to local panel/core scales. Recompute that limit for refined meshes. Fail qualification on unresolved stability warnings rather than accepting the previous h = 0.01 s warning. Pilot runs estimate cost and capacity, but cannot close scientific gates.

Save time histories and spatial states around the encounter, including particle support/extents and receiving-wing forces. Compare the receiving-wing-on/off wake deformation and load differences with discretization uncertainty. Comparing responsive versus lagged alone does not isolate receiving-wing feedback.

**Exit gates G4–G5:** every required case covers its intended event; two-way feedback exceeds independently estimated numerical uncertainty on the selected nontrivial benchmark.

## 5. Make spatial/model failure trigger implementation work

1. On the frozen resolved benchmark, evaluate independent interior probes and both trace sides for the point field and actual finite-target transport at fixed representative radii. Report physical edge/tip bands separately. Keep the band fixed in physical coordinates across refinements; do not enlarge it or exclude probes after seeing failures.
2. Refine chordwise and spanwise panels separately with at least three levels each, then refine particle spacing at fixed physical vortex profile/core and controlled overlap. Record sampling error of the initial ring so altered discretization is not mistaken for changed physical vorticity. If a distributed sheet is introduced, separately refine its quadrature.
3. Use the contract targets on the resolved smooth-interior benchmark: mean normalized transport leakage ≤0.005, maximum ≤0.05, converged trajectory difference ≤0.01 reference chord, and converged encounter-load measures ≤0.02 of their fixed characteristic scales. These are proposed engineering targets for this task, not claims of universal model accuracy. Require improving error trends above the numerical floor, not merely one passing fine case.
4. Compare with the preserved lagged baseline using the same initial state, geometry, operator, physical window, and normalization. Require a demonstrable improvement beyond estimated numerical uncertainty for the original leakage/interaction problem, without significant regression in qualified loads or far-field behavior.
5. If temporal convergence is established but spatial leakage or unresolved interior penetration persists, execute the original plan's distributed bound-surface escalation. Derive its circulation/potential-jump basis, edge attachment, regularization, velocity/Jacobian, exchange and impulse mapping. Replace the corresponding source representation in both the solve and transport; never add it on top of equivalent filaments. Iterate until the predefined resolved benchmark passes.
6. Keep the stronger unsupported case in the report with its failure evidence. Such a limit can define the model's validity range, but it cannot excuse the absence of any passing intended interaction benchmark. Core overlap alone is not a material crossing; every claimed explanation of an interior event must identify the relevant reconstruction and supporting reference evidence.

**Exit gates G6–G7:** the resolved benchmark meets the leakage, trajectory and load targets with refinement evidence and no unexplained interior crossings. Retaining the old representation is justified by measured results, not by implementation cost.

## 6. Close budgets, refinement safety, and restart

1. Export vector-valued bound/free exchange and wake/bound/total impulse histories under the solver's documented conventions. Include imposed motion, external forces, shedding, and truncation terms. Do not equate particle-only impulse with total body force or assume every impulse quantity must stay constant in a forced flow.
2. Verify discrete exchange closure at an algebraic tolerance scaled by precision, operation count and conditioning. Independently derived force/impulse and reference errors must converge within the frozen physical accuracy targets. Compare every component, not only a scalar strength sum or its dominant direction.
3. Verify that diagnostics and rejected work leave particles, accepted histories, exchange, clock, IDs, RNG, wake buffers, and scheduled output unchanged. Implement safe prescribed global refinement first. If automatic retry/subcycling is needed for the selected algorithm, introduce a real trial/rollback boundary and test rejected versus equivalently refined accepted runs. Do not claim retry support from a strict observer or a delayed clock.
4. Classify advection, permitted emission, and diffusion/regeneration relocation separately. Exercise unsupported-viscosity detection without presenting random displacements as integrated material paths.
5. Restart during approach and peak interaction, including moving geometry and deferred emission where supported. Compare positions, strengths, radii, volumes, IDs/groups, bound/old/cumulative circulation, geometry history, exchange, loads, output time series and budgets. A two-step comparison of three arrays is insufficient.

**Exit gates G8–G9:** all ledgers and reference balances are accounted for, restart agrees under declared f64 tolerances, and optional retries are either verified or explicitly outside the shipped algorithm with safe prescribed refinement verified.

## 7. Final integration and the completion decision

1. Run the relevant VLM, stage, RK, exchange, evolution-transaction, boundary-refresh and restart suites in the repository's declared environment. Add focused tests only for new contracts and discovered failures. Preserve exact commands and result logs; do not add counts from overlapping selections as unique tests.
2. Run supported f32 comparisons through a real encounter. Report available GPU/backend execution separately; unavailability may be a documented coverage limitation under the original plan, not a fabricated pass.
3. Build a regular wheel from the frozen final source. Install it into an isolated environment and run the real maintained tutorial outside the repository without `PYTHONPATH` or shell startup modifications. Verify the actual imported package location. The installed run must cover the required encounter, not only launch successfully.
4. Keep durable raw evidence and source hashes outside paths removed by `allclean.sh`. Remove only temporary files owned by this assignment; do not delete the baseline or acceptance data. Update the permanent numerical-method documentation with the measured formulation, order and validity limits.
5. Run the acceptance validator after the final relevant code change. If that change invalidates earlier numerical evidence, rerun only the affected studies and dependent gates. No stale certificate is valid against different source/contract hashes.
6. Finish with the original section 9 report, generated from the evidence: source IDs and changed files, method, exact checks, baseline/final/refinement results, restart/installed evidence, validity and cleanup. Do not write “100% complete” unless all mandatory gates pass and every conditional decision is substantiated.

**Exit gate G10:** final `acceptance.json` passes, evidence is inspectable, and the report has no unresolved mandatory item.

## Worker handoff instructions

Execute this document and the original plan as one continuous assignment. Start with the acceptance validator, then close the numerical defects, then run the physically complete campaign. Treat any failed gate as the next implementation or diagnosis action. Do not stop after creating files, running smoke cases, passing unit tests, or merely producing tables. Keep short progress updates identifying the current gate and its evidence. Request user input only for a genuinely unavailable external prerequisite that prevents further useful work. Notify the user when G0–G10 and every applicable original-plan requirement are satisfied; otherwise report the exact unresolved gates without claiming completion.
