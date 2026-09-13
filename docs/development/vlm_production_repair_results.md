# VLM implementation and qualification results

13 September 2026. Follow-up to the [repair plan](vlm_production_repair_plan.md)
and [original audit](vlm_production_audit.md). The original discovery evidence is
preserved under `studies/vlm_production_audit/`.

**Release decision: rotor production qualification remains open.** Concrete
software defects have been repaired and independently tested. A healthy,
converged full rotor with complete terminal induction histories has not yet
been demonstrated. The `PENDING` tutorial status is intentional.

## Implemented repairs

| Area | Result | Evidence |
|---|---|---|
| Numerical backups | VPM owns the accepted-step clock and paired VLM exports. Fresh complete velocity fields include bound surfaces and external providers; serialization cannot reuse stale CPU caches. | Moving two-surface physical-field regression; five native Metal HDF5/VTP pairs at steps 4, 8, 12, 16, 20. |
| Wake–body coupling | Every surface receives the complete free wake, and the global solve includes off-diagonal bound interaction. Particle groups cannot suppress induction. | Independent host Biot–Savart oracle and circulation response tests. |
| Native fields | Empty-wake samples retain freestream/bound fields. Streamwise CSV histories append across output calls/restarts. Four lines retain all signed velocity components from −1D to 3D. | Empty-cloud, output-schema and complete-window postprocessor tests. |
| Surface observation | Compiled f64 classification, batched geometry and distant-particle culling preserve the host oracle. Observation cadence is explicit; strict policy cannot skip intervals. Events publish only after acceptance. | Random/adversarial moving-surface tests. Warm initial replay: 15.26 s → 0.214 s (71×), excluding compilation. |
| Responsive row | Accepted and virtual rows share an affine source representation, including old circulation and body displacement. Virtual sources affect velocity/gradient/stretching; reaction history uses RK stage coefficients. | Seven independent closure/operator tests, including SSPRK3's non-monotonic stage times, plus existing exchange/restart tests. |
| Kernel precision | High-order and super-Gaussian functions derive from accurate ordinary-Gaussian identities. Removed the old erf approximation floor and corrected the super-Gaussian scaling. | Independent density quadrature and potential checks in f64; no relaxed high-order tolerance. |
| Diagnostic cost | Uniform finite-target probes can use the configured induction backend with the exact arithmetic-mean core transformation. Approximation is explicitly identified. | Direct/tree agreement with heterogeneous source cores, untouched source arrays, background/empty-cloud tests. |
| Documentation/configuration | All 337 VLM function definitions, including nested helpers, have docstrings. Invalid nonfinite physics and malformed solver/allocation settings fail early. | AST inventory and configuration regressions. |
| Legacy restart | An older lagged-field checkpoint may migrate only if its entire reconstructed legacy hash matches, including physics, geometry, motion and original group labels. | Real rotor checkpoint hash matched exactly; changed-model and incomplete-metadata tests reject. |
| Tree scheduling | Public target-order and thread-block controls preserve source arithmetic. Rotor uses spatial target order and block size 32. | Fixed 155,520-particle Metal replay: 4.42 s → 3.12 s per warm evaluation (29% less time); velocity, gradient and strength rate are bit-identical. |
| Tree capacity | Validate actual hierarchy depth against the traversal stack before field evaluation; use every allocated stack entry. Build multipoles through actual depth rather than a guessed loop bound. | A constrained-stack reproduction failed before the repair and is rejected afterward. The real mature rotor tree has depth 25, within capacity 48; its final GPU fields remain bit-identical to baseline. |

Particles are not deleted, reflected or clipped when they cross a VLM surface.
The repaired observer records crossings without changing physical state.

## What the rotor diagnosis establishes

A fresh native 20-step Metal startup completed at t=0.12 s with 2,700 particles.
Its five HDF5/VTP pairs pass the checkpoint publication audit. At matched startup
step 8, CPU and Metal circulation differ by relative L2 6.8e-7; maximum particle
velocity difference is 3.6e-5 m/s after matching particle positions within
provenance groups. Atomic insertion order is not a persistent particle identity.
These are short-run backend/output checks, not steady turbine validation.

A continuation from the retained t=6.912 s rotor checkpoint was deliberately
interrupted after profiling approximately 16 steps. Its native interrupted status
is preserved. About 363 of 427 profiled seconds were spent in VPM tree induction;
bound VLM stage contributions used approximately 17 s. This identifies the
dominant runtime cost without completing another long failing campaign.

The fixed-state diagnostic evaluates an independent f64 direct sum at the 32
largest free-wake strain locations and 32 seeded random particles. Relative L2
differences from the GPU tree are 6.93e-5 for velocity, 6.91e-5 for its gradient,
and 3.53e-5 for stretching rate. These errors are too small to explain the large
instantaneous growth at these sampled locations; this does not bound error at
every particle or during later evolution.

The largest sampled strain is in the old downstream wake near x=31 m, well away
from the blades. Its strain norm is about 9.02 s⁻¹ at t=6.912 s. Local relative
strength growth is about 6.00 s⁻¹ while relative core growth from the recorded
viscosity is about 0.0316 s⁻¹. Several strongest-strain particles have strength
nearly opposed to the curl of the filtered velocity. These are diagnostic signs
of a particle-representation/evolution problem to investigate, not a proven
causal explanation or permission to add arbitrary damping. The native late
failure remains at t=7.68 s in the original run.

The result points toward wake evolution/resolution rather than a missing
cross-body induction term or a large tree approximation error at the sampled
state. A smaller timestep alone has not been shown to cure it. Traditional VPM
stability under strong stretching is also a recognized formulation issue;
different element-evolution and subfilter models require their own validation.
See [Alvarez and Ning's formulation paper](https://scholarsarchive.byu.edu/facpub/7123/).
The literature is context, not evidence that a particular replacement would fix
this rotor.

## Remaining production gates

1. Resolve the demonstrated late wake growth using a controlled numerical study
   that separates timestep, particle/core resolution and any change in wake
   evolution. Preserve the baseline and native health limits. Greater survival
   time alone is not accuracy evidence.
2. Complete a healthy rotor horizon long enough to bracket five terminal
   revolutions after field establishment. Retain axial, radial/tangential and
   streamwise fields as well as CT/CP and sectional loading. The old run and
   the new short startup do not supply that evidence.
3. Pass the plan's unchanged load/field stationarity, reference and numerical
   refinement criteria. Quantify spatial, temporal and core sensitivity
   separately. The matched thin-plate BEM reference cannot validate stall,
   airfoil profile drag, tower or nacelle effects absent from this model.

At dt=0.001 s, the current 135-particle-per-step emission exceeds one million
particles over 7.5 s, beyond the authored 600,000 capacity. The measured mature
stage cost also makes that campaign expensive on this host. Capacity, wake
representation and runtime must therefore be resolved together; merely launching
the long run again would not close these gates.

The repaired default coupling has bounded two-wing temporal-refinement evidence;
it does not establish global third-order convergence from the SSPRK3 particle
integrator. The responsive option remains experimental outside the measured
checks until a broader coupled qualification supports it.

The moving two-wing study completes t=0.2 s at dt=0.02, 0.01, 0.005 and
0.0025 s for each policy, keeping all emitted cores at 1.25 m. Observed orders
are 0.989–0.997 for the lagged policy and 0.990–0.998 for responsive coupling
across circulation, integrated force and the full velocity vectors at 16 probes.
Between the two finest steps, integrated force changes by 0.0104% and 0.0105%,
respectively. This coarse, short two-wing case supports temporal consistency of
the implementation; it supplies no rotor CT/CP convergence or spatial accuracy
claim. The native rotor validator correctly rejects the completed Metal startup
for having fewer than five revolutions of final force samples.

## Reproduction and evidence

Run the scripts in `studies/vlm_production_audit/` from the repository root:

- `profile_observer_compiled.py`: warm observer replay against the retained baseline.
- `profile_mature_induction.py`: fixed-source Metal timing and saved fields;
  `--sort --block 32` selects the measured rotor schedule. Existing result tags
  cannot be overwritten.
- `diagnose_mature_operator.py`: bounded direct-sum check at selected mature-wake
  targets, including finite self gradients.
- `check_coupled_time_refinement.py`: actual moving two-wing evolution with four
  timestep levels for both coupling policies; fixed geometry and core rule.
- `run_rotor_qualification.py`: isolated native pilots with explicit timestep,
  backend, capacity, resource bounds and strict checkpoint identity.

Final test counts and source hashes are recorded alongside the results. Earlier
failed and interrupted logs remain discovery evidence and are not counted as
successful physical qualification.

**411 distinct tests pass** across the broad repair suite, restart/output
follow-ups and final kernel checks. There are no failed or skipped tests in
those final suites. This is a CPU regression result. The separate native Metal
checks described above do not imply coverage of every backend or physical case.
Lint and formatting pass for the 52 scoped Python files; a whole-project CI run
is not claimed. `repair_validation_summary.json`, `repair_source_sha256.json`
and the JUnit XML files provide the test counts, source identity and raw results.

The final observer replay measures 0.182 s after warming, about 84× faster than
the original 15.26 s replay. It still reports the same 16 synthetic events.

Timing varies with compilation and host load. The final guarded-tree replay
measures 3.44 s per warm evaluation, compared with 3.12 s in the scheduling
comparison and 4.42 s in the original replay. These are bounded measurements,
not a promised full-simulation speedup.
