# VLM production repair and qualification

This is the implementation follow-up to `vlm_production_audit.md`. Earlier
surface-crossing campaigns are superseded. Particles retain their circulation
and follow transport; an observed crossing does not authorize deletion.

## Ordered actionables

1. **Compile the moving finite-surface observer.** Preserve the host geometric
   oracle and event priority. Precompute geometry per panel/substep instead of
   per particle. Compare all event fields on adversarial and randomized moving
   panels, then replay the recorded rotor workload and measure warm throughput.
2. **Repair responsive near-wake closure.** Include old circulation and moving
   geometry consistently in the boundary solve and stage transport. Test the
   identical-row counterexample, moving surfaces, exchange conservation and
   stage purity. Keep the default policy explicit; do not infer coupled RK order
   from the particle integrator's name.
3. **Finish numerical contracts and documentation.** Fix demonstrated precision
   defects, document VLM functions and supported assumptions, retain independent
   field/load/backup tests, and measure expensive diagnostic work.
4. **Diagnose and qualify rotor flow.** Use isolated output directories and
   native histories. Investigate the late strain growth and timestep/core/wake
   resolution with short informative pilots before the production-length run.
   Record all parameter changes and resource limits. Run a healthy complete
   case with axial, radial and tangential induction, including streamwise lines.
5. **Run release checks and publish the supported envelope.** Execute relevant
   regression tests after the final edits, inspect native output consistency,
   record source hashes and numerical evidence, and update the production
   decision from measured results.

## Acceptance fixed before replacement runs

- Compiled observation agrees with the retained double-precision host oracle
  for event class, closest distance and interval/position, including moving,
  triangular, degenerate, grazing and reversed-winding cases. No physical state
  changes. Warm replay must improve the measured 15.26 s / 16-particle baseline
  by at least 20x on this host.
- Responsive accepted/stage row contributions agree for identical geometry,
  including the old-circulation term. Moving-row and transport checks must pass;
  any unsupported formulation remains explicitly outside the release envelope.
- Existing backup ownership, complete induction, group-independent cross-body
  coupling, load/power signs and restart/history contracts continue to pass.
- Rotor evidence must finish the declared interval without run-health failure,
  have complete terminal five-revolution native field/load histories and finite
  outputs, and pass the existing positive-load, <=2% load drift, <=15% matched
  BEM load comparison and <=1% field stationarity checks.
- Numerical refinement must demonstrate <=3% change in terminal mean CT/CP and
  <=0.02 U RMS change in each sampled velocity component (report near-filament
  extrema separately). Report spatial, temporal and core sensitivity separately;
  a BEM comparison alone is insufficient. Finite-distance vortex-cylinder
  comparisons remain idealized reference diagnostics, not experimental truth.
- CPU and GPU support are reported only for the backends actually tested.
  No thresholds are loosened to turn a failure into a pass.

## Status

Implementation repairs and bounded checks are recorded in
[the results report](vlm_production_repair_results.md). Items 1–3 are implemented.
Item 4 has narrowed the downstream-wake diagnosis but remains incomplete;
the full rotor release gate in item 5 remains open. The previous audit's
not-ready decision stays in effect until rotor qualification is complete.

- Compiled moving observation passes the scalar oracle, including event
  priority and positions. The initial warm replay improved from 15.26 s to
  0.214 s (71x). Mature wakes also use batched broad-phase culling. Observation
  cadence is now explicit, with strict policy requiring every interval.
- Responsive row assembly now shares affine source data with virtual-row
  transport, includes the old closing vector and body displacement, and uses
  the actual RK stage coefficients for intermediate reaction history. Seven
  independent closure/operator tests and 21 existing coupling/restart tests
  passed. Coupled temporal convergence remains a separate qualification.
- Corrected Gaussian kernels now derive from accurate ordinary-Gaussian
  identities. Independent quadrature passes at f64 precision. The VLM
  inventory has 337 functions including nested helpers, zero missing docstrings.
- The 20-step native Metal startup completed in 84 s. A matched CPU/Metal
  startup comparison gives circulation relative L2 difference 6.8e-7 and
  velocity maximum difference 3.6e-5 m/s, after accounting for atomic particle
  insertion order. This is backend agreement, not mature-flow accuracy.
- The retained version-6 rotor checkpoint's full original identity was
  reproduced exactly, including its original all-zero particle groups.
  Migration accepts only that exact legacy hash and unchanged lagged field
  model. Changed physics, motion, geometry or groups are rejected.
- A mature-wake diagnostic continuation was deliberately interrupted after
  profiling about 16 steps; its native interrupted status is preserved. It
  is not a completed stability result. Approximately 363 of 427 profiled
  seconds were in VPM stage tree induction, while bound VLM stage contributions
  used 17 s. GPU tree traversal, not the dense VLM solve, now dominates cost.
  A fixed 155,520-particle replay measures 4.42 s per induction evaluation.
- Spatial target scheduling reduces the measured warm induction time to
  3.12–3.44 s with bit-identical velocity, gradient and rate fields. The actual
  tree depth now also must fit its traversal stack before evaluation.
- Four timestep levels of actual moving two-wing evolution completed for each
  policy with fixed core radii. Both converge at approximately first order
  in circulation, force and sampled velocity; no global RK3 claim is made.
- At 64 selected mature-wake targets, including the largest strain locations,
  the GPU tree differs from the independent f64 direct gradient by relative
  L2 6.91e-5. Strong stretching is already present far downstream. The exact
  cause and validated remedy for late wake growth remain unresolved.
