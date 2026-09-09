# VLM tutorial completion checklist

User scope (2026-09-08): complete the VLM audit, then run actual `allrun.sh` workflows for flat plate, delta wing, rotor flow and quadcopter in isolated copies; monitor completion and steady/periodic/statistical convergence; plot meaningful velocity/loading/force comparisons, including rotor thrust and power against appropriate theory; use only native metadata and sampled data; keep setups and launchers simple; reduce checkpoint frequency relative to scientific sampling; commit all accumulated progress after completion.

## Current state — 9 September, installed v17; turbine and rotor refinement active

- **Active Metal:** actual full `workflows-v16/rotor_flow/allrun.sh`, native common overlap2.5, Gaussian, Pedrizzetti0.3, original dt.006/horizon14.4s, fields every.06s, forces.012s, backups one revolution. `/tmp/openonda-v16-rotor-full.log`; session40669. This is the candidate full turbine qualification, not an already validated setup.
- **Active final flat sweep:** actual20-case `workflows-v17-cpu/flat_plate/allrun.sh`, installedv17, no PYTHONPATH, two CPU threads. Only compute_device changes AUTO→CPU; time step, precision, physics, mesh and output schedules match the source. Session7845; `/tmp/openonda-v17-flat-full-cpu.log`. A read-only memory check showed4.27GB free before launch, permitting overlap with the turbine and two small rotor studies while keeping onlyone Metal simulation.
- **Delta failure:** v13 actual full run stopped at step995/t2.4875s with strain CFL1.06, before10cycles. All outputs and last valid backup800 retained. Session91393 exited1. The proposed native runtime compatibility wrapper found the process already gone and exited before any pause/signal; it did not stop this run. The delta case needs a demonstrated resolution/stabilization repair before another full attempt.
- **Completed relaxed refinement:** `single-rotor-native-overlap-fine` finished all576 steps/three revolutions, native runtime-v14, Gaussian/FMM/two threads,1.875 degrees, overlap2.5, Pedrizzetti.3. Session44823 exited0; final checkpoint576 retained. Third-cycle means T=1.389860N/P=5.621868W, compared with coarse relaxed1.364319N/5.562355W (1.87%/1.07% difference). Both still lose load; this does not validate relaxation as the default.
- **Active unrelaxed refinement:** actual `single-rotor-native-overlap-unrelaxed-fine/allrun.sh` from a fresh copied case, installed `installer-v17/bin/python`, PYTHONPATH unset, Gaussian/FMM/two CPU threads,1.875 degrees,3 revolutions/576 steps,40kcapacity,overlap2.5,no relaxation. Session33830; `/tmp/openonda-single-rotor-native-overlap-unrelaxed-fine.log`. The solver update in v17 only adds native progress metadata; numerical evolution matchesv16.
- **Completed coarse Gaussian:** six revolutions/576steps, exit0, all data/checkpoint576 preserved in `single-rotor-core-overlap-gaussian-fmm`. Revolution6 means T=1.225002N, P=4.793300W; last3revolution means T=1.282398N, P=5.100444W. Loads are still falling and do not qualify as steady or within the20% isolated-BEM thrust target. Do not equate small CFL with an accurate solution.
- **Completed unrelaxed coarse / active continuation:** native v16 common-overlap Gaussian single rotor without Pedrizzetti relaxation finished six revolutions/576steps, exit0; session61017 closed. Sixth-cycle CT=.0044278241 / CP=.0002752256 versus BEM .0047432617 / .0002871502 (−6.65% / −4.15%). Recent three-cycle thrust drift remains >3%. Actual `single-rotor-native-overlap-unrelaxed-extension/allrun.sh` now continues read-only checkpoint576 for two additional revolutions with installedv17, same numerical settings, separate outputs, 40kcapacity. Its launch used `TI_NUM_THREADS=2`, which Taichi does not honor, so this continuation uses the default CPU pool; subsequent launches must use `TI_CPU_MAX_NUM_THREADS=2`. BLAS remains limited to two threads. Session30584; `/tmp/openonda-single-rotor-native-overlap-unrelaxed-extension.log`. Never modify or remove the original completed run.
- **Short budget diagnostics complete:** external-direct bound stretching gives .0397% residual at t=.25s; adding total trailing-edge convection .1232%; overlap-only .1161%, versus native medium .1580%. These hypotheses do not establish a conservation fix and are not adopted.
- **Verified mixed-core tree acceleration:** accept distant nodes only outside every source regularization tail, retaining near-core descent. Direct-reference/origin tests39passed. Fixed delta checkpoint400: warm fused evaluation ~2.66s → .675s; velocity and gradient relative L2 differences both3.97e-5 (`tree-tail-comparison.json`). Target Morton ordering alone gave no speedup and stays disabled. The original test failure was the host reference zeroing a finite source-centre Jacobian; corrected with independent finite differences. Particle-only reference tests now explicitly exclude self-pairs, as the particle solvers do. Broader FMM follow-up:27passed (`/tmp/openonda-kernel-origin-fmm-followup.log`). Fixed-state CPU FMM acceleration is separately verified as described above.
- **Complete flat sweep:** all20 actual v7 runs and native contracts pass. PNG/PDF all5 exported and visually inspected at `workflows-v7-full/flat_plate/figures`. Final metrics `full-v7-qualification-metrics.json`; CL5=.428478 (−2.71% versus LL), CD5=.0060275 (−10.10%), sectional loading L2=3.97%, inboard downwash L2=2.12%, full-span25.37%. Max lift tail range.152%. **Strict vector-strength validator still fails2.089% at8deg**; do not claim allphysics passes.
- **Earlier failed candidate:** standard native Pedrizzetti.3 extends single-rotor run tostep420/576 (4.375rev), then strainCFL1.01. Thrust/power drift substantially; not an accepted final setup. Preserved `single-rotor-v8-relaxation-standard`, log `/tmp/openonda-single-rotor-standard-relax.log`. A prepared four-rev wind version remains unstarted at `rotor-v8-relaxation-standard`.
- **Tests:** fullv13 regression423passed,0failed,0skipped in682.909s (`/tmp/openonda-vlm-v13-regression.xml`); fullv7 regression394passed; post-v7 relaxation24passed; bound-vector/restart19passed; installed/style22passed. Batches overlap; do not sum them. Numerical solver frozen in runtime-v16; installedv17 adds metadata progress (restart version4 unchanged). Do not mutate any frozen runtime or active installed environment. Current source-only tutorial cleanup afterv9 is safe alongside these runs.
- **Current source cleanup:** wind single input deck with exact native-config/sampler equivalence. Flat now one physical moving/static branch; removed unused translation mode/builders and unplotted final-plane samplers. All20 physics/run/backup/plotted-sampler configurations match (`flat-input-equivalence.json`). Removed one-line cadence wrappers from delta/quad. Delta now explicitly uses DNS (noSGS), replacing LES withzero coefficient; correcteddelta fullrun uses this setting. Flat spanwise plot shows native cell-centred stations only, and budget annotation is explicit.
- **Budget experiments complete:** common-time refinement residual .115%,.158%,.206%; unfiltered-bound diagnostic worsens to.278%; adding boundTE convection gives.177% (all at t=.25s). No diagnostic monkeypatch adopted. These older diagnostics used restart version3; current native version4 is required after the shared-root fix. Full three-leg bound accounting leaves rectangular flat-plate values unchanged.
- Original5,552 result/geometry hashes match (`preservation-check-v7.json`). Finalrecheck, installer/outside-checkoutverification, fullwind/quadqualification, finalaudit andgitcommit still outstanding. No commit has been made.

### Latest verified diagnostic and logging changes

- Fourier integrals now pad each compact component inside the FFT and accumulate quadratic integrals component by component. On delta checkpoint400, runtime-v14 versus v15 gives 4.432s versus 3.588s and peak RSS3.564GB versus2.544GB: 19% faster and29% less peak memory, with unchanged grid and integrals (relative differences around machine precision). Evidence: `fourier-memory-benchmark-v14.json` and `fourier-memory-benchmark-v15.json`; 17 Fourier/free-space tests pass. Delta was briefly paused for this owned-process benchmark and resumed in `finally`; it later failed at step995 as recorded above.
- Native optional `wake_core_overlap=2.5` makes trailing and transverse core overlap explicit in the VLM configuration. No option retains exactly the previous geometry, emissions and restart identity (`core-default-compat-v13.json` / `core-default-compat-v14.json`). Native overlap/restart qualification:32passed. Fine single-rotor refinement is using this API; the tutorial rotor decks await the comparison.
- Elapsed run logging now includes initial conditions, accepted diagnostics, sampling and backups. The evolution profiler retains its separate timing. A simulated-clock regression verifies28s total including output versus4s evolution and frozen terminal elapsed time. Lifecycle/logging/output suite:58passed (`/tmp/openonda-elapsed-time-tests.log`). Initial invocation named a nonexistent test file; subsequent run exposed only a formatting mismatch in the new test, which was corrected before the passing rerun. This source change is newer than runtime-v15.

## Protection and evidence

- Existing repository contains substantial authorized work from earlier tasks. Preserve it.
- Audit workspace: `/var/folders/kw/njsv3xwj69qf8p4bp4jw035h0000gn/T/openonda-vlm-audit-0fyxi5zd`.
- `original-data-hashes.json` protects 4,093 original flat-plate result and geometry files. Other-tutorial inventory is `other-original-file-hashes.json`. Latest 5,552 original result/geometry hashes all match (`preservation-check-v7.json`).
- Source findings, theory references and completed evidence: `2026-09-vlm-audit.md`.
- Do not edit Taichi source while a running process may compile it lazily.

## Completed

- [x] Start with existing plots and raw samples; document audit before fixes.
- [x] Fix rank-deficient lifting-line reference, hidden moment curve, inconsistent bound induction, finite-difference gradients, frame-dependent shedding, force sampling, motion clock, duplicate diagnostics, restart omission and linear-solver issues.
- [x] Route VLM VTP/PVD through `VLMSampler` under `samples/`; include VLM in velocity probes.
- [x] Add native sampled velocity plot; improve lift, moment, loading and budget plots.
- [x] Verify analytic gradient and LU component speedups (2.63x and 16.0x).
- [x] Pass focused numerical/restart tests and isolated two-step runs of rotor, delta and quadcopter.

## Required before completion

- [x] Finish current 20-case full flat-plate diagnostic sweep; retain results and quantify all errors.
- [ ] Investigate and fix coupled bound/wake vector-strength drift (2.1% in current static 8-degree run). Do not relax existing conservation checks to hide it.
- [ ] Complete final regression suite and validate output/restart contracts.
- [x] Audit rotor/delta/quad physics, sample cadence, checkpoint cadence and native force/power data contracts.
- [x] Add native solver diagnostics for rotor power/thrust; no tutorial metadata writers or duplicate extraction methods.
- [x] Select and cite appropriate rotor theory, stating inviscid/viscous, Reynolds-number and transient limitations.
- [ ] Run actual `allrun.sh` for all four tutorials using final installed code in fresh copies.
- [ ] Monitor full runs; extend duration if steady/periodic/statistical convergence is not reached.
- [ ] Run each `allplot.sh`, inspect all figures and numerical values; ensure plots use native solver samples/metadata only.
- [ ] Verify original datasets unchanged and keep generated validation data out of git.
- [ ] Reinstall final code through `python install.py`; verify direct commands outside checkout.
- [ ] Update audit with final results, remaining physical limits, commands, artifact locations and exact test counts.
- [ ] Inspect staged content and git commit all accumulated authorized progress; explain commit purpose.

## Full-workflow follow-through

### Historical v2 execution (superseded by current state above)

- Frozen pip-installed package: audit `runtime-v2/`; never mutate it while simulations use it. Checkout edits are safe.
- Actual `allrun.sh` running under audit `workflows-v2/flat_plate` (session 45880, `/tmp/openonda-v2-flat.log`) and `workflows-v2/rotor_flow` (session 70037, `/tmp/openonda-v2-rotor.log`). Earlier rotor session 34975 was deliberately interrupted and preserved after the coupling defect was identified.
- Near-wake rewrite completed: transport old wake, move/solve body, emit completed row once; local TE velocities and cumulative strip circulation; stage-predicted source geometry. Accepted-state Galilean test passes after eight coupled steps. Focused coupling tests: 16 passed.
- Three complete corrected runs passed: static5 CL=.42699423, moving5 CL=.42697995, static8 CL=.68182079. Final-five-chord CL range is ~0.14%; moving/static5 mean difference ~0.0033%. Vector-strength residual persists: .788% at5°, 2.030% at8°. Keep conservation investigation open.
- Native rotating motion now supports an exactly integrated one-revolution sin² spin-up; force/power uses actual angular speed. Latest power/restart/BEM checks: 7 passed.
- Quadcopter candidate uses4000rpm,24revolutions,3.75° per step, LES.17, forceevery2steps, fieldevery12steps, checkpointevery2revolutions,500kcapacity. Audit blade orientation before launching: old geometry comments contradict Ω×r.
- Delta placement now uses the local initial pivot (previously translated twice);10heavecycles, fields.04s, forces.01s, checkpoints1s. Not yet run in full.
- Wind BEM reference has two independent annular-momentum/scaling tests passing. Finish native-data plotters/validators for delta and quad and inspect figures.

- Initial 20 full flat-plate cases all completed. Static 5° tail CL=0.430802 vs lifting-line 0.440416; downwash inboard relative L2=2.79%, full-span=25.0%, outer station error=-33.3%. Native output/time/row contracts passed for the first 12 cases; recompute for all 20.
- Full regression suite: 371 passed; two old metadata fixtures failed and both pass in the corrected 14-test follow-up. New power/restart tests pass after using the native geometry serializer.
- Added native `vlm_surface_forces.csv`, per-surface moments about actual pivots, sampled motion/centroids, rotational/translational/total fluid-on-body power; summed shaft power remains meaningful when counter-rotating torques cancel.
- Fixed portable VLM restart identity (geometry content and references, not path).
- Full actual rotor `allrun.sh` is running from regular installed code in `full-workflows/rotor_flow` (session 34975, log `/tmp/openonda-full-rotor.log`). Do not reinstall while its Taichi process is active.
- Full actual quadcopter `allrun.sh` failed its strain/CFL check at step 72 (1.07 > 1); preserved under `full-workflows/quadcopter`.
- Revised sampling/checkpoints: flat checkpoints every 0.5 s; delta fields every .04 s, forces every .01 s, checkpoints every 1 s; rotor forces every .012 s, fields .12 s, checkpoints one revolution, capacity400k. Quadcopter test operating point4000 rpm (original200 rpm has incidence below pitch for the stated climb),24revolutions, forceevery step, fields6steps, checkpoints2revs,capacity300k. Reassess dt/LES before final runs.
- Major coupling finding: the AIC extends only TE-panel Γ (not cumulative chordwise Γ) using a global mean-flow offset, while actual rotating-blade wake travel is local relative velocity. Starting-vortex augmentation sits at TE rather than the downstream row edge. Step order advances body before transporting old wake and transports newborn particles for a full step. Correct these together with actual TE-point velocities and consistent row placement; validate static/moving equivalence and conservation before keeping the change.
- One-step budget trace at static8° step192: external transposed stretch net_y=.50986 m³/s², direct=.06448; shedding+bound-change net_y=-.00373385 m³/s; total step budget change+.00434494. Both shedding geometry and external stretching contribute. Do not hide this via an arbitrary global strength correction.
- `other-original-file-hashes.json` contains1,591 files including source files. Preservation comparison should apply to original scientific data/geometry, excluding intentional source edits.
- Rotor BEM reference rewritten to bracket the inflow-angle residual with Prandtl hub/tip losses and Buhl high induction. Needs independent tests and plotting validation. Rotor `_common.py` and performance plot now read native metadata/geometry/motion/power; remaining loading/wake/validator scripts need updating.
- Snapshot before the near-wake coupling change: audit `before-near-wake-fix/` (VPM source only).

### Later findings and plot changes

- Corrected quadcopter blade geometry: original CCW/CW leading edges were reversed relative to Ω×r. Quarter chords now lie on the radial axis. Both directional geometry tests and both BEM tests pass (4 total).
- Corrected geometry with one-revolution spin-up fails at step5 (CFL3.94) because almost stationary pitched blades see full climb inflow. Full-speed start passed step60 before diagnostic interruption; complete stability/refinement qualification remains open.
- Stopped CPU FMM rotor session70037 and quad full-speed diagnostic58126 after observed18GB swap / severe compression. All output preserved. FMM fixed-capacity workspace and concurrent runs are too expensive for this16GB machine. Flat sweep45880 continues.
- Metal Treecode(theta=.3, order3) benchmark uses a separate copy of rotor checkpoint128. Diagnostic copy explicitly changes numerical backend identity; never treat this as a production restart. Session37564, `/tmp/openonda-rotor-tree-benchmark.log`; CPU comparison still needed.
- Shared attached-flow reference moved to `openonda/rotor_theory.py` for both wind rotor and propeller. No duplicate tutorial algorithm.
- Delta plots now read native per-surface force/motion/power and native PVD velocity, with last-three-cycle overlays; fixed vector-strength units to m³/s.
- Quad plots now show individual rotor CT/CP, summed input shaft power, BEM/ideal-momentum references and two velocity planes. Wind wake plot uses actual PVD times and expanded ideal streamtube.
- Rewrote all three validators to use native configured horizons and real sampled coefficients, not old hardcoded steps/rpm. Thresholds are explicit; no claim of pass yet.
- `VPMSolver.load_backup(Path)` throws AttributeError because it assumes `.endswith`; fix PathLike support and test after current source-based Taichi benchmark finishes.

### Active v3 qualification

- Backend benchmark: identical rotor checkpoint128, four steps; warmed FMM CPU5.4596s versus Metal Treecode(theta.3,order3)1.6445s. Final force_x relative difference1.76e-7, moment_x8.43e-8, Γ L2difference3.56e-7, particle strength1.85e-6, maxpositiondifference6.36e-6m. Evidence: `rotor-backend-comparison.json`.
- Frozen pip-installed `runtime-v3/` includes shared BEM reference, corrected quad geometry, full-speed quad startup, Path backup support and Treecode/AUTO for rotor/quad/delta. Delta capacity250k. Do not edit this runtime.
- Actual quad `allrun.sh` now running at `workflows-v3/quadcopter`, session2483, log `/tmp/openonda-v3-quad.log`; first revolution passes health checks. Fullwind/delta-v3 not started yet.
- Full regression (VPM + related tutorial/installation tests) running session48011, `/tmp/openonda-vlm-v3-regression.log`, JUnit `/tmp/openonda-vlm-v3-regression.xml`. Do not edit solver Python while it can compile Taichi kernels.
- All13 rotor/delta/quad plotter/validator `--help` invocations pass without saved results. Preliminary quad PNGs render and were visually inspected. Source plot adjustments after runtime-v3: shorter coefficient labels, distinct power style, honest averaging-window title, fewer colorbar digits. Sync source assets into run copies before finalplotting.
- Added READMEs for all three with coefficient conventions, native-output contract, cadence, reference scope and inviscid limitations.
- Remaining code cleanup after regression: stale near-wake comments and unused private wake arguments; consider reducing per-step particle-add log spam (scientific sampler records already provide population), correct logging energy units (native integrals are per density). Do not disrupt ongoing frozen runs.

### v4 runtime / last cleanup

- Full regression382passed (449s). Kernel-reuse/output/restart follow-up45passed; native area-weighted centroid test3passed.
- Moved grouped wake-copy Taichi kernel out of the per-step Python function. The old path compiled20extra kernels across20appends; fixed path compiled0. CPU32-particle transfer median5.88ms ->.0439ms (component benchmark only). Four-step rotor checkpoint comparison gives bit-identical positions, strengths and circulation. Whole-step timings differ with concurrent workload and are not claimed as an additional speedup.
- Removed unused near-wake arguments and dead previous-velocity download, refreshed coupling comments, made buffer/particle-capacity overflow fail explicitly instead of dropping wake strength, reduced particle-add log spam while retaining progress/capacity warnings, fixed per-density energy logging units.
- Native per-surface centroid now uses actual panel centers weighted by area. Delta wake-plane z bounds expanded to±.9m to include the prescribed heave.
- Frozen installed `runtime-v4/`; fresh `workflows-v4/rotor_flow` and `workflows-v4/delta_wing` copies ready. Start wind when flat finishes, then delta when a heavy-run slot becomes available. Quad session2483 continues on v3; physical solution unchanged by v4 fixes.

### Full sweep and rotor stability follow-up (9 September)

- All20 corrected flat-plate runs completed; native output contracts passed. PNG/PDF plots rendered. At5deg CL=.426994 (3.05% below lifting-line), inboard downwash L2error2.15%; full-span28.76%. Tail CLrange <.157% for all20. Strict vector-strength budget remains unresolved: .788% at5deg,2.030% at8deg,7.332% at15deg. Do not waive the existing threshold. Metrics: audit `full-v2-qualification-metrics.json`.
- Quadcopter corrected/full-speed v3 actual allrun failed atstep227/2304, strain CFL1.01. Before failure, native misalignment reached49deg and divergence .30; no steady-state qualification.
- Wind rotor v4 actual allrun failed atstep188/2400, strain CFL1.24. Preserve `workflows-v4/rotor_flow`, log `/tmp/openonda-v4-rotor.log`.
- Delta v4 actual allrun continues, session35098, log `/tmp/openonda-v4-delta.log`.
- Quadcopter3x finer time diagnostic (1.25deg,3revolutions) also scales sigma_factor2.5->7.5 to hold transverse initial core width. CPU session61688 deliberately stopped due to poor throughput; restart same diagnostic in separate `quad-time-refinement-metal`, preserving CPU outputs. This isolates time refinement before changing stabilization. Source tutorial settings are not yet qualified.

- Additional code audit: VLM stage stretching always used J.T@Gamma even when users selected direct/mixed, and downloaded/uploaded every particle gradient and rate at every RK stage. Make the VLM contribution honor the chosen formulation and accumulate on device; retain transposed default and verify all three contractions independently. Correct Pedrizzetti option docs: per-particle magnitude preservation is not net-vector preservation.

- VLM device contraction and selected-formulation checks26passed, including coupled restart and Galilean invariance. New frozen runtime-v5. Four-rotor refined diagnostics stopped/preserved before completion because their population made the stability isolation too expensive. Three single-rotor copies compare3.75deg/1.25deg (matched transverse core width) and native Pedrizzetti.3 with moment preservation at3.75deg; these are diagnostics, not final quad qualification. Delta fullrun still active.

- Single-rotor coarse baseline reproduces instability atstep252/288 (2.625rev), CFL1.07. Fine1.25deg comparison is running session5844. Native Pedrizzetti+moment-preservation failed before first shedding because it tried setting properties on an empty cloud. Fixed the solver worker to skip an empty cloud and added regression; do not make tutorials special-case startup.
- Incomplete-run validators now stop with an explicit incomplete-horizon result before computing empty final windows. Allthree reject incomplete native metadata cleanly. Removed unreferenced rotor shedding helper (which checked a separate design schedule) and unused quad particle-count plot; native performance/wake/enstrophy figures remain.

- Native Pedrizzetti.3 with per-particle magnitude+net/impulse preservation was correctly rejected atstep2: moment correction increased total strength by1.139%, exceeding unchanged.1% worker gate. Next controlled candidate uses native.01blend without individual-magnitude renormalization, retaining net/impulse preservation. Do not call rejected runs qualified. Empty-wake/shared-contraction tests28passed.
- Flat8 one-step source trace from fullstep192: self-stretching net_y~-1.6e-4m3/s2; bound externalstage rates~.4-.6m3/s2. Shedding closure=-.002252m3/s; totalstepbudgetchange+.003961. Direct-bound-only diagnostic flips totalchange to-.001381, so changing contraction alone does not close the budget. Evidence `budget-v6-{transposed,external-direct}.json`; these are explicitly diagnostic transfers, not production continuations.

### Additional near-wake audit before correction

- Accepted-state checkpoint velocity exposes a no-penetration defect after insertion: static8deg step192 normal-velocity RMS.02974m/s, max.30183m/s; single rotor step192 RMS.20062m/s, max1.10616m/s. The circulation solve uses finite straight near-wake filaments, but inserts regularized midpoint particles afterward. The implicit starting-vortex NumPy formula also adds a length-squared core to an area-squared denominator. Thus the solve and accepted flow use different wake fields.
- Correct the coupled matrix to use the actual newborn particle positions, strength coefficients, radii and the configured native radial VPM kernel. Remove the obsolete separate finite-line starting-vortex formula. Verify accepted no-penetration for multiple kernels, scales and rotating motion, plus continuation/Galilean tests, before running new production cases. Frozen v4/v5/v6 diagnostic runs are unaffected while this is implemented.

- Stopped/preserved v4 delta and v5 fine-rotor diagnostic after proving their boundary-condition defect, rather than spending full horizons on that discretization. Native.01moment-preserving relaxation also fails atstep252CFL1.25: it does not repair the coupling problem.
- New native-particle matrix implementation removes finite-line near-wake and dimensional starting-vortex bug. Bound finite-line evaluator now uses the same Rosenhead formula as stage/probe fields. Restart subgroup version3 rejects incompatible older coupled history. First boundary tests: Winckelmans passes; Gaussian residual~2.8e-7m/s reflects device A&S erf versus exact host erf and is tested against the known approximation scale. Rechecking after consistent bound regularization. Fresh v7 workflow copies prepared; launch only after checks.

### v7 validated coupling and execution

- Focused native-wake/stage/output/restart suite27passed. Configured Gaussian/Winckelmans source kernels, rotating motion and geometry scale1/.02 all satisfy accepted no-penetration within native-kernel numerical accuracy.
- Full static8deg v7 case completed192steps in46s: CL=.684268,CD=.015358,CM=.002933; tail CLrange.113%. Normal-velocity RMS1.01e-6m/s,max3.10e-6m/s (was RMS.02974/max.30183). Vector-strength budget still2.089%; keep investigation open. `v7-flat8-metrics.json`.
- v7 single rotor completed3rev/288steps (previouscoarsefailed252). FinalCFL.240,divergence.292,misalignment29.6deg; thrust/power still oscillate, so this is a stability probe, not steady-state validation. `single-rotor-native-wake`.
- Only long Metal workflow now: actual fullwind `workflows-v7/rotor_flow`, log `/tmp/openonda-v7-rotor-full.log`. Broad CPU regression underway. Delta and fullquad still need corrected runs; flat20sweep must be rerun once solver changes settle. Allrun/allplot fourcases now have one `cd` line and direct Python calls so invocation from another directory works.
- Preservation recheck:5,552 original result+geometry files matched SHA256, no changes (`preservation-check-v7.json`).
- Another task asked about its PID60073 SIGKILL. No known SIGKILL from this audit; only identified-own SIGINT stops. Relaying a status via send_message_to_thread was rejected by automatic approval review as sharing private machine/project details without explicit authorization. Do not retry or work around it. Answer/status kept in this task; keep long Metal jobs sequential and check active processes.

- Additional native-relaxation audit: the worker changes strengths before the manager can reject its growth, and accept() records an event before testing its limits. Rejected relaxation must restore original strengths and must not count as an accepted event. Fix and test that transaction without changing any admissibility thresholds.

### Continued audit and controlled stabilization studies

- Frozen runtime-v8 includes transactional Pedrizzetti rejection rollback. Its 24 focused tests pass. Source changes after v8 are diagnostic all-three-bound-leg integration and a clearer incompatible-checkpoint-version error; qualification/restart tests session16682 (`/tmp/openonda-vlm-bound-budget-tests.log`).
- Wind setup reduced from two configurable builders to one input deck. Native case configuration and every sampler identity match exactly (`rotor-setup-cleanup-equivalence.json`). Installed-construction and tutorial-style follow-up: 22 passed (`/tmp/openonda-tutorial-cleanup-followup.log`). The single case-local `cd` is necessary to launch direct Python commands from any caller directory.
- New single-rotor six-revolution diagnostics prepared: `single-rotor-v8-relaxation-standard` (native Pedrizzetti .3, default per-particle magnitude preservation, native transfer reporting; no net-moment constraint) and `single-rotor-v8-direct-stretching`. Neither is final quad qualification. Wind four-revolution diagnostic `rotor-v8-relaxation-standard` prepared. Run sequentially after full flat sweep frees Metal.
- CPU flat diagnostic session80774 (`/tmp/openonda-flat-bound-filter-diagnostic.log`) removes only target-volume filtering from the external bound-field gradient in an isolated monkeypatched harness. This is a diagnostic, not an adopted solver change. Native baseline medium grid is retained for comparison.
- Another task reported its unrelated VPM battery PID60073 had been SIGKILLed. No such signal was issued by this audit; prior severe memory pressure was observed. Automatic approval review rejected a cross-task status relay for lack of explicit destination/payload authorization. Do not retry that relay or interrupt unrelated processes. Latest process inventory showed only this flat sweep and short qualification commands.

### Native geometry records and additional cleanup

- Frozen runtime-v10 built after new native geometry metadata, shared plotting reader, flat input cleanup, and dead code removal. Do not modify it. Current delta continues immutablev9; overlap experiment is separate and has NOT been adopted into source.
- VPM metadata now embeds each loaded VLM surface geometry using the existing solver surface serializer, so input files can be removed/edited without changing plot references. `openonda.plotting.read_vlm_surface` reads this record, with file-based compatibility for historical cases. Flat, wind and quad readers use it. Regression removes the source geometry after solver construction and checks exactnativegeometry/readback. Initialbatch39passed withonebadfixture (missingν); correctedfixture passes separately.
- Removed two uncalled scalar mesh generators, their exclusive helpers, an unused preconditioned matvec pair, and the unused quad particle-count helper. Fixed ignored public initial trailing-leg length and added a direct geometry regression. Combined VLM/restart/setup/style batch42passed (`/tmp/openonda-vlm-cleanup-regression.log`).
- Fullflat all5PNG/PDF reviewed. All20 strict native clocks/tables/PVD/checkpoints pass. Kelvinvalidator alonefails2.089%. Updated spanwiseplot usesonlynativecell-centredstations; no synthetic tip markers. Nativegeometryreader compatibility tests pass for oldrecords.
- AnotherVPM battery PID68992 (notownedbythisaudit) is active; narrow earlierprocessfilter missedstudy.py. Memory free68%,swapused7.8GB at latestcheck; deltaphysicalfootprint827MB,peak8.9GB. Neverinterruptthatbattery. Explicitpermission for a minimal cross-task schedulingnote was requested asynchronously; NOanswer yet, NOmessage sent. Earlierrelay wasauto-reviewrejected. Rootcancontinueread-only/ownisolatedwork withoutthatpermission.
- OS three-second profiles: `/tmp/openonda-delta-stack.txt` shows mainthread~95%waitingforMetal; `/tmp/openonda-core-cpu-stack.txt` shows CPUwaitingforTaichiworker. Nativephaseprofiling nowrunning oncopiedoverlapcheckpoint192. Originalbenchmark/direct/failedoutputsallpreserved.

### Diagnostic cost correction and resumed runs

- Second OS sample of delta at the prolonged step352 found large CPU Fourier-array operations, following the earlier GPU waits. Persistent diagnostic grid growth expanded all axes by25% whenever one axis needed more support; this compounds transverse memory unnecessarily for a long thin wake. Source now grows only axes lacking slack. Independent elongated-support sizing test plus Fourier/free-space regressions:13passed (`/tmp/openonda-fourier-grid-growth-tests.log`). Numerical evolution and output cadence are unchanged. Frozen runtimev11 includes this fix.
- Own delta PID71337 and slow CPU overlap PID71587 were SIGINT-interrupted, with all prior samples retained. No unrelated process was signaled. Cross-task scheduling question is still pending; no unauthorized message has been sent.
- Larger-core rotor had reached roughly2.5revolutions with misalignment~1.5deg before the backend switch; this remains an experiment, not a full qualification. Standard-core Pedrizzetti and full-direct-stretching failures remain preserved. Fresh FMM experiment starts from zero; the separately adapted two-step checkpoint is only an explicit backend comparison, not a production restart.

### Native diagnostic follow-through

- Bound-strength CSV/history still bypassed the corrected full-bound integration and reconstructed only quarter-chord TE legs. It now calls the solver's canonical full-bound diagnostic, and wake sums accumulate in float64. Tapered/contracted-span history regression plus VLM/restart/output suite:52passed (`/tmp/openonda-native-budget-recording-tests.log`). Fixed helicity log units to m4/s2 and Fourier viscous-rate doc units to m5/s3.
- Accepted-step health already evaluates current velocity/gradient/LES fields; the immediately following integral-output path unnecessarily evaluated them again. Reuse those known-fresh fields in advance() and coupled sampler dispatch without adding a public cache switch. Lifecycle/health suite:31passed (`/tmp/openonda-output-refresh-tests.log`). These source changes are after frozenv11; active long runs are unaffected.
- External-bound stretching diagnostics at t=.25: direct-only residual.0397% versus native.1580%; direct plus total TE convection.1232%; uniform-overlap-only.1161%. None closes the budget, so no stage/convection monkeypatch was adopted. Detailed comparison `budget-external-field-comparison.json`; overlap log `/tmp/openonda-budget-v11-overlap.log`.
- Additional topology audit: mirrored root edges are always skipped by emitter and newborn AIC. They cancel only for antisymmetric signed circulation. Unequal loads require one root filament proportional to the sum of the two signed root circulations. Reproduce closed-row budget with asymmetric halves before fixing both paths; preserve full boundary/continuation checks.

### Asymmetric shared-root closure

- The independently prescribed unequal-half-wing startup test failed before correction with missing vector[.12,.03,.018]m3/s. Both emitter and newborn AIC now represent one shared-root filament with the sum of the two signed root circulations (mirrored halves both point root-to-tip). The first corrected suite had27passes and one Galilean failure caused by roundoff-sized root particles being emitted in different frames, not by a resolved flow discrepancy. A32-ulp relative root-cancellation floor now suppresses only that numerical noise; the asymmetric closure test retains the physical root filament. Follow-up currently `/tmp/openonda-asymmetric-root-followup.log`, session27818.
- Native VLM restart version is now4 in current source. Frozenv11 and the core-overlap experiment remainversion3 and must never be silently mixed withversion4 continuation.
- Source afterv11 also includes native full-bound CSV/history, unit corrections, and removal of redundant immediate post-health gradient refresh. No core-radius or external-stretching experiment has been adopted.
- Latest single-rotor FMM experiment passed3revolutions; cyclemeans T/P are2.4067N/5.7972W,1.5762N/5.9318W,1.4076N/5.5898W. Startup is still relaxing; no steady-state claim. Audit output total2.1GB at this check. Full wind/quad and final commit remain outstanding.

### Native history, restart and output refresh follow-up

- Runtime-v12 contains canonical all-bound-leg vector history, correct helicity/viscous-rate units, and removal of immediately repeated derived-field refreshes after accepted-state health evaluation. Focused native-history/qualification checks52passed and output/health checks31passed.
- Asymmetrically loaded mirrored wing halves now emit their shared root filament once, with the sum of their signed circulations; AIC and emission agree. Symmetric cancellation uses32machine-epsilon relative tolerance to suppress frame-dependent roundoff blobs. Coupled restart version4 rejects previous history explicitly. Asymmetric/Galilean/restart/stage follow-up14passed.
- Delta v11 has passed step400 and written a native checkpoint. Benchmark at57,600particles found sorted and unsorted fields bit-identical but no throughput benefit. The source tree tail-admissibility candidate needs independent verification before use.

### Native geometry I/O and regularization comparison

- Removed duplicate tutorial geometry readers/writers in flat, delta, quad and wind assets in favor of the existing native `surface_io` implementation. Wind reference origin remains the shaft axis. All four loaded geometries and reference values match exactly (`geometry-io-cleanup-equivalence.json`); geometry/style/plotting/theory follow-up7passed.
- Single-rotor startup comparison figure: audit `rotor-core-kernel-comparison-progress.png`. Both kernels remain transient; no steady-state claim. Winckelmans full-revolution mean thrust2.407,1.576,1.408,1.328N; power5.797,5.932,5.590,5.341W for rev1–4. Gaussian rev1–3 approximately follows the same trend.
- Additional isolated flat-budget diagnostic integrates the native Gaussian kernel along bound filaments using16-point quadrature, instead of Rosenhead volume filtering. Transposed result at t=.25s=.1801% (native baseline .1580%); thus kernel mismatch alone does not repair the budget. Direct version completed32steps: .0415% at t=.25s, versus Rosenhead direct .0397%; no quadrature/stage change adopted.
- An extra installed-environment Taylor–Green run/plot/clean check completed with the full CSV identical to the saved pre-cleanup result. Its command was accidentally selected when intending a memory probe; it touched only existing isolated `/tmp` qualification workspaces, never tutorial originals. Actual subsequent memory probe found31% free, with substantial pre-existing swap use, so no third long rotor study was launched.

### Next implementation decision (before final rotor input decks)

- The existing `sigma_factor` scales only transverse newborn elements; it cannot specify common core overlap for trailing and transverse elements. Implemented a native optional `wake_core_overlap` setting: positive core radius divided by max(local span spacing, local row length), applied consistently in both AIC and emitter. DefaultNone retains the existing discretization, so the accepted flat and running delta physics remain unchanged. Rotor cases can then state `wake_core_overlap=2.5` once, removing their redundant legacy sigma_factor line. Preserve old restart identity when the new option isNone; include a supplied overlap in the identity so changed shedding controls reject continuation. The preceding source-based regression session82916 completed before the implementation (423passed).
- Tests before adoption: native emitted-radius coverage, both-kernel accepted no-penetration with the option, restart identity/continuation and legacy default equivalence. Frozen core-overlap studies remain immutable. A timestep-refined Gaussian comparison is still needed before full rotor qualification.
- Broad v13 regression completed:423passed,0failed,0skipped in682.909s, `/tmp/openonda-vlm-v13-regression.log` and matching XML. Native overlap follow-up32passed.
- Wind operating-point figures now use a physical six-revolution averaging window, matching loading/validation, instead of a fraction of total run duration.

### Native overlap implementation completed

- Added optional `VLMSetup.wake_core_overlap` to specify common trailing/transverse blob overlap. AIC and emitter use the identical rule. Omitted setting retains the previous formulas and restart identity; supplied overlap is part of native restart/configuration identity. Input validation stays inside the solver. Native no-penetration, emitted radii, unequal-root and restart follow-up32passed (`/tmp/openonda-native-core-overlap-followup.log`). Initial pytest collection issue with a parametrized default argument was corrected; no solver failure.
- v13/v14 default compatibility: generated restart identity, emitted positions, strength, radii, volumes and counts match exactly (`core-default-compat-v13.json`, `core-default-compat-v14.json`). Existing delta/flat input physics therefore does not need to change for this option. Frozen runtime-v14 successfully installed.
- The fine single-rotor allrun starts from zero, uses native v14 API rather than patched emitter code, and keeps all output isolated. Main rotor tutorials have not yet been switched to this candidate pending the timestep comparison.

- Fine-run launch initially found no allrun.sh because the previous single-rotor diagnostic folder contained only setup/assets. Copied the unchanged source quadcopter launcher into the fresh fine directory and launched successfully; failure log preserved separately. No simulation data existed during that initial launcher error.
- Final rotor cadence review: wind .12s field snapshots are only~2.1samples per blade-passage period; prefer .06s (~4.3) while retaining .012s force samples and one-revolution backups. Quad eight field snapshots/rev give onlyfour per two-blade passage; prefer12/rev (every8steps at3.75deg) while retaining48force samples/rev and backups every2rev. Apply with final selected rotor inputs, after timestep study.

- Delta cycle2 plotting QA exposed identical teal aliases for both wings. The rear wing now uses the shared purple palette; cycle title counts actually available complete cycles. Wake figure now shows native streamwise and vertical velocity separately with common per-component scales, making downwash visible. Actual allplot PNG preview passed and was inspected in `delta-preview-cycle2`; its samples/solution links are read-only inputs to plotting, and original run output is untouched. Far-wake downwash approaches the lower sampled-plane edge, so the final figure must state its sampled extent rather than imply complete wake coverage.
- First two complete revolutions of fine/coarse Gaussian comparison: mean thrust changes1.35%,6.29%, mean power0.05%,3.42%; waveform relativeL2 thrust10.52%,12.55%, power3.84%,4.57%. Both retain relaxation0.3 per step, so refinement also doubles relaxation frequency; this is a numerical-resolution comparison, not an isolated continuous-rate relaxation experiment. `rotor-time-refinement-progress.json`. Third revolution remains active.

- The lifecycle/coupling/output/style integration suite passes all97checks (`/tmp/openonda-v16-integration-tests.log`), including native overlap continuation and physical stage-field tests.
- Prepared but unstarted actual full workflows in `workflows-v16/delta_wing` and `workflows-v16/quadcopter`: native overlap2.5 without extra relaxation. Quad uses Gaussian and12field samples/revolution; delta keeps its original physics/time step and extends the sampled z window to−1.5m after the preview showed clipping. No particle-evolution bounds or acceptance threshold was relaxed. Adopt rotor changes only after the no-relaxation/refinement evidence is complete.
- Unrelaxed overlap rotor completes its third revolution with meanT=1.566855N/P=6.071350W (BEM1.621457N/6.167613W), substantially closer than the relaxed result. Six revolutions and an unrelaxed time-refinement comparison remain necessary.
- Reviewed possible self-gradient/relaxation changes but adopted none: native particle backends deliberately omit self-pairs, while arbitrary-target gradients retain the finite centre limit. The primary FLOWVPM direct/FMM implementation likewise omits zero-separation pairs. This does not by itself establish a solver bug; do not change the scheme merely to improve a plotted coefficient.

- `python install.py` completed in isolated `installer-v16` (system dependencies shared, OpenONDA itself installed inside that environment), including pip check and `python -I -m openonda.verify_install --require-site-packages`. Verification ran outside the checkout with no PYTHONPATH workaround and exercised native FVM, compiled Cartesian meshing, plotting/resources and direct-script help. Log: `/tmp/openonda-installer-v16-network-verification.log`. The first sandboxed attempt failed only because pip build isolation could not reach PyPI; normal network-authorized retry succeeded. The existing machine environment package was left untouched; final reinstall there remains.
- Two-second owned-wind OS profile: physical footprint1.0GB, peak3.1GB at step~400; most main-thread samples wait for Metal completion. No evidence of per-step recompilation in that sample. Keep long GPU jobs sequential. `/tmp/openonda-v16-wind-sample.txt`.
- Native turbine performance PNG preview rendered and inspected around three nominal revolutions. Startup overshoot decays toward the BEM references (CT=.6838172, CP=.4862516). The averaging legend now counts the available revolutions, rather than claiming six before six exist. Full horizon, final six-revolution statistics and downstream plane sampling remain pending.

- Stronger preservation recheck: all5,653 inventoried non-source files (results, geometry, figures and logs) match their original SHA256 hashes; none changed. `preservation-check-v16-all-nonsource.json`. The narrower scientific-data selection separately checks5,538files, also unchanged.
- Broad v16 regression is running: `/tmp/openonda-vlm-v16-regression.log`, JUnit `/tmp/openonda-vlm-v16-regression.xml`. Do not edit source Taichi code until it completes.
- Prepared `single-rotor-native-overlap-unrelaxed-fine` (1.875degrees, three revolutions,40kcapacity, Gaussian/FMM/two CPU threads). Start its actual allrun after the existing relaxed fine process78153/session44823 finishes. Use the verified `installer-v16/bin` Python with PYTHONPATH unset to combine the numerical refinement with a full direct installed-script run.

### Four-cycle comparison and next launch

- The unrelaxed single rotor has completed four revolutions. Revolution4 CT=.004647654 and CP=.000283992, versus attached-flow BEM .004743262 and .000287150 (−2.02% thrust, −1.10% power). Its third-to-fourth cycle mean change is1.40% thrust and0.47% power. Six revolutions and the independent unrelaxed time refinement remain required; this is not final certification. Updated native-data figure: `rotor-overlap-comparison-progress.png`.
- The ordinary base environment still contains an earlier installed OpenONDA version. A diagnostic launched with that interpreter from `/tmp` correctly failed to import the new rotor theory module; rerunning with the verified `installer-v16/bin/python` succeeded. This reinforces the pending final machine installation, not a need for tutorial path hacks.
- The transposed versus direct external-bound stretching question remains a hypothesis. Primary hybrid implementations also use total transposed stretching, including static bound particles. Do not adopt a global strength correction or change contraction solely to make the budget metric pass.

### Regression and live-metadata follow-up

- Full v16 regression completed:435 passed,0 failed,0 skipped in796.175s (`/tmp/openonda-vlm-v16-regression.xml`). Its source-based process has ended.
- Monitoring found that native metadata still reported created/step0 after several live checkpoints. The native solver now publishes running state after initial-condition construction and refreshes the same atomic metadata after checkpoints; manual stepping records partial state until close. No tutorial writer or extra metadata file is added. Focused lifecycle/output/backup/VLM-restart tests are active in `/tmp/openonda-v17-progress-metadata-tests.log`, session44285; do not edit source Taichi code until completion. Existing long runs retain their immutable v14/v16 runtimes.
- Turbine cycle4 means CT=.744625, CP=.557213; instantaneous values at t3.36s are .735163 and .545588. The wake-impulse force/blade-thrust ratio over cycles1–4 is1.030,1.044,1.039,1.061. These checks support continuing the candidate; neither four cycles nor instantaneous BEM proximity establishes final stationarity.
- Wind loading legends now state their actual averaging duration; the power envelope is labeled ideal steady disk so transient startup is not presented as a violation of a steady bound.

### Installed v17 and completed relaxed refinement

- Native progress metadata follow-up passes93 lifecycle/output/backup/VLM-restart tests; session44285 exited0. `python install.py` in fresh `installer-v17` completed pip check and isolated installed-package verification; session72763 exited0, `/tmp/openonda-installer-v17-verification.log`. The base machine environment remains untouched.
- Relaxed fine run completed576 steps. Its final native backup and all samples are preserved; Taichi emitted cache-lock warnings during shutdown only, with exit0. Do not clear the shared cache during other active simulations.
- Unrelaxed coarse rotor now completes5 revolutions; fifth-cycle means T=1.553748N and P=6.003245W (BEM1.621457N/6.167613W). The last three cycle means stay close to theory, but six cycles and time refinement are pending. `rotor-complete-cycle-means-v17.json` records both relaxed resolutions and unrelaxed coarse cycles.
- Late turbine-cycle section lift coefficient differs from BEM by9.86% in radial relativeL2 at this progress check. The earlier four-cycle average includes near-zero-rpm startup and is deliberately labeled as such; it is not the final six-revolution operating average.
- Tutorial-style documentation now agrees with the actual launchers: one case-local cd, direct Python commands, no custom extraction or metadata writers.

- Started the prepared final flat sweep after checking available memory; no need to wait for the coarse rotor. All current CPU runs use at mosttwo Taichi/BLAS threads each. Its original20-case source launcher is unchanged. No original outputs are destinations.
- Current long-run log monitoring reads both `Progress` and `Diagnostics` clocks. The fine unrelaxed case already reachedstep156 even though its last heartbeat-only line wasstep1; native diagnostic records were current throughout. Its installedv17 metadata correctly reports `running`; its first scheduled checkpoint is atstep384.

- Installedv17 versus frozenv16 VPM source comparison differs only in `core/solver.py`: four executable lines for metadata status/writes, plus the lifecycle docstring. All numerical source files are identical (`v16-v17-source-comparison.json`).
- First full unrelaxed rotor revolution: coarseT=2.406225N/P=5.991492W, refinedT=2.410509N/P=5.987590W (0.178% thrust,0.065% power change). This is an initial-cycle refinement result, not steady-state validation. The comparison figure now includes both unrelaxed resolutions.

- First final installed CPU flat case (`exp_moving_aoan10`) completed197 steps, tail CL=−.8538096085; its difference from the earlier full v7 Metal result is.00775%. Native metadata now advances at scheduled checkpoints during subsequent cases. Per-completed-case metrics are in `flat-installed-v17-completed-cases.json`.
- Latest progress at03:01UTC: turbine690/2400, coarse unrelaxed rotor526/576, fine unrelaxed rotor276/576, and second flat case nearing completion. Keep long GPU simulations sequential. Delta and full quad remain prepared but unstarted; both need complete final runs before committing.

### 9 September, 03:25 UTC — completion checks and cutoff benchmark

- Unrelaxed coarse rotor completed at576/t.09s in1h25m. Two-revolution continuation uses the native PathLike checkpoint API and separate output directory; it reached577/768 without a compatibility bypass. Fine unrelaxed refinement remains active near its second complete revolution. Final installed flat sweep has five complete cases and the sixth running.
- Fixed turbine checkpoint640 contains86,400 particles. An independent f64 host direct sum at67 stratified/extreme particle targets compared the standard mixed-core tail tolerance1e-7 with1e-5 and1e-4, preserving theta.3/order3. For1e-7, warmed fused evaluations1.45–1.48s;1e-5,1.17–1.18s;1e-4,1.06s. Velocity/gradient relative L2 direct-reference errors: default4.00e-5/5.08e-5;1e-5 gives4.39e-5/6.14e-5;1e-4 gives4.92e-5/7.34e-5. Record `wind-cutoff-benchmark-direct.json`. The first benchmark had slower default timing from transient contention; use the repeated direct-reference benchmark. Decision: retain the existing default and running numerical settings; the modest component gain does not justify another numerical configuration during this qualification. Both owned turbine pauses verified PID80399 and exact working directory, and resumed in `finally`; neither changed the running solver.
- Native VLM console loads now include dimensional Cartesian force, reference pressure and reference area so generic wing coefficients in axial rotor flow cannot be mistaken for the rotorcraft disk/tip-speed coefficients in the plots. Small surface areas print with four significant digits instead of rounding to0.00m². A rendered console check passed; these are display-only source changes newer than installedv17. Active immutable runtimes remain untouched.

- Fine unrelaxed second-cycle means: T=1.663789064N/P=6.209471890W, versus coarse1.662183736/6.207290566 (.0966%/.0351% differences). Third cycle remains incomplete. This supports early load time resolution, not long-time stationarity.


# Superseded checklist: before the user revised the validation sequence

The following record is historical. Its isolated-rotor prerequisites and delta convergence campaign were superseded by the user on 9 September. Use the current todos file for priorities.

# VLM tutorial completion checklist

Updated 9 September 2026, 08:04 UTC. **Work is ongoing; no commit has been made.**

Latest user requirements: report completed executions separately from physically
reliable results; explain the bound/wake drift and remaining solver fixes; match
Lamb–Oseen's fonts and the project's maximum figure width of12.5cm. Every future
permanent simulation must launch from its actual tutorial case directory. The
two remaining jobs retain their original working directories and write
through the previously established symlinks into physical tutorial output
folders. Do not repeat or restart them just to change their working directory.
No reference_flow simulation was found in the process/cwd inventory this turn.

Complete the VLM audit and actual `allrun.sh` workflows for flat plate, delta wing, wind turbine and quadcopter. Monitor steady/periodic/statistical convergence, extend when required, validate theoretical loading/velocity and rotor thrust/power, render and inspect all PNG/PDF figures, preserve existing data, then commit all accumulated authorized progress. Keep user-facing setups physics-focused, launchers direct, and all scientific output in native `samples/` with infrequent backups in `solution/`.

## Durable records

- Audit: [2026-09-vlm-audit.md](2026-09-vlm-audit.md).
- Complete earlier checklist/journal, retained without loss: [2026-09-vlm-history.md](2026-09-vlm-history.md). Treat its old active-state sections as historical.
- Earlier global tutorial qualification: [2026-09-tutorial-qualification.md](2026-09-tutorial-qualification.md). It records 52/66 bounded scenario passes, not successful full execution of every tutorial.
- Audit workspace: `/var/folders/kw/njsv3xwj69qf8p4bp4jw035h0000gn/T/openonda-vlm-audit-0fyxi5zd`; also stored in `/tmp/openonda-active-vlm-audit`.
- Active thread heartbeat: `complete-vlm-tutorial-qualification`, every ten minutes. Continue substantive work; remain quiet on unchanged status. Pause after completion or a blocker requiring user input.

## Active runs

The user explicitly requested current results inside tutorials, matching the case scripts, without rerunning completed simulations or backing up superseded outputs. The active audit working directories now link their output paths to physical directories inside tutorials. Read both **Diagnostics** and **Progress**; v16 turbine metadata predates progress updates, while v18 metadata updates at checkpoints. Never run allclean or another allrun against an active output directory.

| Run | State at update | Runtime and evidence |
|---|---|---|
| `workflows-v16/rotor_flow` | running, step1485/2400, t8.91s, N200475; process elapsed6h05m at08:03 | Frozen `runtime-v16`; session40669; owned PID80399; `/tmp/openonda-v16-rotor-full.log` |
| `workflows-v18-cpu/delta_wing` | running, step640/4000, t1.6s, N91390; process elapsed2h48m at08:03 | Immutable `installer-v18`; FMM two CPU threads; session36191; owned PID88579; `/tmp/openonda-v18-delta-full-cpu.log` |
| `continue_12` | completed1152/t.18s/N57600, native output checks pass; physically unqualified | session89037 exit0; PID87132 exited; final plots regenerated |
| `relaxed_moments` | failed step3/t.00046875: strength growth.1206% exceeds.1% gate | session88707 exit1; `/tmp/openonda-rotor-relaxed-moments.log`; native failed records retained |

The two-revolution extension completed at768/t.12s with38400particles in34m21s; session30584 exited0. Cycle8 means T=1.748437N/P=6.473058W, with thrust standard deviation.098966N. Cycle7→8 mean drift7.75%/4.69% remains above the intended stationary target. Final native CFL.04371/divergence.26588/misalignment34.18degrees. Original and extension final checkpoints are retained.

The extension reads `single-rotor-native-overlap-unrelaxed/solution/vpm_000576.h5` and writes a separate directory. The original completed run is untouched. Its launch mistakenly used `TI_NUM_THREADS=2`, which Taichi does not honor: this continuation uses the default Taichi CPU pool, with BLAS limited to two threads. **Use `TI_CPU_MAX_NUM_THREADS=2` for subsequent bounded CPU launches.** Do not describe this continuation as using two Taichi threads. A later memory-pressure check reported48% free; do not add more simulations concurrently.

Only one Metal simulation may run at a time. Verify exact owned PID and working directory before any necessary signal; never signal unrelated jobs or use SIGKILL. Do not edit frozen runtimes/installed environments used by active jobs. Do not edit Taichi source while source-importing jobs may compile it. Current jobs use frozen/installed code, so checkout edits are safe. Never reinstall `installer-v17` while its jobs run.

## Current numerical evidence

### Flat plate

- Earlier complete v7 actual sweep: all20 solver runs and native output contracts completed. Five PNG/PDF figure pairs were inspected in `workflows-v7-full/flat_plate/figures`.
- At5degrees: CL=.4284780597 versus lifting-line .4404155827 (−2.71%); CD=.00602752187 versus .0067045262 (−10.10%); quarter-chord CM=.0018865. Sectional lift L2 error3.97%; inboard downwash L2 error2.12%, full span25.37% with tip discrepancy. Moving/static CL difference.0039%. Maximum20-case late lift range.1517%, below.2%;24chords suffice for loads.
- **Existing strict bound/wake vector-strength validator still fails**:8degree residual2.0888%,15degree7.5512%,5degree.8106%, versus criterion1e-4. Do not weaken the criterion or hide the failure. Numerical budget diagnostics below have not fixed it.
- Installed-v17 CPU sweep **completed all20 cases**, session7845 exit0; only compute_device AUTO→CPU differs. Actual allplot PNG and PDF launchers both exited0 (sessions2949/66930). All five PDF renders and all five original PNG figures were visually checked: readable labels, no clipped panels, and the conservation failure is explicitly plotted. The PNGs were inspected individually after an earlier oversized output was truncated.
- Final native output/row/time/geometry/VTP/PVD/checkpoint contracts pass for all20 cases; `/tmp/openonda-vlm-full-v17-metrics.py`, session83890 exit0, `full-v17-qualification-metrics.json`. At5degrees CL=.4285115263 (−2.7029% LL), CD=.00602541561 (−10.1291%), sectional lift L2=3.9637%, inboard downwash L2=2.1181%, full-span25.3703%. Maximum20-case late CL range=.15238%; maximum nonzero CL CPU/Metal difference=.009082%. `flat-v17-cpu-versus-v7-metal.json`.
- Both preplot and final validators exit1 **only for the unchanged strict vector-strength closure failure**,2.089e-2 versus1e-4. `/tmp/openonda-flat-v17-preplot-validation.log` and `/tmp/openonda-flat-v17-final-validation.log`. No full certification claim.

### Single rotor / quadcopter qualification

- Isolated attached-flow BEM per rotor: thrust1.6214565N, shaft input6.16761345W; CT=.0047432617, CP=.0002871502 using rho*A*(OmegaR)^2/3.
- Completed coarse unrelaxed Gaussian/common overlap2.5:576steps/six revolutions, exit0 (session61017 closed), native checkpoint576 and completed metadata retained. Cycle thrust means1–6:2.4062253,1.6621837,1.5668546,1.5887736,1.5537478, approximately1.5136N. Sixth CT=.0044278241/CP=.0002752256 (−6.65%/−4.15% versus BEM). Last-three-cycle thrust trend remained >3%; the two-revolution continuation completed and a further continuation through twelve revolutions is active. CFL stayed about.04 but particle/vorticity misalignment grew; small CFL alone is not proof of accuracy.
- Continuation cycle7 means T=1.622750683N/P=6.183003582W, but larger oscillations remain: thrust standard deviation .0663N versus .0260N in cycle6. Native misalignment reaches36.75degrees/divergence.241 at7.25rev, with CFL.054. No steady-state claim. Native restart boundary is smooth (T1.56015N at576;1.56521N at578); full configurations differ only in surface file paths and requested additional steps, with embedded geometry identical. Evidence: `rotor-continuation-configuration-differences.json` and `rotor-unrelaxed-continuation-cycle-statistics.json`. The comparison plot now includes continuation CSV samples and marks the native restart.
- Fine unrelaxed study **completed576steps/3revolutions**, t.045s/N28800,1h49m11.8s, session33830 exit0. Fine third-cycle T1.566965370N/P6.071311064W; coarse1.566854590N/6.071350285W. Relative mean changes+.007070%/−.000646%; waveform L2 changes.02164%/.05197% at matching times. `rotor-unrelaxed-time-refinement-final.json`. This supports3.75degree early-load temporal resolution, not long-time wake stationarity. First/second mean T differences.1780%/.0966%; P differences−.0651%/+.0351%.
- Completed coarse Pedrizzetti.3 variant lost load continuously: sixth-cycle T1.225002N/P4.793300W. Fine Pedrizzetti.3 three-revolution variant completed; third T1.389860N/P5.621868W versus coarse1.364319N/5.562355W. This is not a qualified steady default. Same relaxation factor per step means refinement also changes relaxation frequency.
- Diagnostic comparison figure/script: audit `rotor-overlap-comparison-progress.png/.json`, external `/tmp/openonda-rotor-overlap-comparison.py`. Regenerate with installedv17 Python from `/tmp`, `MPLCONFIGDIR=/tmp/openonda-mpl`. Reads native CSV/metadata only. Include separate continuation data deliberately if extending this comparison.

### Wind turbine

- Active candidate: Gaussian, native `wake_core_overlap=2.5`, native Pedrizzetti.3, dt.006,14.4s/2400steps,400kcapacity, Tree theta.3/order3. Forces.012s; fields.06s; backups one nominal revolution. Planes start at about9.78s and have not yet written.
- BEM: thrust2325.5465N, torque1417.4222Nm, power11575.6146W; CT=.6838172262, CP=.4862515866.
- Complete cycle means1–8 CT:.53275,.80708,.76302,.74462,.73419,.72738,.722715,.719495; CP:.43632,.63596,.58003,.55721,.54441,.53610,.530445,.526594. Cycle7→8 drift−.4456%CT/−.7260%CP. Still trending; do not certify steady.
- Impulse-force/blade-thrust ratios cycles1–4:1.030,1.044,1.039,1.061. Latest late-cycle sectional Cl L2 difference from BEM9.86%. An early all-available-window loading plot includes startup at near-zero angular speed and can look much higher; do not misdiagnose that as a normalization bug.
- Final validator uses last six revolutions: CT/CP drift2%, BEM15%, impulse-force10%, plane drift1%. Assess actual downstream field convergence, not loads alone.

### Delta wing

- Earlier v13 actual full attempt failed at step995/t2.4875s of10s with strain CFL1.06. `workflows-v13/delta_wing` outputs and last backup800 retained; session91393 exited1. No process was killed by this task.
- Prepared candidate `workflows-v16/delta_wing`: Gaussian/common overlap2.5/no relaxation, DNSnu.001,dt.0025,10cycles/4000steps,250kcapacity, Tree theta.3/order3/AUTO. Sampling planes now span z−1.5..+.9 to include the descending wake; evolution bounds remain z±1.5. It has not run yet. Demonstrate stability/convergence before adopting in source.
- Final validator compares phase-resolved forces over the last three complete cycles,5% drift.

## Prepared full workflows and source differences

- Full quad prepared in `workflows-v16/quadcopter`: four rotors, Gaussian/common overlap2.5/no relaxation,24rev/2304steps at3.75degrees,500kcapacity, Tree theta.3/order3/AUTO. Forces48/rev, fields12/rev, backups every2rev. Await the isolated refinement/continuation evidence before full launch. Prefer a fresh installed runtime with final display-only changes; preserve existing prepared inputs if making another copy.
- To satisfy the latest reproducibility request, source turbine and delta now exactly match their active candidate inputs (turbine Gaussian/overlap2.5/PDR.3/fields.06s; delta Gaussian/overlap2.5/CPU-FMM/noPDR). Their READMEs identify ongoing qualification; this is not a completed-convergence claim. Source full quad still uses legacy Winckelmans/no common overlap and8field samples/rev; select its final inputs after the isolated continuation assessment.
- Flat original20-case configuration equivalence is recorded in `flat-input-equivalence.json`; wind input unification also retained exact native configuration. User launchers already use a single case-local cd and direct Python commands; cleanup is cd plus removal of output directories. Never run cleanup in original data-bearing tutorials.

## Verified solver changes

Detailed audit and history contain code paths, tests and primary sources. Preserve the completed fixes:

- Consistent three-leg finite bound induction and analytic Jacobian, accepted-state near-wake AIC/emission ordering, local TE motion, shared-root filament merge, native per-surface force/power and VLM sampling under samples/, exact native restart version4, reference frames and cached/validated linear solves.
- Optional native common wake-core overlap, with legacy default geometry/emissions/restart identity bit-identical. Native overlap/restart tests32passed.
- Mixed-core tree far-tail admissibility, finite host origin Jacobian, correct particle self-pair exclusion, grouped GPU wake uploads, removed per-step host transfers, reduced Fourier integral memory, native elapsed timing and progress metadata.
- Fourier benchmark at fixed delta400:4.432s/3.564GB →3.588s/2.544GB, same integrals. Native elapsed-time test distinguishes28s total from4s evolution.
- v17 differs numerically from v16 **only in metadata progress writes**; exact source comparison recorded in `v16-v17-source-comparison.json`. Metadata reports running at run entry and refreshes at successful backups; managed runs stay running, manual advances partial, finalizer gives terminal state. No extra metadata file or tutorial code.
- New checkout-only display changes afterv17: VLM log adds Cartesian force in N and coefficient reference pressure/area; mesh total area uses four significant digits. Rendered console check passed. No numerical/sample changes.
- Cutoff benchmark at turbine640/86,400particles: independent f64 direct sum at67targets. Default1e-7 warmed fused1.45–1.48s, velocity/gradient L2 errors4.00e-5/5.08e-5. Candidate1e-5 takes1.17–1.18s, errors4.39e-5/6.14e-5. **Default retained**; no live configuration change. `wind-cutoff-benchmark-direct.json`. Owned wind pauses always resumed in finally.

## Unresolved conservation investigation

The six-case native versus external-direct contraction refinement completed with installedv17 at exactly t.25s. Native coarse/medium/fine maximum residuals:.115165%,.157715%,.205599%; external-direct:.027158%,.039422%,.058339%. Both increase with refinement. **This rejects external-direct contraction as an established conservation fix; production behavior stays unchanged.** All six numerical runs completed, session25554 exit0. Records: `budget-direct-refinement-v17-results.json` and `budget-v17-*/audit-result.json`. These are instrumented diagnostics; the external contraction is not represented by production metadata. The first attempt failed before solving because the old fixture omitted its geometry helper; its directory/log were preserved with an import-failure suffix, and the copied diagnostic fixture was repaired using the current geometry helper.

At common t.25s, refinement residuals .115%,.158%,.206% did not approach zero. External-direct bound stretching reduced medium residual to.0397% but did not pass. Direct+total TE convection.1232%; overlap-only.1161%; Gaussian bound quadrature/transposed.1801%, direct.0415%; unfiltered bound.278%. No diagnostic monkeypatch was adopted. Do not add a global strength correction merely to fit the check. PDR is a pre-strength stage using the correct old-state gradient; it does not modify the accepted newborn row after the AIC. Upstream particle solvers likewise omit self pairs. Neither hypothesis established a new bug.

## Validation and preservation

- Full v16 regression:435passed,0failed,0skipped,796.175s; `/tmp/openonda-vlm-v16-regression.xml`. Earlier fullv13:423passed; batches overlap, do not sum.
- v17 progress metadata/lifecycle/output checks:93passed; `/tmp/openonda-v17-progress-metadata-tests.log`, session44285 closed0.
- Ordinary `installer-v17/bin/python install.py` passed, pip check clean, and `python -I -m openonda.verify_install --require-site-packages` passed outside the checkout including compiled mesher/resources/direct CLI. This venv shares system dependencies, so it is not a pristine dependency bootstrap. Log `/tmp/openonda-installer-v17-verification.log`; session72763 closed0. Base environment still needs final refresh; do not rely on its old package from /tmp.
- Latest full non-source preservation check:5653 original files unchanged, `preservation-check-v16-all-nonsource.json`. Protect originals using `original-data-hashes.json` and `other-original-file-hashes.json`; exclude deliberately edited .py/.sh entries from the latter. Prior global19,515-file check differed only in a .DS_Store. Recheck before commit.
- Repository `development` has hundreds of authorized preexisting modified/untracked FVM, coupler, docs and tutorial files. Preserve and review all accumulated progress before the requested commit. No commit or push has occurred.
- The other VPM task now has a new official four-case suite queued for an exclusive Metal slot after our turbine; it reports that it will wait without interrupting these jobs. Preserve vortex_interactions/setup_les.py, allrun/allplot and its readers/docs. Its main Anaconda installation was refreshed with the validated Fourier memory changes; our frozen/installed jobs were untouched. Do not reinstall shared environments while its jobs depend on them. No cross-task tool message was sent: an earlier detailed message was automatically rejected and optional authorization remains unanswered. Do not resend without authorization, and do not launch a second Metal job if its suite starts.

## Remaining completion steps

1. Monitor the two active jobs. The isolated refinements and12revolution continuation have completed; diagnose an admissible rotor stabilization approach before selecting full-quad inputs. The moment-restoring proposal rejection is not caused by conditioning; do not repeat that diagnostic.
2. Resolve or explicitly retain the failed flat-plate conservation evidence; all20 runs, native contracts, PNG/PDF plotting and visual QA are complete. Do not claim a full certification pass.
3. Finish turbine full horizon and native plane/load/impulse validation, extending if required.
4. Finish the active delta workflow and run full quad from its actual tutorial directory when resources and diagnosed rotor settings permit; assess health, phase/cycle convergence, rotor theory and native fields.
5. Keep final setups/cadences/READMEs in agreement; run allplot PNG/PDF and inspect every figure. No duplicate native metadata or sampling implementations.
6. Verify final regression/output/restart/installer contracts; refresh the base installation through ordinary installer only when safe.
7. Recheck original data hashes, update final audit and qualifications, inspect git status/diff/staged content, commit all authorized accumulated progress, and explain commit purpose. Pause the heartbeat only when the requested work is complete or genuinely needs user input.

## Verified capacity extension and active twelve-revolution continuation

The native restart storage increase is implemented and verified. It permits larger allocations only when filament refinement and regularization are disabled. Checksums, all physical settings, smaller allocations and capacity-dependent adaptation remain protected. All45 backup/coupled-restart tests passed (0failed/skipped,416.828s), including Direct/Treecode/FMM particle trajectories and f64 coupled motion/circulation/sample comparisons across256→512 allocation. Session3154 closed0; `/tmp/openonda-restart-capacity-tests.xml` and log. No source-importing test remains active.

Ordinary `installer-v18/bin/python install.py` passed, pip check clean and outside-checkout `python -I -m openonda.verify_install --require-site-packages` passed. Session52945 closed0; `/tmp/openonda-installer-v18-verification.log`. V18 changes only three Python files relative to v17: backup compatibility and the two display-only changes; `v17-current-source-comparison.json`. Numerical evolution is unchanged. **Do not mutate installer-v18 while its continuation runs.** Base installation still awaits final refresh.

Actual `single-rotor-native-overlap-revolutions9-12/allrun.sh` is active in session89037 with installedv18 and no PYTHONPATH. It loaded original extension checkpoint768 read-only, allocated60kparticles, and requests384 additional steps, ending at1152/t.18s/twelve revolutions with nominal57600particles. Native metadata confirms initial_step768, initial_time.12, initial_count38400, statusrunning. All physics, dt, Gaussian/overlap2.5/no-relaxation and samplers match; only storage and the continuation horizon change. Output remains separate. The launch correctly uses TI_CPU_MAX_NUM_THREADS=2, two numerical-library threads and an isolated Taichi cache path, `/tmp/openonda-taichi-v18-rotor12`.

## Completed spatial comparison

`single-rotor-native-overlap-spatial-fine` started after the temporal study completed, using actualallrun, installedv18, no PYTHONPATH, two CPU threads and isolated cache `/tmp/openonda-taichi-v18-spatial`. It restores3.75degrees per step and doubles blade mesh from4×12 to8×24 per blade, retaining Gaussian/common overlap2.5/no relaxation. Capacity40k covers nominal28224 particles. Native VLM snapshots once per revolution supplement force/integral samples. All288steps/3revolutions completed at t.045s/N28224, elapsed27m53.2s; session4600 closed0, native checkpoint288 and completed metadata retained. Third-cycle thrust1.514127767N/inputpower5.938204581W differ−3.3651%/−2.1930% from the4×12 mesh. This complements the temporal study; it is not another relaxation sweep.

## 05:06 UTC follow-through

A source audit found another silent force-reporting failure: ConservationTracker passed the removed reference_speed keyword to VLMSolver.compute_forces and swallowed the resulting TypeError. It now calls the supported native reference-velocity resolution and propagates errors. Two genuine solved-lattice tests verify nonzero dimensional forces for static/moving plates and failure propagation (2passed,14.91s; /tmp/openonda-conservation-tracker-tests.xml, session9979 closed0). This checkout-only diagnostic change does not affect active installed/frozen numerical runs, and does not fix vector-strength drift. No source-importing tests remain active.

Prepared `workflows-v18-cpu/delta_wing`, preserving the full v16 physical inputs/horizon/sampling but selecting native CPU/FMM induction to run alongside the Metal turbine. Current source plotters read embedded native geometry. Its actual allrun started after the spatial rotor finished, in session36191. Native CPU/FMM/Gaussian startup and running metadata verified; isolated cache `/tmp/openonda-taichi-v18-delta`. CPU/FMM is already covered by production regression and rotor refinement, but this full delta configuration still needs qualification. Do not overwrite the prepared v16 case. Memory pressure currently reports58% free; disk39GiB available.

The new external audit plot `/tmp/openonda-rotor-resolution-comparison.py` reads native metadata and samples, and shows matching first-three-revolution comparisons. Its current PNG was visually inspected. First/second complete refined-mesh thrust differs−3.35%/−3.61%, power−1.54%/−2.21%; the final third-cycle comparison is recorded above. `rotor-resolution-comparison-progress.json/.png/.pdf`. The overlap comparison now includes the separate9–12rev native continuation and uses the current embedded-geometry reader.

## Native plotting window labels

Rotor performance/loading labels previously inferred averaging duration from the absolute final clock. A restarted segment could therefore be labelled six revolutions despite containing fewer samples. Source wind performance/loading and quad performance now report the actual sampled revolution range. Wind and quad wake plots likewise identify their sampled intervals. No new metadata or extraction path is introduced. Three live-native-sample previews rendered and were visually inspected under `plot-window-review-v18/`: wind performance/loading and the isolated rotor9–12continuation performance (currently onlyrev8.0–9.2). Session39455 exit0. Wake-label render checks await published full candidate planes. Copy current source plotters into completed isolated workflow copies before their final actualallplot runs; do not alter active frozen runtimes. Source plotting edits do not change numerical evolution.

## Completed rotor loading comparison

`rotor-loading-refinement.json/.png/.pdf` and external `/tmp/openonda-rotor-loading-refinement.py` compare the third revolution from native sectional forces/circulation and sampled bound positions. The native constant rotation/pivot maps positions back to radial stations. Integrated blade thrust is.78342784N coarse,.78348647N temporal-fine,.75706825N spatial-fine, consistent with half the rotor loads. At the coarse stations, temporal refinement changes axial-loading/circulation L2 by.0758%/.1157%; spatial refinement changes them5.948%/10.395%, concentrated toward the root. Global thrust/power are much less sensitive, but **do not claim pointwise loading is fully mesh converged**. The final resolution and loading PNGs and both PDF renders were visually inspected; readable labels/no clipped panels. `rotor-spatial-refinement-final.json` preserves the completed cycle comparison. The initial plotting script read the native ndarray wrapper incorrectly, failed before writing figures, and was corrected to its serialized values.

The active full delta now uses the installed CPU/FMM backend to progress alongside the Metal turbine; its source physical parameters and samplers match the prepared v16 candidate. The source delta setup remains unmodified pending full qualification. Full quad remains unstarted; use the temporal/spatial evidence and twelve-revolution health assessment when selecting its global-force resolution. Avoid further broad parameter sweeps without a specific diagnosed issue.

## Results now physically in tutorials — latest user instruction completed

The user explicitly replaced the old preservation requirement for these result
folders: do not back up old outputs, do not rerun completed simulations, and make
case scripts reproduce the transferred solutions in separate output folders.
Old outputs in these four VLM tutorials were therefore superseded or removed
without archives. Original-data hash checks must account for these authorized
replacements; unrelated original datasets remain protected.

- `tutorials/vpm/flat_plate/solution/exp_<mode>_aoa<angle>/` and matching `samples/`
  contain all20 installed-v17 CPU cases. Ten PNG/PDF figures are in its `figures/`.
  Copied setup differs from the previous source only by AUTO→CPU. Source geometry,
  physics, horizon and sampler identities match all20 native records exactly.
- `tutorials/vpm/rotor_flow/solution/` and `samples/rotor/` are the physical live
  v16 turbine outputs. Current native-data performance/loading PNGs are in
  `figures/`; downstream plane figures await samples. No restart was performed.
- `tutorials/vpm/delta_wing/solution/` and `samples/delta_wing/` are the physical
  live v18 CPU/FMM outputs. No restart was performed.
- `tutorials/vpm/quadcopter/studies/solution/<case>/` and `samples/<case>/` contain
  coarse, time_refined, mesh_refined, relaxed, relaxed_time_refined, continue_8,
  and the live continue_12. Six cases completed; continue_12 remains active.
  Comparison PNG/PDF figures and the continuation performance preview are in
  studies/figures. The obsolete full-four-rotor outputs were removed, so no old
  solution is displayed as a new full-quad qualification. Full quad is pending.

The three live moves were same-volume directory renames with source-path
symlinks pointing INTO the physical tutorial directories, not tutorial symlinks
pointing into temporary storage. Exact owned PIDs/cwd were reverified with lsof;
all three jobs were briefly stopped and resumed in finally, taking about.5s
altogether, with no simulation repeated. Later Progress records confirmed all
continued; continue_12 also published checkpoint960 in its tutorial directory.
`live-tutorial-output-relocation.json` is the audit mapping. Do not remove those
audit-path symlinks while active jobs run.

Study setup is a single direct `python setup.py --case <name>` with an explicit
physics table and native restart paths. Its allrun/allplot/allclean are direct
commands. Geometry generation and native-data plotting reuse the parent quad
helpers. The standalone installed tutorial catalog now includes
vpm/quadcopter/studies, with two explicit support_files copied only when missing.
Main base installation does not yet include this latest catalog change; final
ordinary installer refresh remains necessary when the shared environment is safe.

All29 configurations were constructed without creating or advancing a solver and
compared with their native records: all20 plates plus turbine/delta are exact
apart from absolute surface paths; six native rotor cases differ only in output
locations, with identical geometry. The older coarse relaxed experiment lacks
native wake_core_overlap and embedded reference fields, because it predates the
API; its explicit experimental formula is now expressed as native overlap2.5.
Vertices and mesh are unchanged; original metadata is retained without edits.
`/tmp/openonda-check-relocated-configs.py` and
`tutorial-relocated-input-equivalence.json` preserve the comparison.

All1214 completed native checkpoint/sample/log/metadata files were rehashed after
transfer and remain byte-for-byte identical. Records: flat-tutorial-result-transfer,
rotor-study-result-transfer and tutorial-transfer-verification JSON in the audit.
The relocated actual allplot launchers passed PNG and PDF from /tmp with the
normal installed Python and PYTHONPATH unset; no simulation was rerun. Flat
velocity and updated rotor-relaxation PNGs were visually rechecked. Source/style,
geometry/results and installed-tutorial tests passed26/26 in37.793s, no skips,
`/tmp/openonda-relocated-tutorial-tests-final.xml`; session12665 closed0. The first
pass had one catalog omission, fixed by registering the independent study and
its shared source dependencies, then rerun successfully. The final four-command study allplot passed both PNG and PDF after removing
verbose audit JSON output (sessions28588/80185 closed0). All four study PDFs
were rendered with pdftoppm and visually inspected; the final launcher/style
check passed too (session22087 closed0).

A plain git diff --check tried to run Git LFS's clean filter against live samples
and hit sandbox write restrictions in .git/lfs/tmp. Code/docs whitespace check
passed with LFS filters disabled and explicit *.py/*.sh/*.md path selection.
This was not an approval-review rejection, and no data was altered. A complete
staged LFS review belongs to the authorized final git commit after qualification.

## Diagnostic reporting changes after v18

The native energy-rate source now has a read-only public property. Console
logging labels direct/Fourier grid-transition viscous fallbacks as a viscous
estimate instead of claiming a complete energy derivative. ConservationTracker
now records the actual viscous kinetic-energy rate, rather than the total-rate
field, and its force reporting uses the supported compute_forces API instead of
silently swallowing TypeError. Nine focused reporting/force tests passed
(20deselected,16.84s), `/tmp/openonda-diagnostic-reporting-v19-tests.xml`,
session28249 closed0. These are checkout diagnostic changes; active frozen/
installed numerical evolution is unchanged. No new Fourier/evolution edits
were made after declaring the memory fixes validated to the other task.

## Thesis-style figures and updated qualification evidence

Lamb–Oseen calls set_thesis_style(), not the portable set_style(): its actual
PDFs use TeX Gyre Pagella text and NewPX mathematics at the shared10.95pt size.
All four VLM tutorial plotters and isolated-study plotters now use that same
style. Wide layouts were stacked; all measured PDF widths are at most12.5cm,
with bbox_inches=None preserving physical dimensions and shared400dpi PNGs.
Late-window labels use actual sample times. LaTeX percent signs and degree
symbols were corrected after rendered inspection; no numerical data changed.
The style guidelines are now explicit in docs/development/tutorial_style.md.

Actual flat and isolated-study allplot launchers completed both PNG/PDF, as did
current turbine performance/loading. All nine flat/study PDFs and both turbine
PDFs were rendered and inspected. Fonts and page widths were checked with
pdffonts/pdfinfo. Final percent/degree corrections and larger restart annotation
were re-rendered. Delta's three figures and full-quad wake layout were exercised
with preserved older native data **only in /tmp/openonda-vlm-thesis-native-previews**;
these are layout checks, not current physical qualification results. Do not copy
those old-data previews into the active tutorial figures. Actual final delta,
full quad and turbine-plane figures still require their current samples.
Evidence: /tmp/openonda-vlm-thesis-figure-review/verified.json and
wind-and-preview-verified.json; previews and logs share that prefix. Latest
style/results tests passed3/3, /tmp/openonda-vlm-thesis-style-tests.xml; code/docs
whitespace check passed. Earlier26-case contract/style checks remain applicable.

The native per-angle flat closure check was measured across all20 cases:

| Positive angle | Moving maximum residual | Static maximum residual |
|---|---:|---:|
| 2deg | .129768% | .129928% |
| 5deg | .811924% | .813993% |
| 8deg | 2.082341% | 2.088793% |
| 10deg | 3.274062% | 3.287071% |
| 12deg | 4.747276% | 4.769032% |
| 15deg | 7.511433% | 7.554447% |

Negative-angle residuals closely match their positive counterparts. All18
nonzero-angle cases fail the unchanged0.01% criterion. The two zero-angle cases
have zero strength and are trivial passes, not evidence that nonzero-lift wakes
are qualified. Approximately quadratic incidence scaling and frame agreement
point to a systematic nonlinear coupling discrepancy, not a graph or reference
frame error. The earlier stage budget locates the dominant imbalance in bound
stretching versus the accepted bound/shedding update. No conservative repair
has been established; do not describe the residual as roundoff, claim it is
fixed, or apply a global strength adjustment merely to satisfy the check.

At06:26UTC the turbine's complete cycle9/10 means are CT=.716220/.713556 and
CP=.522598/.519312. Cycle9→10 drift is−.372%/−.629%; final six-cycle and downstream
checks remain pending. Native health at t7.8: CFL.03781, divergence.12682,
misalignment.454deg. Isolated rotor cycle10 means CT=.00457830/CP=.000273595,
with CT standard deviation.00034497 (7.5% of its mean). At10.75rev its native
misalignment reaches71.04deg, divergence.35916 and CFL.08696. The low CFL and
mean proximity to BEM do not establish long-time wake reliability. Finish the
twelve-revolution diagnostic before selecting full-quad settings. Delta has
passed one cycle; at t1.08 CFL.05211/divergence.13470/misalignment22.08deg, which
is not enough history for the three-cycle periodic criterion.

These values are recorded by /tmp/openonda-vlm-current-native-status.py in the
audit's native-progress-2026-09-09-thesis-style.json. It only reads native CSVs
and solver metadata; the JSON is an external audit record, not tutorial metadata.

## Deformed-wake shedding check and eleventh rotor revolution

Two new independent closure regressions prescribe an affine deformation of a
previous closing wake polyline, unequal mirrored-half loads, a change in every
panel's circulation, and stationary/rotating body geometry. The integrated
old wake, current bound field and emitted particles close to3e-14m³/s. Together
with existing bound-leg and shared-root checks, five tests passed with no skips,
session12276 closed0; /tmp/openonda-vlm-deformed-wake-budget-tests.xml and log.
The test sums transported polygon segments independently of the emitter's
spanwise differencing. It narrows the full-run investigation: basic row topology
and changing-circulation closure pass under prescribed deformation; interaction
with the evolving regularized particle representation remains unresolved.
No production evolution change or new simulation was made during this check.

Native progress at06:41UTC: turbine field step1340/t8.04s, CFL.03995,
divergence.12423, misalignment.915deg. Delta field step480/t1.2s,
CFL.04610/divergence.15369/misalignment25.04deg. Isolated continuation passed
eleven revolutions: cycle11 CT=.0043086181/CP=.0002572242, changes−5.890% and
−5.983% versus cycle10. CT standard deviation.000669740 is15.54% of its mean.
Atstep1056/t.165s the native CFL=.07786/divergence=.35793/misalignment54.62deg;
the misalignment oscillates and had reached71deg previously. No steady claim.
The three original processes remain active and all physical output paths are
inside their tutorials. Their installed/frozen health implementation was checked:
misalignment uses freshly evaluated curl(u), not the backup-only blob-vorticity
array, so the growing diagnostic is not an artifact of backup frequency.

## Rotor checkpoint localization and targeted moment-preserving candidate

A read-only diagnostic reconstructed native VLM state from continue_12/checkpoint960
and independently summed the Gaussian particle Jacobian in f64 at the512 native
health probe indices, excluding self pairs. Adding the native bound Jacobian gives
53.2896deg weighted misalignment, versus native FMM53.3024deg. Thus FMM approximation
does not explain the large angle. Within half a rotor radius of the rotor plane,
weighted misalignment is65.14deg; particles younger than one revolution give60.28deg.
The defect is not confined to the old far wake. These are stratified probe estimates,
not exhaustive per-particle statistics. Native count48000 matches50particles/step
with no removal. No time advance or native result write occurred; the diagnostic
only generated temporary input geometry and an external audit record.
Evidence: /tmp/openonda-rotor-checkpoint-health.py and .log, session53270 exit0;
rotor-checkpoint10-misalignment-localization.json in the audit.

Native wake impulse provides another independent consistency check. The unrelaxed
coarse run has -rho*dI_z/dt divided by integrated blade thrust of.992–.995 over
cycles2–6, then.803 in cycle9,1.146 in10 and1.438 in11 as the wake deteriorates.
Ordinary P-relaxation yields1.255,1.856,1.999,2.428,2.304 in cycles2–6. The refined
relaxed run also reaches1.988 atcycle3, while unrelaxed refinement remains.996.
Sparse native bound-polyline impulse changes over revolutions2–4 and4–6 contribute
at most.00218N, too small to explain the relaxed mismatch of order1N. This remains
a diagnostic, not complete force certification with every unsteady term.
Evidence: /tmp/openonda-rotor-impulse-balance.py; rotor-native-wake-impulse-balance.json.
Its first invocation used removed np.trapz and failed before writing a result;
it now uses scipy.integrate.trapezoid and completed successfully (session31253).

This diagnoses a concrete reason to test the **existing native**
pedrizzetti_relaxation_preserve_moments option, which restores the pre-relaxation
strength/impulse moments, rather than altering the physical evolution's budget.
`tutorials/vpm/quadcopter/studies/setup.py --case relaxed_moments` is prepared:
same Gaussian overlap2.5,3.75degree step,4×12panels,PDR.3,80kcapacity as relaxed,
with native moment preservation and12revolutions. Its separate outputs will be
solution/relaxed_moments and samples/relaxed_moments. allrun contains the direct
eighth command; README identifies this as unstarted and unqualified. The existing
seven cases' settings are unchanged. Input-only comparison confirms byte-identical
geometry and no other physical changes; rotor-moment-input-check.json. Tutorial
style check passed, /tmp/openonda-rotor-moments-tutorial-style.xml; whitespace clean.

**Next:** after owned continue_12 PID87132 actually exits, inspect its terminal
metadata/native checkpoint and final stats, then launch this one targeted candidate
from the actual studies directory with immutable installer-v18 Python and external
TI_CPU_MAX_NUM_THREADS=2 / BLAS thread limits. Do not rerun the seven previous cases.
Do not start it concurrently with the existing three simulations. Candidate is
not yet launched, and no result exists. Once native samples appear, add its series
to the relaxation comparison and its own performance command to allplot, then
regenerate/inspect those figures. The ordinary moment-restoration mechanism was
not changed. This is a diagnosed comparison, not a broad parameter sweep.

## Native-sample wake-health figure

Added studies/assets/plot_rotor_health.py and its direct allplot command. It reads
only native flow_integrals/vlm_surface_forces and solver metadata. Three panels
show curl/strength misalignment, normalized divergence, and full-revolution wake
impulse change divided by integrated blade thrust. The unrelaxed history includes
both native continuations. The denominator uses trapezoidal integration of the
dense force history; incomplete revolutions and the unsampled startup interval
are excluded. The README explains that the impulse ratio excludes bound and other
unsteady contributions and is a diagnostic rather than a complete certificate.

Actual PNG/PDF exports passed (sessions72791/86335 exit0) and both were visually
inspected. PDF page354.331×510.236pt is12.5×18cm; pdffonts confirms Pagella/NewPX.
Files are studies/figures/rotor_health.png and .pdf; rendered review at
/tmp/openonda-vlm-thesis-figure-review/rotor_health.png. The tutorial-style check
passed, /tmp/openonda-rotor-health-tutorial-style.xml; whitespace clean. Current
series are coarse and relaxed; **when relaxed_moments native samples appear, add
that name to this allplot command too**. Its supported label/color already exists.

At07:20 the isolated continuation still had21steps left; it has not yet exited
and relaxed_moments has not started. The turbine and delta also remain active.
Avoid merely reporting unchanged progress on subsequent checks.

Before certifying full quad, audit its validator's wake checks: it currently
checks that two native plane collections exist, but does not yet assess their
temporal convergence or wake-impulse/load consistency. The existing load/BEM/
ideal-power checks remain useful; plane existence alone must not imply a qualified
downstream field. No full-quad pass has been claimed.

## Twelve-revolution completion and failed moment-preserving comparison (07:47 UTC)

The native continue_12 run completed1152steps/t.18s/57600particles in3h12m21.3s.
Session89037 closed0; PID87132 exited. Native lifecycle, final checkpoint1152,
coupled restart version4, finite data and sample clocks pass. Final check evidence:
rotor-twelve-revolution-final.json and /tmp/openonda-rotor-12-final-check.py.
All seven original rotor studies are now completed; completed execution is not
physical qualification. Last six-revolution means T1.60588176N/P5.87815373W differ
from BEM by−.96054%/−4.69322%, but half-window drift6.01162%/12.22327% exceeds3%.
Cycle12 thrust/power fluctuations are about30%/33% of their means. Final native
CFL.240597/divergence.465000/misalignment75.0828degrees. The full-cycle wake-impulse
thrust / blade-thrust ratio reaches1.965255 atcycle12; mean agreement alone is
misleading. All five actual allplot PNG/PDF commands were rerun successfully
(sessions56347/86043 exit0); final changed figure visual review follows.

The targeted relaxed_moments case was launched from the actual tutorial studies
directory using immutable installer-v18 and TI_CPU_MAX_NUM_THREADS=2/BLAS2. It
FAILED atstep3/t.00046875 with150particles after12.6seconds. Session88707 exit1.
Native minimum-norm moment restoration produced total strength-magnitude growth
1.206e-3, beyond the existing1e-3 limit. The solver restored the original strengths
and rejected the event. Keep that acceptance gate unchanged. Native failed
metadata and partial force samples are retained in solution/relaxed_moments and
samples/relaxed_moments; flow_integrals do not exist because step12 was not reached.
Do not rerun/overwrite this failed case, or add it to a plot requiring missing
integrals. The previous unstarted/active notes above are historical, superseded
by this terminal result. No native relaxation fix has yet been implemented.

Next: audit moment-restoration constraints, rank/conditioning and acceptance
semantics before any fix. Do not add an arbitrary global budget correction or
raise limits. Any justified follow-up must retain the failed records and use a
separate actual tutorial case directory and a fresh runtime if code changes.
Full quad is still unrun; neither existing isolated default is qualified.

## Correction audit, native coupled moments and stricter wake qualification (08:01 UTC)

The bounded four-step diagnostic replay reproduced relaxed_moments' step3
rejection exactly. It ran in a new audit-only directory, never touched the
permanent failed candidate, and stopped at the same native gate. An independent
row-scaled rectangular SVD gives the same strength growth.00120609105934;
relative proposal difference1.72e-14. The150-particle constraint matrix condition
is1406.50 (row-scaled53.87), and native moment residuals are below1.4e-17.
Thus this failure is not explained by Gram conditioning or moment-restoration
arithmetic. The minimum-norm restoration preserves the nine requested moments,
but those linear constraints do not guarantee the separate nonlinear magnitude
bound. No gate was relaxed and no damping or alternative relaxation algorithm
was silently substituted. Further rotor stabilization work remains necessary.
Evidence: /tmp/openonda-audit-rotor-moment-proposals.py/.log, session29200 exit0
(the wrapper catches the expected native rejection), and
rotor-moment-proposal-audit-jma2t958/proposal-audit.json + three proposal NPZs.
These are instrumented diagnostic records, not additional qualified tutorials.

Fixed a distinct reporting defect: ConservationTracker.impulse_total contained
only wake impulse. Native VLMSolver.compute_bound_linear_impulse now integrates
all three finite on-wing filament legs in coupled mode (quarter-chord-only in
standalone mode). Tracker total includes bound plus wake with density, and its
CSV adds explicit bound/total columns while retaining established wake-only
linear_impulse columns. Removed its arbitrary EXCELLENT/GOOD/ACCEPTABLE labels,
which could conflict with the case's much stricter declared closure criterion.
Independent Gauss-Legendre filament integration and origin-translation checks,
force tracking and CSV semantics:5passed,19.996s, session10084 exit0,
/tmp/openonda-vlm-bound-impulse-tests.xml. First fixture mistakenly let a
standalone solve reset coupled mode; corrected the fixture before the final pass.
This diagnostic correction does not fix or excuse the physical drift.

Native FlowIntegralsSampler now publishes bound_linear_impulse_xyz,
coupled_linear_impulse_xyz, bound_vortex_strength_xyz and coupled_vortex_strength_xyz
for VLM runs at the existing sampler cadence. Impulse stays per density[m4/s];
old particle-only columns keep their meaning. No tutorial extractor or metadata
writer was added. A two-step native coupled sampler test passes, including
non-unit density, native samples/ location and absent checkpoints. Test is in
output_contracts; /tmp/openonda-vlm-native-coupled-moments-tests.xml, session4575
exit0. Earlier fixture failures were inconsistent inviscid viscosities and reading
Taichi fields after managed run teardown; the final test configures zero viscosity
consistently and reads before close using two normal native advances.
Full output-contract regression is currently session33622; record its result.
All active simulations and shared installed runtimes remain unchanged. These
new sampler columns require a fresh runtime for subsequent runs; do not rewrite
or fabricate them in older saved samples.

Full-quad wake qualification is stricter. A shared native PVD/VTK reader serves
the plotter and validator. Validation now requires the declared two planes,
finite vectors, unchanged grids, ordered times and complete final six-revolution
cadence. Two three-revolution mean vector fields must differ by <=3% relative to
the induced velocity (freestream subtracted); spatial cancellation or a large
uniform inflow cannot conceal drift. Missing rotor force histories also fail.
Native bound-plus-wake axial impulse change must agree with integrated thrust
over the final six revolutions within10%; this assumes retained wake, and
boundary losses require separate accounting. Old samples without native coupled
moments remain unqualified, not retroactively filled. Nine wake/momentum/style
checks passed1.810s: periodic-field pass, stale/gapped/duplicate/nonfinite failures,
spatially cancelling drift failure and correct/incorrect impulse budgets.
/tmp/openonda-quadcopter-wake-validation-tests.xml; session11026 exit0.

Final twelve-revolution study PNG/PDFs were visually reviewed. The relaxation
legend obscured late oscillations, so it is now outside the data area; regenerated
PNG/PDF session74590 exit0 and inspected PDF render. Its page is12.5x25cm;
Pagella/NewPX embedded. Other final health/performance panels were inspected and
fit the shared fonts and width. Delta current native allplot exports are now
being generated inside its actual tutorial, with explicit common wake time
window. First invocation accidentally replaced PATH and hid LaTeX; it exited
before plotting. Retried with inherited PATH plus the OpenONDA Python directory;
no launcher or dependency changes were needed. Record final sessions/results.

## Completed verification and current delta figures (08:04 UTC)

The complete output-contract module passed34tests in21.268s, no failures/skips;
session33622 exit0; /tmp/openonda-vlm-coupled-moment-output-contracts.xml.
The native coupled sampler integration passed independently in14.415s before
that full module pass. No test process remains active.

Actual delta allplot PDF and PNG commands both completed (sessions45180/36267
exit0). All three PDF renders and all three original PNGs were visually inspected.
They use Pagella/NewPX and widths12.4968cm; force height21.9964cm, wake17.9959cm,
strength history7.99465cm. /tmp/openonda-vlm-thesis-figure-review/delta-current-verified.json.
These are now current native progress figures inside tutorials/vpm/delta_wing/figures,
not the older temporary layout previews. Force history reaches about1.6s and
only one full cycle is available; no convergence claim. The wake average explicitly
shows t.60–1.56s and all three planes use that same published time window. The
expanded z-range contains the depicted wake at this early time. The sampled
centroid panel also exposes the authored motion: integrating opposite-phase
sinusoidal velocities from the same initial height gives different mean heave
heights, rather than both wings oscillating about the same height. Assess whether
that is the intended crossing-wake geometry before changing any physical inputs;
do not silently alter the live run. A final small caption edit uses 'Complete
cycles shown' for singular/plural clarity; force-only PNG/PDF exports completed
(sessions5186/97038 exit0), and the final PDF render was inspected.

At08:03, only owned wind PID80399 and delta PID88579 remain active; no reference_flow
simulation was found. Wind progress1485/2400/t8.91s/N200475; delta640/4000/t1.6s/N91390.
Their physical results continue inside tutorials. Full quad remains unrun, and no
new permanent simulation was launched after the failed relaxed_moments comparison.
No installed runtime or running case was changed. Further work: rotor stabilization
and bound/wake strength closure, final wind/delta physical qualification, fullquad,
final ordinary installation/regression/preservation review, then the requested commit.
