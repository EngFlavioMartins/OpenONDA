# VLM audit and flat-plate qualification

Current status is in [the sequential checklist](2026-09-vlm-todos.md) and
[the bound/free exchange diagnosis](2026-09-vlm-circulation-budget.md). Later
runs mentioned below were stopped at the user's request. Only flat-plate work
is active; the entries below retain the chronological audit evidence.

## Audit before implementation

The audit began with the four existing flat-plate figures and their raw historical CSVs. Existing results were copied and hashed before testing. Disposable audit directory: `/var/folders/kw/njsv3xwj69qf8p4bp4jw035h0000gn/T/openonda-vlm-audit-0fyxi5zd`. `original-data-hashes.json` protects 4,093 original result and surface files. Historical data, short diagnostic probes and full workflows are retained in separate directories.

Plot contract: compare signed lift, induced drag, quarter-chord moment, spanwise circulation/loading, and induced velocity against attached-flow theory. Use the tutorial's Matplotlib style, dimensional velocity axes and nondimensional force axes, direct solver CSV/VTP samples, explicit final-time windows, and exported PNG/PDF files in the isolated case's `figures/`. Inspect exported figures and check the underlying numerical errors; a successful image export alone is not a physics pass.

### Findings recorded before fixes

1. **Incorrect reference calculation.** Both lifting-line implementations use odd Fourier modes at symmetric full-span collocation points. The 20/40/120-term matrices have ranks 10/20/60 and condition numbers 2.0e17/1.4e16/2.6e17. Their attractive curves are not reliable reference solutions. Share one well-posed implementation and verify convergence independently.
2. **Hidden moment data.** The polar requests `pitching_moment_coefficient_c4`, whereas solver samples contain `pitching_moment_coefficient_quarter_chord`. Replotting raw samples exposes differences between moving/static moments, especially at larger angles.
3. **Inconsistent induced field.** The coupled circulation system uses on-surface horseshoes ending at the wing trailing edge, whereas RK wake advection and collocation postprocessing use closed rings extending 10,000 m downstream. This reintroduces artificial wake legs alongside the free VPM wake. Arbitrary-point samplers omit VLM induction altogether.
4. **Ill-resolved wake gradients and excess work.** Stage gradients use six additional full lattice evaluations and a tiny fixed finite-difference interval in f32, with near-singular filaments passing through newly shed particles. Derive and verify an analytic regularized filament Jacobian, with a documented particle-scale regularization for particle targets. Preserve accumulation of independent contributions. The apparent gradient overwrite is a private scratch-field write, not overwriting the particle gradient; its caller correctly accumulates it.
5. **Frame-dependent shedding.** Rotor-oriented normal kicks also affect a translating plate in still air. The local branch stores relative wind as the new particle's lab-frame velocity. Use physical relative convection for shed strength and lab-frame fluid velocity for particle state; verify moving/static equivalence.
6. **Forces sample the wrong locations.** Wake and kinematic velocities evaluated at collocation points are reused at bound midpoints. This changes force and moment in nonuniform/rotating flow. Evaluate each at its actual target.
7. **Reference/output inconsistency.** VTK pressure jump assumes 1 m/s for any static surface instead of the configured reference speed. Quarter-chord moments assume an unrotated point `(c/4,0,0)` even when the plate geometry is pitched or translated. Single-surface transforms and per-surface force selection match wing UIDs against surface names and miss ordinary `flat_plate` / `main_wing` geometry.
8. **Missing VLM sample dispatch.** Force/distribution CSVs already use samples, but the old VLM VTK exporter targets the backup directory and is no longer dispatched. Add a normal scientific sampler using the existing output manager, with sample cadence and restart-safe indexes.
9. **Avoidable allocation/transfer.** Every solve constructs a new linear solver (including persistent GPU fields); mesh summary downloads the area field once per panel; initial transforms upload every scalar individually, even when no transform was requested. Reuse solver state and transfer arrays in batches. Cache a dense LU only when the matrix is exactly unchanged.
10. **Unverified iterative convergence.** BiCGSTAB returns on breakdown or exhausted iterations without checking the true residual. Add explicit residual-based success/failure and test difficult/small right-hand sides.
11. **Dead/incorrect diagnostics.** Uncalled near-wake and disabled-CG helper kernels remain, a debug claim `rhs[0]/A[0,0] == gamma[0]` ignores matrix coupling, and several comments claim unsteady force components that are not implemented. Remove dead private paths and state the actual Kutta–Joukowski scope.

The audit covered the VLM orchestration, influence/force kernels, mesh/topology and transforms, loading/VTK/CSV output, linear solver strategies, wake shedding, kinematics interfaces, VPM stage/target interfaces, and tutorial readers/reference functions. It is not a certification of arbitrary aircraft, separated flow, rotor performance, or noncirculatory unsteady loads.

### Theory sources

MIT 16.100, [Force Calculations for Lifting Line](https://ocw.mit.edu/courses/16-100-aerodynamics-fall-2005/0d4e4bdb7badca18ff23efc20f99a285_16100lectre18_cg.pdf): circulation Fourier series, sectional Kutta–Joukowski lift, integrated lift/induced drag, induced angle and elliptic limit. NASA Glenn, [Downwash Effects on Lift](https://www1.grc.nasa.gov/beginners-guide-to-aeronautics/downwash-effects-on-lift/): downwash reduces effective incidence and tilts the aerodynamic force downstream. These references describe attached inviscid flow; they do not establish viscous drag or stall accuracy at high incidence.

## Follow-through findings

Tracing accepted-state output also exposed duplicate VLM CSV writes during flow-integral refresh and writes under the previous step's clock. VLM output now has one accepted-step path. Smooth-ramp geometry was advanced over `[t, t+dt]` even though the caller supplied the new clock; it now reaches the requested time exactly.

VPM backups previously omitted all live VLM continuation state. A validated VLM subgroup now records lattice fields and mutable motion, rejects missing/incompatible state before mutation, and restores the next circulation solve. Non-VLM backup layout remains unchanged. Runtime motion is copied from the declaration, so solving a case does not mutate its input metadata.

The public force evaluator also used zero background speed to normalize a moving plate and hid failures behind zero forces. It now uses the VLM reference flow, defaults to the case density, and propagates actual errors.

## Implementation and validation

Initial isolated full 5-degree cases completed 192 static and 197 moving steps. Subsequent full sweeps and their runtime versions are recorded below. Original result files remain protected.

Independent regression coverage includes filament quadrature/Jacobians, well-posed lifting-line convergence, finite-wing loading integration, moving/static forces and moments, single-surface placement/name overrides, ramp pose timing, LU invalidation, BiCGSTAB relative residuals including very small right-hand sides, sample-directory VTK/PVD output and pressure normalization, and VLM continuation. The first related integration batch passed all 94 tests.

On CPU (f64, two Taichi threads, 224 panels, 3,000 target points), warmed median gradient evaluation fell from 0.0961 s to 0.0366 s, a 2.63x speedup. Away from filament cores, the analytic and previous finite-difference gradients differ by 1.75e-9 in relative L2 norm. Repeated solves of an identical 224x224 dense matrix fell from 1.465 ms to 0.0915 ms with LU reuse, a 16.0x speedup. These are component benchmarks, not whole-simulation speedups. Full results are in `performance.json`.


## Coupled-wake audit during full workflows

Full rotor runs exposed additional issues that short probes did not reveal. Each change below followed a source trace or a reproducing run.

- **Accepted step and near-wake geometry:** the old method advanced the body before transporting the old wake, used a global mean convection offset, extended only the last chordwise panel's circulation, and advected newborn particles again during their birth step. The current sequence transports the old wake, advances prescribed geometry to the new clock, solves circulation with a local completed near-wake row, and deposits that row once. Trailing strength uses cumulative chordwise circulation and shared spanwise edges. An eight-step coupled Galilean-invariance regression compares full particle states.
- **Newborn AIC consistency:** thin near-wake filaments in the circulation system were replaced after solving by smoothed midpoint particles with a different induced velocity. At an accepted static8-degree checkpoint the resulting normal-velocity residual reached .302m/s; for a small rotor it reached1.106m/s. The AIC now includes the exact positions, linear strength coefficients, radii and native radial kernel of the emitted row. The static8-degree maximum residual is3.10e-6m/s (RMS1.01e-6m/s) in Metal/f32. Independent four-step tests cover Gaussian and Winckelmans kernels and two geometry scales. Native Gaussian host/device erf approximations set the corresponding regression tolerance.
- **Dimensionally inconsistent core:** a finite-filament denominator mixed squared length and squared area. It now uses the same analytic Rosenhead line integral as the stage evaluator. Coupled restart version3 prevents silently continuing pre-fix wake histories.
- **Stretching selection and transfer:** the VLM contribution always used the transposed contraction even when the case selected direct or mixed stretching. It now uses the selected native contraction, on device, without downloading/uploading every particle gradient and rate at each RK stage. Six independent linear-field tests cover all formulations with and without a requested diagnostic gradient.
- **Per-step kernel compilation:** grouped wake insertion declared a new Taichi kernel on each call. Moving it onto the particle container eliminated20extra compilations in20appends. A32-particle CPU component benchmark fell from5.88ms to.0439ms; a four-step rotor comparison was bit-identical. This is a component result, not a134x whole-simulation claim.
- **Rotor geometry and power:** quadcopter leading edges were reversed relative to rotation. Both rotation senses now have the correct blade orientation and radial quarter chord. Per-surface force, pivot moment, actual angular/translational velocity and fluid-on-body power are native sampled outputs. Shaft power is summed per blade, so counter-rotating shaft torques cannot cancel its magnitude. Centroids use actual panel areas.
- **Native stabilization edge cases:** Pedrizzetti relaxation tried mutating an empty starting cloud, and a rejected event could leave modified particle strengths and accepted-event counters. Empty clouds are skipped; rejected or failed relaxation restores the exact previous strengths, and acceptance gates run before event bookkeeping. No health threshold was increased.
- **Bound-strength accounting:** a coupled horseshoe's integrated vector includes all three on-wing segments. The diagnostic now telescopes their endpoints at the strip trailing edge, which matters for taper/twist. A closed-polygon test verifies cancellation with its closing wake segment; rectangular flat-plate values are unchanged.

## Tutorial and output contract

All four launchers run direct `python setup.py` commands after one case-local directory change. Plot launchers similarly run individual Python plotters. The wind setup's two configurable builders and optional configuration gate were removed; native serialized configuration and all sampler identities are unchanged (`rotor-setup-cleanup-equivalence.json`). User-facing metadata writers and checkpoint extraction paths are not part of these workflows.

VLM force/distribution CSVs, per-surface power CSVs, geometry snapshots and velocity planes are native outputs under `samples/`. Plotters read native VPM metadata, including operating parameters, motion, sampled clocks and lifecycle. Backups have a separate, slower schedule. Incomplete runs are rejected before empty averaging windows are evaluated. Rotor and delta figures label their averaging or phase windows; signed forces, per-rotor power and wake velocity are retained.

Wind and propeller references share `openonda/rotor_theory.py`: bracketed blade-element/momentum inflow, Prandtl hub/tip factors, Buhl high-induction treatment for the turbine, and the appropriate propeller annular momentum equations. Tests independently verify momentum balances and dimensional scaling. See [CCBlade theory](https://wisdem.readthedocs.io/en/master/wisdem/ccblade/theory.html). The reference uses attached thin-plate lift and no profile drag; it is not a high-Re stall or viscous-drag model. Rotor thrust and power agreement must therefore be reported with that scope. Native Pedrizzetti relaxation follows the established [particle-alignment model](https://flow.byu.edu/FLOWUnsteady/api/flowvpm-relaxation/); its transfer must be measured, not hidden by a conservation claim.

## Current qualification evidence (work remains)

- Latest full VPM plus related tutorial/output/installation regression:423passed,0failed,0skipped in682.909s (`/tmp/openonda-vlm-v13-regression.xml`). Later focused native-relaxation tests:24passed; bound-vector/restart tests:19passed; installation/style tests:22passed. These batches overlap and must not be added into an inflated unique-test count.
- The runtime-v7 actual 20-case flat sweep and all native output contracts passed at `workflows-v7-full/flat_plate`. At 5 degrees, CL=.428478 versus lifting-line .440416 (−2.71%), CD=.00602752 versus .00670453 (−10.10%), and quarter-chord CM=.00188651. Sectional lift L2 error is3.97%; downwash L2error is2.12% inboard and25.37% over the full span, concentrated at tips. Moving/static CL differs by.0039%. Final-five-chord lift range stays below.152% in all20 cases. Allfive PNG/PDF figures were exported and visually checked; native cell-centred span stations are plotted without synthetic tip markers. Metrics: `full-v7-qualification-metrics.json`.
- **Unresolved vector-strength drift:** current static8-degree full-run residual is2.089%, above the existing1e-4 criterion. Joint grid/time refinement at the same physical time t=.25s gives.115%,.158%,.206% from coarse to fine; it does not demonstrate convergence to zero. Removing target filtering worsens the medium-grid residual to.278%, and adding bound-induced trailing-edge convection gives.177%; neither diagnostic was adopted. Data are in `budget-refinement-common-time.json`, `budget-bound-filter-comparison.json` and `budget-te-velocity-comparison.json`.
- **Unqualified long rotor stability:** corrected native-wake single rotor completes3revolutions at3.75degrees per step, whereas the previous coupling failed at2.625revolutions. It is not yet a steady-state result. The full wind v7 attempt failed atstep170/2400 with strain CFL1.19. Previous quad and wind failures, deliberately interrupted obsolete-code runs, and all outputs are retained separately. Full corrected delta and quad execution remains required.
- Original preservation audit:5,552 result/geometry files retain their hashes (`preservation-check-v7.json`). Scientific output from qualification runs is kept outside the checkout.

### Further native output and diagnostic cost audit

Native VPM metadata now snapshots the actual loaded VLM surface geometry through the existing surface serializer. The shared plotting reader uses this embedded record; historical records can still read the original local geometry file. A regression deletes the input file after construction and verifies exact native metadata and plotting readback. Flat input cleanup preserves all20 physics/run/backup/plotted-sampler configurations while removing unused final crossflow planes. Uncalled private mesh generators and preconditioned matvec kernels were removed; the public initial trailing-leg length is now honored and independently tested.

A full delta run exposed excessive Fourier diagnostic allocation. When one support dimension grew, the persistent-grid code enlarged allthree dimensions by25%; repeated streamwise growth unnecessarily expanded the transverse grid. The fix expands only axes that lack stencil slack. An elongated-wake sizing regression and the Fourier/free-space batch pass all13 tests. The physical particle update and scientific sampling cadence are unchanged. The affected v9 outputs remain preserved; the current fullv13 allrun includes this fix.

A single-rotor checkpoint comparison quantified CPU backend cost: warmed treecode step12.56s versus FMM3.86s on two threads. Aftertwo steps, particle-strength relativeL2difference is2.44e-6, circulation8.69e-6, and maximum positiondifference4.66e-8m. These are explicit isolated backend comparisons, not silently modified production restarts. The six-revolution core-overlap experiment is rerunning fromzero on FMM; no core-model change has yet been accepted into repository physics.

The asymmetric-wing audit also found an omitted common-root wake filament. The two root-to-tip parameterizations have opposite signed circulation; their root-edge strengths add, and cancel only for equal physical loading. A prescribed unequal-load startup row had a missing[.12,.03,.018]m3/s vector. Emission and the newborn AIC now include this shared element once. A precision-scaled cancellation floor prevents the circulation solve's last-bit symmetric noise from changing particle topology between Galilean frames. Native restart version4 distinguishes this coupling history. Final checks are recorded in the completion checklist.

Mixed core radii previously prevented distant tree nodes from being accepted at any distance, making core-spreading wakes approach direct pair cost. Acceptance now also permits nodes wholly beyond the native kernel regularization tail at1e-7 tolerance; near-core and cancelled nodes still descend. At the exact57,600-particle delta checkpoint, warmed fused evaluation falls from~2.66s to~.675s, with velocity and gradient relative L2differences3.97e-5. Independent near/far native-kernel checks pass. Their new source-centre probe exposed a separate host-reference error: regularized velocity is zero there but its Jacobian is finite. The host contract now uses that finite limit, verified by finite differences for allfour kernels. Particle reference tests explicitly exclude self-pairs, matching particle backends. The subsequent FMM batch passes27checks.

Allfour tutorial geometry helpers now use native surface I/O, removing duplicate serializers/deserializers. Loaded geometry and reference values remain exactly identical in `geometry-io-cleanup-equivalence.json`. Nominal chord Reynolds numbers from native rotor inputs are~0.47–1.15million for the turbine and~22,000–63,000 for the small quadcopter; the latter must not be described as the same high-Reynolds operating regime.

Further Fourier memory reduction preserves the exact grid, padding, source smoothing and quadratic integrals, while padding each component inside the FFT and accumulating components sequentially. At the same delta checkpoint400, wall time falls4.432s→3.588s and peak RSS3.564GB→2.544GB. The17 Fourier/free-space regressions pass, including comparison with explicitly centered three-component padding. This is diagnostic cost reduction, with no change to particle evolution or output cadence.

Run progress previously reported only the evolution profiler, excluding expensive accepted-state diagnostics, sampling and backups. The solver now owns a separate elapsed run clock; terminal reporting freezes after final sampling/backup. The detailed evolution timer retains its meaning. All58 lifecycle, output and logging checks pass, including a simulated clock that independently accounts for each category of work.

### Live progress metadata and installer follow-through

Monitoring the long turbine and rotor studies showed that native metadata stayed at `created`, step0 until finalization, despite existing numerical checkpoints. The solver now records `running` after constructing initial conditions and updates the same atomic metadata file after checkpoint writes. Manual stepping records `partial`; finalization still owns the terminal status. Sampling and checkpoint cadence are unchanged, and tutorials need no metadata code. Two real-solver regressions observe the file before completion and check scheduled/manual checkpoints. The lifecycle/output/backup/VLM-restart batch passes93 tests.

The preceding broad v16 suite passes435 tests with no failures or skips in796.175s (`/tmp/openonda-vlm-v16-regression.xml`). Normal `python install.py` also succeeds in isolated `installer-v17`, sharing existing dependency packages while installing OpenONDA locally. Its outside-checkout isolated verifier exercises FVM, compiled meshing, Taichi, resources, plotting and direct-script help; pip check reports no broken requirements. The machine's existing OpenONDA installation has not yet been replaced.

The three-revolution refined rotor with relaxation completed all576 steps. Third-cycle thrust/power means are1.389860N/5.621868W, versus1.364319N/5.562355W on the coarser relaxed grid. Both retain a falling-load trend and remain unsuitable as steady reference results. Without relaxation, the coarse rotor's fourth and fifth cycle means are1.588774N/6.099780W and1.553748N/6.003245W, against BEM1.621457N/6.167613W. The matching unrelaxed refinement is now running through the actual direct launcher and installedv17 interpreter. Full steady-state qualification remains open.

## 9 September: completed unrelaxed rotor and bounded tree benchmark

The six-revolution unrelaxed Gaussian/FMM single rotor completed at576steps with native metadata reporting completed. Its sixth-cycle CT=.0044278241 and CP=.0002752256 differ from isolated attached-flow BEM by−6.65% and−4.15%. The last-three-cycle thrust trend still exceeds3%, so a separate native-checkpoint continuation is running for two additional revolutions. Do not classify this as steady yet.

A fixed-state tree benchmark at turbine checkpoint640 tested mixed-core tail cutoffs against independent f64 direct summation at67 particle targets. Default1e-7 gives velocity/gradient relative L2 errors4.00e-5/5.08e-5. A1e-5 candidate gives4.39e-5/6.14e-5 and reduces warmed fused induction from about1.46s to1.17s. The default remains unchanged during qualification. Evidence: `wind-cutoff-benchmark-direct.json` under the audit workspace. Owned-process pauses were bounded and always resumed.

The native VLM console now reports dimensional Cartesian force and the generic force-coefficient reference pressure/area, making rotor normalization explicit without tutorial diagnostics code. Mesh surface areas use four significant digits. These display-only edits do not alter evolution or samples.

## 9 September, 03:49 UTC: continuation and convergence monitoring

Nine installed flat-plate cases have completed; the20-case workflow remains active. The turbine completed six nominal revolutions, with sixth-cycle CT=.727380 and CP=.536101 using the recorded quarter-chord rotor radius6.005739m. The corresponding fifth-cycle values are.734194/.544407; downstream plane sampling has not started yet.

The isolated rotor's seventh cycle gives T1.622750683N/P6.183003582W, close to BEM, but thrust oscillation standard deviation increased from.0260N in cycle6 to.0663N in cycle7. Native misalignment/divergence reached36.75degrees/.241 near7.25rev while CFL remained.054. This is not steady-state qualification. Force/power records are continuous across the native restart; the recorded configuration differs only in moved geometry paths and requested continuation steps, with identical embedded geometry. The comparison figure now includes the continuation and marks the restart.

Prepared a six-case, common-time native versus external-direct bound-contraction refinement diagnostic using installedv17. It has not started; wait until the ongoing rotor continuation finishes before consuming another CPU slot. Production contraction is unchanged.

## 9 September: completed conservation refinement and restart allocation audit

The common-time six-case study completed. Native maximum bound/wake vector-strength residuals at t.25s rise from.115165% to.157715% to.205599% with refinement. Direct external-bound contraction lowers their magnitude, but likewise rises from.027158% to.039422% to.058339%. It is not an established convergent repair and is not adopted. The initial diagnostic fixture import failure was corrected in an isolated copy, preserving the failed attempt; all six numerical cases then completed.

The isolated rotor reached eight revolutions. Eighth-cycle T=1.748437N/P=6.473058W still differs from cycle7 by7.75%/4.69%, so mean agreement with BEM does not establish steady loading. Its final checkpoint768 is preserved. Extending further requires a larger allocation; source audit identified that the existing restart comparison rejects even a pure storage increase. A narrow compatibility change and trajectory/adaptation guard tests are being verified before another isolated installation or continuation.

## 9 September, 04:25 UTC: verified restart storage increase

All45 backup/coupled-restart tests passed with no failures or skips in416.828s. Direct, Treecode and FMM preserve a continued particle trajectory with larger allocation; f64 VLM cases preserve motion, circulation and sampled fields across256→512. Smaller capacities and enabled capacity-dependent adaptation still fail before state mutation.

The ordinary isolatedv18 installer, dependency check and outside-checkout verification passed. Only backup compatibility and console display files differ from v17; evolution is unchanged. The single rotor is now continuing from its preserved eighth-cycle checkpoint to twelve revolutions with60kstorage and two CPU threads. All prior outputs remain separate and intact. The installed flat sweep has17/20 complete cases.

## 9 September: final installed flat sweep and temporal refinement

The installed-v17 CPU flat sweep completed all20 actual allrun cases. All native output/row/time/geometry/VTP/PVD/checkpoint contracts passed. Actual PNG and PDF allplot launchers both exited0; allfive original PNGs and five PDF renders were visually inspected with readable labels and no clipped panels. At5degrees CL=.4285115263 differs−2.7029% from lifting-line; CD=.00602541561 differs−10.1291%. Sectional lift L2error3.9637%; inboard downwash2.1181%, fullspan25.3703%. The maximum20-case late lift range is.15238%, and maximum nonzero CPU/Metal lift difference.009082%. Both numerical validators still fail only the existing strict bound/wake vector-strength check:2.089e-2 versus1e-4. There is no full certification pass. Evidence: full-v17-qualification-metrics.json and flat-v17-cpu-versus-v7-metal.json.

Unrelaxed temporal rotor refinement completed576steps/3revolutions. Halving3.75degrees to1.875degrees changes third-cycle mean thrust+.007070% and power−.000646%; waveform L2changes are.02164%/.05197% at common times. This establishes early-load time-step resolution, not long-time wake stationarity. The doubled blade mesh study is running separately; the coarse rotor continues through twelve revolutions because its eighth-cycle oscillations and means are still changing.

ConservationTracker silently returned zero Kutta–Joukowski force because it passed an obsolete reference_speed keyword and caught every exception. Its native force call now uses the supported reference resolution and propagates errors. Two solved static/moving-lattice regressions pass, including nonzero dimensional force and failed-record behavior. This is an independent diagnostic repair; it does not resolve the vector-strength failure.

Rotor figure labels now show the actual sampled revolution interval rather than inferring an averaging duration from absolute time. This corrects misleading six-revolution labels for short native-checkpoint continuation segments. Wind/quad wake labels also identify their time windows. Three native-data performance/loading previews rendered and were visually inspected; wake label inspection remains pending the full-run plane outputs. No metadata writer or data extractor was added.

The turbine completed eight nominal revolutions with CT=.719495 and CP=.526594. Their successive-cycle changes are−.4456%/−.7260%; downstream plane sampling has not yet begun, so this is improving load convergence, not full wake qualification.

## 9 September, 05:19 UTC: mesh comparison complete; full delta started

The doubled rotor blade mesh completed all288steps/3revolutions, retaining its native final checkpoint and completed metadata. Third-cycle T=1.514127767N/P=5.938204581W changes−3.3651%/−2.1930% versus4×12panels. Native sampled blade-loading comparison gives spatial L2changes5.948% in axial loading and10.395% in circulation at coarse stations, concentrated near the root; temporal changes are.0758%/.1157%. Integrated blade forces agree with half the complete rotor means. This supports modest global-force sensitivity but does not establish pointwise loading convergence. Both comparison PNGs and their PDF renders were visually inspected. Evidence: rotor-spatial-refinement-final.json and rotor-loading-refinement.json/.png/.pdf.

The full actual delta-wing allrun is now active in workflows-v18-cpu/delta_wing with installedv18, native CPU/FMM/Gaussian and two CPU threads. Physical inputs, ten-cycle horizon,100force samples/25field snapshots per cycle and one checkpoint per cycle match the prepared v16 candidate. Source adoption awaits actual stability/convergence/plane evidence. Wind and the separate twelve-revolution rotor continuation remain active. No original data was overwritten and no commit has been made.


## 9 September, 05:54 UTC: reproducible results placed in tutorials

The user's latest instruction was executed without repeating simulations:
completed flat20 and six isolated-rotor datasets were transferred and1214 native
files verified byte-for-byte. The three active output trees were moved on the
same volume into physical tutorial folders, with audit-path symlinks preserving
live writes. No active cwd or runtime was changed; verified owned jobs were
resumed immediately and continued advancing. Superseded outputs were removed
without backups, as expressly requested. Full quad remains pending and has no
new full-four-rotor result to present.

Flat setup now matches the completed CPU sweep. Turbine/delta setups match their
active candidates and identify ongoing qualification in README. Isolated cases
have one parameterized physics setup, separate solution/<case> and samples/<case>,
direct launchers, native restart dependencies and native-data comparison plots.
The installed catalog includes this study with its two shared helper dependencies.
Twenty-nine constructed configurations were compared with original native
records without initializing a solver; output-path changes are intentional, and
the pre-native relaxed experiment's core formula is now stated by the native
wake_core_overlap option. Metadata was not rewritten. Native figures can be
regenerated by the tutorial allplot launchers, which passed PNG/PDF from /tmp
using installed OpenONDA. Twenty-six relevant tests passed. Full details and
artifact paths are in the TODO's latest results-placement section.

The last source-only diagnostic fixes distinguish a native viscous energy-rate
estimate from an actual finite-difference energy derivative, expose its existing
source label, and correct ConservationTracker force/viscous-rate reporting.
Nine focused tests passed. Neither fix changes active numerical trajectories or
repairs the still-failing flat-plate vector-strength closure criterion.

## 9 September, 06:26 UTC: figure consistency and honest qualification status

The VLM figures now use the exact thesis style used by Lamb–Oseen: Pagella/
Palatino text and NewPX mathematics. Previous portable-serif figures and several
oversized layouts were corrected. Panels are stacked within12.5cm width, shared
font sizes remain readable, and uncropped saves retain the declared dimensions.
PDF font embedding and page widths were measured; all available flat/study and
turbine load figures were visually inspected. Escaped LaTeX percent signs and
degree symbols fixed missing labels. Delta and full-quad wake layouts were also
checked using preserved older native samples in /tmp only; those previews are
not current qualification results. Actual final plane figures remain pending.
Three existing style/results tests passed, with a clean code/docs whitespace
check. The completion checklist records all rendering evidence and active jobs.

All20 flat executions completed, but all18 nonzero-angle cases fail the existing
0.01% signed vector-strength closure threshold. Static positive-angle maxima
are .129928%, .813993%, 2.088793%, 3.287071%, 4.769032% and7.554447% at2,5,8,10,12
and15degrees. Moving results and opposite-incidence magnitudes closely agree.
The approximately quadratic angle dependence supports a systematic nonlinear
coupling discrepancy. It does not prove the precise cause or repair. Stage
budget evidence points to external bound stretching versus the bound/shedding
update; alternative contractions did not demonstrate convergence. Production
behavior and the strict validator remain unchanged. This integrated signed
vector budget is distinct from the unsigned sum of wake-strength magnitudes,
which is not a conserved quantity, and from an individual filament's circulation.

Turbine cycle10 CT=.713556 and CP=.519312 are approaching their BEM references
but still drift; plane samples have not begun. The isolated rotor's tenth-cycle
CT standard deviation is7.5% of its mean, and native misalignment reaches71deg
at10.75revolutions. This remains an unreliable long-time steady-flow result.
The revised delta run has passed its first cycle only. The full-four-rotor case
has not been run with the candidate settings. No blanket success or final commit
is justified yet.

The three existing processes remain active with physical outputs inside their
tutorials. No reference_flow process was found. The user now explicitly requires
every subsequent permanent simulation to launch from its actual tutorial case;
do not rerun completed cases or restart these existing processes merely to
change their working directories.

## 9 September, 06:46 UTC: independent shedding closure under deformation

Added two numerical regressions that transport an old closing wake polyline
through an affine fluid deformation, change the panel circulations, and move
the bound body independently. Unequal loads on mirrored halves exercise the
shared root. For stationary and rotated body geometry, the integrated old wake,
current bound field and newly emitted row cancel to3e-14m³/s. The expected old
wake comes from independent polygon-segment integration, not a duplicate of
the emitter's spanwise formula. These plus three existing bound/root checks
passed5/5; /tmp/openonda-vlm-deformed-wake-budget-tests.xml. Production solver
evolution was not changed. This excludes a basic shedding-topology explanation
under prescribed deformation, but does not fix or qualify the full coupled run.

The isolated rotor completed its eleventh revolution with CT=.0043086181 and
CP=.0002572242, about5.9% lower than the prior cycle for both. Thrust coefficient
standard deviation is.000669740 (15.54% of the mean). Thus extending the horizon
has exposed worsening load fluctuations, not established a steady solution.
Native health uses accepted-state curl(u) in both immutable runtimes and the
current source; it does not depend on the stale backup-vorticity array. The
twelve-revolution continuation, turbine and delta runs remain active.

## 9 September, 07:05 UTC: near-wake health and relaxation impulse imbalance

At the isolated rotor's native ten-revolution checkpoint, f64 direct Gaussian
particle-pair Jacobians plus the native bound-field Jacobian reproduce the native
512-probe misalignment:53.2896deg versus FMM53.3024deg. Close to the rotor plane
(within half a rotor radius) the weighted angle is65.14deg; particles younger
than one revolution average60.28deg. This is neither an FMM approximation artifact
nor a defect confined to old wake particles. The diagnostic only read checkpoint
data; it did not advance or overwrite the simulation. Details and probe limitations
are in rotor-checkpoint10-misalignment-localization.json in the audit workspace.

Native impulse/load comparison gives wake-impulse thrust within1% of blade thrust
for the unrelaxed case over cycles2–6. The ordinary relaxed case instead gives
ratios1.255,1.856,1.999,2.428,2.304 over those cycles. The refined relaxed case also
fails this comparison, while the unrelaxed refinement agrees. Changes in integrated
bound-polyline impulse at the saved2,4,6revolution checkpoints account for at most
.00218N, far below the mismatch. This is a diagnostic of the relaxation/coupling
budget, not final certification of the complete unsteady force identity.

Prepared one targeted `relaxed_moments` study using the existing native option
to preserve vector strength and impulse moments during P-relaxation. All physical
inputs match the ordinary relaxed case except that flag and the12revolution horizon;
geometry is byte-identical. Outputs are separate and no simulation has started.
The direct command is part of the studies/allrun launcher. After the current
continuation finishes, run this candidate from the actual tutorial studies folder
and compare native loads, impulse and wake health. No numerical operator was
modified and no qualified-default claim is made. The checklist has launch and
plotting follow-up instructions.

## 9 September, 07:23 UTC: native-sample wake-health figure

The isolated-study allplot now includes a wake-health figure with native
misalignment, normalized divergence and revolution-integrated wake/blade impulse
ratio. It joins the two existing continuations without altering samples. Only
complete sampled intervals are used for the impulse comparison, with trapezoidal
integration of dense blade-force samples. The README states the wake-only
limitation. Actual PNG and PDF were rendered and inspected with the shared
Pagella/NewPX fonts and12.5cm width; tutorial-style checks pass. The figure makes
both the unrelaxed late degradation and ordinary relaxation's impulse mismatch
visible in the tutorial's reproducible outputs.

The current full-quad validator checks the existence of its native velocity-plane
collections, but lacks a temporal field-convergence test. This needs strengthening
before full-quad certification, alongside the newly motivated impulse/load check.
The full-quad candidate has not been run or claimed qualified.

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


## User-revised validation scope, 9 September

The current priority is flat-plate physics and the unexplained integrated
vector-strength budget, followed by delta-wing combined motion and force sanity,
then deep assessment of the one wind-turbine rotor at its authored operating
point, and finally a complete quadcopter demonstration. The isolated quadcopter
rotor studies are closed historical diagnostics, not prerequisites for the
demonstration and not additional wind-turbine tutorials. Their native outputs
and reproducible case scripts remain under the tutorial. The installed public
catalog no longer presents these studies as a separate tutorial.

The unlaunched raw-moment permanent candidate was removed. Its four-step
temporary probe is retained as startup-only evidence and is not pursued further.
Neither active wind-turbine nor delta-wing simulation was restarted or altered.
The continuation heartbeat now follows the new sequence. Current actions and
evidence are in [the revised checklist](2026-09-vlm-todos.md); the old checklist
was preserved in the history file to avoid losing diagnostic findings.


## 9 September: bound/free stretching incompatibility isolated

A read-only instantaneous budget at the actual completed 8-degree checkpoint
shows that geometric endpoint shedding and transposed bound-to-particle
stretching balance different reverse interactions. Even with matched kernels,
the remainder is integral curl(u_w) cross (Gamma ds) on the bound lattice,
measured as +0.5044335840 m³/s² for Gaussian filtering. The actual mixed kernels
give +0.4424096413. Independent endpoint integration and reciprocal pair
identities close to about 6e-13. Smoothed wake-core overlap dominates, with a
nonzero correction because the split wake vorticity is not independently
solenoidal. This is not explained by a missing root filament or only a kernel
mismatch. No production numerical change has been adopted. The derivation,
measurements and limits are in [the budget review](2026-09-vlm-circulation-budget.md).
A consistent local exchange formulation remains to be implemented and validated.


## User-directed stop of later cases

The user clarified that the entire workflow must be strictly sequential, with no
later-case simulation while flat-plate issues remain. Exact PIDs80399/88579 and
their turbine/delta working directories were verified, both received SIGINT, and
both exited. Native finalization preserved metadata and partial tutorial output.
Turbine final recorded state:1570/t9.42s, last restart1536. Delta:736/t1.84s, last
restart400. Their native `failed` status reflects KeyboardInterrupt at the user's
request, not a numerical failure. No simulation from this task remains running.
The heartbeat and working checklist now prohibit restarting later cases until
their predecessors pass. The flat plate is the exclusive current focus.
