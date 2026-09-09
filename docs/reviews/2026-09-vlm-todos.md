# VLM tutorial completion checklist

Updated 9 September 2026. **Work is ongoing; no commit has been made.**

**Current:** direct-induction static12° passed the complete native run and physics
checks. The remaining direct-reference angles now run sequentially in session28419,
driver `/tmp/openonda-flat-direct-v20-sweep.py`, master log
`/tmp/openonda-vlm-direct-v20-flat-sweep.log`. Audit status:
`flat-direct-v20-status.json`. The first case is moving-10°. Never duplicate this
driver. It retains qualified direct static12° and stops at the first failed check.
The preceding tree sweep stopped at its1.10721e-4 strength residual. Complete-step
budgets identify free/free tree approximation as the dominant remainder; no native
solver equation changed. All diagnostic probes and the single-run session17442
have exited. The sweep uses the frozen v20 install and the updated authored setup.
Do not count preceding tree cases as current direct-reference results. That campaign remains in
`flat-v20-sweep-status.json` and `/tmp/openonda-vlm-v20-flat-sweep.log`.
No delta, turbine or quad run may resume yet.

## Pending thesis handoff — explicitly authorized 9 September

- [ ] After the complete sequential VLM qualification and final commit, send one
  consolidated message to **Complete ch05 TODO study**, task
  `01a081a4-a01e-78b0-9dac-60770868f496`, host `local` (thesis project).
  The user explicitly authorizes this message. It has **not been sent**.
- Ask that task to update **Chapter 3** with the established solver formulation,
  correct equations and explanations. The thesis is unpublished: present the
  final formulation directly, without an erratum or a narrative admitting past
  corrections. Translate the algorithm into academic, physics-first prose rather
  than implementation details, following the thesis's existing notation.
- Include the final commit, the initial VLM audit, the circulation and force
  budget documents, relevant primary references, verification evidence and
  remaining model limitations. The initial audit also covers the consistent
  finite bound-field representation and its analytic regularized Jacobian,
  motion timing, physical force evaluation points, reference frames and moments.
  At present the key topics are consistent bound/free vortex-strength exchange
  over accepted time integration stages, the relation between horseshoe
  circulation increments and surface potential jump, and unsteady pressure
  forces with their moments and power. Include further findings from subsequent
  cases only after verification; do not imply universal high-Reynolds validation.
- Record the successful delivery here so future continuations do not send a
  duplicate batch. This handoff is a completion requirement of the current task.

## Strict sequential execution — latest user instruction

The user explicitly rejected running later cases while flat-plate issues remain.
Only work on the current stage. **No tutorial simulations in parallel.** Do not
resume delta wing, turbine or quadcopter work until the flat plate has passed its
physics and numerical checks with no unexplained discrepancies. Complete each
subsequent stage before beginning the next. The earlier exception preserving
active later-case runs has been revoked; both jobs have now been stopped.

## User's revised sequence — authoritative

1. **Flat plate: establish the physics first.** Use the completed runs and native
   samples to check force magnitude, lift distribution, velocity profiles,
   moving/static equivalence and bound/shed circulation. Diagnose the unresolved
   integrated vector-strength drift with a complete budget. Do not call the
   solver validated while this discrepancy is unexplained.
2. **Delta wing: test combined motion and force sanity.** Verify translation and
   rotation composition against the authored motion, inspect the actual sampled
   geometry, and check finite, physically reasonable loads and a stable wake.
   A separate periodic/statistical convergence campaign is not required.
3. **One wind-turbine rotor: the principal rotor validation.** The case is
   `tutorials/vpm/rotor_flow`, at its single authored operating point. Assess Ct,
   Cp, spanwise loading, native velocity planes, momentum balance and startup
   decay against applicable theory. Give this stage the main effort. Extend the
   same run only if the available evidence shows it needs more development.
   Do not create a family of rotor operating conditions.
4. **Quadcopter: demonstrate the solver after 1–3.** Obtain a stable, complete
   run with useful load and wake figures. It is a toy problem, not another rotor
   qualification campaign. Isolated quadcopter-rotor studies are closed and are
   not a prerequisite. Keep their existing scripts/results as historical data.

A passing wind-turbine case will support the modeled regime; it is not universal
proof of separated, viscous or arbitrary high-Reynolds-number flow accuracy.

Preserve the stopped jobs' existing partial results. No production simulation
was running from this task immediately after the stops. The current flat-plate
run is recorded above. Any future permanent run starts from the
actual tutorial directory after its prerequisite stage passes.

## Reproducibility and execution rules

- All new permanent runs start from their actual tutorial directories, one at a
  time. The stopped audit working directories linked output into the physical
  tutorial directories; those files remain preserved.
- Keep user-facing setup files minimal and physics-focused. Direct installed
  `python setup.py` commands and simple shell launchers are the intended interface.
- Use native scientific output in `samples/` and native metadata/logs/sparse
  checkpoints in `solution/`. Plotters read those records. Do not add tutorial
  metadata writers or checkpoint extraction to duplicate native sampling.
- Do not repeat completed simulations just to relocate results. Replotting native
  data is allowed. Keep inputs and output folders reproducible from the case.
- Use Lamb–Oseen's shared Pagella/Palatino + NewPX style, 10.95 pt, maximum
  12.5 cm figure width. Render and visually inspect final PNG/PDF plots.
- At most one Metal simulation. The other VPM task has a four-case Metal suite
  queued after the turbine; preserve its work. For bounded CPU work use
  `TI_CPU_MAX_NUM_THREADS=2` and two BLAS threads. Do not modify active runtimes or
  reinstall shared environments. Verify exact owned PID/cwd before any signal.
- Preserve hundreds of authorized preexisting changes across the repository.
  Review before the eventual requested commit; do not reset unrelated work.

## Stopped later-case simulations

Both exact PIDs and working directories were verified, then sent SIGINT at the
user's explicit request. Both processes exited, their native finalizers ran,
and their stored output remains inside the tutorials. **Both later-case simulations remain stopped.** Native lifecycle status is `failed` because the runtime records
KeyboardInterrupt that way; these two endings are user-requested interruptions,
not demonstrated numerical failures.

| Case | Final recorded state | Preserved restart / evidence |
|---|---|---|
| Wind turbine | step 1570/2400, t=9.42 s, N=211950; PID80399 exited | last checkpoint1536; `tutorials/vpm/rotor_flow/{solution,samples}`; `/tmp/openonda-v16-rotor-full.log` |
| Delta wing | step736/4000, t=1.84 s, N=98190; PID88579 exited | last checkpoint400; `tutorials/vpm/delta_wing/{solution,samples}`; `/tmp/openonda-v18-delta-full-cpu.log` |

Do not claim a restart can resume from the final recorded state: the available
sparse checkpoints are earlier. Do not clean these partial data or restart either
case while flat-plate issues remain. The immutable runtimes and original audit
working directories remain available for provenance. Full quadcopter has not run.

## Flat plate: first unresolved checkpoint

All 20 installed-v17 CPU cases completed; native output/time/geometry/checkpoint
contracts passed. Five final PNG/PDF pairs were regenerated in the actual tutorial
and visually inspected. Moving runs end at 197 steps/t=2.4625 s; static runs at
192 steps/t=2.4 s. Twenty-four chord lengths suffice for the observed load plateau:
maximum late CL range is 0.15238%. These are successful executions, not full
physical qualification.

At 5 degrees, CL=0.4285115263 versus rectangular lifting-line 0.4404155827
(−2.7029%), CD=0.00602541561 versus 0.0067045262 (−10.1291%), and quarter-chord
CM=0.0018871. Sectional lift L2 error is 3.9637%; inboard downwash L2 error is
2.1181%, full-span 25.3703% with the discrepancy concentrated near the tips.
Maximum nonzero CL CPU/Metal difference is 0.009082%. Apply the reference model's
limitations when judging high angles and near-tip velocities.

**All 18 nonzero-angle cases fail the current signed integrated vector-strength
closure tolerance of 1e-4.** Static residuals at 2, 5, 8, 10, 12, 15 degrees are
0.129928%, 0.813993%, 2.088793%, 3.287071%, 4.769032%, 7.554447%; moving values
are nearly identical. The two zero-angle cases trivially pass. Do not weaken or
hide this criterion. The current case validator only checks a subset of the
external theory/profile metrics; bring the decisive checks into native-data
postprocessing as needed, without distracting the setup.

Distinguish circulation Gamma [m²/s], signed integrated vector strength sum(alpha)
[m³/s], and unsigned sum(|alpha|). The last is not a circulation conservation law.
The existing drift plot is a vector-strength budget, not a pointwise comparison of
all historical wake circulation with the present bound circulation.

Completed diagnostics narrow the issue:

- Finite bound strength includes all three on-wing legs and telescopes to the TE
  endpoints. Independent deforming old closing polyline + changing circulation +
  stationary/rotating-body emission tests close to 3e-14 m³/s (5 tests passed).
- At 8 degrees, step 192: external bound stretching dominates the vector-strength
  budget; per-step bound-plus-shed change is about −0.0022521 m³/s and net budget
  change about +0.0039606 m³/s. Evidence: `budget-v6-transposed.json`.
- Replacing external transposed stretching with direct stretching reduced but did
  not eliminate drift and did not converge under common-time refinement. Native
  coarse/medium/fine residuals at t=0.25 are 0.115165/0.157715/0.205599%; external
  direct values are 0.027158/0.039422/0.058339%. This is not a validated fix.
- Total-TE convection, removing bound filtering, overlap alone and Gaussian
  quadrature changes did not resolve the discrepancy. Do not repeat these as new
  hypotheses or apply an arbitrary global strength correction.
- No particle removal, capacity loss, Pedrizzetti relaxation or divergence
  projection occurs in these flat-plate cases. Root cause remains unresolved.

Metrics: `full-v17-qualification-metrics.json`, `flat-v17-cpu-versus-v7-metal.json`
in the audit workspace; `/tmp/openonda-vlm-full-v17-metrics.py` reads native output.
Both case validators currently fail only the unchanged strength closure check.

## New circulation diagnosis: bound/free exchange

Read [the bound/wake budget derivation](2026-09-vlm-circulation-budget.md) before
further changes. A read-only calculation on the actual completed 8-degree CPU
checkpoint isolates the instantaneous inconsistency. Transposed particle
stretching pairs with a reverse transposed interaction, whereas the geometric
shedding stencil balances an endpoint/direct interaction. Their difference is
integral curl(u_w) cross (Gamma ds) on the bound lattice. With matched Gaussian
kernels its y rate is +0.5044335840 m³/s²; the actual Rosenhead/Gaussian mismatch
changes it to +0.4424096413. Pair reciprocity and endpoint quadrature identities
agree to about 6e-13 at order 32. Core overlap contributes +0.5936617537 and the
non-solenoidal correction −0.0892281697 to the matched-Gaussian term.

This is a concrete coupling incompatibility, not a solved defect. It explains
why geometric shedding tests pass while the full particle budget drifts, and
why matching kernels alone previously failed. Next work must derive and test
consistent local bound/free exchange with the boundary condition and force
model; no global wake rescaling or isolated disabling of stretching. No new
simulation or production evolution change was made in this diagnostic.

## Subsequent case evidence

**Delta:** 4000 steps/10 s, dt=0.0025, Gaussian overlap 2.5, no relaxation,
DNS nu=0.001, CPU/FMM, 250k capacity. Forces 100/cycle, fields 25/cycle, backup
one/cycle. The earlier v13 candidate failed at step 995 with strain CFL 1.06;
retain that history. Current three PNG/PDF pairs are native progress plots in the
actual tutorial, with explicit common wake averaging window and completed-cycle
count. All were visually checked at 12.4968 cm width. Verify the authored motion:
integrating opposite-phase sinusoidal velocities from a common initial height
produces different mean heave heights. Do not silently change the active inputs.
When stage 2 is reached, reconcile the strict periodic-force validator with the
user's motion/force-sanity scope; do not mislabel periodic stationarity as demonstrated.
The run is now stopped by request, with native partial data retained.

**Wind turbine:** dt=0.006, 2400 steps/14.4 s, Gaussian overlap 2.5, ordinary
Pedrizzetti 0.3, Tree theta=0.3/order=3, 400k capacity. Forces every 0.012 s,
fields 0.06 s, backup about one revolution. Final-six-revolution wake planes start
near 9.78 s and are not yet available. BEM reference T=2325.5465 N,
Q=1417.4222 Nm, P=11575.6146 W, Ct=0.6838172262, Cp=0.4862515866.
Cycle 10 Ct=0.7135560/Cp=0.5193123; cycle 9→10 drift −0.372%/−0.629%.
The run is now stopped by request; defer its assessment until flat plate and delta
wing pass. Current validation limits: final-six-revolution coefficient drift 2%, BEM 15%,
wake impulse/force 10%, native plane drift 1%. Evaluate their physical basis;
steady loads alone do not establish a correct wake. The native bound/coupled
impulse columns were added after this frozen run; do not fabricate old columns.

**Quadcopter:** not yet run in full. Current strict BEM/drift/wake/impulse gates
were written before the revised scope. For stage 4, retain useful diagnostics as
informational and use completion, finite output, numerical health and meaningful
figures for demonstration acceptance. Do not require an isolated-propeller study.
Select a supported setup informed by stages 1–3, then run from the actual tutorial.

## Closed historical quadcopter-rotor diagnostics

Seven completed variants and one failed variant remain under
`tutorials/vpm/quadcopter/studies/`, with reproducible scripts/native results.
They are no longer separate entries in the installed public tutorial catalog.
Their README identifies this closed history. The seven completed cases are
coarse, time_refined, mesh_refined, relaxed, relaxed_time_refined, continue_8,
continue_12. The relaxed_moments candidate failed at step 3 at its unchanged
strength-growth gate. The numerical projection/conditioning audit already passed;
the rejection is not an arithmetic-conditioning error.

The 12-revolution unrelaxed run completed but did not qualify as stationary:
last-six-revolution mean T/P differed from BEM by −0.96%/−4.69%, but half-window
drift was 6.01%/12.22%, final misalignment 75.08 degrees, and wake impulse/thrust
about 1.97. These findings remain visible; they do not gate the toy demonstration.

Before the revised user request, a four-step temporary raw-moment relaxation
probe completed (no per-particle magnitude renormalization, native moment
preservation and factor 0.3). Magnitude changes were −3.25/−3.97/−2.83% with
moment residuals around 1e-17. It establishes startup admissibility only. Evidence:
`rotor-raw-moment-proposal-audit-lsx0at4v/proposal-audit.json` and
`/tmp/openonda-audit-rotor-raw-moment-proposals.log`. The unlaunched permanent
moment_filter candidate was removed. An empty installer-v19 venv was created,
but no installation or production simulation was launched. Do not pursue this
branch. Existing installed/frozen runtimes remain untouched.

## Verified source work to preserve

See the audit/history for full details: finite three-leg induction/Jacobian;
accepted-state near-wake AIC/emission; local TE motion/root merging; force/power
frames; native VLM sampling in samples/; restart v4; optional common core overlap;
Tree mixed-core/self-pair fixes; Fourier memory reduction; safe restart capacity
extension; elapsed/progress reporting; native bound and coupled impulse diagnostics.

Recent evidence: full v16 regression 435 passed; v17 metadata 93 passed; optional
overlap/restart 32 passed; restart/capacity 45 passed; bound/coupled impulse 5 passed;
complete native output contracts 34 passed; nine quad native wake/style checks
passed. The last diagnostic additions do not change v18 particle evolution
(`v18-current-nondoc-vpm-changes.json`). Do not treat diagnostic improvements as a
fix for the flat-plate vector-strength discrepancy.

## Finish and report

Follow stages 1–4 above. Keep case scripts/cadences, native outputs and current
figures aligned. Report completed executions separately from reliable physics,
explain any remaining circulation issue, and state the scope of solver validation.
Then perform appropriate final regression/installer checks when environments are
free, inspect changes and preserved data, and git commit all authorized progress
with a clear purpose. No commit or push has occurred yet.

## Durable records

- [Solver audit](2026-09-vlm-audit.md).
- [Detailed prior history and superseded checklist](2026-09-vlm-history.md).
- [Earlier global tutorial qualification](2026-09-tutorial-qualification.md):
  52/66 bounded passes, 10 numerical failures and 4 meshing timeouts; not full
  successful execution of every tutorial.
- Audit workspace: `/var/folders/kw/njsv3xwj69qf8p4bp4jw035h0000gn/T/openonda-vlm-audit-0fyxi5zd`,
  also stored in `/tmp/openonda-active-vlm-audit`.
- Heartbeat `complete-vlm-tutorial-qualification`, every ten minutes, now updated
  to this sequence. Quiet on unchanged state; substantive progress between checks.

Revised installed tutorial catalog/materialization verification: 21 tests passed in 47.784 s; no failures or errors. `/tmp/openonda-vlm-revised-catalog-tests.xml`. The historical studies retain their reproducible launcher but are excluded from public tutorial discovery. Source diff whitespace check passed.

Read-only reciprocal-budget audit completed, session89586 exit0; curl split
session35723 exit0. Orders16/32/64 completed for both kernels; 32→64 differences
in the reported curl cross bound rate are below1.3e-12. Native snapshot was
opened read-only; no simulation advanced. Reproduction scripts and JSON retained
in the audit workspace. All source/docs whitespace checks passed. Those process observations predate the explicit stop request above; both owned
processes subsequently exited. No production solver setting, runtime, or physical qualification threshold changed.

## Provisional local exchange repair — current work

A disposable 32-step static 8° prototype completed without editing or replacing
tutorial results. Peak signed strength closure 4.37490146e-6 (unchanged gate
1e-4); final -4.03586032e-6. At t=0.4 s, CL=0.6459336016, CD=0.01745393214,
CMc4=0.00427582312; changes from baseline at that time are -0.0155%, -0.1165%,
-0.0423%. This is promising startup evidence, not physics qualification.

The candidate accumulates the opposite bound-induced particle stretching rate
per spanwise strip with the exact accepted RK weights. The resulting transported
old bound vector replaces the geometrically transported closing vector in both
newborn deposition and the implicit AIC constant term. Existing free particles
are unchanged by the source operation. Check impulse, loading, velocity, motion,
and refinement before accepting this model; zero net strength alone is insufficient.

Native implementation now in progress in the checkout: fused bound induction
and strip reaction (no second particle/panel traversal), transactional RK provider
contexts, vector-valued transverse emission, compatible near-wake RHS, and restart
coupling version5. Existing v4 checkpoints remain preserved, but cannot claim
exact continuation across this numerical change. No later tutorial is running.

Prototype: `flat-local-exchange-nxjora9p` in the audit workspace, including native
samples, checkpoint, `local-exchange-rates.json`, and `probe-metrics.json`. The
reproduction script is `/tmp/openonda-flat-local-exchange-probe.py`. Current focused
regression log: `/tmp/openonda-vlm-native-exchange-existing.log`.

Native exchange regression: all55 tests in `test_vlm_bound_exchange.py`,
`test_vlm_coupled_restart.py`, and `test_vlm_qualification.py` passed. This covers
independent per-strip filtered-line quadrature for all three stretching forms,
RK2/SSPRK3/RK4 actual-increment closure and disabled stretching, failure rollback,
nonparallel transverse emission with unchanged scalar circulation, accepted
rotating-body no-penetration at two geometric scales/two kernels, exact restart
and Galilean equivalence, plus existing filament/force/loading regressions.
Report: `/tmp/openonda-vlm-native-exchange-physics.xml`. The first implementation
run found a misplaced kernel signature argument; it was fixed before this passing
run. Its18 generic RK/provider tests had passed, and its13 coupled failures were
all the signature error. Source/style/whitespace checks pass.

Next: install the candidate into isolated `installer-v19`, verify imports outside
the checkout, then run only static8° from the actual flat-plate tutorial. Preserve
its preceding native results for comparison. No later case will be launched.

Before production use, the fused accumulator was refined to sum each RK stage's
strip reaction in a zeroed field, then add it once to the larger transported
bound vector. This avoids losing tiny f32 terms against the old bound magnitude
and keeps one particle/panel traversal. A4096-particle precision regression now
checks the actual RK increment in f32 and f64; both passed. The fourteen local
exchange tests also pass after this change. Coupled restart/motion tests are
being repeated because the numerical accumulation changed. The initial unused
v19 installation precedes this refinement and will be refreshed before any run.

The other VPM task has claimed the free Metal slot for its official four-case
battery. Preserve its installed main environment. This task uses CPU/two threads
and the isolated candidate environment; do not send cross-task messages without
explicit authorization.

After stage-wise precision accumulation, all30 focused local-exchange/precision/
coupled restart-motion regressions passed (zero skips). Report:
`/tmp/openonda-vlm-exchange-precision.xml`. Isolated v19 was refreshed before its
first simulation; all six changed native source files match checkout hashes.
The frozen patch and hashes are retained as `native-local-exchange.patch` and
`native-local-exchange-source-hashes.json` in the audit workspace.

### Current single production run

Static8° is now running from `tutorials/vpm/flat_plate` through ordinary installed
`python setup.py --mode static --angle 8`, using the isolated v19 runtime on CPU
with two threads. Session/monitor log: `/tmp/openonda-vlm-v19-flat8-full.log`.
The outside-checkout isolated installation verifier passed before launch. The
previous56 native files from this case were moved intact under the audit
workspace's `flat-before-native-exchange/{solution,samples}` for comparison,
with a SHA256 manifest; no existing simulation was rerun just to relocate data.
New results write directly to the canonical tutorial solution/samples folders.
Delta wing and rotor runs remain stopped; no other tutorial from this task runs.
Do not start another simulation until this one completes and its budget/force/
velocity/impulse measurements are assessed.

At the in-progress static8° native sample step135/t1.6875, the maximum full-vector
closure was3.87654e-5 (y3.68653e-5). Over t1.1875–1.6875, -d(coupled impulse)/dt
agreed with sampled lift/drag at ratios1.0012757/0.9930780. CL=.682427; the
last-five-chord load range still0.4501%, so this interim window was not accepted
as the final plateau. These are partial observations, not final qualification.
An audit-only calculation initially used removed NumPy `trapz`; switching to
SciPy's compatible `trapezoid` corrected the analysis, with no simulation change.

### Completed native v19 static8° — not the whole flat suite

Session92305 exited0 after192 steps/t2.4; native metadata says completed and
N10752. Last scheduled force/field sample is190/t2.375, as prescribed.
Peak y/full-vector closure3.67861e-5/3.86821e-5, below unchanged1e-4.
Final-five-chord CL range0.13422% is below the existing0.2% plateau criterion.
CL=.6839129706, CD=.01530621591, CMc4=.00295341444; against rectangular
lifting-line: CL -2.94494%, CD -10.82158%. These finite-chord/viscous-wake
reference differences predate the repair; do not pretend exact theory agreement.
Physical span-width-weighted sectional-lift L2=3.4843% over the span (2.6151%
inboard80%); downwash L2=22.9945% overall and4.71994% inboard80%. The tip
difference still requires resolution/reference-model assessment.

Native coupled impulse over t1.875–2.375 yields lift/KJ ratio1.00085775 and
drag/KJ ratio0.98508732. Force histories finite; max sampled strain increment
.05106596. Audit report `flat-v19-static8-full-metrics.json` and reusable native
CSV reader `flat-native-metrics.py` are retained outside user-facing cases.
The new budget PNG/PDF are being regenerated in the canonical tutorial figures
folder, using the shared thesis fonts and12cm width. Other19 flat cases still
contain the preceding v17 results and have NOT been rerun with this repair.
No later tutorial has resumed. Before the sweep, assess representative time/grid
refinement and complete the key5° force/velocity comparison under the new model.

Budget figure PNG/PDF were regenerated and visually inspected. The first export
placed an annotation over the now-small residual curve; moving it to empty lower
space fixed the overlap. The PDF is340.157pt wide (12cm), with embedded TeX Gyre
Pagella/NewPX fonts verified by `pdffonts`. No font or width exception.

A bounded stationary-lattice diagnostic has now completed sequentially after the
full8° run, from the actual flat tutorial directory. At5°, it compared8×14,8×28,
16×28 and16×56 panel counts per half-wing (224→1792 total), with the tutorial's
geometric spacing and an inviscid fixed horseshoe wake. This separates lattice/
reference-model effects from VPM wake transport. Results:
`flat-steady-lattice-refinement.json`; script `flat-steady-refinement.py` retained
in the audit workspace. No tutorial outputs were modified by this diagnostic.
Assess these results before choosing a coupled refinement.

Stationary5° lattice refinement gives CL .427430→.424376→.424395→.422809,
CD .00587812→.00588019→.00588114→.00587949. Full-span downwash relativeL2
.25833→.25457→.25172→.24731; inboard about4–5%. Thus the discrepancy is also
present without VPM and is not eliminated by this lattice refinement. This does
NOT prove the force or reference model correct. A useful additional check of
current coupled8° native strip Gamma uses well-conditioned odd Fourier fits:
12→14 terms change predicted far-wake drag .0160235→.0160310, fitted CL .68405
(close to sampled .683913), whereas native CD=.0153062. The remaining~4.5%
drag difference deserves an explicit near-/far-wake assessment, not dismissal
as generic theory error. Fourier far-wake drag assumes an idealized planar
wake and is not an exact measurement of the rolled-up viscous free wake.

Current bounded diagnostic: `/tmp/openonda-flat-steady-trefftz.py`, log
`/tmp/openonda-flat-steady-trefftz.log`, running sequential stationary lattices
from the actual flat tutorial cwd. It retains strip Gamma arrays and Fourier
conditioning/truncation evidence in `flat-steady-lattice-trefftz.json`. No
production time-marching simulation or later case has been launched. Examine
these results before proceeding.

The stationary Trefftz diagnostic completed, session20300 exit0. Its near-field
CD versus circulation-series CD gap decreases with span refinement: -5.144%
(14 strips/half), -2.922% (28), -2.918% (28 with doubled chord resolution),
-1.682% (56). Fourier fits remain well conditioned and12→16-term drag changes
are small. This is evidence for a spanwise discretization contribution to the
near/far discrepancy; it does not establish a coupled free-wake refinement.
Native near-field CD itself changes less than0.06% across these grids. The
remaining lifting-line reference difference should be judged with the actual
finite-wing model and resolved loading, not corrected by scaling force output.

**Next bounded work:** perform sequential coupled time/span refinement at a
common early physical time, using the completed v19 static8° native samples as
the coarse reference where clocks match. Assess strength closure, forces,
impulse and velocity together. Then complete the native5° force/profile cases
and the rest of the20-case flat sweep only after that assessment. Keep all
permanent outputs/case launches in their real tutorial folders. No delta, rotor
or quad simulation may start. At this checkpoint all owned processes (including
stationary diagnostics and plotters) have exited; no production simulation runs.

Tooling note: unqualified `python` in `exec_command` resolves the project Conda
environment at the repository root but not when its cwd is a tutorial subfolder.
The user's zshrc already activates OpenONDA. Use the explicit isolated interpreter
for tool launches; do not add Python-discovery boilerplate to tutorial launchers
or change the shared installed environment while the other VPM battery runs.
Taichi emitted cache-lock warnings for the protected shared home cache; numerical
runs exited0. Do not clean that shared cache or interrupt the other task.

## Coupled refinement continuation

Verified no existing owned setup/refinement process before launch. Only
`time_half` is now running from the real flat tutorial cwd, v19 CPU/two threads,
dt=.00625,8×14 panels/half,60 steps/t.375. It uses the authored tutorial case
constructor with the refinement controls overridden in a process-local audit
harness; no solver algorithm is patched. Native outputs go directly to
`tutorials/vpm/flat_plate/{solution,samples}/qualification/exchange_v19/time_half`.
Its generated input surface is retained in the audit workspace and embedded by
the native metadata writer. Main geometry and coarse full-run samples are intact.
Sampler interval=.03125 (five refined steps), one end checkpoint. Log:
`/tmp/openonda-flat-coupled-time-half.log`. Reproduction script
`flat-coupled-refinement.py` is retained in the audit workspace.
After completion, compare against the existing coarse step30/t.375 sample, then
run `span_half` sequentially (same dt,8×28 panels/half) only after assessment.
No later tutorial has resumed.

Time-half completed60 steps/t.375, N3360, session37382 exit0 in28.6s. At the
common time, coarse/time-half CL .642301934/.636719856 (-0.869%); CD
.0176039996/.0175933477 (-0.0605%); CMc4 .004450427/.005172692.
Peak full-vector closure4.80012e-6/1.85289e-6, both below1e-4. All forces finite.
The comparison is during startup, not a steady-theory pass. Native sample clocks
differ at the first output; use common impulse start.0625 when comparing budgets.

To establish time convergence rather than one difference, `time_quarter` is now
running sequentially: dt=.003125,120 steps/t.375, same8×14 lattice. Its native
outputs are in the adjacent `qualification/exchange_v19/time_quarter` folders;
log `/tmp/openonda-flat-coupled-time-quarter.log`, session70678. After assessing
this result, proceed to the single span-half check. No concurrent simulation
from this task and no later case have been launched.

Quarter-step and span-half runs both completed sequentially, sessions70678 and
51437 exited0. Both reach t.375 with6720 particles (120/60 steps respectively);
span-half took59.4s. At quarter dt CL=.634918676, CD=.0175759545, CMc4=.005434097;
time-half→quarter CL change contracts to0.283% from coarse→half0.869%. Drag
shifts remain below0.16% overall but do not demonstrate a monotone order.
Span-half CL=.631315321, CD=.0174443075, CMc4=.005707223; compared to time-half
at the same dt, CL differs0.849% and drag0.847%. Full-vector closure peaks for
coarse/time-half/time-quarter/span-half are4.8001e-6/1.8529e-6/1.8061e-5/3.6687e-5;
all below the unchanged1e-4 limit. Main case files were not overwritten.

**New force audit finding, before additional full runs:** the current force path
is explicitly quasi-steady Kutta–Joukowski. Its startup force/impulse mismatch
includes the omitted unsteady pressure term from changing surface potential
jump. Independently integrating that term from the native chordwise Gamma
samples closes most of the mismatch: over common t.0625–.375, lift/impulse
ratios after adding the estimated term are1.00761/1.00394/1.00254/1.00387 for
coarse/time-half/time-quarter/span-half. Drag ratios are.98384/1.01089/1.01563/
1.02103. This is an audit calculation; the native force outputs have NOT yet
been changed and the field evolution is untouched. Native data and audit plot
are in the actual tutorial. New analysis JSON: `flat-unsteady-pressure-force-audit.json`;
script `flat-unsteady-force-audit.py` retained in the audit workspace.

Next: finish the unsteady-force audit and implement the native pressure-time
contribution with correct spatial moments before more full flat runs. Also
assess whether trailing on-wing leg forces are material; some actuator-line
models deliberately omit them, so do not add them merely to fit theory. See
`2026-09-vlm-force-budget.md` for the mathematical mapping and implementation
requirements. Do not modify frozen v19 or the shared main environment. No delta,
wind-turbine or quad run is authorized to resume before flat qualification.

## Native unsteady-pressure implementation (candidate v20)

The native pressure-time force and exact triangle-centroid moments are implemented
in `solver/unsteady.py`. Each physical panel is split at its bound line; the
fore/aft portions carry upstream/cumulative potential jump respectively. A new
`ForceConfig.kutta_joukowski(unsteady=True)` option selects this contribution.
Pure KJ remains the default. The flat setup selects the new option and records
forces/distributions every step, with geometry/flow samples every5 and backups40.
Native total force samples include the separate unsteady components; pressure
jumps include the pressure-time contribution. Surface torque/power and total
reference/quarter-chord moments include the intrinsic panel moment. Conservation
CSV retains true KJ force and reports the pressure term separately. Restart
format6 stores the new load fields. Existing v19 outputs remain untouched.

Analytical tests check independent whole downstream pressure patches against the
physical-panel partition, mirrored signs, negative circulation, rigid rotations
and shifts, reference/quarter-chord moments, density scaling, downstream loading,
power at the pressure centroid, constant-circulation no-op and invalid timesteps.
The first six tests pass in the active regression batch (session24739, log
`/tmp/openonda-vlm-unsteady-physics.log`); coupled/restart/qualification tests are
still running. A later edit extends the Galilean regression to pressure forces
and moments and requires a focused rerun after this batch finishes.

The v19 installation differs from current VPM source in exactly the nine owned
unsteady-force files; `openonda/` is identical. This permits a clean comparison
of wake evolution. Candidate installer-v20 is being created separately. Do not
update the shared main environment or frozen v19. No tutorial simulation is
currently running; do not start a bounded v20 case until the regression batch
passes. No later tutorial may run while flat qualification remains open.

### v20 tests and first coupled force budget

- Solver/force/analytical/restart batch:50 passed,0 failures,226.7s;
  `/tmp/openonda-vlm-unsteady-physics.xml`.
- Output contracts, extended unsteady Galilean invariance and native-metadata
  reader batch:37 passed,0 failures; `/tmp/openonda-vlm-unsteady-output.xml`.
- Installed v20 into a separate system-site-packages venv, without modifying
  the shared main environment. Verified imports from `/tmp` and all177 VPM/public
  Python source hashes; report `native-unsteady-v20-source-hashes.json`.
- Bounded time-half run completed60 steps/t.375,N3360,31.6s,exit0. Native outputs:
  `{solution,samples}/qualification/unsteady_v20/time_half`. Full-vector strength
  closure1.8529e-6, below the unchanged1e-4 limit.
- Over common t.0625–.375, native integrated pressure impulse is5.805235055 N s
  along z; independently integrated surface potential jump gives5.805234806 N s.
  With interval weights for the backward pressure derivative and trapezoidal KJ,
  surface/fluid impulse discrepancy is -0.3285% in lift and -1.0018% in drag.
  Over the complete sampled t.03125–.375 window both are about -0.42%. These
  are finite discretization residuals, not exact conservation claims.
- Compared native final checkpoints against v19 time-half: position maximum
  change4.77e-7 m, circulation2.61e-7 m²/s, strengths7.15e-7 m³/s. Only the nine
  force/reporting source files differ; no material wake change, but f32 threaded
  reductions are not bitwise deterministic.
- Native plot `assets/plot_plate_impulse.py` now reproduces the force/impulse
  comparison directly from force and flow-integral CSV samples. It is added to
  `allplot.sh`; no checkpoint extraction or metadata writer. Bounded PNG visually
  checked, matching the shared font/style. PDF export also requested.

The bounded report and source are `flat-v20-native-force-budget.json` and
`openonda-v20-native-force-report.py` in the audit workspace. The installed v20
source is frozen. Full static8° now runs192 steps from the real tutorial directory
with its authored setup; all current outputs go to the normal exp_static_aoa08
folders. Check force plateau, strength closure and the impulse figure before
advancing to the5° static/moving flat cases. The old19 remaining cases are still
pre-repair results. Do not treat the mixed sweep as qualified.

### Full v20 static8° completed; static5° underway

Static8° completed192 steps/t2.4, N10752,212.2s, session3923 exit0. Native force
CSV now reaches the final step192 (the separate flow/VTK sampler last runs190).
CL=.684006331, CD=.015313790, CMc4=.002932722. Final5-chord CL variation.1163%
(<.2%). Peak full-vector closure3.93366e-5 (<1e-4), yclosure3.70966e-5. All forces
finite. Last5-chord native fluid-impulse force agrees within.066% in lift and
1.61% in drag; full integrated sampled-window errors -.134% lift/+.639% drag.
LL comparison: CL-2.932%, CD-10.777%; physical span-weighted sectional CL L2
3.47% overall/2.60% inboard80%; downwash23.0% overall/4.72% inboard80%. Retain
the documented finite-chord/tip limitations of lifting-line comparison.

Read-only final-checkpoint force quadrature completed, session70263 exit0; no
time advance. On-wing trailing legs contribute.125662 N vertically (about.037%
of vertical force, .23% of wind-axis drag), too small to explain a large missing
load. Do not replace native midpoint loads with the nonconvergent bound-line
quadrature of the unrefined, nearly singular strip representation. Details/raw
values and limits are in the force-budget document and
`flat-v20-on-wing-force-quadrature.json`; probe script retained in audit workspace.

The static8° impulse and strength PNGs were visually inspected. Matching PDFs
were exported. The bounded native impulse PDF is354.24pt (12.497cm) wide with
embedded TeX Gyre Pagella, matching the shared theme; native strength figure is
12cm wide. Only the two current8° figures were refreshed; other comparison
figures wait for current5° static/moving data.

Now running the authored `python setup.py --mode static --angle 5` sequentially
from the actual flat tutorial directory, frozen v20 CPU/two threads. Output
uses normal `solution/exp_static_aoa05` and `samples/exp_static_aoa05`. After it
completes, assess physical metrics before moving5°. No delta, turbine or quad
run has resumed. All test/plot/diagnostic processes mentioned above have exited
apart from the final PDF plot session31145, which should be closed on polling.

### Static5° passed; moving5° running

Static5° completed192 steps/t2.4,N10752,213.6s,session19365 exit0.
CL=.428650111, CD=.006013640, CMc4=.001874743. Final5-chord CL range.1089%
(<.2%), vector closure4.00187e-5 (<1e-4), yclosure3.88204e-5. LL errors
-2.671%CL/-10.305%CD; inboard80% sectional CL L2 2.378%, downwash4.527%.
Late fluid-impulse ratios1.000260 lift/.983584 drag. All forces finite, maximum
logged strain increment.0302344. Metrics saved in `flat-v20-static5-full-metrics.json`.

Moving5° now runs its authored setup alone from the tutorial, frozen v20 CPU/two
threads. Preserve its old data in the qualification/pre_v20 folder. Assess its
completed native force/strength histories, then compare both frames' steady
loads and profiles before proceeding with other flat angles.

The frame-history plotter now emits two figures from unchanged native samples:
`plate_startup` (first2 chord lengths, including the true finite-step impulsive
load) and `plate_staticvsmoving` (2–24 chords). This keeps the initial pressure
peak from compressing the later convergence comparison. No hidden smoothing
or clipped first-step point. Wait for moving5° to finish before rendering and
visually checking these plus the span/velocity figures. The validator now
requires both figures. Native force/impulse plotting remains an additional
normal allplot command. Style/reader tests:3 passed; Ruff/diff checks pass.

Avoiding unnecessary new simulations: the recorded old/new circulation fields
in completed v19 refinement checkpoints independently determine the native
unsteady force at t.375. Whole downstream-patch integration predicts total
CL/CD .64891657/.01930748 (time-half), .64734353/.01932215 (time-quarter),
.64326365/.01912354 (span-half). The time-half prediction agrees with actual
v20 .64891665/.01930746 to f32 levels. Refining dt changes total CL by.242%;
span refinement at fixed dt changes it.871%. This is an audit prediction from
native history, not a rewritten native dataset or another simulation. Report:
`flat-unsteady-refinement-from-native-history.json` in the audit workspace.

### Paired5° qualification and remaining flat-angle sweep

Moving5° completed197 steps/t2.4625,N11032,231.4s,session71216 exit0.
CL=.428653717, CD=.006013515, CMc4=.001874811. Final5-chord CL range.1132%
(<.2%), vector closure7.28990e-5 (<1e-4), yclosure7.10665e-5. Late fluid-impulse
ratios1.000289 lift/.982746 drag. All force histories finite. Final static/moving
CL and CD match within.003%. Inboard80% sectional CL L2≈2.378% and downwash
L2≈4.52% for both frames. Metrics: `flat-v20-moving5-full-metrics.json`.

Current5° startup/wake-development, spanwise loading and velocity figures were
rendered in PNG and PDF and visually checked. The reference line now has
headroom; legends do not cover the later force curves. Moving profiles use
hollow circles around the static squares so coincident curves remain visible.
The velocity reference is extended to both actual tips; do not truncate the
lifting-line curve and hide the largest finite-chord/tip differences. The final
full-span velocity rerender is session39872 and should be closed/inspected.
PDF width354.24pt=12.497cm, with embedded Pagella and NewPX families.

The external qualification driver reads the unchanged `allrun.sh` commands,
runs each one with the frozen v20 interpreter in the actual flat tutorial cwd,
and immediately checks native completion/clock/sample count, finite forces,
24-chord travel, last5-chord CL variation≤.2%, and y/full-vector strength closure
≤1e-4. The three current qualified cases are checked and retained, not rerun.
Each preceding dataset is moved to the tutorial's qualification/pre_v20 folder
just before its replacement run. The driver preserves both old and new data
and stops at the first error. No Python/bootstrap logic was added to launchers.

Driver: `/tmp/openonda-flat-v20-sweep.py` (also retained in audit workspace).
Session5609; master log `/tmp/openonda-vlm-v20-flat-sweep.log`; per-case log
`/tmp/openonda-vlm-v20-<case>.log`. Status is atomically written only in the
external audit workspace, not as tutorial metadata. Do not start a second driver
while it runs. Inspect `flat-v20-sweep-status.json` for current/completed/stopped
state. After completion, review all-angle polar/sign/symmetry and frame agreement,
run allplot PNG/PDF and the complete flat validator, inspect all figures, and
resolve any failure before moving to delta wing. Shared main installation still
awaits final safe update; do not alter it while another task is using it.

Sweep first replacement passed: moving-10°,197 steps, CL=-.8528963013,
CD=.0238111782, CL tail range.10724%, yclosure7.3002e-5 and vectorclosure
7.5414e-5. It completed before the driver started moving-5°. The scheduler
prompt now reads the driver status and force-budget document before doing any
work; no duplicate runner or later tutorial may be launched.

Final full-span velocity PNG/PDF rerender (session39872) exited0 and was visually
inspected. The entire lifting-line tip behavior is now visible, including its
difference from finite-chord VLM near the tips. All plotter sessions have exited.
The only owned active work is the sequential flat sweep, session5609. Latest
observed case is moving-5° after moving-10° passed; use its status JSON for live
state rather than this historical observation. No commit has been made.

### Sequential sweep and reproducible cross-case checks, 9 September

Eleven current cases are complete: moving -10,-5,-2,0,2,5,8,10,12 degrees and
static5,8 degrees. Moving15° is active in the existing session5609 driver; the
remaining static cases follow one at a time. No later tutorial has resumed.
Read the status JSON for newer state. No current case has failed its driver
checks; maximum observed full-vector closure is7.5414e-5 (<1e-4).

The tutorial validator now retains the audit's native full-vector strength check
as well as its every-step y budget, checks solver completion and the recorded
force cadence (missing/duplicate steps), and rejects preceding results without
the authored unsteady force model. It also checks settled CL/CD/CM reflection
parity, zero-incidence loads and moving/static agreement. These use0.2% relative
tolerance plus1e-8 absolute near zero; the native lift-plateau and circulation
criteria are unchanged. No gates/configuration were added to setup or launchers.
Nine reader/validation regressions passed, including wrong lift/drag/moment
reflection signs, a frame bias preserving symmetry, zero-angle spurious drag,
and vector drift outside the spanwise component. Log:
`/tmp/openonda-vlm-flat-validation-tests.log`. Ruff and diff checks passed.

Read-only current-sweep report:
`flat-v20-polar-progress.json` in the audit workspace. Reproduce with retained
`openonda-flat-sweep-report.py` using the checkout on PYTHONPATH. It reads only
completed current cases listed by the driver, never mixed preceding results,
and writes its report outside the tutorial. The available reflection/frame
comparisons pass. The5° and8° settled frame CL/CD differ by at most.0066%; CM
differs by.0512% and.1317%. Moving +/-2,+/-5,+/-10 reflection differences are
below.009% for all three coefficients. Zero incidence gives exactly zero loads
and strength, rather than a denominator-scaled nonzero result.

The report also integrates native pressure loads with backward-interval weights
and KJ loads trapezoidally, comparing them against density-scaled native coupled
fluid impulse. Across these eleven cases, whole sampled-window lift discrepancy
is at most.301%, drag at most.871%; late-window drag differs by up to1.758%.
These are finite-resolution measurements, not exact identities. Continue these
checks through the full sweep. The late CM range is larger in relative terms
(up to1.74% of the small pitching coefficient), so do not describe the CL
plateau criterion as a0.2% bound on every load coefficient.

The polar plot now uses the same hollow moving circles/smaller static squares
as the profile figures so matching frames remain visible. It still awaits the
full current sweep before the final allplot PNG/PDF rendering and visual check.
The circulation-budget document now explicitly separates the preceding
checkpoint's diagnosis from the current implementation and records the retained
checkpoint location, avoiding attribution of old rates to the new canonical data.
No native solver/runtime change was made in this continuation. No commit or
thesis handoff has been made; both await the complete sequential qualification.

Moving15° subsequently passed at197 steps: CL=1.268751831, CD=.052730667,
lift tail range.1283%, yclosure9.47964e-5, vectorclosure9.52431e-5. This is
below the unchanged1e-4 criterion but closer than the lower angles; retain the
actual margin rather than rounding it into an exact-conservation claim. All ten
moving angles are now complete. Static-10° is running in the same driver, with
no overlapping tutorial simulation. The progress report has been refreshed for
all twelve completed cases and its available polar checks still pass.

### Validator integration and native diagnostic health, 9 September

Seventeen current cases are complete; static10°,12°,15° remain, with static10°
running sequentially. The native validator was exercised against the real mixed
directory while the sweep continued: all sixteen then-completed current cases
passed; only the active static2° and preceding static10°,12°,15° were rejected.
Static2° has since completed and passed the driver. No physical criterion failed.

The validator initially raised a KeyError when it reached a preceding sample
without the new vector-budget fields. It now reports missing fields as a failed
budget and skips older force models after explicitly rejecting them. It does not
fabricate or extract replacement samples. Nine reader/validation regressions
passed again, now including missing native vector-budget fields; the real-data
pre-plot invocation exits1 only for the four then-pending cases. Evidence:
`/tmp/openonda-vlm-flat-preplot-progress.log` and
`/tmp/openonda-vlm-flat-validation-tests.log`. Ruff/diff checks pass.

Across the seventeen completed native flow-integral histories, every numeric
diagnostic is finite, energy and enstrophy are nonnegative, and particle counts
are nondecreasing. Maximum logged strain increment/CFL is0.095805. Report:
`flat-v20-native-integral-health.json` in the audit workspace. These are native
diagnostic checks, not a substitute for spatial flow validation. The retained
`openonda-flat-sweep-report.py` now includes these health measurements so its
final full-sweep refresh covers them without advancing simulations. No native
solver, immutable runtime or tutorial setup changed in this continuation.

Static10° subsequently passed at192 steps: CL=.8529217141, CD=.0238200192,
CL tail range.11065%, full-vector closure5.53492e-5. Static12° is now active,
then static15° is the final remaining command. The eighteen-case polar/impulse
report has been refreshed; the available frame/reflection checks still pass.

### Static12° strength failure — current blocking physics check

The sequential driver stopped after static12° completed192 steps/t2.4,N10752.
Native lifecycle is `completed`; this is a qualification failure, not a failed
simulation. Maximum y residual magnitude is.0056705449 m³/s against maximum
bound strength51.2146530 m³/s: normalized1.10721e-4, above1e-4. The residual
changes sign over the history (+.00259 around steps60–70, -.00563 at190), so do
not assume simple monotonic roundoff accumulation without a rate budget.
Native force and flow records and sparse checkpoints remain in their canonical
tutorial directories. Static15° still has its preceding results; no current run
was started. All later tutorials remain stopped.

First diagnosis is read-only evaluation at the existing static12° checkpoint160.
`/tmp/openonda-flat-strength-rate-probe.py` compares native free/free stretching
at tree theta.1,.05,.025 and order2 with direct f32/f64 evaluations of the same
state. Source fields must remain unchanged; no time step is taken. It runs with
the frozen v20 install, CPU/two threads, from the actual flat tutorial directory.
Its native constructor and temporary scratch outputs are outside the tutorial.
Session79241; log `/tmp/openonda-vlm-static12-rate-probe.log`; report target
`flat-v20-static12-rate-probe-160.json` in the audit workspace. Inspect before
launching another probe. Next, close an actual accepted-step budget if needed,
separating free/free approximation from bound exchange, RK rounding and emission.
Do not loosen the criterion, rescale the wake or rerun for a lucky numerical pass.

### Free/free approximation identified; direct static12° passed

All diagnostic sessions are closed. Native read-only rate evaluations at the
retained step160 state and two complete disposable accepted-step budgets are
documented in `2026-09-vlm-circulation-budget.md`. Free/free tree truncation
dominates the observed positive drift at40→41 and negative drift at160→161.
Bound exchange cancels its actual stage rate to rounding accuracy. Direct f64
free/free rate sums to machine precision on the same state. This is an accuracy
choice for the small flat reference case, not a new source correction or a
weakened conservation criterion.

The first diagnostic order2 row was invalid because only an existing evaluator
attribute had been changed; the allocated tree remained order1. The corrected
probe uses a fresh order2 PhysicsEngine and completed successfully. An initial
step-budget wrapper also counted the pre-RK diagnostic RHS call; the successful
version explicitly records only calls inside the native RK advance. Both audit
helpers are retained with their final reports; neither changed production source
or original tutorial files. The earlier initial rate report is retained with an
`-initial` suffix and must not be used for the order2 comparison.

The authored flat setup now uses `vpm.DirectInduction(stretching_scheme="transposed")`.
That is the only case-physics/numerical change; native solver source and the frozen
v20 runtime remain unchanged. The validator now rejects preceding tree results
so a mixed-backend directory cannot be certified as the current reference.
The failed tree static12° result was moved intact into both roots'
`qualification/pre_direct_v20/exp_static_aoa12`; the direct run writes into the
normal canonical folders and is reproducible with `python setup.py --mode static --angle 12`.

Direct static12° completed192 steps/t2.4,N10752 in282.4s, session17442 exit0.
Every requested force row is present. CL=1.020440869, CD=.0341096465,
CMc4=.004168679. Last5-chord CL range.11716% (<.2%); peak y closure8.95186e-7
and full-vector closure4.91637e-6 (<1e-4). Native integral diagnostics are finite,
energy/enstrophy nonnegative and populated core radii positive. Maximum strain/CFL
increment.0795494. Direct versus preceding tree final CL changes-.000121%, CD
+.02906%, CM-.35244%; the same native force formulation is used in both.

Proper interval-integrated total surface load versus native fluid impulse differs
by-.19437% lift/-.13511% drag over t.0625–2.375; late t1.875–2.375 differences
are-.12173% lift/+.41726% drag. Loading/velocity profiles retain the expected
finite-chord/tip differences from lifting-line: inboard80% CL L2=3.061%, downwash
L2=5.137%; full-span CL3.959%, downwash23.771%. Final CL/CD differ from the
small-incidence lifting-line reference by-3.459%/-11.674%; direct induction does
not remove the physical/discretization limits of that comparison.

Metrics: `flat-direct-v20-static12-full-metrics.json`,
`flat-tree-v20-static12-full-metrics.json`, and
`flat-direct-v20-polar-progress.json` in the audit workspace.
Native-data aggregate script: `/tmp/openonda-flat-direct-sweep-report.py`
(also retained there). The remaining19 direct cases now run sequentially in
session28419 via `/tmp/openonda-flat-direct-v20-sweep.py`, with master log
`/tmp/openonda-vlm-direct-v20-flat-sweep.log` and atomic audit status
`flat-direct-v20-status.json`. It checks/retains static12° without rerunning it,
preserves preceding datasets under `qualification/pre_direct_v20`, and stops on
the first failed native completion, finite-load, plateau or strength check.
Do not duplicate the driver or resume any later tutorial. Once it finishes,
refresh the aggregate report, check full frame/reflection and impulse results,
run allplot PNG/PDF and the full native validator, and visually inspect every
figure. The current figures still await the complete direct-reference data.
No final installation, commit or thesis handoff has been performed.

### Direct-reference progress check, 9 September

Current direct cases static12°, moving-10°,-5°,-2°,0° passed the driver's
completion, finite-force, plateau and strength gates. Moving+2° is active; all
later tutorials remain stopped. The aggregate native report has been refreshed
for these five cases. Maximum full-vector closure is1.01108e-5 (<1e-4), and all
numeric integral diagnostics are finite. Zero incidence gives exactly zero load
and strength. Direct moving-5° has whole sampled-window impulse discrepancies
of.206% lift/1.006% drag; its late drag discrepancy is1.861%, consistent with the
documented finite-resolution force budget. Do not generalize the smaller
static12° impulse errors to all cases or claim exact force identity.

The force-budget document now explicitly distinguishes the ongoing direct
reference from preceding tree results. No solver, setup or run control changed
in this check. Session28419 remains the sole owned simulation driver; inspect
its atomic status before any future launch.
