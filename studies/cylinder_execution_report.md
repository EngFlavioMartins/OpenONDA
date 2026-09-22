# Cylinder execution evidence

This report separates software checks and short measurements from production
convergence. Full cylinder runs remain the user's production workload. Neither
a short pilot nor three mesh names establishes grid independence.

## Final corrected short cohort

All five corrected CPU/four-rank cases reached the same transient time
**t=0.4D/U** with hxy=dz=.08D, span=.96D and three interface sweeps. They
share one [numerical source fingerprint and measured record](cylinder_final_short_cohort.json).
Whole wall time includes mesh construction and startup; the baseline's later
3.3-second completed-run check is excluded from its 300.1-second initial cost.

| Case | Realized hp/D | Whole wall [s] | Final interval [s] | Particles | Cd at t=.4 | Converged steps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline, exchange dt=.04 | .080 | 300.1 | 16.56 | 4,228 | 1.5382 | 2/10 |
| Exchange dt=.02 | .080 | 418.1 | 16.51 | 4,887 | 1.7642 | 17/20 |
| Exchange dt=.08 | .080 | 239.7 | 18.92 | 3,660 | 1.4584 | 0/5 |
| Particle-spacing ratio 1.25 | .096 | 264.5 | 13.36 | 2,745 | 1.5366 | 4/10 |
| Particle-spacing ratio 1.50 | .120 | 242.0 | 10.67 | 1,592 | 1.5860 | 5/10 |

The spacing ratios quantize hp to divide the slab. Fixed ratios make the
physical core/blend/release widths (.08/.48/.16)D at baseline,
(.096/.576/.192)D at ratio 1.25, and (.12/.72/.24)D at ratio 1.50. The spacing
comparison therefore changes those widths as well as particle count. Endpoint
lift coefficients were -.00525, -.00593, -.00121, -.02064 and -.02532 in table
order; the differences from baseline are absolute, since baseline lift is near
zero. At x/D=1 the endpoint velocity-vector RMS differences from baseline were
.00965, .00177, .00191 and .00477 U in variant order. Peak sampled
process-tree RSS was 5.34–5.36 GiB. [Raw cohort journals and manifests](../tutorials/coupled_fvm_vpm/01_cylinder_shedding_flow/study_results/execution_pilots/final-short-cohort/)
retain the force, profile, wall-projection and interface budgets.

These are transient sensitivity measurements, not stationary force, shedding
frequency or grid-convergence qualifications. In particular, fewer particles
and shorter late intervals at larger hp do not establish physical accuracy or
completion of the full 100D/U run. A separate [real native-restart smoke](cylinder_native_restart_journal_smoke.json)
copied the final baseline, injected future diagnostic and force/profile rows,
then replayed from its step-6 checkpoint through step 10. It exited cleanly,
left exactly one record per accepted step and no future samples, and archived
11 superseded streams. The measured cohort was untouched.

## Machine observed during execution

- Intel i7-12700H, 14 physical cores, 20 logical CPUs; about 14 GiB RAM.
- Vulkan outside the sandbox exposes Intel Iris Xe (ADL GT2), Mesa 26.0.8.
  The NVIDIA device listed by PCI is not exposed by the current Vulkan drivers.
- The sandbox exposes only llvmpipe, a CPU software renderer. Its Vulkan timings
  must not be interpreted as GPU measurements. Hardware tests require access to
  the actual render device.
- `env -u DISPLAY vulkaninfo --summary` works without an X connection. The VPM
  memory-budget probe now also omits DISPLAY in its child environment; the
  caller's environment is unchanged.

## Measured optimization

Fixed renewal velocity traces now retain their donor indices, weights and
wall factors. Each interface sweep still consumes the current donor velocity
and gradient. Rebuilding transfer geometry invalidates these plans.

[Raw trace timing](cylinder_trace_benchmark.json): 448 manufactured donors,
54,872 renewal nodes, 39,304 queried nodes, seven alternating repeats. Warm
median trace time improved by **2.61 times**, with exactly equal output. This
is a component benchmark with box geometry, not an end-to-end cylinder speedup.
The recorded first-call costs share a process/JIT cache and must not be used
as an independent cold-start comparison.

The interpolation, tied-donor, stable-renewal and body-centred-lattice tests
passed together (64 tests). The updated prepared-trace checks cover fresh
provisional fields, caller mutation, transient-cache eviction and exact
agreement with the previous circulation formula. Interface/reporting checks
also pass (five tests).

The interface diagnostics now distinguish state capture, restore, boundary
refresh and accepted output; capture time is included in the aggregate timing.
Donor gathering is exposed for the last sweep. These host timings complement
whole-process wall time; they do not replace GPU synchronization measurements.

## Qualification status

The implementation now includes shared curved-solid exclusion, full 3D slip-slab
induction, matched case factories, campaign/resume handling and statistical
reports. Production qualification remains open. In particular, the first
real coupled GPU pilot passed three accepted geometry-checked intervals but
failed the runtime budget; the following measurements are not a runtime
certificate. The common-span pipeline and its native restart now pass. The
paired CPU restart is substantially faster, and CPU is the campaign default;
the finest proposed mesh has completed bounded four- and six-rank CPU pilots.
Four ranks remains the measured startup choice. Neither has sufficient measured
margin to certify a complete developed-wake trajectory within twelve hours.

## Reference mesh and rank pilots

[Raw numerical summary](cylinder_machine_pilots.json). Journals, console logs
and the exact pilot scripts are retained under the tutorial's ignored
`study_results/execution_pilots/` directory. Runs were sequential, with one
BLAS/Numba thread per MPI rank. These are single trials, not confidence intervals
or a controlled thermal/affinity sweep.

| h/D | Span/D | Cells | MPI ranks | Late median step [s] | Aggregate peak RSS [GiB] |
| --- | --- | ---: | ---: | ---: | ---: |
| .08 | .96 | 42,624 | 2 | .440 | 1.19 |
| .08 | .96 | 42,624 | 4 | .312 | 2.14 |
| .08 | .96 | 42,624 | 6 | .244 | 3.02 |
| .08 | .96 | 42,624 | 8 | .481 | 3.93 |
| .0565685 | .96 | 114,104 | 6 | .708 | 3.83 |

The rank trials reach t=1; the finer pilot reaches only t=.16. Six ranks is
the measured reference choice. The first complete cold h=.08 construction took
180 s before solving; later rank trials reused its identical native mesh. The
finer pilot re-extruded the verified historical XY mesh section to 17 uniform
layers and therefore does **not** measure a fresh source-mesh build.

The finest tested reference appears feasible for a longer qualification run,
but developed-wake timing, repeated trials and full wall time remain necessary
before certifying the twelve-hour limit. Coupled cost cannot be inferred from
these standalone results: it includes provisional FVM sweeps and reflected VPM
induction. The launch pipeline applies a twelve-hour wall limit to each case
and terminates its MPI process group on timeout; that limit is not an accuracy
or completion guarantee.

## Statistical reporting

Force statistics interpolate exact window endpoints, reject missing coverage,
and estimate periods using subsample shedding peaks. At least ten complete
periods are required. Period blocks supply uncertainty estimates and a drift
check; individual adjacent CSV samples are not counted as independent trials.
Richardson/GCI rejects divergent, zero-order and statistically unresolved mesh
differences. Reports always distinguish force-grid qualification from the
remaining temporal, domain, span and velocity-profile gates. Six manufactured
postprocessing tests pass, including invalid-GCI and incomplete-window cases.

## Historical coupled integration and performance gate

The first two-rank Intel Vulkan pilot uses h=.08, span=.48, hp=.08,
8,368 global FVM cells and capacity 100,000. It exercises all three velocity
and vorticity components with the small deterministic initial perturbation.
The accepted intervals at t=.04/.08/.12 took approximately 174.6/349.1/250.6
seconds. The final interval contained 1,389 renewed particles, and an intact
coupled checkpoint was written. [Raw interval data](cylinder_coupled_initial_pilot.json). These timings belong to the original per-shell image dispatch,
not the subsequently batched implementation. All exceed the 17.28-second
per-interval allowance implied by 2,500 intervals in twelve hours.

The dominant measured cost was image induction in the VPM predictor and
repeated boundary updates (the latter is included in transfer timing).
Donor gathering was about a millisecond in the first interval, so optimizing
MPI gathers before the image computation would not address this bottleneck.
Image queries now batch complete doubling blocks and reuse one fixed-source
FMM tree, retaining the existing velocity/gradient tail criterion. Numerical
agreement tests pass; new runtime evidence is required before selecting the
backend and target-buffer capacity.

The real pilot also found shallow RK/accepted-step cylinder crossings. The
verified analytic cylinder now projects only bounded shallow penetrations
back onto its fluid side, keeps circulation unchanged, and reports affected
strength, displacement and impulse change. Deep crossings and unsupported
general-wall crossings remain errors. This is a numerical exclusion inside
the FVM-owned wall region, not a physical wall-vorticity boundary model.

The current batched implementation also completed the reference/coupled pilot
pipeline on the common .96D span at h=.1, hp=.12 and four coupled MPI ranks.
Its first three accepted intervals took 130.6, 131.1 and 100.8 seconds. A native
restart loaded step 3, accepted step 4 at t=.16 and wrote the next canonical
checkpoint; that interval took 162.3 seconds including cold restart/JIT work.
[Measured interval diagnostics](cylinder_common_span_pilot.json) retain the
image-tail, wall-exclusion and interface residual evidence. All four startup
intervals exhausted the three permitted interface sweeps without convergence;
this is explicitly reported and is not accepted as stationary accuracy evidence.

The same saved step-3 state was continued on CPU with four owner threads and
four MPI ranks. Its first interval took 34.89 seconds including compilation;
the next warm interval took **6.452 seconds** (VPM 1.075, boundary .603, FVM
2.764, transfer 2.011). The matched step-4 result has 1,195 particles on both
backends; raw boundary flux differs by 7.97e-9, maximum outflow mismatch by
2.96e-8 and wall-remap correction L1 by 6.69e-8. These are diagnostic comparisons,
not full-field norms: CPU did not save a step-4 field checkpoint.

This CPU run exposed and verified a portability fix: slip-slab GBD now fixes
its lattice origin on CPU as well as GPU. An adaptive CPU origin previously
made the otherwise valid slip-plane/grid alignment fail after restart. The
CPU workspace may still grow; only the required lattice phase is fixed.

The warm CPU result passes the startup per-interval budget. It does not predict
the cost of a fully developed wake with more particles. The GPU/CPU times above
are single observations, with different cold/warm conditions, so no overall
speedup factor or twelve-hour completion certificate is inferred from them.

Independent near-wall field tests hold the Gaussian core and total strength
fixed while refining the remap. At spacings .1/.05/.025, normalized induced
velocity errors fall from .02067 to .01238 to .003159 at one lattice phase,
and .01952 to .005781 to .001150 at a translated phase. Circulation and first
moment residuals remain below 2.4e-9. This validates that remapping component;
it does not establish a physical no-slip boundary model or complete wake accuracy.

## Prepared campaign and interpretation

### Finest default mesh: rank comparison before final renewal parity correction

The final common-domain h=.064 case has **36,555 cells and 114,052 faces**,
hp=.08 and dz=.064 on span .96. Both CPU trials completed three intervals and
wrote native checkpoints. See [the preserved measurements](cylinder_finest_cpu_pilot.json).
These measurements precede the final Gaussian renewal parity correction
described below; they are retained for diagnosis, not as final numerical
qualification of the corrected method.

| Coupled ranks / owner threads | Cold interval [s] | Warm intervals [s] | Whole invocation [s] | Peak process-tree RSS [GiB] |
| --- | ---: | --- | ---: | ---: |
| 4 | 47.290 | 16.312, 17.031 | 225.39 | 5.54 |
| 6 | 51.393 | 16.384, 17.366 | 248.68 | 6.45 |

Six ranks reduced FVM time but increased owner-side VPM/transfer work. Both
trials produced 1,921/2,146/2,316 particles at the three accepted states.
Four ranks therefore remains the default; six is not an overall optimization.
At exchange dt=.04, the entire twelve-hour allowance is 17.28 seconds per
interval before startup. These startup timings are too close to that ceiling
to claim a robust runtime prediction. Memory also exceeds the plan's initial
5 GiB incremental target; production monitoring remains necessary.

The first four-rank interval projected 24 RK-stage and 24 accepted particles
out of the cylinder. Maximum penetrations were .01482/.01532D, below the
explicit .25hp=.02D rejection bound. Accepted projection changed impulse by
approximately (-2.03e-4, -6.72e-5, -4.48e-8). These are material diagnostics,
not an assertion of negligible wall error. All three startup intervals still
exhausted the interface-sweep limit; stationary acceptance requires convergence.

### Finest mesh after Gaussian/halo corrections, before sparse wall-image correction

This pilot predates the subsequent sparse wall-image correction and is historical execution/timing evidence. The [fresh four-rank CPU pilot](cylinder_finest_cpu_parity_fixed_pilot.json) completed three intervals and a native checkpoint
on the same 36,555-cell mesh. Cold/warm interval times were **48.093,
16.418 and 17.078 seconds**, with peak sampled process-tree RSS
**5,872,181,248 bytes (5.47 GiB)**. Transfer particle counts were
1,933 / 2,147 / 2,324. No solid/slab validity check failed.

The third interface solve still reached its three-sweep limit (normal residual
1.90e-5, gradient residual 2.17e-5), and the largest accepted projection
x-impulse change was approximately 3.61e-4 at step 2. These startup states
do not qualify stationary coupling accuracy. The 17.078-second warm interval
also leaves almost no allowance for developed-wake cost growth within the
17.28-second gross budget; a full twelve-hour completion remains unproven.

### Remaining performance opportunities

On the four-rank finest third interval, VPM/boundary/FVM/transfer consumed
2.016/1.516/9.117/4.383 seconds. FVM and transfer now deserve profiling ahead
of unmeasured optimization. The longer pre-wall-image-fix screen resolves that transfer
cost: at baseline interval 9, transfer took 4.647 seconds, of which boundary
refresh took 4.435. Capture, restore and the last donor gather were only
.0026/.0060/.0021 seconds. Preallocated rollback and fewer gathers therefore
have little measured upside here. Induced boundary-field evaluation is the
meaningful target, alongside the FVM solve; the aggregate transfer label alone
was misleading. Removing interface sweeps or relaxing tolerances is not the
solution.
The follow-up code audit confirms that velocity and gradient already share
one FMM traversal, and the physical source tree is reused across image batches.
There is no duplicate velocity/gradient tree build to remove for this Gaussian
cylinder configuration. Further substantial improvement would need a validated
faster evaluation of the slab image field, such as aggregating distant image
groups, rather than claiming another cache fixes the measured bottleneck.
The repeated near-threshold interface residuals also motivate safeguarded
fixed-point acceleration of the boundary trace. That would need flux-consistent
velocity/gradient updates, a fallback on residual growth and matched-state
tests before adoption; it is an algorithmic candidate, not an implemented
shortcut or a reason to weaken the existing convergence gate.

The default launcher uses h=.10/.08/.064 on the common .96 span, with
reference and coupled grid comparisons, matched off-midspan profiles and
PNG figures. These are initial candidates, not an accepted grid family.
The final coupled XY box is **[-1.6,1.6]D in both directions**. Its three
default spacings fit that box exactly; the transfer/authority box remains
[-1.25,1.25]D. Earlier pilot timings above used the historical requested
±1.48D box and must not be mistaken for timings of this final geometry.

The finest-mesh pilot exposed two distinct geometry errors before solving:
Cartesian quantization moved the physical slip planes to ±.512D, and extrusion
then tried to move the resolved outer XY boundary inward from ±1.536D to
±1.48D. The latter folded 92 section edges. The original section was valid,
and varying the slice plane did not cure the boundary projection. Generated
variants now use exact span levels and the source mesher's resolved XY bounds;
the common ±1.6D default eliminates that XY variation across the grid family.
Custom grid requests retain their actual bounds in the campaign identity.

The separate sensitivity baseline uses h=hp=.08 so the .48/.96/1.92 spans
retain the same particle resolution. Its eight factors leave interface
iteration limits, error tolerances and vorticity cutoff fixed; changing
exchange dt is explicitly a temporal study, not an independent emission rate.

Each individual process group has a twelve-hour wall limit. A timeout is a
failed case, not a guarantee that its physical horizon will finish. Full
grid plus sensitivity execution may take days. Completed results are only
reused for compatible requests, incomplete reference solves are not silently
restarted, and coupled continuation requires an intact native backup.
The report records resumed execution cost cumulatively, so a cheap completion
check cannot masquerade as a solver speedup.
Resume identity also includes relative Python-source hashes and numerical
dependency versions, without machine paths or third-party binary hashes. Each
worker records sampled aggregate process-tree RSS, including MPI children;
missing solver-journal memory is reported as unavailable instead of zero.

### Mirrored wall correction and endpoint audit

A further wall audit found that the physical body-bounds shortcut omitted
near-wall image particles outside the slab. Recomputing their M4 corrections
with a folded shortcut was also insufficient: an image 2.6 grid spacings beyond
a plane can influence the required diffusion halo while its complete correction
stencil extends beyond the allocated three-cell halo. The implementation now
computes the physical sparse wall correction once and reflects its node indices
and axial-vector values, clipping only deposits outside the active lattice.
An independent recomputation on a larger grid gives identical retained
float32 deposits in the bounded regression. This also removes two repeated
wall-correction solves per remesh; no whole-case speedup is inferred from that
operation count.

Generic and analytic solid masks now reflect ghost coordinates into the
physical slab, including repeated reflections, while physical coordinates
remain unchanged. Mask identity includes the slip planes. The verified cylinder
classifier already extended through z; its original defect was the image
correction bounds shortcut, not classification of cylinder ghost nodes.
Historical pilots above predate this correction and are labeled accordingly.

A node-aligned physical source exactly on a slip plane also produced twice its
normal circulation on zero-viscosity remeshing because its normal image
coincides with it. Regeneration now applies half control-volume strength and
volume at those endpoint nodes. Repeated-remesh checks preserve normal
circulation, while half-node lattices are unchanged. Endpoint-volume handling
is explicitly GBD-owned; unsupported slab diffusion combinations (core
spreading, RWM, DVH and LES) fail at configuration rather than silently using
incompatible physics.

### Restart and runtime accounting

The completed pre-wall-fix exchange screen exposed two extra diagnostic rows
from a timed-out attempt beyond its last committed checkpoint. Native restart
now rewinds the coupler-owned journal as well as the already-rewound FVM/VPM
histories. Superseded streams are archived with a `.superseded` suffix so
recursive force/cost readers see only the active trajectory. Reconciliation
errors are broadcast to all MPI ranks. Incomplete append tails are recovered;
malformed completed records remain errors.

The runner enforces its wall allowance cumulatively across resumed attempts,
including changed compatible end horizons. It validates saved attempt costs
and does not spawn another worker when the case budget is exhausted. Attempt
wall time and archived histories remain available even when their provisional
trajectory is superseded by a checkpoint restart.

## Software verification and production commands

### Final span audit

The final audit found that renewal scattered mirrored M4 strengths but then
removed ghost nodes before applying the Gaussian representation filter.
Constant-zero convolution consequently lost the mirror contribution used by
slab induction. This affected the local strength correction, not just a
diagnostic. The initial exchange-clock screen was stopped and retained as
rejected pre-fix evidence. The corrected representation includes repeated
translated/reflected sources over its full kernel support and excludes temporary
ghost output from physical particles and residual norms. Direct image-sum
tests cover half-node and node-aligned planes, wide cores, repeated images,
and endpoint cancellation/doubling consistent with the induction convention.
These and the existing stable-renewal tests passed together (27 tests).

The audit also found that cloud-bounded GBD extents could omit mirrored
support after a cloud retreats from a slip plane. The default sensitivity
family requires at most two explicit diffusion substeps, but the shared
implementation must maintain the physical slab and its stencil support
independently of where the current cloud happens to lie. It now retains both
planes with a halo based on the explicit substep count. On-node and half-node
two-substep diffusion tests match an independently mirrored, extra-padded
reference at every physical node. Unsupported halo reach or insufficient
configured padding fails explicitly instead of silently dropping images.

The changed geometry, GBD, slab, transfer, checkpoint and launcher paths passed
their focused regressions, including real two-rank MPI completion and
collective rejection without a hang. The final geometry/builder/style checks
passed. The changed Python files passed Ruff; the repository type check still
has 123 baseline errors, not a new clean type-check result. All 84 changed
Python files pass Ruff checks and formatting after the final edits. The final portability/runtime/
campaign/profile group passed 61 tests, and launcher/plot/style checks passed
five. The campaign figure renderer was also exercised with synthetic data;
those figures are rendering checks, not physical results.
The earlier combined numerical suite passed all 93 tests after the Gaussian/halo corrections,
covering represented renewal, solid projection, transfer fields, checkpoints,
case construction, slab induction/diffusion, wall masks and RK-stage handling.
After the sparse wall-image, endpoint and restart fixes, the final targeted
runs passed **92 numerical tests, 52 campaign/restart tests, and 21
campaign/plot/style tests**. Type diagnostic signatures still match all 123
baseline errors exactly. A rebuilt wheel (SHA256
`f1886d8b8e21ecbc2864faf65d7233d154222a077bca5c991f5a240c632c8366`)
passed both current- and minimum-dependency installed verifiers outside the
checkout. Its 567 entries include the campaign helpers and no generated
simulation outputs.
Raw software logs and the final wheel checksum are retained under
`study_results/execution_pilots/software_checks` in the tutorial.

The final parity-corrected wheel was installed outside the checkout into separate current and
minimum-dependency environments (NumPy 1.26.4 / SciPy 1.12). Both passed
`python -m openonda.verify_install --require-site-packages`, including an
actual iterative FVM solve, VPM evolution, HDF5 restart, compiled mesh code,
packaged resources and plotting. These local verification environments inherit the existing conda dependency
installation; the minimum environment overrides NumPy/SciPy. They verify an
installed wheel outside the checkout, not a fresh dependency-resolution run.
Linux/macOS CI uses clean virtual environments; a local Linux run does not
verify macOS hardware. MPI remains an optional external
runtime; serial CPU use works with the base package dependencies.

From `tutorials/coupled_fvm_vpm/01_cylinder_shedding_flow`, in the OpenONDA
environment:

```bash
./allrun.sh
./allplot.sh
```

The first command runs reference and coupled grids followed by the full
17-case sensitivity study (and at most one admissible interaction). For the
grid comparison alone, use `./allrun.sh --sensitivity none`. To continue an
interrupted compatible campaign, use
`./allrun.sh --run-dir <saved-directory> --resume`; an incomplete reference
solve requires a fresh campaign because it has no native restart support.
An inconclusive statistical comparison is saved and returns status 2; it is
not called grid independent. Numerical errors/timeouts retain logs and a
failure status. `allplot.sh` selects the latest campaign explicitly and refuses
to hide an incomplete run behind older successful figures.

Unchecked scientific qualification items in the plan remain open. In
particular, full runtime, developed-wake stationarity, temporal/domain/span
independence and the complete stationary sensitivity study require production
results. The launchers make those outcomes inspectable; they cannot guarantee
them before they are measured.

## Optimization follow-up

The default solver now reduces reflected-image contributions locally per target,
instead of issuing competing atomic writes for every image. The image family,
tail criterion and stretching definitions are unchanged. In the matched f32
CPU component benchmark, the complete 1,000-source/1,000-target slab evaluation
fell from 0.33262 to 0.30183 seconds (9.3%). The isolated accumulation kernel
was 24.2 times faster; that kernel ratio is not an overall solver speedup.
Maximum complete-call differences were 8.94e-8 in velocity and 7.15e-7 in its
gradient, with the same shell count, batches and target evaluations. See
`cylinder_slip_slab_accumulation_2026-09-22.json` and its benchmark script.
Additive accumulation, disabled outputs, all stretching modes, f32/f64 and
target tiling are covered by tests. A separate Intel Iris Xe Vulkan correctness
check also passed; macOS hardware was not exercised locally.

The final matched four-rank comparison uses hxy=hp=dz=0.08, span=0.96,
exchange dt=0.04 and ten intervals through t=0.4. Production tolerances,
cutoffs, physical dimensions and sweep counts were held fixed.

| Variant | Whole invocation | Mean warm interval, steps 4–10 | Final particles | Converged intervals |
| --- | ---: | ---: | ---: | ---: |
| Pre-optimization baseline, quiet repeat | 249.11 s | 12.206 s | 4,228 | 2/10 |
| Default optimizations | 222.73 s | 10.416 s | 4,228 | 2/10 |
| Optimizations with experimental Aitken | 228.65 s | 10.451 s | 4,230 | 10/10 |

The default changes reduced measured mean warm-interval time by 14.7% and
whole-invocation time by 10.6% in this comparison. These are single-run
observations, not a repeatability estimate or a production-runtime forecast.
The initial baseline overlapped verification work and was excluded from the
final comparison; `baseline_repeat` ran without concurrent agent tests or
benchmarks. Startup, caches, thermal state and unrelated host load can still
affect timings. The original pre-optimization package snapshot and all four
raw runs are preserved under
`study_results/execution_pilots/optimization_comparison`.

Against the quiet baseline, default optimizations changed final Cd by 2.21e-8
and Cl by 8.71e-7. Across aligned sampled profiles, maximum absolute differences
were 1.30e-6 in velocity, 1.97e-6 in vorticity and 6.31e-7 in kinematic pressure.
The same 4,228 final particles were retained. Flux and renewal conservation
checks passed. Commands, configurations, source identities, force values,
component costs and sampled-field comparisons are saved in
`cylinder_optimization_comparison_2026-09-22.json`; regenerate the summary with
`python studies/summarize_cylinder_optimizations.py --baseline-name baseline_repeat --output studies/cylinder_optimization_comparison_2026-09-22.json`.

The partitioned PETSc workspace now retains the normalization vectors and
reuses the already-computed current-matrix/initial-guess product. It still
updates the equation and applies the same normalized residual acceptance rules.
An interleaved 40,000-row repeated-solve benchmark reduced median solve-call
wall time by 3.2% on one rank and 1.6% on two ranks; solution errors were at most
4.44e-16. These are synthetic component measurements, recorded with commands
and source hashes in `petsc_partitioned_workspace_reuse_benchmark.json`.

The remaining audit did not justify changing donor gathers, rollback snapshot
allocation, particle copying or output cadence. Recorded donor gathers and
state capture/restore take milliseconds, while repeated boundary induction
takes seconds. Static donor/trace preparation and compatible FMM source reuse
were already implemented. Optional VLM/source-panel GPU round trips do not
occur in this cylinder configuration. Further GBD memory reductions would need
to retain the validated diffusion halos and wall correction; no additional
support truncation or diagnostic removal was used to obtain these gains.

Experimental `interface_acceleration="aitken"` mixes velocity and tangential
gradient with one bounded coefficient, retaining normal-flux consistency.
Both residual components must remain finite without growing, or the last coherent endpoint is
restored, including local FVM fields, patch data, particles and reported
diagnostics. The failed attempt consumes its sweep. The default remains
`"none"`: under the unchanged three-sweep limit, two ordinary residuals are
needed before acceleration can affect the third sweep, so it cannot save a
sweep. Its configuration is included in campaign and restart identities.

In the four-rank, ten-interval comparison through t=0.4, Aitken met both
unchanged interface tolerances in all ten intervals, versus two for ordinary
iteration. All ten proposed accelerated sweeps were accepted; factors ranged
from 0.89794 to 0.97740. Both variants still used three sweeps per interval.
The mean warm interval cost was 10.451 seconds with acceleration and 10.416
without it, so this screen supports a convergence benefit, not a speedup.
Final Cd changed from 1.5381932 to 1.5376291 and Cl from -0.0052469 to
-0.0053943. These differences reflect changed interface iterates and are not
a claim of numerical equivalence. Flux and renewal conservation checks passed
in both runs. Longer-wake qualification is still required before promotion.

The final targeted verification passed 80 tests, including a two-rank
accelerated-rejection rollback test, with the 12 affected interface/PETSc
tests rechecked after typing fixes. Ruff passed. Pyrefly reports 116 existing
errors, down from the pre-change baseline of 123; it is not a clean repository
type check, but the optimization introduces no new diagnostic signatures.

The optimized wheel, SHA256
`14cc31cfcf4fbc452bbf35a8827b64695d80ef53f723202f4745a006bd4e2b4c`,
passed the installed verifier outside the checkout. The verification venv
inherits the conda dependency installation; this is not a fresh dependency
resolution or a macOS hardware check. Wheel, identity and logs are retained in
the tutorial's `study_results/execution_pilots/optimization_final_checks`.

## Numerical interpretation and references

The span images impose planar free-slip symmetry, not a cylinder-wall model.
For reflection R=diag(1,1,-1), velocity transforms as R u and vorticity as
det(R) R omega=diag(-1,-1,1) omega. Repeating both plane reflections generates
the translated image sequence. The implementation checks velocity and gradient
increments over doubling blocks; this is an empirical convergence criterion,
not a rigorous bound on the infinite tail. Direct-sum comparison tests are
therefore retained alongside the criterion.

For background, [MIT's image-vortex derivation](https://web.mit.edu/fluids-modules/www/potential_flows/LecturesHTML/lec1011/node39.html)
shows how planar images enforce no normal flow. [Cottet, Koumoutsakos and Ould Salihi](https://www.sciencedirect.com/science/article/pii/S0021999100965318)
study vortex methods with spatially varying cores and local mappings.
[Eldredge's viscous vortex formulation](https://www.sciencedirect.com/science/article/pii/S0021999106003093)
explicitly supplies wall-vorticity flux to enforce no slip. Here that physical
wall treatment belongs to the FVM solver; the new constrained GBD redistribution
only excludes the solid while retaining supported fluid moments. These sources
provide context, not validation of this implementation's particular wall mask
or its coupled cylinder results.
