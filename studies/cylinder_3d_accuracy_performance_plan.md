# Cylinder Re = 150: 3D accuracy, geometry, and runtime plan

Status: software fixes and bounded execution checks completed; full production
qualification and the 12-hour target remain open. See cylinder_execution_report.md for measured evidence.
The additional optimization pass is complete; see
[its checked execution list](cylinder_remaining_optimizations.md) and
[matched short-run evidence](cylinder_optimization_comparison_2026-09-22.json).
Created: 2026-09-21. Update checkboxes only when the linked evidence exists.

## Objective and decisions

Deliver comparable coupled FVM–VPM and fully meshed FVM cylinder cases with a resolved 3D span, performant curved-body handling, and a measured choice of mesh/dimensions whose finest production run fits within 12 hours on this laptop in parallel. Optimize the coupling algorithm after correctness fixes, then measure the accuracy–cost sensitivity of renewal/injection and other consequential numerical controls.

- [x] Confirm with the user: preserve **Re = 150** and resolve/test the span. Do not change Reynolds number to manufacture a 3D instability.
- [x] Inspect both case definitions, mesh-family tests, coupling/GBD geometry paths, and local hardware.
- [x] Obtain two bounded, read-only audits from smaller agents: geometry/GBD and coupling performance.
- [x] Add shared case parameters, unique resumable campaign outputs, manifests, bounded runners, and runtime contracts. Evidence: [campaign runtime tests](../tests/coupler/test_cylinder_campaign_runtime.py), [reference-case tests](../tests/tutorials/test_reference_cases.py), and [cylinder grid tests](../tests/coupler/test_cylinder_grid_campaign.py).
- [x] Record machine/diff provenance and bounded timing evidence. Evidence: [execution report](cylinder_execution_report.md) and [trace benchmark](cylinder_trace_benchmark.json).
- [ ] Agree on the final physical comparison through numerical evidence: common span, compatible span conditions, force normalization, observation locations, time horizon, and statistics window.
- [ ] Complete fixes, short qualifications, measured sizing, optimizations, and sensitivity reporting in the order below.

A 3D discretization at Re = 150 need not develop a self-sustained spanwise instability. For an ideal unconfined cylinder wake, Barkley & Henderson report the first 3D instability near Re = 188.5; that is context, not a validation target for this case. Use a small, controlled 3D perturbation and measure its response rather than forcing a planar solution or expecting finite spanwise fluctuations. [Primary paper](https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/abs/threedimensional-floquet-stability-analysis-of-the-wake-of-a-circular-cylinder/61575FBF0BC45054592D46382DEF30BB).

The 12-hour limit applies **per finest complete production case**, including cold startup, meshing, solver initialization, accepted steps, sampling, and checkpoints. The total sensitivity campaign is a separate cost and must be reported. Preserve the physical observation horizon and convergence tolerances when pursuing performance improvements.

## Confirmed starting evidence (historical; see execution report for updates)

| Item | State observed during planning | Consequence |
| --- | --- | --- |
| CPU | Intel i7-12700H; 14 physical cores / 20 logical CPUs | Benchmark MPI rank count and affinity; 20 ranks is not an automatic optimum on this hybrid CPU. |
| Memory | About 14 GiB RAM; about 7.8 GiB available and 3.2 GiB swap occupied at inspection | Establish an idle baseline and incremental working-set budget. Include all ranks, root gathers, host/device staging, and shared GPU memory. |
| Graphics | PCI lists Intel Iris Xe and NVIDIA RTX 3060 Mobile | Availability to Taichi is unverified. `vulkaninfo --summary` failed to connect to X in this tool environment; no conclusion about GPU usability follows. `nvidia-smi` was not on PATH. |
| Coupled FVM | h = 0.04D; box [-1.48, 1.48] × [-1.48, 1.48]; span 0.96D; 24 uniform z layers | This is already geometrically 3D and does not stretch z. The small outer box and span treatment still need qualification. |
| Reference FVM | x/D = [-8, 24], y/D = [-10, 10]; resolved span 0.96D; uniform z levels; XY refinement | The matched-span family is implemented; complete force-grid and profile qualification remain open. |
| Times | Coupled: 100 s, FVM dt = .004, VPM dt = .04. Reference: 80 s, adaptive dt ≤ .01 | Current force records do not establish an identical comparison. Separate temporal errors from spatial/transfer errors. |
| Body geometry | Renewal uses `TriangulatedWall`; GBD receives a body mask only for `_body_bounds` boxes | A cylinder can be excluded at renewal yet have invalid intermediate particles after GBD. |
| Existing GBD API | `configure_body_cylinder` already exists | Reuse and qualify this capability; the missing geometry connection is a shared solver problem. |
| VPM support | Free-space 3D domain [-5, 15] × [-5, 5] × [-6.6, 6.6], h = .05 | A large z box alone does not enforce the FVM slip-span physics. |
| Diffusion allocation | 411 × 211 × 275 ≈ 23.85 million nodes, ~728 MiB for four GBD fields at current settings | Particle capacity is not a diffusion-memory bound. FMM, particles, temporary arrays, and MPI memory are additional. |
| Reference selection | Campaign writes `grid_h*` plus explicit `reference_selection.json` | The comparison reader rejects missing or unqualified selection metadata. |

Sources: [coupled setup](../tutorials/coupled_fvm_vpm/01_cylinder_shedding_flow/setup.py), [reference setup](../tutorials/coupled_fvm_vpm/01_cylinder_shedding_flow/reference_flow/setup.py), [coupler initialization](../source/coupler/solver.py), [transfer geometry](../source/coupler/vorticity_transfer.py), [wall classifier](../source/coupler/geometry.py), [GBD](../source/solvers/vpm/physics/diffusion/grid.py), [reference-selection reader](../tutorials/coupled_fvm_vpm/01_cylinder_shedding_flow/assets/postprocess.py).

The planning pass did not run new cylinder solves. Execution subsequently measured
matched-span reference meshes/ranks and coupled GPU pilots; consult the execution report. Existing ignored reference outputs were inspected separately; their historical configuration must be checked before reuse. Mesh sizes below are candidates, not certified runtime predictions.

### Historical reference measurements available locally

| Case | Final cells from `samples/<case>/grid_run.json` | Logged progress | Sum of per-step maximum-rank times | Median / final-500 median step time |
| --- | ---: | --- | ---: | ---: |
| grid_h008 | 14,208 | 10,654 steps; t = 80 | 0.372 h | 0.123 / 0.117 s |
| grid_h00565685 | 33,560 | 10,654 steps; t = 80 | 3.114 h | 1.055 / 1.297 s |
| grid_h004 | Completion metadata absent | 66 steps; t = 0.424 | 0.042 h | 1.989 s median; insufficient developed-wake data |

All three logs report six MPI ranks. These sums are solver-step cost, **not full wall time**; they exclude some startup/meshing costs. Use final solver cell counts, not intermediate Cartesian source-mesh counts. Sources are the local `reference_flow/solution/<case>/performance.jsonl` files and corresponding `samples/<case>/grid_run.json`. Existing h=.04 history is too short to certify a full fine run. Increasing the span from .25D to .96D changes the problem size substantially, so the .0566 run's 3.114 h is not a runtime certificate for the proposed matched-span family.

## 1. Establish reproducible case and measurement infrastructure

- [x] Add case-construction functions and a bounded study driver using the installed package API. Share validated physical and mesh-family parameters between the reference and coupled runs; keep numerical owners explicit.
- [ ] Record source revision plus dirty-diff hash, resolved configuration, geometry hash, Python/package versions, actual GPU/device/driver, MPI implementation, rank/thread counts, affinity, memory, and output paths for every trial.
- [x] Write trials to unique study directories. Do not invoke cleaning launchers or overwrite existing tutorial results. Preserve the user's current uncommitted solver changes.
- [ ] Record measured mesh counts, realized wall spacing and dz, aspect ratio/skewness/nonorthogonality, partition imbalance, mesh-generation time, peak resident memory, and output bytes.
- [ ] Extend existing coupling timers to distinguish GBD, FMM build/evaluation, device transfers, FVM substeps, global gathers, donor interpolation, renewal, solid classification, interface state capture/restore, diagnostics, and I/O. Synchronize only where needed for trustworthy GPU timing; also measure uninstrumented end-to-end elapsed time.
- [x] Audit existing reference logs/configuration and seed the benchmark report with the verified historical table above. Include cold versus developed-wake cost and explain superlinear changes before extrapolating.
- [x] Use maximum rank elapsed time and whole-process wall time. The existing sum of four phase timers is not sufficient for a 12-hour estimate because reporting, checkpointing, and waits may be outside those timers.

Evidence: machine manifest, mesh manifest, per-step JSONL timing/memory records, and a repeatable benchmark command.

## 2. Fix solid geometry consistently before tuning

- [x] Introduce one geometry contract, owned by the configured physical body/FVM mesh and consumed by renewal, interpolation, GBD, and diagnostics. Include geometry identity in restart/configuration compatibility.
- [x] Wire the existing analytic cylinder mask into coupled GBD when a circular, axis-aligned cylinder is explicitly declared or verified against the authoritative wall surface within a recorded geometric tolerance. Do not infer a cylinder from an arbitrary bounding box or patch name.
- [x] For general triangulated stationary walls, rasterize/cache the solid mask on the GBD lattice with accelerated geometry queries. Key it by geometry revision, lattice origin/spacing, shape and span model. Rebuild only on a changed lattice/body, not every step; keep device masks resident.
- [x] Keep a single, precision-aware interior/surface convention across CPU and GPU. Do not silently truncate an infinite/extruded body at the edge of a clipped FVM wall mesh or invent endcaps; body extent and span boundary treatment must agree.
- [x] Apply the geometry policy through the whole particle lifecycle: donor sampling, scatter, diffusion stencil, regeneration, advection/RK intermediate states, renewal, and restart. Diagnose crossing particles before they can contaminate induction or a boundary trace. Evidence: geometry projection, GBD mask, stage-RHS and coupled-backup tests plus the native cylinder restart; shallow cylinder corrections remain explicitly measured.
- [x] Prevent diffusion flux through solid faces and regeneration at solid nodes. Treat the existing masked zero-flux GBD stencil as a numerical exclusion boundary; it is not a replacement for physical no-slip wall-vorticity production, which remains FVM-owned.
- [x] Audit M4 weights and Gaussian representation near a curved wall. Signed remeshing weights mean naive positive renormalization can change moments. Prefer bounded local fluid-support redistribution with moment constraints and independent field-error checks; use a documented fallback or reject under-resolved support when constraints cannot be met. Evidence: `test_gbd_body_mask.py` and the two-phase independent field-refinement test `test_gbd_wall_field_refinement.py`.
- [ ] Account separately for physical FVM wall-vorticity input, fluid circulation/impulse redistributed during remeshing, explicitly invalid in-solid input, and truncation/pruning. Do not claim physical wall impulse is globally invariant, or hide lost fluid circulation with an unconstrained global correction.
- [ ] Compare the minimal existing-mask fix with the conservative near-wall treatment before adding cut-cell/fractional-aperture complexity. Introduce that complexity only if geometry-shift/refinement tests show the simpler method cannot meet accuracy gates.

Tests: translated cylinders at different lattice phases; wall-adjacent manufactured vorticity; constant/linear field transfer; zero/nonzero viscosity; variable viscosity if supported; multiple axial resolutions; genuine non-box STL; domain-crossing cylinder; CPU/GPU agreement; restart equivalence. Require zero strictly interior accepted particles, no cross-solid diffusion leakage, explicit strength/impulse accounting, and bounded near-wall velocity/curl errors under refinement. Check conservation against the intended fluid exchange budget, not an isolated-body conservation assumption.

## 3. Make the resolved 3D span physically comparable

- [x] Use full 3D induction and all three velocity/vorticity components. The coupled setup uses `SlipSlabInduction` with physical planes at `z=±0.48D`; evidence: [coupled setup](../tutorials/coupled_fvm_vpm/01_cylinder_shedding_flow/setup.py) and [slip-slab tests](../tests/vpm/test_slip_slab_induction.py).
- [x] Start with a common span of **0.96D** and keep force reference area D × the actual physical span. Evidence: [reference-case tests](../tests/tutorials/test_reference_cases.py) and [cylinder grid tests](../tests/coupler/test_cylinder_grid_campaign.py).
- [ ] Test **0.48D, 0.96D, 1.92D** at fixed central resolution and qualify span dependence.
- [x] Resolve the FVM-slip / VPM-free-space mismatch before accepting comparisons. Assess a compatible 3D slip-slab extension using mirrored/image vorticity and a converged induction tail, including the correct axial-vector reflection parity. Evidence: `test_slip_slab_induction.py`, `test_gbd_body_mask.py` and the common-span pilot; the tail test is empirical, with independent direct-sum comparisons. Treat image particles as boundary representation, not independent injected physical particles. Extend diffusion and transfers consistently.
- [ ] If no affordable compatible slab formulation passes, report that limitation and test a larger matched physical span with a qualified central observation region. Simply enlarging the VPM z bounding box is not a valid fix. Do not silently substitute a finite cylinder with endcaps or periodicity.
- [x] Distinguish physical span faces from x/y exchange faces when constructing ownership ramps. Check that thin spans and buffer widths do not erase the FVM-authoritative interior through unintended z tapering.
- [ ] Keep uniform dz near the body and exchange region initially. Quantify anisotropic FVM donor transfer by refining dz independently of hxy; use cell volumes and physical distances, not nearest-centre distance as a solid/fluid classifier.
- [ ] Measure spanwise profiles of drag/lift, mean velocity, w RMS, cross-span vorticity components, and growth/decay of a small reproducible 3D perturbation. Include midspan versus off-midspan errors and span-boundary leakage.

Gate: compatible span physics, stable 3D particle support, and converged quantities in the observation region. A span-invariant long-time wake at Re = 150 is acceptable if the test actually resolves and exercises 3D perturbations.

## 4. Select mesh sizes and domains from measured cost

The original planning candidates below remain optional. The execution pipeline
uses the initial candidate geometric family **.10, .08, .064D** (ratio 1.25)
on the same .96D span. Neither family is certified grid independent or within
12 hours before full measurements. Use `--grids` to choose another geometric family.

Original candidates:

| Candidate | hxy/D | Uniform z layers | Realized dz/D | Role |
| --- | ---: | ---: | ---: | --- |
| A | 0.080000 | 12 | 0.080000 | Screening and domain/span experiments |
| B | 0.056569 | 17 | 0.056471 | Intermediate accuracy and timing |
| C | 0.040000 | 24 | 0.040000 | Initial target for production qualification |
| D | 0.028284 | 34 | 0.028235 | Optional finer candidate; accept only if measured budget permits |

These are dimensional spacings for D = 1 m, not claimed cell counts. Near-wall realized dimensions, curvature resolution, and the extrusion's outer aspect ratios must be measured. If the finest candidate fails the budget, select a justified coarser family and disclose its remaining error; do not reduce the sampling horizon or hide the rejected fine case.

- [ ] Mesh A/B first. Reject excessive skewness/conditioning or memory before timing solves. Qualify hxy and dz separately before interpreting combined 3D refinement/GCI.
- [ ] Retain the reference outer domain [-8,24]D × [-10,10]D as the initial comparison; test a compact [-6,16]D × [-6,6]D alternative against it at fixed inner resolution. These are experimental dimensions, not accepted replacements.
- [ ] Compare the current coupled box against at least one larger candidate, initially [-2,4]D × [-2,2]D. Construct transfer/ownership bands from mesh and stencil support, and verify the resulting FVM interior remains nonempty.
- [ ] Start particle spacing from the current hp/hwall = 1.25 and keep hGBD = hp. Refine the particle/FVM resolution ratio separately so reference grid convergence is not confused with transfer error.
- [ ] Size downstream VPM extent from observed wake convection/residence and truncation influence. Budget **GBD grid volume**, FMM workspace, particle high-water count, and staging separately. Revisit the current 13.2D VPM axial extent only after choosing the span model.
- [ ] Benchmark 1, 2, 4, 6, and 8 MPI ranks on A/B, including CPU affinity on performance/efficiency cores. Preserve one BLAS thread per MPI rank and separately measure owner-side VPM CPU/GPU work. Run trials serially to avoid contaminating timings.
- [ ] Use bounded warm and cold pilots (initially 2–5 wall minutes per trial), with at least three repeats of final candidates. Include developed-wake checkpoints where available, plus a larger-particle-count replay to test growth; a near-empty startup cloud is not representative.
- [ ] Match a provisional physical horizon of tU/D = 100 for both cases and choose the statistics start from stationarity, provisionally no earlier than 40. Require enough resolved shedding periods for uncertainty estimates (target at least ten); revise the runtime estimate if a longer observation window is needed.
- [ ] Fit measured phase costs against mesh size, active particles, diffusion nodes, actual FVM substeps, and output volume. Forecast the remaining trajectory using conservative late-wake growth, not startup cost alone.
- [ ] Require a conservative upper runtime estimate below 12 h; target a central estimate below 9 h to leave margin. Include cold meshing/JIT and observed thermal drift. Report extrapolation uncertainty and validate with a longer bounded pilot before certification.
- [ ] Set a measured RAM/device-memory budget that leaves desktop/OS headroom and does not produce sustained swapping. Initially target ≤5 GiB incremental host working set until an idle baseline supports another limit. Do not count swap as usable solver memory.

Reference arithmetic only: a 100 s run at exchange dt = .04 has 2,500 accepted coupling intervals. At the current maximum of three interface sweeps and ten FVM substeps per interval, this can require **75,000 FVM advances**, not merely the 25,000 accepted FVM steps; use observed sweep counts and account for provisional work. The full 12 h allowance is 17.28 s/interval before startup; a 9 h target is 12.96 s/interval before startup. This is a budget constraint, not a measured speed. Adaptive FVM step counts must be forecast from measured accepted timesteps.

Deliverable: machine-specific sizing table with actual cells/faces, hp, dz, domain, ranks/backend, measured timings, peak host/device memory, predicted full runtime interval, numerical errors, and accepted/rejected decision.

## 5. Optimize coupling after correctness qualification

Preserve accepted timesteps, interface iteration limits, convergence tolerances, physical horizon, and diagnostic acceptance gates during A/B comparisons. Rank opportunities by measured time saved per implementation risk.

- [x] Reuse static donor search indices, interpolation weights/geometry factors, body masks, transfer lattice maps, and communication schedules. Specifically precompute the six offset velocity-trace stencils used to reconstruct renewal curl; inspect the existing six-entry interpolation LRU before adding another cache. Cache keys must invalidate for mesh/body/lattice/partition changes.
- [x] Audit repeated MPI gathers before changing communication. Recorded donor gathers take milliseconds versus seconds for boundary induction; retain the existing collective ordering and failure behavior rather than add an unmeasured communication optimization.
- [x] Keep interface rollback in memory and verify coherent restoration. Canonical payloads already avoid disk serialization; accelerated rollback restores fields, patch data, particles and diagnostics. Two-rank rejection tests pass. Further snapshot preallocation was not justified by measured millisecond costs.
- [x] Audit particle copies and concatenations. No additional dominant cylinder transfer-copy bottleneck was found; optional VLM/source-panel GPU round trips are inactive here. Preserve the measured path and record process-tree memory instead of claiming an unmeasured buffer improvement.
- [x] Reuse an FMM tree across compatible velocity/gradient target evaluations only while source positions/support are unchanged; use explicit state versions. Rebuild on renewal/advection as required. Consider fusing target evaluation where it preserves numerical accuracy.
- [x] Evaluate further GBD mask/active-region work against measured costs. Retain validated halos, wall corrections and bounded allocation; further support truncation or dynamic allocation was not justified in this pass.
- [x] Retain requested audit/output records. Accepted-output costs are small compared with induction; do not remove diagnostics or alter output cadence for a nominal speedup.
- [x] Measure accepted optimizations using repeated component inputs and matched short coupled runs, reporting setup, phase costs, forces, fields, conservation and restart/rollback tests. Record single-run timing uncertainty explicitly. The quiet baseline repeat and full comparison are in `cylinder_optimization_comparison_2026-09-22.json`; production runtime remains a separate open qualification.

Primary files: [interface iteration](../source/coupler/interface_iteration.py), [coupler driver](../source/coupler/solver.py), [boundary evaluation](../source/coupler/boundary.py), [interpolation](../source/coupler/interpolation.py), [renewal](../source/coupler/stable_renewal.py), and [diffusion](../source/solvers/vpm/physics/diffusion/grid.py).

## 6. Conduct a staged accuracy–cost sensitivity study

Define “injection rate” explicitly: the present method renews an authoritative overlap once per accepted coupling interval, with possible provisional interface sweeps. It is not an independent arbitrary particle-count-per-second source. Measure net released strength, replaced particles, retained exterior particles, and accepted renewal frequency; do not count provisional sweeps as physical injection events.

| Factor | Initial study levels | Control / question |
| --- | --- | --- |
| Accepted renewal interval | 1×, 2× baseline; add .5× only through a separately controlled exchange-timestep study | Keep FVM/VPM integration and boundary-refresh clocks fixed when testing skipped renewal; introduce this capability only with explicit ownership/history and conservation tests. |
| hp/hwall | 1.0, 1.25, 1.5 | Quantify transfer resolution versus count/cost; do not change vorticity cutoff. |
| Core radius / hp | 0.8, 1.0, 1.2, subject to validated overlap limits | Representation error, smoothness, induced boundary trace, and FMM cost. |
| Authority blend width / hp | 4, 6, 8 | Geometry containment and interface reflection at constant solver clocks. |
| VPM-only release buffer / hp | 1, 2, 3, always less than blend width | Renewal imprint and safe convection/diffusion between exchanges. |
| Represented-state amplification cap | 1.4, 1.8, 2.2, only where the current method uses this control | Quantify correction/local-error tradeoffs without relaxing conservation or field-error gates. |
| Transfer-region location / box size | Current and qualified larger configuration | Separate boundary placement from mesh resolution and slab effects. |
| dz/hxy and span | Independent dz refinement; 0.48D/.96D/1.92D span | Diagnose anisotropic donor accuracy and span-boundary dependence. |
| VPM exchange dt | .5×, 1×, 2× qualified baseline in a separate temporal study | Keep accuracy tolerances and physical horizon fixed; account for explicit diffusion substeps and interpolation lag. |
| Induction accuracy | Two qualified FMM accuracy settings if exposed | Field-error versus work; compare selected samples to direct induction on a bounded cloud. |

- [ ] Enforce a release-buffer constraint based on measured travel Umax × renewal interval, diffusion length sqrt(2 nu × interval), and remeshing/kernel support. Reject a cadence that lets vorticity cross the ownership region without a valid exchange.
- [ ] Screen roughly 12–20 carefully selected medium/coarse configurations with short bounded windows. Use paired initial states, consistent accepted clocks, and separate warm/cold timing. Short screening establishes local errors and cost, not converged shedding statistics.
- [ ] Select dominant factors; examine interactions such as cadence × buffer width and particle spacing × core overlap with a small factorial design. Avoid an exhaustive Cartesian product and avoid attributing a factor's effect to a simultaneous mesh change.
- [ ] Confirm shortlisted settings over common statistically stationary windows against a spatially/time-qualified fully meshed reference. Re-equilibrate after changing a parameter; replay timing alone is not a physical sensitivity result.
- [ ] Report mean Cd, Cl RMS, St = fD/U, side-force RMS, mean/RMS velocity profiles, spanwise variation, interface normal/gradient error, mass imbalance, vorticity/impulse exchange budget, near-wall error, particle high-water count, peak memory, and seconds per physical flow time.
- [x] Separate shedding phase error from amplitude/statistical error. Use time weighting for irregular samples, verify window coverage and stationarity, and report uncertainty from cycle/block statistics rather than treating adjacent samples as independent.
- [x] Pre-register provisional acceptance targets: mean drag within 2%, lift RMS within 5%, Strouhal within 2%, and normalized profile L2 error within 3% of the qualified reference. Refine these only from reference uncertainty/physical requirements, not to admit a preferred fast result. Mark differences below uncertainty unresolved.
- [ ] Produce an accuracy–time–memory comparison and explain the recommended configuration and rejected alternatives. Keep existing scientific failure thresholds active; do not optimize vorticity cutoff, delete observation cycles, or simply lower corrector/iteration counts.

## 7. Completion and handoff

- [x] Fix canonical VPM backup discovery in RWM/postprocessing, retain explicit legacy layouts, and test numeric frame ordering. This addresses the Linux `RWM member ... has no backups` failure without depending on a checkout-specific layout.
- [x] Align SciPy's declared minimum with the iterative-solver API and make trapezoidal integration work with the declared NumPy 1.26 minimum.
- [x] Remove eager MPI initialization from the coupled solver and keep GPU memory probing independent of a display connection.
- [x] Make CPU slip-slab diffusion use the same fixed lattice phase as GPU. The matched CPU native restart passes; see `test_slip_slab_cpu_lattice.py` and the execution report.
- [x] Record portable numerical-source/dependency identity for campaign reuse and process-tree RSS for actual MPI worker memory.
- [x] Rebuild the final wheel after wall-image/restart changes and repeat installed-package verification outside the checkout with current and minimum numerical dependencies. Both final wall-image/restart-corrected installations passed `openonda.verify_install --require-site-packages`; the wheel includes the campaign plotting helper and excludes generated outputs.
- [x] Run geometry, conservation, anisotropy, installed-package, CPU/GPU, and serial/MPI regressions relevant to changed code. Test short coupled save/restart with the geometry and renewal state intact. Evidence: execution report and saved Linux pilot logs; macOS is covered by configured CI, not a claimed local hardware run.
- [x] Update tutorial configurations, launchers, reference selection, force normalization, and mesh-family tests together. Existing tests that enforce only two reference CLI options must evolve with the supported study interface rather than block reproducible experiments.
- [x] Save benchmark/sensitivity manifests, raw summary tables, rejected trials, exact commands, and limitations under a dedicated study output root. The final five-case transient screen and real native-restart recovery check are recorded in `cylinder_final_short_cohort.json` and `cylinder_native_restart_journal_smoke.json`; the full stationary study remains open. Check large solver output into neither Git nor the distribution wheel.
- [x] Publish a study report distinguishing measured results, runtime predictions, numerical qualifications, and outstanding long-run evidence. Cite primary numerical references for any new boundary/remeshing treatment.
- [x] Provide commands for the candidate coupled/reference cases and their postprocessing in the execution report and tutorial README. Full production completion remains distinct from short-case verification; do not mark the 12-hour target “measured” until a complete run establishes it.

Execution order: **measurement scaffold → shared geometry + span correctness → numerical qualification → mesh/rank pilots → coupling optimization → retiming → sensitivity confirmation → final case selection and report**. Stop advancing a candidate when it fails geometry, memory, conservation, or runtime gates; preserve its diagnostic evidence.
