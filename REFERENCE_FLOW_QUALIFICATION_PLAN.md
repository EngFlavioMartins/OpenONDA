# Reference-flow qualification and grid-independence plan

Status: **not yet qualified for a production grid-independence claim**.
Date: 2026-09-06.

Implementation follow-up: `reference_flow/allrun.sh` now invokes a resumable
eleven-run spatial/temporal/iterative/domain/span campaign with cycle statistics, error budgets,
mesh selection and plots. See the reference-flow README and study tests.
This implements the campaign infrastructure described here; it does not complete
the outstanding mesh qualification or produce a physical grid-independence result.

Laptop sizing follow-up: the current dimensions and D/8, D/16, D/32 family are
defined in `reference_flow/case_definition.py` and documented in
`reference_flow/LAPTOP_SIZING.md` (both under the cylinder tutorial). These
supersede the old resolution/capacity assumptions below, not the qualification
gates. The 266,429-cell enlarged-domain candidate passed independent checkMesh
at 1.74 GiB peak meshing/checking RSS, but still fails production LSQ admission
(11.67 versus 9). Full FVM capacity and the remaining final-profile meshes are
unmeasured. The long campaign has not been launched.

## Decision

Keep the current cfMesh-style mesher and its overall appearance. Do not restart
the translation or require identical native cell counts. The built-in D/40 mesh
exists and the FVM advances on it. There is **no demonstrated general FVM failure**.
However, a short successful startup does not establish accurate cylinder forces,
and the current study postprocessor can accept insufficient evidence. Complete
the qualification gates below before launching the expensive three-grid campaign.

This is the qualification/repair-plan branch of the user's request, not a claim
that the production study has been configured or completed. Production settings
and existing solutions have not been replaced by the diagnostic runs.

## Evidence available now

Evidence directory: `artifacts/mesher-recovery-20260906/` in the repository.

| Check | Built-in D/40 result | Interpretation |
|---|---|---|
| Cells | 523,534 | Full generated mesh, not a preview |
| Connectivity and volumes | One fluid region; closed cells; positive volumes and face pyramids | Essential checks pass |
| Maximum non-orthogonality | 42.274 degrees | No reason here to reject the overall mesh appearance |
| Wall distance to clipped STL | Less than 4.81e-13 | Geometric wall recovery passes |
| Independent OpenFOAM check | 16 concave cells; other checks pass | Localized qualification issue, not proof the solver is broken |
| Maximum OpenONDA LSQ condition | 11.7791 versus configured limit 9 | Current production startup rejects it |
| Native cfMesh LSQ conditions | 13.3840 for STL; 11.6135 for double-precision diagnostic input | Limit 9 also excludes independently passing native meshes |
| Built-in FVM diagnostic | 20 steps; final time 0.0230096; all fields finite; every recorded linear solve converged | Startup works with diagnostic LSQ limit 15 |
| Diagnostic maximum Courant number | 0.992662 | Above warning 0.9, below abort 1.5; not a temporal-accuracy test |

The native STL mesh also completed 20 FVM steps with the same diagnostic
admission limit: final time 0.0229682, finite fields, every recorded linear solve
converged, maximum Courant number 0.875227, and total elapsed time 204.08 seconds.
Both startup tests therefore work. Neither reaches even one shedding cycle or
qualifies the force statistics. Their different adaptive time histories also
prevent interpreting this experiment as matched-time solution equivalence.

The startup changed only the LSQ admission limit to 15 and execution to one core;
the configured discretization and run-acceptance limits were retained. Its total
elapsed time was 173.56 seconds including initialization. The value 15 is an
experimental admission limit, **not a validated production quality threshold**.

Inspect the actual data rather than treating this table as a solver certificate:

- `qualification-builtin/result.json`: every step and linear result.
- `qualification-builtin/solution/fvm.log`: solver log, including Courant warning.
- `qualification-native/`: corresponding independently generated native-mesh startup experiment.
- `builtin-check-ordered/checkMesh.log`: independent mesh check.
- `concavity.json`: affected cell IDs and before/after wall-projection measurements.
- `conditioning.json`: conditioning and worst-cell positions for all three meshes.
- `builtin_vs_native.png`: actual mesh cross-sections, not an illustration.
- `MESHER_RECOVERY_2026-09-06.md`: preceding implementation repairs and checkpoints.

## Gate 1 — qualify the existing mesh, with localized repairs only

1. Freeze the current candidate, full mesh-archive SHA256, STL, configuration,
   mesher source hashes, dependency versions, and native checker identity. Keep
   diagnostic artifacts separate from accepted `solution/<case>` products.
2. Reproduce the 16 concave cells from saved checkpoints. They already exist
   before final wall projection, so do not blame or repeatedly retune that step.
   Extract each cell plus its neighboring cells and locate the first stage that
   introduces concavity. Inspect native operations only around that stage.
3. Preferred acceptance: repair these local configurations while preserving
   positive volumes, face pyramids, wall constraints, and refinement transitions;
   require a full passing independent `checkMesh`. Do not delete the affected
   cells, skip their checks, or smooth the whole domain to conceal the defect.
   If intentionally retaining concave but star-shaped polyhedra, first document
   their support in the actual FVM operators and demonstrate operator accuracy
   on them. That is an explicit alternative qualification, not a silent waiver.
4. Calibrate LSQ admission using native and built-in meshes: constant-field
   preservation, linear-gradient reconstruction, conservative internal-face
   cancellation, and a pressure/diffusion manufactured solution. Include the
   worst-condition cells and the 16-cell neighborhood. Verify mesh-refinement
   error reduction, not just finite answers. Change limit 9 only with these
   results, a documented rationale, and a regression for rejection above the
   chosen supported range. Do not increase it afresh for each subsequent grid.
5. Resolve the existing failing wall/core-interface test against the actual
   all-patch wrapper/intersection-cell contract. Retain real geometric and
   topological assertions; do not simply remove the failing test.

**Exit:** accepted mesh manifest, independent quality verdict, operator tests,
and consistent mesh-publication/solver-startup gates. Stop this stage with a
specific failing fixture if it cannot pass; do not launch the grid campaign.
Arbitrary-STL robustness remains separate from qualifying this cylinder case.

## Gate 2 — demonstrate useful FVM operation

1. Extend the native and built-in startup comparison in fresh directories.
   Record continuity, equation residuals, Courant number, field extrema and
   forces every step during qualification. Different meshes and adaptive times
   mean raw step-by-step fields are not a parity test; compare physical times.
2. Run the accepted built-in coarse mesh through transient decay and sustained
   shedding. If symmetric initialization delays shedding, use one documented
   smooth, small physical perturbation, identical across grids and restarts.
   Do not rely on different grid noise to trigger different transients.
3. Check force integration normalization, wall-patch selection, reference area
   D times span, pressure/viscous contributions, and global flux balance.
   A plausible picture or a converged pressure solve alone is insufficient.
4. Repeat a settled segment with tighter linear tolerances and more PIMPLE
   correction. Quantify changes in all reported force metrics. If both native
   and built-in meshes exhibit the same numerical failure, isolate the FVM
   operator before changing the mesher. If only the built-in fails, localize
   its failing cells/operators.

**Exit:** bounded fields and conservation plus a resolved, stationary periodic
signal and quantified iterative sensitivity. No full arbitrary-geometry rewrite.

## Gate 3 — configure a feasible, reproducible spatial family

Preserve the existing physical problem: Re=150, D=1, U=1, domain
[-8,20] x [-8,8] x [-0.6,0.6], cylinder no-slip and current external patches.
Preserve the STL and fixed physical refinement-box extents. Scale wall, near-body,
wake and background target sizes together; keep wrapper policy consistent.
Do not substitute a two-dimensional model or alter domain/physics to save time.

The requested family remains D/40, D/80, D/160, with D/12 for preflight only.
From the measured coarse count, simple three-dimensional scaling estimates
about 4.2 million and 33.5 million cells for the other grids. These are estimates,
not completed builds. Measure mesh and solver peak memory and warm step costs;
reserve capacity for simultaneous geometry, operators, preconditioners and output.
Do not launch fine merely because coarse fits. The recovered coarse meshing run
alone measured 3.48 GiB peak RSS; linear extrapolation is not a memory guarantee.

If this family exceeds available resources, explicitly choose an adequately
provisioned machine or a revised three-grid spacing family before the campaign.
Do not silently skip fine, relabel D/12 as a production grid, or claim independence
from two grids. Native-like dyadic local transitions do not require an inter-grid
refinement ratio of exactly two.

Implementation targets:

- `setup.py`: explicit study configuration, fixed-step mode, duration/continuation
  controls, consistent perturbation, and documented qualified mesh thresholds.
- `mesh.py`: all-grid mesh-only qualification, exact archive checksum and immutable
  accepted manifest. Reuse verifies geometry, configuration and code identity.
- `allrun.sh` or a campaign driver: mesh qualification before flow, resource gate,
  atomic run states, safe checkpoint resume and no overwrites of unrelated results.
- A campaign manifest: each spatial/temporal run and its parent mesh/checkpoint,
  actual timestep history, full numerical setup and acceptance evidence.

**Exit:** three distinct accepted meshes and a dry-run campaign manifest that
fits the measured resource budget. A build failure is not an analysis result.

## Gate 4 — separate time, iteration and sampling errors from spatial error

1. Establish a stable fixed timestep on the finest grid and use that same step
   for the spatial family. Align sampling without introducing unrecorded extra
   substeps. If adaptive stepping remains necessary, demonstrate equivalent
   temporal accuracy using recorded actual steps; equal Courant targets do not
   by themselves isolate spatial error.
2. Run a temporal family dt, dt/2, dt/4 on a qualified mesh. Check fine-grid
   sensitivity as well if this initial family uses medium. The current Euler
   scheme is first order; do not presume second-order time accuracy.
3. Require at least 20 settled lift cycles for final statistics, using whole
   cycles and the same physical acceptance rules on every run. This is a proposed
   study minimum, not a universal theorem. Compare adjacent cycle blocks for
   drift and extend the run when necessary. Fixed end time 60 and window [30,60]
   must not automatically count as sufficient.
4. Use time-weighted mean/RMS and cycle-based amplitudes and periods. Resolve
   force sampling with a sampling-refinement check. Estimate sampling uncertainty
   from cycle blocks, accounting for correlation rather than treating every
   sample as independent. Cross-check frequency against the resolved spectrum.
5. Freeze tolerances before inspecting convergence: retain 1% for mean Cd and
   Strouhal number, 2% for the existing RMS/amplitude metrics. Allocate the
   budget among spatial, temporal, iterative and sampling contributions. A
   conservative working allocation is 40%, 20%, 20%, 20% respectively of each
   metric's tolerance. Report estimates and limitations, not rigorous error bounds.
   Use an explicitly defined absolute tolerance when a metric is near zero.

**Exit:** stationary statistics with sampling, temporal and iterative sensitivity
below their registered budgets. No-shedding or unresolved differences are
`inconclusive`, not evidence of zero discretization error.

## Gate 5 — repair the study verdict and test it

Update `assets/postprocess.py` and its tests:

- Exclude D/12 preflight from production completion and common-window selection.
- Verify complete mesh checksums, distinct grids and consistent refinement of
  all target sizes; report realized sizes and cell counts, not labels alone.
- Require all earlier quality, conservation, time and sampling gates.
- Retain three-grid Richardson/GCI estimates where applicable. Positive observed
  order alone does not establish an asymptotic regime. The ratio constructed
  from the same fitted three values is not independent proof either. Check
  stability with additional refinement when the estimate is questionable.
- Replace automatic `converged_to_roundoff` success with an unresolved-difference
  status unless independent uncertainty evidence supports the requested tolerance.
- Treat noisy, non-monotone, divergent or near-zero metrics explicitly. Report
  `passed`, `failed`, or `inconclusive`, with reasons and missing prerequisites.
- Publish spatial GCI alongside temporal/iterative/sampling estimates. Require
  their conservative combined estimate within the registered tolerance for every
  primary metric; do not claim physical validation from grid convergence alone.

Regression fixtures: known first/second-order synthetic convergence; duplicate
meshes; missing fine run; missing preflight (must not invalidate good production
data); constant/no-shedding histories; drifting and short histories; noisy equal
metrics; non-monotone convergence; near-zero metrics; altered mesh archive;
restart seams; failed linear solve and over-budget temporal/sampling error.

Methodological basis: NASA's [spatial-convergence guidance](https://www.grc.nasa.gov/www/wind/valid/tutorial/spatconv.html)
recommends three grids for order/GCI estimation and consistent refinement; its
[temporal-convergence guidance](https://www.grc.nasa.gov/www/wind/valid/tutorial/tempconv.html)
treats time accuracy separately. The numerical budgets and cycle minimum above
are proposed acceptance rules for this study, not values prescribed by NASA.

## Definition of done

The reference-flow directory produces independently qualified meshes and a
resumable, resource-checked campaign. Its report contains force histories,
stationarity evidence, all convergence/sensitivity estimates, mesh identities,
actual numerical settings, and a justified per-metric verdict. Mesh and study
regressions pass. Existing solutions remain recoverable.

This plan cannot guarantee that the present three spacings will meet the requested
accuracy. It requires the code to report honestly when more resolution, longer
sampling, or a numerical repair is needed. The next action is Gate 1 on the saved
candidate, not another unconstrained mesher rewrite or an unattended `allrun.sh`.
