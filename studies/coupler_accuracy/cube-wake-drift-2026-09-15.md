# Cube wake drift: controlled diagnosis, 15 September 2026

The transfer introduces a reproducible directional bias at the actual particle
spacing of 0.06. Its box lattice was anchored to the lower body corner, followed
by a half-cell shift. A side length of 1 is not an integer multiple of 0.06:
opposite cube walls consequently receive different particle stencils. Centring
the lattice on the body and choosing the phase from control-cell geometry
removes this defect. A separate donor-interpolation
defect arbitrarily truncated equal-distance neighbour groups; that is corrected
as well.

This phase defect is hidden at the earlier power-of-two particle spacings,
which divide the cube side exactly. The corrected phase leaves those lattices
unchanged.

An 800-step fresh, fully 3D trial through reduced time 8 confirms a substantial
improvement: native coupled-FVM velocity RMS falls from 7.17% to 1.95% of Uinf
at time 8. The complete matched drag history from times 1 to 8 improves from
3.12% to 1.28% relative RMS. **The downstream VPM error is not cured.** Its
time-8 plane error near xmax is still 16.27% Uinf, versus 17.41% before, and
its transverse components are worse. This report distinguishes the proven
lattice defect from that unresolved second problem; it does not validate
a complete run to time 20.

The production correction and its regressions are committed as `06422b46`.
The completed component studies and analysis tools are committed as `06b91e77`.

## Preserved baseline and comparison

Commit `333fc4bb75f582aa975eeaf346bb80526ea56ca5` precedes the experiments.
The [baseline manifest](results/cube-wake-drift-2026-09-15/baseline/manifest.json)
identifies 1,490 preserved files, including the interrupted coupled run and the
completed **new fine reference**. Native solutions are independent local
filesystem clones, not hardlinks to active runs. Their hashes are recorded;
native restart/solution files remain outside Git under the repository's data
policy. The versioned samples and figures accompany the manifest.
All 1,490 files passed the post-study SHA-256 integrity check. The archived
run did not record its executing Git revision; the fresh baseline replay
below verifies startup equivalence, not an independent second baseline run
through time 8. The frozen-reference phase experiments establish causality
without relying on a long-run source-identity assumption.

The reference has 692,604 cells; the coupled FVM has 303,264 cells in
`[-1.5,1.5]^3`. Both use the requested fine wall sizing 0.06, the same
linearUpwind/Gauss/backward schemes, PIMPLE counts, relaxation, fluid and LES
settings. Their independently generated cell geometries are not identical.
No coupled cell centre coincides with a reference centre; nearest-centre
distance is typically 0.01 and reaches 0.0263. Matching a sizing request does
not make these identical discrete FVM problems. The reference also uses an
adaptive startup step, while the coupled run uses 0.01 throughout.

The main error measure uses all native coupled fluid cells and all three
velocity components:

\[
 E_u=\frac{1}{U_\infty}
 \left(\frac{\sum_i V_i\lVert u_{C,i}-\mathcal I u_{R,i}\rVert^2}
 {\sum_i V_i}\right)^{1/2}.
\]

Here `I` is the tutorial's affine, 12-donor reconstruction. Comparisons use
exactly coincident physical times, without temporal interpolation. Near-zero
reference velocity does not enter the denominator. Field data, regional
errors, setup checks and reconstruction details are in
[audit.json](results/cube-wake-drift-2026-09-15/audit/audit.json).

An [interpolation-sensitivity check](results/cube-wake-drift-2026-09-15/interpolation-sensitivity-t6/sensitivity.json)
varies the reference donor count from 8 to 12 to 20 at time 6. The baseline
velocity RMS is 0.03905, 0.03892 and 0.03886; the corrected RMS is 0.01023,
0.01060 and 0.01123. The resulting reduction stays between 71% and 74%.
The exact residual depends on reconstruction, but the improvement does not
depend on choosing twelve donors.

| Reduced time | Baseline velocity RMS [% Uinf] | Reference reattachment x/D | Baseline VPM reattachment x/D |
|---:|---:|---:|---:|
| 1 | 0.73 | 0.985 | 0.968 |
| 3 | 1.51 | 1.484 | 1.290 |
| 5 | 2.95 | 1.721 | 1.407 |
| 6 | 3.89 | 1.713 | 1.487 |
| 8 | 7.17 | 1.829 | 1.632 |
| 10 | 12.03 | 1.982 | 1.795 |
| 15 | 21.02 | 2.401 | 1.336 |

Reattachment is the first negative-to-positive centreline crossing downstream
of the rear face, with linear interpolation between samples. These are **x
coordinates**, not lengths from the rear face at x=0.5. When reattachment lies
outside the small FVM domain, its FVM-only value is censored, not set to zero.
The discrepancy starts before time 6: the reference wake reaches the downstream
FVM boundary around time 3, and the coupled recirculation is already short at 5.

## Isolated lattice-phase experiment

The frozen fine-reference fields at times 1 and 6 supply the same 3D velocity
and native Gauss gradients to production velocity-trace curl, Gaussian
representation correction, pruning and moment recovery. The native curl
reconstruction agrees with the stored FVM vorticity to about 5e-7 RMS.
No particles are advected and no physical time elapses. The initially empty
outer wake makes this a transfer-component test, not a complete reference
velocity reconstruction.

Only the lattice phase changes. The actual production anchor is
`(-0.53,-0.53,-0.53)`; the body-centred half-cell lattice uses
`(-0.03,-0.03,-0.03)`. Velocity is measured at paired 3D probes related by a
180-degree rotation about x; vectors are rotated along with their positions.

| Frozen reference time | Lower-corner rotation defect RMS/Uinf | Body-centred defect RMS/Uinf | Reduction |
|---:|---:|---:|---:|
| 1 | 1.02e-2 | 1.25e-5 | 819 times |
| 6 | 4.38e-3 | 2.56e-5 | 171 times |

At time 1, reversing the transverse phase reverses the induced net transverse
vortex strength: approximately `(-0.03616,+0.03616)` becomes
`(+0.03615,-0.03615)`. The centred values are approximately
`(-3.7e-6,+3.5e-6)`. This sign reversal with unchanged donors isolates the
directional source. After ten frozen renewals the centred advantage persists.
The vorticity representation residual also decreases, from 0.323 to 0.284 at
time 1 and from 0.253 to 0.245 at time 6, after one renewal.

The exact [production-phase results](results/cube-wake-drift-2026-09-15/lattice-phase-production/phase.json)
supersede the preliminary `lattice-phase` directory, which omitted the existing
half-cell shift. The early 500-times estimate from that preliminary comparison
must not be used as a production result.

The source correction applies to buffered renewal around an identified box.
The existing coupler also aligns its GBD diffusion grid to this anchor, so
the advancing correction changes both particle-grid phases consistently.
At the actual spacing, the old solid-centre mask covers control volumes up to
0.02 beyond the three positive cube faces. There are **817 partly fluid cells
with solid centres**, containing 0.061208 D³ of fluid. Their nodal strengths
are masked and redistributed by the moment repair, which cannot restore their
original spatial distribution. The centred lattice has no such cells. This
[native-field geometry check](results/cube-wake-drift-2026-09-15/lattice-phase-geometry/phase.json)
identifies one concrete geometric inconsistency in the original phase.

A [mask-only ablation](results/cube-wake-drift-2026-09-15/body-mask-ablation/phase.json)
keeps the old lattice but retains every wall-crossing control cell, including
those centred inside the body. It makes the time-1 rotation defect worse,
from 0.0102 to 0.0151 Uinf. Consequently, the evidence establishes the
**lattice phase and its resulting transfer stencils** as the causal defect;
it does not establish that solid-centre deletion alone explains its magnitude.
Retaining particles inside the body is not a usable correction. The released
phase fixes the opposite-wall geometry without introducing such particles.

For other spacings, simply applying the same half-cell shift is insufficient.
The correction selects between the two body-centred symmetric phases so a
wall-crossing control cell has a fluid centre. This preserves the old lattice
for integer side-length/spacing ratios, including odd ratios. A manufactured,
fully 3D divergence-free field verifies that finite-volume curl is not dropped
in a partly fluid cell with a solid centre. Boundary membership also allows
coordinate roundoff, so a node lying on the FVM box is not excluded on just
one side. The correction neither imposes symmetry on evolving fields nor
reduces the problem to 2D.

## Equal-distance donor selection

`FVMVelocityInterpolator` previously kept four donors even when more were
equally close. A fixed cubic manufactured velocity sampled at the centre of
eight equidistant cells changed when those same cells were reordered. The
correct reflected value is zero. The interpolator now includes the complete
last distance shell; exact donor hits and untied stencils retain their previous
values.

The [separate tie-control experiment](results/cube-wake-drift-2026-09-15/lattice-phase-ties/phase.json)
shows negligible additional change at the centred production query geometry.
This is a confirmed correctness fix, not evidence that donor ties dominate
this cube's late wake error.

## Other findings and limits

* Independent, untruncated float64 Biot-Savart evaluation gives tree velocity
  errors of 0.031–0.084% Uinf RMS at 270 saved profile probes at times 3, 6, 10
  and 15. Replaying the complete saved velocity reproduces its CSV samples to
  about 1.4e-8 RMS. Thus a large instantaneous tree-evaluation error is absent
  at these probes. This does not bound accumulated time-integration error or
  instability sensitivity. See [induction check](results/cube-wake-drift-2026-09-15/induction-verified/probe.json).
  The [final candidate replay](results/cube-wake-drift-2026-09-15/candidate-induction-verified/probe.json)
  additionally checks both saved planes and both profiles at time 8: 1,960
  fluid probes. Both planes reproduce exactly; line differences are below
  4.7e-8 maximum. Tree velocity RMS error against direct summation is
  0.000399 Uinf, and its maximum is 0.00153. The large observed plane error
  is therefore present in the computed field, not a surface-export ordering
  defect or a comparably large instantaneous tree error at these probes.
* At time 6, the downstream mixed-boundary full-velocity mismatch has maximum
  0.591 Uinf and mean 0.0252 Uinf, while its normal mismatch remains near
  roundoff. The discrepancy is tangential. This boundary prescribes normal
  velocity and tangential normal derivative; interface convergence does not
  certify tangential velocity agreement. Changing the boundary condition
  solely because this mismatch exists would not establish its cause.
* A C4 decomposition of sampled 3D errors attributes about 71% of squared
  error at time 10 to the asymmetric part, leaving a significant symmetric
  component. The [checked projection](results/cube-wake-drift-2026-09-15/symmetry-checked/symmetry.json)
  samples a closed query grid once and then permutes its rotation orbits;
  the two components are orthogonal to roundoff. The small FVM mesh itself
  has slight geometric asymmetry, so symmetry measurements are not an
  assertion of an exactly symmetric discrete FVM problem. Halving the probe
  spacing from 0.1 to 0.05 changes the asymmetric squared-error fraction at
  time 10 from 70.75% to 71.07%, supporting the attribution on this probe region.
* Directly evaluating Gaussian-sum vorticity and the analytical curl of its
  velocity shows another discrepancy in the outer wake: about 21% relative
  RMS at time 6 and 39% at time 15 on the recorded 3D probe grid. The source
  potential has zero curl. This [representation check](results/cube-wake-drift-2026-09-15/vorticity-consistency-checked/consistency.json)
  is distinct from the lattice-phase experiment and from the other agent's
  stability correction. It does not establish which transfer or evolution
  operation created the non-solenoidal component. Ratios in regions with
  essentially zero vorticity are reported as null, alongside absolute errors.
* At time 7, a [frozen core-width control](results/cube-wake-drift-2026-09-15/core-sensitivity-t7/sensitivity.json)
  keeps particle positions and strengths fixed, and adds independent direct-sum
  velocity changes to the saved plane field. Scaling radii by 0.8, 1.0 and 1.2
  gives outflow-plane errors of 0.09557, 0.09042 and 0.08931 Uinf RMS. This
  range does not resolve the remaining error. The body potential is held fixed
  in this diagnostic; it is not validation of a different evolving core model.

## Advancing validation

A fresh 50-step baseline replay reproduced the original particle positions
exactly and FVM velocity within 1.2e-7 maximum absolute difference. A second
50-step control reduced only the allocated particle capacity from 1,500,000
to 300,000, avoiding memory pressure during experiments. Positions again
matched exactly; velocity differences remained at 1.2e-7 maximum, with RMS
about 2.3e-9. Spacing, core radius, GBD grid limits, thresholds, timestep and
FVM operators were unchanged. The [capacity control](results/cube-wake-drift-2026-09-15/storage-control-equivalence.json)
records the numerical differences rather than claiming bitwise identity.

The advancing candidate uses the frozen baseline source plus the two explicit
[source patches](results/cube-wake-drift-2026-09-15/source-variants/centred-ties.patch),
the identical 303,264-cell FVM mesh, 0.01 timestep and the verified smaller
storage allocation. It runs separately from the user's solution directories.
The generalised production geometry has exactly the same positions and masks
as the trial at spacing 0.06; [array hashes](results/cube-wake-drift-2026-09-15/release-trial-geometry-equivalence.json)
verify this equivalence. Capacity audits check both transfer and GBD counts;
no capacity-driven population pruning has occurred.
The trial completed all 800 steps, exited successfully and wrote the time-8
coupled checkpoint and scheduled samples. The peak population was 60,689,
well below the 300,000 allocation; all 800 transfer records report zero
capacity pruning. The [output manifest](results/cube-wake-drift-2026-09-15/live-centred-ties-storage/output-manifest.json)
records all 627 native and sampled files. The [complete comparison](results/cube-wake-drift-2026-09-15/advancing-comparison/comparison.json)
and its adjacent CSV tables provide the figure inputs.
The concurrently developed vorticity-alignment stability change is deliberately
absent from this causal comparison; their combined long-time behaviour is not
yet established.

| Time | Baseline velocity RMS/Uinf | Candidate RMS/Uinf | Reference Cd | Baseline Cd | Candidate Cd |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.007271 | 0.007142 | 1.28769 | 1.27297 | 1.27104 |
| 2 | 0.010365 | 0.008709 | 0.95416 | 0.97003 | 0.95545 |
| 3 | 0.015064 | 0.007983 | 0.94099 | 0.97418 | 0.93565 |
| 4 | 0.021891 | 0.008990 | 0.90016 | 0.90905 | 0.90321 |
| 5 | 0.029546 | 0.009760 | 1.03522 | 1.06628 | 1.03326 |
| 6 | 0.038924 | 0.010603 | 1.02490 | 1.05018 | 1.01731 |
| 7 | 0.051751 | 0.012314 | 1.08879 | 1.11979 | 1.07279 |
| 8 | 0.071698 | 0.019467 | 1.01398 | 1.07671 | 0.99961 |

At time 5, VPM reattachment changes from 1.40715 to 1.72075, compared with
reference 1.72089. The native 3D velocity RMS decreases 67%; drag differs
from the reference by 0.19%. The candidate is not better on every measure at
every early state: its time-1 drag error is slightly larger.

At time 6, whole-domain velocity error is 73% lower, and the native downstream
strip (`1.25 <= x/D <= 1.5`) improves from 0.07292 to 0.01893 Uinf RMS, a 74%
reduction. Reattachment moves from 1.4871 to 1.6932 versus reference 1.7128.
The FVM–VPM outflow tangential velocity discrepancy decreases from 0.591 to
0.323 Uinf maximum (mean 0.0252 to 0.0199); it has **not** disappeared. The
normal discrepancy remains at roundoff in both runs.

The saved VPM wake-plane samples give a separate, less favourable result:
near `xmax` (`1.25 <= x/D <= 1.75`, z=0), three-component point RMS falls from
0.11438 to 0.06938 Uinf, a 39% reduction. Across the sampled wake it falls from
0.08203 to 0.04201. These use the actual 0.12-spaced VPM probes and reference
reconstruction at the same coordinates. They are neither an upsampled error
image nor a native 3D volume norm. The remaining VPM error cannot be hidden
behind the better coupled-FVM statistics.

At time 8, native whole-domain and downstream-strip errors are respectively
73% and 63% lower. Reattachment is x/D=1.75825, versus baseline 1.63156 and
reference 1.82889. However, the saved VPM plane gives:

| Time-8 near-xmax error [% Uinf RMS] | Baseline | Corrected |
|---|---:|---:|
| All three components | 17.41 | 16.27 |
| Streamwise component | 13.27 | 5.30 |
| Transverse y component | 8.75 | 11.30 |
| Transverse z component | 7.10 | 10.44 |

Thus a centreline streamwise plot alone would exaggerate success. The
corrected wake retains a growing transverse disturbance. The full-velocity
outflow trace mismatch is again 0.588 Uinf maximum and 0.0310 mean, despite
converged interface iterations and normal agreement at roundoff. Native FVM
continuity remains 1.94e-10 maximum with zero nonfinite values. This locates
the remaining disagreement in the particle representation/outflow coupling;
it does not yet isolate its generation to stretching, diffusion, renewal,
or their interaction. No untested damping or boundary change was added.

The [streamwise profiles](results/cube-wake-drift-2026-09-15/figures/velocity_profiles_t8.png)
and [full vector outflow error](results/cube-wake-drift-2026-09-15/figures/vpm_outflow_velocity_error.png)
must be read together. The latter exposes the remaining error rather than
using the good centreline agreement as evidence of a fully accurate 3D wake.

Likewise, at time 6 the candidate's outer-wake Gaussian vorticity differs from
its velocity curl by 25% relative RMS, compared with 21% in the baseline.
Although the inner-domain ratio falls from 1.78% to 1.31%, the lattice fix
does not repair the outer-wake representation inconsistency. This confirms
that the separate stability/representation question must not be described as
resolved by the present accuracy correction.

All 141 force samples in each run from times 1 through 8 coincide physically,
at intervals of 0.05. Trapezoidal time weighting gives drag RMS errors of
0.03153 before and 0.01296 after correction. Dividing by the reference drag
RMS gives 3.12% and 1.28%, respectively. The maximum instantaneous absolute
Cd error changes only from 0.0681 to 0.0611; visible excursions remain.
This assessment uses the complete matched force interval, with no smoothing
or phase adjustment. See the [force history](results/cube-wake-drift-2026-09-15/figures/matched_drag_history.png).

The focused geometry, interpolation, stable-renewal and representation suite
initially passed 61 tests. A final rerun against the current checkout, after
concurrent changes added three checks, passed all 64 tests. This includes
affine exactness, second-order reconstruction
on graded donors, anisotropic constant-vorticity transfer, complete tied
shells, translated/reflected box lattices, and control-cell exclusion. The new
regressions fail against the original source, as recorded in
[the original-source test log](results/cube-wake-drift-2026-09-15/regressions-original-source.txt).
The native time-advancing boundary qualification tests are separate from these
component checks.

Ruff lint and formatting checks pass for the changed code. The configured
repository-wide [Pyrefly check](results/cube-wake-drift-2026-09-15/pyrefly-final.txt)
still reports 163 errors, including SciPy type information and other solver
paths; a clean repository-wide type check is not claimed. Its two warnings
about the new donor cutoff/tolerance initialization were corrected by making
those definitions unconditional, with no change to the numerical stencil.
Bandit reports no medium/high findings. The final repository-wide Vulture
check flags one unused `body_stl` parameter in the concurrently modified
`source/solvers/vpm/physics/diffusion/grid.py`; that unrelated file is outside
this study's commit. These static-check failures remain recorded rather than
being reported as passes.

## Reproduction

Study entry points are `cube_wake_drift_audit.py`,
`cube_lattice_phase_study.py`, `cube_wake_particle_probe.py`,
`cube_wake_symmetry.py`, `cube_wake_vorticity_consistency.py`,
`cube_wake_live_trial.py`, `cube_wake_trial_comparison.py`,
`cube_wake_interpolation_sensitivity.py` and `cube_wake_core_sensitivity.py`
in this directory.
Each prints its accepted options with `--help`. Use the preserved native
baseline directory for `--baseline`; solution-based checks additionally
require its local native files listed in the manifest.

The production change is selected internally by the transfer geometry; it
requires no new tutorial option. Use a fresh simulation when assessing its
accuracy, because restarting the old checkpoint retains the distorted outer
wake. Neither the tutorial launchers nor their plotters were changed in this
study. No wall-clock speedup is claimed from runs sharing a laptop with other
workloads and diagnostics.

The frozen trees were created from `git archive 333fc4bb` with `source/`,
`openonda/`, `pyproject.toml`, the cube `setup.py` and its `assets/cube.stl`.
The source variant patches apply with `patch -p1` from that tree. The trial
driver records source hashes and prepends the frozen tree before importing
the package, overriding the local environment's editable-checkout path.
Create a `SOURCE_REVISION` file containing `333fc4bb` in a reconstructed tree;
copy the variant description into `STUDY_CHANGE.txt` if applicable.

`plot_cube_wake_drift.py` renders the audit and supplied completed comparison
JSON files at 12.5 cm width with the shared thesis font requirements, validates
layout, and defaults to PNG. All seven figures also passed PDF export and
fixed-page-size verification. It does not change the tutorial plotters.
