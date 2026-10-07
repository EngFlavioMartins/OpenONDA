# Cylinder force investigation and cleanup

Investigation stopped at the user's request on 7 October 2026.

## Outcome

The requested sustained, nearly 100% agreement of the coupled lift and drag
coefficients with `reference_flow` was **not achieved**. No experimental force
correction is being promoted to the tutorial or solver. The investigation found
several measurable numerical issues, but none of the tested combinations proved
that they resolve the long-time force discrepancy.

The latest wider-domain, common-3D coupled control reached 8.12 s. Its drag at that
time was 1.289% below the matched reference. Its first post-ramp lift minimum was
4.512% too large in magnitude and occurred one 0.04 s exchange earlier. These
results fail the requested accuracy target. High correlation in the preceding
short interval did not establish correct amplitude or long-time phase.

## Comparison conditions

The original coupled and reference meshes both had 0.04 m near-body resolution,
104 cylinder faces and 1,540 cells within a radius of 1 m in their original
single-layer meshes. The investigation therefore did not establish a near-body
mesh-resolution mismatch as the cause. Wider temporary coupled meshes were
extracted from the reference mesh and checked for bitwise-identical near-body
geometry.

Matched runs used diameter 1 m, Reynolds number 150, viscosity 1/150 m²/s,
FVM step 0.008 s and coupling step 0.04 s. Both prescribed the same initial
velocity disturbance and quintic freestream change from (1, 0.1, 0) m/s to
(1, 0, 0) m/s between 1 and 2 s. Force coefficients were compared at the same
physical times without fitted coefficient offsets, amplitude scaling or phase
shifts. Forces from unconverged exchanges and incomplete intervals were not
treated as proof of successful convergence.

The final wider temporary case used 0.04 m cubic particles, core radius 0.04 m,
six particle layers across a 0.24 m span, a particle domain extending to x=80 m,
physical side-slip planes at y=±10 m, and a vorticity floor of 0.001/s. Its nominal
FVM box was x=[−3.2, 7.68] m and y=[−3.2, 3.2] m; actual mesh bounds were checked
separately. These experimental inputs are not tutorial defaults. A matched
full-FVM reference ran from startup through 25 s. Reference force histories for
0.24 m and 0.48 m spans agreed with the unit-span normalization to roundoff.

## Tested hypotheses

| Hypothesis or control | Measured result | Conclusion and disposition |
| --- | --- | --- |
| The apparent late loss of correlation is primarily a force-amplitude error. | In the original planar history, lift-peak delay grew from approximately zero at 10.76 s to 2.2 s near 97 s. Unshifted lift correlation fell from 0.999748 over 5–15 s to −0.779 over 90–100 s. The reference period was about 5.34 s versus about 5.50 s coupled. | Accumulating phase error is a substantial part of the historical correlation loss. These are historical planar results, not qualification of the retained 3D formulation. |
| The original x=15 m particle cutoff removes the wake when accuracy deteriorates. | The wake reached the cutoff around 17–18 s. Retaining it to x=80 m recovered part of lift growth, but did not remove the early drag deficit or phase error. | Wake truncation is a contributing mechanism, not a complete explanation. No expanded domain was promoted. |
| Insufficient interface convergence causes the mean-force deficit. | Tightening interface tolerances reduced drag second-difference RMS by about 12.8 times in a matched two-second control. Mean drag remained wrong. | It explains part of the force noise, not the main mean-force or long-time discrepancy. Tutorial tolerances were not changed. |
| Smaller steps or finer particles alone recover the forces. | Historical reduced-step and finer-particle controls retained substantial lift-amplitude deficits. The wider 0.08 m particle control gave about 96.94% of reference lift amplitude and 97.39% of mean drag, with a 5.472 s period. Several finer 3D trials exceeded particle capacity, diffusion/stability limits, or device memory. | No qualified resolution/time-step sequence demonstrated convergence to the requested target. Failed runs were excluded rather than interpreted as accuracy results. |
| Higher arithmetic precision is the missing correction. | Induction arithmetic and an expensive full-f64 experiment did not demonstrate force recovery. | The experimental full-f64 planar-GBD exemption was reverted. |
| Six-point remeshing removes excess numerical diffusion. | Qualification and evolved force controls did not establish the required force improvement. | Experimental changes and case activation were reverted; retired investigation helpers were removed. The pre-existing standalone LAGRANGE6 option was not introduced by this investigation and is preserved. |
| A lower outer-wake vorticity floor improves boundary induction. | Reducing the floor from 0.01/s to 0.001/s reduced frozen normal-velocity RMS error at the wider interface from 0.009764 to 0.001871 m/s. | A real frozen-field improvement, but sustained autonomous force recovery was not proved. No new tutorial floor was adopted. |
| Moving the coupling boundary downstream solves the trace error. | With the same frozen reference particles, normal-velocity RMS error increased from about 0.009766 to 0.010294 m/s; gradient error also increased. | Boundary location alone is not a demonstrated fix. |
| The FVM execution backend changes the reference solution. | A matched 12-second full-FVM comparison using NumPy and Numba differed in forces by less than 1.5×10⁻¹³. | Ruled out at the measured precision. |
| The internal coupling surface should have the prescribed physical-inlet mean velocity. | The reference mean incoming speed at the internal surface was 0.98973448 m/s even though the physical inlet prescribed 1 m/s. Imposing the latter on the internal surface could make a frozen force look better while imposing the wrong condition. | Rejected as a general coupling correction. All experimental inlet-mean controls were removed. |
| Enforcing the external physical-inlet flux recovers the forces. | A frozen control reduced first-step drag error from 1.94% to 0.024%. An evolving short control had first-exchange drag error 0.067%, but later inlet-only runs still had low drag and incorrect forces. | A first-step improvement did not establish sustained recovery. No inlet-specific force correction was retained. |
| Omitting the initial reference outer wake confounds developed-flow comparisons. | A one-time transfer of the full saved reference wake removed that initialization omission. Fresh-start cases also used the same compact disturbance, entirely inside both FVM domains. | Earlier inner-only developed controls were treated as confounded. Complete initialization did not prove force agreement. |
| Missing physical side-wall induction explains the trace and forces. | Full-vector 3D reflected fields reduced frozen normal-velocity RMS from about 0.00954 to 0.00511 m/s. In the small-domain evolving case, the first positive lift peak reached 98.16% of reference, 0.12 s early; the following negative peak reached 96.42%, 0.16 s early. | Partial improvement, with amplitude and phase errors remaining. Experimental channel-field code was removed. |
| A body no-through-flow potential correction solves the missing boundary response. | A 3D dipole correction improved some frozen drag values and short-window mean drag. The small-domain later lift peak was 0.16 s early, and lift RMS error increased. The wider case still had drag −1.289% at 8.12 s and first lift-minimum magnitude +4.512%. | No sustained force match. The experimental correction and Fourier implementation were removed. |
| More complete body-surface potential modes are needed. | A 624-source 3D basis reduced the body-normal RMS residual at 8.12 s from about 0.0411 to 0.00523 m/s. Forces stayed almost unchanged, and all three exchanges failed the unchanged 10⁻⁷ interface tolerance within eight sweeps. | Smaller body residual did not imply force recovery. Rejected and removed. |
| The extra fluid-side velocity taper near the wall suppresses circulation. | Removing it improved first-exchange drag error from −1.271% to −0.797%, but errors returned to −1.262% and −1.276% on the next two exchanges. Transfer amplification increased toward its existing cap. | The improvement was transient. The native taper was preserved and the experiment removed. |
| Reflection at the finite physical inlet corrects the remaining induced flow. | Frozen inlet-normal RMS fell from about 0.000646 m/s to 10⁻⁸ m/s, but tangential error increased. The frozen drag coefficient changed by only 9.37×10⁻⁶ and lift worsened. | No demonstrated force benefit; no inlet reflection was added to the solver. |
| The changing freestream is evaluated at the wrong RK times. | In a native three-component advection test, endpoint evaluation caused a maximum displacement error of 0.012 m; stage-time evaluation reduced it to 4.77×10⁻⁸ m. In the paired cylinder exchanges at 1.04, 1.08 and 1.12 s, Cd changed by at most 6.24×10⁻⁷ and Cl by at most 7.89×10⁻⁶, while Cl differed from reference by as much as 0.02437. | The timing defect is real, but the measured cylinder test did not explain the force discrepancy. The test-only correction was removed and no timing change was promoted. |

## Isolation using reference boundary data

Temporary FVM cases received the actual reference face-normal fluxes and
interpolated velocity gradients on the same coupling surface. This isolated the
near-body FVM solution and the mixed boundary condition from evolving VPM errors.
It was a diagnostic replay, not an autonomous coupled solution.

For the wider mesh over 2–12 s, mean drag differed by +0.1404%, lift correlation
was 0.999550, and one drift-corrected lift-cycle amplitude differed by +0.3173%.
Sampled lift periods were 5.32 s versus 5.36 s, with 0.04 s sampling. These results
show that accurately supplied boundary data can produce much closer forces.
They do not establish which VPM evolution or transfer error is responsible for
the remaining autonomous mismatch.

The autonomous wider control over 3.32–6.48 s had mean Cd 1.2253375 versus
1.2316021 reference (−0.50865%), Cl RMS difference 0.0043417 and Cl correlation
0.9999987. It contained no complete lift cycle. Its worsening drag and incorrect
first lift minimum subsequently invalidated any claim that this early
correlation constituted a successful force match.

## Proven numerical and execution improvements retained

The retained changes are not presented as a force-accuracy solution:

- Dedicated planar induction, planar-channel induction, planar GBD, scalar
  transfer, planar invariant recovery and their special runtime branches were
  removed. The tutorial uses the common full-vector 3D induction, stretching,
  diffusion and cubic particle volume. Particle components are not clamped.
- Interpolation distances are measured in directions resolved by the donor
  geometry. The complete 3D Taylor reconstruction still uses every supplied
  derivative. A real frozen cylinder field's artificial spanwise velocity
  variation fell from 0.00391369 m/s to 1.67×10⁻¹⁶ m/s; artificial curl fell from
  0.00113927/s to 1.27×10⁻¹⁵/s. Rotated donor-plane regression tests verify that
  this is geometric interpolation, not a special 2D physics branch.
- Cubic transfer thresholds consistently use h³. A floating-point equality
  inconsistency between the physical transfer and GBD floors was corrected.
- Gaussian reflected-image evaluation selects complete shells using the
  existing velocity and gradient remainder bounds and the unchanged hard
  shell limit. A frozen 27,598-particle cylinder test took 62.734 s versus
  227.105 s, a 3.62-fold speedup. Velocity difference was 4.7×10⁻⁵ m/s within the
  10⁻⁴ budget, and gradient difference was 1.48×10⁻⁷/s. Independent tests include
  all vorticity components, off-plane velocity and all nine Jacobian entries.

Private GPU experiments initially failed because returned device arrays kept
temporary memory pools alive after a field session closed. Releasing those
arrays before closing the session stopped the accumulation. Paired forces
changed by at most 4.41×10⁻⁷ in Cd and 1.16×10⁻⁷ in Cl; a longer run remained
memory-stable. This was a correction to experiment callers, not a solver
physics change. Those callers have now been deleted.

## Direction-agnostic coupling requirement

The coupling boundary retains its local-normal mixed condition on every face:
normal velocity and the tangential part of the full 3D normal velocity
derivative. It does not select a different physical condition using an inlet or
outlet label. Vorticity may enter or leave any coupling face. Physical solid or
slip walls remain distinct physical boundaries.

Operator checks passed for mixed reconstruction, convection with positive and
negative flux, and skew geometry. New tests reverse a field with all three
vorticity components on each of the six box-face orientations. This verifies
the operators; a full dynamic incoming-vortex simulation was not completed.
No experimental inlet-only correction is retained.

## Cleanup

The experiment workers were stopped. Investigation-specific temporary cases,
checkpoints, logs, archives, generated figures, result tables and auxiliary
scripts were removed. Retired planar experiment helpers and their dependent
tests were also removed. The tutorial's existing file inventory and original
run outputs were preserved; `report.md` is its only added file. The unrelated
`assets/paraview_state.py` edit was preserved and excluded from this commit.

The original near-body mesh resolution, startup inputs, tutorial domain and
coupling tolerances remain unchanged. No wider-mesh, channel/body correction,
inlet constraint, custom core radius, precision exemption, newly added remeshing
scheme or RK ramp experiment remains in the solver. Existing standalone
remeshing options were preserved.

## Validation after cleanup

- 161 selected regression cases passed, covering changed tests, native 3D
  coupling and restart, interpolation, renewal, reflected-image evaluation,
  tutorial configuration and mixed boundaries.
- The independent full-vector CUDA reflected-image test passed (one GPU case).
- Test collection for `tests/coupler` and `tests/vpm` completed without errors.
- `git diff --check` passed. Searches found no remaining runtime references to
  the removed planar backends or investigation correction modules.
- A Git-based tutorial inventory check found no missing original files and
  only the requested `report.md` addition. No untracked investigation files
  remained under `tests/support/cylinder`.

The force accuracy investigation is stopped and unresolved; this report does
not claim that the retained code achieves the requested force match.
