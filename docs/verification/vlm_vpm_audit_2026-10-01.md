# VLM–VPM tutorial audit, 1 October 2026

The saved flat-plate runs demonstrate useful coupling consistency, but do not
establish a converged physical error bound. The saved rotor run does **not**
qualify a stationary wake or the requested final time. Smooth figures, frame
agreement and circulation closure are insufficient to make those claims.

This audit preserves the native results. It corrects the reference calculation,
sampling implementation and presentation; it does not adjust aerodynamic loads,
particle strengths, viscosity, thresholds or reference curves to obtain agreement.

## What the figures establish

| `allplot.sh` output | Evidence and limitation |
|---|---|
| Flat `plate_polar` | Completed moving/static polars agree. Lift and induced drag differ systematically from rectangular-wing lifting-line theory; a separate residual panel makes that visible. The model excludes stall, so high-angle points are not experimental validation. |
| Flat `plate_startup` | Preserves the large first static pressure load and the moving ramp. These are different acceleration histories, not an exact transient equivalence test. |
| Flat `plate_staticvsmoving` | Shows late settling and eventual frame equivalence after startup. Tests kinematics and force transformations, not absolute force accuracy. |
| Flat `plate_spanwise` | Final local loading compared with the same rectangular-wing lifting-line solution. The elliptic curve is only a shape comparison. Tip differences remain visible. |
| Flat `plate_velocity` | Final circulation-weighted bound-midpoint velocities, used in sectional forces. Finite-chord/core effects matter particularly near tips; these are not centerline or far-wake velocities. |
| Flat `flat_plate_kelvin` | Bound/wake strength and full-vector closure residual. Necessary conservation evidence, not a load or wake-shape benchmark. Maximum normalized full-vector residual across the suite is about `2.9e-6`. |
| Flat `plate_impulse` | Independently compares integrated surface forces with native bound-plus-wake fluid impulse. At static 8°, final lift/drag residuals are approximately −0.134%/+0.72% of the corresponding fluid impulse. Pressure-time loads use their backward-difference interval weights. |
| Flat `flat_plate_wake` | Qualitative native particle/plate geometry at the recorded time. Glyph sizes are visual, not physical core radii; wake appearance alone is not validation. |
| Rotor `rotor_performance` | Full startup retained, with separate late-window detail and a steady operating-point comparison. Startup points are not connected into a misleading steady actuator-disk envelope. |
| Rotor `rotor_loading_validation` | First-blade circulation and sectional lift compared with matched-geometry BEM, using a bracketed physical-time mean. This is not a blade-to-blade symmetry test. |
| Rotor `rotor_wake_planes` | Five-revolution radial means plus first-two/final-three revolution comparisons. Full induced-vector drift is about 14.0% at 1D and 87.5% at 2D: neither plane is stationary. Ideal far-wake theory is not an exact finite-distance profile. |
| Rotor `rotor_induction_validation` | Signed axial/swirl profiles against corrected finite-distance vortex-cylinder theory. All finite radial bins enter errors, including weak/zero-reference regions. Model mismatch and nonstationarity remain unresolved. |
| Rotor `rotor_streamwise` | Native axial deficit and both signed transverse velocities at four radii. Spatial development is useful evidence, but a time mean by itself does not establish stationarity. |
| Rotor `rotor_impulse` (added) | Surface thrust impulse versus native bound-plus-wake impulse, with an explicit residual and recorded relaxation transfer. Common native interval ends at 7.500 s; fluid/load ratio is about 1.0106, relaxation transfer zero. |
| Rotor animation | Native blade panels and the full native 1D field, with fixed, labeled color scales and physical clocks. The previous every-fourth-point scatter produced diagonal sampling stripes. Neither playback nor a wake image establishes convergence. |

All static charts use a common readable publication layout, separate legends,
explicit units and comparison windows. Startup excursions, nonstationary curves
and residuals are retained. Shared wake-axis limits come from the plotted data.
The exporter checks text bounds and collisions before saving.

## Flat plate: why the lifting-line gap is not a diagnosed coupling bug

The 20 authored runs completed. Final-five-chord-length means give approximately:

| Incidence | CL versus lifting-line | CD versus lifting-line |
|---|---:|---:|
| 2° | −2.59% | −9.81% |
| 5° | −2.73% | −10.06% |
| 8° | −2.98% | −10.52% |
| 10° | −3.22% | −10.95% |
| 15° | −4.04% | −12.42% |

A separate steady VLM calculation removes the particle solver and free wake.
At 5°, the authored 8 × 14-half-span mesh gives CL=0.427430 and CD=0.0058781,
already −2.95%/−12.33% against lifting-line. The coupled case is approximately
+0.23%/+2.59% relative to this VLM baseline. Most of the theory discrepancy
therefore exists without VLM–VPM coupling.

Refining to 16 × 56-half-span panels gives CL=0.422809 and CD=0.0058795.
Chordwise and spanwise refinements were separated; the drag gap does not disappear.
At fixed 8 × 28-half-span resolution, increasing AR from 5 to 40 reduces the lift
gap from −7.69% to −0.67%, consistent with the high-AR assumption of lifting-line
theory. This is evidence of a reference/model difference, **not** independent
proof that every VLM load is correct. The residual drag discrepancy still needs
an independent finite-chord force benchmark and coupled space/time refinement.
Changing force factors to match lifting-line would be unjustified.

A second, independent force integral projects the prescribed straight wake onto
a Trefftz plane and evaluates its infinite-filament crossflow. At the authored
mesh it gives CD=0.0058414, versus 0.0058781 from surface forces (0.63% difference).
Across the AR=10 refinement sequence that difference is 0.26–1.61%, much smaller
than the 12% lifting-line gap. This supports a discrepancy in predicted loading
or model assumptions rather than a 12% surface-force integration error. It uses
the solved circulation, so it is not an independent loading benchmark. The
[Trefftz-plane momentum relation](https://ocw.mit.edu/courses/16-100-aerodynamics-fall-2005/a50f5b9f91c0ad125d141e041e086167_16100lectre19_cg.pdf)
and the straight prescribed wake are distinct from the evolving viscous VPM wake.

Reproduce the numerical isolation study from the repository root:

```sh
python tutorials/vpm/04_flat_plate/assets/verify_steady_vlm.py \
  --output docs/verification/vlm_vpm_steady_comparison_2026-10-01.json
```

The adjacent [JSON](vlm_vpm_steady_comparison_2026-10-01.json) contains every mesh,
aspect ratio, coefficient and reference value. This study does not refine VPM.

## Rotor: completion, reference error and sampling error

Native metadata records step **1259/1667**, time **7.554/10.002 s**, lifecycle
`failed`. The associated log ends with **KeyboardInterrupt**, after about
29 h 47 min. This interruption is not evidence of numerical instability, and
must not be confused with the earlier historical failure described in the README.
Force records end at 7.554 s; field records end at 7.500 s.

Late mean CT≈0.7233 and CP≈0.5311 exceed matched BEM by about **5.78% and 9.23%**.
The loading may look settled, but the wake is not. The 2D particle-front crossing
is bracketed by 3.720–3.744 s, while the plane mean starts near 3.653 s. Early
induced signal at a plane is not proof that a convected wake has reached it.

Three concrete defects were identified and corrected:

1. **Vortex-cylinder assembly sign.** The reference formed sheet jumps as
   `−2 U diff([0, a..., 0])`. Its influence kernel acts inside each cylinder, so
   the correct jump is `+2 U diff(...)`; the outer jump then induces the expected
   velocity deficit. Conversion to positive axial induction uses `−u_induced/U`
   exactly once. Assembly tests now recover prescribed disk induction and its
   far-wake limit for every annulus and nonuniform loading, including swirl.
   This follows [Li et al. (2025), Eqs. 41 and 43](https://wes.copernicus.org/articles/10/2515/2025/).
2. **Weak-reference bins were discarded.** A reference floor is now a denominator
   floor, not an exclusion criterion. After the sign correction, axial RMS
   differences remain **0.762 m/s at 1D and 1.204 m/s at 2D**. Swirl RMS differences
   are **0.161 and 0.136 m/s**. All 48/48 finite bins are included at each station.
   Scaled axial errors are approximately 197% and 238%; weak model signal near
   the hub and outside its nonexpanding cylinder makes these ratios large.
   Selected-bin errors would substantially understate the discrepancy.
3. **General surface-sampling geometry bug.** Float32 `arange` could overshoot
   bounds and produce an asymmetric plane. The saved rotor grid reaches roughly
   −7.800 to +7.867 m, despite declaring symmetric ±7.800 m bounds. New grids use
   bounded uniform coordinates including both endpoints, with spacing no greater
   than requested. This applies to every axis and case. Restart output rejects
   appending a changed grid to an existing series. Existing samples are preserved
   and correctly fail the declared-geometry check.

The cylinder reference is inviscid, nonexpanding and infinite-bladed; BEM also
approximates finite-blade losses. The simulated wake is finite-bladed, expanding,
viscous/LES and still evolving. Those differences prevent assigning the remaining
profile error uniquely to a solver defect. The diagnostic thresholds are unchanged;
no uncertainty-based agreement claim is made. See also
[Li et al. (2022), Section 3](https://wes.copernicus.org/articles/7/75/2022/) and
[CCBlade theory](https://wisdem.readthedocs.io/en/master/wisdem/ccblade/theory.html).

## Qualification still required

The revised `allplot.sh` scripts render the evidence and then run the validators.
Flat plate reports completion/consistency success with an explicit physical-accuracy
qualification gap. Rotor returns a nonzero exit for incomplete duration,
nonstationary fields and the legacy plane geometry; producing files is not a pass.

Remaining work is a completed rotor history with correct sampling, stationary
downstream windows, matched time-step/panel/particle/core refinement, and an
independent finite-chord load comparison. An unchanged native numerical checkpoint
can supply a continuation, but corrected plane sampling must use a fresh output
namespace; old and new grids must not be spliced. Extending a run alone does not
establish accuracy. No long rotor continuation or full coupled refinement campaign
was performed in this audit, so the remaining physical discrepancies are explicitly
unresolved.

## Verification record

All PNG and PDF charts were regenerated, and the complete 315-frame native rotor
animation was rendered. Every chart was inspected; chart exports also passed the
shared text-layout checks. ParaView's local MPI initialization required running
the flat wake rendering outside the socket-restricted sandbox.

The focused regression checks passed 177 tests covering sampling bounds/restart
compatibility, averaging/error definitions, reference-cylinder assembly, native
loading/field clocks, impulse, flat-plate consistency, native scene selection and
plot-layout contracts. The adjacent [source manifest](vlm_vpm_audit_sources_2026-10-01.json)
records hashes for the audited clocks/load histories and VLM implementation.
The saved validator outputs are
[flat plate](vlm_vpm_flat_validation_2026-10-01.txt) and
[rotor](vlm_vpm_rotor_validation_2026-10-01.txt). Reproduce them with:

```sh
python -m tests.support.vpm.flat_plate.validate_results
python -m tests.support.vpm.rotor_flow.validate_results
```

The rotor command is expected to exit 1 for the recorded qualification failures.
