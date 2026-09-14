# Stretching and reconstructed vorticity in the fully 3D cube

The [saved-state consistency audit](particle_stretching_consistency_3d.py)
finds a material difference between the two particle stretching forms even
when the velocity Jacobian is evaluated by the qualified direct Gaussian
sum. More accurate FMM evaluation therefore addresses only one part of the
particle evolution problem. This does not establish that changing the
stretching form improves the coupled solution.

The audit uses the accepted physical state at time `1.5`: all 28,441 source
particles, their saved strengths and uniform `0.0625` cores, and the same
128 selected near-body and 128 near-wake targets as the
[FMM stage audit](particle-stage-induction-3d.md). It advances neither solver.

## What the two stretching forms measure

With `J[i,j] = du_i/dx_j`, the existing transposed form is `J.T @ Gamma`;
the direct form is `J @ Gamma`. Their exact algebraic difference is

```text
(J - J.T) @ Gamma = curl(u) × Gamma.
```

They agree when the particle strength is parallel to the velocity curl.
The raw Gaussian sum `omega_G = sum Gamma_p zeta_p` is a different field
from `curl(u)` for a general finite particle distribution. Its longitudinal
part contributes no Biot–Savart velocity. These properties and the
conservation rationale for transposed stretching are discussed by
[Winckelmans (1989), sections 3.2 and 3.4](https://thesis.caltech.edu/697/5/winckelmans-gs_1989.pdf).

| Selected region | Direct-minus-transposed rate RMS / transposed-rate RMS (%) | Native FMM rate error / direct-sum transposed-rate RMS (%) | Gaussian vorticity minus velocity curl, relative RMS (%) |
| --- | ---: | ---: | ---: |
| Near body | 194.5118 | 0.44908 | 4.1270 |
| Near wake | 25.3372 | 1.10512 | 15.2279 |

The first column is a **contrast between discrete formulations**, not an
error against a known physical stretching rate or the full FVM trajectory.
Its large near-body value also reflects the transposed rate used as the
denominator. The FMM column retains that same denominator and measures
evaluation error in the existing formulation.

The vorticity column uses `curl(u)` as its denominator. Both fields are
evaluated at the same targets, with every source included and no Gaussian
support truncation. The ratio of Gaussian-field divergence RMS to its
gradient Frobenius RMS is `0.10832` near the body and `0.20523` in the wake.
These are sampled field diagnostics, not bounds on velocity or force error.

| Selected region | Strength-weighted angle to Gaussian sum (degrees) | Strength-weighted angle to velocity curl (degrees) |
| --- | ---: | ---: |
| Near body | 14.5830 | 14.6741 |
| Near wake | 6.3372 | 10.0309 |

The native resolution diagnostic measures the angle to the Gaussian sum.
That angle is useful as a particle-field diagnostic, but it cannot be
interpreted as the exact direct/transposed stretching difference. Substituting
the Gaussian sum in the cross-product identity leaves discrepancies of
`26.24%` and `20.37%` of the reference transposed-rate RMS in these regions.

The mixed FVM boundary already obtains its tangential normal derivative
from the velocity Jacobian. These results therefore do not identify a
raw-Gaussian-vorticity substitution bug in that boundary condition. The
exact body-potential Jacobian is symmetric, so adding that field to an
identical particle state contributes equally to both stretching forms.

## Independent checks and implications

The cross-product identity closes to `3.469e-18` on the physical targets.
Independent fourth-order differences of the Gaussian vorticity sum verify
its analytical gradient to maximum differences `3.536e-11` and `3.120e-10`
at probe sizes `2e-5` and `1e-5`. Rotation checks exercise all three vector
and derivative directions, with maximum gradient discrepancy `1.421e-14`.

A separate nine-particle 3D component example gives a summed transposed
strength rate within `6.655e-16` per component, while the summed direct rate
is approximately `(2.099, -1.485, 1.155)`. This checks why replacing the
transposed form also changes its conservation properties. It does not infer
the physical cloud's total-rate error from the 256-target subset.

The [complete audit record](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/results/interface-precision-frozen-workspace-qualified/studies/coupler_accuracy/results/particle-stretching-consistency-at-one-point-five/particle-stretching-consistency-3d.json)
retains the fields, checks and 328 source/artifact records. All 799 original
frozen files remain unchanged.

The next component experiment applies the
[existing guarded divergence correction](particle_divergence_correction_probe_3d.py)
to this saved particle state, using the actual backup volumes, its unchanged
default gates and grid spacing `0.0625`. It preserves the snapshot's moments;
it does not reproduce the stabilization manager's separate initial-reference
history. The default three-sweep proposal has completed and was rejected:
the reported per-sweep residual ratio `0.9670` exceeded the required
`0.9655`. This is a component rejection, with the input arrays and all 799
original files unchanged. The
[proposal record](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/results/interface-precision-frozen-workspace-qualified/studies/coupler_accuracy/results/particle-divergence-correction-at-one-point-five/particle-divergence-correction-probe-3d.json)
preserves the exact default settings and rejection message.

A [separate work-limit experiment](particle_divergence_work_limit_probe_3d.py)
allows six projection sweeps. The final residual ratio limit remains
`0.9`, the total relative correction limit remains `0.02`, and the moment,
energy, enstrophy, helicity and variation gates retain their existing values.
That proposal has also completed and was rejected, with reported residual
ratio `0.9835` above its per-sweep limit `0.9826`. The
[six-sweep record](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/results/interface-precision-frozen-workspace-qualified/studies/coupler_accuracy/results/particle-divergence-six-sweeps-at-one-point-five/particle-divergence-correction-probe-3d.json)
confirms unchanged inputs and original frozen files.

Doubling this work budget did not produce an admissible correction. No
corrected strengths or post-correction flow improvement were accepted, and
the operator has not been enabled in a coupled run. This bounds the tested
option; it does not prove that a different representation or correction
formulation cannot help. The consistency audit remains a diagnosis, with
the advancing FMM and body-transport tests kept separate.

## Advancing formulation comparison

The [stretching-form runner](cube_stretching_form_3d.py) now isolates the
existing public `DIRECT` option from the native `TRANSPOSED` option. It
retains factor-three FMM, the small matched mesh, `f32` particles, the
original RK/GBD/exchange schedule and the selected panel scope. Every actual
stage checks its contraction against the native Jacobian and temporary
particle strengths, and records the summed strength rate. Outer total
strengths are recorded before and after VPM advancement, before FVM renewal.

The transposed three-interval control is complete. Its histories, six
comparison arrays, 17 FVM entries, 11 boundary-history entries and 11 numeric
VPM datasets match the original control bitwise. Its profile figure matches
the previously inspected control figure. The
[schedule qualification](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/results/interface-precision-frozen-workspace-qualified/studies/coupler_accuracy/results/stretching-transposed-three-schedule.json)
confirms six stage contractions, three GBD calls and twelve outer phases.
Maximum contraction discrepancy is `2.275e-9`, within the declared `f32`
accumulation bound. All 799 original frozen files remain unchanged.

The maximum recorded norm of the summed transposed strength rate, divided
by the sum of individual rate norms, is `2.636e-5` in this control. Thus the
hierarchical evaluator does not reproduce the exact pair-sum conservation
identity to double-precision roundoff; the finite nine-particle example
above uses a different, direct evaluation. These scopes must remain distinct.

A direct-form run is now advancing through 20 intervals over physical time
`0.5–1.5`. The [prepared comparator](compare_stretching_form_3d.py) will check
its forces and profiles, and the first-stage conservation contrast on
identical initial source fields. An accuracy benefit or an acceptable
conservation tradeoff has not yet been demonstrated.
