# VLM–VPM surface-interaction evidence and qualification limits

Retaining particles during normal transport avoids introducing the vorticity
sink and impulse change caused by contact deletion. The available evidence
does not establish that deleting particles improves physical accuracy. That
comparison needs a matched independent reference; the short campaign below
does not provide one.

## Retained evidence

The reviewed campaign is in `tutorials/vpm/08_surface_interaction/studies/`.
Its seven cases used two baseline steps or four half-size steps and ended at
0.02 s. Ring centres remained at x = −0.398 to −0.311 m, upstream of the first
wing at x = 0. The physical encounter had not occurred.

Complete-field transport leakage was approximately R1 = 0.04860 and
Rinf = 0.15343, almost identical for lagged and responsive coupling. Surface
refinement did not establish decreasing transport leakage. Halving the time
step changed the recorded front-wing peak lift from 9.8058 to 18.5701;
initialization and the unsteady-load contribution therefore remain separate
from any encounter interpretation. The original report does not state units
for these peak-load values.

The recorded software checks comprised 74 disjoint relevant regressions plus
installed-tutorial checks. Two-step restart position, strength and bound
circulation agreed exactly. These results do not establish full-encounter
accuracy or complete restart/history equivalence.

## Unqualified benchmark criteria

The accompanying [contract](vlm_vpm_completion_contract.json) preserves the
proposed benchmark definitions and numerical targets. It is inactive and does
not certify a completed campaign. The targets are engineering criteria for
that benchmark, not universal physical constants.

| Quantity | Proposed requirement |
| --- | --- |
| Smooth temporal order | At least 1.8 from four levels h, h/2, h/4, h/8 at fixed spatial resolution |
| Interior normalized transport leakage | R1 ≤ 0.005 and Rinf ≤ 0.05 |
| Finest-level trajectory difference | ≤ 0.01 reference chord |
| Load history, peak, integrated load and physical balance differences | ≤ 0.02 of declared fixed scales |
| Receiving-wing feedback | At least three times estimated numerical uncertainty |
| Independent reference uncertainty | No more than 0.2 of the corresponding target |
| Spatial refinement | At least three chordwise, spanwise and particle levels, refined separately |
| Unexplained interior crossings | Zero; core overlap alone is not a material crossing |

The proposed cases include stationary and moving smooth references, a single
wing encounter, offset and stronger tandem encounters, receiving-wing-removed
and one-way controls, encounter restarts, supported precision comparisons and
an installed-package encounter. Their physical vortex profiles, target radii,
edge bands and comparison windows must remain fixed across refinements.
Changing wake release spacing or core size with integration step combines
spatial and temporal errors and cannot establish pure temporal order.

For the recorded geometry, U∞ = 4 m/s, the initial ring centre is x = −0.45 m
and the rear-wing chord extends to approximately x = 1.2 m. The proposed
0.6 s pilot at a 0.005 s time step would contain 120 steps. This is a cost and
coverage estimate, not an endpoint criterion. Actual approach, closest
encounter, departure beyond the rear trailing edge by one reference chord,
and a finite post-encounter window are required. The preceding 0.01 s step
had an unresolved stability warning.

## Numerical questions that the short campaign cannot close

Stage verification must include the newborn-row matrix, old-circulation and
accepted-exchange right-hand side, induced velocity and Jacobian. A nonzero
matrix alone does not verify the field. Independent actual-source comparisons
at zero, intermediate and final stage fractions must exclude double counting
when a virtual row becomes accepted particles. Moving trailing edges,
relative transport, startup and deferred emission affect the same contract.

Diagnostic evaluations must leave accepted fields, circulation histories,
exchange ledgers and clocks unchanged. Tests with nonempty incident particles,
equal-time/different-state stages, changed radii/count and rejected trials are
needed to establish stage-state ownership. DIRECT, TRANSPOSED and MIXED
stretching remain distinct physical choices.

A temporal reference needs consistent initial bound/free circulation, an
identical dimensional startup ramp across levels, raw steady and unsteady
loads, circulation relaxation of one and no force smoothing. An independent
smooth reference must exercise the production boundary solve and varying
incident field. Filament quadrature verifies a component, not the complete
coupled response. The initial proposed reference is inviscid, without
remeshing or optional stabilization.

Spatial comparisons require independent interior and two-sided trace probes
with fixed radii and fixed physical edge bands. A persistent spatial error
would require investigating the bound-surface representation, including its
circulation/potential-jump basis, edge attachment, regularization, induced
field and exchange mapping. Adding an equivalent sheet on top of existing
filaments would double count the bound field. A stronger unsupported encounter
cannot substitute for a passing resolved benchmark.

Bound/free vector exchange and wake/bound/total impulse need their documented
sign and frame conventions, including motion, external forces, shedding and
truncation. Particle impulse alone is not total body force. Encounter restart
checks must include particle fields and identities, RNG, wake buffers,
bound/old/cumulative circulation, geometry, loads, schedules and budgets.
Advection, emission and diffusion relocation are distinct processes. Optional
rollback or refinement claims require accepted-versus-rejected-state evidence.

No complete-encounter, convergence or contact-deletion accuracy claim follows
from the retained short-run measurements. Their source/input identity,
physical coverage, independent reference and uncertainty must accompany any
future comparison.
