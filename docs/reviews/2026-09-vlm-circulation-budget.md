# Flat-plate bound/wake vector-strength budget

9 September 2026. The first sections diagnose the preceding coupling. The native
strip-exchange implementation described below has since passed analytical,
restart and completed current flat-case checks. The full sequential flat sweep
is still in progress; later tutorials remain stopped. The existing 1e-4 closure
criterion remains unchanged. See the working checklist for live qualification
state and the force-budget document for independent loading checks.

## What the residual measures

The native force samples record the signed integrated bound vector strength and
the sum of the free particles' strength vectors, both in m³/s. Their sum should
close for the intended initially irrotational, fully retained bound/wake system.
This is distinct from filament circulation in m²/s and from the unsigned sum of
particle-strength magnitudes. The latter is not a conserved circulation measure.

All 20 flat-plate executions completed, but the 18 nonzero-angle cases fail this
budget. At 8 degrees the maximum normalized residual is about 2.09%. Independent
polygon tests already establish that emission closes a prescribed, advected old
filament row. That alone does not establish compatibility with particle stretching.

## The interface between filament shedding and particle transport

Write `J_w = grad(u_w)` for the wake-induced velocity Jacobian evaluated on the
bound three-leg polylines. Let `a_b = Gamma ds` denote a bound line element and
`a_p` a free particle strength. There are two distinct reverse interaction rates:

```
B_direct    = integral J_w a_b
B_transpose = integral J_w.T a_b
```

The first is an endpoint velocity flux, by integration along each line. The
geometric new-row construction compensates the transported old closing line
using this flux. Changes in the solved circulation cancel between the bound and
newly shed terms. For a fixed body, the remaining instantaneous geometric
contribution to `d(bound + wake)/dt` is `B_direct`.

Free particles instead receive the bound field's transposed stretching rate:

```
A_transpose = sum_p J_b(x_p).T a_p
```

With the same radial smoothing kernel and pair radius in both directions,
pairwise antisymmetry gives `A_transpose = -B_transpose`. Consequently, even with
matched kernels, the combined instantaneous rate is

```
A_transpose + B_direct
    = integral (J_w - J_w.T) a_b
    = integral curl(u_w) cross a_b.
```

That integral is not zero at the completed plate checkpoint. Thus a geometrically
closed shedding stencil and conservative free/free particle stretching do not
by themselves make this mixed filament/particle coupling conservative. The
discrete wake component is not independently solenoidal, and its smoothed cores
also overlap the bound lattice. Both affect `curl(u_w)` at the bound lines.

The actual implementation adds another difference: bound-to-particle induction
uses Rosenhead filtering with the target particle radius; particle-to-bound
queries use the case's Gaussian source kernel. Kernel matching alone cannot
remove the above term, consistent with the earlier unsuccessful Gaussian bound
quadrature experiment.

## Read-only numerical decomposition

Input: the preceding completed tutorial checkpoint, originally at
`tutorials/vpm/flat_plate/solution/exp_static_aoa08/vpm_000192.h5`, with 10,752
particles and 224 bound panels. That checkpoint is now preserved under the audit
workspace's `flat-before-native-exchange/solution/exp_static_aoa08/`, with its
SHA256 manifest. The canonical tutorial path now holds the current v20 result;
the numerical decomposition below belongs to the preceding checkpoint, not v20.
No state was advanced or overwritten by the diagnostic. Calculations
use f64 arithmetic, analytic Rosenhead line derivatives, independent Gaussian
particle derivatives, and Gauss–Legendre integration of all three bound legs.
The source core radii at this checkpoint range from 0.1631 to 0.7131 m.

All rates below are their y component in m³/s²:

| Quantity | Rate |
|---|---:|
| Actual Rosenhead bound-to-wake transposed rate | +0.6299911054 |
| Reverse Rosenhead transposed rate | −0.6299911054 |
| Reverse Rosenhead endpoint/direct rate | −0.1663408509 |
| Matched-Rosenhead residual | +0.4636502545 |
| Reverse Gaussian transposed rate | −0.6920150481 |
| Actual Gaussian endpoint/direct rate | −0.1875814641 |
| Matched-Gaussian residual | +0.5044335840 |
| Actual mixed-kernel instantaneous residual | +0.4424096413 |

At 32 quadrature points per line, reverse direct integration matches the
independently evaluated endpoint flux within 6.7e-14 (Rosenhead) and 4.8e-13
(Gaussian). Rosenhead pair reciprocity agrees within 5.8e-13. Sixteen-point
results already agree to about 3e-9. The completed 64-point check changes the
reported curl-cross-bound components by less than 1.3e-12 from order 32. These
identities establish the mechanism at
this snapshot; an instantaneous rate is not the finite SSPRK3 step increment or
a quantitative integration of the entire 24-chord history.

The Gaussian `curl(u_w) cross a_b` contribution further splits into:

| Contribution | Rate |
|---|---:|
| Smoothed particle vorticity overlapping the bound lattice | +0.5936617537 |
| Difference between `curl(u_w)` and that particle vorticity field | −0.0892281697 |
| Total matched-Gaussian term | +0.5044335840 |
| Additional Rosenhead/Gaussian interaction difference | −0.0620239427 |
| Actual mixed-kernel residual | +0.4424096413 |

The second row includes the non-solenoidal part of the separately represented
wake; it is not a separate particle-removal loss. Overlap dominates this snapshot,
but attributing the entire discrepancy to core size would omit the other terms.

## Consequence for the repair

The diagnosed mismatch requires a consistent discrete bound/free vorticity exchange. It
must reconcile the line-based shedding source with the chosen particle transport
equation, preserve the accepted boundary condition, and pass force/velocity and
motion checks as well as the strength budget. Merely matching kernels, changing
the Jacobian contraction, or rescaling the total wake strength is not a justified
repair. Earlier contraction/refinement experiments did not establish convergence.

There are deliberate modeling choices in published lifting-line/particle
couplings. For example, Dufour et al. include bound-particle velocity in wake
advection while excluding those particles from stretching and diffusion. That is
part of their complete shedding formulation; it does not justify removing one
term from this solver in isolation. See [Dufour et al., Wind Energy (2024),
section 3](https://onlinelibrary.wiley.com/doi/full/10.1002/we.2905).

The free-particle transpose conservation argument also depends on symmetric
pair interactions; it is not a proof for a separately constrained bound lattice.
See [Winckelmans, Topics in vortex methods (1989), sections 3.2–3.4](https://thesis.library.caltech.edu/4385/5/winckelmans-gs_1989.pdf).

Detailed calculation records in the retained audit workspace:
`flat-reciprocal-stretching-budget.json` and
`flat-reciprocal-stretching-curl-split.json`. Reproduction scripts are retained
alongside them. See [the working checklist](2026-09-vlm-todos.md) for the current
simulation state and remaining sequence.

## Native discrete strip exchange (full-suite qualification in progress)

The implemented exchange retains the existing bound-field velocity and stretching
at each free-particle RK stage. For strip `s`, containing its chordwise panels,
it accumulates their opposite contribution with the actual integrator weights:

```
A_s = Gamma_old,s (TE_right,old - TE_left,old)
      - dt sum_k b_k sum_p stretching(J_bound,s(x_p,k), alpha_p,k).
```

The new transverse source at the convected far-edge midpoint is
`A_s - Gamma_new,s (far_right - far_left)`. The trailing side sources retain
their native positions, radii, shared-root ownership and circulation jumps.
The same vector `A_s` supplies the constant term in the near-wake AIC right-hand
side. This matters: updating deposition alone would break the accepted
no-penetration condition. Even with unchanged scalar circulation, the vector
exchange can be nonzero and must be emitted.

The sum of the new bound and newborn side/transverse vectors equals `sum_s A_s`.
It therefore cancels the actual RK increment from bound-induced free-particle
stretching, to accumulation accuracy, independently of the final circulation
solve or rigid-body orientation. This is a local source exchange; it does not
rescale or modify existing wake particles. It also does not compensate unrelated
free/free approximation error, diffusion, stabilization or particle removal.
External callbacks that replace the full particle RHS introduce their own
forcing and are outside this unforced conservation identity.

This zeroth-moment identity is necessary, not sufficient. Depositing each strip's
reaction at the transverse source is a modeling/discretization choice. Its
impulse, force, velocity and refinement behavior must pass independent checks.
The candidate must not be labeled qualified solely because this identity closes.

The native implementation fuses strip accumulation with the existing bound
velocity/Jacobian traversal, retaining only one particle/panel walk and removing
the provider's full particle-gradient scratch field. A provider context publishes
the exchange only after successful RK integration. Diagnostic probes have zero
weight, and failed stages leave accepted particle fields and ledger validity
unchanged. The next accepted VLM update consumes the ledger; no new tutorial
metadata or sampled-data extraction is introduced. Restart coupling version5
introduced this shedding formulation; the current version6 also stores the native
unsteady force and moment state. Older exact restart formats are rejected.

The 32-step prototype at static 8° reached t=0.4 s with maximum normalized
closure4.3749e-6, below the unchanged1e-4 criterion. CL/CD/CMc4 changed by
-0.0155%/-0.1165%/-0.0423% against the previous run at the same time. These are
startup measurements only. Subsequent native regression and full static8°,
static5° and moving5° checks passed, including force/impulse and velocity/loading
comparisons documented in the force budget. Remaining flat angles are being
checked sequentially; delta wing and both rotor demonstrations remain deferred.

## Residual from approximate free/free induction

The static12° run completed192 steps but exceeded the unchanged1e-4 closure
criterion: peak y residual.00567054 m³/s divided by peak bound strength51.21465
m³/s gives1.10721e-4. The sweep stopped before static15°. This is a separate,
smaller discrepancy from the original bound/free exchange mismatch.

At its retained step160 checkpoint, the exact symmetric particle-pair operation
has zero net transposed stretching rate. The target-dependent tree approximation
does not preserve the pair cancellation exactly. Native evaluations on the
same unchanged8960-particle state gave:

| Free/free induction | Net y rate [m³/s²] | Relative full-rate L2 error against direct f64 |
|---|---:|---:|
| Tree, theta0.1, order1 | −.003070826 | 3.4271e-4 |
| Tree, theta0.05, order1 | −.000339724 | 9.6222e-5 |
| Tree, theta0.025, order1 | −.000323917 | 1.9349e-5 |
| Tree, theta0.1, order2 | −.000152757 | 1.8999e-4 |
| Direct f32 | −.000026294 | 7.5483e-6 |
| Direct f64 | −6.11e-16 | reference |

Two complete accepted-step budgets, evaluated only in disposable native solver
state, distinguish this free/free approximation from bound exchange and storage:

- Step40→41: net y change +6.71944e-5 m³/s; free/free RK contribution
  +6.91904e-5. Remaining terms sum to −1.99601e-6. Here the same stored bound
  filament field was integrated in f64 to separate diagnostic accumulation noise.
- Step160→161: native reported net y change −4.48439e-5 m³/s; free/free RK
  contribution −5.41861e-5. The native bound ledger and emission/diagnostic
  rounding supply +9.32539e-6, and RK storage contributes +1.77078e-8. The
  bound-rate/reaction discrepancy is only −8.35e-10. No preceding/post-RK
  particle-strength mutation contributes in either budget.

Thus the free/free approximation is dominant at both the positive and negative
parts of the residual history. These two steps do not reconstruct every step of
the full history; they identify and quantify the local mechanism. The bound
source must not absorb this unrelated tree error, and wake rescaling would hide
it. The flat tutorial now selects direct induction as its small-system reference.
The full direct static12° run has passed: peak y closure8.95186e-7 and vector
closure4.91637e-6; integrated lift/drag versus fluid impulse differ by-.194%/-.135%
over the sampled window. The remaining direct-reference angles are now running
sequentially. No new bound-exchange equation was introduced for this finding,
and no tolerance was relaxed.

Reports and scripts are retained in the audit workspace:
`flat-v20-static12-rate-probe-160.json`,
`flat-v20-static12-step-budget-{40,160}.json`,
`openonda-flat-strength-rate-probe.py` and
`openonda-flat-strength-step-budget.py`. The final order2 probe uses a fresh
physics workspace because multipole storage is fixed at allocation. The initial
attempt to change an existing order1 workspace was discarded; its first
mislabeled order2 row is not evidence. Production output was not overwritten by
these diagnostics. The preceding static12° dataset is now retained inside the
tutorial under `qualification/pre_direct_v20/exp_static_aoa12` in both roots.
