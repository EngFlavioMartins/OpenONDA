# VPM–LES audit follow-up: a defect in the Stage 8A dynamic procedure

**Status:** Stage 8A re-run and re-gated. Stage 8B audited but **not** re-run — it
does not need the fix, it needs a design decision (§5).

**Relationship to [vpm-les-validation.md](vpm-les-validation.md):** that document's
*decision* survives. Its stated *reason* for rejecting Mansfield does not. §4.4 and
the Stage 8A row of the §3 gate table are superseded by this note.

## 1. Defect

`scripts/experiments/stage_8a_particle_functional_gate.py`,
`mansfield_dynamic_coefficient`, built the Germano/Leonard term as

```python
ell = convection - convection_test - stretching + stretching_test
```

Two independent errors in one line:

1. **Missing test filter.** The model difference was built correctly as
   `m = test_filter(base_basis) - test_basis`, but the two base terms of `ell`
   never had `test_filter` applied. That leaves the unfiltered high-wavenumber
   content $(I-T)N(\bar u)$ inside the estimator. It is large and essentially
   uncorrelated with $M$.
2. **Inverted sign** relative to this project's own SGS convention in
   `exact_sgs_for_filter`, namely
   $g = -F(\mathrm{conv}) + \mathrm{conv}(Fu) + F(\mathrm{stre}) - \mathrm{stre}(Fu)$.

Consequence: $C_r^2 = -0.006673$. Because the Mansfield eddy diffusivity must be
non-negative, the clip set $C_r = 0$, the closure supplied no SGS transfer at all,
and — because the gate block evaluates `selected = model_records["mansfield_dynamic"]`
— **every** predeclared Stage 8A check failed as a downstream consequence of that
zero. The reported "the dynamic Mansfield model is inadmissible" was an artifact.

## 2. Fix

The Leonard term is not a free choice. It is identically the exact SGS source of
the test filter acting on the resolved field:

$$
L = -\big[T(\bar u\cdot\nabla\bar\omega) - T\bar u\cdot\nabla T\bar\omega\big]
   +\big[T(\bar\omega\cdot\nabla\bar u) - T\bar\omega\cdot\nabla T\bar u\big]
$$

so it must equal `exact_sgs_for_filter(grid, u_resolved, T)["g"]` to machine
precision. That reference path is independent of the dynamic procedure and was
already validated in Stage 7A, where the two-filter decomposition closed to
$3.1\times10^{-15}$.

Changes:

- extracted `leonard_term(...)` so the invariant is explicit and testable;
- added a scale-separation diagnostic (`test_filter_width_over_box`), because a
  positive $C_r$ is still untrustworthy without a self-similar range;
- new `tests/experiments/test_mansfield_dynamic_coefficient.py`, 9 tests.

Four of the nine fail when the fix is reverted, including the identity test and
the AGARD admissibility test; the reverted run reproduces the historical
$-0.0066732810849936515$ exactly.

## 3. Stage 8A after the fix

| Quantity | Before | After |
|---|---:|---:|
| $C_{r,\mathrm{raw}}^2$ | $-0.006673$ | $+0.002396$ |
| $C_r$ | $0$ (clipped) | $0.048949$ |
| Correlation | n/a | $0.518056$ |
| Transfer ratio | $-0$ | $0.088676$ |
| Shell error | $1.0$ | $0.915036$ |
| Enstrophy transfer | $0$ | $-0.039363$ |
| Relative divergence | $0$ | $4.86\times10^{-15}$ |

| Predeclared check | Before | After |
|---|---:|---:|
| reference has forward mean transfer | PASS | PASS |
| operator is solenoidal | PASS | PASS |
| operator is mean dissipative | **FAIL** | **PASS** |
| transfer within 50% | FAIL | FAIL |
| shell transfer error below 50% | FAIL | FAIL |

**Stage 8A still fails, for a different and more defensible reason.** The
coefficient is now admissible and the operator is mean-dissipative, but it
under-transfers by a factor of about eleven.

## 4. Why it still under-predicts

Not the model — the procedure has no scale separation. At the audited operating
point the energy-equivalent particle-filter width is $\Delta_p/h = 7.7749$, so on
the $2\pi$ box:

- base filter: $0.243$ of the box;
- test filter at ratio 2: $0.486$ of the box.

The Germano identity assumes the test filter sits in a self-similar range above
the base filter. Here it spans half the domain. A coefficient estimated across
that gap is not meaningful even when its sign is right.

Independent calibration on the same field (`audit/two_filter_sigma_sweep.py`):

| Coefficient | $C_r$ | Transfer ratio |
|---|---:|---:|
| dynamic, after fix | $0.0489$ | $0.089$ |
| Mansfield paper | $0.1200$ | $0.533$ |
| appendix-A (used in 8B) | $0.1367$ | $0.692$ |
| least-squares optimal | $0.1420$ | $0.746$ |
| **transfer-matched** | $0.1644$ | $1.000$ |

The *fixed* coefficient is well calibrated — appendix-A is within 20% of the
transfer-matched value. The dynamic procedure is what is broken, and it is broken
by resolution, not by algebra.

Note also that correlation $0.518$ is normal for a functional model; Smagorinsky
in homogeneous turbulence reaches roughly $0.2$–$0.4$ a priori. It should not be
read as a failure, and Stage 8A correctly does not gate on it.

## 5. Stage 8B: not re-run, and why

Stage 8B calls `mansfield_gaussian_coefficient()` — the fixed appendix-A value. It
never calls `mansfield_dynamic_coefficient`. **The bug does not touch it**, and
re-running it unchanged reproduces the published numbers.

But its result contains a contradiction that should be resolved before it is
trusted: the model *under*-transfers a priori (ratio $0.692$) yet the posterior
reports it as roughly twelve times more dissipative than the current closure
($-0.04081$ vs $-0.00324$ mean SGS power), over-damping energy and spectrum.

Most likely cause: a Reynolds-number mismatch between the two tests. The
coefficient is calibrated on the AGARD $128^3$ field, but the posterior branches
from a $64^3$ reference that is close to fully resolved — §4.5 reports its
high-wavenumber energy fraction below $5.26\times10^{-5}$. The true SGS transfer
that needs supplying there is far smaller than AGARD's, so an AGARD-calibrated
coefficient over-dissipates.

This is a design decision, not a bug fix, so it is left open:

1. measure the true SGS power of the filtered $64^3$ reference and report it
   alongside the model's, rather than comparing branches only to each other;
2. re-run with the transfer-matched $C_r = 0.1644$, or with $C_r$ calibrated on
   the posterior reference itself;
3. run the a-priori and posterior tests on fields of comparable Reynolds number.

## 6. Also found: a production defect, unrelated to LES

`openonda_current` — the shipped $\nu_t\nabla^2\boldsymbol\omega$ closure — has
`divergence_relative = 0.5256`. A spatially varying $\nu_t$ applied component-wise
to $\nabla^2\boldsymbol\omega$ does not preserve $\nabla\cdot\boldsymbol\omega = 0$,
so the closure injects spurious vorticity divergence at 53% of gradient scale.
Mansfield's curl–curl form is solenoidal to $5\times10^{-15}$.

This is currently recorded only as an `implementation_warning` in the Stage 8A
results JSON. It affects production regardless of the LES question and deserves
its own fix. Pinned by `test_current_openonda_operator_is_not_solenoidal`.

## 7. Revised recommendation

Do not merge either closure — unchanged. But the archive decision in
[vpm-les-validation.md](vpm-les-validation.md) §7 should be deferred, because the
evidence that motivated it was partly an artifact:

- the DIAD rejection **stands**, and is in fact stronger than claimed. The
  particle share of enstrophy transfer is $0.891$ at $\sigma/h=0.75$ and $0.996$
  at $\sigma/h=2.5$ (`audit/two_filter_sigma_sweep.py`), so auxiliary-filter
  deconvolution models at most ~11% of the transfer at *any* usable overlap. This
  is not an artifact of the chosen operating point;
- the Mansfield rejection **does not stand** on the stated grounds. Its fixed
  coefficient is near-optimal a priori; only the dynamic procedure fails, and it
  fails because $32^3$ with $\Delta_p/h = 7.77$ leaves about five effective modes
  per direction — too coarse for a Germano estimate;
- worth noting that $\sigma/h = 1.0$ and $1.5$ were excluded by Stage 6A on soft
  5% anisotropy and phase-sensitivity thresholds only, with **zero** monotonicity
  or amplification violations. Fixing the M4′ phase sensitivity (M6′, or a
  compensated P2M/M2P symbol) rather than over-smoothing would roughly double the
  effective resolved band and make a Germano estimate viable.

Suggested next steps, cheapest first: re-run Stage 8A at $64^3$ or $96^3$ nominal
LES resolution with localized or Lagrangian averaging; then Stage 8B per §5.

## 8. Reproducing

```bash
pytest tests/experiments/test_mansfield_dynamic_coefficient.py -v
python scripts/experiments/stage_8a_particle_functional_gate.py
python scripts/experiments/audit/independent_germano_crosscheck.py
python scripts/experiments/audit/two_filter_sigma_sweep.py
```

`audit/independent_germano_crosscheck.py` and `audit/two_filter_sigma_sweep.py`
reimplement the operators from scratch with no repository imports, as an
independent check. The cross-check reproduces $\Delta_p/h = 7.774940$, exact
enstrophy transfer $-0.443896$, and the legacy $C_r^2 = -0.00667328$ to all
printed digits, then yields $C_r = 0.048949$ for the corrected term — matching the
patched Stage 8A exactly.

`audit/sitecustomize_no_taichi.py` stubs `taichi` and `numba` for environments
without them; put it on `PYTHONPATH` as `sitecustomize.py`. It raises if a stubbed
compute object is ever actually invoked, so it cannot silently produce a wrong
number. Not needed on a normal OpenONDA install.

## 9. Changed files

| File | Change |
|---|---|
| `scripts/experiments/stage_8a_particle_functional_gate.py` | fixed Leonard term; extracted `leonard_term`; added scale-separation diagnostic |
| `scripts/experiments/stage_8a_particle_functional_results.json` | regenerated — see `audit/stage_8a_results_before_fix.json` for the superseded values |
| `docs/figures/vpm_les/stage_8a_particle_functional_gate.png` | regenerated |
| `tests/experiments/test_mansfield_dynamic_coefficient.py` | new, 9 tests |
| `scripts/experiments/audit/` | new: independent cross-checks, before/after results, taichi-free shim |
