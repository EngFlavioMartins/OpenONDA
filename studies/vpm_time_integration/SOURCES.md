# Source and evidence ledger

The accepted source assessment is preserved verbatim in
[`fixed_core_theorem_source_ledger.md`](provenance/reviews/fixed_core_theorem_source_ledger.md).
This page provides the short public index.

## Primary sources

- G.-H. Cottet, “A new approach for the analysis of Vortex Methods in two and
  three dimensions,” *Annales de l'Institut Henri Poincaré C* 5(3), 227–285
  (1988), [publisher PDF](https://ems.press/content/serial-article-files/17460).
  Used to bound what the continuous-time grid-free spatial convergence theory
  actually establishes; not used as a theorem for the full OpenONDA step.
- G. S. Winckelmans, *Topics in Vortex Methods for the Computation of Three-
  and Two-Dimensional Incompressible Unsteady Flows*, Caltech PhD thesis
  (1989), [DOI](https://doi.org/10.7907/19HD-DF80). Used for particle variables,
  regularized velocity, and strength-law context.
- S. Gottlieb, C.-W. Shu, and E. Tadmor, “Strong Stability-Preserving
  High-Order Time Discretization Methods,” *SIAM Review* 43(1), 89–112 (2001),
  [DOI](https://doi.org/10.1137/S003614450036757X). Used to state the
  forward-Euler premise required by an SSP conclusion.
- G. Strang, “On the Construction and Comparison of Difference Schemes,”
  *SIAM Journal on Numerical Analysis* 5(3), 506–517 (1968),
  [DOI](https://doi.org/10.1137/0705041). Historical attribution for symmetric
  splitting; this package's commutator statement was derived directly.

## Historical OpenONDA records

- [`fixed_core_review.md`](provenance/reviews/fixed_core_review.md): accepted
  fixed-core analysis as written on 2026-09-10. It is intentionally not
  rewritten as a post-fix report.
- [`fixed_core_equation_map.md`](provenance/reviews/fixed_core_equation_map.md):
  line-level mapping to the working tree and historical parent inspected by
  the study.
- [`comparator_review.md`](provenance/reviews/comparator_review.md): earlier
  exact-linear and synthetic comparison.
- [`manifest.json`](provenance/manifest.json): SHA-256 mapping from every copied
  original to the packaged artifact, plus hashes for derived figures.

## Provenance caution

The fixed script is a production-convention standalone NumPy probe. It freezes
the evaluator formula and stage conventions tested at the time, but it is not
native production-solver execution. Repository kernel files were corrected
after the accepted study. Those later defect/patch diagnostics are deliberately
outside this scientific package and are not relabeled as inputs.
