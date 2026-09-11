# Thesis integration brief

Owner: Writer. Source package: `studies/vpm_time_integration/` in OpenONDA.

The user requests a succinct Chapter 3 justification of the common-stage
position–strength update, with supporting plots and demonstrations in an
appendix. Treat this as an established, carefully justified design choice;
do not write a correction narrative or make it a central thesis contribution.

## Chapter 3

Consolidate the existing material in
`chapters/ch03_vortex_particle_method.tex`, around the coupled Runge–Kutta
paragraph, `sec:time_advection`, and
`eq:vpm_position_strength_commutator`. Do not append a duplicate explanation.
Aim for one or two short paragraphs that:

- State that positions and vector strengths use common SSPRK3 stages, with
  velocity and gradient recomputed from each common state.
- Explain that this avoids the leading advection–stretching splitting defect.
- Report third-order temporal convergence in the bounded fixed-core study,
  compared with first-order sequential RK3 substeps and second-order symmetric
  splitting, and cross-reference the appendix.
- State that this is an accuracy justification, without claiming universal
  nonlinear stability or third-order accuracy of the entire viscous/LES step.

Match the thesis prose and nomenclature. Use its existing vector-strength
symbol (currently alpha) consistently; do not confuse strength with vorticity,
or claim that velocity is an independently integrated state. Check the
nomenclature table and update it only where needed.

## Appendix demonstrations

Use an appropriate subsection of the existing appendices. Include the brief
commutator derivation with its smooth fixed-particle/fixed-core assumptions,
the experiment definition and reference/error norm, and a compact table of
the observed orders. The accepted ranges are:

| Method | Finest-pair order range, 12 cases |
|---|---:|
| Common-stage SSPRK3 | 2.974–3.000 |
| Sequential RK3 substeps | 0.99993–1.00010 |
| Symmetric splitting | 1.99995–2.00007 |

Use `figures/fixed_core_temporal_convergence.pdf` as the principal demonstration;
`fixed_core_order_summary.pdf` summarizes all cases if it adds useful evidence.
Include `oscillator_stability.pdf` with a short counterexample discussion to
bound the stability claim: SSPRK3 imaginary-axis boundary sqrt(3), exact split
subflow power-bounded interval 0 < omega*dt < 2, non-preservation of physical
Euclidean amplitude, defective boundary at 2. The other figures are optional
supporting material, not a requirement to fill the appendix.

Do not describe the Gaussian-convention NumPy probe as an execution of the
compiled production solver. Preserve the evaluator provenance: the accepted
tables predate the Gaussian precision correction. The separate synthetic
particle comparator is a method-level illustration. Small-cloud tangent and
geometry measurements are empirical diagnostics, not a general stability or
spatial-convergence theorem. Very small temporal errors approach reference
resolution; avoid promotional error-ratio claims.

## Sources and delivery

Use the package's methodology, source ledger, and frozen reports to trace every
claim. Cite Strang (1968), DOI 10.1137/0705041, for symmetric splitting and
Gottlieb, Shu and Tadmor (2001), DOI 10.1137/S003614450036757X, for SSPRK theory
and its prerequisite. Label the Taylor derivation as an application of
standard analysis, and the numerical results as this project's experiment.
If discussing spatial convergence, retain the precise hypotheses from the
verified Cottet source; there is no need to expand that topic for this addition.

Copy selected vector figures into the thesis's established asset structure,
use neutral method labels, and verify consistency with its plotting style.
Compile and visually inspect the affected Chapter 3 and appendix pages,
citations and cross-references. Report completion immediately with changed
files, selected plots and validation outcome. Keep this addition brief and
coordinate it with the existing Chapter 3 work and other active Writer tasks.
