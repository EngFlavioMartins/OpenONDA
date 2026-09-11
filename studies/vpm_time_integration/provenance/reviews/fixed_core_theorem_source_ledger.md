# Theorem and source ledger

Access and verification date: 2026-09-10.

| Source | Primary-source material verified | Exact use here | Deliberate non-use |
|---|---|---|---|
| G.-H. Cottet, “A new approach for the analysis of Vortex Methods in two and three dimensions,” *Ann. IHP Analyse non linéaire* 5(3), 227–285 (1988), [publisher PDF](https://ems.press/content/serial-article-files/17460) | Full paper. Three-dimensional particle ODEs (5.7)–(5.9), pp. 269–270; local well-posedness discussion, p. 270; Theorem 5.2 and conditions (5.10)–(5.13), p. 271. | Establishes that a classical 3-D grid-free convergence theorem is a continuous-time spatial result. It assumes a smooth, normalized regularizer with prescribed vanishing moments and `h <= C epsilon^(1+s)`; it yields a velocity error of order `epsilon^d` on a smooth-Euler interval. It also explicitly notes that the finite strength system is nonlinear and only locally well posed absent further control. | Not presented as a theorem for OpenONDA’s complete accepted-step solver. The production Gaussian has nonzero second moment, so the moment hypothesis only supports the low-order (`d=2`) specialization, and the paper does not analyze SSPRK3, split diffusion, LES, VLM driving, or topology changes. |
| G. S. Winckelmans, *Topics in Vortex Methods for the Computation of Three- and Two-Dimensional Incompressible Unsteady Flows*, Caltech PhD thesis (1989), [full thesis](https://thesis.caltech.edu/697/5/winckelmans-gs_1989.pdf), [DOI](https://doi.org/10.7907/19HD-DF80) | Full thesis. Particle state and strength definition, pp. 62–64; regularized Biot–Savart and classical/transposed/mixed strength laws, pp. 70–71; solenoidality and overlap discussion, pp. 71–74. | Confirms the historical particle variables, the regularized velocity/strength equations, the transposed formulation’s conservation motivation, and the distinct spatial-consistency concern caused by non-solenoidal particle vorticity. | Its summaries of other authors’ convergence proofs are treated as secondary summaries, not substitutes for those original theorems. Its aligned-array examples are not generalized into a universal overlap threshold. |
| S. Gottlieb, C.-W. Shu, and E. Tadmor, “Strong Stability-Preserving High-Order Time Discretization Methods,” *SIAM Review* 43(1), 89–112 (2001), [author-hosted PDF](https://math.umd.edu/~tadmor/pub/linear-stability/Gottlieb-Shu-Tadmor.SIREV-01.pdf), [DOI](https://doi.org/10.1137/S003614450036757X) | Full paper. Forward-Euler strong-stability premise and time-step restriction, pp. 90–91; convex Runge–Kutta representation (2.9), p. 93; explicit SSPRK results and CFL coefficient, p. 96. | Establishes what “SSP” does and does not imply: inheritance of a specified forward-Euler monotonicity/contractivity property under a step restriction. | No such functional and forward-Euler estimate is known for the coupled VPM ODE here, so SSP nomenclature is not used as a VPM stability proof. |
| G. Strang, “On the Construction and Comparison of Difference Schemes,” *SIAM J. Numer. Anal.* 5(3), 506–517 (1968), [DOI](https://doi.org/10.1137/0705041) | Bibliographic record and original-paper identity. | Historical attribution for symmetric splitting. The order and commutator statements in this report are derived directly by Taylor expansion for the defined vector fields. | No inaccessible theorem wording or hypotheses are quoted. |
| J. T. Beale, “On the accuracy of vortex methods at large times,” *Math. Comp.* 46, 463–468 (1986), [DOI](https://doi.org/10.1090/S0025-5718-1986-0829616-6) | Primary abstract/metadata only; publisher full text was not accessible in this run. | Context only: the abstract identifies a three-dimensional convergence result and dependence on a zero-average kernel for improved velocity error. | No exact hypothesis, rate, or overlap condition from this paper is asserted. |
| G.-H. Cottet, J. Goodman, and T. Y. Hou, “Convergence of the Grid-Free Point Vortex Method for the Three-Dimensional Euler Equations,” *SIAM J. Numer. Anal.* 28(2), 291–307 (1991), [DOI](https://doi.org/10.1137/0728016) | Primary abstract/metadata only. | Context only: the abstract states convergence to a smooth 3-D Euler solution and identifies consistency plus nonlinear stability as the proof structure. | No exact theorem hypothesis is imported from the abstract. |
| OpenONDA working tree and historical parent `bb1718c9` | Current source was read directly; the predecessor was inspected with `git show`. Exact anchors are in `equation-map.md`. | Defines the actual Gaussian convention, pair core, gradient, transposed strength law, coupled SSPRK3 stages, split core spreading, and the historical advection-then-stretching sequence used by the probes. | The dirty working tree’s unrelated changes were neither edited nor interpreted as part of this study. |

## Claim classification

### Proved here

1. For fixed finite `N` and fixed pair cores bounded below by a positive
   `sigma_min`, the *mathematical* Gaussian pair field built with exact `erf`
   has a smooth analytic origin extension.  Therefore its particle vector
   field is locally Lipschitz and the ODE has a unique local solution.  The
   literal production splice is only piecewise smooth and is treated
   separately below.  Neither statement proves global-in-time bounded strength.
2. The kernel derivatives obey
   `||D^m K_sigma|| <= C_m sigma^(-(2+m))`; the block-Jacobian bounds in the
   report follow by differentiation and finite sums.
3. In the stated scaled block-maximum norm, with Lipschitz constants `L_F`,
   `L_A`, and `L_B` valid on a tube containing both stage trajectories,
   SSPRK3 has the finite nonlinear one-step difference bounds stated in the
   report.  They are growth bounds, not contractivity results.  The separate
   `mu_2` statement is an infinitesimal bound in the scaled full Euclidean norm.
4. The historical `A`-then-`B` composition has leading local split defect
   `(dt^2/2)(B'A-A'B)`, hence first global order generically; the defined
   `A/2`-`B`-`A/2` composition is second order; coupled SSPRK3 is third order
   for the smooth ODE.

### Observed in the bounded probe

1. The production-form gradient matched centered differences to a maximum
   relative operator error `3.15e-5`, attained next to the piecewise
   evaluator’s `rho=0.2` crossover; origin error was `2.40e-12`.  Formula-level
   device branch jumps are `1.214e-8` (`2.597e-5` relative) in 64-bit arithmetic
   and `5.210e-9` (`1.114e-5`) under explicit 32-bit rounding.  The host exact-
   `erf`/series splice is `1.689e-9` (`3.612e-6`).
2. Across twelve `N=27` cloud/core cases, finest-pair observed orders were
   `2.974–3.000` (coupled), `0.99993–1.00010` (historical Lie), and
   `1.99995–2.00007` (symmetric split).
3. In the four `N=8` tangent cases, the coupled centered-difference tangent
   estimate differed from the refined-flow estimate by less than `9e-10`
   relatively; the largest historical estimated excess was `5.57e-5`, on the
   clustered, underlapped cloud.  Five-point logarithmic-norm quadrature is a
   sampled diagnostic, not a certified continuous bound.

### Unresolved

1. No norm/functional was found in which forward Euler is nonexpansive for
   general 3-D stretching, so no SSP CFL theorem for this VPM was established.
2. No universal operational threshold in `sigma/ell` or `sigma/d_nn` follows
   from the theory or the finite-cloud experiment.
3. Global-in-time strength boundedness, spatial convergence for the full
   production Gaussian solver, and stability of the complete driven,
   diffusive, adaptive accepted-step map remain outside the proved result.
4. The exact-Gaussian smooth-ODE theorem does not literally cross the
   production `rho=0.2` switch hypersurface until the two evaluator branches
   and their derivatives are matched.
5. The probe’s sampled fill-distance/mesh-ratio values are lower-estimate
   proxies for the continuum fill-distance supremum, not certified geometry
   bounds; its 3/6/9 call counts do not establish production wall-time ratios.
