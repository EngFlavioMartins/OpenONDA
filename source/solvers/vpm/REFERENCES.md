# VPM solver — source references

Citation keys used from source docstrings and comments. Keep entries short; put
the derivation in the code comment, the provenance here.

PDFs available under `docs/literature/` are marked **[local]**.

---

## Particle approximation, consistency, overlap

- **[CK2000]** Cottet, G.-H. & Koumoutsakos, P. (2000). *Vortex Methods: Theory
  and Practice.* Cambridge University Press.
  Particle quadrature consistency and the overlap requirement h/σ < 1;
  §5.3 on the 3-D vortex-method instability driven by ∇·ω ≠ 0.
  → `diagnostics/resolution.py` (`mean_overlap_ratio`, `vorticity_divergence_error`)

- **[Beale1985]** Beale, J. T. (1986). A convergent 3-D vortex method with
  grid-free stretching. *Math. Comp.* 46(174), 401–424.
  Convergence and the (h/σ)^m error estimate underlying the overlap diagnostic.

## Regularization kernels

- **[WL1993]** Winckelmans, G. S. & Leonard, A. (1993). Contributions to vortex
  particle methods for the computation of three-dimensional incompressible
  unsteady flows. *J. Comput. Phys.* 109(2), 247–273.
  The high-order algebraic kernel ζ = (15/8π)(1+ρ²)^(-7/2) with its q and g;
  the DIRECT / TRANSPOSED / MIXED stretching forms.
  → `kernels/winckelmans.py`, `numerics/kernels_common.py:_stretching_contribution`

- **[AS1964]** Abramowitz, M. & Stegun, I. A. (1964). *Handbook of Mathematical
  Functions*, eq. 7.1.26.
  The erf approximation used by the Gaussian q and g. **Max absolute error
  1.39e-7** — adequate for f32, and the binding accuracy limit if the solver is
  ever run in f64.
  → `kernels/gaussian.py:err_func`, `acceleration/treecode_gpu.py:_erf_approx`

Verified in-repo (see `tests/vpm/test_kernels_math.py` and
`tests/vpm/test_audit_2026_08_regressions.py`): both production kernels
normalize to 1 and have second moment m₂ = 3/2, which is the value the
angular-impulse correction assumes.

## Biot–Savart evaluation and fast summation

- **[BH1986]** Barnes, J. & Hut, P. (1986). A hierarchical O(N log N)
  force-calculation algorithm. *Nature* 324, 446–449.
  Opening-angle (θ) multipole acceptance criterion.

- **[Karras2012]** Karras, T. (2012). Maximizing parallelism in the construction
  of BVHs, octrees, and k-d trees. *High-Performance Graphics.*
  Morton-code LBVH built entirely on device.
  → `acceleration/treecode_gpu.py`

- **[GR1987]** Greengard, L. & Rokhlin, V. (1987). A fast algorithm for particle
  simulations. *J. Comput. Phys.* 73(2), 325–348.
  Background for the multipole orders 1–3 available on tree nodes.

## Vortex stretching

- **[WL1993]** as above — the three discrete stretching forms.
- **[Pedrizzetti1992]** Pedrizzetti, G. (1992). Insight into singular vortex
  flows. *Fluid Dyn. Res.* 10, 101–115.
  Relaxation of particle α toward the local vorticity direction; the misalignment
  diagnostic measures exactly the quantity this addresses.
  → `diagnostics/resolution.py:strength_misalignment_deg`,
  `stabilization/operators.py:apply_pedrizzetti_relaxation`

Only the TRANSPOSED form conserves total particle vortex strength Σα exactly — proved in
`tests/vpm/test_conservation_structure.py`.

## Viscous diffusion

- **[Leonard1980]** Leonard, A. (1980). Vortex methods for flow simulation.
  *J. Comput. Phys.* 37(3), 289–335. Core spreading.
  For the Gaussian, dσ²/dt = 4ν is the *exact* self-similar heat-kernel
  solution.  Both production kernels have m₂ = 3/2, so both use C = 6/m₂ = 4.
  **For an algebraic kernel core spreading is a model, not a discretization** —
  the diffused algebraic blob is not an algebraic blob, so a calibrated constant
  is defensible there provided it is labelled as one.
  → `physics/diffusion.py:core_spreading_diffusion`

- **[Chorin1973]** Chorin, A. J. (1973). Numerical study of slightly viscous
  flow. *J. Fluid Mech.* 57(4), 785–796. Random-walk method, Δx ~ N(0, 2νΔt).
  → `numerics/kernels_common.py:_make_rwm_kernel`

- **[Degond1989]** Degond, P. & Mas-Gallic, S. (1989). The weighted particle
  method for convection-diffusion equations. *Math. Comp.* 53(188), 485–526. PSE.

- **[Durante2024]** Durante, D. et al. (2024). **[local]**
  `docs/literature/durante2024.pdf`, eq. 14–15.
  Diffused Vortex Hydrodynamics truncation parameter β ≈ 0.077 and the fixed
  viscous step Δt_d = β R_d²/(4ν).
  → `physics/diffusion.py:_DVH_BETA`

- **[Rossi2005]** Rossi, L. F. (2005). Achieving high-order convergence rates
  with deforming basis functions. *SIAM J. Sci. Comput.* 26(3), 885–906.
  Background for particle regeneration in DVH.

> **RESOLVED 2026-08-07.** The Winckelmans core-spreading constant
> was `dσ²/dt = (256/45)ν`, which the author confirmed was **hand-calibrated**, with no
> derivation. It is now the derived `4ν` (= 6/m₂, m₂ = 3/2). Three standard matching
> principles give 4ν (second moment), 9.625ν (enstrophy dissipation), 11ν
> (L²/Galerkin) and 14ν (origin curvature); none is 256/45 ≈ 5.689ν, and all give
> exactly 4ν for the Gaussian.  The value is also published, with this kernel and
> σ convention, as eq. (13) of Martins, van Zuijlen & Simão Ferreira,
> *Toward Meshless Turbulent Flow Simulation*, arXiv:2601.06942 (2026).
> **The change is result-affecting: dσ²/dt drops 29.7 %.**
> See `docs/reviews/2026-08-vpm-audit.md` finding N-4.

## LES / subgrid modelling

- **[Smagorinsky1963]** Smagorinsky, J. (1963). General circulation experiments
  with the primitive equations. *Mon. Weather Rev.* 91(3), 99–164.
- **[Lilly1966]** Lilly, D. K. (1966). On the application of the eddy viscosity
  concept in the inertial subrange of turbulence. NCAR Manuscript 123. C_s ≈ 0.17.
  → `config/constants.py:SMAGORINSKY_CONSTANT`
- **[Yoshizawa1985]** Yoshizawa, A. (1985). A statistically-derived subgrid model.
  *Phys. Fluids* 28, 1377. The k-equilibrium form ν_t = C_k Δ √k_eq used here,
  with C_k = (C_s²√C_e)^(2/3).
  → `turbulence/smagorinsky.py`
- **[MKM1998]** Mansfield, J. R., Knio, O. M. & Meneveau, C. (1998). A dynamic
  LES scheme for the vorticity transport equation. *J. Comput. Phys.* 145, 693–730.
  The vortex-method-specific LES literature. Note this implementation uses
  Δ = V^(1/3) rather than σ, deliberately, so the filter width does not inflate
  under core spreading — a VPM-specific choice worth reading alongside [MKM1998].

## Vortex lattice method and wake shedding

- **[KP2001]** Katz, J. & Plotkin, A. (2001). *Low-Speed Aerodynamics*, 2nd ed.
  Cambridge University Press. Lattice construction, the 1/4–3/4 chord rule,
  Kutta condition, and horseshoe influence coefficients.
  → `boundary_elements/vlm/solver/`

- **Kelvin's circulation theorem** — the shed-wake contract. The shipped kernel
  sheds the spanwise *difference* of cumulative circulation at each trailing-edge
  edge (−ΔΓ at interior edges, −Γ₁ and +Γ_n at the tips), which telescopes to
  exactly zero net shed streamwise circulation, plus a transverse particle
  carrying −(Γ(t) − Γ(t−Δt)) span for the unsteady term.
  → `boundary_elements/vlm/solver/kernels.py:shed_wake_particles_kernel`,
  pinned by `tests/vpm/test_audit_2026_08_regressions.py`

## Panel / boundary-element methods

- **[Hess1967]** Hess, J. L. & Smith, A. M. O. (1967). Calculation of potential
  flow about arbitrary bodies. *Prog. Aerosp. Sci.* 8, 1–138.
- **[KP2001]** as above, ch. 10–11, for the source/doublet formulation and the
  inside/outside convention.
  → `boundary_elements/panels/`

## Divergence control and stabilization

- **[WL1993]**, **[CK2000 §5.3]** — the discrete vorticity field is not
  solenoidal and stretching amplifies its divergent part; this motivates the
  projection.
  → `stabilization/divergence_relaxation.py`

- **[Pedrizzetti1992]** as above — the local alternative to that projection:
  rotate α_p toward ω(x_p) by a fixed fraction each step. It is not a
  projection onto a conserved subspace, so the rotation's transfer of
  vortex strength and impulse is reported rather than gated.
  → `config/types.py:StabilizationConfig.pedrizzetti_relaxation`,
  `stabilization/manager.py:StabilizationManager.apply_relaxation`

- **[Alvarez2022]** Alvarez, E. J. & Ning, A. (2022). **[local]**
  `docs/literature/alvarez2022.pdf`, Sec. II.F.
  Reformulated VPM; adopts Pedrizzetti's relaxation as the divergence
  treatment that makes a meshless LES stable in practice.

## Local literature not yet linked from source

`docs/literature/` also holds Constant2016, Cooper2009, Meunier2005,
billuart2023, builland2024, carretelli2003, cottet2014,
kornev2019, rention2025, themas2025, way2024, zeng2024. Where these back a
specific implementation choice, add the citation key above and reference it from
the relevant docstring rather than restating the argument in the source file.
