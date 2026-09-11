# Appendix figure list

Recommended thesis order and captions are below. PDF is the preferred print
asset. SVG is supplied for editable vector workflows and PNG for rapid review.

1. **Temporal refinement of the fixed-core Gaussian particle ODE.** Scaled
   full-state RMS error for regular and cluster/void clouds at
   `sigma/h=0.6`. The common-stage coupled SSPRK3 update shows cubic decay,
   the full sequential RK3 update linear decay, and the symmetric split
   quadratic decay over the resolved time-step range. This is fixed-`N`, fixed-core
   temporal evidence, not full-solver or rotor validation.
   [`fixed_core_temporal_convergence.pdf`](figures/fixed_core_temporal_convergence.pdf)
2. **Observed-order summary across all 12 cloud/core cases.** Individual
   finest-pair orders and medians: 2.974–3.000 for coupled SSPRK3,
   0.99993–1.00010 for the sequential update, and 1.99995–2.00007 for the
   symmetric split. Use this summary for the main method-order claim; do not
   emphasize ratios involving the finest coupled errors near the
   binary64/reference floor.
   [`fixed_core_order_summary.pdf`](figures/fixed_core_order_summary.pdf)
3. **Finite-cloud particle-site moment diagnostics.** Zeroth- and first-moment
   RMS defects versus core/volume spacing for regular, jittered, and
   cluster/void clouds. Boundary truncation is included. The measurements show
   that one overlap ratio is not a complete description and do not define an
   acceptance threshold.
   [`geometry_moment_defects.pdf`](figures/geometry_moment_defects.pdf)
4. **Short-horizon tangent excess relative to a refined flow map.** Signed
   parts-per-million difference between centered-difference numerical tangent
   estimates and the refined-flow estimate. The symlog scale preserves values
   close to zero. These estimates show no resolved extra coupled-method
   amplification at the tested step. They are not certified nonlinear bounds.
   [`tangent_excess.pdf`](figures/tangent_excess.pdf)
5. **Earlier exact-linear and synthetic convergence comparator.** The
   oscillator, growing mode, and Gaussian-like two-particle field independently
   reproduce orders three/one/two. The two-particle panel is synthetic and does
   not use the production Gaussian/Winckelmans kernel.
   [`comparator_convergence.pdf`](figures/comparator_convergence.pdf)
6. **Separate oscillator stability counterexample.** One-step spectral radius
   versus `q=omega dt`. Coupled SSPRK3 crosses unity at `sqrt(3)` whereas the
   exact-subflow split maps remain elliptic for `0<q<2`. This demonstrates that
   the temporal-accuracy ranking is not a universal stability-region ranking.
   It is not a VPM stability threshold.
   [`oscillator_stability.pdf`](figures/oscillator_stability.pdf)
