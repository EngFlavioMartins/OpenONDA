# VPM numerical references

Physical models and numerical methods used by OpenONDA. See the [VPM guide](../../../docs/vpm.md) for variables, units, configuration and tutorial cases.

## Particle approximation and regularization

- **[CK2000]** Cottet, G.-H. & Koumoutsakos, P. (2000). *Vortex Methods: Theory and Practice.* Cambridge University Press. Particle quadrature, overlapping cores and divergence error in three-dimensional vortex methods.
- **[Beale1986]** Beale, J. T. (1986). A convergent 3-D vortex method with grid-free stretching. *Mathematics of Computation* 46(174), 401–424. Resolution error associated with $h/\sigma$.
- **[WL1993]** Winckelmans, G. S. & Leonard, A. (1993). Contributions to vortex particle methods for the computation of three-dimensional incompressible unsteady flows. *Journal of Computational Physics* 109(2), 247–273. Algebraic regularization and direct/transposed/mixed stretching. Only the symmetric transposed pair formulation conserves total particle vector strength exactly.
- **[DLMF7.6]** NIST, [Digital Library of Mathematical Functions, §7.6](https://dlmf.nist.gov/7.6). Error-function series for Gaussian regularization; see [coefficient attribution](kernels/THIRD_PARTY_NOTICES.md).

## Biot–Savart summation

- **[BH1986]** Barnes, J. & Hut, P. (1986). A hierarchical $O(N\log N)$ force-calculation algorithm. *Nature* 324, 446–449. Tree opening-angle criterion.
- **[Karras2012]** Karras, T. (2012). Maximizing parallelism in the construction of BVHs, octrees, and k-d trees. *High-Performance Graphics.* Device hierarchy construction.
- **[GR1987]** Greengard, L. & Rokhlin, V. (1987). A fast algorithm for particle simulations. *Journal of Computational Physics* 73(2), 325–348. Multipole summation.

## Viscous diffusion

The [Lamb–Oseen tutorial](../../../tutorials/vpm/01_lamb_oseen_vortex/README.md) compares the implemented diffusion schemes.

- **[Leonard1980]** Leonard, A. (1980). Vortex methods for flow simulation. *Journal of Computational Physics* 37(3), 289–335. Core spreading. OpenONDA uses $d\sigma^2/dt=4\nu$: exact uniform-viscosity Gaussian heat spreading, but a second-moment model for Winckelmans cores.
- **[Chorin1973]** Chorin, A. J. (1973). Numerical study of slightly viscous flow. *Journal of Fluid Mechanics* 57(4), 785–796. Random walk with displacement covariance $2\nu\Delta t\,I$.
- **[Degond1989]** Degond, P. & Mas-Gallic, S. (1989). The weighted particle method for convection-diffusion equations. *Mathematics of Computation* 53(188), 485–526. Particle-strength exchange background.
- **[Durante2024]** Durante, D. et al. (2024). [Numerical simulation of 3D vorticity dynamics with the Diffused Vortex Hydrodynamics method](https://doi.org/10.1016/j.matcom.2024.06.003). *Mathematics and Computers in Simulation* 225, 528–544. DVH heat transfer. The reference relation $\Delta t_d=\beta R_d^2/(4\nu)$, with $\beta\approx0.077$, assumes matched steps; OpenONDA uses the accepted physical diffusion interval.
- **[Rossi2005]** Rossi, L. F. (2005). Achieving high-order convergence rates with deforming basis functions. *SIAM Journal on Scientific Computing* 26(3), 885–906. Particle regeneration background.

## LES

The [ring](../../../tutorials/vpm/02_vortex_ring/README.md) compares DNS/LES; the [rotor](../../../tutorials/vpm/06_rotor_flow/README.md) uses an LES wake.

- **[Smagorinsky1963]** Smagorinsky, J. (1963). General circulation experiments with the primitive equations. *Monthly Weather Review* 91(3), 99–164. Strain-based eddy viscosity.
- **[Lilly1966]** Lilly, D. K. (1966). On the application of the eddy viscosity concept in the inertial subrange of turbulence. NCAR Manuscript 123. Coefficient scaling.
- **[Yoshizawa1985]** Yoshizawa, A. (1985). A statistically-derived subgrid model. *Physics of Fluids* 28, 1377. Equilibrium subgrid kinetic energy.
- **[MKM1998]** Mansfield, J. R., Knio, O. M. & Meneveau, C. (1998). A dynamic LES scheme for the vorticity transport equation. *Journal of Computational Physics* 145, 693–730. Vortex-method LES context. OpenONDA uses $\Delta=V_p^{1/3}$ by default, not particle core radius.

## Vortex lattice method

- **[KP2001]** Katz, J. & Plotkin, A. (2001). *Low-Speed Aerodynamics*, 2nd ed. Cambridge University Press. Lattice geometry, quarter-/three-quarter-chord rule, Kutta condition and horseshoe influence coefficients.
- **Kelvin's circulation theorem:** spanwise circulation differences and temporal bound-circulation changes supply the emitted wake. The [flat-plate case](../../../tutorials/vpm/04_flat_plate/README.md) checks bound/wake closure and force/impulse balance.

## Redistribution and stabilization

The [leapfrogging tutorial](../../../tutorials/vpm/03_vortex_interactions/readme.md) compares stabilization models separately.

- **[Pedrizzetti1992]** Pedrizzetti, G. (1992). Insight into singular vortex flows. *Fluid Dynamics Research* 10, 101–115. Strength alignment toward local vorticity. OpenONDA's optional global moment restoration is an additional correction.
- **[vR2011]** van Rees, W. M., Leonard, A., Pullin, D. I. & Koumoutsakos, P. (2011). [A comparison of vortex and pseudo-spectral methods for the simulation of periodic vortical flows at high Reynolds numbers](https://doi.org/10.1016/j.jcp.2010.11.031). *Journal of Computational Physics* 230, 2794–2805. Remeshing and solenoidal reprojection; OpenONDA's padded-grid adaptation remains experimental.
- **[W1995]** Winckelmans, G. S. (1995). [Some progress in large-eddy simulation using the 3-D vortex particle method](https://ntrs.nasa.gov/citations/19960022324). CTR Annual Research Briefs, 391–415. Relaxation and selective eddy viscosity. Global moment restoration is an OpenONDA adaptation.
- **[Rossi1996]** Rossi, L. F. (1996). [Resurrecting core spreading vortex methods: a new scheme that is both deterministic and convergent](https://doi.org/10.1137/S1064827593254397). *SIAM Journal on Scientific Computing* 17(2), 370–397. Core-size control and redistribution. Fixed-core particle splitting alone does not perform core redistribution.
