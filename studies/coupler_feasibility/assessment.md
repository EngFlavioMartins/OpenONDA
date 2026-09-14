# Prospects for OpenONDA’s FVM–VPM coupling

**The method has a credible future as a specialist solver for incompressible, wake-dominated external flows. OpenONDA has not yet demonstrated the accuracy–cost advantage needed to make that claim for its implementation.** The physical premise is sound, but it does not guarantee a beneficial domain decomposition. A compatible representation of the flow, a controlled time coupling, and a sufficiently inexpensive near-body solve must all be achieved together.

The strongest potential application is a long, interacting wake whose accurate propagation makes a fully meshed calculation expensive, while the region requiring wall-resolved CFD remains comparatively small. A general replacement for meshed CFD is a much less convincing objective. The next investment should be a bounded numerical qualification program with explicit continuation criteria, rather than another sequence of case-specific parameter adjustments.

**1. Evidence and present maturity**

This assessment concerns the working tree based on commit `ebe9382414e450abf2bcd0bae8c7c2fc45ac0825`, including the local performance changes present on 14 September 2026. Historical studies use frozen configurations and are distinguished from the current tutorial. Source fingerprints, a new production-operator probe, and recomputed timing ratios are retained in [assessment_checks.json](/Users/flaviomartins/OpenONDA/studies/coupler_feasibility/assessment_checks.json). The numerical evidence does not include a newly completed, developed-wake simulation.

| Evidence | Result | What it establishes |
|---|---|---|
| Matched 3D cube, physical time 0.5–1.5 | Drag-history relative RMS difference falls from 2.082% to 0.555% with interface iteration, and to 0.521% with body transport. Near-body velocity error changes from 0.2005% to 0.1835% of freestream speed. | A substantial improvement in one short-run observable; a smaller improvement in the local velocity field. |
| Independently verified, older iterated configuration through time 4.0 | Drag-history RMS difference 0.759%; endpoint near-wake vector velocity differences 2.846% and 2.444% of freestream on two lines. | Small interface residuals and close drag do not guarantee an accurate wake. |
| Current tutorial performance audit, physical time 0.05–0.10 | Optimized hybrid 43.51 s; full-FVM reference 22.59 s. | The hybrid costs 1.93 times the reference in this startup measurement. |
| Current tutorial startup force at time 0.1 | Hybrid drag coefficient 2.7221; reference 2.3934, a 13.74% difference. | The short matched-laminar improvements are not an accuracy certificate for the current LES tutorial. |
| New isolated production-blend probe | An already represented target changes by 6.87% in represented-field norm under full authority; 3.99% under a ramp. | A concrete finite-resolution transfer inconsistency, isolated from time stepping and physical flow. |

The first two comparisons use a matched laminar cube with 16,936 inner cells and 53,752 full-domain cells. The performance comparison uses the different, current LES tutorial with 303,264 inner cells and 692,604 reference cells. Their errors and timings must not be combined into a single accuracy–cost measurement.[^1][^2][^3]

Later raw records from the older long-wake experiment extend beyond the verified time-4.0 prefix. The snapshot retained here ends at time 9.9, with a signed instantaneous drag difference of −4.14%. That portion has not received the independent force/checkpoint qualification used for the published prefix; its report remains marked unfinished. It is an additional reason to complete the validation, not a qualified result for today’s solver.[^4]

Eighteen existing focused checks passed on the inspected working tree: interface restoration and output scheduling, interpolation accuracy, native face-flux moments, and stress/curl operators. Five of these checks concern an explicitly experimental residual-blend candidate. Passing them does not establish that the production blend has the same properties. The distinction is material here.[^5]

**2. Why the physical idea can work**

For constant-density incompressible flow with constant viscosity, velocity–pressure and velocity–vorticity formulations describe the same continuum equations, provided boundary conditions, initial conditions, and velocity reconstruction are consistent. Taking a curl removes the pressure gradient from the vorticity equation. It does not remove the influence of the pressure field on the velocity that transports the vorticity.

A useful decomposition is

\[
u=U_\infty+\mathcal K[\omega]+u_{\mathrm{harmonic}},
\]

where \(\mathcal K\) is the free-space Biot–Savart operator and the harmonic contribution supplies the boundary and topology-dependent part of the velocity. Near a body, a mesh efficiently resolves anisotropic boundary layers and their production of vorticity. Farther away, particles can follow concentrated vortical structures without repeatedly advecting them across a stationary mesh. This is a legitimate numerical division of labor.

It is not necessary for the outer flow to be inviscid: a viscous VPM can evolve diffusion and three-dimensional stretching. Nor is VPM intrinsically a low-fidelity model. Convergence results exist for particular three-dimensional vortex discretizations, although their assumptions do not prove convergence of this bounded, viscous, remeshing hybrid.[^6]

Published hybrid work supplies useful existence evidence:

| Work | Relevant evidence | Limit of the inference |
|---|---|---|
| Palha et al., 2015 | A circulation-conserving Eulerian–Lagrangian scheme was exercised on a dipole, cylinder, and stalled airfoil without Schwarz iteration. | These are 2D results using a finite-element inner solver. They do not qualify 3D vortex stretching or OpenONDA’s transfer. |
| Billuart et al., 2023 | A weak coupling combines near-wall velocity correction, a mixed boundary condition, and circulation control. | Also 2D, and its outer method is particle–mesh. Weak coupling is possible; its stability cannot simply be assumed for another implementation. |
| Pasolari et al., 2023 | OpenFOAM–VPM coupling reproduces 2D cylinder aerodynamic coefficients and shedding frequency within the reported approximately 1% variation. | Good evidence for coupling feasibility, not a general industrial accuracy or performance guarantee. |
| Alvarez and Ning, 2024 | Reformulated VPM includes a stretching-aware subfilter model and demonstrates turbulent-jet statistics and a rotor application. | The reported 100× rotor speed comparison is specific to its formulation, body modeling, and comparison setup. It does not transfer to a wall-resolved FVM–VPM implementation. |
| *Eulerian–Lagrangian coupling method for simulating dynamic evolution of long-trajectory rotor wake*, 2026 | The publisher’s abstract reports rotor hover, forward-flight/BVI, and pitch-maneuver results, with a 55× overall acceleration. | The authors attribute 4.97× to coupling and 11.1× to GPU acceleration. It uses load–velocity coupling with rVPM; it is not a validation of OpenONDA’s volume-renewal algorithm. |

The table separates published findings from their applicability to this solver. The older complete hybrid papers are predominantly two-dimensional. The more recent rotor results support the specialist opportunity, while also showing why method speedup must be separated from hardware acceleration.[^7][^8][^9][^10][^11]

The competitive comparison must include capable meshed methods: appropriate wake refinement, higher-order discretization where useful, and efficient parallel execution. Low numerical diffusion is valuable, but a practical VPM still has regularization, remeshing, pruning, time-integration, and model errors. In turbulent flow, the objective is the correct transfer and dissipation of energy across scales, not simply the smallest possible dissipation.

**3. The main numerical requirements**

**One compatible flow representation.** In 3D, a particle cloud with Gaussian vector strengths does not generally satisfy

\[
\omega_G=\sum_p\Gamma_p\zeta_\sigma(x-x_p)=\nabla\times u_P.
\]

The Biot–Savart velocity corresponds to the solenoidal part of that vector field. Consequently, preserving a raw vorticity sum is not equivalent to preserving induced velocity or stretching. The local saved-state audit measures a 4.13% difference between Gaussian vorticity and velocity curl near the body and 15.23% in sampled wake targets. These are representation differences, not percentages of physical flow error.[^12]

The FVM has a related discrete distinction. Pressure-corrected face fluxes are conservative, while a simple interpolation of cell velocities produces different face fluxes. The repository verifies this in both the full FVM and the hybrid. The transfer cannot assume that cell velocity, native circulation, conservative flux, and an arbitrary continuous interpolant are interchangeable. A hard fit that tries to satisfy incompatible interpretations can introduce large, nonphysical face variations while reporting tiny constraint residuals.[^13]

The appropriate objective is a stable reconstruction that respects conservative flux and circulation in a declared discrete sense, while approximating the resolved velocity and its derivatives to the expected truncation accuracy. Exact equality to every stored FVM quantity is neither generally available nor necessary. A divergence-conforming reconstruction, a constrained weak projection, or a curl-based correction are plausible design directions; each still needs an advancing comparison.

Naive blending itself needs care. Even if both input vorticities are divergence-free,

\[
\nabla\cdot[\eta\omega_F+(1-\eta)\omega_P]
=\nabla\eta\cdot(\omega_F-\omega_P).
\]

Taking the curl of a blended velocity instead introduces the necessary term
\(\nabla\eta\times(u_F-u_P)\). Omitting that term changes the field. This identity explains why smooth authority weights and conserved integrated strength do not, by themselves, constitute a compatible 3D coupling.

**A transfer that preserves agreement.** The selected `buffered_m4_renewal` path does considerably more than deposit cell vorticity. It samples reconstructed FVM velocities on particle-control-volume faces, computes a discrete curl, blends an inner target with the retained particle field, applies a bounded correction, and prunes and repairs moment budgets.[^14]

The new operator probe makes one limitation explicit. Let \(G\) map particle coefficients to their represented Gaussian vorticity multiplied by lattice-cell volume, so that target and coefficients have the same units, and supply the exact target \(f=G\Gamma\). At full FVM authority, the present blend produces

\[
\Gamma_{new}=f+a(f-Gf),\qquad a=0.8
\]

for the configured amplification cap 1.8. This is one approximate inverse-smoothing correction; it is not generally \(\Gamma\). On a 729-particle, fully three-dimensional smooth test field, direct Gaussian evaluation agrees with the production representation to \(1.74\times10^{-17}\) absolute, but the blend changes the represented target by 6.87% in relative norm. Zero authority preserves it exactly.[^15]

This does not establish a 6.87% cube error, perpetual decay under a fixed independent FVM target, or an error floor under joint refinement. It establishes that an already matched represented state is not preserved by this production operator at finite resolution. The experimental residual-only alternative satisfies the corresponding algebraic check, but its physical reconstruction trials were rejected. Replacing the production formula with that candidate is therefore not an established fix.[^5][^15]

A better transfer must control the represented velocity, its boundary derivatives, and the curl relationship together. Its error should decrease under refinement and remain bounded under repeated exchange. A circulation or first-moment repair can preserve selected integrals while redistributing error spatially; it cannot substitute for those checks.

**Compatible pressure and boundary information.** Incompressibility makes the velocity–pressure problem nonlocal. Errors imposed at a close artificial boundary can alter body pressure and force immediately in the incompressible model. An interface must transmit appropriate normal flux and tangential information while maintaining a consistent pressure projection.

The repository’s boundary-oracle studies are useful because they isolate the small-domain FVM from particle errors. Supplying reference pressure-gradient information improves some oracle results substantially; supplying the present VPM pressure gradient does not consistently improve live coupling. It would be premature to prescribe more pressure data by default. Pressure reconstructed from acceleration must include compatible temporal history, convection, viscous/SGS terms, and the body contribution. A particle replacement at fixed physical time must not be interpreted as physical acceleration.[^16]

Direct pressure transfer is not universally required: an accurate velocity trace with a compatible pressure treatment can also be valid. The requirement is mathematical and discrete boundary consistency, rather than maximizing the number of imposed boundary quantities.

**The same resolved physics.** Today’s LES paths do not simply differ in discretization. The FVM uses the vorticity counterpart of complete viscous stress,

\[
\nabla\times\nabla\cdot(2\nu_{eff}S),
\]

whereas the GBD kernel advances

\[
\nabla\cdot(\nu_{eff}\nabla\omega).
\]

For constant viscosity and incompressible velocity these coincide. For spatially varying eddy viscosity they generally do not. The existing manufactured 3D study finds approximately 20.32% source disagreement with a smooth variable coefficient and 14.95% with its fixed-filter Smagorinsky field; the discrepancy persists under refinement. These are source differences, not drag errors. A complete-stress component candidate converges, but remains outside production GBD.[^17]

This is a necessary repair or explicit modeling decision for consistent LES coupling. It is not the complete explanation of the current errors: matched laminar comparisons also disagree. Filter definitions matter as well. Matching a Smagorinsky constant does not make an anisotropic FVM cell and an isotropic particle represent the same resolved scales. If a future version couples near-wall RANS to wake LES, the conversion from mean modeled stresses to resolved fluctuations needs its own formulation and validation.

**Three-dimensional evolution and time consistency.** Vortex stretching makes 3D qualitatively harder than 2D. Strength orientation, particle overlap, core size, and the velocity gradient must evolve consistently. Choosing the direct rather than transposed stretching form changes conservation properties; it is not merely a way of correcting a derivative implementation. The recorded stretching audit shows that the forms differ substantially on an actual cloud.[^12]

The current tutorial already selects `coupling_scope="fvm_vpm"`, including the accepted body velocity and analytical Jacobian in particle stages. The older omission is therefore not an outstanding defect in that configuration. However, the panel strengths are held between refreshes rather than solved for every temporary RK state. Its time-discretization effect remains part of the full coupled convergence problem.[^18]

Current interface iteration replays the FVM and renewal against a fixed advected VPM predictor. Converging its endpoint normal-velocity and gradient residuals does not recompute the complete particle trajectory with the corrected history. Nor does it establish the temporal order of the hybrid. RK2 particles, backward FVM stepping, endpoint interpolation, diffusion splitting, and renewal must be tested as a combined method.[^19]

The implementation also publishes the final sweep when the iteration cap is reached, with a warning and `converged=false`. Qualification should explicitly fail or reject an interval that misses its declared interface criterion. Application tolerances can be scaled to the actual error budget, but cap exhaustion cannot count as convergence.[^19]

**Resolution, overlap, and budgets.** The interface need not lie outside every separated structure. It must lie where both representations can resolve what crosses it. There must be enough overlap for interpolation, kernel support, advection during exchange, and diffusion. A buffer estimate should consider local transport speed, effective viscosity, and strain; a freestream-only distance is not a general guarantee for recirculating or strongly accelerated flow.

Qualification should vary FVM resolution, particle spacing, core size, overlap width, exchange step, and remeshing/pruning controls with a declared scaling. The current `sigma/h` setting is not a convergence result. A particle scheme avoids the usual fixed-grid advective CFL mechanism, but deformation, stretching, diffusion stability, and interface travel still constrain usable time steps.

For constant viscosity, the conservative vorticity flux through a stationary interface is

\[
F_\omega=(u\cdot n)\omega-(\omega\cdot n)u-\nu\partial_n\omega.
\]

Tracking only the advective term misses stretching transport and diffusion. The repository contains an experimental flux-handoff implementation of this expression, but the current production driver selects volume renewal. The handoff’s one-way release mechanism is not yet a substitute for general two-way recirculating coupling.[^20]

A combined budget should distinguish physical wall production and outer transport from replacement, pruning, and stabilization changes. Net vector strength, circulation through selected surfaces, impulse, mass flux, and momentum/energy balance answer different questions. For a free-space compact vorticity distribution, the impulse volume contribution is \(\rho\sum x_p\times\Gamma_p/2\); body and truncated-domain terms require appropriate treatment in a force balance. Energy and enstrophy need physical source/sink accounting, rather than blanket demands that they remain constant. In 3D, physical stretching can increase enstrophy.

**4. Why the current implementation does not yet save cost**

The numerical partition is less favorable than the geometric picture suggests. In the current cube, the inner FVM retains 43.8% of the full reference’s cell count. Each accepted interval needs three interface sweeps: fifteen PIMPLE advances for five accepted FVM steps. The inner domain is smaller, but it is solved repeatedly.[^3][^19]

Let \(C_N\) be the cost of advancing the inner FVM once over a physical interval, \(m\) the interface sweep count, and \(C_O\) all remaining hybrid work. Then a simple accounting model is

\[
\frac{T_H}{T_F}=m\frac{C_N}{T_F}+\frac{C_O}{T_F}.
\]

The recorded optimized interval contains 34.33 s of FVM work, 1.12 s of particle advancement, 0.82 s of initial boundary evaluation, and 7.25 s of transfer/interface refresh. Dividing the FVM work by three estimates 11.44 s per sweep, or 50.7% of the entire full-FVM interval. **The repeated FVM work alone is 1.52 times the reference cost. Even eliminating all non-FVM work would not make this measured configuration faster.**[^15]

That is a structural diagnosis for this configuration, not a universal lower bound. The estimate assumes comparable cost per sweep; it is not a measured one-sweep run. Timings come from a short startup interval with profiling and concurrent jobs. The reference also receives the optimized FVM kernels, but these runs have not demonstrated equal error or developed-wake throughput.[^3]

Three additional costs deserve explicit attention:

* GBD uses dense Cartesian diffusion arrays. Its allocation can cover the entire VPM domain, and active diffusion visits a rectangular grid extent. Particle sparsity therefore does not automatically imply sparse diffusion cost or memory. Long, thin, moving, or multiple separated wakes need a spatially sparse or adaptive treatment if this overhead becomes dominant.
* The coupler owns VPM on rank zero and gathers global donor velocity/gradient information. This limits the scalability of the coupled workflow even when the FVM scales across ranks.
* Repeated reconstruction, host/device transfer, pruning recovery, and body evaluation add work at every exchange. Fine particle spacing over a volume that the meshed reference can coarsen aggressively may eliminate the expected benefit.[^21]

The most promising routes are reducing the number and cost of *necessary* inner solves, preserving anisotropic near-wall efficiency, improving the interface so that inexpensive exchange is accurate, and retaining efficient particle/diffusion representations as the wake grows. Smaller domains, looser convergence, or weaker wake resolution only constitute improvements if the accuracy target still holds.

For an illustrative 2× speed target, the accounting requirement is \(mC_N/T_F+C_O/T_F\leq0.5\). With three sweeps and non-FVM work equal to 10% of the reference cost, a single inner advance would have to cost at most 13.3% of the full solve. The current estimated 50.7% is far from that regime. The 10% overhead is a hypothetical design budget, not a measured result.

**5. Where it could be a strong candidate**

| Application | Assessment | Decisive condition |
|---|---|---|
| Long rotor/propeller wakes and repeated wake interactions | Strongest potential | Accurate wake transport must dominate the competing mesh cost; moving bodies and wake feedback need qualification. |
| Finite-wing wakes, separated components, remote interference | Plausible | Resolved vortical structures occupy a small fraction of the surrounding domain, and force prediction benefits from the retained near-body CFD. |
| Compact bluff-body flow when only mean forces are required | Uncertain advantage | A well-designed meshed calculation may already be cheap; the coupling needs a measurable benefit at the required force accuracy. |
| Turbulence filling most of the outer domain | Possible physics, weaker cost argument | Particle counts, diffusion grids, and stretching work can lose the sparse-wake advantage. |
| Strongly confined/internal flows or many closely packed walls | Poor fit for the present architecture | Much of the fluid needs boundary-aware treatment, leaving little inexpensive outer region. |
| Compressible shocks, acoustic propagation, variable-density or multiphase flow | Outside the current formulation | Additional governing variables and interface physics would be required. An incompressible vortex cloud does not carry these automatically. |

This application ranking is an analytical judgment based on the equations and cost structure, not a published performance ranking. Even in the favorable category, a mature meshed solver may remain preferable for a particular geometry, required uncertainty, or hardware allocation.

**6. What remains to be demonstrated**

The first change should be to the acceptance question. Near-roundoff equality to another discretization is useful for exact replay or deliberately equivalent discrete operators. It is not a sensible general target for different spatial representations, different outer boundary approximations, or developed chaotic flow. A hybrid can disagree with one full-FVM trajectory while being equally accurate; it can also match drag through cancellation while its wake is wrong.

Verification should establish convergence and consistency. Validation should establish agreement with physical evidence at quantified uncertainty. Matching OpenONDA’s own full FVM is a valuable isolation test, but shared FVM errors can survive that comparison. Domain truncation matters too: a finite-box reference and free-space induction with a finite retained particle wake are not automatically the same boundary-value problem.[^22]

The following sequence provides a practical basis for continuing or redirecting development. Proposed numerical targets are project decisions, not universal CFD standards.

| Gate | Required demonstration | Acceptance and decision |
|---|---|---|
| A. Representation and equations | Production transfer on smooth 3D fields and frozen physical snapshots; normal flux, velocity curl, induced velocity/gradient, body correction, and LES source mapping assessed together. | Errors decrease under refinement; already matched fields change only within the declared transfer budget; repeated exchange remains bounded. A candidate must improve advancing results before promotion. |
| B. Interface crossing | A 3D vortex ring or localized divergence-free structure crosses each face, enters, exits, and crosses obliquely; include diffusion and a returning structure. | Compare with standalone VPM and a converged reference. Demonstrate no persistent interface source, unacceptable reflection, or secular budget drift. Use three resolution levels and separate exchange-step refinement. |
| C. Coupled body flow | Matched 3D laminar cube or sphere, followed by a wake that becomes statistically developed. | Demonstrate short-time field convergence and long-time convergence of mean forces, force RMS, shedding frequency, mean velocity deficit, Reynolds stresses, and wake geometry. Use uncertainty intervals and volume/plane evidence beyond two lines. |
| D. Intended application | A finite wing, rotor, or interacting-wake geometry chosen for a realistic use case. | Validate against independent measurements and a qualified meshed reference. Include wall resolution/model, motion, reversal, initialization, and restart requirements. No per-case fitting of force scales or phases. |
| E. Equal-error cost | Accuracy–cost curves for hybrid and a competently configured full mesh over the same physical duration and hardware/resource budget. | Include all sweeps, diffusion, transfer, setup, memory, and output. A repeatable approximately 2× advantage on the intended application would be a reasonable development gate; quantify variability rather than relying on one interval. |

For an initial engineering target, mean loads and shedding frequency within roughly 1–2% of qualified reference values, and selected wake mean/RMS quantities within a few percent, would be reasonable starting points **only where reference and sampling uncertainties are smaller and the application permits them**. Blade–vortex interaction, aeroacoustics, and small drag increments may require substantially tighter and more specific criteria. Near-zero force components should use absolute, dimensionally appropriate tolerances.

The component error budget should be appreciably smaller than the total observable tolerance. Interface iteration should terminate based on its effect on that budget, while nonconverged intervals remain explicit failures. Refinement should separate spatial discretization, LES filtering, time integration, exchange frequency, remeshing, and wake truncation; changing all of them at once cannot diagnose a nonmonotone outcome.

For developed turbulence, confidence intervals need to account for correlated samples and sufficient flow-through or shedding times. Matching one instantaneous profile or one late drag coefficient is insufficient. Existing records already illustrate force-curve crossings and pressure/viscous cancellation that can make an endpoint appear more accurate than the history.[^2]

No universal mathematical proof for all turbulent flows is required before the solver becomes useful. A convincing package would combine consistency identities, measured convergence on smooth problems, stability/budget evidence through repeated interfaces, and independent physical validation in a clearly bounded application class. The presently missing evidence is this combined package, not another large inventory of isolated passing tests.

**7. Recommended development decision**

Continue if the intended product is accurate long-wake external aerodynamics, and make representation/LES consistency and an equal-error cost target the organizing objectives. Preserve the existing matched-cell oracles, source archives, and independent force reconstructions: they are useful foundations. Complete a developed 3D comparison before interpreting small short-run drag errors as maturity.

Prioritize the production transfer and the meaning of its data over adding more boundary-condition variants. In parallel with the numerical design work—not necessarily with additional simulations—use the measured cost model to reject architectures that cannot beat the full mesh even under optimistic non-FVM costs. The current three-sweep, large-inner-mesh configuration falls into that category.

If a compatible scheme converges but still cannot beat a modern meshed reference on a deliberately favorable wake problem, the appropriate conclusion is to narrow or redirect the hybrid effort. The VPM/VLM components can remain useful for less expensive interactional aerodynamics, and the FVM remains useful independently. Conversely, passing the five gates would justify describing the hybrid as a strong candidate within its validated range.

The evidence supports cautious continuation with a narrower objective. It does not support abandoning the physical idea, claiming that a few remaining bugs will certainly unlock it, or treating years of effort as evidence that the necessary advantage must eventually appear.

**Sources and reproducibility**

[^1]: OpenONDA, [Verified simulation improvement and recorded computational cost](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/verified-progress-3d.md). Frozen matched 3D laminar experiments, physical time 0.5–1.5. Local repository evidence.
[^2]: OpenONDA, [Verified 3D wake comparison through physical time 4.0](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/long-wake-prefix-through-four-3d.md). Independent force, profile, and checkpoint checks for the older iterated configuration. Local repository evidence.
[^3]: OpenONDA, [Cube runtime audit, 14 September 2026](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/cube-runtime-2026-09-14.md), and [machine-readable measurements](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/results/cube-runtime-2026-09-14/measurements.json). Profiled startup timings with concurrent jobs; current tutorial, not the matched-laminar campaign.
[^4]: OpenONDA, [unfinished long-wake comparison](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/long-wake-comparison-3d.md). The later raw history, exact snapshot hash, and last-row arithmetic are preserved in [assessment_checks.json](/Users/flaviomartins/OpenONDA/studies/coupler_feasibility/assessment_checks.json). Its completion/status text is not evidence that a process is currently active.
[^5]: OpenONDA, [interface tests](/Users/flaviomartins/OpenONDA/tests/coupler/test_interface_iteration.py), [experimental renewal checks](/Users/flaviomartins/OpenONDA/tests/coupler/test_renewal_fixed_point.py), [native flux-moment checks](/Users/flaviomartins/OpenONDA/tests/coupler/test_native_flux_moments_3d.py), [stress/curl checks](/Users/flaviomartins/OpenONDA/tests/coupler/test_stress_curl_3d.py), and [interpolation qualification](/Users/flaviomartins/OpenONDA/tests/coupler/test_interpolation_qualification.py). All 18 selected checks passed on 14 September 2026. These include component experiments and do not amount to complete coupled validation.
[^6]: J. T. Beale, [A convergent 3-D vortex method with grid-free stretching](https://doi.org/10.1090/S0025-5718-1986-0829616-6), *Mathematics of Computation* 46 (1986), 401–424. The theorem concerns an incompressible inviscid problem without boundaries; the publisher’s abstract bounds the claim used here.
[^7]: A. Palha, L. Manickathan, C. S. Ferreira, G. van Bussel, [A hybrid Eulerian–Lagrangian flow solver](https://arxiv.org/abs/1505.03368), 2015, especially the coupling algorithm and conclusions. [Full manuscript](https://arxiv.org/pdf/1505.03368).
[^8]: P. Billuart, M. Duponcheel, G. Winckelmans, P. Chatelain, [A weak coupling between a near-wall Eulerian solver and a Vortex Particle-Mesh method for the efficient simulation of 2D external flows](https://www.sciencedirect.com/science/article/pii/S0021999122007896), *Journal of Computational Physics* 473 (2023), 111726. Publisher abstract and highlights; DOI 10.1016/j.jcp.2022.111726.
[^9]: R. Pasolari, C. Ferreira, A. van Zuijlen, [Coupling of OpenFOAM with a Lagrangian vortex particle method for external aerodynamic simulations](https://repository.tudelft.nl/file/File_18f02160-c089-48bc-b774-563c9c5e5707), *Physics of Fluids* 35 (2023), 107115. Sections II, III, V, and VI; DOI 10.1063/5.0165878. Full published text in the TU Delft repository.
[^10]: E. J. Alvarez and A. Ning, [Stable Vortex Particle Method Formulation for Meshless Large-Eddy Simulation](https://par.nsf.gov/servlets/purl/10514726), *AIAA Journal* 62(2), 637–656 (2024 issue; online 2023), DOI 10.2514/1.J063045. Abstract-level validation/performance claims, corroborated by the authors’ [publication list](https://flow.byu.edu/publications/) and [solver description](https://flow.byu.edu/FLOWUnsteady/).
[^11]: [Eulerian–Lagrangian coupling method for simulating dynamic evolution of long-trajectory rotor wake](https://www.sciencedirect.com/science/article/pii/S1000936126001731), *Chinese Journal of Aeronautics* (2026), 104236, DOI 10.1016/j.cja.2026.104236. Publisher abstract and conclusions. The detailed algorithm and reported speedups have not been independently reproduced here.
[^12]: OpenONDA, [Stretching and reconstructed vorticity in the fully 3D cube](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/particle-stretching-consistency-3d.md). Qualified saved-state diagnostics; not errors against physical stretching or a converged flow solution.
[^13]: OpenONDA, [Velocity projection and conservative face flux in 3D](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/velocity-projection-and-mass-flux-3d.md), and [Face-flux moment compatibility](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/flux-moment-compatibility-3d.md). Both reference and hybrid controls are included.
[^14]: OpenONDA, [production velocity-trace curl](/Users/flaviomartins/OpenONDA/source/coupler/stable_renewal.py:107), [represented-state blend](/Users/flaviomartins/OpenONDA/source/coupler/stable_renewal.py:467), and [whole-belt renewal](/Users/flaviomartins/OpenONDA/source/coupler/stable_renewal.py:642).
[^15]: New bounded assessment calculations: [reproduction script](/Users/flaviomartins/OpenONDA/studies/coupler_feasibility/check_assessment.py), [numerical results and source hashes](/Users/flaviomartins/OpenONDA/studies/coupler_feasibility/assessment_checks.json). The probe uses fixed positions, no body, no pruning, no FVM, and no time advancement. Cost ratios are recalculated from source [3].
[^16]: OpenONDA, [Small-domain accuracy investigation](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/README.md), [3D cube boundary and pressure experiments](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/cube-3d-findings.md), and [post-replacement boundary history](/Users/flaviomartins/OpenONDA/source/coupler/boundary.py:471). Historical component corrections are not all outstanding defects.
[^17]: OpenONDA, [SGS operator comparison and isolated complete-stress candidate](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/cube-3d-findings.md:249), [production GBD variable-viscosity kernel](/Users/flaviomartins/OpenONDA/source/solvers/vpm/physics/diffusion/grid.py:3074), and [isolated stress/curl implementation](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/stress_curl_3d.py).
[^18]: OpenONDA, [current cube configuration](/Users/flaviomartins/OpenONDA/tutorials/coupled_fvm_vpm/02_cube_flow/setup.py:263), [panel scope contract](/Users/flaviomartins/OpenONDA/source/solvers/vpm/boundary_elements/panels/solver/panel_solver.py:138), and [held-strength gradient evaluation](/Users/flaviomartins/OpenONDA/source/solvers/vpm/boundary_elements/panels/solver/panel_solver.py:1547).
[^19]: OpenONDA, [fixed-predictor interface iteration](/Users/flaviomartins/OpenONDA/source/coupler/interface_iteration.py:63), [coupled advance sequence](/Users/flaviomartins/OpenONDA/source/coupler/solver.py:771), [substep interpolation](/Users/flaviomartins/OpenONDA/source/coupler/boundary.py:754), and [configuration restrictions](/Users/flaviomartins/OpenONDA/source/coupler/config/types.py:194).
[^20]: OpenONDA, [experimental flux handoff](/Users/flaviomartins/OpenONDA/source/coupler/flux_handoff.py). Its constant-viscosity flux formula, inward-flux accounting, and one-way release scope are explicit in the source.
[^21]: OpenONDA, [dense diffusion allocation](/Users/flaviomartins/OpenONDA/source/solvers/vpm/physics/diffusion/grid.py:654), [domain allocation estimate](/Users/flaviomartins/OpenONDA/source/solvers/vpm/physics/diffusion/grid.py:798), [grid diffusion workflow](/Users/flaviomartins/OpenONDA/source/solvers/vpm/physics/diffusion/grid.py:2180), and [global donor collection](/Users/flaviomartins/OpenONDA/source/coupler/solver.py:887). These are scalability considerations, not measured developed-wake bottleneck shares.
[^22]: NASA/NPARC Alliance, [CFD Verification and Validation tutorial](https://www.grc.nasa.gov/www/wind/valid/tutorial/tutorial.html), [grid convergence](https://www.grc.nasa.gov/www/wind/valid/tutorial/spatconv.html), and [validation assessment](https://www.grc.nasa.gov/www/wind/valid/tutorial/valassess), updated 2021. Used for the distinction between numerical verification, physical validation, and uncertainty; the proposed project thresholds in this report are independent recommendations.

To reproduce the bounded assessment probe from the repository root with the OpenONDA environment:

```sh
env PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    python studies/coupler_feasibility/check_assessment.py
```

The selected existing checks were run with:

```sh
env PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1 TI_CPU_MAX_NUM_THREADS=2 \
    python -m pytest -q -p no:cacheprovider \
    tests/coupler/test_interface_iteration.py \
    tests/coupler/test_renewal_fixed_point.py \
    tests/coupler/test_native_flux_moments_3d.py \
    tests/coupler/test_stress_curl_3d.py \
    tests/coupler/test_interpolation_qualification.py
```
