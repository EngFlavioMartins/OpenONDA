# Terminology of the FVM–VPM coupling

This file is the single source of the names, symbols and code identifiers of the
OpenONDA coupler. The thesis (Chapter 4 and every coupling passage), Paper 1 and
the code use exactly these. The two LaTeX projects share the macro file
`coupling_notation.tex`, which defines the symbols below. Approved 26 October 2026.

## Rules

1. One object has one name, one symbol and one code identifier, identical in the
   thesis, the paper and OpenONDA.
2. The solver labels are FVM and VPM, upright. A subscript names the solver that
   owns a domain, surface, step, parameter or operator (`Ω_FVM`, `Δt_VPM`,
   `τ_VPM`). A superscript names the solver that computed a field value
   (`u^VPM`, `ω^FVM`, `Γ^FVM_j`).
3. The superscript slot holds only that solver label, a time level `n`, a
   coupling iteration `(m)`, a Runge–Kutta stage `(r)`, the predictor `*`, or a
   transpose. Stage names, limits and tolerances are never superscripts.
4. Other labels are words or plain abbreviations, upright, in the subscript:
   transfer, wall, inset, remesh, blend, mis, corr, lim, ret, mom, tot. A
   combined subscript lists stage, then component, then index:
   `u^VPM_{corr,n,f}`, `Γ_{lim,j}`.
5. Indices are italic and keep one meaning: `p` particle (`q` the source
   particle in pair sums), `j` lattice node, `f` boundary face, `C` FVM cell,
   `i` stencil cell, `n` coupling time, `m` coupling iteration, `k` FVM
   substep, `ℓ` diffusion substep.
6. A name says what the object is or what the operation changes: the quantity,
   its new value, and where.
7. Circulation is only a line integral of velocity (bound circulation, a planar
   shed vortex). The volume integral of three-dimensional vorticity is the total
   vortex strength `Γ_tot`.
8. Remeshing is the `M4′` scatter onto a lattice; regeneration is the creation
   of particles at lattice nodes after GBD diffusion; vorticity transfer is the
   whole FVM-to-particle operation. "GBD remeshing" is not used.
9. The interface is the outer FVM boundary and the transfer region together,
   never one of them alone.
10. First use in each section gives name and symbol; captions name every symbol
    they show; table headers carry units; values are normalised groups
    (`Δt_VPM U∞/D`).
11. Every name, parameter and symbol is immediately meaningful to a CFD reader:
    stencil, flux, moment, threshold, mask. Closure means only a turbulence
    closure.

## Domains, surfaces and zones

Distances `d` are measured from the transfer surface, positive into the transfer
region. From the wall outward:

| Term | Symbol | Code |
| --- | --- | --- |
| FVM domain | `Ω_FVM` | `fvm_domain_bounds` |
| Particle domain (all fluid) | `Ω_VPM` | VPM `domain_bounds` |
| Body wall | `Σ_wall` | wall |
| Outer FVM boundary | `Σ_FVM` | `outer_boundary_patch` (mesh patch label `numericalBoundary`) |
| Transfer region | `Ω_transfer` | `transfer_region` |
| Transfer surface | `Σ_transfer = ∂Ω_transfer \ Σ_wall` | `transfer_surface` |
| Transfer inset (distance from `Σ_FVM` to `Σ_transfer`) | `L_inset` | `transfer_inset_spacings` (in `h`) |
| Inset band (between `Σ_transfer` and `Σ_FVM`; free particles, no FVM vorticity) | none | `inset_band` |
| Transfer-surface distance | `d` | `transfer_surface_distance` |
| Zero-weight layer, `0 < d ≤ δ_0`, `η = 0` | `δ_0` | `blend_start_depth` |
| Blending ramp, `δ_0 < d < δ_1` | `δ_1` (depth where `η` reaches 1) | `blend_end_depth` |
| Replacement region, `d ≥ δ_1`, `η = 1` (FVM vorticity replaces particle vorticity) | none | `replacement_region` |
| Remeshing region (`Ω_transfer` extended outward by `L_remesh`) | `L_remesh` | `remeshing_extension_length` |
| Signed wall distance (positive in fluid); wall normal into the fluid | `d_wall`; `n_wall` | `wall_distance` |
| Smallest extent of the FVM domain | `L_FVM` | none |
| Total area of `Σ_FVM` | `A_FVM` | none |
| Interface | `Σ_FVM` and `Ω_transfer` together | none |

Defaults: `L_inset = 8.75h`, `δ_0 = 2h`, `δ_1 = 6h`, so the ramp width
`δ_1 − δ_0` is `4h`.

## Time integration and coupling iteration

| Term | Symbol | Code |
| --- | --- | --- |
| FVM time step | `Δt_FVM` | FVM `time_step_size` |
| Coupling interval, equal to the VPM time step | `Δt_VPM` | `coupling_time_step_size` |
| FVM substeps per coupling interval | `N` | `n_fvm_substeps` |
| Coupling time; coupling step | `t^n`; `n` | `coupling_time(step)` |
| Elapsed fraction of the interval at substep `k` | `θ_k = k/N` | none |
| Coupling intervals per convective time | `D/(U∞ Δt_VPM)` | none |
| Lagrangian CFL number (strain), the `LS_str` of thesis Ch. 3 | `Λ = Δt_VPM ‖S‖∞` | `lagrangian_cfl` |
| VPM predictor | `𝒫*` | `predictor` |
| FVM and VPM states at `t^n` | `ℱ^n`, `𝒫^n` | none |
| Coupling iteration; its index; maximum number | `m`, `M` | `max_coupling_iterations` |
| Boundary data | `b` | `BoundaryData` |
| Start-of-interval data | `b^n` | none |
| End-of-interval estimate | `b^(m)` | `estimate` |
| Substep data | `b_k^(m)` | none |
| Previous-interval correction | `Δb^n` | `boundary_data_correction` |
| Normal-velocity residual; tolerance | `r_{u_n}`; `ε_{u_n}` | `normal_velocity_residual`; `normal_velocity_tolerance` |
| Normal-gradient residual; tolerance | `r_{g_t}`; `ε_{g_t}` | `normal_gradient_residual`; `normal_gradient_tolerance` |
| FVM advance over one interval | `𝒜_FVM` | none |
| Boundary-data operator | `ℬ` | none |
| Vorticity-transfer operator | `𝒯` | none |
| Accepted state (kept whether or not converged) | none | accepted |

## Boundary data

| Term | Symbol | Code |
| --- | --- | --- |
| Particle velocity; particle velocity gradient | `u^VPM`; `G^VPM` | none |
| FVM velocity, velocity gradient, vorticity | `u^FVM`, `G^FVM`, `ω^FVM` | none |
| Normal and tangential velocity; tangential projector | `u_n`, `u_t`; `P_t = J − n nᵀ` | none |
| Normal gradient of the tangential velocity (Neumann data) | `g_t^VPM = P_t G^VPM n` | `tangential_velocity_normal_gradient` |
| Mixed boundary condition: boundary operator; data | `𝒞`; `b` | FVM velocity boundary type `mixed_velocity` |
| Net volume flux through `Σ_FVM` | `q_net` | none |
| Flux-corrected normal velocity | `u^VPM_{corr,n,f}` | none |
| Net-flux tolerance | `(h/L_FVM)²` | `boundary_flux_tolerance` |
| Flux-consistent pressure: normal pressure gradient at face `f`; pressure-free flux; coefficient | `(∂p/∂n)_f`; `φ^H_f`; `D_{n,f}` | `fixedFluxPressure` |

`g_t` is the normal gradient of the tangential velocity, never a "tangential
gradient" (a derivative along the surface).

## Vorticity transfer

In the order of the operation:

| Term | Symbol | Code |
| --- | --- | --- |
| Vorticity transfer (the whole operation) | `𝒯` | `VorticityTransfer`, `transfer_vorticity` |
| Transfer lattice: spacing, node, lattice cube | `h`, `x_j`, `Ω_j` | `TransferLattice` |
| Transfer-lattice particles | none | `transfer_lattice_particles` |
| Blending weight | `η(x)` | `blending_weight` |
| Remeshed strengths | `Γ^VPM_j` | `remeshed_strength` |
| Representation operator; particle-represented vorticity | `ℳ_σ`; `ℳ_σ[Γ^VPM]` | `gaussian_representation` |
| FVM target strength (Stokes integral) | `Γ^FVM_j` | `fvm_target_strength` |
| Reconstructed FVM velocity; interpolation stencil; stencil cell | `ũ^FVM`; `𝒮(x)`; `i` | `interpolation_stencil` |
| No-slip extension | `s(ξ)` | none |
| Blended target | `Γ_blend` | `blended_target` |
| Weighted mismatch | `Γ_mis` | `weighted_mismatch` |
| Strength correction; correction gain | `δΓ_corr`; `γ = min(κ_max − 1, 1)` | `strength_correction`; `correction_gain` |
| Update fraction; limited strengths | `ϑ_j`; `Γ_{lim,j}` | `limited_strength` |
| Peak target strength | `Γ_peak = max_j ‖Γ_{blend,j}‖` | none |
| Strength ratio; its limit | `κ = max_j ‖Γ_{lim,j}‖ / Γ_peak`; `κ_max` | `strength_ratio`; `max_strength_ratio` |
| Removal threshold (a vorticity): local, interior, GBD | `τ(x)`, `τ_transfer`, `τ_VPM` | `transfer_removal_threshold` |
| Strength threshold | `Γ_{τ,j} = h³ τ(x_j)` | none |
| Weak-strength removal (smooth thresholding) | none | `remove_weak_strengths` |
| Retained strengths | `Γ_{ret,j}` | `retained_strength` |
| Moment correction (restores total strength and linear impulse); its limit | `δΓ_{mom,j}`; 8 % of the transferred L1 norm | `max_moment_correction_fraction` |
| Total vortex strength; linear and angular impulse per unit density | `Γ_tot`; `I`, `A` | `total_vortex_strength`, `linear_impulse`, `angular_impulse` |

## Particle operators next to the wall

| Term | Symbol | Code |
| --- | --- | --- |
| Wall projection (of a particle that crosses the wall) | `δx_p` | `wall_projection` |
| Fluid-side remeshing stencil of particle `p` (fluid nodes reached without crossing the wall) | `𝒮_p` | `fluid_side_stencil` |
| Deposition weights | `w_{jp}` | none |
| Diffusion-flux mask (1 for a fluid link, 0 when the link touches the solid or crosses the wall) | `𝟙_{jk}` | `diffusion_flux_mask` |
| Nodes connected through the fluid (support of the GBD moment correction) | none | `fluid_connected_nodes` |

## Particles, errors and flow scales

| Term | Symbol |
| --- | --- |
| Particle strength (volume-integrated vorticity, m³/s) | `α_p` |
| Lattice-node strength | `Γ_j` |
| Core radius; particle spacing | `σ`; `h` |
| Error split of a coupled statistic `Q` | `e = e_FVM + e_coupling` |
| FVM-only value extrapolated to zero cell size and step; its uncertainty | `Q_ext`; `U_ext` |
| Vortex-formation length (centreline peak of `u_rms`) | `L_f` |
| Vorticity thickness of the separated shear layer | `δ_ω` |
| Lateral clearance: body surface to the side faces of the replacement region | `ℓ_side` |
| Wake reach: formation point to the downstream transfer surface | `ℓ_wake` |

`η(x)` is the blending weight; `η_σ` (thesis Ch. 3) is a diffusion kernel and
`η_K` the Kolmogorov length.

## Code configuration (`CouplerConfig`)

| Field | Meaning | Default |
| --- | --- | --- |
| `freestream_velocity` | Free-stream velocity in m/s | (1, 0, 0) |
| `transfer_region_bounds` | Transfer-region bounds in m, or None for the inset | None |
| `transfer_inset_spacings` | `L_inset / h` | 8.75 |
| `blend_start_depth` | `δ_0` in m, or None for `2h` (zero when `blend_end_depth` is zero) | None |
| `blend_end_depth` | `δ_1` in m, or None for `6h`; zero disables blending | None |
| `transfer_removal_threshold` | `τ_transfer` in 1/s | 0.05 |
| `max_strength_ratio` | `κ_max` | 1.8 |
| `max_moment_correction_fraction` | moment-correction limit | 0.08 |
| `transfer_diagnostic_interval_steps` | accepted transfers between full diagnostics | 1 |
| `outer_boundary_patch` | mesh patch of `Σ_FVM` | `numericalBoundary` |
| `max_coupling_iterations` | `M` | 6 |
| `normal_velocity_tolerance` | `ε_{u_n}` in m/s | 1e-5 |
| `normal_gradient_tolerance` | `ε_{g_t}` in 1/s | 1e-5 |

Constants: `TRANSFER_INSET_SPACINGS = 8.75`, `BLEND_START_SPACINGS = 2`,
`BLEND_END_SPACINGS = 6`. `CouplerConfig.blend_depths(h)` returns `(δ_0, δ_1)` in m.

## Code modules and identifiers

| Module | Contents |
| --- | --- |
| `coupler/transfer_lattice.py` | `TransferLattice`, `build_transfer_lattice`, `blending_weight(positions, transfer_region_bounds, blend_start_depth, blend_end_depth)`, `fvm_target_strength`, `scatter_m4_prime_to_lattice`, `gaussian_representation`, `correct_lattice_strength` → `StrengthCorrection` (`limited_strength`, `blended_target`, `represented_strength`, `strength_ratio`), `remove_weak_strengths`, `redistribute_removed_strength`, `VortexMoments`, `vortex_moments`, `correct_vortex_moments` (moment correction), `vortex_moment_residual`, `transfer_on_lattice` → `LatticeTransferResult` |
| `coupler/m4_kernel.py` | `m4_prime` |
| `coupler/vorticity_transfer.py` | `VorticityTransfer`, `transfer_vorticity_to_particles`, `remeshing_extension_length`, `TransferResult` |
| `coupler/coupling_iteration.py` | `advance_coupling_iterations`, `_first_estimate`, `BoundaryData`, `CouplingIterationRow` |
| `coupler/interpolation.py` | `FVMVelocityInterpolator`, `InterpolationStencil` |
| `coupler/boundary.py` | `tangential_velocity_normal_gradient`, `boundary_flux_tolerance`, `evaluate_vpm_boundary` |
| `coupler/solid.py` | `WallProjection` |
| `coupler/geometry.py` | `segments_cross_wall` (whether a segment crosses the wall) |
| `fvm/fields/mixed_velocity_boundary.py` | `update_mixed_velocity_boundary`, `reconstruct_mixed_velocity_boundary`; set by `FVMSolver.set_mixed_velocity_boundary_condition`; patch fields `normal_velocity_field`, `tangential_velocity_normal_gradient_field` |
| `coupler/solver.py` | `CoupledSolver.coupling_time(step)`, `boundary_data_correction`, `fvm_domain_bounds` |

Lattice fields: `transfer_region_bounds`, `remeshing_region_bounds`,
`remeshing_extension_length` (`L_remesh`), `blending_weight` (`η` per node),
`wall_crossing_links` (lattice links that cross the wall; the diffusion-flux mask
is their complement). Strength thresholds: `strength_threshold = h³ τ_transfer` in
the replacement region and `surface_strength_threshold`, the absolute GBD strength
threshold, at the transfer surface.
The VPM GBD diagnostic is `last_gbd_moment_correction`.

## Retired terms

These words and symbols do not appear in coupling text, code or figures.

| Retired | Use |
| --- | --- |
| renewal, renew, renewed particles | vorticity transfer; transfer-lattice particles |
| exchange interval, exchange step, exchange (for one interval) | coupling interval, coupling step |
| trace, boundary trace, interface trace, velocity trace | boundary data; reconstructed FVM velocity |
| sweep, pass, interface iteration | coupling iteration |
| endpoint estimate, candidate | end-of-interval estimate |
| final (for the accepted state) | accepted |
| coupling boundary, numerical boundary (in prose), box faces | outer FVM boundary `Σ_FVM` |
| FVM core, full-FVM region, unblended region | replacement region |
| particle-only layer, free-particle band, VPM-only width, dead zone | zero-weight layer, `δ_0` |
| `δ_buf`, `δ_blend`, blend width | `δ_1` |
| transfer box, renewal buffer, renewal belt, release buffer, `L_buf` | remeshing region, `L_remesh` |
| `L_T` | `L_inset` |
| `Ω_F`, `Ω_P`, `Ω_T`, `Σ_F`, `Σ_T`, `Σ_b`, `Δt_F`, `Δt_P`, `A_F`, `L_F`, `L_box` | `Ω_FVM`, `Ω_VPM`, `Ω_transfer`, `Σ_FVM`, `Σ_transfer`, `Σ_wall`, `Δt_FVM`, `Δt_VPM`, `A_FVM`, `L_FVM`, `L_FVM` |
| `u^P`, `u^F`, `G^P`, `Γ^P`, `Γ^F`, `𝒜_F`, `ℬ_F`, `ℬ_FVM`, `𝒞_F`, `𝒞_FVM`, `ℛ` | `u^VPM`, `u^FVM`, `G^VPM`, `Γ^VPM`, `Γ^FVM`, `𝒜_FVM`, `ℬ`, `ℬ`, `𝒞`, `𝒞`, `𝒯` |
| `a_cap`, amplification cap, strength correction factor | `κ_max`, strength-ratio limit; `κ`, strength ratio |
| `Γ_phys`, physical target | `Γ_blend`, blended target |
| `Γ_R` | `Γ_mis` |
| `δΓ_T` | `δΓ_corr` |
| `Γ_A`, `Γ^new`, bounded coefficients | `Γ_lim`, limited strengths |
| `Γ′_j` | `Γ_{ret,j}` |
| `T_j` | `Γ_{τ,j}` |
| `τ_T`, `τ_P`, transfer vorticity cutoff | `τ_transfer`, `τ_VPM`, removal threshold |
| `r_n`, `r_g`, `r_{∂u_t}`, `ε_n`, `ε_g`, `r^max` | `r_{u_n}`, `r_{g_t}`, `ε_{u_n}`, `ε_{g_t}` |
| tangential gradient (for `g_t`), tangential Neumann data | normal gradient of the tangential velocity |
| `e_F`, `e_C`, `ε_FVM`, `ε_coupling` | `e_FVM`, `e_coupling` |
| `L_form` | `L_f` |
| `n_R`, `N_x` | `D/(U∞ Δt_VPM)` |
| Lagrangian step number (for the coupling limit) | Lagrangian CFL number `Λ` |
| donor, donor cell, donor set `𝒟(x)` | interpolation stencil `𝒮(x)`, stencil cell |
| visible node set `𝒱_p`, visible support, wall-visible support | fluid-side remeshing stencil `𝒮_p`; nodes connected through the fluid |
| open-link indicator, blocked link | diffusion-flux mask `𝟙_{jk}`; zero diffusive flux across the wall |
| represented field, Gaussian-represented field | particle-represented vorticity |
| pruning, soft pruning, garrote shrinkage | weak-strength removal, smooth thresholding |
| closure correction, budget closure, moment recovery, circulation and impulse restoration, vortex invariants | moment correction `δΓ_mom`; total vortex strength and impulse |
| released (to the VPM), release | evolve under the VPM alone; left free |
| authority, owned, hand-off, handoff | state which strengths change and where |
| safeguard | name the check |
| provenance | which solver computed the value |
| population (of particles) | particle set |
| Lagrangian–Eulerian transfer | vorticity transfer |
| buffered lattice | transfer lattice extended by `L_remesh` |

## Stored data written before 26 October 2026

Diagnostics keys were renamed without aliases. Keys of `coupler_diagnostics.jsonl`:

| Old | New |
| --- | --- |
| `fvm_boundary_trace` | `fvm_boundary_data` |
| `boundary_normal_velocity` | `transfer_surface_normal_velocity` |
| `vortex_line_closure` | `transfer_surface_normal_vorticity` |
| `gbd_moment_recovery` (with `pruned_node_count`) | `gbd_moment_correction` (with `removed_node_count`) |
| `interface_iteration` | `coupling_iteration` |
| `interface_iteration.sweeps` | `coupling_iteration.iterations` |
| `interface_iteration.initial_guess` (`vpm_trace`, `previous_correction`) | `coupling_iteration.first_estimate` (`vpm_predictor`, `previous_correction`) |
| rows `sweep`, `normal_residual_rms`, `gradient_residual_rms` | `iteration`, `normal_velocity_residual`, `normal_gradient_residual` |
| `timing_seconds.last_sweep_donor_gather` | `timing_seconds.last_transfer_fvm_gather` |
| `transfer.transfer_method` | dropped |
| `transfer.eta_blending_enabled` | `transfer.blending_enabled` |
| `transfer.maximum_transfer_amplification` | `transfer.strength_ratio` |
| `transfer.representation_residual_{before,after}_prune` | `transfer.representation_residual_{before,after}_removal` |
| `transfer.population_renewal_*` | `transfer.particle_set_transfer_*` |
| `transfer.population_pruned_*` | `transfer.particle_set_removed_*` |
| `transfer.renewal_*`, `transfer.renewed_*` | `transfer.transfer_*` |
| `transfer.pruned_*` | `transfer.removed_*` |

Keys of `coupler_metadata.json`: the `configuration` fields follow the
`CouplerConfig` table (`coupling_patch` → `outer_boundary_patch`, `eta_blend_width` →
`blend_end_depth`, `vpm_only_width` → `blend_start_depth`, `interface_iterations` →
`max_coupling_iterations`, `interface_normal_tolerance` → `normal_velocity_tolerance`,
`interface_gradient_tolerance` → `normal_gradient_tolerance`,
`transfer_amplification_cap` → `max_strength_ratio`,
`transfer_discretization_error_limit` → `max_moment_correction_fraction`,
`transfer_vorticity_cutoff` → `transfer_removal_threshold`), and `coupling.fvm_box`
became `coupling.fvm_domain_bounds`.

Columns written by the cylinder tutorial's `assets/error_analysis.py` and
`assets/plot_error_analysis.py` (`figures/auxiliary/*.csv`): `exchanges per convective
time` → `coupling intervals per convective time`, `FVM steps per exchange` → `FVM
substeps per coupling interval`, `stability number` → `Lagrangian CFL number`,
`interface sweeps` → `coupling iterations`, `trace velocity mismatch mean|max` →
`boundary velocity mismatch mean|max`, `first-sweep normal residual` →
`first-iteration normal-velocity residual`, `first-sweep gradient residual` →
`first-iteration normal-gradient residual`, `last-sweep gradient residual` →
`last-iteration normal-gradient residual`, `e_c <quantity> [%]` → `e_coupling
<quantity> [%]`, `|e_c| <quantity> [%]` → `|e_coupling| <quantity> [%]`; run parameters
`interface_sweeps` → `coupling_iterations`, `mean sweeps` → `mean coupling iterations`,
`stability_number` → `lagrangian_cfl`. The tutorial command-line flags
`--interface-iterations`, `--interface-tolerance` became `--coupling-iterations`,
`--coupling-tolerance`.

Coupled backups are at `BACKUP_FORMAT_VERSION` 15 and boundary schema 5; the
boundary-data keys `tangential_gradient`, `has_tangential_gradient` became
`normal_gradient`, `has_normal_gradient`.

`python scripts/convert_coupler_diagnostics.py <case>/solution [...]` converts an older
run in place: it renames the keys of `coupler_diagnostics.jsonl` and
`coupler_metadata.json`, and renames the boundary data and the configuration of every
coupled backup and renews its digests, so that the run restarts with the current coupler.
