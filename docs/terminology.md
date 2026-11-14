# Terminology of the FVM–VPM coupling

The names, symbols and code identifiers of the OpenONDA coupler. The code, its
messages, the coupled tutorials and the publications on the method use these.

## Notation

1. The solver labels are FVM and VPM, upright. A subscript names the solver that
   owns a domain, surface, step, parameter or operator (`Ω_FVM`, `Δt_VPM`,
   `τ_VPM`). A superscript names the solver that computed a field value
   (`u^VPM`, `ω^FVM`, `Γ^FVM_j`).
2. The superscript slot holds only that solver label, a time level `n`, a
   coupling iteration `(m)`, a Runge–Kutta stage `(r)`, the predictor `*`, or a
   transpose. Stage names, limits and tolerances are never superscripts.
3. Other labels are words or plain abbreviations, upright, in the subscript:
   transfer, wall, inset, remesh, blend, mis, corr, lim, ret, mom, tot. A
   combined subscript lists stage, then component, then index:
   `u^VPM_{corr,n,f}`, `Γ_{lim,j}`.
4. Indices are italic and keep one meaning: `p` particle (`q` the source
   particle in pair sums), `j` lattice node, `f` boundary face, `C` FVM cell,
   `i` stencil cell, `n` coupling time, `m` coupling iteration, `k` FVM
   substep, `ℓ` diffusion substep.
5. Circulation is only a line integral of velocity (bound circulation, a planar
   shed vortex). The volume integral of three-dimensional vorticity is the total
   vortex strength `Γ_tot`.
6. Remeshing is the `M4′` scatter onto a lattice; regeneration is the creation
   of particles at lattice nodes after GBD diffusion; vorticity transfer is the
   whole FVM-to-particle operation. "GBD remeshing" is not used.
7. The interface is the outer FVM boundary and the transfer region together,
   never one of them alone.

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
| Transfer surface | `Σ_transfer = ∂Ω_transfer \ Σ_wall` | none |
| Transfer inset (distance from `Σ_FVM` to `Σ_transfer`) | `L_inset` | `transfer_inset_spacings` (in `h`) |
| Inset band (between `Σ_transfer` and `Σ_FVM`; free particles, no FVM vorticity) | none | none |
| Transfer-surface distance | `d` | none |
| Zero-weight layer, `0 < d ≤ δ_0`, `η = 0` | `δ_0` | `blend_start_depth` |
| Blending ramp, `δ_0 < d < δ_1` | `δ_1` (depth where `η` reaches 1) | `blend_end_depth` |
| Replacement region, `d ≥ δ_1`, `η = 1` (FVM vorticity replaces particle vorticity) | none | none |
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
| Flux-consistent pressure: normal pressure gradient at face `f`; pressure-free flux; coefficient | `(∂p/∂n)_f`; `φ^H_f`; `D_{n,f}` | `fixed_flux_pressure` |

`g_t` is the normal gradient of the tangential velocity, never a "tangential
gradient" (a derivative along the surface).

## Vorticity transfer

In the order of the operation:

| Term | Symbol | Code |
| --- | --- | --- |
| Vorticity transfer (the whole operation) | `𝒯` | `VorticityTransfer`, `transfer_vorticity_to_particles` |
| Transfer lattice: spacing, node, lattice cube | `h`, `x_j`, `Ω_j` | `TransferLattice` |
| Transfer-lattice particles | none | none |
| Blending weight | `η(x)` | `blending_weight` |
| Remeshed strengths | `Γ^VPM_j` | none |
| FVM target strength (Stokes integral) | `Γ^FVM_j` | `fvm_target_strength` |
| Reconstructed FVM velocity; interpolation stencil; stencil cell | `ũ^FVM`; `𝒮(x)`; `i` | none |
| No-slip extension | `s(ξ)` | none |
| Node strengths, blended | `Γ_j = Γ^VPM_j + η (Γ^FVM_j − Γ^VPM_j)` | `blend_lattice_strength` |
| Removal threshold (a vorticity): local, interior, GBD | `τ(x)`, `τ_transfer`, `τ_VPM` | `transfer_removal_threshold` |
| Strength threshold | `Γ_{τ,j} = h³ τ(x_j)` | none |
| Weak-strength removal (smooth thresholding) | none | `remove_weak_strengths` |
| Retained strengths | `Γ_{ret,j}` | `retained_vortex_strength` |
| Moment correction (restores total strength and linear impulse); its limit | `δΓ_{mom,j}`; 8 % of the transferred L1 norm | `max_moment_correction_fraction` |
| Total vortex strength; linear and angular impulse per unit density | `Γ_tot`; `I`, `A` | `total_vortex_strength`, `linear_impulse`, `angular_impulse` |

## Particle operators next to the wall

| Term | Symbol | Code |
| --- | --- | --- |
| Wall projection (of a particle that crosses the wall) | `δx_p` | `WallProjection` |
| Fluid-side remeshing stencil of particle `p` (fluid nodes reached without crossing the wall) | `𝒮_p` | none |
| Deposition weights | `w_{jp}` | none |
| Diffusion-flux mask (1 for a fluid link, 0 when the link touches the solid or crosses the wall) | `𝟙_{jk}` | none |
| Nodes connected through the fluid (support of the GBD moment correction) | none | none |

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
| `freestream_velocity` | Free-stream velocity in m/s; equal to the VPM case's | (1, 0, 0) |
| `freestream` | Optional free-stream velocity history (`VelocityRamp`) | None |
| `transfer_region_bounds` | Transfer-region bounds in m, or None for the inset | None |
| `transfer_inset_spacings` | `L_inset / h` | 8.75 |
| `blend_start_depth` | `δ_0` in m, or None for `2h` (zero when `blend_end_depth` is zero) | None |
| `blend_end_depth` | `δ_1` in m, or None for `6h`; zero disables blending | None |
| `transfer_removal_threshold` | `τ_transfer` in 1/s | 0.05 |
| `max_moment_correction_fraction` | moment-correction limit | 0.08 |
| `transfer_diagnostic_interval_steps` | accepted transfers between two transfer log entries | 10 |
| `outer_boundary_patch` | mesh patch of `Σ_FVM` | `numericalBoundary` |
| `max_coupling_iterations` | `M` | 6 |
| `normal_velocity_tolerance` | `ε_{u_n}` in m/s | 1e-5 |
| `normal_gradient_tolerance` | `ε_{g_t}` in 1/s | 1e-5 |
| `boundary_mismatch_warning` | tangential FVM–particle mismatch on `Σ_FVM`, in units of the reference speed, above which a warning names the face; None disables it | 0.2 |
| `boundary_mismatch_limit` | mismatch above which the run stops naming the face; None disables it | 1.0 |
| `backup` | coupled restart backups (`Backup(schedule, write_at_end)`) | `Backup()` |

Constants: `TRANSFER_INSET_SPACINGS = 8.75`, `BLEND_START_SPACINGS = 2`,
`BLEND_END_SPACINGS = 6`. `CouplerConfig.blend_depths(h)` returns `(δ_0, δ_1)` in m.

## Code modules and identifiers

| Module | Contents |
| --- | --- |
| `coupler/transfer_lattice.py` | `TransferLattice`, `build_transfer_lattice`, `blending_weight(positions, transfer_region_bounds, blend_start_depth, blend_end_depth)`, `fvm_target_strength`, `scatter_m4_prime_to_lattice`, `blend_lattice_strength`, `remove_weak_strengths`, `redistribute_removed_strength`, `VortexMoments`, `vortex_moments`, `correct_vortex_moments` (moment correction), `vortex_moment_residual`, `transfer_on_lattice` → `LatticeTransferResult` |
| `core/m4_kernel.py` | `m4_prime`, `m4_prime_scalar`, `m4_prime_ti`, `m4_axis_weights` |
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
