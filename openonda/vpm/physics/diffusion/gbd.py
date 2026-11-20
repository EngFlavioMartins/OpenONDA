"""Gaussian blob diffusion (GBD) with particle regeneration (Cottet & Koumoutsakos 2000).

The vortex strength is remeshed onto the lattice with the M4' kernel (or six-point
Lagrange), advanced by stable explicit stages of the 7-point Laplacian, thresholded, and
the surviving nodes become the new particles. Weak-node removal keeps the moments of the
diffused field (``moment_correction.py``); a solid body zeroes the grid inside it and
cuts the Laplacian links across its wall (``wall.py``); a slip slab adds the image fields
of its two planes.
"""

import math
from typing import TYPE_CHECKING

import numpy as np
import taichi as ti

from openonda.core.errors import NonFiniteSolutionError
from openonda.core.m4_kernel import m4_prime_ti
from openonda.core.slip_slab_images import even_slab_field, slab_image_lattice

from .lattice import empty_gbd_moment_correction, empty_gbd_wall_transfer
from .moment_correction import (
    GBDMomentCorrectionError,
    augment_moment_correction_support,
    redistribute_removed_moments,
    retain_diffused_component,
)
from .regeneration import (
    build_particle_arrays,
    nearest_node_mapping,
    read_diffusion_sources,
    scatter_scalar_weighted,
    select_regenerated_nodes,
)

if TYPE_CHECKING:
    from ...particles.container import Particles
    from ...turbulence.spectral_cusp import SpectralCuspClosure
    from .lattice import DiffusionLattice

# Bound the 64 atomic deposits per particle to keep GPU dispatches short.
_M4_SCATTER_BATCH_SIZE = 4096


def _explicit_diffusion_substep_count(
    max_diffusivity: float,
    time_step_size: float,
    particle_spacing: float,
) -> tuple[int, float]:
    """Return stable forward-Euler stages for the 3-D 7-point Laplacian."""
    values = np.asarray([max_diffusivity, time_step_size, particle_spacing], dtype=np.float64)
    if not np.isfinite(values).all():
        raise NonFiniteSolutionError(
            "GBD diffusion requires finite diffusivity, time step, and grid spacing."
        )
    if max_diffusivity < 0.0:
        raise ValueError("GBD diffusion requires non-negative effective viscosity.")
    if time_step_size < 0.0:
        raise ValueError("GBD diffusion requires a non-negative time step.")
    if particle_spacing <= 0.0:
        raise ValueError("GBD diffusion requires a positive particle spacing.")

    max_diffusion_number = float(max_diffusivity * time_step_size / particle_spacing**2)
    if not np.isfinite(max_diffusion_number):
        raise NonFiniteSolutionError("GBD diffusion produced a non-finite Fourier number.")
    if max_diffusion_number == 0.0:
        return 1, 0.0

    # The most negative eigenvalue of the dimensionless 3-D 7-point
    # Laplacian is -12.  Requiring alpha < 1/12 keeps every modal
    # amplification in (0, 1], so diffusion cannot reverse a grid mode.
    scaled = 12.0 * max_diffusion_number
    if not np.isfinite(scaled):
        raise FloatingPointError("GBD diffusion requires an unrepresentable stage count.")
    n_diffusion_substeps = max(1, math.floor(scaled) + 1)

    # Kernels run in f32 in production.  Guard against a nominally safe
    # host-side value rounding back onto or above the stability boundary.
    diffusivity_f32 = np.float32(max_diffusivity)
    spacing_f32 = np.float32(particle_spacing)
    spacing_sq_f32 = np.float32(spacing_f32 * spacing_f32)
    if not np.isfinite(diffusivity_f32) or not np.isfinite(spacing_sq_f32):
        raise FloatingPointError("GBD diffusion inputs exceed the production precision.")
    if spacing_sq_f32 == 0.0:
        raise FloatingPointError("GBD grid spacing underflows the production precision.")
    while True:
        substep_f32 = np.float32(time_step_size / n_diffusion_substeps)
        variable_alpha_f32 = np.float32(np.float32(diffusivity_f32 * substep_f32) / spacing_sq_f32)
        scalar_alpha_f32 = np.float32(
            max_diffusivity * (time_step_size / n_diffusion_substeps) / particle_spacing**2
        )
        if max(float(variable_alpha_f32), float(scalar_alpha_f32)) < 1.0 / 12.0:
            break
        n_diffusion_substeps += 1
    return n_diffusion_substeps, max_diffusion_number


def gbd_diffusion_substep_count(
    max_diffusivity: float,
    time_step_size: float,
    particle_spacing: float,
) -> int:
    """Return the production GBD stage count for the supplied runtime state."""
    return _explicit_diffusion_substep_count(
        max_diffusivity,
        time_step_size,
        particle_spacing,
    )[0]


def _advance_gbd_laplacian(
    lattice: "DiffusionLattice",
    *,
    nx: int,
    ny: int,
    nz: int,
    time_step_size: float,
    particle_spacing: float,
    kinematic_viscosity: float,
    effective_viscosity_grid: np.ndarray | None,
) -> tuple[int, float]:
    """Advance the frozen GBD grid operator with stable explicit stages."""
    if effective_viscosity_grid is None:
        max_diffusivity = float(kinematic_viscosity)
    else:
        max_diffusivity = (
            float(np.max(effective_viscosity_grid)) if effective_viscosity_grid.size else 0.0
        )

    n_diffusion_substeps, max_diffusion_number = _explicit_diffusion_substep_count(
        max_diffusivity,
        time_step_size,
        particle_spacing,
    )
    diffusion_substep_size = float(time_step_size) / n_diffusion_substeps
    max_substep_diffusion_number = max_diffusion_number / n_diffusion_substeps

    if n_diffusion_substeps > 1:
        lattice.logger.section(
            "gaussian blob diffusion substepping",
            ("substeps", f"{n_diffusion_substeps:,}"),
            ("diffusion number, max", f"{max_diffusion_number:.6f}"),
            ("diffusion number, max per substep", f"{max_substep_diffusion_number:.6f}"),
            ("substep size", f"{diffusion_substep_size:.6e}", "s"),
        )

    lattice.zero_grid_kernel(lattice.other_grid, nx, ny, nz)
    if effective_viscosity_grid is not None:
        lattice.upload_active_scalar_grid(
            lattice.effective_viscosity_grid, effective_viscosity_grid, nx, ny, nz
        )

    for _ in range(n_diffusion_substeps):
        if effective_viscosity_grid is None:
            diffusion_number = float(
                kinematic_viscosity * diffusion_substep_size / particle_spacing**2
            )
            lattice.laplacian_step_gpu_kernel(
                lattice.current_grid,
                lattice.other_grid,
                lattice.body_mask_grid,
                diffusion_number,
                nx,
                ny,
                nz,
            )
        else:
            lattice.laplacian_step_variable_gpu_kernel(
                lattice.current_grid,
                lattice.other_grid,
                lattice.effective_viscosity_grid,
                lattice.body_mask_grid,
                diffusion_substep_size,
                particle_spacing,
                nx,
                ny,
                nz,
            )
        lattice.ping = not lattice.ping

    return n_diffusion_substeps, max_diffusion_number


def gbd_diffusion(
    lattice: "DiffusionLattice",
    particles: "Particles",
    time_step_size: float,
    particle_spacing: float,
    kinematic_viscosity: float,
    domain_padding: float = 3.0,
    regen_threshold: float = 0.01,
    regen_threshold_mode: str = "budget",
    effective_viscosity: np.ndarray | None = None,
    remeshing_kernel: str = "m4_prime",
    wall_fallback_limit: float = 0.0,
    spectral_closure: "SpectralCuspClosure | None" = None,
) -> dict[str, np.ndarray] | None:
    """GBD diffusion + particle regeneration (Cottet & Koumoutsakos 2000).

    A ``spectral_closure`` damps the diffused lattice in Fourier space before the
    regeneration (:class:`~openonda.vpm.turbulence.spectral_cusp.SpectralCuspClosure`);
    it needs a lattice without walls and slip planes.

    Moment-preserving scatter → explicit 7-point Laplacian → weak-strength
    removal → spawn. Per-particle ``effective_viscosity`` (ν+ν_t) enables the
    conservative componentwise operator ``div(nu_eff grad(omega))``; otherwise
    the scalar molecular Laplacian is used. Regenerated particles inherit the
    pre-regeneration ν_t. ``wall_fallback_limit`` is the fraction of the total
    strength that may sit on wall-adjacent particles without bounded moment
    support (see :meth:`compute_m4_wall_corrections`).

    Algorithm
    ---------
    1. M4' or six-point Lagrange scatter: vortex strength → grid.
    2. Explicit diffusion: stable forward-Euler substeps of the constant
       molecular Laplacian, or ``div(nu_eff grad(omega))`` with arithmetic
       face viscosities when per-particle ν+ν_t is supplied (GPU kernel).
       The latter is componentwise vorticity diffusion. For variable
       viscosity it is not ``curl(div(2 nu_eff S))``, the complete
       incompressible viscous-stress source used by the FVM solver.
       Matching Smagorinsky constants does not make those models equivalent.
    3. Weak-strength removal: discard weak grid nodes  (CPU NumPy — small).
    4. Spawn new particles at surviving nodes       (CPU NumPy).

    Vortex-strength scatter and diffusion run on the device. Variable
    viscosity is mapped on the host and uploaded before diffusion; the
    diffused strength grid is downloaded once for weak-strength removal.

    If the macro-step exceeds the 3-D explicit limit, only the Laplacian is
    substepped. Scatter, thresholding, and regeneration still occur once.
    """
    # One is also the correct reach for a skipped/zero-diffusivity GBD
    # operation. A completed Laplacian below replaces this with the exact
    # runtime stage count used by the current regeneration.
    lattice.last_gbd_diffusion_substeps = 1
    if remeshing_kernel not in {"m4_prime", "lagrange6"}:
        raise ValueError("unsupported GBD remeshing kernel")
    if remeshing_kernel == "lagrange6" and domain_padding < 4:
        raise ValueError("remeshing_kernel='lagrange6' needs at least four grid cells of padding")
    lattice.last_gbd_moment_correction = empty_gbd_moment_correction()
    lattice.last_gbd_wall_transfer = empty_gbd_wall_transfer()
    N = particles.n_particles_total
    if N == 0:
        return None
    lattice.ping = True

    slab_bounds = lattice.slip_slab_bounds
    sources = read_diffusion_sources(particles, "GBD", slab_bounds)
    pos_np = sources.position
    vortex_strength = sources.vortex_strength
    if slab_bounds is not None and remeshing_kernel != "m4_prime":
        raise ValueError("slip-slab GBD requires remeshing_kernel='m4_prime'")

    invalid = lattice.wall.interior_at_particles(pos_np)
    if np.any(invalid):
        invalid_strength = vortex_strength[invalid]
        raise RuntimeError(
            "GBD received particles strictly inside the solid before scatter: "
            f"count={int(np.count_nonzero(invalid))}, "
            f"vortex_strength_l1={float(np.linalg.norm(invalid_strength, axis=1).sum()):.6e}"
        )

    # -- Grid setup --------------------------------------------------------
    # Use a fixed grid origin when the domain was pre-configured, to avoid
    # the asymmetric flat-end artefact (see DiffusionLattice.fixed_grid_min).
    required_z_bounds = None
    if slab_bounds is not None:
        lattice.require_fixed_slab_extent("GBD")
        max_diffusivity = (
            float(kinematic_viscosity)
            if effective_viscosity is None
            else float(np.max(effective_viscosity[:N], initial=0.0))
        )
        substeps = gbd_diffusion_substep_count(max_diffusivity, time_step_size, particle_spacing)
        # Slab nodes see only halo nodes the explicit stages reach.
        halo_cells = max(3, substeps + 1)
        required_z_bounds = (
            slab_bounds[0] - halo_cells * particle_spacing,
            slab_bounds[1] + halo_cells * particle_spacing,
        )
    grid_min_np, (nx, ny, nz) = lattice.fit_active_box(
        pos_np, particle_spacing, domain_padding, required_z_bounds=required_z_bounds
    )
    if slab_bounds is not None:
        phase = 2.0 * (np.asarray(slab_bounds) - grid_min_np[2]) / particle_spacing
        if np.max(np.abs(phase - np.rint(phase))) > 1e-4:
            raise ValueError(
                "slip planes must align to grid nodes or half nodes: "
                f"phase={phase.tolist()}, grid_min_z={grid_min_np[2]:.9g}, "
                f"spacing={particle_spacing:.9g}, fixed_grid_min="
                f"{None if lattice.fixed_grid_min is None else float(lattice.fixed_grid_min[2])}"
            )
    node_mapping = nearest_node_mapping(
        pos_np, vortex_strength, grid_min_np, particle_spacing, nx, ny, nz
    )

    # -- M4' scatter (GPU) -------------------------------------------------
    with lattice.profiler.section("GBD remeshing"):
        lattice.wall.prepare_mask(grid_min_np, particle_spacing, nx, ny, nz)
        lattice.zero_grid_kernel(lattice.current_grid, nx, ny, nz)
        gmin = grid_min_np.astype(float)
        scatter = (
            m4_scatter_gpu_kernel
            if remeshing_kernel == "m4_prime"
            else lagrange6_scatter_gpu_kernel
        )
        batch_size = _M4_SCATTER_BATCH_SIZE if remeshing_kernel == "m4_prime" else 1024
        for start in range(0, N, batch_size):
            count = min(batch_size, N - start)
            scatter(
                particles.position,
                particles.vortex_strength,
                lattice.current_grid,
                gmin[0],
                gmin[1],
                gmin[2],
                float(particle_spacing),
                nx,
                ny,
                nz,
                start,
                count,
            )
            ti.sync()
    with lattice.profiler.section("GBD wall correction"):
        if lattice.wall.active:
            if remeshing_kernel != "m4_prime":
                raise RuntimeError(
                    "solid-wall GBD redistribution requires remeshing_kernel='m4_prime'"
                )
            correction_nodes, correction_values, wall_budget = (
                lattice.wall.compute_m4_wall_corrections(
                    pos_np,
                    vortex_strength,
                    grid_min_np,
                    particle_spacing,
                    (nx, ny, nz),
                    wall_fallback_limit,
                )
            )
            lattice.last_gbd_wall_transfer = wall_budget
            if len(correction_nodes):
                linear = (
                    correction_nodes[:, 0].astype(np.int64) * (ny * nz)
                    + correction_nodes[:, 1].astype(np.int64) * nz
                    + correction_nodes[:, 2]
                )
                unique, inverse = np.unique(linear, return_inverse=True)
                correction_values = np.column_stack(
                    [
                        np.bincount(
                            inverse,
                            weights=correction_values[:, component],
                            minlength=len(unique),
                        )
                        for component in range(3)
                    ]
                ).astype(np.float32)
                correction_nodes = np.column_stack(
                    (unique // (ny * nz), (unique // nz) % ny, unique % nz)
                ).astype(np.int32)
            for start in range(0, len(correction_nodes), lattice.TRANSFER_CHUNK):
                stop = min(start + lattice.TRANSFER_CHUNK, len(correction_nodes))
                add_sparse_wall_corrections_kernel(
                    lattice.current_grid,
                    np.ascontiguousarray(correction_nodes[start:stop].reshape(-1)),
                    np.ascontiguousarray(correction_values[start:stop]),
                    stop - start,
                )
            if len(correction_nodes):
                ti.sync()
    if slab_bounds is not None:
        # The M4 kernel, wall mask and moment corrections are equivariant
        # under the slab reflections, so the physical deposits fix the
        # field of every image on the active grid, halo included.
        lower, period = slab_image_lattice(grid_min_np[2], particle_spacing, slab_bounds)
        add_slab_images_kernel(lattice.current_grid, lattice.other_grid, nx, ny, nz, lower, period)
        lattice.ping = not lattice.ping
    lattice.apply_body_mask(nx, ny, nz)

    # -- Explicit Laplacian diffusion (GPU) --------------------------------
    # Variable ν_eff is stored at nodes and averaged to conservative face
    # fluxes. This advances div(ν_eff grad(ω)), not curl(div(2 ν_eff S)).
    # With no LES coefficient, use the scalar molecular Laplacian.
    effective_viscosity_grid_np = None
    fill_radius = None
    if effective_viscosity is not None:
        effective_viscosity_np = np.ascontiguousarray(effective_viscosity[:N], dtype=np.float32)
        # Clip negatives (defensive: Smagorinsky guards already, but the
        # grid scatter can introduce tiny excursions via round-off).
        np.clip(effective_viscosity_np, 0.0, None, out=effective_viscosity_np)
        if slab_bounds is None and not lattice.wall.active and len(effective_viscosity_np):
            # In free space, vorticity reaches at most two cells beyond a particle's
            # nearest node (M4' support) plus one cell per explicit stage; a node one
            # cell farther still meets it in a face flux. The particle maximum bounds
            # the node maximum, so these stages are at least those of the grid.
            stages, _ = _explicit_diffusion_substep_count(
                float(effective_viscosity_np.max()), float(time_step_size), float(particle_spacing)
            )
            fill_radius = stages + 4
        effective_viscosity_grid_np = scatter_scalar_weighted(
            pos_np,
            vortex_strength,
            effective_viscosity_np,
            grid_min_np,
            particle_spacing,
            nx,
            ny,
            nz,
            mapping=node_mapping,
            fill_empty=True,
            fill_radius=fill_radius,
        )
        if slab_bounds is not None:
            # Viscosity is even under every slab image; halo nodes take
            # the value of the physical node their image maps onto them.
            effective_viscosity_grid_np = even_slab_field(
                effective_viscosity_grid_np,
                float(grid_min_np[2]),
                particle_spacing,
                slab_bounds,
            ).astype(np.float32)
    with lattice.profiler.section("GBD diffusion"):
        lattice.last_gbd_diffusion_substeps, _max_diffusion_number = _advance_gbd_laplacian(
            lattice,
            nx=nx,
            ny=ny,
            nz=nz,
            time_step_size=float(time_step_size),
            particle_spacing=float(particle_spacing),
            kinematic_viscosity=float(kinematic_viscosity),
            effective_viscosity_grid=effective_viscosity_grid_np,
        )

    # -- Body mask (GPU, optional) -----------------------------------------
    lattice.apply_body_mask(nx, ny, nz)

    # -- Weak-strength removal (CPU — read diffused grid back once) --------
    with lattice.profiler.section("GBD grid download"):
        grid_np = lattice.download_active_vec_grid(lattice.current_grid, nx, ny, nz)

    if spectral_closure is not None:
        if slab_bounds is not None or lattice.wall.active:
            raise ValueError(
                "The spectral eddy viscosity transforms an unbounded lattice; "
                "it is not available with walls or slip planes."
            )
        with lattice.profiler.section("Spectral closure"):
            closure = spectral_closure.apply(grid_np, particle_spacing, time_step_size)
        lattice.logger.section(
            "spectral eddy viscosity",
            ("cut-off wavenumber", f"{closure.cutoff_wavenumber:.4e}", "rad/m"),
            ("E(k_c)", f"{closure.cutoff_energy_spectrum:.4e}", "m^6/s^2"),
            ("eddy viscosity, plateau", f"{closure.plateau_eddy_viscosity:.4e}", "m^2/s"),
            ("eddy viscosity, cut-off", f"{closure.cutoff_eddy_viscosity:.4e}", "m^2/s"),
            ("energy removed / step", f"{closure.subgrid_kinetic_energy_rate:.4e}", "m^5/s^3"),
        )

    regenerated = select_regenerated_nodes(
        lattice.logger,
        "GBD",
        particles,
        sources,
        grid_np,
        grid_min_np,
        particle_spacing,
        node_mapping,
        regen_threshold,
        regen_threshold_mode,
        slab_bounds,
        fill_support_only=fill_radius is not None,
    )
    if regenerated is None:
        lattice.ping = True
        return None
    grid_np = regenerated.grid
    vortex_strength_magnitude = regenerated.magnitude
    vortex_strength_total = regenerated.magnitude_total
    ix, iy, iz = regenerated.ix, regenerated.iy, regenerated.iz
    group_winner_grid = regenerated.group_winner_grid
    cap = regenerated.capacity

    with lattice.profiler.section("GBD moment redistribution"):
        nonzero_node_count = int(np.count_nonzero(vortex_strength_magnitude > 0.0))
        support_augmented_node_count = 0
        preserve_group_correction = False
        wall_correction = lattice.wall.segment_classifier is not None
        correction_labels = group_winner_grid
        if wall_correction and len(ix) < nonzero_node_count:
            correction_labels = lattice.wall.correction_labels(
                vortex_strength_magnitude, group_winner_grid
            )
        if len(ix) < nonzero_node_count:
            retained_before_support = len(ix)
            (
                ix,
                iy,
                iz,
                support_augmented_node_count,
                preserve_group_correction,
            ) = augment_moment_correction_support(
                grid_np,
                vortex_strength_magnitude,
                ix,
                iy,
                iz,
                grid_min_np,
                particle_spacing,
                cap,
                labels=correction_labels,
                strict_labels=wall_correction,
            )
            if support_augmented_node_count:
                lattice.logger.section(
                    "gaussian blob diffusion moment support",
                    ("nodes, threshold/cap result", f"{retained_before_support:,}"),
                    ("nodes, support added", f"{support_augmented_node_count:,}"),
                    ("nodes, final retained", f"{len(ix):,}"),
                    ("regeneration cap", f"{cap:,}"),
                )
        correction_diagnostics = empty_gbd_moment_correction()
        correction_diagnostics.update(
            nonzero_node_count=nonzero_node_count,
            retained_node_count=int(len(ix)),
            removed_node_count=int(nonzero_node_count - len(ix)),
            support_augmented_node_count=int(support_augmented_node_count),
        )
        lattice.last_gbd_moment_correction = correction_diagnostics
        if len(ix) < nonzero_node_count:
            moment_diagnostics: dict[str, bool | int | float] = {}
            active_correction_labels = correction_labels if preserve_group_correction else None
            while True:
                try:
                    corrected_retained = redistribute_removed_moments(
                        grid_np,
                        vortex_strength_magnitude,
                        ix,
                        iy,
                        iz,
                        grid_min_np,
                        particle_spacing,
                        labels=active_correction_labels,
                        diagnostics=moment_diagnostics,
                        strict_labels=wall_correction,
                        segments_cross_wall=lattice.wall.segments_cross_wall
                        if wall_correction
                        else None,
                    )
                    break
                except GBDMomentCorrectionError as error:
                    # A geometrically complete basis can still misrepresent a
                    # small fluid component that lost most of its nodes. Keep that
                    # component's diffused field; never relax the moment or
                    # correction-fraction limits.
                    ix, iy, iz, added = retain_diffused_component(
                        vortex_strength_magnitude,
                        ix,
                        iy,
                        iz,
                        cap,
                        active_correction_labels,
                        error.correction_label,
                    )
                    support_augmented_node_count += added
                    correction_diagnostics.update(
                        retained_node_count=int(len(ix)),
                        removed_node_count=int(nonzero_node_count - len(ix)),
                        support_augmented_node_count=int(support_augmented_node_count),
                    )
                    lattice.logger.section(
                        "gaussian blob diffusion retained diffused component",
                        ("component", str(error.correction_label)),
                        ("diffused nodes retained", f"{added:,}"),
                        ("nodes, final retained", f"{len(ix):,}"),
                        ("regeneration cap", f"{cap:,}"),
                    )
            raw_retained = grid_np[ix, iy, iz].astype(np.float64)
            correction_l1 = float(
                np.linalg.norm(corrected_retained.astype(np.float64) - raw_retained, axis=1).sum(
                    dtype=np.float64
                )
            )
            complete_net = grid_np.sum(axis=(0, 1, 2), dtype=np.float64)
            raw_net = raw_retained.sum(axis=0, dtype=np.float64)
            corrected_net = corrected_retained.sum(axis=0, dtype=np.float64)
            net_scale = vortex_strength_total + 1.0e-30
            lattice.logger.section(
                "gaussian blob diffusion moment correction",
                ("correction strength fraction", f"{correction_l1 / net_scale:.6e}"),
                (
                    "net residual, before",
                    f"{np.linalg.norm(raw_net - complete_net) / net_scale:.6e}",
                ),
                (
                    "net residual, after",
                    f"{np.linalg.norm(corrected_net - complete_net) / net_scale:.6e}",
                ),
            )
            grid_np[ix, iy, iz] = corrected_retained
            correction_diagnostics.update(moment_diagnostics)
            correction_diagnostics["applied"] = True
            lattice.last_gbd_moment_correction = correction_diagnostics

    result = build_particle_arrays(
        ix,
        iy,
        iz,
        grid_np,
        grid_min_np,
        particle_spacing,
        kinematic_viscosity,
        float(lattice.core_radius_ratio) * float(particle_spacing),
        regenerated.zone_winner_grid,
        group_winner_grid,
        eddy_viscosity_grid=regenerated.eddy_viscosity_grid,
        slab_bounds=slab_bounds,
    )
    lattice.ping = True
    return result


@ti.func
def _lagrange6_weights_ti(fraction: ti.f32):
    weights = ti.Vector.zero(ti.f32, 6)
    for j in ti.static(range(6)):
        value = 1.0
        for k in ti.static(range(6)):
            if ti.static(j != k):
                value *= (fraction - ti.cast(k - 2, ti.f32)) / ti.cast(j - k, ti.f32)
        weights[j] = value
    return weights


@ti.kernel
def lagrange6_scatter_gpu_kernel(
    position: ti.template(),
    vortex_strength: ti.template(),
    grid: ti.template(),
    gmin_x: ti.f32,
    gmin_y: ti.f32,
    gmin_z: ti.f32,
    particle_spacing: ti.f32,
    nx: ti.i32,
    ny: ti.i32,
    nz: ti.i32,
    start_particle: ti.i32,
    count: ti.i32,
):
    """Six-point tensor-product Lagrange scatter, with bounded dispatches."""
    for local_particle in range(count):
        p = start_particle + local_particle
        scaled = (position[p] - ti.Vector([gmin_x, gmin_y, gmin_z])) / particle_spacing
        base = ti.cast(ti.floor(scaled), ti.i32)
        fraction = scaled - ti.cast(base, ti.f32)
        wx = _lagrange6_weights_ti(fraction[0])
        wy = _lagrange6_weights_ti(fraction[1])
        wz = _lagrange6_weights_ti(fraction[2])
        for di, dj, dk in ti.ndrange(6, 6, 6):
            i, j, k = base[0] + di - 2, base[1] + dj - 2, base[2] + dk - 2
            if 0 <= i < nx and 0 <= j < ny and 0 <= k < nz:
                weight = wx[di] * wy[dj] * wz[dk]
                for component in ti.static(range(3)):
                    ti.atomic_add(grid[i, j, k][component], weight * vortex_strength[p][component])


@ti.kernel
def add_slab_images_kernel(
    deposit: ti.template(),
    images: ti.template(),
    nx: ti.i32,
    ny: ti.i32,
    nz: ti.i32,
    lower: ti.i32,
    period: ti.i32,
):
    """Field of every slip-slab image of the physical deposits.

    Node k receives node m directly when k - m is a multiple of the image
    period and as an axial-vector reflection when k + m - lower is; see
    ``openonda.core.slip_slab_images``. Offsets keep both remainders nonnegative.
    """
    offset = 4 * period * nz
    for i, j, k in ti.ndrange(nx, ny, nz):
        total = ti.Vector.zero(ti.f32, 3)
        for m in range(nz):
            value = deposit[i, j, m]
            if (k - m + offset) % period == 0:
                total += value
            if (k + m - lower + offset) % period == 0:
                total += ti.Vector([-value[0], -value[1], value[2]])
        images[i, j, k] = total


@ti.kernel
def m4_scatter_gpu_kernel(
    position: ti.template(),
    vortex_strength: ti.template(),
    grid: ti.template(),
    gmin_x: ti.f32,
    gmin_y: ti.f32,
    gmin_z: ti.f32,
    particle_spacing: ti.f32,
    nx: ti.i32,
    ny: ti.i32,
    nz: ti.i32,
    start_particle: ti.i32,
    count: ti.i32,
):
    """Device M4' remeshing scatter: each particle deposits to 4³=64 grid nodes.

    The scatter needs no host-device data transfer. Atomic adds ensure correctness
    when multiple particles contribute to the same grid node.

    Accuracy: same quadratic moment reproduction as the host M4' scatters.
    """
    for local_particle in range(count):
        p = start_particle + local_particle
        pos = position[p]
        particle_vortex_strength = vortex_strength[p]

        # Fractional grid coordinates
        fx = (pos[0] - gmin_x) / particle_spacing
        fy = (pos[1] - gmin_y) / particle_spacing
        fz = (pos[2] - gmin_z) / particle_spacing

        # Base index (floor)
        ix0 = int(ti.floor(fx))
        iy0 = int(ti.floor(fy))
        iz0 = int(ti.floor(fz))

        # M4' 4³ stencil: offsets -1, 0, 1, 2 relative to base index
        for di in ti.static(range(-1, 3)):
            rx = ti.abs(fx - ti.cast(ix0 + di, ti.f32))
            wx = m4_prime_ti(rx)
            for dj in ti.static(range(-1, 3)):
                ry = ti.abs(fy - ti.cast(iy0 + dj, ti.f32))
                wy = m4_prime_ti(ry)
                for dk in ti.static(range(-1, 3)):
                    rz = ti.abs(fz - ti.cast(iz0 + dk, ti.f32))
                    wz = m4_prime_ti(rz)
                    w = wx * wy * wz

                    ii = ix0 + di
                    jj = iy0 + dj
                    kk = iz0 + dk

                    if 0 <= ii < nx and 0 <= jj < ny and 0 <= kk < nz:
                        ti.atomic_add(grid[ii, jj, kk][0], w * particle_vortex_strength[0])
                        ti.atomic_add(grid[ii, jj, kk][1], w * particle_vortex_strength[1])
                        ti.atomic_add(grid[ii, jj, kk][2], w * particle_vortex_strength[2])


@ti.kernel
def add_sparse_wall_corrections_kernel(
    grid: ti.template(),
    indices: ti.types.ndarray(),
    corrections: ti.types.ndarray(),
    count: ti.i32,
):
    """Add host-reduced corrections; input nodes are unique."""
    for item in range(count):
        i, j, k = indices[3 * item], indices[3 * item + 1], indices[3 * item + 2]
        grid[i, j, k] += ti.Vector(
            [corrections[item, 0], corrections[item, 1], corrections[item, 2]]
        )
