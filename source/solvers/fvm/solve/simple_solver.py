"""SIMPLE pressure-velocity coupling for incompressible flow."""

from dataclasses import dataclass
from typing import Any, Literal, overload

from numba import njit
import numpy as np

from ..assemble import matrix_assembly, momentum
from ..fields import diagnostics as field_diagnostics
from ..fields import gradients
from ..fields.mixed_velocity_boundary import (
    update_normal_velocity_tangential_gradient_boundary,
)
from ..schemes.boundaries import BOUNDARIES, BoundaryStrategy
from ..utils import cavity_utils
from .contracts import OuterCorrectorDiagnostics
from .linear_interface import normalized_residual, solve_linear_system


@dataclass(frozen=True)
class PressureBoundaryLayout:
    """Immutable boundary-face indexing used by pressure assembly.

    Patch values are intentionally excluded: coupling and freestream updates
    may change them between correctors.  The layout only captures topology and
    resolved boundary behaviour, which are stable for an algorithm instance.
    """

    signature: tuple[tuple[int, int, str], ...]
    face_indices: np.ndarray
    type_codes: np.ndarray


@dataclass(frozen=True)
class PressureCorrectionWorkspace:
    """Dynamic Rhie–Chow data shared by pressure assembly and velocity correction.

    ``pressure_velocity_coefficient`` (the inverse diagonal of the momentum matrix at each cell) and
    ``face_conductance`` (the interpolated face flux coefficient) are
    computed once per PIMPLE corrector and reused by both the pressure
    Poisson assembly and the subsequent face-flux and velocity correction.

    Attributes
    ----------
    pressure_velocity_coefficient : np.ndarray
        Inverse of the momentum matrix diagonal. The production momentum
        path stores the shared scalar diagonal as ``(n_cells,)``. Anisotropic
        operators may supply component diagonals as ``(n_cells, 3)``.
    face_conductance : np.ndarray
        Interpolated face flux coefficient, shape ``(n_faces,)``.
    matrix : Any or None
        Pressure operator assembled from ``face_conductance``. It is reusable
        for every pressure/non-orthogonal correction belonging to the same
        momentum predictor because ``momentum_diagonal`` is unchanged within that loop.
    """

    pressure_velocity_coefficient: np.ndarray
    face_conductance: np.ndarray
    matrix: Any | None = None


def _resolve_pressure_constraint(params) -> str:
    """Select the configured all-Neumann pressure treatment for this backend."""
    policy = str(params.get("pressure_nullspace_policy", "auto")).lower()
    backend = str(params.get("_linear_backend", "scipy")).lower()
    if policy == "auto":
        return "nullspace" if backend == "petsc" else "reference"
    if policy == "petsc":
        if backend != "petsc":
            raise ValueError("The PETSc pressure null space requires backend='petsc'")
        return "nullspace"
    if policy == "reference":
        return "reference"
    raise ValueError(f"Unknown pressure null-space policy {policy!r}")


def _pressure_requires_constraint(boundaries, velocity_star, mesh_data, geo_data) -> bool:
    """Return whether the assembled pressure operator has a constant null space."""
    n_cells = mesh_data["n_cells"]
    n_interior = mesh_data["n_interior_faces"]
    for boundary in boundaries:
        boundary_condition_type = boundary.get("pressure_type")
        strategy = BOUNDARIES.strategy(boundary_condition_type, "kinematic_pressure", "pressure")
        if strategy is BoundaryStrategy.FIXED_VALUE:
            local_requires_constraint = False
            break
        if strategy is BoundaryStrategy.FREESTREAM:
            start = boundary["start_face"]
            nf = boundary["n_faces"]
            ghosts = n_cells + np.arange(start - n_interior, start - n_interior + nf)
            flux = np.sum(
                velocity_star[ghosts] * geo_data["face_area_vector"][start : start + nf], axis=1
            )
            if np.any(flux >= 0.0):
                local_requires_constraint = False
                break
    else:
        local_requires_constraint = True
    parallel = mesh_data.get("_parallel_context")
    if parallel is not None and parallel.is_partitioned:
        return parallel.global_all(local_requires_constraint)
    return local_requires_constraint


def compute_ddt_flux_correction(
    velocity_old,
    velocity_older,
    volumetric_face_flux_old,
    volumetric_face_flux_older,
    time_step_size,
    mesh_data,
    geo_data,
    boundaries,
    ddt_scheme,
):
    r"""Return the transient Rhie--Chow face-history correction.

    This is the limited difference between the committed face flux and the
    flux obtained by interpolating the committed cell velocity, divided by the
    time step.  BDF2 uses the same two-level history combination as the cell
    derivative, and every non-coupled boundary contribution is zero.

    Fully periodic domains are special here: their conservative face flux is
    already the complete old-time history used by this solver's H/A
    reconstruction.  Applying a second correction double-counts that history.
    Boundary-driven domains still require the correction to couple the cell
    and face transient operators.
    """
    if volumetric_face_flux_old is None or time_step_size is None or velocity_old is None:
        return None

    strategies = [
        BOUNDARIES.strategy(b.get("velocity_type"), "velocity", "ghost") for b in boundaries
    ]
    if strategies and all(
        strategy in (BoundaryStrategy.CYCLIC, BoundaryStrategy.EMPTY) for strategy in strategies
    ):
        return None

    n_faces = mesh_data["n_faces"]
    n_interior = mesh_data["n_interior_faces"]

    if str(ddt_scheme).lower() in ("backward", "bdf2") and (
        velocity_older is not None and volumetric_face_flux_older is not None
    ):
        coefft0, coefft00 = 2.0, 0.5
    else:
        coefft0, coefft00 = 1.0, 0.0

    volumetric_face_flux_old = np.asarray(volumetric_face_flux_old, dtype=np.float64)
    velocity_old = np.asarray(velocity_old, dtype=np.float64)
    older_flux = (
        np.zeros_like(volumetric_face_flux_old)
        if volumetric_face_flux_older is None
        else np.asarray(volumetric_face_flux_older, dtype=np.float64)
    )
    velocity_history = coefft0 * velocity_old
    if coefft00 != 0.0:
        velocity_history = velocity_history - coefft00 * np.asarray(
            velocity_older, dtype=np.float64
        )

    owners = mesh_data["owners"]
    neighbours = mesh_data["neighbours"]
    weights = geo_data["face_interpolation_weight"]
    face_area_vector = geo_data["face_area_vector"]
    correction = np.zeros(n_faces, dtype=np.float64)
    chunk_size = 250_000

    for start in range(0, n_interior, chunk_size):
        stop = min(start + chunk_size, n_interior)
        own = owners[start:stop]
        nei = neighbours[start:stop]
        w = weights[start:stop, np.newaxis]
        history_face = w * velocity_history[nei] + (1.0 - w) * velocity_history[own]
        old_face = w * velocity_old[nei] + (1.0 - w) * velocity_old[own]
        history_flux = coefft0 * volumetric_face_flux_old[start:stop]
        if coefft00 != 0.0:
            history_flux -= coefft00 * older_flux[start:stop]
        phi_corr = history_flux - np.einsum("ij,ij->i", history_face, face_area_vector[start:stop])
        reference = volumetric_face_flux_old[start:stop] - np.einsum(
            "ij,ij->i", old_face, face_area_vector[start:stop]
        )
        coupling = 1.0 - np.minimum(
            np.abs(reference)
            / (np.abs(volumetric_face_flux_old[start:stop]) + np.finfo(np.float64).tiny),
            1.0,
        )
        correction[start:stop] = coupling * phi_corr / float(time_step_size)

    boundary_neighbour_cell = mesh_data.get("boundary_neighbour_cell")
    for boundary, strategy in zip(boundaries, strategies, strict=True):
        if strategy is not BoundaryStrategy.CYCLIC:
            continue
        start = boundary["start_face"]
        stop = start + boundary["n_faces"]
        faces = np.arange(start, stop)
        own = owners[faces]
        paired = boundary_neighbour_cell[faces]
        w = weights[faces, np.newaxis]
        history_face = w * velocity_history[paired] + (1.0 - w) * velocity_history[own]
        old_face = w * velocity_old[paired] + (1.0 - w) * velocity_old[own]
        history_flux = coefft0 * volumetric_face_flux_old[faces]
        if coefft00 != 0.0:
            history_flux -= coefft00 * older_flux[faces]
        phi_corr = history_flux - np.einsum("ij,ij->i", history_face, face_area_vector[faces])
        reference = volumetric_face_flux_old[faces] - np.einsum(
            "ij,ij->i", old_face, face_area_vector[faces]
        )
        coupling = 1.0 - np.minimum(
            np.abs(reference)
            / (np.abs(volumetric_face_flux_old[faces]) + np.finfo(np.float64).tiny),
            1.0,
        )
        correction[faces] = coupling * phi_corr / float(time_step_size)

    return correction


def _compute_rhie_chow_coefficients(volumes, momentum_diagonal):
    """Compute the pressure_velocity_coefficient coefficients for Rhie-Chow interpolation.

    ``pressure_velocity_coefficient = V / momentum_diagonal`` converts pressure-gradient cell values to velocity
    corrections: ``Δvelocity = −pressure_velocity_coefficient · ∇kinematic_pressure'``.

    Args:
        volumes: Cell volumes ``(n_elements,)``.
        momentum_diagonal:     Diagonal coefficients from the momentum equation
                 ``(n_elements, 3)`` (per component).

    Returns:
        pressure_velocity_coefficient array ``(n_elements, 3)``, with a small regulariser to avoid
        division by zero.
    """
    diagonal = np.asarray(momentum_diagonal, dtype=np.float64)
    if diagonal.ndim == 1:
        return volumes / (diagonal + 1e-10)
    return volumes[:, np.newaxis] / (diagonal + 1e-10)


def _validate_reference_density(density) -> float:
    """Validate the scalar reference density used for dimensional forces.

    Constant density cancels from the kinematic-pressure flow equations, but
    remains part of the public setup because dimensional forces scale with it.
    """
    density = np.asarray(density, dtype=np.float64)
    if density.ndim != 0:
        raise ValueError("constant-density FVM requires density to be a scalar")
    value = float(density.item())
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("density must be finite and positive")
    return value


def _compute_pressure_face_conductance(mesh_data, geo_data, pressure_velocity_coefficient):
    """Return the geometric Rhie--Chow conductance for every face.

    The pressure matrix and the post-solve flux correction must use exactly
    the same conductance.  Keeping this calculation in one function prevents
    the non-orthogonal inconsistency that previously used ``Sf·e`` during
    assembly but ``|Sf|`` during correction.

    ``pressure_velocity_coefficient`` is the cell-centred diagonal pressure-to-velocity coefficient.  It
    is linearly interpolated on interior faces and taken from the owner on
    boundary faces. Because pressure is kinematic and ``volumetric_face_flux`` is volumetric
    flux, density does not enter this conductance.
    """
    n_faces = mesh_data["n_faces"]
    n_interior = mesh_data["n_interior_faces"]
    owners = mesh_data["owners"]
    neighbours = mesh_data["neighbours"]

    scalar_diagonal = np.asarray(pressure_velocity_coefficient).ndim == 1
    sf = geo_data["face_area_vector"]
    cf_vec = geo_data["cell_connection_vector"]
    weights = geo_data["face_interpolation_weight"]
    boundary_neighbour_cell = np.asarray(
        mesh_data.get("boundary_neighbour_cell", np.full(n_faces, -1, dtype=np.int32))
    )
    conductance = np.empty(n_faces, dtype=np.float64)
    chunk_size = 250_000
    for start in range(0, n_faces, chunk_size):
        stop = min(start + chunk_size, n_faces)
        face_slice = slice(start, stop)
        sf_block = sf[face_slice]
        cf_block = cf_vec[face_slice]
        mag_sf = np.linalg.norm(sf_block, axis=1)
        mag_cf = np.linalg.norm(cf_block, axis=1)
        if np.any(mag_sf <= 1e-30) or np.any(mag_cf <= 1e-30):
            raise ValueError("Pressure conductance requires non-zero face area and cell distance")

        own = owners[face_slice]
        neighbour = np.full(stop - start, -1, dtype=np.int32)
        interior_stop = min(stop, n_interior)
        if interior_stop > start:
            neighbour[: interior_stop - start] = neighbours[start:interior_stop]
        boundary_start = max(start, n_interior)
        if stop > boundary_start:
            neighbour[boundary_start - start :] = boundary_neighbour_cell[boundary_start:stop]
        coupled = neighbour >= 0

        face_pressure_velocity_coefficient = np.asarray(
            pressure_velocity_coefficient[own], dtype=np.float64
        ).copy()
        if np.any(coupled):
            w = weights[face_slice][coupled]
            if scalar_diagonal:
                face_pressure_velocity_coefficient[coupled] = (
                    w * pressure_velocity_coefficient[neighbour[coupled]]
                    + (1.0 - w) * pressure_velocity_coefficient[own[coupled]]
                )
            else:
                w_vector = w[:, np.newaxis]
                face_pressure_velocity_coefficient[coupled] = (
                    w_vector * pressure_velocity_coefficient[neighbour[coupled]]
                    + (1.0 - w_vector) * pressure_velocity_coefficient[own[coupled]]
                )
        if scalar_diagonal:
            effective_pressure_velocity_coefficient = face_pressure_velocity_coefficient
        else:
            normal = sf_block / mag_sf[:, np.newaxis]
            effective_pressure_velocity_coefficient = np.sum(
                normal * normal * face_pressure_velocity_coefficient, axis=1
            )
        edge = cf_block / mag_cf[:, np.newaxis]
        orthogonal_area = np.sum(sf_block * edge, axis=1)
        conductance[face_slice] = effective_pressure_velocity_coefficient * orthogonal_area / mag_cf

    if np.any(conductance < -1e-14):
        raise ValueError(
            "Negative pressure-face conductance; check face orientation and mesh geometry"
        )
    return np.maximum(conductance, 0.0)


def _update_fixed_flux_pressure_boundaries(
    kinematic_pressure,
    velocity_star,
    pressure_velocity_coefficient,
    mesh_data,
    geo_data,
    boundaries,
    kinematic_pressure_gradient=None,
    pressure_free_face_flux=None,
):
    """Update ``fixedFluxPressure`` from the pressure-free face flux."""
    n_cells = mesh_data["n_cells"]
    n_interior = mesh_data["n_interior_faces"]
    owners = mesh_data["owners"]
    face_area_vector = geo_data["face_area_vector"]
    face_cf = geo_data["cell_connection_vector"]
    _grad_fn = gradients._resolve_gradient_fn(geo_data)

    fixed_flux_patches = []
    for boundary in boundaries:
        strategy = BOUNDARIES.strategy(boundary.get("pressure_type"), "kinematic_pressure", "ghost")
        if strategy is BoundaryStrategy.FIXED_FLUX_PRESSURE or (
            strategy is BoundaryStrategy.FREESTREAM
            and boundary.get("_directional_fixed_flux_pressure", False)
        ):
            fixed_flux_patches.append(boundary)
    if not fixed_flux_patches:
        return kinematic_pressure_gradient

    if kinematic_pressure_gradient is None:
        kinematic_pressure_gradient = _grad_fn(kinematic_pressure, mesh_data, geo_data)
        if kinematic_pressure_gradient.ndim == 3:
            kinematic_pressure_gradient = kinematic_pressure_gradient.squeeze(-1)
    assert kinematic_pressure_gradient is not None

    velocity_h_over_a = None
    if pressure_free_face_flux is not None:
        pressure_free_face_flux = np.asarray(pressure_free_face_flux, dtype=float)
        if pressure_free_face_flux.shape != (mesh_data["n_faces"],):
            raise ValueError("pressure_free_face_flux must have one value per face")
    else:
        pressure_velocity_coefficient_vector = (
            pressure_velocity_coefficient[:, np.newaxis]
            if np.asarray(pressure_velocity_coefficient).ndim == 1
            else pressure_velocity_coefficient
        )
        velocity_h_over_a = (
            velocity_star[:n_cells]
            + pressure_velocity_coefficient_vector * kinematic_pressure_gradient[:n_cells]
        )

    changed = False
    for boundary in fixed_flux_patches:
        start = boundary["start_face"]
        nf = boundary["n_faces"]
        ghost = n_cells + (start - n_interior) + np.arange(nf)
        own = owners[start : start + nf]
        if boundary.get("fixed_flux_pressure_external", False):
            delta = boundary.get("fixed_flux_pressure_delta")
            if delta is not None:
                kinematic_pressure[ghost] = kinematic_pressure[own] + np.asarray(delta, dtype=float)
                changed = True
            continue

        sf = face_area_vector[start : start + nf]
        mag_sf = np.linalg.norm(sf, axis=1)
        normal = sf / mag_sf[:, np.newaxis]
        dr = face_cf[start : start + nf]
        normal_distance = np.einsum("ij,ij->i", dr, normal)
        if np.asarray(pressure_velocity_coefficient).ndim == 1:
            D_normal = pressure_velocity_coefficient[own]
        else:
            D_normal = np.einsum("ij,ij->i", pressure_velocity_coefficient[own], normal * normal)
        phi_target = np.einsum("ij,ij->i", velocity_star[ghost], sf)
        if pressure_free_face_flux is None:
            assert velocity_h_over_a is not None
            volumetric_face_flux_h_over_a = np.einsum("ij,ij->i", velocity_h_over_a[own], sf)
        else:
            volumetric_face_flux_h_over_a = pressure_free_face_flux[start : start + nf]
        pressure_flux_coefficient = mag_sf * D_normal
        dpdn = (volumetric_face_flux_h_over_a - phi_target) / np.maximum(
            pressure_flux_coefficient, 1.0e-30
        )
        delta = dpdn * normal_distance
        boundary["fixed_flux_pressure_delta"] = delta
        if boundary.get("_directional_fixed_flux_pressure", False):
            outflow = np.asarray(boundary["_fixed_freestream_outflow"], dtype=bool)
            kinematic_pressure[ghost] = np.where(
                outflow,
                float(boundary.get("kinematic_pressure_value", 0.0)),
                kinematic_pressure[own] + delta,
            )
        else:
            kinematic_pressure[ghost] = kinematic_pressure[own] + delta
        changed = True

    if changed:
        kinematic_pressure_gradient = _grad_fn(kinematic_pressure, mesh_data, geo_data)
        if kinematic_pressure_gradient.ndim == 3:
            kinematic_pressure_gradient = kinematic_pressure_gradient.squeeze(-1)
    return kinematic_pressure_gradient


@njit(cache=True)
def _process_boundary_faces_jit(
    n_boundary_faces,
    n_interior,
    n_cells,
    owners,
    face_area_vector,
    cell_connection_vector,
    velocity_star,
    pressure_velocity_coefficient_x,
    pressure_velocity_coefficient_y,
    pressure_velocity_coefficient_z,
    kinematic_pressure_gradient,
    kinematic_pressure,
    bc_type_codes,
    kinematic_pressure_boundary_values,
    boundary_face_indices,
    face_conductance,
):
    """Numba-JITted boundary-face processing for Rhie-Chow assembly.

    Vectorised over all boundary faces using integer-coded BC types.
    Significantly faster than looping per patch in pure Python.

    Args:
        n_boundary_faces:     Total number of boundary faces.
        n_interior:           Number of interior faces.
        n_elements:           Number of interior elements.
        owners:               Owner index array.
        face_area_vector:              Face area vectors ``(n_faces, 3)``.
        cell_connection_vector:       Centre-to-centre vectors ``(n_faces, 3)``.
        velocity_star:               Predicted velocity ``(n_total, 3)``.
        pressure_velocity_coefficient_x, pressure_velocity_coefficient_y, pressure_velocity_coefficient_z:       Component views of the Rhie-Chow coefficients.
                              For a scalar momentum diagonal all three
                              arguments are the same one-dimensional array.
        kinematic_pressure_gradient:               Pressure gradient ``(n_total, 3)``.
        kinematic_pressure:                    Pressure field ``(n_total,)``.
        bc_type_codes:        Integer-coded BC: 0=zeroGradient, 1=fixedValue, 2=empty.
        kinematic_pressure_boundary_values:    Fixed pressure values for fixedValue patches.
        boundary_face_indices: Global face index for each boundary face.

    Returns:
        Tuple ``(flux_cf_out, flux_ff_out, flux_vf_out)`` — coefficient
        arrays for all boundary faces.
    """
    n_total_boundary = len(boundary_face_indices)
    flux_cf_out = np.zeros(n_total_boundary, dtype=np.float64)
    flux_ff_out = np.zeros(n_total_boundary, dtype=np.float64)
    flux_vf_out = np.zeros(n_total_boundary, dtype=np.float64)

    for i in range(n_total_boundary):
        i_face = boundary_face_indices[i]
        own = owners[i_face]

        Sf0, Sf1, Sf2 = (
            face_area_vector[i_face, 0],
            face_area_vector[i_face, 1],
            face_area_vector[i_face, 2],
        )

        bc_code = bc_type_codes[i]

        b_elem_idx = n_cells + (i_face - n_interior)
        Ub0, Ub1, Ub2 = (
            velocity_star[b_elem_idx, 0],
            velocity_star[b_elem_idx, 1],
            velocity_star[b_elem_idx, 2],
        )
        velocity_flux = Ub0 * Sf0 + Ub1 * Sf1 + Ub2 * Sf2

        # zeroGradient pressure, including the inflow side of freestream
        if bc_code == 0 or (bc_code == 3 and velocity_flux < 0.0):
            flux_vf_out[i] = Ub0 * Sf0 + Ub1 * Sf1 + Ub2 * Sf2
            continue

        # fixedValue pressure, including the outflow side of freestream
        if bc_code == 1 or bc_code == 3:
            CF0, CF1, CF2 = (
                cell_connection_vector[i_face, 0],
                cell_connection_vector[i_face, 1],
                cell_connection_vector[i_face, 2],
            )
            mag_CF = (CF0 * CF0 + CF1 * CF1 + CF2 * CF2) ** 0.5

            e0 = CF0 / (mag_CF + 1e-10)
            e1 = CF1 / (mag_CF + 1e-10)
            e2 = CF2 / (mag_CF + 1e-10)

            (
                owner_pressure_velocity_coefficient_x,
                owner_pressure_velocity_coefficient_y,
                owner_pressure_velocity_coefficient_z,
            ) = (
                pressure_velocity_coefficient_x[own],
                pressure_velocity_coefficient_y[own],
                pressure_velocity_coefficient_z[own],
            )
            gp0, gp1, gp2 = (
                kinematic_pressure_gradient[own, 0],
                kinematic_pressure_gradient[own, 1],
                kinematic_pressure_gradient[own, 2],
            )

            # Base velocity flux
            flux_vf = Ub0 * Sf0 + Ub1 * Sf1 + Ub2 * Sf2

            # Geometric diffusion.  This value is shared with pressure-matrix
            # assembly and post-solve flux correction.
            mag_Sf = (Sf0 * Sf0 + Sf1 * Sf1 + Sf2 * Sf2) ** 0.5
            if mag_Sf < 1e-30:
                continue
            sf_dot_e = Sf0 * e0 + Sf1 * e1 + Sf2 * e2
            geoDiff = face_conductance[i_face]

            if geoDiff > 0:
                cf = geoDiff
                ff = -geoDiff

                # Interpolated gradient flux
                term_interp = (
                    owner_pressure_velocity_coefficient_x * gp0 * Sf0
                    + owner_pressure_velocity_coefficient_y * gp1 * Sf1
                    + owner_pressure_velocity_coefficient_z * gp2 * Sf2
                )

                # Compact pressure drive
                val = kinematic_pressure_boundary_values[i]
                term_compact = cf * kinematic_pressure[own] + ff * val

                # Non-orthogonal correction
                k0 = Sf0 - sf_dot_e * e0
                k1 = Sf1 - sf_dot_e * e1
                k2 = Sf2 - sf_dot_e * e2
                k_norm = (k0 * k0 + k1 * k1 + k2 * k2) ** 0.5
                if k_norm > 1e-12:
                    flux_nonortho = (
                        k0 * owner_pressure_velocity_coefficient_x * gp0
                        + k1 * owner_pressure_velocity_coefficient_y * gp1
                        + k2 * owner_pressure_velocity_coefficient_z * gp2
                    )
                else:
                    flux_nonortho = 0.0

                flux_vf = flux_vf + term_interp + term_compact + flux_nonortho
                flux_cf_out[i] = cf
                flux_ff_out[i] = ff

            flux_vf_out[i] = flux_vf
            continue

        # bc_code == 2 (empty): nothing to do — already zero-initialized
    return flux_cf_out, flux_ff_out, flux_vf_out


def _pressure_boundary_signature(boundaries) -> tuple[tuple[int, int, str], ...]:
    """Return the structural part of the pressure-boundary configuration."""
    signature = []
    for boundary in boundaries:
        strategy = BOUNDARIES.strategy(
            boundary.get("pressure_type"), "kinematic_pressure", "pressure"
        )
        signature.append((int(boundary["start_face"]), int(boundary["n_faces"]), strategy.name))
    return tuple(signature)


def _pressure_boundary_matrix_is_reusable(boundaries) -> bool:
    """Return whether pressure boundary coefficients stay fixed this step.

    Ordinary freestream faces switch between pressure Dirichlet and Neumann
    with the evolving face flux. A directional coupling patch carries an
    immutable geometric outflow mask, so its pressure matrix is just as static
    as a conventional fixed-value/fixed-gradient layout.
    """
    for boundary in boundaries:
        strategy = BOUNDARIES.strategy(
            boundary.get("pressure_type"), "kinematic_pressure", "pressure"
        )
        if (
            strategy is BoundaryStrategy.FREESTREAM
            and boundary.get("_fixed_freestream_outflow") is None
        ):
            return False
    return True


def build_pressure_boundary_layout(boundaries, n_interior, n_faces) -> PressureBoundaryLayout:
    """Build immutable, vectorized pressure boundary-face metadata."""
    n_bnd = n_faces - n_interior
    codes = np.empty(n_bnd, dtype=np.int32)
    face_indices = np.arange(n_interior, n_faces, dtype=np.int32)
    for boundary in boundaries:
        start = int(boundary["start_face"])
        nf = int(boundary["n_faces"])
        local = slice(start - n_interior, start - n_interior + nf)
        strategy = BOUNDARIES.strategy(
            boundary.get("pressure_type"), "kinematic_pressure", "pressure"
        )
        if strategy is BoundaryStrategy.FIXED_VALUE:
            code = 1
        elif strategy is BoundaryStrategy.EMPTY:
            code = 2
        elif strategy is BoundaryStrategy.FREESTREAM:
            code = 3
        elif strategy in (
            BoundaryStrategy.ZERO_GRADIENT,
            BoundaryStrategy.FIXED_FLUX_PRESSURE,
            BoundaryStrategy.FIXED_GRADIENT,
            BoundaryStrategy.CYCLIC,
        ):
            code = 0
        else:
            raise RuntimeError(f"Unhandled pressure boundary strategy {strategy!r}")
        codes[local] = code
    return PressureBoundaryLayout(_pressure_boundary_signature(boundaries), face_indices, codes)


def _pressure_boundary_values(boundaries, n_interior, n_faces) -> np.ndarray:
    """Read kinematic-pressure values without rebuilding face topology."""
    values = np.zeros(n_faces - n_interior, dtype=np.float64)
    for boundary in boundaries:
        start = int(boundary["start_face"])
        nf = int(boundary["n_faces"])
        local = slice(start - n_interior, start - n_interior + nf)
        field = boundary.get("kinematic_pressure_value_field")
        if field is not None:
            field_values = np.asarray(field, dtype=np.float64)
            if field_values.shape != (nf,):
                raise ValueError(
                    f"Per-face pressure value for patch {boundary.get('name')!r} "
                    f"must have shape ({nf},), got {field_values.shape}"
                )
            values[local] = field_values
        else:
            val = boundary.get("kinematic_pressure_value", 0.0)
            if val is not None:
                values[local] = val
    return values


def _build_boundary_face_arrays(boundaries, n_interior, n_faces, layout=None):
    """Return the canonical pressure boundary arrays.

    Returns
    -------
    bc_type_codes : ndarray
        0=zeroGradient, 1=fixedValue, 2=empty, 3=freestream
    kinematic_pressure_boundary_values : ndarray
        Fixed pressure value (or 0.0 if not applicable)
    boundary_face_indices : ndarray
        Global face index for each boundary face
    """
    signature = _pressure_boundary_signature(boundaries)
    if layout is None or layout.signature != signature:
        layout = build_pressure_boundary_layout(boundaries, n_interior, n_faces)
    return (
        layout.type_codes,
        _pressure_boundary_values(boundaries, n_interior, n_faces),
        layout.face_indices,
    )


def adjust_boundary_flux_for_continuity(
    volumetric_face_flux, boundaries, mesh_data, n_interior, n_faces
):
    """Rescale adjustable outflow boundary fluxes so the net is zero.

    An incompressible domain requires ``∮ volumetric_face_flux·dS = 0``; otherwise the pressure
    Poisson problem is incompatible
    and the solver leaves a residual divergence (a checkerboard, typically at
    the outflow).  Faces whose velocity is *fixed* (``fixedValue`` inlet,
    ``noSlip`` wall) carry a prescribed flux and cannot absorb the mismatch;
    faces whose velocity *floats* (``freestream``, ``zeroGradient``,
    ``inletOutlet``) can.  The net imbalance is removed by a single
    multiplicative scaling of the floating **outflow** faces, exactly matching
    the mass the fixed faces admit.

    When the boundary flux already balances (a standalone inlet/pressure-outlet
    case), the scale factor is unity and this is a no-op — so it never
    perturbs cases that were already conservative.  The reduction is global so
    a partitioned coupling patch spread over several ranks is balanced once,
    not per rank.

    ``volumetric_face_flux`` is mutated in place.
    """
    floating = (
        BoundaryStrategy.FREESTREAM,
        BoundaryStrategy.ZERO_GRADIENT,
        BoundaryStrategy.INLET_OUTLET,
    )
    boundary_neighbour_cell = np.asarray(
        mesh_data.get("boundary_neighbour_cell", np.full(n_faces, -1, dtype=np.int32))
    )

    net_local = 0.0
    outflow_local = 0.0
    adjustable_slices: list[tuple[int, int, np.ndarray]] = []
    for boundary in boundaries:
        start = int(boundary["start_face"])
        nf = int(boundary["n_faces"])
        if nf == 0:
            continue
        patch_flux = volumetric_face_flux[start : start + nf]
        # Cyclic faces pair internally and never carry net domain flux.
        if np.any(boundary_neighbour_cell[start : start + nf] >= 0):
            continue
        net_local += float(np.sum(patch_flux))
        strategy = BOUNDARIES.strategy(boundary.get("velocity_type"), "velocity", "flux")
        if strategy in floating:
            # Scale exactly the faces the pressure matrix treats as
            # adjustable: the assembly's own classification when present.
            mask = boundary.get("_freestream_outflow")
            outflow = (patch_flux > 0.0) if mask is None else mask & (patch_flux > 0.0)
            outflow_local += float(np.sum(patch_flux[outflow]))
            adjustable_slices.append((start, nf, outflow))

    parallel = mesh_data.get("_parallel_context")
    if parallel is not None and parallel.is_partitioned:
        net = float(parallel.global_sum(net_local))
        outflow_total = float(parallel.global_sum(outflow_local))
    else:
        net, outflow_total = net_local, outflow_local

    # Already balanced (relative test, dimensionless): nothing to do.
    if abs(net) <= 1e-11 * outflow_total:
        return volumetric_face_flux
    # Not enough floating outflow to absorb the imbalance without reversing a
    # face; leave it for the pressure solve rather than manufacture backflow.
    if outflow_total <= abs(net):
        return volumetric_face_flux

    # Scale the floating outflow faces so their total drops by exactly `net`.
    scale = (outflow_total - net) / outflow_total
    for start, nf, outflow in adjustable_slices:
        patch = volumetric_face_flux[start : start + nf]
        patch[outflow] *= scale
        volumetric_face_flux[start : start + nf] = patch
    return volumetric_face_flux


@njit(cache=True)
def _pressure_interior_flux_scalar(
    owners,
    neighbours,
    weights,
    face_area_vector,
    cell_connection_vector,
    pressure_velocity_coefficient,
    velocity_h_over_a,
    kinematic_pressure_gradient,
    kinematic_pressure,
    face_conductance,
    ddt_flux_correction,
    flux_vf,
):
    """Fuse scalar-diagonal Rhie--Chow face work without large temporaries."""
    use_ddt = len(ddt_flux_correction) != 0
    for face in range(len(neighbours)):
        own = owners[face]
        nei = neighbours[face]
        weight = weights[face]
        owner_weight = 1.0 - weight
        face_pressure_velocity_coefficient = (
            weight * pressure_velocity_coefficient[nei]
            + owner_weight * pressure_velocity_coefficient[own]
        )

        cf0 = cell_connection_vector[face, 0]
        cf1 = cell_connection_vector[face, 1]
        cf2 = cell_connection_vector[face, 2]
        mag_cf = np.sqrt(cf0 * cf0 + cf1 * cf1 + cf2 * cf2) + 1e-12
        edge0 = cf0 / mag_cf
        edge1 = cf1 / mag_cf
        edge2 = cf2 / mag_cf
        sf0 = face_area_vector[face, 0]
        sf1 = face_area_vector[face, 1]
        sf2 = face_area_vector[face, 2]
        orthogonal_area = sf0 * edge0 + sf1 * edge1 + sf2 * edge2

        grad0 = (
            weight * kinematic_pressure_gradient[nei, 0]
            + owner_weight * kinematic_pressure_gradient[own, 0]
        )
        grad1 = (
            weight * kinematic_pressure_gradient[nei, 1]
            + owner_weight * kinematic_pressure_gradient[own, 1]
        )
        grad2 = (
            weight * kinematic_pressure_gradient[nei, 2]
            + owner_weight * kinematic_pressure_gradient[own, 2]
        )
        nonorthogonal_flux = face_pressure_velocity_coefficient * (
            (sf0 - orthogonal_area * edge0) * grad0
            + (sf1 - orthogonal_area * edge1) * grad1
            + (sf2 - orthogonal_area * edge2) * grad2
        )
        velocity_h_over_a_face_flux = (
            (weight * velocity_h_over_a[nei, 0] + owner_weight * velocity_h_over_a[own, 0]) * sf0
            + (weight * velocity_h_over_a[nei, 1] + owner_weight * velocity_h_over_a[own, 1]) * sf1
            + (weight * velocity_h_over_a[nei, 2] + owner_weight * velocity_h_over_a[own, 2]) * sf2
        )
        if use_ddt:
            velocity_h_over_a_face_flux += (
                face_pressure_velocity_coefficient * ddt_flux_correction[face]
            )
        conductance = face_conductance[face]
        flux_vf[face] = (
            velocity_h_over_a_face_flux
            + conductance * (kinematic_pressure[own] - kinematic_pressure[nei])
            + nonorthogonal_flux
        )


@njit(cache=True)
def _pressure_interior_flux_vector(
    owners,
    neighbours,
    weights,
    face_area_vector,
    cell_connection_vector,
    pressure_velocity_coefficient,
    velocity_h_over_a,
    kinematic_pressure_gradient,
    kinematic_pressure,
    face_conductance,
    ddt_flux_correction,
    flux_vf,
):
    """Fused component-diagonal Rhie--Chow interior-face kernel."""
    use_ddt = len(ddt_flux_correction) != 0
    for face in range(len(neighbours)):
        own = owners[face]
        nei = neighbours[face]
        weight = weights[face]
        owner_weight = 1.0 - weight
        pressure_velocity_coefficient_x = (
            weight * pressure_velocity_coefficient[nei, 0]
            + owner_weight * pressure_velocity_coefficient[own, 0]
        )
        pressure_velocity_coefficient_y = (
            weight * pressure_velocity_coefficient[nei, 1]
            + owner_weight * pressure_velocity_coefficient[own, 1]
        )
        pressure_velocity_coefficient_z = (
            weight * pressure_velocity_coefficient[nei, 2]
            + owner_weight * pressure_velocity_coefficient[own, 2]
        )

        cf0 = cell_connection_vector[face, 0]
        cf1 = cell_connection_vector[face, 1]
        cf2 = cell_connection_vector[face, 2]
        mag_cf = np.sqrt(cf0 * cf0 + cf1 * cf1 + cf2 * cf2) + 1e-12
        edge0 = cf0 / mag_cf
        edge1 = cf1 / mag_cf
        edge2 = cf2 / mag_cf
        sf0 = face_area_vector[face, 0]
        sf1 = face_area_vector[face, 1]
        sf2 = face_area_vector[face, 2]
        orthogonal_area = sf0 * edge0 + sf1 * edge1 + sf2 * edge2

        grad0 = (
            weight * kinematic_pressure_gradient[nei, 0]
            + owner_weight * kinematic_pressure_gradient[own, 0]
        )
        grad1 = (
            weight * kinematic_pressure_gradient[nei, 1]
            + owner_weight * kinematic_pressure_gradient[own, 1]
        )
        grad2 = (
            weight * kinematic_pressure_gradient[nei, 2]
            + owner_weight * kinematic_pressure_gradient[own, 2]
        )
        nonorthogonal_flux = (
            (sf0 - orthogonal_area * edge0) * pressure_velocity_coefficient_x * grad0
            + (sf1 - orthogonal_area * edge1) * pressure_velocity_coefficient_y * grad1
            + (sf2 - orthogonal_area * edge2) * pressure_velocity_coefficient_z * grad2
        )
        velocity_h_over_a_face_flux = (
            (weight * velocity_h_over_a[nei, 0] + owner_weight * velocity_h_over_a[own, 0]) * sf0
            + (weight * velocity_h_over_a[nei, 1] + owner_weight * velocity_h_over_a[own, 1]) * sf1
            + (weight * velocity_h_over_a[nei, 2] + owner_weight * velocity_h_over_a[own, 2]) * sf2
        )
        if use_ddt:
            velocity_h_over_a_face_flux += (
                (
                    pressure_velocity_coefficient_x
                    + pressure_velocity_coefficient_y
                    + pressure_velocity_coefficient_z
                )
                / 3.0
            ) * ddt_flux_correction[face]
        conductance = face_conductance[face]
        flux_vf[face] = (
            velocity_h_over_a_face_flux
            + conductance * (kinematic_pressure[own] - kinematic_pressure[nei])
            + nonorthogonal_flux
        )


@overload
def assemble_pressure_correction_equation_rhie_chow(
    *args: Any, return_workspace: Literal[False] = False, **kwargs: Any
) -> tuple[Any, Any, np.ndarray]: ...


@overload
def assemble_pressure_correction_equation_rhie_chow(
    *args: Any, return_workspace: Literal[True], **kwargs: Any
) -> tuple[Any, Any, np.ndarray, PressureCorrectionWorkspace]: ...


def assemble_pressure_correction_equation_rhie_chow(
    velocity_star,
    momentum_diagonal,
    kinematic_pressure,
    density,
    mesh_data,
    geo_data,
    boundaries,
    velocity_relaxation=1.0,
    pressure_constraint="reference",
    matrix_workspace=None,
    operator_backend="numpy",
    boundary_layout=None,
    ddt_flux_correction=None,
    correction_workspace=None,
    reuse_matrix=False,
    return_workspace=False,
):
    """
    Assemble pressure correction equation using Modified Rhie-Chow interpolation.

    This implementation uses the "H-by-A" reconstruction method:
    1. Reconstruct velocity without pressure gradient at cell centres (velocity_h_over_a).
    2. Interpolate velocity_h_over_a to faces.
    3. Add compact pressure gradient drive at faces.

    This is more robust against checkerboarding than the standard correction method.

    Args:
        velocity_star: Predicted velocity field
        momentum_diagonal: Momentum diagonal coefficients
        kinematic_pressure: Current kinematic-pressure field ``kinematic_pressure/ρ`` [m²/s²].
        density: Positive constant reference density [kg/m³]. It is validated
            for API compatibility but cancels from this pressure equation.
        mesh_data: Mesh connectivity
        geo_data: Geometric data
        boundaries: Boundary conditions
        velocity_relaxation: Velocity under-relaxation factor

    Returns:
        tuple: (pressure_matrix, pressure_right_hand_side, f_vf) where f_vf is the Rhie-Chow corrected flux (volumetric_face_flux_star).
    """
    n_cells = mesh_data["n_cells"]
    n_interior = mesh_data["n_interior_faces"]
    n_faces = mesh_data["n_faces"]
    owners = mesh_data["owners"]
    neighbours = mesh_data["neighbours"]
    _validate_reference_density(density)

    # 1. Compute pressure_velocity_coefficient and kinematic_pressure_gradient. The momentum diagonal is fixed for all
    # pressure/non-orthogonal corrections in one PIMPLE outer iteration, so
    # retain its inverse and face conductance instead of rebuilding two
    # full-mesh arrays on every inner solve.
    volumes = geo_data["cell_volume"]
    if correction_workspace is None:
        # Restore physical momentum_diagonal from relaxed momentum_diagonal for Rhie-Chow D-coefficients
        physical_momentum_diagonal = momentum_diagonal * velocity_relaxation
        pressure_velocity_coefficient = _compute_rhie_chow_coefficients(
            volumes, physical_momentum_diagonal
        )
        face_conductance = _compute_pressure_face_conductance(
            mesh_data, geo_data, pressure_velocity_coefficient
        )
    else:
        pressure_velocity_coefficient = correction_workspace.pressure_velocity_coefficient
        face_conductance = correction_workspace.face_conductance
    if reuse_matrix and (correction_workspace is None or correction_workspace.matrix is None):
        raise ValueError("reuse_matrix requires an assembled pressure-correction workspace")

    # Use direct gradient computation for full pressure field kinematic_pressure
    _grad_fn = gradients._resolve_gradient_fn(geo_data)
    kinematic_pressure_gradient = _grad_fn(kinematic_pressure, mesh_data, geo_data)
    if kinematic_pressure_gradient.ndim == 3:
        kinematic_pressure_gradient = kinematic_pressure_gradient.squeeze(-1)
    kinematic_pressure_gradient = _update_fixed_flux_pressure_boundaries(
        kinematic_pressure,
        velocity_star,
        pressure_velocity_coefficient,
        mesh_data,
        geo_data,
        boundaries,
        kinematic_pressure_gradient=kinematic_pressure_gradient,
    )
    assert kinematic_pressure_gradient is not None

    # Pre-allocate flux arrays
    flux_cf = None if reuse_matrix else np.zeros(n_faces)
    flux_ff = None if reuse_matrix else np.zeros(n_faces)
    flux_vf = np.zeros(n_faces)

    # --- MODIFIED RHIE-CHOW INTERPOLATION ---
    # Standard Rhie-Chow: U_f = Avg(velocity) + Avg(D)*(Avg(GradP) - CompactGradP)
    # Modified Rhie-Chow: U_f = Avg(velocity + D*GradP) - Avg(D)*CompactGradP
    # This is much more robust on persistent checkerboarding.

    # 1. Reconstruct "H/A" velocity at cell centres (Velocity without pressure gradient)
    # U_centre = H/A - kinematic_pressure_gradient * pressure_velocity_coefficient
    # So H/A = U_centre + kinematic_pressure_gradient * pressure_velocity_coefficient
    # We use velocity_star as U_centre (it includes -kinematic_pressure_gradient * pressure_velocity_coefficient approx).
    scalar_diagonal = np.asarray(pressure_velocity_coefficient).ndim == 1
    pressure_velocity_coefficient_vector = (
        pressure_velocity_coefficient[:, np.newaxis]
        if scalar_diagonal
        else pressure_velocity_coefficient
    )
    velocity_h_over_a = (
        velocity_star[:n_cells]
        + pressure_velocity_coefficient_vector * kinematic_pressure_gradient[:n_cells]
    )

    # Fuse the interior-face interpolation and non-orthogonal correction in a
    # compiled loop. Besides being faster than eight chains of NumPy advanced
    # indexing per PIMPLE step, this avoids retaining pressure_velocity_coefficient, velocity_h_over_a, edge,
    # non-orthogonal, and gradient temporaries at the same time.
    grad_p_interior = kinematic_pressure_gradient[:n_cells]
    if not reuse_matrix:
        assert flux_cf is not None and flux_ff is not None
        flux_cf[:n_interior] = face_conductance[:n_interior]
        np.negative(face_conductance[:n_interior], out=flux_ff[:n_interior])
    ddt_values = (
        np.empty(0, dtype=np.float64)
        if ddt_flux_correction is None
        else np.asarray(ddt_flux_correction, dtype=np.float64)
    )
    interior_kernel = (
        _pressure_interior_flux_scalar if scalar_diagonal else _pressure_interior_flux_vector
    )
    interior_kernel(
        owners[:n_interior],
        neighbours[:n_interior],
        geo_data["face_interpolation_weight"][:n_interior],
        geo_data["face_area_vector"][:n_interior],
        geo_data["cell_connection_vector"][:n_interior],
        pressure_velocity_coefficient,
        velocity_h_over_a,
        grad_p_interior,
        kinematic_pressure,
        face_conductance,
        ddt_values,
        flux_vf,
    )

    # 3. Boundary Faces (Numba JIT)
    n_boundary_faces = n_faces - n_interior
    if n_boundary_faces > 0:
        bc_codes, p_vals, bnd_face_idx = _build_boundary_face_arrays(
            boundaries, n_interior, n_faces, layout=boundary_layout
        )

        # Authoritative freestream inflow/outflow switch for THIS assembly:
        # the same ghost-velocity flux the JIT thresholds below.  Every later
        # stage of the correction (kinematic_pressure' ghost extension, boundary-flux
        # correction, velocity/pressure ghost updates) must reuse this mask.
        # Re-deriving the switch from the evolving flux field lets grazing
        # faces (u·n ≈ 0, e.g. the lateral sides of a coupling box) change
        # class between assembly and correction, injecting boundary flux the
        # pressure matrix never saw — which surfaces as a divergence
        # checkerboard anchored at the patch corners.
        for boundary in boundaries:
            strategy = BOUNDARIES.strategy(
                boundary.get("pressure_type"), "kinematic_pressure", "pressure"
            )
            if strategy is BoundaryStrategy.FREESTREAM:
                start = int(boundary["start_face"])
                nf = int(boundary["n_faces"])
                fixed_outflow = boundary.get("_fixed_freestream_outflow")
                if fixed_outflow is not None:
                    boundary["_freestream_outflow"] = fixed_outflow
                else:
                    ghost = velocity_star[
                        n_cells + (start - n_interior) : n_cells + (start - n_interior) + nf
                    ]
                    sf_patch = geo_data["face_area_vector"][start : start + nf]
                    boundary["_freestream_outflow"] = np.einsum("ij,ij->i", ghost, sf_patch) >= 0.0
        pressure_velocity_coefficient_components = (
            (
                pressure_velocity_coefficient,
                pressure_velocity_coefficient,
                pressure_velocity_coefficient,
            )
            if scalar_diagonal
            else (
                pressure_velocity_coefficient[:, 0],
                pressure_velocity_coefficient[:, 1],
                pressure_velocity_coefficient[:, 2],
            )
        )
        cf_b, ff_b, vf_b = _process_boundary_faces_jit(
            n_boundary_faces,
            n_interior,
            n_cells,
            owners,
            geo_data["face_area_vector"],
            geo_data["cell_connection_vector"],
            velocity_star,
            *pressure_velocity_coefficient_components,
            kinematic_pressure_gradient,
            kinematic_pressure,
            bc_codes,
            p_vals,
            bnd_face_idx,
            face_conductance,
        )
        if not reuse_matrix:
            assert flux_cf is not None and flux_ff is not None
            flux_cf[bnd_face_idx] = cf_b
            flux_ff[bnd_face_idx] = ff_b
        flux_vf[bnd_face_idx] = vf_b

        # A replay experiment may supply the face flux measured by the same
        # monolithic discretisation.  Keep it as the pressure-equation
        # boundary flux (and, for fixedFluxPressure, through the correction)
        # instead of reconstructing it from an interpolated boundary velocity.
        # Normal coupled runs never populate this optional patch field.
        for boundary in boundaries:
            external_flux = boundary.get("external_face_flux")
            if external_flux is None:
                continue
            start = int(boundary["start_face"])
            nf = int(boundary["n_faces"])
            field = np.asarray(external_flux, dtype=float).reshape(-1)
            if field.shape != (nf,) or not np.all(np.isfinite(field)):
                raise ValueError(
                    f"External face flux for patch {boundary.get('name')!r} must have "
                    f"shape ({nf},) and finite values"
                )
            flux_vf[start : start + nf] = field

        boundary_neighbour_cell = np.asarray(
            mesh_data.get("boundary_neighbour_cell", np.full(n_faces, -1, dtype=np.int32))
        )
        cyclic_faces = np.flatnonzero(boundary_neighbour_cell >= 0)
        if cyclic_faces.size:
            own_b = owners[cyclic_faces]
            nei_b = boundary_neighbour_cell[cyclic_faces]
            weight_b_scalar = geo_data["face_interpolation_weight"][cyclic_faces]
            weight_b = weight_b_scalar[:, np.newaxis]
            sf_b = geo_data["face_area_vector"][cyclic_faces]
            cf_b = geo_data["cell_connection_vector"][cyclic_faces]
            mag_cf_b = np.linalg.norm(cf_b, axis=1)
            edge_b = cf_b / mag_cf_b[:, np.newaxis]
            if scalar_diagonal:
                boundary_pressure_velocity_coefficient = (
                    weight_b_scalar * pressure_velocity_coefficient[nei_b]
                    + (1.0 - weight_b_scalar) * pressure_velocity_coefficient[own_b]
                )
            else:
                boundary_pressure_velocity_coefficient = (
                    weight_b * pressure_velocity_coefficient[nei_b]
                    + (1.0 - weight_b) * pressure_velocity_coefficient[own_b]
                )
            hbya_b = (
                weight_b * velocity_h_over_a[nei_b] + (1.0 - weight_b) * velocity_h_over_a[own_b]
            )
            grad_b = (
                weight_b * kinematic_pressure_gradient[nei_b]
                + (1.0 - weight_b) * kinematic_pressure_gradient[own_b]
            )
            orthogonal_area = np.sum(sf_b * edge_b, axis=1)
            nonorthogonal = sf_b - orthogonal_area[:, np.newaxis] * edge_b
            conductance_b = face_conductance[cyclic_faces]
            if not reuse_matrix:
                assert flux_cf is not None and flux_ff is not None
                flux_cf[cyclic_faces] = conductance_b
                flux_ff[cyclic_faces] = -conductance_b
            velocity_h_over_a_face_flux = np.sum(hbya_b * sf_b, axis=1)
            compact = conductance_b * (kinematic_pressure[own_b] - kinematic_pressure[nei_b])
            if scalar_diagonal:
                nonorthogonal_flux = boundary_pressure_velocity_coefficient * np.sum(
                    nonorthogonal * grad_b, axis=1
                )
            else:
                nonorthogonal_flux = np.sum(
                    nonorthogonal * boundary_pressure_velocity_coefficient * grad_b, axis=1
                )
            flux_vf[cyclic_faces] = velocity_h_over_a_face_flux + compact + nonorthogonal_flux

    # 3b. Enforce global continuity of the predicted boundary flux (adjustPhi)
    # so the pressure Poisson problem is compatible.  A no-op when the boundary
    # flux already balances, so conservative inlet/outlet cases are untouched.
    if n_boundary_faces > 0:
        adjust_boundary_flux_for_continuity(flux_vf, boundaries, mesh_data, n_interior, n_faces)

    # 4. Assemble Matrix and RHS
    flux_data = {"flux_vf": flux_vf}
    if reuse_matrix:
        assert correction_workspace is not None
        assert correction_workspace.matrix is not None
        pressure_matrix = correction_workspace.matrix
    else:
        assert flux_cf is not None and flux_ff is not None
        flux_data.update({"flux_cf": flux_cf, "flux_ff": flux_ff})
        pressure_matrix = matrix_assembly.assemble_matrix_from_fluxes_vectorized(
            flux_data, mesh_data, workspace=matrix_workspace, backend=operator_backend
        )
    pressure_right_hand_side = matrix_assembly.assemble_rhs_from_fluxes_vectorized(
        flux_data, mesh_data, backend=operator_backend
    )

    # 5. Fix Pressure Reference only for an all-Neumann pressure problem.
    if _pressure_requires_constraint(boundaries, velocity_star, mesh_data, geo_data):
        if pressure_constraint == "reference":
            pressure_matrix, pressure_right_hand_side = cavity_utils.fix_pressure_reference(
                pressure_matrix, pressure_right_hand_side
            )
        elif pressure_constraint == "nullspace":
            # A finite-volume all-Neumann RHS should already be compatible;
            # remove only accumulated roundoff before the backend projection.
            parallel = mesh_data.get("_parallel_context")
            if parallel is not None and parallel.is_partitioned:
                n_owned = parallel.n_owned
                global_sum = parallel.global_sum(float(np.sum(pressure_right_hand_side[:n_owned])))
                pressure_right_hand_side = (
                    pressure_right_hand_side - global_sum / parallel.partition.n_global_cells
                )
            else:
                pressure_right_hand_side = pressure_right_hand_side - np.mean(
                    pressure_right_hand_side
                )
        else:
            raise ValueError(
                "All-Neumann pressure requires pressure_constraint='reference' or 'nullspace'"
            )

    if return_workspace:
        if correction_workspace is None:
            correction_workspace = PressureCorrectionWorkspace(
                pressure_velocity_coefficient, face_conductance, pressure_matrix
            )
        return pressure_matrix, pressure_right_hand_side, flux_vf, correction_workspace
    return pressure_matrix, pressure_right_hand_side, flux_vf


def _extend_kinematic_pressure_correction_bcs(
    kinematic_pressure_correction, mesh_data, boundaries, volumetric_face_flux=None
):
    """Extend the pressure-correction array with ghost-cell values.

    For ``fixedValue`` pressure boundaries, the ghost value is set to
    zero (``kinematic_pressure' = 0`` at a fixed-pressure face).  For all other types,
    the ghost cell inherits the owner cell value (zero-gradient).

    Args:
        kinematic_pressure_correction:    Pressure correction for interior cells ``(n_elements,)``.
        mesh_data:  Mesh dictionary.
        boundaries: List of boundary patch dictionaries.

    Returns:
        Extended ``kinematic_pressure_correction`` array ``(n_elements + n_boundary_faces,)``.
    """
    n_cells = mesh_data["n_cells"]
    n_interior = mesh_data["n_interior_faces"]
    n_faces = mesh_data["n_faces"]
    owners = mesh_data["owners"]
    kinematic_pressure_correction_extended = np.zeros(n_cells + (n_faces - n_interior))
    kinematic_pressure_correction_extended[:n_cells] = kinematic_pressure_correction
    for boundary in boundaries:
        start = boundary["start_face"]
        nf = boundary["n_faces"]
        idx = n_cells + (start - n_interior)
        own = owners[start : start + nf]
        pressure_type = boundary.get("pressure_type")
        strategy = BOUNDARIES.strategy(pressure_type, "kinematic_pressure", "ghost")
        if strategy is BoundaryStrategy.FIXED_VALUE:
            kinematic_pressure_correction_extended[idx : idx + nf] = 0.0
        elif strategy is BoundaryStrategy.CYCLIC:
            paired = mesh_data["boundary_neighbour_cell"][start : start + nf]
            kinematic_pressure_correction_extended[idx : idx + nf] = kinematic_pressure_correction[
                paired
            ]
        elif strategy is BoundaryStrategy.FREESTREAM:
            outflow = boundary.get("_freestream_outflow")
            if outflow is None:
                if volumetric_face_flux is None:
                    raise ValueError(
                        "Freestream pressure correction requires the predicted face flux"
                    )
                outflow = np.asarray(volumetric_face_flux)[start : start + nf] >= 0.0
            kinematic_pressure_correction_extended[idx : idx + nf] = np.where(
                outflow, 0.0, kinematic_pressure_correction[own]
            )
        elif strategy in (
            BoundaryStrategy.ZERO_GRADIENT,
            BoundaryStrategy.FIXED_FLUX_PRESSURE,
            BoundaryStrategy.FIXED_GRADIENT,
            BoundaryStrategy.EMPTY,
        ):
            kinematic_pressure_correction_extended[idx : idx + nf] = kinematic_pressure_correction[
                own
            ]
        else:
            raise RuntimeError(f"Unhandled pressure ghost strategy {strategy!r}")
    return kinematic_pressure_correction_extended


def _correct_interior_fluxes(
    volumetric_face_flux, kinematic_pressure_correction, mesh_data, face_conductance
):
    """Correct interior face fluxes with the Rhie-Chow pressure correction.

    Applies the volumetric-flux correction ``Δφ = g⋅(kinematic_pressure'_P − kinematic_pressure'_N)`` where *g* is
    the geometric diffusion coefficient based on the interpolated pressure_velocity_coefficient and
    face-normal projection.

    Args:
        volumetric_face_flux:      Face flux array ``(n_faces,)`` (mutated in place).
        kinematic_pressure_correction:  Pressure correction ``(n_elements,)``.
        mesh_data: Mesh dictionary.
        face_conductance: Shared pressure-face conductance array.
    """
    n_interior = mesh_data["n_interior_faces"]
    owners = mesh_data["owners"]
    neighbours = mesh_data["neighbours"]
    geo_diff = face_conductance[:n_interior]
    volumetric_face_flux[:n_interior] += geo_diff * (
        kinematic_pressure_correction[owners[:n_interior]]
        - kinematic_pressure_correction[neighbours[:n_interior]]
    )


def _correct_boundary_fluxes(
    volumetric_face_flux, kinematic_pressure_correction, boundaries, owners, face_conductance
):
    """Correct boundary-face fluxes for ``fixedValue`` pressure boundaries.

    Only patches whose pressure BC type is ``fixedValue`` receive a
    correction, computed from the geometric diffusion and wall distance.

    Args:
        volumetric_face_flux:        Face flux array ``(n_faces,)`` (mutated in place).
        kinematic_pressure_correction:    Pressure correction ``(n_elements,)``.
        boundaries: List of boundary patch dictionaries.
        owners:     Owner index array.
        face_conductance: Shared pressure-face conductance array.
    """
    for boundary in boundaries:
        start = boundary["start_face"]
        nf = boundary["n_faces"]
        idx = np.arange(start, start + nf)
        own = owners[idx]
        geo_diff_b = face_conductance[idx]
        boundary_condition_type = boundary.get("pressure_type")
        strategy = BOUNDARIES.strategy(boundary_condition_type, "kinematic_pressure", "flux")
        if strategy is BoundaryStrategy.FIXED_VALUE:
            volumetric_face_flux[idx] += geo_diff_b * kinematic_pressure_correction[own]
        elif strategy is BoundaryStrategy.CYCLIC:
            paired = boundary.get("_paired_cells")
            if paired is None:
                # The mesh-level array is not part of this helper's historical
                # signature; cyclic setup stores the same view on each patch.
                raise ValueError(f"Cyclic patch {boundary.get('name')!r} lacks paired cells")
            volumetric_face_flux[idx] += geo_diff_b * (
                kinematic_pressure_correction[own] - kinematic_pressure_correction[paired]
            )
        elif strategy is BoundaryStrategy.FREESTREAM:
            outflow = boundary.get("_freestream_outflow")
            if outflow is None:
                outflow = volumetric_face_flux[idx] >= 0.0
            volumetric_face_flux[idx] += np.where(
                outflow, geo_diff_b * kinematic_pressure_correction[own], 0.0
            )
        elif strategy not in (
            BoundaryStrategy.ZERO_GRADIENT,
            BoundaryStrategy.FIXED_FLUX_PRESSURE,
            BoundaryStrategy.FIXED_GRADIENT,
            BoundaryStrategy.EMPTY,
        ):
            raise RuntimeError(f"Unhandled pressure flux strategy {strategy!r}")


def _apply_inlet_outlet_bc(velocity, volumetric_face_flux, boundary, owners, n_cells, n_interior):
    """inletOutlet velocity BC: zeroGradient on outflow, fixed value on inflow.

    Per face: outgoing flux (φ ≥ 0) → extrapolate from the owner cell
    (zeroGradient); incoming flux (φ < 0) → impose ``value_velocity_field`` when
    present, otherwise the uniform ``value_velocity`` (the inletValue, default 0).
    The per-face path is required by the FVM--VPM characteristic VPM boundary condition:
    pressure-correction refreshes must not replace its non-uniform VPM boundary-condition trace
    with the uniform freestream.
    """
    start = boundary["start_face"]
    nf = boundary["n_faces"]
    idx = n_cells + (start - n_interior)
    own = owners[start : start + nf]
    if boundary.get("velocity_value_field") is not None:
        inlet_val = np.asarray(boundary["velocity_value_field"], dtype=float)
        if inlet_val.shape != (nf, 3):
            raise ValueError(
                f"Per-face inlet velocity for patch {boundary.get('name')!r} "
                f"must have shape ({nf}, 3), got {inlet_val.shape}"
            )
    else:
        inlet_val = np.asarray(boundary.get("velocity_value", [0.0, 0.0, 0.0]), dtype=float)
        if inlet_val.shape != (3,):
            raise ValueError(
                f"Uniform inlet velocity for patch {boundary.get('name')!r} "
                f"must have shape (3,), got {inlet_val.shape}"
            )
    outflow = boundary.get("_freestream_outflow")
    if outflow is None:
        outflow = volumetric_face_flux[start : start + nf] >= 0.0
    velocity[idx : idx + nf] = np.where(outflow[:, np.newaxis], velocity[own], inlet_val)


def _apply_zero_gradient_bc(
    velocity, volumetric_face_flux, boundary, owners, n_cells, n_interior, boundaries
):
    """Apply a zero-gradient velocity BC (extrapolate from the owner cell).

    Sets the ghost-cell velocity equal to the owner-cell value.

    Args:
        velocity:           Velocity array (mutated in place).
        volumetric_face_flux:         Face flux array.
        boundary:    Boundary patch dictionary.
        owners:      Owner index array.
        n_elements:  Number of interior elements.
        n_interior:  Number of interior faces.
        boundaries:  List of all boundary patches.
    """
    start = boundary["start_face"]
    nf = boundary["n_faces"]
    idx = n_cells + (start - n_interior)
    own = owners[start : start + nf]
    velocity[idx : idx + nf] = velocity[own]


def _apply_cyclic_bc(velocity, boundary, mesh_data, n_cells, n_interior):
    """Copy paired owner values into the cyclic patch ghost layer."""
    start = boundary["start_face"]
    nf = boundary["n_faces"]
    idx = n_cells + (start - n_interior)
    paired = mesh_data["boundary_neighbour_cell"][start : start + nf]
    if np.any(paired < 0):
        raise ValueError(f"Cyclic patch {boundary['name']!r} is not paired")
    velocity[idx : idx + nf] = velocity[paired]


def _apply_fixed_value_bc(velocity, boundary, n_cells, n_interior, strategy):
    """Apply fixedValue or noSlip velocity BC.

    Honours a per-face ``value_velocity_field`` (n_faces_patch, 3) when present (e.g. a
    non-uniform coupler VPM boundary condition), otherwise the uniform ``value_velocity``.
    """
    start = boundary["start_face"]
    nf = boundary["n_faces"]
    idx = n_cells + (start - n_interior)
    if strategy is BoundaryStrategy.NO_SLIP:
        velocity[idx : idx + nf] = [0.0, 0.0, 0.0]
    elif boundary.get("velocity_value_field") is not None:
        velocity[idx : idx + nf] = boundary["velocity_value_field"]
    elif "velocity_value" in boundary:
        velocity[idx : idx + nf] = np.array(boundary["velocity_value"])


def _apply_slip_bc(velocity, boundary, owners, geo_data, n_cells, n_interior):
    """Apply a slip / symmetry / empty velocity BC.

    Removes the component normal to the boundary face from the
    ghost-cell velocity, leaving only the tangential part.

    Args:
        velocity:          Velocity array (mutated in place).
        boundary:   Boundary patch dictionary.
        owners:     Owner index array.
        geo_data:   Geometry dictionary (needs ``face_area_vector``).
        n_elements: Number of interior elements.
        n_interior: Number of interior faces.
    """
    start = boundary["start_face"]
    nf = boundary["n_faces"]
    idx = n_cells + (start - n_interior)
    own = owners[start : start + nf]
    face_area_vector = geo_data["face_area_vector"][start : start + nf]
    # This is deliberately the same array expression as the predictor's
    # empty-boundary update.  In particular, a degenerate face retains the
    # owner value as the old scalar helper did.
    owner_velocity = velocity[own]
    magnitudes = np.linalg.norm(face_area_vector, axis=1)
    valid = magnitudes > 1e-10
    projected = owner_velocity.copy()
    if np.any(valid):
        normals = face_area_vector[valid] / magnitudes[valid, np.newaxis]
        normal_velocity = np.sum(owner_velocity[valid] * normals, axis=1)
        projected[valid] -= normal_velocity[:, np.newaxis] * normals
    velocity[idx : idx + nf] = projected


def _update_velocity_bcs(
    velocity,
    volumetric_face_flux,
    boundaries,
    owners,
    geo_data,
    n_cells,
    n_interior,
    mesh_data=None,
):
    """Update all velocity boundary conditions after a pressure-correction step.

    Dispatches to individual BC handlers based on each patch's
    ``bc_type_velocity``:

    - ``zeroGradient`` → :func:`_apply_zero_gradient_bc`
    - ``inletOutlet``  → :func:`_apply_inlet_outlet_bc`
    - ``fixedValue`` / ``noSlip`` → :func:`_apply_fixed_value_bc`
    - ``empty`` / ``slip`` / ``symmetry`` → :func:`_apply_slip_bc`

    Args:
        velocity:           Velocity array (mutated in place).
        volumetric_face_flux:         Face flux array.
        boundaries:  List of boundary patch dictionaries.
        owners:      Owner index array.
        geo_data:    Geometry dictionary.
        n_elements:  Number of interior elements.
        n_interior:  Number of interior faces.
    """
    for boundary in boundaries:
        bc_type_u = boundary.get("velocity_type") or boundary.get("boundary_condition_type")
        strategy = BOUNDARIES.strategy(bc_type_u, "velocity", "ghost")
        if strategy is BoundaryStrategy.ZERO_GRADIENT:
            _apply_zero_gradient_bc(
                velocity, volumetric_face_flux, boundary, owners, n_cells, n_interior, boundaries
            )
        elif strategy in (BoundaryStrategy.INLET_OUTLET, BoundaryStrategy.FREESTREAM):
            _apply_inlet_outlet_bc(
                velocity, volumetric_face_flux, boundary, owners, n_cells, n_interior
            )
        elif strategy in (BoundaryStrategy.FIXED_VALUE, BoundaryStrategy.NO_SLIP):
            _apply_fixed_value_bc(velocity, boundary, n_cells, n_interior, strategy)
        elif strategy is BoundaryStrategy.NORMAL_VALUE_TANGENTIAL_GRADIENT:
            if mesh_data is None:
                raise ValueError("Mixed velocity boundary update requires mesh_data")
            update_normal_velocity_tangential_gradient_boundary(
                velocity, boundary, mesh_data, geo_data
            )
        elif strategy in (
            BoundaryStrategy.EMPTY,
            BoundaryStrategy.SLIP,
            BoundaryStrategy.SYMMETRY,
        ):
            _apply_slip_bc(velocity, boundary, owners, geo_data, n_cells, n_interior)
        elif strategy is BoundaryStrategy.CYCLIC:
            if mesh_data is None:
                raise ValueError("Cyclic boundary update requires mesh_data")
            _apply_cyclic_bc(velocity, boundary, mesh_data, n_cells, n_interior)
        else:
            raise RuntimeError(f"Unhandled velocity ghost strategy {strategy!r}")


def correct_velocity_and_flux(
    velocity,
    volumetric_face_flux,
    kinematic_pressure_correction,
    momentum_diagonal,
    mesh_data,
    geo_data,
    boundaries,
    density=1.0,
    velocity_relaxation=1.0,
    pressure_relaxation=1.0,
    workspace: PressureCorrectionWorkspace | None = None,
):
    """
    Apply pressure correction to velocity and persistent flux.

    velocity = velocity* - pressure_relaxation * pressure_velocity_coefficient
    * grad(kinematic_pressure')
    volumetric_face_flux = volumetric_face_flux* - DU_f * (grad(kinematic_pressure') . S)

    Uses the unrelaxed diagonal for pressure-velocity-coefficient consistency:
    pressure_velocity_coefficient = cell_volume /
    (momentum_diagonal * velocity_relaxation).
    This matches the Rhie-Chow assembly which also uses the un-relaxed momentum_diagonal.

    ``pressure_relaxation`` scales the **cell-velocity** correction only, never the face
    flux. The pressure equation corrects ``volumetric_face_flux`` with the full flux correction
    (so mass conservation never depends on a relaxation factor), then rebuilds
    the cell velocity from the relaxed pressure. Passing the full
    correction to ``velocity`` while the caller stores only
    ``pressure_relaxation * kinematic_pressure_correction``
    leaves velocity and pressure describing different states, and the outer
    loop then converges to a fixed point that depends on ``pressure_relaxation`` instead of
    the pressure-relaxation-independent solution.
    """
    n_cells = mesh_data["n_cells"]
    n_interior = mesh_data["n_interior_faces"]
    owners = mesh_data["owners"]
    _validate_reference_density(density)

    if workspace is None:
        volumes = geo_data["cell_volume"]
        # Restore un-relaxed diagonal for pressure_velocity_coefficient consistency.
        physical_momentum_diagonal = momentum_diagonal * velocity_relaxation
        pressure_velocity_coefficient = _compute_rhie_chow_coefficients(
            volumes, physical_momentum_diagonal
        )
        face_conductance = _compute_pressure_face_conductance(
            mesh_data, geo_data, pressure_velocity_coefficient
        )
    else:
        pressure_velocity_coefficient = workspace.pressure_velocity_coefficient
        face_conductance = workspace.face_conductance

    # 1. Correct Cell Velocity
    kinematic_pressure_correction_extended = _extend_kinematic_pressure_correction_bcs(
        kinematic_pressure_correction,
        mesh_data,
        boundaries,
        volumetric_face_flux=volumetric_face_flux,
    )
    _grad_fn = gradients._resolve_gradient_fn(geo_data)
    kinematic_pressure_correction_gradient = _grad_fn(
        kinematic_pressure_correction_extended, mesh_data, geo_data
    )
    if kinematic_pressure_correction_gradient.ndim == 3:
        kinematic_pressure_correction_gradient = kinematic_pressure_correction_gradient.squeeze(-1)
    pressure_velocity_coefficient_vector = (
        pressure_velocity_coefficient[:, np.newaxis]
        if np.asarray(pressure_velocity_coefficient).ndim == 1
        else pressure_velocity_coefficient
    )
    velocity[:n_cells] -= (
        pressure_relaxation
        * pressure_velocity_coefficient_vector
        * kinematic_pressure_correction_gradient[:n_cells]
    )

    # 2. Correct Face Fluxes
    _correct_interior_fluxes(
        volumetric_face_flux, kinematic_pressure_correction, mesh_data, face_conductance
    )
    _correct_boundary_fluxes(
        volumetric_face_flux, kinematic_pressure_correction, boundaries, owners, face_conductance
    )

    # 3. Update Velocity BCs
    _update_velocity_bcs(
        velocity,
        volumetric_face_flux,
        boundaries,
        owners,
        geo_data,
        n_cells,
        n_interior,
        mesh_data=mesh_data,
    )

    return velocity, volumetric_face_flux


def _remove_normal_component(velocity_owner, face_vector):
    """Remove the normal component of velocity (slip / symmetry / empty BC).

    Projects out the velocity component along the face normal, leaving
    only the tangential component: ``U_t = velocity − (velocity·n̂) n̂``.

    Args:
        velocity_owner:     Velocity vector at the owner cell ``(3,)``.
        face_vector: Face area vector ``(3,)`` (direction defines normal).

    Returns:
        Tangential velocity vector ``(3,)``.
    """
    norm_Sf = np.linalg.norm(face_vector)
    if norm_Sf > 1e-10:
        n = face_vector / norm_Sf
        normal_velocity_magnitude = np.dot(velocity_owner, n)
        normal_velocity = normal_velocity_magnitude * n
        return velocity_owner - normal_velocity
    return velocity_owner


def _apply_scalar_bc(
    field_values,
    indices,
    owners_b,
    strategy,
    boundary,
    field_name,
    paired_owners=None,
    volumetric_face_flux=None,
):
    """Apply a boundary condition to a scalar field ghost-cell block.

    Zero-gradient BCs (including inlet, outlet, symmetry, empty, slip,
    noSlip) copy the owner value.  ``fixedValue`` sets the prescribed
    boundary value.

    Args:
        field_values: Scalar field array (mutated in place).
        indices:    Ghost-cell indices for this patch.
        owners_b:   Owner cell indices for the boundary faces.
        strategy:   Validated boundary behavior.
        boundary:   Boundary patch dictionary with canonical field-value keys.
        field_name: Field name for value lookup (e.g. ``"kinematic_pressure"``,
            ``"scalar_field"``).
    """
    if strategy in (BoundaryStrategy.ZERO_GRADIENT, BoundaryStrategy.EMPTY):
        field_values[indices] = field_values[owners_b]
    elif strategy in (
        BoundaryStrategy.FIXED_FLUX_PRESSURE,
        BoundaryStrategy.FIXED_GRADIENT,
    ):
        key = (
            "fixed_flux_pressure_delta"
            if strategy is BoundaryStrategy.FIXED_FLUX_PRESSURE
            else "fixed_gradient_delta"
        )
        delta = boundary.get(key)
        if delta is None:
            field_values[indices] = field_values[owners_b]
        else:
            field_values[indices] = field_values[owners_b] + np.asarray(delta, dtype=float)
    elif strategy is BoundaryStrategy.FIXED_VALUE:
        value_key = (
            "kinematic_pressure_value"
            if field_name == "kinematic_pressure"
            else f"{field_name}_value"
        )
        val = boundary.get(value_key)
        if val is None:
            raise ValueError(
                f"Fixed-value {field_name} boundary {boundary.get('name')!r} has no value"
            )
        field_values[indices] = val
    elif strategy is BoundaryStrategy.CYCLIC:
        if paired_owners is None or np.any(paired_owners < 0):
            raise ValueError(f"Cyclic patch {boundary.get('name')!r} is not paired")
        field_values[indices] = field_values[paired_owners]
    elif strategy is BoundaryStrategy.FREESTREAM:
        outflow = boundary.get("_freestream_outflow")
        if outflow is None:
            if volumetric_face_flux is None:
                raise ValueError("Freestream scalar update requires face fluxes")
            outflow = np.asarray(volumetric_face_flux) >= 0.0
        value_key = (
            "kinematic_pressure_value"
            if field_name == "kinematic_pressure"
            else f"{field_name}_value"
        )
        val = boundary.get(value_key)
        if val is None:
            raise ValueError(
                f"Freestream {field_name} boundary {boundary.get('name')!r} has no value"
            )
        if field_name == "kinematic_pressure" and boundary.get(
            "_directional_fixed_flux_pressure", False
        ):
            delta = boundary.get("fixed_flux_pressure_delta")
            owner_value = (
                field_values[owners_b]
                if delta is None
                else field_values[owners_b] + np.asarray(delta)
            )
            field_values[indices] = np.where(outflow, val, owner_value)
        else:
            field_values[indices] = np.where(outflow, val, field_values[owners_b])
    else:
        raise ValueError(
            f"Unsupported scalar boundary strategy {strategy!r} "
            f"for {field_name} on patch {boundary.get('name')!r}"
        )


def update_scalar_boundaries(
    field_values, mesh_data, boundaries, field_name="kinematic_pressure", volumetric_face_flux=None
):
    """
    Update boundary values (ghost cells) for a scalar field.

    Args:
        field_values: Scalar field (n_elements + n_boundary)
        mesh_data: Mesh connectivity
        boundaries: Boundary conditions
        field_name: Name of field (for example ``kinematic_pressure``) to check BC type
    """
    n_cells = mesh_data["n_cells"]
    n_interior = mesh_data["n_interior_faces"]
    owners = mesh_data["owners"]

    for boundary in boundaries:
        start = boundary["start_face"]
        n_bfaces = boundary["n_faces"]

        # Boundary element indices
        b_elem_start = n_cells + (start - n_interior)
        b_elem_indices = np.arange(b_elem_start, b_elem_start + n_bfaces)
        owners_b = owners[start : start + n_bfaces]
        paired = mesh_data.get("boundary_neighbour_cell")
        paired_owners = None if paired is None else paired[start : start + n_bfaces]
        patch_flux = (
            None
            if volumetric_face_flux is None
            else np.asarray(volumetric_face_flux)[start : start + n_bfaces]
        )

        # Get BC type
        boundary_type_key = (
            "pressure_type" if field_name == "kinematic_pressure" else f"{field_name}_type"
        )
        boundary_condition_type = boundary.get(boundary_type_key) or boundary.get(
            "boundary_condition_type"
        )
        registry_field = "kinematic_pressure" if field_name == "kinematic_pressure" else "scalar"
        strategy = BOUNDARIES.strategy(boundary_condition_type, registry_field, "ghost")
        _apply_scalar_bc(
            field_values,
            b_elem_indices,
            owners_b,
            strategy,
            boundary,
            field_name,
            paired_owners=paired_owners,
            volumetric_face_flux=patch_flux,
        )


class SIMPLESolver:
    """SIMPLE algorithm for incompressible Navier–Stokes.

    Semi-Implicit Method for Pressure-Linked Equations: solves the
    steady incompressible Navier–Stokes equations through a predictor–
    corrector loop that alternates between a momentum solve and a
    pressure-correction Poisson solve.

    The algorithm iterates until ``max_iterations`` is reached or the residuals
    fall below ``tolerance``.  Under-relaxation is applied through
    ``velocity_relaxation`` (velocity) and ``pressure_relaxation`` (pressure).

    This class also serves as the base for the transient :class:`PIMPLESolver`.

    References
    ----------
    - Patankar, S. V. and Spalding, D. B. "A calculation procedure for
      heat, mass and momentum transfer in three-dimensional parabolic
      flows." *Int. J. Heat Mass Transfer*, 15(10):1787–1806, 1972.
    - Ferziger, J. H. and Perić, M. *Computational Methods for Fluid
      Dynamics*, 3rd ed., Springer, 2002 (Chapter 8).

    Examples
    --------
    >>> solver = SIMPLESolver(mesh_data, geo_data, boundaries)  # doctest: +SKIP
    >>> velocity, kinematic_pressure, volumetric_face_flux, converged = solver.solve(  # doctest: +SKIP
    ...     initial_velocity, initial_kinematic_pressure, density=1.225,
    ...     kinematic_viscosity=1.5e-5
    ... )
    """

    def __init__(self, mesh_data, geo_data, boundaries, params=None):
        """Initialise the SIMPLE solver.

        Args:
            mesh_data: Mesh connectivity dictionary.
            geo_data: Geometric quantities dictionary.
            boundaries: List of boundary condition dictionaries.
            params: Optional dict of solver parameters overriding defaults.
                Supported keys: ``velocity_relaxation``, ``pressure_relaxation``, ``max_iterations``,
                ``tolerance``, ``convection_scheme``, ``linear_solver``.
        """
        self.mesh_data = mesh_data
        self.geo_data = geo_data
        self.boundaries = boundaries

        # Default parameters
        self.params: dict[str, Any] = {
            "velocity_relaxation": 0.7,
            "pressure_relaxation": 0.3,
            "max_iterations": 100,
            "tolerance": 1e-6,
            "convection_scheme": "deferred",
            "linear_solver": "spsolve",
        }

        if params:
            self.params.update(params)

        # Optional immersed-boundary forcing (set via FVMSolver.set_immersed_bodies).
        self.ibm = None
        self.residuals = []
        self.last_linear_results = ()
        self.last_outer_diagnostics = ()
        # Momentum and pressure are sequential and share one static-topology
        # workspace. Its CSR values are overwritten between equations.
        self._momentum_matrix_workspace = matrix_assembly.MatrixAssemblyWorkspace.create(mesh_data)
        self._pressure_matrix_workspace = self._momentum_matrix_workspace
        self._pressure_boundary_layout = build_pressure_boundary_layout(
            boundaries,
            mesh_data["n_interior_faces"],
            mesh_data["n_faces"],
        )

    def close(self) -> None:
        """Release solver-owned transient preconditioner cache entries."""
        from .linear_interface import clear_linear_solver_caches

        namespace = self._momentum_matrix_workspace.cache_namespace
        clear_linear_solver_caches(("momentum", namespace))
        clear_linear_solver_caches(("kinematic_pressure", namespace))

    def step(
        self,
        velocity,
        kinematic_pressure,
        volumetric_face_flux,
        velocity_old=None,
        time_step_size=None,
        density=1.0,
        kinematic_viscosity=0.01,
        velocity_older=None,
        source_explicit=None,
        source_implicit=None,
        volumetric_face_flux_old=None,
        volumetric_face_flux_older=None,
    ):
        """Perform one SIMPLE pressure–velocity correction.

        ``velocity_older`` and the flux histories are accepted for interface parity
        with the transient driver but are unused by steady SIMPLE.
        ``source_explicit``/``source_implicit`` are optional volumetric momentum
        sources forwarded to the momentum predictor.

        Args:
            velocity: Cell and boundary-ghost velocity [m/s], shape
                ``(n_cells_with_ghosts, 3)``.
            kinematic_pressure: Kinematic pressure ``kinematic_pressure/ρ`` [m²/s²], shape
                ``(n_cells_with_ghosts,)``.
            volumetric_face_flux: Volumetric face flux ``velocity·Sf`` [m³/s], shape
                ``(n_faces,)``.
            velocity_old: Previous velocity [m/s], unused by steady SIMPLE.
            time_step_size: Time-step size [s], normally ``None`` for steady SIMPLE.
            density: Positive constant reference density [kg/m³]. It cancels
                from the kinematic-pressure flow equations.
            kinematic_viscosity: Positive kinematic viscosity [m²/s].
            velocity_older: Older velocity [m/s], unused by steady SIMPLE.
            source_explicit: Explicit acceleration source [m/s²].
            source_implicit: Non-negative implicit source coefficient [1/s].

        Returns:
            tuple: Updated ``(velocity, kinematic_pressure, volumetric_face_flux, residuals)``. The three field arrays
            retain the shapes and units described above.
        """
        # 1. Solve momentum predictor
        velocity_star, momentum_diagonal, momentum_diagnostics = momentum.solve_momentum_predictor(
            velocity,
            kinematic_pressure,
            volumetric_face_flux,
            density,
            kinematic_viscosity,
            self.mesh_data,
            self.geo_data,
            self.boundaries,
            convection_scheme=self.params["convection_scheme"],
            solver=self.params.get("momentum_solver") or self.params["linear_solver"],
            under_relaxation=self.params["velocity_relaxation"],
            time_step_size=time_step_size,  # Use time_step_size if provided (e.g. for PIMPLE)
            source_explicit=source_explicit,
            source_implicit=source_implicit,
            linear_backend=self.params.get("_linear_backend", "scipy"),
            parallel_context=self.params.get("_parallel_context"),
            failure_policy=self.params.get("linear_failure_policy", "raise"),
            log_sink=self.params.get("_logger"),
            momentum_tolerance=self.params.get("momentum_tolerance", 1e-4),
            maxiter=self.params.get("momentum_max_iterations", 1000),
            reuse_ilu=self.params.get("reuse_ilu", False),
            ilu_drop_tolerance=self.params.get("ilu_drop_tolerance", 1e-4),
            ilu_fill_factor=self.params.get("ilu_fill_factor", 10),
            ilu_reuse_tolerance=self.params.get("ilu_reuse_tolerance"),
            matrix_workspace=self._momentum_matrix_workspace,
            operator_backend=self.params.get("_operator_backend", "numpy"),
            return_diagnostics=True,
        )

        # 2. Solve pressure correction
        pressure_constraint = _resolve_pressure_constraint(self.params)
        has_pressure_nullspace = _pressure_requires_constraint(
            self.boundaries, velocity_star, self.mesh_data, self.geo_data
        )
        pressure_matrix, pressure_right_hand_side, volumetric_face_flux_star = (
            assemble_pressure_correction_equation_rhie_chow(
                velocity_star,
                momentum_diagonal,
                kinematic_pressure,
                density,
                self.mesh_data,
                self.geo_data,
                self.boundaries,
                velocity_relaxation=self.params["velocity_relaxation"],
                pressure_constraint=pressure_constraint,
                matrix_workspace=self._pressure_matrix_workspace,
                operator_backend=self.params.get("_operator_backend", "numpy"),
            )
        )

        kinematic_pressure_correction, kinematic_pressure_result = solve_linear_system(
            pressure_matrix,
            pressure_right_hand_side,
            method=self.params.get("pressure_solver") or self.params["linear_solver"],
            equation_type="kinematic_pressure",
            tol=self.params.get("pressure_tolerance", 1e-8),
            maxiter=self.params.get("pressure_max_iterations", 500),
            backend=self.params.get("_linear_backend", "scipy"),
            parallel_context=self.params.get("_parallel_context"),
            failure_policy=self.params.get("linear_failure_policy", "raise"),
            log_sink=self.params.get("_logger"),
            nullspace=(
                "constant"
                if has_pressure_nullspace and pressure_constraint == "nullspace"
                else None
            ),
            return_info=True,
        )

        # 3. Correct velocity and flux
        # Calculate residual before in-place modification of velocity_star
        velocity_increment = np.linalg.norm(
            velocity_star[: self.mesh_data["n_cells"]] - velocity[: self.mesh_data["n_cells"]]
        ) / (np.linalg.norm(velocity[: self.mesh_data["n_cells"]]) + 1e-10)

        velocity, volumetric_face_flux = correct_velocity_and_flux(
            velocity_star,
            volumetric_face_flux_star,
            kinematic_pressure_correction,
            momentum_diagonal,
            self.mesh_data,
            self.geo_data,
            self.boundaries,
            density=density,
            velocity_relaxation=self.params["velocity_relaxation"],
            pressure_relaxation=self.params["pressure_relaxation"],
        )

        # 4. Update pressure
        kinematic_pressure[: self.mesh_data["n_cells"]] += (
            self.params["pressure_relaxation"] * kinematic_pressure_correction
        )

        # Update pressure boundaries
        update_scalar_boundaries(
            kinematic_pressure,
            self.mesh_data,
            self.boundaries,
            field_name="kinematic_pressure",
            volumetric_face_flux=volumetric_face_flux,
        )

        # 5. Residuals
        self.last_kinematic_pressure_residual = normalized_residual(
            pressure_matrix, kinematic_pressure_correction, pressure_right_hand_side
        )
        self.last_velocity_residual = max(
            (values["final_residual"] for values in momentum_diagnostics.values()),
            default=0.0,
        )
        continuity = field_diagnostics.compute_continuity_error(
            volumetric_face_flux, self.mesh_data, self.geo_data
        )
        volumes = self.geo_data["cell_volume"]
        max_continuity_error = float(np.max(np.abs(continuity) / (volumes + 1e-30)))
        self.last_linear_results = tuple(
            values["linear_result"] for values in momentum_diagnostics.values()
        ) + (kinematic_pressure_result,)
        self.last_outer_diagnostics = (
            OuterCorrectorDiagnostics(
                index=0,
                velocity_residual=self.last_velocity_residual,
                kinematic_pressure_residual=self.last_kinematic_pressure_residual,
                max_continuity_error=max_continuity_error,
            ),
        )

        residuals = {
            "kinematic_pressure": self.last_kinematic_pressure_residual,
            "velocity": self.last_velocity_residual,
            "velocity_increment": velocity_increment,
        }
        residuals.update(
            {
                f"velocity_{component}": values["final_residual"]
                for component, values in momentum_diagnostics.items()
            }
        )
        return velocity, kinematic_pressure, volumetric_face_flux, residuals

    def solve(
        self, initial_velocity, initial_kinematic_pressure, density=1.0, kinematic_viscosity=0.01
    ):
        """Solve a steady incompressible flow using the SIMPLE algorithm.

        Iterates over ``max_iterations`` SIMPLE steps, computing the momentum
        predictor, pressure correction, and velocity/pressure update at
        each iteration.  Convergence is declared when both the pressure
        and velocity residuals fall below ``tolerance``.

        Args:
            initial_velocity: Initial velocity [m/s], shape ``(n_total, 3)``.
            initial_kinematic_pressure: Initial kinematic pressure ``kinematic_pressure/ρ`` [m²/s²], shape
                ``(n_total,)``.
            density: Positive constant reference density [kg/m³]. It cancels
                from the kinematic-pressure flow equations. Defaults to 1.0.
            kinematic_viscosity: Kinematic viscosity [m²/s]. Defaults to 0.01.

        Returns:
            Tuple ``(velocity, kinematic_pressure, volumetric_face_flux, converged)`` where *converged* is a bool.
        """
        velocity = initial_velocity.copy()
        kinematic_pressure = initial_kinematic_pressure.copy()
        logger: Any = self.params.get("_logger")

        if logger is not None:
            logger.section(
                "SIMPLE SOLVER",
                [
                    ("Maximum iterations", str(self.params["max_iterations"])),
                    ("Tolerance", f"{self.params['tolerance']:.3e}"),
                    (
                        "Under-relaxation",
                        f"velocity={self.params['velocity_relaxation']}, kinematic_pressure={self.params['pressure_relaxation']}",
                    ),
                ],
            )

        # Initialize Flux (volumetric_face_flux) if not provided
        from ..assemble import convection

        volumetric_face_flux = convection.compute_volumetric_face_flux(
            velocity, self.mesh_data, self.geo_data
        )

        for iteration in range(int(self.params["max_iterations"])):
            velocity, kinematic_pressure, volumetric_face_flux, residuals = self.step(
                velocity,
                kinematic_pressure,
                volumetric_face_flux,
                density=density,
                kinematic_viscosity=kinematic_viscosity,
            )

            kinematic_pressure_residual = self.last_kinematic_pressure_residual
            velocity_residual = residuals["velocity_increment"]
            continuity = self.last_outer_diagnostics[-1].max_continuity_error

            self.residuals.append(
                {
                    "iter": iteration,
                    "kinematic_pressure_residual": kinematic_pressure_residual,
                    "velocity_residual": velocity_residual,
                    "max_continuity_error": continuity,
                }
            )

            if logger is not None and (
                iteration % 10 == 0 or kinematic_pressure_residual < self.params["tolerance"]
            ):
                logger.message(
                    f"  Iter {iteration:3d}: kinematic_pressure_residual={kinematic_pressure_residual:.3e}, "
                    f"Δvelocity={velocity_residual:.3e}, continuity={continuity:.3e}"
                )

            if (
                kinematic_pressure_residual < self.params["tolerance"]
                and velocity_residual < self.params["tolerance"]
                and continuity < self.params["tolerance"]
            ):
                if logger is not None:
                    logger.info(
                        f"component=SIMPLE status=converged iterations={iteration} "
                        f"kinematic_pressure_residual={kinematic_pressure_residual:.3e} "
                        f"velocity_increment={velocity_residual:.3e} continuity={continuity:.3e}"
                    )
                return velocity, kinematic_pressure, volumetric_face_flux, True

        if logger is not None:
            logger.warning(
                f"component=SIMPLE status=not_converged "
                f"iterations={self.params['max_iterations']} "
                f"kinematic_pressure_residual={kinematic_pressure_residual:.3e} "
                f"velocity_increment={velocity_residual:.3e} continuity={continuity:.3e}"
            )
        return velocity, kinematic_pressure, volumetric_face_flux, False
