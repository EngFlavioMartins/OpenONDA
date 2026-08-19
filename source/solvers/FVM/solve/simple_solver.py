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

    ``DU`` (the inverse diagonal of the momentum matrix at each cell) and
    ``face_conductance`` (the interpolated face flux coefficient) are
    computed once per PIMPLE corrector and reused by both the pressure
    Poisson assembly and the subsequent face-flux and velocity correction.

    Attributes
    ----------
    DU : np.ndarray
        Inverse of the momentum matrix diagonal. The production momentum
        path stores the shared scalar diagonal as ``(n_cells,)``; legacy
        callers may supply component diagonals as ``(n_cells, 3)``.
    face_conductance : np.ndarray
        Interpolated face flux coefficient, shape ``(n_faces,)``.
    matrix : Any or None
        Pressure operator assembled from ``face_conductance``. It is reusable
        for every pressure/non-orthogonal correction belonging to the same
        momentum predictor because ``A_U`` is unchanged within that loop.
    """

    DU: np.ndarray
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


def _pressure_requires_constraint(boundaries, U_star, mesh_data, geo_data) -> bool:
    """Return whether the assembled pressure operator has a constant null space."""
    n_elements = mesh_data["n_elements"]
    n_interior = mesh_data["n_interior_faces"]
    for boundary in boundaries:
        bc_type = boundary.get("bc_type_p")
        strategy = BOUNDARIES.strategy(bc_type, "p", "pressure")
        if strategy is BoundaryStrategy.FIXED_VALUE:
            local_requires_constraint = False
            break
        if strategy is BoundaryStrategy.FREESTREAM:
            start = boundary["startFace"]
            nf = boundary["nFaces"]
            ghosts = n_elements + np.arange(start - n_interior, start - n_interior + nf)
            flux = np.sum(U_star[ghosts] * geo_data["face_sf"][start : start + nf], axis=1)
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
    U_old,
    U_old_old,
    phi_old,
    phi_old_old,
    time_step_size,
    mesh_data,
    geo_data,
    boundaries,
    ddt_scheme,
):
    r"""Return the transient Rhie-Chow flux correction.

    The correction is added to the predicted face flux before the pressure
    solve. For ``backward`` it is

    .. math::

        C \, \frac{1}{\Delta t}\Big[
            (c_0 \phi^{n} - c_{00}\phi^{n-1})
          - S_f \cdot \overline{(c_0 U^{n} - c_{00} U^{n-1})} \Big]

    with the ``backward`` coefficients ``c_0 = 2``, ``c_{00}`` = 0.5 at constant
    ``time_step_size`` (Euler uses ``c_0 = 1``, ``c_{00} = 0``).

    It exists because the Rhie-Chow damping is proportional to ``rAU``, and in
    a transient run ``rAU`` is dominated by the ``V/dt`` of the time
    derivative.  Without this term the face flux carries a spurious
    fourth-order pressure dissipation that scales with ``time_step_size`` and never
    vanishes under mesh refinement; the correction removes exactly the part of
    the flux-velocity mismatch that the time derivative introduced, leaving the
    convection/diffusion part.  In a bluff-body wake that surplus dissipation
    is enough to hold a shear layer steady.

    ``C`` is the time-step coupling coefficient
    ``C = 1 - min(|phiCorr| / (|phi| + SMALL), 1)``, which switches the
    correction off wherever the flux and the interpolated velocity have already
    fully decoupled, and ``C = 0`` on patches whose velocity condition fixes a
    value (inlets, no-slip walls).

    Returns a volumetric flux increment per unit ``rAU`` — the caller scales it
    by the face-interpolated ``DU``.
    """
    if phi_old is None or time_step_size is None or U_old is None:
        return None

    n_faces = mesh_data["n_faces"]
    n_interior = mesh_data["n_interior_faces"]
    n_elements = mesh_data["n_elements"]

    if str(ddt_scheme).lower() in ("backward", "bdf2") and (
        U_old_old is not None and phi_old_old is not None
    ):
        coefft0, coefft00 = 2.0, 0.5
    else:
        coefft0, coefft00 = 1.0, 0.0

    phi_old = np.asarray(phi_old, dtype=np.float64)
    U_old = np.asarray(U_old, dtype=np.float64)
    U_history = U_old.copy()
    U_history *= coefft0
    if coefft00 != 0.0:
        U_history -= coefft00 * np.asarray(U_old_old, dtype=np.float64)

    # Sf . interpolate(U_history), matching fvc::dotInterpolate.
    owners = mesh_data["owners"]
    neighbours = mesh_data["neighbours"]
    weights = geo_data["face_weights"]
    face_sf = geo_data["face_sf"]

    correction = np.empty(n_faces, dtype=np.float64)
    chunk_size = 250_000
    for start in range(0, n_interior, chunk_size):
        stop = min(start + chunk_size, n_interior)
        own = owners[start:stop]
        nei = neighbours[start:stop]
        w = weights[start:stop, np.newaxis]
        history_face = w * U_history[nei] + (1.0 - w) * U_history[own]
        old_face = w * U_old[nei] + (1.0 - w) * U_old[own]
        history_flux = coefft0 * phi_old[start:stop]
        if coefft00 != 0.0:
            history_flux -= coefft00 * np.asarray(phi_old_old)[start:stop]
        phi_corr = history_flux - np.einsum("ij,ij->i", history_face, face_sf[start:stop])
        reference = phi_old[start:stop] - np.einsum("ij,ij->i", old_face, face_sf[start:stop])
        coupling = 1.0 - np.minimum(
            np.abs(reference) / (np.abs(phi_old[start:stop]) + np.finfo(np.float64).tiny),
            1.0,
        )
        correction[start:stop] = coupling * phi_corr / float(time_step_size)

    for start in range(n_interior, n_faces, chunk_size):
        stop = min(start + chunk_size, n_faces)
        ghosts = n_elements + np.arange(start - n_interior, stop - n_interior)
        history_flux = coefft0 * phi_old[start:stop]
        if coefft00 != 0.0:
            history_flux -= coefft00 * np.asarray(phi_old_old)[start:stop]
        phi_corr = history_flux - np.einsum("ij,ij->i", U_history[ghosts], face_sf[start:stop])
        reference = phi_old[start:stop] - np.einsum("ij,ij->i", U_old[ghosts], face_sf[start:stop])
        coupling = 1.0 - np.minimum(
            np.abs(reference) / (np.abs(phi_old[start:stop]) + np.finfo(np.float64).tiny),
            1.0,
        )
        correction[start:stop] = coupling * phi_corr / float(time_step_size)

    for boundary in boundaries:
        strategy = BOUNDARIES.strategy(boundary.get("bc_type_velocity"), "U", "ghost")
        if strategy in (
            BoundaryStrategy.FIXED_VALUE,
            BoundaryStrategy.NO_SLIP,
            BoundaryStrategy.NORMAL_VALUE_TANGENTIAL_GRADIENT,
        ):
            start = boundary["startFace"]
            correction[start : start + boundary["nFaces"]] = 0.0

    return correction


def _compute_rhie_chow_coefficients(volumes, A_U):
    """Compute the DU coefficients for Rhie-Chow interpolation.

    ``DU = V / A_U`` converts pressure-gradient cell values to velocity
    corrections: ``ΔU = −DU · ∇p'``.

    Args:
        volumes: Cell volumes ``(n_elements,)``.
        A_U:     Diagonal coefficients from the momentum equation
                 ``(n_elements, 3)`` (per component).

    Returns:
        DU array ``(n_elements, 3)``, with a small regulariser to avoid
        division by zero.
    """
    diagonal = np.asarray(A_U, dtype=np.float64)
    if diagonal.ndim == 1:
        return volumes / (diagonal + 1e-10)
    return volumes[:, np.newaxis] / (diagonal + 1e-10)


def _validate_reference_density(rho) -> float:
    """Validate the scalar reference density used for dimensional forces.

    Constant density cancels from the kinematic-pressure flow equations, but
    remains part of the public setup because dimensional forces scale with it.
    """
    density = np.asarray(rho, dtype=np.float64)
    if density.ndim != 0:
        raise ValueError("constant-density FVM requires rho to be a scalar")
    value = float(density.item())
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("rho must be finite and positive")
    return value


def _compute_pressure_face_conductance(mesh_data, geo_data, DU):
    """Return the geometric Rhie--Chow conductance for every face.

    The pressure matrix and the post-solve flux correction must use exactly
    the same conductance.  Keeping this calculation in one function prevents
    the non-orthogonal inconsistency that previously used ``Sf·e`` during
    assembly but ``|Sf|`` during correction.

    ``DU`` is the cell-centred diagonal pressure-to-velocity coefficient.  It
    is linearly interpolated on interior faces and taken from the owner on
    boundary faces. Because pressure is kinematic and ``phi`` is volumetric
    flux, density does not enter this conductance.
    """
    n_faces = mesh_data["n_faces"]
    n_interior = mesh_data["n_interior_faces"]
    owners = mesh_data["owners"]
    neighbours = mesh_data["neighbours"]

    scalar_diagonal = np.asarray(DU).ndim == 1
    sf = geo_data["face_sf"]
    cf_vec = geo_data["face_cf_vector"]
    weights = geo_data["face_weights"]
    boundary_neighbours = np.asarray(
        mesh_data.get("boundary_neighbours", np.full(n_faces, -1, dtype=np.int32))
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
            neighbour[boundary_start - start :] = boundary_neighbours[boundary_start:stop]
        coupled = neighbour >= 0

        du_face = np.asarray(DU[own], dtype=np.float64).copy()
        if np.any(coupled):
            w = weights[face_slice][coupled]
            if scalar_diagonal:
                du_face[coupled] = w * DU[neighbour[coupled]] + (1.0 - w) * DU[own[coupled]]
            else:
                w_vector = w[:, np.newaxis]
                du_face[coupled] = (
                    w_vector * DU[neighbour[coupled]] + (1.0 - w_vector) * DU[own[coupled]]
                )
        if scalar_diagonal:
            d_eff = du_face
        else:
            normal = sf_block / mag_sf[:, np.newaxis]
            d_eff = np.sum(normal * normal * du_face, axis=1)
        edge = cf_block / mag_cf[:, np.newaxis]
        orthogonal_area = np.sum(sf_block * edge, axis=1)
        conductance[face_slice] = d_eff * orthogonal_area / mag_cf

    if np.any(conductance < -1e-14):
        raise ValueError(
            "Negative pressure-face conductance; check face orientation and mesh geometry"
        )
    return np.maximum(conductance, 0.0)


def _update_fixed_flux_pressure_boundaries(
    p,
    U_star,
    DU,
    mesh_data,
    geo_data,
    boundaries,
    grad_p=None,
    pressure_free_face_flux=None,
):
    """Update ``fixedFluxPressure`` from the pressure-free face flux."""
    n_elements = mesh_data["n_elements"]
    n_interior = mesh_data["n_interior_faces"]
    owners = mesh_data["owners"]
    face_sf = geo_data["face_sf"]
    face_cf = geo_data["face_cf_vector"]
    _grad_fn = gradients._resolve_gradient_fn(geo_data)

    fixed_flux_patches = []
    for boundary in boundaries:
        strategy = BOUNDARIES.strategy(boundary.get("bc_type_p"), "p", "ghost")
        if strategy is BoundaryStrategy.FIXED_FLUX_PRESSURE or (
            strategy is BoundaryStrategy.FREESTREAM
            and boundary.get("_directional_fixed_flux_pressure", False)
        ):
            fixed_flux_patches.append(boundary)
    if not fixed_flux_patches:
        return grad_p

    if grad_p is None:
        grad_p = _grad_fn(p, mesh_data, geo_data)
        if grad_p.ndim == 3:
            grad_p = grad_p.squeeze(-1)
    assert grad_p is not None

    U_hbya = None
    if pressure_free_face_flux is not None:
        pressure_free_face_flux = np.asarray(pressure_free_face_flux, dtype=float)
        if pressure_free_face_flux.shape != (mesh_data["n_faces"],):
            raise ValueError("pressure_free_face_flux must have one value per face")
    else:
        DU_vector = DU[:, np.newaxis] if np.asarray(DU).ndim == 1 else DU
        U_hbya = U_star[:n_elements] + DU_vector * grad_p[:n_elements]

    changed = False
    for boundary in fixed_flux_patches:
        start = boundary["startFace"]
        nf = boundary["nFaces"]
        ghost = n_elements + (start - n_interior) + np.arange(nf)
        own = owners[start : start + nf]
        if boundary.get("fixed_flux_pressure_external", False):
            delta = boundary.get("fixed_flux_pressure_delta")
            if delta is not None:
                p[ghost] = p[own] + np.asarray(delta, dtype=float)
                changed = True
            continue

        sf = face_sf[start : start + nf]
        mag_sf = np.linalg.norm(sf, axis=1)
        normal = sf / mag_sf[:, np.newaxis]
        dr = face_cf[start : start + nf]
        normal_distance = np.einsum("ij,ij->i", dr, normal)
        if np.asarray(DU).ndim == 1:
            D_normal = DU[own]
        else:
            D_normal = np.einsum("ij,ij->i", DU[own], normal * normal)
        phi_target = np.einsum("ij,ij->i", U_star[ghost], sf)
        if pressure_free_face_flux is None:
            assert U_hbya is not None
            phi_hbya = np.einsum("ij,ij->i", U_hbya[own], sf)
        else:
            phi_hbya = pressure_free_face_flux[start : start + nf]
        pressure_flux_coefficient = mag_sf * D_normal
        dpdn = (phi_hbya - phi_target) / np.maximum(pressure_flux_coefficient, 1.0e-30)
        delta = dpdn * normal_distance
        boundary["fixed_flux_pressure_delta"] = delta
        if boundary.get("_directional_fixed_flux_pressure", False):
            outflow = np.asarray(boundary["_fixed_freestream_outflow"], dtype=bool)
            p[ghost] = np.where(outflow, float(boundary.get("value_p", 0.0)), p[own] + delta)
        else:
            p[ghost] = p[own] + delta
        changed = True

    if changed:
        grad_p = _grad_fn(p, mesh_data, geo_data)
        if grad_p.ndim == 3:
            grad_p = grad_p.squeeze(-1)
    return grad_p


@njit(cache=True)
def _process_boundary_faces_jit(
    n_boundary_faces,
    n_interior,
    n_elements,
    owners,
    face_sf,
    face_cf_vector,
    U_star,
    DU0,
    DU1,
    DU2,
    grad_p,
    p,
    bc_type_codes,
    p_boundary_values,
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
        face_sf:              Face area vectors ``(n_faces, 3)``.
        face_cf_vector:       Centre-to-centre vectors ``(n_faces, 3)``.
        U_star:               Predicted velocity ``(n_total, 3)``.
        DU0, DU1, DU2:       Component views of the Rhie-Chow coefficients.
                              For a scalar momentum diagonal all three
                              arguments are the same one-dimensional array.
        grad_p:               Pressure gradient ``(n_total, 3)``.
        p:                    Pressure field ``(n_total,)``.
        bc_type_codes:        Integer-coded BC: 0=zeroGradient, 1=fixedValue, 2=empty.
        p_boundary_values:    Fixed pressure values for fixedValue patches.
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

        Sf0, Sf1, Sf2 = face_sf[i_face, 0], face_sf[i_face, 1], face_sf[i_face, 2]

        bc_code = bc_type_codes[i]

        b_elem_idx = n_elements + (i_face - n_interior)
        Ub0, Ub1, Ub2 = U_star[b_elem_idx, 0], U_star[b_elem_idx, 1], U_star[b_elem_idx, 2]
        velocity_flux = Ub0 * Sf0 + Ub1 * Sf1 + Ub2 * Sf2

        # zeroGradient pressure, including the inflow side of freestream
        if bc_code == 0 or (bc_code == 3 and velocity_flux < 0.0):
            flux_vf_out[i] = Ub0 * Sf0 + Ub1 * Sf1 + Ub2 * Sf2
            continue

        # fixedValue pressure, including the outflow side of freestream
        if bc_code == 1 or bc_code == 3:
            CF0, CF1, CF2 = (
                face_cf_vector[i_face, 0],
                face_cf_vector[i_face, 1],
                face_cf_vector[i_face, 2],
            )
            mag_CF = (CF0 * CF0 + CF1 * CF1 + CF2 * CF2) ** 0.5

            e0 = CF0 / (mag_CF + 1e-10)
            e1 = CF1 / (mag_CF + 1e-10)
            e2 = CF2 / (mag_CF + 1e-10)

            du0, du1, du2 = DU0[own], DU1[own], DU2[own]
            gp0, gp1, gp2 = grad_p[own, 0], grad_p[own, 1], grad_p[own, 2]

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
                term_interp = du0 * gp0 * Sf0 + du1 * gp1 * Sf1 + du2 * gp2 * Sf2

                # Compact pressure drive
                val = p_boundary_values[i]
                term_compact = cf * p[own] + ff * val

                # Non-orthogonal correction
                k0 = Sf0 - sf_dot_e * e0
                k1 = Sf1 - sf_dot_e * e1
                k2 = Sf2 - sf_dot_e * e2
                k_norm = (k0 * k0 + k1 * k1 + k2 * k2) ** 0.5
                if k_norm > 1e-12:
                    flux_nonortho = k0 * du0 * gp0 + k1 * du1 * gp1 + k2 * du2 * gp2
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
        strategy = BOUNDARIES.strategy(boundary.get("bc_type_p"), "p", "pressure")
        signature.append((int(boundary["startFace"]), int(boundary["nFaces"]), strategy.name))
    return tuple(signature)


def _pressure_boundary_matrix_is_reusable(boundaries) -> bool:
    """Return whether pressure boundary coefficients stay fixed this step.

    Ordinary freestream faces switch between pressure Dirichlet and Neumann
    with the evolving face flux. A directional coupling patch carries an
    immutable geometric outflow mask, so its pressure matrix is just as static
    as a conventional fixed-value/fixed-gradient layout.
    """
    for boundary in boundaries:
        strategy = BOUNDARIES.strategy(boundary.get("bc_type_p"), "p", "pressure")
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
        start = int(boundary["startFace"])
        nf = int(boundary["nFaces"])
        local = slice(start - n_interior, start - n_interior + nf)
        strategy = BOUNDARIES.strategy(boundary.get("bc_type_p"), "p", "pressure")
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
        start = int(boundary["startFace"])
        nf = int(boundary["nFaces"])
        local = slice(start - n_interior, start - n_interior + nf)
        field = boundary.get("value_p_field")
        if field is not None:
            field_values = np.asarray(field, dtype=np.float64)
            if field_values.shape != (nf,):
                raise ValueError(
                    f"Per-face pressure value for patch {boundary.get('name')!r} "
                    f"must have shape ({nf},), got {field_values.shape}"
                )
            values[local] = field_values
        else:
            val = boundary.get("value_p", boundary.get("value", 0.0))
            if val is not None:
                values[local] = val
    return values


def _build_boundary_face_arrays(boundaries, n_interior, n_faces, layout=None):
    """Return pressure boundary arrays, preserving the legacy helper API.

    Returns
    -------
    bc_type_codes : ndarray
        0=zeroGradient, 1=fixedValue, 2=empty, 3=freestream
    p_boundary_values : ndarray
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


def adjust_boundary_flux_for_continuity(flux_vf, boundaries, mesh_data, n_interior, n_faces):
    """Rescale adjustable outflow boundary fluxes so the net is zero.

    An incompressible domain requires ``∮ phi·dS = 0``; otherwise the pressure
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

    ``flux_vf`` is mutated in place.
    """
    floating = (
        BoundaryStrategy.FREESTREAM,
        BoundaryStrategy.ZERO_GRADIENT,
        BoundaryStrategy.INLET_OUTLET,
    )
    boundary_neighbours = np.asarray(
        mesh_data.get("boundary_neighbours", np.full(n_faces, -1, dtype=np.int32))
    )

    net_local = 0.0
    outflow_local = 0.0
    adjustable_slices: list[tuple[int, int, np.ndarray]] = []
    for boundary in boundaries:
        start = int(boundary["startFace"])
        nf = int(boundary["nFaces"])
        if nf == 0:
            continue
        patch_flux = flux_vf[start : start + nf]
        # Cyclic faces pair internally and never carry net domain flux.
        if np.any(boundary_neighbours[start : start + nf] >= 0):
            continue
        net_local += float(np.sum(patch_flux))
        strategy = BOUNDARIES.strategy(boundary.get("bc_type_velocity"), "U", "flux")
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
        return flux_vf
    # Not enough floating outflow to absorb the imbalance without reversing a
    # face; leave it for the pressure solve rather than manufacture backflow.
    if outflow_total <= abs(net):
        return flux_vf

    # Scale the floating outflow faces so their total drops by exactly `net`.
    scale = (outflow_total - net) / outflow_total
    for start, nf, outflow in adjustable_slices:
        patch = flux_vf[start : start + nf]
        patch[outflow] *= scale
        flux_vf[start : start + nf] = patch
    return flux_vf


@njit(cache=True)
def _pressure_interior_flux_scalar(
    owners,
    neighbours,
    weights,
    face_sf,
    face_cf_vector,
    DU,
    U_HbyA,
    grad_p,
    p,
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
        du_face = weight * DU[nei] + owner_weight * DU[own]

        cf0 = face_cf_vector[face, 0]
        cf1 = face_cf_vector[face, 1]
        cf2 = face_cf_vector[face, 2]
        mag_cf = np.sqrt(cf0 * cf0 + cf1 * cf1 + cf2 * cf2) + 1e-12
        edge0 = cf0 / mag_cf
        edge1 = cf1 / mag_cf
        edge2 = cf2 / mag_cf
        sf0 = face_sf[face, 0]
        sf1 = face_sf[face, 1]
        sf2 = face_sf[face, 2]
        orthogonal_area = sf0 * edge0 + sf1 * edge1 + sf2 * edge2

        grad0 = weight * grad_p[nei, 0] + owner_weight * grad_p[own, 0]
        grad1 = weight * grad_p[nei, 1] + owner_weight * grad_p[own, 1]
        grad2 = weight * grad_p[nei, 2] + owner_weight * grad_p[own, 2]
        nonorthogonal_flux = du_face * (
            (sf0 - orthogonal_area * edge0) * grad0
            + (sf1 - orthogonal_area * edge1) * grad1
            + (sf2 - orthogonal_area * edge2) * grad2
        )
        flux_hbya = (
            (weight * U_HbyA[nei, 0] + owner_weight * U_HbyA[own, 0]) * sf0
            + (weight * U_HbyA[nei, 1] + owner_weight * U_HbyA[own, 1]) * sf1
            + (weight * U_HbyA[nei, 2] + owner_weight * U_HbyA[own, 2]) * sf2
        )
        if use_ddt:
            flux_hbya += du_face * ddt_flux_correction[face]
        conductance = face_conductance[face]
        flux_vf[face] = flux_hbya + conductance * (p[own] - p[nei]) + nonorthogonal_flux


@njit(cache=True)
def _pressure_interior_flux_vector(
    owners,
    neighbours,
    weights,
    face_sf,
    face_cf_vector,
    DU,
    U_HbyA,
    grad_p,
    p,
    face_conductance,
    ddt_flux_correction,
    flux_vf,
):
    """Fused legacy component-diagonal Rhie--Chow interior-face kernel."""
    use_ddt = len(ddt_flux_correction) != 0
    for face in range(len(neighbours)):
        own = owners[face]
        nei = neighbours[face]
        weight = weights[face]
        owner_weight = 1.0 - weight
        du0 = weight * DU[nei, 0] + owner_weight * DU[own, 0]
        du1 = weight * DU[nei, 1] + owner_weight * DU[own, 1]
        du2 = weight * DU[nei, 2] + owner_weight * DU[own, 2]

        cf0 = face_cf_vector[face, 0]
        cf1 = face_cf_vector[face, 1]
        cf2 = face_cf_vector[face, 2]
        mag_cf = np.sqrt(cf0 * cf0 + cf1 * cf1 + cf2 * cf2) + 1e-12
        edge0 = cf0 / mag_cf
        edge1 = cf1 / mag_cf
        edge2 = cf2 / mag_cf
        sf0 = face_sf[face, 0]
        sf1 = face_sf[face, 1]
        sf2 = face_sf[face, 2]
        orthogonal_area = sf0 * edge0 + sf1 * edge1 + sf2 * edge2

        grad0 = weight * grad_p[nei, 0] + owner_weight * grad_p[own, 0]
        grad1 = weight * grad_p[nei, 1] + owner_weight * grad_p[own, 1]
        grad2 = weight * grad_p[nei, 2] + owner_weight * grad_p[own, 2]
        nonorthogonal_flux = (
            (sf0 - orthogonal_area * edge0) * du0 * grad0
            + (sf1 - orthogonal_area * edge1) * du1 * grad1
            + (sf2 - orthogonal_area * edge2) * du2 * grad2
        )
        flux_hbya = (
            (weight * U_HbyA[nei, 0] + owner_weight * U_HbyA[own, 0]) * sf0
            + (weight * U_HbyA[nei, 1] + owner_weight * U_HbyA[own, 1]) * sf1
            + (weight * U_HbyA[nei, 2] + owner_weight * U_HbyA[own, 2]) * sf2
        )
        if use_ddt:
            flux_hbya += ((du0 + du1 + du2) / 3.0) * ddt_flux_correction[face]
        conductance = face_conductance[face]
        flux_vf[face] = flux_hbya + conductance * (p[own] - p[nei]) + nonorthogonal_flux


@overload
def assemble_pressure_correction_equation_rhie_chow(
    *args: Any, return_workspace: Literal[False] = False, **kwargs: Any
) -> tuple[Any, Any, np.ndarray]: ...


@overload
def assemble_pressure_correction_equation_rhie_chow(
    *args: Any, return_workspace: Literal[True], **kwargs: Any
) -> tuple[Any, Any, np.ndarray, PressureCorrectionWorkspace]: ...


def assemble_pressure_correction_equation_rhie_chow(
    U_star,
    A_U,
    p,
    rho,
    mesh_data,
    geo_data,
    boundaries,
    alpha_u=1.0,
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
    1. Reconstruct velocity without pressure gradient at cell centers (HbyA).
    2. Interpolate HbyA to faces.
    3. Add compact pressure gradient drive at faces.

    This is more robust against checkerboarding than the standard correction method.

    Args:
        U_star: Predicted velocity field
        A_U: Momentum diagonal coefficients
        p: Current kinematic-pressure field ``p/ρ`` [m²/s²].
        rho: Positive constant reference density [kg/m³]. It is validated
            for API compatibility but cancels from this pressure equation.
        mesh_data: Mesh connectivity
        geo_data: Geometric data
        boundaries: Boundary conditions
        alpha_u: Velocity under-relaxation factor

    Returns:
        tuple: (A_p, b_p, f_vf) where f_vf is the Rhie-Chow corrected flux (phi_star).
    """
    n_elements = mesh_data["n_elements"]
    n_interior = mesh_data["n_interior_faces"]
    n_faces = mesh_data["n_faces"]
    owners = mesh_data["owners"]
    neighbours = mesh_data["neighbours"]
    _validate_reference_density(rho)

    # 1. Compute DU and grad_p. The momentum diagonal is fixed for all
    # pressure/non-orthogonal corrections in one PIMPLE outer iteration, so
    # retain its inverse and face conductance instead of rebuilding two
    # full-mesh arrays on every inner solve.
    volumes = geo_data["element_volumes"]
    if correction_workspace is None:
        # Restore physical A_U from relaxed A_U for Rhie-Chow D-coefficients
        A_U_physical = A_U * alpha_u
        DU = _compute_rhie_chow_coefficients(volumes, A_U_physical)
        face_conductance = _compute_pressure_face_conductance(mesh_data, geo_data, DU)
    else:
        DU = correction_workspace.DU
        face_conductance = correction_workspace.face_conductance
    if reuse_matrix and (correction_workspace is None or correction_workspace.matrix is None):
        raise ValueError("reuse_matrix requires an assembled pressure-correction workspace")

    # Use direct gradient computation for full pressure field p
    _grad_fn = gradients._resolve_gradient_fn(geo_data)
    grad_p = _grad_fn(p, mesh_data, geo_data)
    if grad_p.ndim == 3:
        grad_p = grad_p.squeeze(-1)
    grad_p = _update_fixed_flux_pressure_boundaries(
        p, U_star, DU, mesh_data, geo_data, boundaries, grad_p=grad_p
    )
    assert grad_p is not None

    # Pre-allocate flux arrays
    flux_cf = None if reuse_matrix else np.zeros(n_faces)
    flux_ff = None if reuse_matrix else np.zeros(n_faces)
    flux_vf = np.zeros(n_faces)

    # --- MODIFIED RHIE-CHOW INTERPOLATION ---
    # Standard Rhie-Chow: U_f = Avg(U) + Avg(D)*(Avg(GradP) - CompactGradP)
    # Modified Rhie-Chow: U_f = Avg(U + D*GradP) - Avg(D)*CompactGradP
    # This is much more robust on persistent checkerboarding.

    # 1. Reconstruct "H/A" velocity at cell centers (Velocity without pressure gradient)
    # U_center = H/A - grad_p * DU
    # So H/A = U_center + grad_p * DU
    # We use U_star as U_center (it includes -grad_p * DU approx)
    scalar_diagonal = np.asarray(DU).ndim == 1
    DU_vector = DU[:, np.newaxis] if scalar_diagonal else DU
    U_HbyA = U_star[:n_elements] + DU_vector * grad_p[:n_elements]

    # Fuse the interior-face interpolation and non-orthogonal correction in a
    # compiled loop. Besides being faster than eight chains of NumPy advanced
    # indexing per PIMPLE step, this avoids retaining DU, HbyA, edge,
    # non-orthogonal, and gradient temporaries at the same time.
    grad_p_interior = grad_p[:n_elements]
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
        geo_data["face_weights"][:n_interior],
        geo_data["face_sf"][:n_interior],
        geo_data["face_cf_vector"][:n_interior],
        DU,
        U_HbyA,
        grad_p_interior,
        p,
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
        # stage of the correction (p' ghost extension, boundary-flux
        # correction, velocity/pressure ghost updates) must reuse this mask.
        # Re-deriving the switch from the evolving flux field lets grazing
        # faces (u·n ≈ 0, e.g. the lateral sides of a coupling box) change
        # class between assembly and correction, injecting boundary flux the
        # pressure matrix never saw — which surfaces as a divergence
        # checkerboard anchored at the patch corners.
        for boundary in boundaries:
            strategy = BOUNDARIES.strategy(boundary.get("bc_type_p"), "p", "pressure")
            if strategy is BoundaryStrategy.FREESTREAM:
                start = int(boundary["startFace"])
                nf = int(boundary["nFaces"])
                fixed_outflow = boundary.get("_fixed_freestream_outflow")
                if fixed_outflow is not None:
                    boundary["_freestream_outflow"] = fixed_outflow
                else:
                    ghost = U_star[
                        n_elements + (start - n_interior) : n_elements + (start - n_interior) + nf
                    ]
                    sf_patch = geo_data["face_sf"][start : start + nf]
                    boundary["_freestream_outflow"] = np.einsum("ij,ij->i", ghost, sf_patch) >= 0.0
        du_components = (DU, DU, DU) if scalar_diagonal else (DU[:, 0], DU[:, 1], DU[:, 2])
        cf_b, ff_b, vf_b = _process_boundary_faces_jit(
            n_boundary_faces,
            n_interior,
            n_elements,
            owners,
            geo_data["face_sf"],
            geo_data["face_cf_vector"],
            U_star,
            *du_components,
            grad_p,
            p,
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
            start = int(boundary["startFace"])
            nf = int(boundary["nFaces"])
            field = np.asarray(external_flux, dtype=float).reshape(-1)
            if field.shape != (nf,) or not np.all(np.isfinite(field)):
                raise ValueError(
                    f"External face flux for patch {boundary.get('name')!r} must have "
                    f"shape ({nf},) and finite values"
                )
            flux_vf[start : start + nf] = field

        boundary_neighbours = np.asarray(
            mesh_data.get("boundary_neighbours", np.full(n_faces, -1, dtype=np.int32))
        )
        cyclic_faces = np.flatnonzero(boundary_neighbours >= 0)
        if cyclic_faces.size:
            own_b = owners[cyclic_faces]
            nei_b = boundary_neighbours[cyclic_faces]
            weight_b_scalar = geo_data["face_weights"][cyclic_faces]
            weight_b = weight_b_scalar[:, np.newaxis]
            sf_b = geo_data["face_sf"][cyclic_faces]
            cf_b = geo_data["face_cf_vector"][cyclic_faces]
            mag_cf_b = np.linalg.norm(cf_b, axis=1)
            edge_b = cf_b / mag_cf_b[:, np.newaxis]
            if scalar_diagonal:
                du_b = weight_b_scalar * DU[nei_b] + (1.0 - weight_b_scalar) * DU[own_b]
            else:
                du_b = weight_b * DU[nei_b] + (1.0 - weight_b) * DU[own_b]
            hbya_b = weight_b * U_HbyA[nei_b] + (1.0 - weight_b) * U_HbyA[own_b]
            grad_b = weight_b * grad_p[nei_b] + (1.0 - weight_b) * grad_p[own_b]
            orthogonal_area = np.sum(sf_b * edge_b, axis=1)
            nonorthogonal = sf_b - orthogonal_area[:, np.newaxis] * edge_b
            conductance_b = face_conductance[cyclic_faces]
            if not reuse_matrix:
                assert flux_cf is not None and flux_ff is not None
                flux_cf[cyclic_faces] = conductance_b
                flux_ff[cyclic_faces] = -conductance_b
            flux_hbya = np.sum(hbya_b * sf_b, axis=1)
            compact = conductance_b * (p[own_b] - p[nei_b])
            if scalar_diagonal:
                nonorthogonal_flux = du_b * np.sum(nonorthogonal * grad_b, axis=1)
            else:
                nonorthogonal_flux = np.sum(nonorthogonal * du_b * grad_b, axis=1)
            flux_vf[cyclic_faces] = flux_hbya + compact + nonorthogonal_flux

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
        A_p = correction_workspace.matrix
    else:
        assert flux_cf is not None and flux_ff is not None
        flux_data.update({"flux_cf": flux_cf, "flux_ff": flux_ff})
        A_p = matrix_assembly.assemble_matrix_from_fluxes_vectorized(
            flux_data, mesh_data, workspace=matrix_workspace, backend=operator_backend
        )
    b_p = matrix_assembly.assemble_rhs_from_fluxes_vectorized(
        flux_data, mesh_data, backend=operator_backend
    )

    # 5. Fix Pressure Reference only for an all-Neumann pressure problem.
    if _pressure_requires_constraint(boundaries, U_star, mesh_data, geo_data):
        if pressure_constraint == "reference":
            A_p, b_p = cavity_utils.fix_pressure_reference(A_p, b_p)
        elif pressure_constraint == "nullspace":
            # A finite-volume all-Neumann RHS should already be compatible;
            # remove only accumulated roundoff before the backend projection.
            parallel = mesh_data.get("_parallel_context")
            if parallel is not None and parallel.is_partitioned:
                n_owned = parallel.n_owned
                global_sum = parallel.global_sum(float(np.sum(b_p[:n_owned])))
                b_p = b_p - global_sum / parallel.partition.global_n_cells
            else:
                b_p = b_p - np.mean(b_p)
        else:
            raise ValueError(
                "All-Neumann pressure requires pressure_constraint='reference' or 'nullspace'"
            )

    if return_workspace:
        if correction_workspace is None:
            correction_workspace = PressureCorrectionWorkspace(DU, face_conductance, A_p)
        return A_p, b_p, flux_vf, correction_workspace
    return A_p, b_p, flux_vf


def _extend_p_prime_bcs(p_prime, mesh_data, boundaries, face_flux=None):
    """Extend the pressure-correction array with ghost-cell values.

    For ``fixedValue`` pressure boundaries, the ghost value is set to
    zero (``p' = 0`` at a fixed-pressure face).  For all other types,
    the ghost cell inherits the owner cell value (zero-gradient).

    Args:
        p_prime:    Pressure correction for interior cells ``(n_elements,)``.
        mesh_data:  Mesh dictionary.
        boundaries: List of boundary patch dictionaries.

    Returns:
        Extended ``p_prime`` array ``(n_elements + n_boundary_faces,)``.
    """
    n_elements = mesh_data["n_elements"]
    n_interior = mesh_data["n_interior_faces"]
    n_faces = mesh_data["n_faces"]
    owners = mesh_data["owners"]
    p_prime_ext = np.zeros(n_elements + (n_faces - n_interior))
    p_prime_ext[:n_elements] = p_prime
    for boundary in boundaries:
        start = boundary["startFace"]
        nf = boundary["nFaces"]
        idx = n_elements + (start - n_interior)
        own = owners[start : start + nf]
        bc_type_p = boundary.get("bc_type_p")
        strategy = BOUNDARIES.strategy(bc_type_p, "p", "ghost")
        if strategy is BoundaryStrategy.FIXED_VALUE:
            p_prime_ext[idx : idx + nf] = 0.0
        elif strategy is BoundaryStrategy.CYCLIC:
            paired = mesh_data["boundary_neighbours"][start : start + nf]
            p_prime_ext[idx : idx + nf] = p_prime[paired]
        elif strategy is BoundaryStrategy.FREESTREAM:
            outflow = boundary.get("_freestream_outflow")
            if outflow is None:
                if face_flux is None:
                    raise ValueError(
                        "Freestream pressure correction requires the predicted face flux"
                    )
                outflow = np.asarray(face_flux)[start : start + nf] >= 0.0
            p_prime_ext[idx : idx + nf] = np.where(outflow, 0.0, p_prime[own])
        elif strategy in (
            BoundaryStrategy.ZERO_GRADIENT,
            BoundaryStrategy.FIXED_FLUX_PRESSURE,
            BoundaryStrategy.FIXED_GRADIENT,
            BoundaryStrategy.EMPTY,
        ):
            p_prime_ext[idx : idx + nf] = p_prime[own]
        else:
            raise RuntimeError(f"Unhandled pressure ghost strategy {strategy!r}")
    return p_prime_ext


def _correct_interior_fluxes(phi, p_prime, mesh_data, face_conductance):
    """Correct interior face fluxes with the Rhie-Chow pressure correction.

    Applies the volumetric-flux correction ``Δφ = g⋅(p'_P − p'_N)`` where *g* is
    the geometric diffusion coefficient based on the interpolated DU and
    face-normal projection.

    Args:
        phi:      Face flux array ``(n_faces,)`` (mutated in place).
        p_prime:  Pressure correction ``(n_elements,)``.
        mesh_data: Mesh dictionary.
        face_conductance: Shared pressure-face conductance array.
    """
    n_interior = mesh_data["n_interior_faces"]
    owners = mesh_data["owners"]
    neighbours = mesh_data["neighbours"]
    geo_diff = face_conductance[:n_interior]
    phi[:n_interior] += geo_diff * (p_prime[owners[:n_interior]] - p_prime[neighbours[:n_interior]])


def _correct_boundary_fluxes(phi, p_prime, boundaries, owners, face_conductance):
    """Correct boundary-face fluxes for ``fixedValue`` pressure boundaries.

    Only patches whose pressure BC type is ``fixedValue`` receive a
    correction, computed from the geometric diffusion and wall distance.

    Args:
        phi:        Face flux array ``(n_faces,)`` (mutated in place).
        p_prime:    Pressure correction ``(n_elements,)``.
        boundaries: List of boundary patch dictionaries.
        owners:     Owner index array.
        face_conductance: Shared pressure-face conductance array.
    """
    for boundary in boundaries:
        start = boundary["startFace"]
        nf = boundary["nFaces"]
        idx = np.arange(start, start + nf)
        own = owners[idx]
        geo_diff_b = face_conductance[idx]
        bc_type = boundary.get("bc_type_p")
        strategy = BOUNDARIES.strategy(bc_type, "p", "flux")
        if strategy is BoundaryStrategy.FIXED_VALUE:
            phi[idx] += geo_diff_b * p_prime[own]
        elif strategy is BoundaryStrategy.CYCLIC:
            paired = boundary.get("_paired_cells")
            if paired is None:
                # The mesh-level array is not part of this helper's historical
                # signature; cyclic setup stores the same view on each patch.
                raise ValueError(f"Cyclic patch {boundary.get('name')!r} lacks paired cells")
            phi[idx] += geo_diff_b * (p_prime[own] - p_prime[paired])
        elif strategy is BoundaryStrategy.FREESTREAM:
            outflow = boundary.get("_freestream_outflow")
            if outflow is None:
                outflow = phi[idx] >= 0.0
            phi[idx] += np.where(outflow, geo_diff_b * p_prime[own], 0.0)
        elif strategy not in (
            BoundaryStrategy.ZERO_GRADIENT,
            BoundaryStrategy.FIXED_FLUX_PRESSURE,
            BoundaryStrategy.FIXED_GRADIENT,
            BoundaryStrategy.EMPTY,
        ):
            raise RuntimeError(f"Unhandled pressure flux strategy {strategy!r}")


def _apply_inlet_outlet_bc(U, phi, boundary, owners, n_elements, n_interior):
    """inletOutlet velocity BC: zeroGradient on outflow, fixed value on inflow.

    Per face: outgoing flux (φ ≥ 0) → extrapolate from the owner cell
    (zeroGradient); incoming flux (φ < 0) → impose ``value_velocity_field`` when
    present, otherwise the uniform ``value_velocity`` (the inletValue, default 0).
    The per-face path is required by the FVM--VPM characteristic VPM BC:
    pressure-correction refreshes must not replace its non-uniform VPM-BC trace
    with the uniform freestream.
    """
    start = boundary["startFace"]
    nf = boundary["nFaces"]
    idx = n_elements + (start - n_interior)
    own = owners[start : start + nf]
    if boundary.get("value_velocity_field") is not None:
        inlet_val = np.asarray(boundary["value_velocity_field"], dtype=float)
        if inlet_val.shape != (nf, 3):
            raise ValueError(
                f"Per-face inlet velocity for patch {boundary.get('name')!r} "
                f"must have shape ({nf}, 3), got {inlet_val.shape}"
            )
    else:
        inlet_val = np.asarray(boundary.get("value_velocity", [0.0, 0.0, 0.0]), dtype=float)
        if inlet_val.shape != (3,):
            raise ValueError(
                f"Uniform inlet velocity for patch {boundary.get('name')!r} "
                f"must have shape (3,), got {inlet_val.shape}"
            )
    outflow = boundary.get("_freestream_outflow")
    if outflow is None:
        outflow = phi[start : start + nf] >= 0.0
    U[idx : idx + nf] = np.where(outflow[:, np.newaxis], U[own], inlet_val)


def _apply_zero_gradient_bc(U, phi, boundary, owners, n_elements, n_interior, boundaries):
    """Apply a zero-gradient velocity BC (extrapolate from the owner cell).

    Sets the ghost-cell velocity equal to the owner-cell value.

    Args:
        U:           Velocity array (mutated in place).
        phi:         Face flux array.
        boundary:    Boundary patch dictionary.
        owners:      Owner index array.
        n_elements:  Number of interior elements.
        n_interior:  Number of interior faces.
        boundaries:  List of all boundary patches.
    """
    start = boundary["startFace"]
    nf = boundary["nFaces"]
    idx = n_elements + (start - n_interior)
    own = owners[start : start + nf]
    U[idx : idx + nf] = U[own]


def _apply_cyclic_bc(U, boundary, mesh_data, n_elements, n_interior):
    """Copy paired owner values into the cyclic patch ghost layer."""
    start = boundary["startFace"]
    nf = boundary["nFaces"]
    idx = n_elements + (start - n_interior)
    paired = mesh_data["boundary_neighbours"][start : start + nf]
    if np.any(paired < 0):
        raise ValueError(f"Cyclic patch {boundary['name']!r} is not paired")
    U[idx : idx + nf] = U[paired]


def _apply_fixed_value_bc(U, boundary, n_elements, n_interior, strategy):
    """Apply fixedValue or noSlip velocity BC.

    Honours a per-face ``value_velocity_field`` (n_faces_patch, 3) when present (e.g. a
    non-uniform coupler VPM BC), otherwise the uniform ``value_velocity``.
    """
    start = boundary["startFace"]
    nf = boundary["nFaces"]
    idx = n_elements + (start - n_interior)
    if strategy is BoundaryStrategy.NO_SLIP:
        U[idx : idx + nf] = [0.0, 0.0, 0.0]
    elif boundary.get("value_velocity_field") is not None:
        U[idx : idx + nf] = boundary["value_velocity_field"]
    elif "value_velocity" in boundary:
        U[idx : idx + nf] = np.array(boundary["value_velocity"])


def _apply_slip_bc(U, boundary, owners, geo_data, n_elements, n_interior):
    """Apply a slip / symmetry / empty velocity BC.

    Removes the component normal to the boundary face from the
    ghost-cell velocity, leaving only the tangential part.

    Args:
        U:          Velocity array (mutated in place).
        boundary:   Boundary patch dictionary.
        owners:     Owner index array.
        geo_data:   Geometry dictionary (needs ``face_sf``).
        n_elements: Number of interior elements.
        n_interior: Number of interior faces.
    """
    start = boundary["startFace"]
    nf = boundary["nFaces"]
    idx = n_elements + (start - n_interior)
    own = owners[start : start + nf]
    face_sf = geo_data["face_sf"][start : start + nf]
    # This is deliberately the same array expression as the predictor's
    # empty-boundary update.  In particular, a degenerate face retains the
    # owner value as the old scalar helper did.
    owner_velocity = U[own]
    magnitudes = np.linalg.norm(face_sf, axis=1)
    valid = magnitudes > 1e-10
    projected = owner_velocity.copy()
    if np.any(valid):
        normals = face_sf[valid] / magnitudes[valid, np.newaxis]
        normal_velocity = np.sum(owner_velocity[valid] * normals, axis=1)
        projected[valid] -= normal_velocity[:, np.newaxis] * normals
    U[idx : idx + nf] = projected


def _update_velocity_bcs(
    U, phi, boundaries, owners, geo_data, n_elements, n_interior, mesh_data=None
):
    """Update all velocity boundary conditions after a pressure-correction step.

    Dispatches to individual BC handlers based on each patch's
    ``bc_type_velocity``:

    - ``zeroGradient`` → :func:`_apply_zero_gradient_bc`
    - ``inletOutlet``  → :func:`_apply_inlet_outlet_bc`
    - ``fixedValue`` / ``noSlip`` → :func:`_apply_fixed_value_bc`
    - ``empty`` / ``slip`` / ``symmetry`` → :func:`_apply_slip_bc`

    Args:
        U:           Velocity array (mutated in place).
        phi:         Face flux array.
        boundaries:  List of boundary patch dictionaries.
        owners:      Owner index array.
        geo_data:    Geometry dictionary.
        n_elements:  Number of interior elements.
        n_interior:  Number of interior faces.
    """
    for boundary in boundaries:
        bc_type_u = boundary.get("bc_type_velocity") or boundary.get("bc_type")
        strategy = BOUNDARIES.strategy(bc_type_u, "U", "ghost")
        if strategy is BoundaryStrategy.ZERO_GRADIENT:
            _apply_zero_gradient_bc(U, phi, boundary, owners, n_elements, n_interior, boundaries)
        elif strategy in (BoundaryStrategy.INLET_OUTLET, BoundaryStrategy.FREESTREAM):
            _apply_inlet_outlet_bc(U, phi, boundary, owners, n_elements, n_interior)
        elif strategy in (BoundaryStrategy.FIXED_VALUE, BoundaryStrategy.NO_SLIP):
            _apply_fixed_value_bc(U, boundary, n_elements, n_interior, strategy)
        elif strategy is BoundaryStrategy.NORMAL_VALUE_TANGENTIAL_GRADIENT:
            if mesh_data is None:
                raise ValueError("Mixed velocity boundary update requires mesh_data")
            update_normal_velocity_tangential_gradient_boundary(U, boundary, mesh_data, geo_data)
        elif strategy in (
            BoundaryStrategy.EMPTY,
            BoundaryStrategy.SLIP,
            BoundaryStrategy.SYMMETRY,
        ):
            _apply_slip_bc(U, boundary, owners, geo_data, n_elements, n_interior)
        elif strategy is BoundaryStrategy.CYCLIC:
            if mesh_data is None:
                raise ValueError("Cyclic boundary update requires mesh_data")
            _apply_cyclic_bc(U, boundary, mesh_data, n_elements, n_interior)
        else:
            raise RuntimeError(f"Unhandled velocity ghost strategy {strategy!r}")


def correct_velocity_and_flux(
    U,
    phi,
    p_prime,
    A_U,
    mesh_data,
    geo_data,
    boundaries,
    rho=1.0,
    alpha_u=1.0,
    alpha_p=1.0,
    workspace: PressureCorrectionWorkspace | None = None,
):
    """
    Apply pressure correction to velocity and persistent flux.

    U = U* - alpha_p * DU * grad(p')
    phi = phi* - DU_f * (grad(p') . S)

    Uses UN-RELAXED diagonal for DU consistency: DU = V / (A_U * alpha_u).
    This matches the Rhie-Chow assembly which also uses the un-relaxed A_U.

    ``alpha_p`` scales the **cell-velocity** correction only, never the face
    flux. The pressure equation corrects ``phi`` with the full flux correction
    (so mass conservation never depends on a relaxation factor), then rebuilds
    the cell velocity from the relaxed pressure. Passing the full
    correction to ``U`` while the caller stores only ``alpha_p * p_prime``
    leaves velocity and pressure describing different states, and the outer
    loop then converges to a fixed point that depends on ``alpha_p`` instead of
    the pressure-relaxation-independent solution.
    """
    n_elements = mesh_data["n_elements"]
    n_interior = mesh_data["n_interior_faces"]
    owners = mesh_data["owners"]
    _validate_reference_density(rho)

    if workspace is None:
        volumes = geo_data["element_volumes"]
        # Restore un-relaxed diagonal for DU consistency.
        A_U_physical = A_U * alpha_u
        DU = _compute_rhie_chow_coefficients(volumes, A_U_physical)
        face_conductance = _compute_pressure_face_conductance(mesh_data, geo_data, DU)
    else:
        DU = workspace.DU
        face_conductance = workspace.face_conductance

    # 1. Correct Cell Velocity
    p_prime_ext = _extend_p_prime_bcs(p_prime, mesh_data, boundaries, face_flux=phi)
    _grad_fn = gradients._resolve_gradient_fn(geo_data)
    grad_p_prime = _grad_fn(p_prime_ext, mesh_data, geo_data)
    if grad_p_prime.ndim == 3:
        grad_p_prime = grad_p_prime.squeeze(-1)
    DU_vector = DU[:, np.newaxis] if np.asarray(DU).ndim == 1 else DU
    U[:n_elements] -= alpha_p * DU_vector * grad_p_prime[:n_elements]

    # 2. Correct Face Fluxes
    _correct_interior_fluxes(phi, p_prime, mesh_data, face_conductance)
    _correct_boundary_fluxes(phi, p_prime, boundaries, owners, face_conductance)

    # 3. Update Velocity BCs
    _update_velocity_bcs(
        U, phi, boundaries, owners, geo_data, n_elements, n_interior, mesh_data=mesh_data
    )

    return U, phi


def _remove_normal_component(U_owner, face_vector):
    """Remove the normal component of velocity (slip / symmetry / empty BC).

    Projects out the velocity component along the face normal, leaving
    only the tangential component: ``U_t = U − (U·n̂) n̂``.

    Args:
        U_owner:     Velocity vector at the owner cell ``(3,)``.
        face_vector: Face area vector ``(3,)`` (direction defines normal).

    Returns:
        Tangential velocity vector ``(3,)``.
    """
    norm_Sf = np.linalg.norm(face_vector)
    if norm_Sf > 1e-10:
        n = face_vector / norm_Sf
        U_normal_mag = np.dot(U_owner, n)
        U_normal = U_normal_mag * n
        return U_owner - U_normal
    return U_owner


def _apply_scalar_bc(
    phi,
    indices,
    owners_b,
    strategy,
    boundary,
    field_name,
    paired_owners=None,
    face_flux=None,
):
    """Apply a boundary condition to a scalar field ghost-cell block.

    Zero-gradient BCs (including inlet, outlet, symmetry, empty, slip,
    noSlip) copy the owner value.  ``fixedValue`` sets the prescribed
    boundary value.

    Args:
        phi:        Scalar field array (mutated in place).
        indices:    Ghost-cell indices for this patch.
        owners_b:   Owner cell indices for the boundary faces.
        strategy:   Validated boundary behavior.
        boundary:   Boundary patch dictionary (may contain ``value_p`` etc.).
        field_name: Field name for value lookup (e.g. ``"p"``, ``"phi"``).
    """
    if strategy in (BoundaryStrategy.ZERO_GRADIENT, BoundaryStrategy.EMPTY):
        phi[indices] = phi[owners_b]
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
            phi[indices] = phi[owners_b]
        else:
            phi[indices] = phi[owners_b] + np.asarray(delta, dtype=float)
    elif strategy is BoundaryStrategy.FIXED_VALUE:
        val = boundary.get(f"value_{field_name}")
        if val is None:
            val = boundary.get("value")
        if val is None:
            raise ValueError(
                f"Fixed-value {field_name} boundary {boundary.get('name')!r} has no value"
            )
        phi[indices] = val
    elif strategy is BoundaryStrategy.CYCLIC:
        if paired_owners is None or np.any(paired_owners < 0):
            raise ValueError(f"Cyclic patch {boundary.get('name')!r} is not paired")
        phi[indices] = phi[paired_owners]
    elif strategy is BoundaryStrategy.FREESTREAM:
        outflow = boundary.get("_freestream_outflow")
        if outflow is None:
            if face_flux is None:
                raise ValueError("Freestream scalar update requires face fluxes")
            outflow = np.asarray(face_flux) >= 0.0
        val = boundary.get(f"value_{field_name}")
        if val is None:
            val = boundary.get("value")
        if val is None:
            raise ValueError(
                f"Freestream {field_name} boundary {boundary.get('name')!r} has no value"
            )
        if field_name == "p" and boundary.get("_directional_fixed_flux_pressure", False):
            delta = boundary.get("fixed_flux_pressure_delta")
            owner_value = phi[owners_b] if delta is None else phi[owners_b] + np.asarray(delta)
            phi[indices] = np.where(outflow, val, owner_value)
        else:
            phi[indices] = np.where(outflow, val, phi[owners_b])
    else:
        raise ValueError(
            f"Unsupported scalar boundary strategy {strategy!r} "
            f"for {field_name} on patch {boundary.get('name')!r}"
        )


def update_scalar_boundaries(phi, mesh_data, boundaries, field_name="p", face_flux=None):
    """
    Update boundary values (ghost cells) for a scalar field.

    Args:
        phi: Scalar field (n_elements + n_boundary)
        mesh_data: Mesh connectivity
        boundaries: Boundary conditions
        field_name: Name of field ('p', etc.) to check BC type
    """
    n_elements = mesh_data["n_elements"]
    n_interior = mesh_data["n_interior_faces"]
    owners = mesh_data["owners"]

    for boundary in boundaries:
        start = boundary["startFace"]
        n_bfaces = boundary["nFaces"]

        # Boundary element indices
        b_elem_start = n_elements + (start - n_interior)
        b_elem_indices = np.arange(b_elem_start, b_elem_start + n_bfaces)
        owners_b = owners[start : start + n_bfaces]
        paired = mesh_data.get("boundary_neighbours")
        paired_owners = None if paired is None else paired[start : start + n_bfaces]
        patch_flux = None if face_flux is None else np.asarray(face_flux)[start : start + n_bfaces]

        # Get BC type
        bc_type = boundary.get(f"bc_type_{field_name}") or boundary.get("bc_type")
        registry_field = "p" if field_name == "p" else "scalar"
        strategy = BOUNDARIES.strategy(bc_type, registry_field, "ghost")
        _apply_scalar_bc(
            phi,
            b_elem_indices,
            owners_b,
            strategy,
            boundary,
            field_name,
            paired_owners=paired_owners,
            face_flux=patch_flux,
        )


class SIMPLESolver:
    """SIMPLE algorithm for incompressible Navier–Stokes.

    Semi-Implicit Method for Pressure-Linked Equations: solves the
    steady incompressible Navier–Stokes equations through a predictor–
    corrector loop that alternates between a momentum solve and a
    pressure-correction Poisson solve.

    The algorithm iterates until ``max_iter`` is reached or the residuals
    fall below ``tolerance``.  Under-relaxation is applied through
    ``alpha_u`` (velocity) and ``alpha_p`` (pressure).

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
    >>> U, p, phi, converged = solver.solve(  # doctest: +SKIP
    ...     U_initial, p_initial, rho=1.225, nu=1.5e-5
    ... )
    """

    def __init__(self, mesh_data, geo_data, boundaries, params=None):
        """Initialise the SIMPLE solver.

        Args:
            mesh_data: Mesh connectivity dictionary.
            geo_data: Geometric quantities dictionary.
            boundaries: List of boundary condition dictionaries.
            params: Optional dict of solver parameters overriding defaults.
                Supported keys: ``alpha_u``, ``alpha_p``, ``max_iter``,
                ``tolerance``, ``convection_scheme``, ``linear_solver``.
        """
        self.mesh_data = mesh_data
        self.geo_data = geo_data
        self.boundaries = boundaries

        # Default parameters
        self.params: dict[str, Any] = {
            "alpha_u": 0.7,
            "alpha_p": 0.3,
            "max_iter": 100,
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
        clear_linear_solver_caches(("pressure", namespace))

    def step(
        self,
        U,
        p,
        phi,
        U_old=None,
        time_step_size=None,
        rho=1.0,
        nu=0.01,
        U_old_old=None,
        source_explicit=None,
        source_implicit=None,
        phi_old=None,
        phi_old_old=None,
    ):
        """Perform one SIMPLE pressure–velocity correction.

        ``U_old_old`` and the flux histories are accepted for interface parity
        with the transient driver but are unused by steady SIMPLE.
        ``source_explicit``/``source_implicit``
        are optional volumetric momentum sources (e.g. the coupling blending source
        S = λ(Utarget − U)) forwarded to the momentum predictor.

        Args:
            U: Cell and boundary-ghost velocity [m/s], shape
                ``(n_cells_with_ghosts, 3)``.
            p: Kinematic pressure ``p/ρ`` [m²/s²], shape
                ``(n_cells_with_ghosts,)``.
            phi: Volumetric face flux ``U·Sf`` [m³/s], shape
                ``(n_faces,)``.
            U_old: Previous velocity [m/s], unused by steady SIMPLE.
            time_step_size: Time-step size [s], normally ``None`` for steady SIMPLE.
            rho: Positive constant reference density [kg/m³]. It cancels
                from the kinematic-pressure flow equations.
            nu: Positive kinematic viscosity [m²/s].
            U_old_old: Older velocity [m/s], unused by steady SIMPLE.
            source_explicit: Explicit acceleration source [m/s²].
            source_implicit: Non-negative implicit source coefficient [1/s].

        Returns:
            tuple: Updated ``(U, p, phi, residuals)``. The three field arrays
            retain the shapes and units described above.
        """
        # 1. Solve momentum predictor
        U_star, A_U, momentum_diagnostics = momentum.solve_momentum_predictor(
            U,
            p,
            phi,
            rho,
            nu,
            self.mesh_data,
            self.geo_data,
            self.boundaries,
            convection_scheme=self.params["convection_scheme"],
            solver=self.params.get("momentum_solver") or self.params["linear_solver"],
            under_relaxation=self.params["alpha_u"],
            time_step_size=time_step_size,  # Use dt if provided (e.g. for PIMPLE)
            source_explicit=source_explicit,
            source_implicit=source_implicit,
            linear_backend=self.params.get("_linear_backend", "scipy"),
            parallel_context=self.params.get("_parallel_context"),
            failure_policy=self.params.get("linear_failure_policy", "raise"),
            log_sink=self.params.get("_logger"),
            momentum_tol=self.params.get("momentum_tol", 1e-4),
            maxiter=self.params.get("momentum_maxiter", 1000),
            reuse_ilu=self.params.get("reuse_ilu", False),
            ilu_drop_tol=self.params.get("ilu_drop_tol", 1e-4),
            ilu_fill_factor=self.params.get("ilu_fill_factor", 10),
            ilu_reuse_tol=self.params.get("ilu_reuse_tol"),
            matrix_workspace=self._momentum_matrix_workspace,
            operator_backend=self.params.get("_operator_backend", "numpy"),
            return_diagnostics=True,
        )

        # 2. Solve pressure correction
        pressure_constraint = _resolve_pressure_constraint(self.params)
        has_pressure_nullspace = _pressure_requires_constraint(
            self.boundaries, U_star, self.mesh_data, self.geo_data
        )
        A_p, b_p, phi_star = assemble_pressure_correction_equation_rhie_chow(
            U_star,
            A_U,
            p,
            rho,
            self.mesh_data,
            self.geo_data,
            self.boundaries,
            alpha_u=self.params["alpha_u"],
            pressure_constraint=pressure_constraint,
            matrix_workspace=self._pressure_matrix_workspace,
            operator_backend=self.params.get("_operator_backend", "numpy"),
        )

        p_prime, pressure_result = solve_linear_system(
            A_p,
            b_p,
            method=self.params.get("pressure_solver") or self.params["linear_solver"],
            equation_type="pressure",
            tol=self.params.get("pressure_tol", 1e-8),
            maxiter=self.params.get("pressure_maxiter", 500),
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
        # Calculate residual before in-place modification of U_star
        velocity_increment = np.linalg.norm(
            U_star[: self.mesh_data["n_elements"]] - U[: self.mesh_data["n_elements"]]
        ) / (np.linalg.norm(U[: self.mesh_data["n_elements"]]) + 1e-10)

        U, phi = correct_velocity_and_flux(
            U_star,
            phi_star,
            p_prime,
            A_U,
            self.mesh_data,
            self.geo_data,
            self.boundaries,
            rho=rho,
            alpha_u=self.params["alpha_u"],
            alpha_p=self.params["alpha_p"],
        )

        # 4. Update pressure
        p[: self.mesh_data["n_elements"]] += self.params["alpha_p"] * p_prime

        # Update pressure boundaries
        update_scalar_boundaries(p, self.mesh_data, self.boundaries, field_name="p", face_flux=phi)

        # 5. Residuals
        self.last_res_p = normalized_residual(A_p, p_prime, b_p)
        self.last_res_u = max(
            (values["final_residual"] for values in momentum_diagnostics.values()),
            default=0.0,
        )
        continuity = field_diagnostics.compute_continuity_error(phi, self.mesh_data, self.geo_data)
        volumes = self.geo_data["element_volumes"]
        continuity_max = float(np.max(np.abs(continuity) / (volumes + 1e-30)))
        self.last_linear_results = tuple(
            values["linear_result"] for values in momentum_diagnostics.values()
        ) + (pressure_result,)
        self.last_outer_diagnostics = (
            OuterCorrectorDiagnostics(
                index=0,
                momentum_residual=self.last_res_u,
                pressure_residual=self.last_res_p,
                continuity_max=continuity_max,
            ),
        )

        residuals = {"p": self.last_res_p, "U": self.last_res_u, "U_increment": velocity_increment}
        residuals.update(
            {
                f"U_{component}": values["final_residual"]
                for component, values in momentum_diagnostics.items()
            }
        )
        return U, p, phi, residuals

    def solve(self, U_init, p_init, rho=1.0, nu=0.01):
        """Solve a steady incompressible flow using the SIMPLE algorithm.

        Iterates over ``max_iter`` SIMPLE steps, computing the momentum
        predictor, pressure correction, and velocity/pressure update at
        each iteration.  Convergence is declared when both the pressure
        and velocity residuals fall below ``tolerance``.

        Args:
            U_init: Initial velocity [m/s], shape ``(n_total, 3)``.
            p_init: Initial kinematic pressure ``p/ρ`` [m²/s²], shape
                ``(n_total,)``.
            rho: Positive constant reference density [kg/m³]. It cancels
                from the kinematic-pressure flow equations. Defaults to 1.0.
            nu: Kinematic viscosity [m²/s]. Defaults to 0.01.

        Returns:
            Tuple ``(U, p, phi, converged)`` where *converged* is a bool.
        """
        U = U_init.copy()
        p = p_init.copy()
        logger: Any = self.params.get("_logger")

        if logger is not None:
            logger.section(
                "SIMPLE SOLVER",
                [
                    ("Maximum iterations", str(self.params["max_iter"])),
                    ("Tolerance", f"{self.params['tolerance']:.3e}"),
                    (
                        "Under-relaxation",
                        f"U={self.params['alpha_u']}, p={self.params['alpha_p']}",
                    ),
                ],
            )

        # Initialize Flux (phi) if not provided
        from ..assemble import convection

        phi = convection.compute_volumetric_face_flux(U, self.mesh_data, self.geo_data)

        for iteration in range(int(self.params["max_iter"])):
            U, p, phi, residuals = self.step(U, p, phi, rho=rho, nu=nu)

            residual_p = self.last_res_p
            residual_u = residuals["U_increment"]
            continuity = self.last_outer_diagnostics[-1].continuity_max

            self.residuals.append(
                {
                    "iter": iteration,
                    "R_p": residual_p,
                    "R_u": residual_u,
                    "continuity": continuity,
                }
            )

            if logger is not None and (
                iteration % 10 == 0 or residual_p < self.params["tolerance"]
            ):
                logger.message(
                    f"  Iter {iteration:3d}: R_p={residual_p:.3e}, "
                    f"ΔU={residual_u:.3e}, continuity={continuity:.3e}"
                )

            if (
                residual_p < self.params["tolerance"]
                and residual_u < self.params["tolerance"]
                and continuity < self.params["tolerance"]
            ):
                if logger is not None:
                    logger.info(f"SIMPLE converged in {iteration} iterations")
                return U, p, phi, True

        if logger is not None:
            logger.warning(f"SIMPLE did not converge in {self.params['max_iter']} iterations")
        return U, p, phi, False
