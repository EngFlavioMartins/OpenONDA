"""Boundary ghost values of the velocity and pressure fields.

One rule serves every site that refreshes the velocity ghosts: field
initialisation and restart, the momentum predictor, and the end of each
pressure correction. Per velocity boundary type:

- ``zero_gradient``: the owner-cell value.
- ``inlet_outlet`` and ``freestream``: the owner-cell value on outflow faces and
  the prescribed value on inflow faces. A face is outflow where
  ``_freestream_outflow`` marks it, otherwise where the face flux is not
  negative.
- ``fixed_value``: the prescribed value.
- ``empty`` and ``slip``: the owner-cell velocity without its normal component.
- ``mixed_velocity``: :func:`update_mixed_velocity_boundary`.
- ``cyclic``: the paired owner cell.

The module also holds the selection of freestream inflow and outflow faces, the
scalar ghost update (kinematic pressure) and the ``fixed_flux_pressure`` value
that makes the boundary flux follow the predicted velocity.
"""

import numpy as np

from ..schemes.boundaries import BoundaryStrategy, boundary_strategy
from . import gradients
from .boundary_reconstruction import boundary_owner_gradient, boundary_resolved_displacement
from .mixed_velocity_boundary import update_mixed_velocity_boundary


def _patch_slices(boundary: dict[str, object], mesh_data: dict[str, object]) -> tuple[slice, slice]:
    """Return the face slice and the ghost-cell slice of one boundary patch."""
    start = boundary["start_face"]
    n_faces = boundary["n_faces"]
    ghost = mesh_data["n_cells"] + (start - mesh_data["n_interior_faces"])
    return slice(start, start + n_faces), slice(ghost, ghost + n_faces)


def _inflow_velocity(boundary: dict[str, object], n_faces: int) -> np.ndarray:
    """Return the prescribed velocity of an inlet_outlet or freestream patch."""
    if boundary.get("velocity_value_field") is not None:
        value = np.asarray(boundary["velocity_value_field"], dtype=float)
        if value.shape != (n_faces, 3):
            raise ValueError(
                f"Per-face inlet velocity for patch {boundary.get('name')!r} "
                f"must have shape ({n_faces}, 3), got {value.shape}"
            )
        return value
    value = np.asarray(boundary.get("velocity_value", [0.0, 0.0, 0.0]), dtype=float)
    if value.shape != (3,):
        raise ValueError(
            f"Uniform inlet velocity for patch {boundary.get('name')!r} "
            f"must have shape (3,), got {value.shape}"
        )
    return value


def _remove_normal_component(
    owner_velocity: np.ndarray, face_area_vector: np.ndarray
) -> np.ndarray:
    """Return ``owner_velocity`` without its normal component; degenerate faces keep it."""
    magnitudes = np.linalg.norm(face_area_vector, axis=1)
    valid = magnitudes > 1e-10
    projected = owner_velocity.copy()
    if np.any(valid):
        normals = face_area_vector[valid] / magnitudes[valid, np.newaxis]
        normal_velocity = np.sum(owner_velocity[valid] * normals, axis=1)
        projected[valid] -= normal_velocity[:, np.newaxis] * normals
    return projected


def update_velocity_boundaries(
    velocity: np.ndarray,
    face_flux: np.ndarray,
    boundaries: list[dict],
    mesh_data: dict[str, object],
    geo_data: dict[str, object],
) -> None:
    """Set the boundary ghost velocities of ``velocity`` in place.

    ``face_flux`` selects the outflow faces of ``inlet_outlet`` and ``freestream``
    patches that carry no ``_freestream_outflow`` mask; on a moving mesh pass the
    flux relative to the moving faces.
    """
    owners = mesh_data["owners"]
    for boundary in boundaries:
        strategy = boundary_strategy(
            boundary.get("velocity_type") or boundary.get("boundary_condition_type"), "velocity"
        )
        faces, ghosts = _patch_slices(boundary, mesh_data)
        if strategy is BoundaryStrategy.ZERO_GRADIENT:
            velocity[ghosts] = velocity[owners[faces]]
        elif strategy in (BoundaryStrategy.INLET_OUTLET, BoundaryStrategy.FREESTREAM):
            outflow = boundary.get("_freestream_outflow")
            if outflow is None:
                outflow = face_flux[faces] >= 0.0
            velocity[ghosts] = np.where(
                outflow[:, np.newaxis],
                velocity[owners[faces]],
                _inflow_velocity(boundary, boundary["n_faces"]),
            )
        elif strategy is BoundaryStrategy.FIXED_VALUE:
            if boundary.get("velocity_value_field") is not None:
                velocity[ghosts] = boundary["velocity_value_field"]
            elif "velocity_value" in boundary:
                velocity[ghosts] = np.array(boundary["velocity_value"])
            else:
                raise ValueError(
                    f"Fixed velocity boundary {boundary.get('name')!r} has no configured value"
                )
        elif strategy is BoundaryStrategy.MIXED_VELOCITY:
            update_mixed_velocity_boundary(velocity, boundary, mesh_data, geo_data)
        elif strategy in (BoundaryStrategy.EMPTY, BoundaryStrategy.SLIP):
            velocity[ghosts] = _remove_normal_component(
                velocity[owners[faces]], geo_data["face_area_vector"][faces]
            )
        elif strategy is BoundaryStrategy.CYCLIC:
            paired = mesh_data["boundary_neighbour_cell"][faces]
            if np.any(paired < 0):
                raise ValueError(f"Cyclic patch {boundary['name']!r} is not paired")
            velocity[ghosts] = velocity[paired]
        else:
            raise RuntimeError(f"Unhandled velocity ghost strategy {strategy!r}")


def select_freestream_faces(
    boundary: dict,
    face_flux: np.ndarray,
    geo_data: dict[str, object],
    mesh_face_flux: np.ndarray | None = None,
) -> None:
    r"""Classify the faces of one freestream patch for pressure and velocity.

    The pressure branch follows the prescribed stream,
    :math:`\mathbf{U}_\infty\cdot\mathbf{S}_f - \phi_{mesh} \ge 0`: downstream
    faces carry the pressure datum, upstream faces a zero normal gradient.  On
    a static mesh this split is fixed in time.  The velocity is the prescribed
    stream on upstream faces and ``inlet_outlet`` on downstream faces (zero
    gradient for an outgoing flux, the stream on reverse flow).

    Selecting the pressure branch with the instantaneous face flux instead
    switches it wherever the local flux reverses: on nearly tangential faces
    during the impulsive start and when wake vortices cross the far field.
    Even a few faces switching between Dirichlet and Neumann pressure leave a
    disturbance that does not decay and later destabilises a circular far
    field. The velocity branch may follow the flux without that effect.

    Sets ``_freestream_pressure_outflow`` (pressure Dirichlet faces) and
    ``_freestream_outflow`` (velocity zero-gradient faces).
    """
    start = int(boundary["start_face"])
    stop = start + int(boundary["n_faces"])
    stream_flux = _freestream_stream_flux(boundary, geo_data)
    flux = np.asarray(face_flux[start:stop], dtype=float)
    if mesh_face_flux is not None:
        stream_flux = stream_flux - mesh_face_flux[start:stop]
        flux = flux - mesh_face_flux[start:stop]
    downstream = stream_flux >= 0.0
    boundary["_freestream_pressure_outflow"] = downstream
    boundary["_freestream_outflow"] = downstream & (flux >= 0.0)


def _freestream_stream_flux(boundary: dict, geo_data: dict[str, object]) -> np.ndarray:
    """Return the prescribed-stream flux ``U_inf . S_f`` through each patch face."""
    start = int(boundary["start_face"])
    surface = geo_data["face_area_vector"][start : start + int(boundary["n_faces"])]
    stream = boundary.get("velocity_value_field")
    if stream is None:
        stream = np.broadcast_to(np.asarray(boundary["velocity_value"], dtype=float), surface.shape)
    return np.einsum("ij,ij->i", stream, surface)


def freestream_pressure_outflow(boundary: dict, geo_data: dict[str, object]) -> np.ndarray:
    """Return the freestream faces with a fixed pressure (downstream of the stream)."""
    outflow = boundary.get("_freestream_pressure_outflow")
    if outflow is None:
        outflow = _freestream_stream_flux(boundary, geo_data) >= 0.0
    return np.asarray(outflow, dtype=bool)


def prepare_freestream_boundaries(
    velocity: np.ndarray,
    pressure: np.ndarray,
    face_flux: np.ndarray,
    mesh_data: dict[str, object],
    geo_data: dict[str, object],
    boundaries: list[dict],
    mesh_face_flux: np.ndarray | None = None,
) -> None:
    """Select each mixed boundary branch once for the momentum/projection solve.

    On a moving mesh the branch follows the flux relative to the moving face,
    ``face_flux - mesh_face_flux``, and ``inlet_outlet`` patches are selected
    once per step as well. Freestream faces are classified by
    :func:`select_freestream_faces`.
    """
    selection_flux = face_flux if mesh_face_flux is None else face_flux - mesh_face_flux
    changed = False
    for boundary in boundaries:
        strategy = boundary_strategy(boundary.get("velocity_type"), "velocity")
        if strategy is BoundaryStrategy.FREESTREAM:
            select_freestream_faces(boundary, face_flux, geo_data, mesh_face_flux)
        elif strategy is BoundaryStrategy.INLET_OUTLET and mesh_face_flux is not None:
            start = int(boundary["start_face"])
            stop = start + int(boundary["n_faces"])
            boundary["_freestream_outflow"] = np.asarray(selection_flux[start:stop] >= 0.0)
        else:
            continue
        changed = True
    if changed:
        update_velocity_boundaries(velocity, face_flux, boundaries, mesh_data, geo_data)
        update_scalar_boundaries(
            pressure,
            mesh_data,
            boundaries,
            field_name="kinematic_pressure",
            volumetric_face_flux=face_flux,
        )


def update_fixed_flux_pressure_boundaries(
    kinematic_pressure: np.ndarray,
    velocity_star: np.ndarray,
    pressure_velocity_coefficient: np.ndarray,
    mesh_data: dict[str, object],
    geo_data: dict[str, object],
    boundaries: list[dict],
    kinematic_pressure_gradient: np.ndarray = None,
    pressure_free_face_flux: np.ndarray | None = None,
) -> np.ndarray | None:
    """Update ``fixed_flux_pressure`` from the pressure-free face flux."""
    n_cells = mesh_data["n_cells"]
    n_interior = mesh_data["n_interior_faces"]
    owners = mesh_data["owners"]
    face_area_vector = geo_data["face_area_vector"]
    face_cf = geo_data["cell_connection_vector"]
    fixed_flux_patches = []
    for boundary in boundaries:
        strategy = boundary_strategy(boundary.get("pressure_type"), "kinematic_pressure")
        if strategy is BoundaryStrategy.FIXED_FLUX_PRESSURE:
            fixed_flux_patches.append(boundary)
    parallel = mesh_data.get("_parallel_context")
    has_fixed_flux = bool(fixed_flux_patches)
    if parallel is not None and parallel.is_partitioned:
        has_fixed_flux = bool(parallel.global_max(int(has_fixed_flux)))
    if not has_fixed_flux:
        return kinematic_pressure_gradient

    if kinematic_pressure_gradient is None:
        kinematic_pressure_gradient = gradients.gradient(kinematic_pressure, mesh_data, geo_data)
    assert kinematic_pressure_gradient is not None

    if pressure_free_face_flux is not None:
        pressure_free_face_flux = np.asarray(pressure_free_face_flux, dtype=float)
        if pressure_free_face_flux.shape != (mesh_data["n_faces"],):
            raise ValueError("pressure_free_face_flux must have one value per face")
    elif fixed_flux_patches:
        pressure_velocity_coefficient_vector = (
            pressure_velocity_coefficient[:, np.newaxis]
            if np.asarray(pressure_velocity_coefficient).ndim == 1
            else pressure_velocity_coefficient
        )
        velocity_h_over_a = (
            velocity_star[:n_cells]
            + pressure_velocity_coefficient_vector * kinematic_pressure_gradient[:n_cells]
        )
        pressure_free_face_flux = fixed_flux_pressure_face_drive(
            velocity_h_over_a,
            mesh_data,
            geo_data,
            boundaries,
            velocity_star=velocity_star,
            pressure_velocity_coefficient=pressure_velocity_coefficient,
            predictor_pressure_gradient=kinematic_pressure_gradient,
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
        grad_owner = kinematic_pressure_gradient[own]
        grad_tangent = grad_owner - np.einsum("ij,ij->i", grad_owner, normal)[:, None] * normal
        if np.asarray(pressure_velocity_coefficient).ndim == 1:
            D_normal = pressure_velocity_coefficient[own]
            tangential_pressure_flux = np.zeros(nf)
        else:
            D_normal = np.einsum("ij,ij->i", pressure_velocity_coefficient[own], normal * normal)
            tangential_pressure_flux = np.einsum(
                "ij,ij->i", normal * pressure_velocity_coefficient[own], grad_tangent
            )
        if np.any(~np.isfinite(D_normal)) or np.any(D_normal <= 0):
            raise ValueError(
                "fixed_flux_pressure requires a finite positive normal momentum inverse"
            )
        phi_target = np.einsum("ij,ij->i", velocity_star[ghost], sf)
        assert pressure_free_face_flux is not None
        volumetric_face_flux_h_over_a = pressure_free_face_flux[start : start + nf]
        pressure_flux_coefficient = mag_sf * D_normal
        dpdn = (
            volumetric_face_flux_h_over_a - phi_target
        ) / pressure_flux_coefficient - tangential_pressure_flux / D_normal
        tangential_displacement = dr - normal_distance[:, None] * normal
        delta = dpdn * normal_distance + np.einsum(
            "ij,ij->i", grad_tangent, tangential_displacement
        )
        boundary["fixed_flux_pressure_delta"] = delta
        kinematic_pressure[ghost] = kinematic_pressure[own] + delta
        changed = True

    if parallel is not None and parallel.is_partitioned:
        # Gradient reconstruction exchanges halo data, including on ranks
        # without physical boundary faces. Its entry must be collective.
        changed = bool(parallel.global_max(int(changed)))
    if changed:
        kinematic_pressure_gradient = gradients.gradient(kinematic_pressure, mesh_data, geo_data)
    return kinematic_pressure_gradient


def fixed_flux_pressure_face_drive(
    velocity_h_over_a: np.ndarray,
    mesh_data: dict[str, object],
    geo_data: dict[str, object],
    boundaries: list[dict],
    *,
    velocity_star: np.ndarray = None,
    pressure_velocity_coefficient: np.ndarray = None,
    predictor_pressure_gradient: np.ndarray = None,
) -> np.ndarray | None:
    """Extrapolate one frozen pressure-free predictor to its boundary faces.

    Real-cell/processor-halo values supply the face location independently of
    physical ghosts or patch update order. This supplies the missing face
    location without imposing a second pressure condition or rebuilding a
    full-domain gradient. The momentum solve and collective pressure-gradient
    reconstruction have already refreshed the required halo values.
    """
    faces = []
    for boundary in boundaries:
        strategy = boundary_strategy(boundary.get("pressure_type"), "kinematic_pressure")
        if boundary.get("fixed_flux_pressure_external", False):
            continue
        if strategy is BoundaryStrategy.FIXED_FLUX_PRESSURE:
            start = int(boundary["start_face"])
            faces.append(np.arange(start, start + int(boundary["n_faces"])))
    if not faces:
        return None
    result = np.zeros(mesh_data["n_faces"])
    faces = np.concatenate(faces)
    owners = mesh_data["owners"]
    owner_gradient = boundary_owner_gradient(
        velocity_h_over_a,
        mesh_data,
        geo_data,
        faces,
        include_physical_ghosts=False,
    )
    dr = geo_data["cell_connection_vector"][faces]
    face_velocity = velocity_h_over_a[owners[faces]] + np.einsum("fji,fj->fi", owner_gradient, dr)
    unresolved = dr - boundary_resolved_displacement(mesh_data, geo_data, faces, dr)
    missing = np.linalg.norm(unresolved, axis=1) > 1e-8 * np.linalg.norm(dr, axis=1) + 1e-13
    if np.any(missing):
        # A one-cell-thick fluid layer cannot determine every derivative from
        # real neighbours. Preserve ALL identifiable components and close only
        # the missing directions with the already accepted/lagged native faces.
        # These inputs are captured before pressure ghosts are refreshed, and
        # the completed drive stays frozen for the whole momentum predictor.
        if (
            velocity_star is None
            or pressure_velocity_coefficient is None
            or predictor_pressure_gradient is None
        ):
            raise ValueError(
                "Unresolved pressure-face direction requires lagged native predictor faces"
            )
        n_cells, n_interior = mesh_data["n_cells"], mesh_data["n_interior_faces"]
        D_vector = (
            pressure_velocity_coefficient[:, None]
            if np.asarray(pressure_velocity_coefficient).ndim == 1
            else pressure_velocity_coefficient
        )
        native_predictor = np.empty_like(velocity_star)
        native_predictor[:n_cells] = velocity_h_over_a
        native_predictor[n_cells:] = (
            velocity_star[n_cells:]
            + D_vector[owners[n_interior:]] * predictor_pressure_gradient[n_cells:]
        )
        missing_gradient = boundary_owner_gradient(
            native_predictor,
            mesh_data,
            geo_data,
            faces[missing],
            include_physical_ghosts=True,
            displacements=unresolved[missing],
        )
        face_velocity[missing] += np.einsum("fji,fj->fi", missing_gradient, unresolved[missing])
    result[faces] = np.einsum("ij,ij->i", face_velocity, geo_data["face_area_vector"][faces])
    return result


def _apply_scalar_bc(
    field_values: np.ndarray,
    indices: np.ndarray,
    owners_b: np.ndarray,
    strategy: "BoundaryStrategy",
    boundary: dict[str, object],
    field_name: str,
    paired_owners: np.ndarray = None,
    volumetric_face_flux: np.ndarray | None = None,
) -> None:
    """Apply a boundary condition to a scalar field ghost-cell block.

    Zero-gradient BCs (including inlet, outlet, empty, slip)
    copy the owner value.  ``fixed_value`` sets the prescribed
    boundary value.

    Args:
        field_values: Scalar field array (mutated in place).
        indices:    Ghost-cell indices for this patch.
        owners_b:   Owner cell indices for the boundary faces.
        strategy:   Validated boundary behavior.
        boundary:   Boundary patch dictionary with standard field-value keys.
        field_name: Field name for value lookup (e.g. ``"kinematic_pressure"``,
            ``"scalar_field"``).
    """
    if strategy in (BoundaryStrategy.ZERO_GRADIENT, BoundaryStrategy.EMPTY):
        field_values[indices] = field_values[owners_b]
    elif strategy is BoundaryStrategy.FIXED_FLUX_PRESSURE:
        delta = boundary.get("fixed_flux_pressure_delta")
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
        outflow = boundary.get(
            "_freestream_pressure_outflow"
            if field_name == "kinematic_pressure"
            else "_freestream_outflow"
        )
        if outflow is None:
            # Before the first step selects the freestream branches.
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
        field_values[indices] = np.where(outflow, val, field_values[owners_b])
    else:
        raise ValueError(
            f"Unsupported scalar boundary strategy {strategy!r} "
            f"for {field_name} on patch {boundary.get('name')!r}"
        )


def update_scalar_boundaries(
    field_values: np.ndarray,
    mesh_data: dict[str, object],
    boundaries: list[dict],
    field_name: str = "kinematic_pressure",
    volumetric_face_flux: np.ndarray | None = None,
) -> None:
    """
    Update boundary values (ghost cells) for a scalar field.

    Args:
        field_values: Scalar field (n_cells + n_boundary)
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

        boundary_type_key = (
            "pressure_type" if field_name == "kinematic_pressure" else f"{field_name}_type"
        )
        boundary_condition_type = boundary.get(boundary_type_key) or boundary.get(
            "boundary_condition_type"
        )
        registry_field = "kinematic_pressure" if field_name == "kinematic_pressure" else "scalar"
        strategy = boundary_strategy(boundary_condition_type, registry_field)
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
