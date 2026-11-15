"""Boundary ghost values of the velocity field.

One rule serves every site that refreshes the ghosts: field initialisation and
restart, the momentum predictor, and the end of each pressure correction. Per
velocity boundary type:

- ``zero_gradient``: the owner-cell value.
- ``inlet_outlet`` and ``freestream``: the owner-cell value on outflow faces and
  the prescribed value on inflow faces. A face is outflow where
  ``_freestream_outflow`` marks it, otherwise where the face flux is not
  negative.
- ``fixed_value``: the prescribed value.
- ``empty`` and ``slip``: the owner-cell velocity without its normal component.
- ``mixed_velocity``: :func:`update_mixed_velocity_boundary`.
- ``cyclic``: the paired owner cell.
"""

import numpy as np

from ..schemes.boundaries import BoundaryStrategy, boundary_strategy
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
