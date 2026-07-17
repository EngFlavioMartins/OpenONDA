import copy

import numpy as np

from source.solvers.FVM.config.types import BoundaryConfig
from source.solvers.FVM.mesh import geometry
from source.solvers.FVM.solve.simple_solver import (
    _pressure_requires_constraint,
    _update_velocity_bcs,
    update_scalar_boundaries,
)


def _freestream_patch(hand_built_3d_mesh):
    mesh = copy.deepcopy(hand_built_3d_mesh)
    patch = mesh["boundary"][0]
    patch.update(
        bc_type_U="freestream",
        bc_type_p="freestream",
        value_U=np.array([4.0, 0.0, 0.0]),
        value_p=2.5,
    )
    return mesh, patch, geometry.compute_mesh_geometry(mesh)


def test_freestream_factory_exposes_switching_contract():
    boundary = BoundaryConfig.freestream("farfield", [1.0, 0.0, 0.0], p=3.0)
    assert boundary.type_U == "freestream"
    assert boundary.type_p == "freestream"
    assert boundary.value_p == 3.0


def test_freestream_switches_velocity_and_pressure_per_face(hand_built_3d_mesh):
    mesh, patch, geo = _freestream_patch(hand_built_3d_mesh)
    n_cells = mesh["n_elements"]
    n_interior = mesh["n_interior_faces"]
    n_total = n_cells + mesh["n_faces"] - n_interior
    owners = mesh["owners"]
    faces = np.arange(patch["startFace"], patch["startFace"] + patch["nFaces"])
    ghosts = n_cells + faces - n_interior

    velocity = np.zeros((n_total, 3))
    velocity[:n_cells, 1] = np.arange(1, n_cells + 1)
    face_flux = np.zeros(mesh["n_faces"])
    face_flux[faces] = [-1.0, 1.0, -2.0, 2.0]
    _update_velocity_bcs(
        velocity,
        face_flux,
        [patch],
        owners,
        geo,
        n_cells,
        n_interior,
    )

    inflow = face_flux[faces] < 0.0
    assert np.allclose(velocity[ghosts[inflow]], patch["value_U"])
    assert np.allclose(velocity[ghosts[~inflow]], velocity[owners[faces[~inflow]]])

    pressure = np.arange(n_total, dtype=float)
    update_scalar_boundaries(pressure, mesh, [patch], field_name="p", face_flux=face_flux)
    assert np.allclose(pressure[ghosts[~inflow]], patch["value_p"])
    assert np.allclose(pressure[ghosts[inflow]], pressure[owners[faces[inflow]]])


def test_freestream_preserves_per_face_inflow_values(hand_built_3d_mesh):
    """A characteristic donor survives pressure-correction BC refreshes."""
    mesh, patch, geo = _freestream_patch(hand_built_3d_mesh)
    n_cells = mesh["n_elements"]
    n_interior = mesh["n_interior_faces"]
    n_total = n_cells + mesh["n_faces"] - n_interior
    owners = mesh["owners"]
    faces = np.arange(patch["startFace"], patch["startFace"] + patch["nFaces"])
    ghosts = n_cells + faces - n_interior
    patch["value_U_field"] = np.column_stack(
        [
            np.linspace(0.6, 0.9, len(faces)),
            np.linspace(-0.2, 0.2, len(faces)),
            np.zeros(len(faces)),
        ]
    )

    velocity = np.zeros((n_total, 3))
    velocity[:n_cells] = [2.0, 1.0, 0.0]
    face_flux = np.zeros(mesh["n_faces"])
    face_flux[faces] = [-1.0, 1.0, -2.0, 2.0]
    _update_velocity_bcs(
        velocity,
        face_flux,
        [patch],
        owners,
        geo,
        n_cells,
        n_interior,
    )

    inflow = face_flux[faces] < 0.0
    np.testing.assert_allclose(
        velocity[ghosts[inflow]],
        patch["value_U_field"][inflow],
    )
    np.testing.assert_allclose(
        velocity[ghosts[~inflow]],
        velocity[owners[faces[~inflow]]],
    )


def test_freestream_pressure_constraint_depends_on_flow_direction(hand_built_3d_mesh):
    mesh, patch, geo = _freestream_patch(hand_built_3d_mesh)
    n_cells = mesh["n_elements"]
    n_interior = mesh["n_interior_faces"]
    n_total = n_cells + mesh["n_faces"] - n_interior
    faces = np.arange(patch["startFace"], patch["startFace"] + patch["nFaces"])
    ghosts = n_cells + faces - n_interior
    velocity = np.zeros((n_total, 3))

    normals = geo["face_sf"][faces]
    velocity[ghosts] = -normals
    assert _pressure_requires_constraint([patch], velocity, mesh, geo)

    velocity[ghosts[0]] = normals[0]
    assert not _pressure_requires_constraint([patch], velocity, mesh, geo)
