"""Diffusive boundary-operator consistency tests."""

import numpy as np

from source.solvers.FVM.assemble.diffusion import assemble_diffusion_term
from source.solvers.FVM.fields.gradients import compute_gauss_gradient
from source.solvers.FVM.mesh.geometry import compute_mesh_geometry

from ._structured_mesh import structured_box


def _scalar_field(mesh, value=1.0):
    n = mesh["n_elements"]
    nb = mesh["n_faces"] - mesh["n_interior_faces"]
    return np.full(n + nb, value, dtype=float)


def test_no_slip_diffusion_does_not_trust_stale_ghost_value():
    mesh = structured_box(2, 1, 1)
    geo = compute_mesh_geometry(mesh)
    field = _scalar_field(mesh, value=3.0)  # deliberately non-zero ghosts
    for patch in mesh["boundary"]:
        patch["bc_type_velocity"] = "zeroGradient"
    wall = mesh["boundary"][0]
    wall["bc_type_velocity"] = "noSlip"

    grad = compute_gauss_gradient(field, mesh, geo)
    flux = assemble_diffusion_term(
        field, grad, np.ones(mesh["n_elements"]), mesh, geo, mesh["boundary"]
    )
    sl = slice(wall["startFace"], wall["startFace"] + wall["nFaces"])
    assert np.all(flux["flux_cf"][sl] > 0.0)
    assert np.allclose(flux["flux_vf"][sl], 0.0)


def test_inlet_outlet_diffusion_acts_only_on_inflow_faces():
    mesh = structured_box(2, 1, 1)
    geo = compute_mesh_geometry(mesh)
    field = _scalar_field(mesh, value=2.0)
    for patch in mesh["boundary"]:
        patch["bc_type_velocity"] = "zeroGradient"
    patch = mesh["boundary"][0]
    patch["bc_type_velocity"] = "inletOutlet"
    face_flux = np.zeros(mesh["n_faces"])
    indices = np.arange(patch["startFace"], patch["startFace"] + patch["nFaces"])
    face_flux[indices] = -1.0

    grad = compute_gauss_gradient(field, mesh, geo)
    inflow = assemble_diffusion_term(
        field,
        grad,
        np.ones(mesh["n_elements"]),
        mesh,
        geo,
        mesh["boundary"],
        face_flux=face_flux,
    )
    assert np.all(inflow["flux_cf"][indices] > 0.0)

    face_flux[indices] = 1.0
    outflow = assemble_diffusion_term(
        field,
        grad,
        np.ones(mesh["n_elements"]),
        mesh,
        geo,
        mesh["boundary"],
        face_flux=face_flux,
    )
    assert np.allclose(outflow["flux_cf"][indices], 0.0)
    assert np.allclose(outflow["flux_vf"][indices], 0.0)
