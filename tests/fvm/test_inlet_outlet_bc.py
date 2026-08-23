"""inletOutlet velocity BC: zeroGradient on outflow, fixed value on inflow.

Replaces the old patch-name backflow heuristic.  This checks the per-face
switching directly: ghost cells of outgoing faces copy the owner; ghost cells of
incoming (reverse-flow) faces take the boundary ``value_velocity`` (inletValue).
"""

import numpy as np

from source.solvers.fvm.mesh.geometry import compute_mesh_geometry
from source.solvers.fvm.solve.simple_solver import _update_velocity_bcs

from ._structured_mesh import structured_box


def test_inlet_outlet_switches_on_flux_sign():
    mesh = structured_box(4, 4, 1)
    geo = compute_mesh_geometry(mesh)
    n_elem = mesh["n_cells"]
    n_int = mesh["n_interior_faces"]
    owners = mesh["owners"]

    patch = next(b for b in mesh["boundary"] if b["name"] == "xmax")
    patch["velocity_type"] = "inletOutlet"
    patch["velocity_value"] = [5.0, 0.0, 0.0]
    start, nf = patch["start_face"], patch["n_faces"]

    n_bnd = mesh["n_faces"] - n_int
    velocity = np.zeros((n_elem + n_bnd, 3))
    own = owners[start : start + nf]
    velocity[own] = [2.0, 1.0, 0.0]  # owner-cell velocity

    # Alternate outflow / inflow across the patch faces.
    volumetric_face_flux = np.zeros(mesh["n_faces"])
    volumetric_face_flux[start : start + nf] = [1.0 if j % 2 == 0 else -1.0 for j in range(nf)]

    _update_velocity_bcs(velocity, volumetric_face_flux, [patch], owners, geo, n_elem, n_int)

    for j in range(nf):
        ghost = n_elem + (start + j - n_int)
        if j % 2 == 0:  # outflow → zeroGradient (owner value)
            assert np.allclose(velocity[ghost], [2.0, 1.0, 0.0]), (
                f"outflow face {j} not extrapolated"
            )
        else:  # inflow → inletValue
            assert np.allclose(velocity[ghost], [5.0, 0.0, 0.0]), (
                f"inflow face {j} not clamped to inletValue"
            )
