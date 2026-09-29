"""Compatibility exports for installed tutorial support helpers."""

from openonda.tutorial_support.fvm_cylinder_ibm.mesh_rectilinear import (
    graded_coords as graded_coords,
    rectilinear_box_2d as rectilinear_box_2d,
    cylinder_ibm_mesh as cylinder_ibm_mesh,
)


if __name__ == "__main__":
    mesh, depth = cylinder_ibm_mesh()
    nx = len(np.unique(mesh["vertex_position"][:, 0])) - 1
    ny = len(np.unique(mesh["vertex_position"][:, 1])) - 1
    print(f"cylinder_ibm mesh: {nx} x {ny} = {mesh['n_cells']} cells, depth {depth}")
