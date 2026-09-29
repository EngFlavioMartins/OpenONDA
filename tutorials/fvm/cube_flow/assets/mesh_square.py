"""Compatibility exports for installed tutorial support helpers."""

from openonda.tutorial_support.fvm_cube_flow.mesh_square import (
    graded_coords as graded_coords,
    rectilinear_box_with_hole as rectilinear_box_with_hole,
    square_cylinder_mesh as square_cylinder_mesh,
)


if __name__ == "__main__":
    mesh, depth = square_cylinder_mesh()
    print(f"cube_flow square-cylinder mesh: {mesh['n_cells']} cells, depth {depth}")
