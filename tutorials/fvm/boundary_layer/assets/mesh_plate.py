"""Compatibility exports for installed tutorial support helpers."""

from openonda.tutorial_support.fvm_boundary_layer.mesh_plate import (
    wall_normal_coords as wall_normal_coords,
    plate_coords as plate_coords,
    flat_plate_mesh as flat_plate_mesh,
)


if __name__ == "__main__":
    mesh, depth = flat_plate_mesh()
    print(f"boundary_layer mesh: {mesh['n_cells']} cells, depth {depth}")
    for b in mesh["boundary"]:
        print(f"  {b['name']:<8} {b['n_faces']:>6} faces  ({b['type']})")
