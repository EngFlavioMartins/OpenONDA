"""Compatibility exports for installed tutorial support helpers."""

from openonda.tutorial_support.fvm_step_profile.mesh_step import (
    backward_facing_step_mesh as backward_facing_step_mesh,
)


if __name__ == "__main__":
    mesh, mesh_depth = backward_facing_step_mesh()
    print(f"step_profile mesh: {mesh['n_cells']} cells, depth {mesh_depth:g}")
    for patch in mesh["boundary"]:
        print(f"  {patch['name']:<8} {patch['n_faces']:>6} faces ({patch['type']})")
