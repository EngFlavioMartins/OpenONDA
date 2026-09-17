#!/usr/bin/env python3
"""Copy-and-run STL to Cartesian FVM tutorial.

Run ``python setup.py`` from this directory.  The ordinary public API builds
the mesh, writes ``solution/fvm/mesh.npz`` and ``solution/fvm/mesh.vtu``, and advances
the short inlet/wall flow case for twenty steps.
"""

from __future__ import annotations

from pathlib import Path

import openonda.fvm as fvm
import openonda.fvm.mesher as msh

CASE_DIR = Path(__file__).resolve().parent
OBJECT_STL = CASE_DIR / "assets" / "object.stl"


def create_mesher() -> msh.CartesianMesher:
    """Declare the STL, box, background size, and both refinement controls."""
    return msh.CartesianMesher(
        surfaces=(msh.STLSurface(OBJECT_STL, patch="body"),),
        domain=msh.BoxDomain(
            bounds=(-2.0, 3.0, -1.5, 1.5, -1.0, 1.0),
            patches=msh.BoxPatches(
                xmin="inlet",
                xmax="outlet",
                ymin="sides",
                ymax="sides",
                zmin="span",
                zmax="span",
            ),
        ),
        max_cell_size=0.5,
        boundary_cell_size=0.25,
        refinements=(
            msh.BoxRefinement(
                name="wake",
                bounds=(-1.5, 2.0, -0.75, 0.75, -0.5, 0.5),
                cell_size=0.25,
            ),
        ),
        patch_refinements=(msh.PatchRefinement("body", cell_size=0.25),),
    )


def create_fvm_setup() -> fvm.FVMSetup:
    """Return the physics configuration independently of mesh construction."""
    return fvm.FVMSetup(
        case_name="cartesian_mesher",
        time=fvm.TimeConfig(
            time_step_size=0.01,
            end_time=0.20,
            output_schedule=fvm.RunSchedule(every_n_steps=20),
        ),
        schemes=fvm.DiscretizationConfig(
            convection_scheme="upwind",
            gradient_scheme="gauss",
        ),
        linear=fvm.LinearSolverConfig(linear_solver="spsolve"),
        pimple=fvm.PimpleControl(n_correctors=1, n_outer_correctors=1),
        transport=fvm.TransportConfig(density=1.0, kinematic_viscosity=0.01),
        boundaries=[
            fvm.BoundaryConfig.inlet("inlet", [1.0, 0.0, 0.0]),
            fvm.BoundaryConfig.outlet("outlet"),
            fvm.BoundaryConfig.slip("sides"),
            fvm.BoundaryConfig.slip("span"),
            fvm.BoundaryConfig.wall("body"),
        ],
        initial_velocity=[1.0, 0.0, 0.0],
    )


def main() -> None:
    mesher = create_mesher()
    solver = fvm.create_fvm_solver(
        create_fvm_setup(),
        case_dir=CASE_DIR,
        mesh=mesher,
    )
    solver.run()


if __name__ == "__main__":
    main()
