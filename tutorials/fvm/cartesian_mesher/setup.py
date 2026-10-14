#!/usr/bin/env python3
"""Copy-and-run STL to Cartesian FVM tutorial.

Run ``python setup.py`` from this directory.  The ordinary public API builds
the mesh, writes ``assets/cartesian_mesher_mesh.npz`` and its ``.vtu`` view, and advances
the short inlet/wall flow case for twenty steps.
"""

from __future__ import annotations

from pathlib import Path

import openonda.fvm as fvm
import openonda.fvm.mesher as msh

START_FROM = "latest"  # Resume the latest backup; ./allrun.sh cleans first.

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


def create_fvm_case() -> fvm.FVMCase:
    """Return the inlet/wall flow around the meshed STL object."""
    every_twenty_steps = fvm.EverySteps(20)
    return fvm.FVMCase(
        name="cartesian_mesher",
        directory=CASE_DIR,
        numerics=fvm.Numerics(
            time_step_size=0.01,
            transport=fvm.TransportConfig(density=1.0, kinematic_viscosity=0.01),
            schemes=fvm.DiscretizationConfig(convection_scheme="upwind", gradient_scheme="gauss"),
            linear=fvm.LinearSolverConfig(linear_solver="spsolve"),
            pimple=fvm.PimpleConfig(n_correctors=1),
        ),
        initial_conditions=fvm.InitialFields(velocity=(1.0, 0.0, 0.0)),
        run=fvm.RunPlan(end_time=0.20),
        output=fvm.Output(schedule=every_twenty_steps),
        backup=fvm.Backup(schedule=every_twenty_steps),
        mesh=create_mesher(),
        boundaries=(
            fvm.BoundaryConfig.inlet("inlet", (1.0, 0.0, 0.0)),
            fvm.BoundaryConfig.outlet("outlet"),
            fvm.BoundaryConfig.slip("sides"),
            fvm.BoundaryConfig.slip("span"),
            fvm.BoundaryConfig.wall("body"),
        ),
    )


def main() -> None:
    with fvm.FVMSolver(create_fvm_case()) as solver:
        solver.run(start_from=START_FROM)


if __name__ == "__main__":
    main()
