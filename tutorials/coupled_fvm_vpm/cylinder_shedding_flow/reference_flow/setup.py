#!/usr/bin/env python3
"""Body-fitted Re=150 cylinder reference used by the grid study.

Example:
    python -u setup.py --dx 0.025 \
        --case-name coarse
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import openonda.fvm as fvm
import openonda.fvm.mesher as msh

try:
    from .canonical_surface import DOMAIN, prepare_canonical_surfaces
    from .case_definition import (
        domain_for,
        construction_domain,
        extrusion_levels,
        REFINEMENT_REGIONS,
    )
except ImportError:  # Direct ``python setup.py`` execution.
    from canonical_surface import (
        DOMAIN,
        prepare_canonical_surfaces,
    )  # pyrefly: ignore [missing-import]
    from case_definition import (
        domain_for,
        construction_domain,
        extrusion_levels,
        REFINEMENT_REGIONS,
    )  # pyrefly: ignore [missing-import]

CASE_DIR = Path(__file__).resolve().parent
CYLINDER_STL = CASE_DIR.parent / "assets" / "cylinder_long.stl"

# ---- Physics -------------------------------------------------------------
DIAMETER = 1.0
REYNOLDS_NUMBER = 150.0
FREESTREAM_VELOCITY = [1.0, 0.0, 0.0]
KINEMATIC_VISCOSITY = 1.0 / REYNOLDS_NUMBER
CYLINDER_LENGTH = DOMAIN[5] - DOMAIN[4]

# ---- Time and output -----------------------------------------------------
TIME_STEP_SIZE = 0.001
MAX_TIME_STEP_SIZE = 4.0 * TIME_STEP_SIZE
MAX_COURANT_NUMBER = 0.9
# Discard the start-up transient and analyse the final 30 convective units.
TOTAL_TIME = 60.0
FORCE_INTERVAL_TIME = 0.02
LINE_INTERVAL_TIME = 0.1
SLICE_INTERVAL_TIME = 0.5
FIELD_INTERVAL_TIME = 2.5
NUMBER_OF_CORES = 6


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dx", required=True, type=float, help="cylinder-wall cell size in D")
    parser.add_argument("--case-name", required=True, help="solution/ and samples/ subdirectory")
    return parser.parse_args()


def background_cell_size(dx: float, *, domain=DOMAIN) -> float:
    """Scale the far field with every other spacing in the study family."""
    return 8.0 * dx


def grid_mesh(dx: float, *, domain=DOMAIN):
    """Extrude a cfMesh section for the span-invariant Re=150 experiment."""
    from source.solvers.fvm.mesh.cartesian.extrusion import ExtrudedCartesianMesher

    background_size = background_cell_size(dx, domain=domain)
    source_domain = construction_domain(dx, domain)
    canonical = prepare_canonical_surfaces(
        CYLINDER_STL,
        CASE_DIR
        / "mesh_evidence"
        / "extruded_inputs"
        / hashlib.sha256(repr(source_domain).encode()).hexdigest()[:12],
        domain=source_domain,
    )
    patches = msh.BoxPatches("inlet", "outlet", "ymin", "ymax", "zmin", "zmax")
    source = msh.CartesianMesher(
        domain=msh.BoxDomain(bounds=source_domain, patches=patches),
        surfaces=(msh.STLSurface(str(canonical["wall_path"]), patch="cylinder", allow_open=True),),
        max_cell_size=background_size,
        boundary_cell_size=background_size,
        min_cell_size=None,
        refinements=tuple(
            msh.BoxRefinement(
                name=name,
                bounds=(*bounds, source_domain[4], source_domain[5]),
                cell_size=factor * dx * (1.0 + 1.0e-12),
            )
            for name, bounds, factor in REFINEMENT_REGIONS
        ),
        patch_refinements=(msh.PatchRefinement("cylinder", dx),),
        boundary_layers=(),
        surface_may_cross_domain_boundary=True,
    )
    return ExtrudedCartesianMesher(
        source=source,
        domain=msh.BoxDomain(bounds=domain, patches=patches),
        levels=extrusion_levels(dx, domain),
    )


def samplers(dx: float, *, domain=DOMAIN) -> tuple:
    force_schedule = fvm.RunSchedule(every_time=FORCE_INTERVAL_TIME)
    line_schedule = fvm.RunSchedule(every_time=LINE_INTERVAL_TIME)
    slice_schedule = fvm.RunSchedule(every_time=SLICE_INTERVAL_TIME)
    sample_spacing = min(0.125, 2.0 * dx)
    return (
        fvm.ForceSampler(
            patch_names=["cylinder"],
            reference_velocity=1.0,
            reference_area=DIAMETER * (domain[5] - domain[4]),
            reference_length=DIAMETER,
            moment_centre=[0.0, 0.0, 0.0],
            file_name="forces_history",
            schedule=force_schedule,
        ),
        fvm.LineSampler(
            start=[1.5, 0.0, 0.0],
            end=[1.5, 0.0, 0.0],
            n_points=1,
            k=12,
            reconstruction="affine",
            file_name="midspan_probe",
            schedule=force_schedule,
        ),
        fvm.LineSampler(
            start=[-2.0, 0.0, 0.0],
            end=[12.0, 0.0, 0.0],
            spacing=sample_spacing,
            k=12,
            reconstruction="affine",
            file_name="centreline",
            schedule=line_schedule,
        ),
        fvm.LineSampler(
            start=[1.0, -3.0, 0.0],
            end=[1.0, 3.0, 0.0],
            spacing=sample_spacing,
            k=12,
            reconstruction="affine",
            file_name="transverse_x1",
            schedule=line_schedule,
        ),
        fvm.LineSampler(
            start=[2.0, -3.0, 0.0],
            end=[2.0, 3.0, 0.0],
            spacing=sample_spacing,
            k=12,
            reconstruction="affine",
            file_name="transverse_x2",
            schedule=line_schedule,
        ),
        fvm.LineSampler(
            start=[4.0, -3.0, 0.0],
            end=[4.0, 3.0, 0.0],
            spacing=sample_spacing,
            k=12,
            reconstruction="affine",
            file_name="transverse_x4",
            schedule=line_schedule,
        ),
        fvm.SurfaceSampler(
            point=[0.0, 0.0, 0.0],
            normal=[0.0, 0.0, 1.0],
            bounds=[domain[0], domain[1], domain[2], domain[3]],
            spacing=sample_spacing,
            k=12,
            reconstruction="affine",
            file_name="midspan",
            schedule=slice_schedule,
            body_bounds=[-0.5, 0.5, -0.5, 0.5, -6.0, 6.0],
            body_geometry="cylinder_z",
        ),
    )


def solver_setup(case_name: str, dx: float) -> fvm.FVMSetup:
    return fvm.FVMSetup(
        case_name=case_name,
        cores=NUMBER_OF_CORES,
        mesh=fvm.MeshQualityConfig(
            # The complete-box D/8 mesh measures 58.81 degrees and 0.971
            # maximum skewness, with no inverted pyramids or VTK intersections.
            # Its 64 worst transition tetrahedra remain rank-3 QR stencils and
            # peak at LSQ condition 18.49.  Retain a measured margin without
            # allowing rank deficiency or the solver's SVD fallback.
            max_non_orthogonality_deg=86.0,
            max_skewness=1.0,
            max_lsq_condition=25.0,
        ),
        execution=fvm.ComputeConfig(operator_backend="numba"),
        output=fvm.OutputConfig(
            format="vtk_xml",
            data_location="cell",
            encoding="appended",
            compression="lz4",
            precision="f32",
            asynchronous=False,
            ghost_layers=0,
        ),
        logging=fvm.LoggingConfig(schedule=fvm.RunSchedule(every_time=0.1)),
        acceptance=fvm.RunAcceptanceLimits(
            sustained_steps=1,
            max_continuity_error_warning=1.0e-4,
            max_continuity_error_abort=1.0e-2,
            max_equation_residual_warning=1.0e-4,
            max_equation_residual_abort=1.0e-2,
            max_courant_number_warning=MAX_COURANT_NUMBER,
            max_courant_number_abort=1.5,
            max_velocity_magnitude_warning=3.0,
            max_velocity_magnitude_abort=5.0,
        ),
        backup=fvm.BackupConfig(
            schedule=fvm.RunSchedule(every_time=FIELD_INTERVAL_TIME),
            write_at_end=True,
        ),
        time=fvm.TimeConfig(
            time_step_size=TIME_STEP_SIZE,
            start_time=0.0,
            end_time=TOTAL_TIME,
            output_schedule=fvm.RunSchedule(every_time=FIELD_INTERVAL_TIME),
            adjustment=fvm.MaximumCourantTimeStep(
                maximum=MAX_COURANT_NUMBER,
                maximum_time_step_size=MAX_TIME_STEP_SIZE,
            ),
        ),
        schemes=fvm.DiscretizationConfig(
            convection_scheme="limitedLinear",
            gradient_scheme="lsq",
            time_scheme="euler_implicit",
        ),
        linear=fvm.LinearSolverConfig(
            linear_solver="bicgstab",
            pressure_solver="amg",
            pressure_tolerance=1.0e-7,
            pressure_relative_tolerance=0.005,
            momentum_tolerance=1.0e-6,
            momentum_relative_tolerance=0.05,
        ),
        pimple=fvm.PimpleControl(
            n_correctors=2,
            n_outer_correctors=2,
            # The Rhie--Chow assembly already includes the explicit
            # non-orthogonal pressure flux.
            n_orthogonal_correctors=0,
            velocity_relaxation=0.7,
            pressure_relaxation=0.3,
        ),
        samplers=samplers(dx, domain=domain_for(case_name)),
        transport=fvm.TransportConfig(
            density=1.0,
            kinematic_viscosity=KINEMATIC_VISCOSITY,
        ),
        turbulence=fvm.TurbulenceConfig.none(),
        boundaries=[
            fvm.BoundaryConfig.inlet("inlet", FREESTREAM_VELOCITY),
            fvm.BoundaryConfig.outlet("outlet", kinematic_pressure=0.0),
            fvm.BoundaryConfig.slip("ymin"),
            fvm.BoundaryConfig.slip("ymax"),
            fvm.BoundaryConfig.slip("zmin"),
            fvm.BoundaryConfig.slip("zmax"),
            fvm.BoundaryConfig.wall("cylinder"),
        ],
        initial_velocity=FREESTREAM_VELOCITY,
        initial_kinematic_pressure=0.0,
    )


def main() -> None:
    arguments = parse_arguments()
    solution_dir = CASE_DIR / "solution" / arguments.case_name
    samples_dir = CASE_DIR / "samples" / arguments.case_name
    mesh_file = solution_dir / "mesh.npz"
    mesh_source: str | Path | msh.CartesianMesher
    if mesh_file.is_file():
        manifest_file = solution_dir / "mesh_manifest.json"
        if not manifest_file.is_file():
            raise RuntimeError(
                f"Saved mesh {mesh_file} has no mesh_manifest.json; refusing stale reuse"
            )
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        source_hash = hashlib.sha256(CYLINDER_STL.read_bytes()).hexdigest()
        if (
            manifest.get("case") != arguments.case_name
            or float(manifest.get("requested_dx", -1.0)) != float(arguments.dx)
            or manifest.get("source_stl_sha256") != source_hash
        ):
            raise RuntimeError(
                f"Saved mesh manifest {manifest_file} does not match case={arguments.case_name!r} "
                f"dx={arguments.dx!r} and the frozen source STL"
            )
        mesh_source = mesh_file
    else:
        mesh_source = grid_mesh(arguments.dx, domain=domain_for(arguments.case_name))
    solver = fvm.create_fvm_solver(
        solver_setup(arguments.case_name, arguments.dx),
        case_dir=CASE_DIR,
        solution_dir=solution_dir,
        samples_dir=samples_dir,
        mesh=mesh_source,
    )
    try:
        solver.write_run_manifest()
        solver.run()
    finally:
        solver.close()


if __name__ == "__main__":
    main()
