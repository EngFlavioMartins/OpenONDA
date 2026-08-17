"""Fully meshed reference for the Re = 300 sphere benchmark.

Matched to the hybrid case by construction: same body, fluid, near-field
spacing, IBM, scheme and time step, all imported from ``_geometry``.  Only the
treatment outside the compact box differs -- mesh here, particles there.

Two things this reference does that the coupled layout must not:

* it gives itself a **real outlet**.  ``coupling_box_mesh`` merges all six sides
  into one patch, which is right when every face is a donor boundary and wrong
  for a reference: a ``fixedValue`` outlet clamps the exit to the freestream, so
  the wake cannot leave and the pressure has no Dirichlet anchor anywhere;
* it **stretches to a distant far field**, so it approximates the same unbounded
  problem the particle method solves.  A near, clamped boundary makes the two
  cases solve different problems, and no coupling accuracy can close that gap.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openonda.fvm import (  # noqa: E402
    BoundaryConfig,
    ExecutionConfig,
    FVMSetup,
    IBMForceSampler,
    ImmersedBody,
    LinearSolverConfig,
    LineSampler,
    OutputSetup,
    PimpleControl,
    SamplingSchedule,
    SchemesConfig,
    SurfaceSampler,
    TimeConfig,
    TransportConfig,
    TurbulenceConfig,
    coupling_box_mesh,
    setup_fvm_solver,
)

import _geometry as G  # noqa: E402

CASE_DIR = Path(__file__).resolve().parent
CORES = int(os.environ.get("OPENONDA_FVM_CORES", "1"))

FORCE_PERIOD = G.step_period("force interval", G.FORCE_INTERVAL, G.DT_FVM)
FIELD_PERIOD = G.step_period("diagnostic interval", G.DIAGNOSTIC_INTERVAL, G.DT_FVM)

MESH = coupling_box_mesh(
    G.REFERENCE_DOMAIN,
    G.SPACING,
    patch_name="farfield",
    nodes=G.reference_nodes(G.SPACING),
    separate_outer=("outlet",),
)
SPHERE = ImmersedBody.sphere(
    centre=[0.0, 0.0, 0.0],
    diameter=G.DIAMETER,
    h=G.SPACING,
    alpha=1.0,
    name="sphere",
)

FORCE_SCHEDULE = SamplingSchedule(every_n_steps=FORCE_PERIOD)
FIELD_SCHEDULE = SamplingSchedule(every_n_steps=FIELD_PERIOD)

SAMPLERS = (
    IBMForceSampler(
        ref_velocity=float(np.linalg.norm(G.U_INF)),
        ref_area=0.25 * np.pi * G.DIAMETER**2,
        schedule=FORCE_SCHEDULE,
    ),
    LineSampler(
        start=[G.FVM_BOX[0], 0.0, 0.0],
        end=[G.FVM_BOX[1], 0.0, 0.0],
        spacing=G.SAMPLE_SPACING,
        file_name="centerline",
        schedule=FIELD_SCHEDULE,
    ),
    LineSampler(
        start=[2.0, G.FVM_BOX[2], 0.0],
        end=[2.0, G.FVM_BOX[3], 0.0],
        spacing=G.SAMPLE_SPACING,
        file_name="section_x200",
        schedule=FIELD_SCHEDULE,
    ),
    SurfaceSampler(
        point=[0.0, 0.0, 0.0],
        normal=[0.0, 0.0, 1.0],
        bounds=[G.FVM_BOX[0], G.FVM_BOX[1], G.FVM_BOX[2], G.FVM_BOX[3]],
        spacing=G.SAMPLE_SPACING,
        file_name="slice_z0",
        schedule=FIELD_SCHEDULE,
    ),
)

SETUP = FVMSetup(
    case_name="reference_sphereFlow",
    cores=CORES,
    execution=ExecutionConfig(
        operator_backend="numba",
        linear_backend="petsc" if CORES > 1 else "scipy",
        parallel_mode="petsc_partitioned" if CORES > 1 else "serial",
    ),
    output=OutputSetup(compression="lz4", precision="float32", asynchronous=False, ghost_layers=0),
    time=TimeConfig(
        delta_t=G.DT_FVM,
        end_time=G.T_END,
        write_interval=10**9,
        write_interval_time=G.VOLUME_INTERVAL,
    ),
    schemes=SchemesConfig(
        convection_scheme="limitedLinear",
        gradient_scheme="gauss",
        time_scheme="backward",
    ),
    linear=LinearSolverConfig(
        pressure_solver="amg",
        pressure_tol=1e-6,
        pressure_rel_tol=0.01,
        momentum_tol=1e-6,
        momentum_rel_tol=0.1,
        momentum_maxiter=2000,
    ),
    pimple=PimpleControl(
        n_correctors=2,
        n_outer_correctors=2,
        alpha_u=0.7,
        alpha_p=0.3,
        ibm_forcing_loops=2,
    ),
    samplers=SAMPLERS,
    transport=TransportConfig(density=G.RHO, nu=G.NU),
    turbulence=TurbulenceConfig(),  # model="None": laminar, matching the hybrid
    boundaries=[
        BoundaryConfig(
            name="farfield",
            type_U="fixedValue",
            value_U=list(G.U_INF),
            type_p="fixedFluxPressure",
        ),
        BoundaryConfig.outlet("outlet", p=0.0),
    ],
    initial_U=list(G.U_INF),
    initial_p=0.0,
)


def main() -> None:
    solver = setup_fvm_solver(SETUP, case_dir=CASE_DIR, mesh=MESH)
    if solver.parallel.is_root:
        print("\n===== REFERENCE SPHERE Re=300 =====")
        print(
            f"  h/D={G.SPACING / G.DIAMETER:g}  dt={G.DT_FVM:g}  t_end={G.T_END:g}  "
            f"cells={solver.mesh_data['n_elements']}"
        )
    solver.set_immersed_bodies(SPHERE, h=G.SPACING)
    solver.write_vtk()
    while solver.flow_time < SETUP.time.end_time:
        solver.evolve()
    solver.close()


if __name__ == "__main__":
    main()
