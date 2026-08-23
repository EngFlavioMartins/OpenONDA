"""FVM reference for the laminar Re = 150 cylinder-shedding experiment.

The geometry, local resolution, numerics, and optional divergence-free seed
match the coupled case in the parent directory.

Usage:
    python reference_flow_setup.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

import openonda.fvm as fvm

CASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CASE_DIR.parents[0]))
from seed_perturbation import build_seed_velocity  # noqa: E402

SMOKE = os.environ.get("OPENONDA_SMOKE", "0") == "1"

# Physical problem
DIAMETER = 1.0
FREESTREAM_VELOCITY = (1.0, 0.0, 0.0)
DENSITY = 1.0
REYNOLDS = 150.0
FREESTREAM_SPEED = float(np.linalg.norm(FREESTREAM_VELOCITY))
KINEMATIC_VISCOSITY = FREESTREAM_SPEED * DIAMETER / REYNOLDS

INITIAL_VELOCITY = FREESTREAM_VELOCITY
# The unseeded reference needs 100 convective units to reach a saturated window.
END_TIME = 1.0 if SMOKE else 100.0

# FVM domain, mesh, and numerics
# Body, wake, and background resolutions match the coupled case.
FVM_CORES = int(os.environ.get("OPENONDA_FVM_CORES", "1"))
if SMOKE:
    FVM_CORES = 1
FVM_DOMAIN = (-5.0, 12.0, -6.0, 6.0, -8.0, 8.0)

# The uncapped cylinder removes tip effects from the Karman instability.
CYLINDER_Z_BOUNDS = (FVM_DOMAIN[4], FVM_DOMAIN[5])
# Force coefficients are normalized per unit span.
CYLINDER_LENGTH = FVM_DOMAIN[5] - FVM_DOMAIN[4]

FVM_CELL_SIZE = 0.5 if SMOKE else 0.25
FVM_WAKE_CELL_SIZE = 0.25 if SMOKE else 0.125
FVM_BODY_CELL_SIZE = 0.125 if SMOKE else 0.0625

BODY_BOX = (-0.65, 0.65, -0.65, 0.65, FVM_DOMAIN[4], FVM_DOMAIN[5])
WAKE_BOX = (-0.75, 3.0, -1.25, 1.25, -5.5, 5.5)

# Backward 2nd-order; CFL ~ 0.32 in the body region, 0.16 wake, 0.08 far field.
FVM_TIME_STEP_SIZE = 0.02
PIMPLE_N_CORRECTORS = 2
PIMPLE_N_OUTER_CORRECTORS = 1
PIMPLE_N_ORTHOGONAL_CORRECTORS = 0
IBM_FORCING_LOOPS = 2

if FVM_CORES > 1:
    # Direct-forcing IBM is not partition-aware: replicated PETSc path.
    FVM_EXECUTION = fvm.ComputeConfig(
        operator_backend="numba",
        linear_backend="petsc",
        parallel_mode="petsc_replicated",
    )
else:
    FVM_EXECUTION = fvm.ComputeConfig(operator_backend="numba")

FVM_VOLUME_INTERVAL_TIME = 2.0

# Output and diagnostics cadence
FORCE_INTERVAL_TIME = 0.05
LINE_INTERVAL_TIME = 0.5
SLICE_INTERVAL_TIME = 1.0
SAMPLE_SPACING = 0.25 if SMOKE else 0.0625
PROBE_X = 1.5
TRANSVERSE_HALF = 1.5
SPAN_HALF = 5.5
MIDSPAN_SLICE_BOUNDS = (FVM_DOMAIN[0], FVM_DOMAIN[1], FVM_DOMAIN[2], FVM_DOMAIN[3])

SEED_AMPLITUDE = float((os.environ.get("OPENONDA_SEED_AMPLITUDE") or "0").strip())

# Case objects
CYLINDER = fvm.ImmersedBody.extruded_cylinder_z(
    centre=[0.0, 0.0, 0.0],
    diameter=DIAMETER,
    z_bounds=CYLINDER_Z_BOUNDS,
    grid_spacing=FVM_BODY_CELL_SIZE,
    name="cylinder",
    caps=False,
)

FVM_MESH = fvm.AdaptiveCartesianMesher(
    domain=FVM_DOMAIN,
    max_cell_size=FVM_CELL_SIZE,
    refinements=(
        fvm.BoxRefinement(BODY_BOX, FVM_BODY_CELL_SIZE, "bodyBox"),
        fvm.BoxRefinement(WAKE_BOX, FVM_WAKE_CELL_SIZE, "wakeBox"),
    ),
)

FVM_SAMPLERS = (
    fvm.IBMForceSampler(
        reference_velocity=FREESTREAM_SPEED,
        reference_area=DIAMETER * CYLINDER_LENGTH,
        schedule=fvm.SamplingSchedule(every_time=FORCE_INTERVAL_TIME),
    ),
    fvm.LineSampler(
        start=[PROBE_X, 0.0, 0.0],
        end=[PROBE_X, 0.0, 0.0],
        n_points=1,
        file_name="midspan_probe",
        schedule=fvm.SamplingSchedule(every_time=FORCE_INTERVAL_TIME),
    ),
    fvm.LineSampler(
        start=[PROBE_X, -TRANSVERSE_HALF, 0.0],
        end=[PROBE_X, TRANSVERSE_HALF, 0.0],
        spacing=SAMPLE_SPACING,
        file_name="transverse_line",
        schedule=fvm.SamplingSchedule(every_time=LINE_INTERVAL_TIME),
    ),
    fvm.LineSampler(
        start=[PROBE_X, 0.0, -SPAN_HALF],
        end=[PROBE_X, 0.0, SPAN_HALF],
        spacing=SAMPLE_SPACING,
        file_name="spanwise_line",
        schedule=fvm.SamplingSchedule(every_time=LINE_INTERVAL_TIME),
    ),
    fvm.SurfaceSampler(
        point=[0.0, 0.0, 0.0],
        normal=[0.0, 0.0, 1.0],
        bounds=MIDSPAN_SLICE_BOUNDS,
        spacing=SAMPLE_SPACING,
        file_name="slice_z0",
        schedule=fvm.SamplingSchedule(every_time=SLICE_INTERVAL_TIME),
    ),
)

FVM_SETUP = fvm.FVMSetup(
    case_name="reference_flow",
    cores=FVM_CORES,
    execution=FVM_EXECUTION,
    output=fvm.OutputConfig(
        format="vtk_xml",
        data_location="cell",
        encoding="appended",
        compression="lz4",
        precision="float32",
        asynchronous=True,
        ghost_layers=0,
    ),
    time=fvm.TimeConfig(
        time_step_size=FVM_TIME_STEP_SIZE,
        start_time=0.0,
        end_time=END_TIME,
        output_interval_steps=10**9,
        output_interval_time=FVM_VOLUME_INTERVAL_TIME,
        adjust_time_step=False,
    ),
    schemes=fvm.DiscretizationConfig(
        convection_scheme="linearUpwind",
        gradient_scheme="gauss",
        time_scheme="backward",
    ),
    linear=fvm.LinearSolverConfig(
        linear_solver="bicgstab",
        pressure_solver="amg",
        pressure_tolerance=1e-6,
        pressure_relative_tolerance=0.01,
        pressure_final_relative_tolerance=0.0,
        momentum_tolerance=1e-4,
        momentum_relative_tolerance=0.1,
        momentum_final_relative_tolerance=0.0,
        momentum_max_iterations=2000,
        ilu_drop_tolerance=1e-4,
        ilu_fill_factor=10.0,
        ilu_reuse_tolerance=0.05,
    ),
    pimple=fvm.PimpleControl(
        n_correctors=PIMPLE_N_CORRECTORS,
        n_outer_correctors=PIMPLE_N_OUTER_CORRECTORS,
        n_orthogonal_correctors=PIMPLE_N_ORTHOGONAL_CORRECTORS,
        velocity_relaxation=0.7,
        pressure_relaxation=0.3,
        ibm_forcing_loops=IBM_FORCING_LOOPS,
    ),
    samplers=FVM_SAMPLERS,
    transport=fvm.TransportConfig(density=DENSITY, kinematic_viscosity=KINEMATIC_VISCOSITY),
    turbulence=fvm.TurbulenceConfig.none(),
    boundaries=[
        fvm.BoundaryConfig.inlet("inlet", list(FREESTREAM_VELOCITY)),
        fvm.BoundaryConfig.outlet("outlet", kinematic_pressure=0.0),
        fvm.BoundaryConfig.slip("ymin"),
        fvm.BoundaryConfig.slip("ymax"),
        fvm.BoundaryConfig.slip("zmin"),
        fvm.BoundaryConfig.slip("zmax"),
    ],
    initial_velocity=list(INITIAL_VELOCITY),
    initial_kinematic_pressure=0.0,
)


def _apply_seed(fvm_solver) -> None:
    """Impose the same controlled, divergence-free perturbation as the hybrid."""
    n_cells = fvm_solver.mesh_data["n_cells"]
    centroids = np.asarray(fvm_solver.geo_data["cell_centre"][:n_cells], dtype=np.float64)
    if centroids.shape[0] != n_cells:
        raise RuntimeError("Seed requires the full cell-centroid array on every rank")
    velocity = build_seed_velocity(
        centroids,
        base_velocity=INITIAL_VELOCITY,
        epsilon=SEED_AMPLITUDE,
        freestream_speed=FREESTREAM_SPEED,
        diameter=DIAMETER,
    )
    fvm_solver.set_initial_velocity(velocity)
    peak = float(np.linalg.norm(velocity - np.asarray(INITIAL_VELOCITY), axis=1).max())
    print(
        f"  Applied controlled seed: eps={SEED_AMPLITUDE:.3e}, "
        f"max|u'|/Uinf={peak / FREESTREAM_SPEED:.3e}"
    )


def main() -> None:
    print("\n===== SIMULATION (FVM reference) =====")
    print(
        f"  Re={REYNOLDS}, infinite cylinder D={DIAMETER}, "
        f"dt={FVM_TIME_STEP_SIZE}s, body/wake cell size={FVM_BODY_CELL_SIZE}/{FVM_WAKE_CELL_SIZE}, "
        f"seed={SEED_AMPLITUDE:g}"
    )
    fvm_solver = fvm.create_fvm_solver(FVM_SETUP, case_dir=CASE_DIR, mesh=FVM_MESH)
    fvm_solver.set_immersed_bodies(CYLINDER, grid_spacing=FVM_BODY_CELL_SIZE)
    if SEED_AMPLITUDE > 0.0:
        _apply_seed(fvm_solver)
    fvm_solver.write_vtk()

    # Optional step cap (OPENONDA_MAX_STEPS) for bounded memory/run probes.
    max_n_steps = int(os.environ.get("OPENONDA_MAX_STEPS", "0"))
    step = 0
    while fvm_solver.time < FVM_SETUP.time.end_time:
        if max_n_steps > 0 and step >= max_n_steps:
            print(f"  [probe] stopped after {step} steps (OPENONDA_MAX_STEPS={max_n_steps})")
            break
        fvm_solver.advance()
        step += 1

    fvm_solver.close()
    print("\n===== DONE =====")
    print("Reference simulation completed. Run ../allvalidate.sh for the instability analysis.")


if __name__ == "__main__":
    main()
