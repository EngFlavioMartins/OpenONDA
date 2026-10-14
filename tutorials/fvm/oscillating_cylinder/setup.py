#!/usr/bin/env python3
"""Circular cylinder heaving across a uniform stream at Re = 100 (moving mesh).

The whole body-fitted O-grid moves with the cylinder, y(t) = A sin(2 pi f t).
Forcing near the natural shedding frequency locks the wake onto the motion.

Run with ``python setup.py``.
"""

from pathlib import Path

import openonda.fvm as fvm

from assets.mesh_cylinder import cylinder_o_grid

# Case definition
START_FROM = "latest"  # Resume the latest backup; ./allrun.sh cleans first.

CASE_NAME = "oscillating_cylinder"
DIAMETER = 1.0  # m
FREESTREAM_VELOCITY = 1.0  # m/s
DENSITY = 1.0  # kg/m^3
REYNOLDS_NUMBER = 100.0
FINAL_TIME = 120.0  # s

# Prescribed transverse motion
AMPLITUDE = 0.2 * DIAMETER  # heave amplitude [m]
NATURAL_STROUHAL_NUMBER = 0.165  # shedding of the fixed cylinder at Re = 100
FREQUENCY_RATIO = 1.0  # forcing / natural shedding frequency
FREQUENCY = FREQUENCY_RATIO * NATURAL_STROUHAL_NUMBER * FREESTREAM_VELOCITY / DIAMETER  # Hz

# Mesh: polar O-grid, one cell across the span
OUTER_DIAMETER = 50.0 * DIAMETER  # far-field circle [m]
N_CIRCUMFERENCE = 128  # cells around the cylinder
N_RADIAL = 96  # cells from the wall to the far field
WALL_SPACING = 0.005 * DIAMETER  # first radial cell [m]
DEPTH = 1.0  # span of the single cell [m]

# Time stepping and output
TIME_STEP_SIZE = 0.01  # [s], maximum Courant number about 0.6 relative to the moving cells
OUTPUT_INTERVAL_TIME = 3.0  # volume fields [s]
FORCE_INTERVAL_STEPS = 2  # force history


def create_fvm_case() -> fvm.FVMCase:
    """Build the moving-mesh FVM case of the heaving cylinder."""
    kinematic_viscosity = FREESTREAM_VELOCITY * DIAMETER / REYNOLDS_NUMBER
    snapshots = fvm.EveryTime(OUTPUT_INTERVAL_TIME)

    # Drag and lift coefficients per unit span; the moment centre moves with
    # the cylinder axis.
    forces = fvm.ForceSampler(
        patch_names=["cylinder"],
        reference_velocity=FREESTREAM_VELOCITY,
        reference_area=DIAMETER * DEPTH,
        reference_length=DIAMETER,
        moment_centre=[0.0, 0.0, 0.5 * DEPTH],
        schedule=fvm.EverySteps(FORCE_INTERVAL_STEPS),
    )

    return fvm.FVMCase(
        name=CASE_NAME,
        directory=Path(__file__).parent,
        numerics=fvm.Numerics(
            time_step_size=TIME_STEP_SIZE,
            transport=fvm.TransportConfig(density=DENSITY, kinematic_viscosity=kinematic_viscosity),
            schemes=fvm.DiscretizationConfig(time_scheme="backward"),
            execution=fvm.ComputeConfig(operator_backend="numba"),
        ),
        initial_conditions=fvm.InitialFields(velocity=(FREESTREAM_VELOCITY, 0.0, 0.0)),
        run=fvm.RunPlan(end_time=FINAL_TIME),
        output=fvm.Output(schedule=snapshots),
        samplers=fvm.Samplers((forces,)),
        backup=fvm.Backup(schedule=snapshots),
        mesh=cylinder_o_grid(
            DIAMETER, OUTER_DIAMETER, N_CIRCUMFERENCE, N_RADIAL, WALL_SPACING, DEPTH
        ),
        boundaries=(
            # The far-field circle moves with the cylinder; its values are
            # absolute-frame. Upstream half: the stream; downstream half: the
            # pressure datum, with the stream entering on reverse flow.
            fvm.BoundaryConfig.inlet("inlet", (FREESTREAM_VELOCITY, 0.0, 0.0)),
            fvm.BoundaryConfig(
                name="outlet",
                velocity_type="inlet_outlet",
                velocity_value=(FREESTREAM_VELOCITY, 0.0, 0.0),
                pressure_type="fixed_value",
            ),
            # A wall on a moving mesh is a no-slip wall moving with the mesh.
            fvm.BoundaryConfig.wall("cylinder"),
            fvm.BoundaryConfig.empty("front"),
            fvm.BoundaryConfig.empty("back"),
        ),
        mesh_motion=fvm.MeshMotion(
            fvm.HeavingMotion(AMPLITUDE, FREQUENCY, direction=(0.0, 1.0, 0.0))
        ),
    )


def main() -> None:
    with fvm.FVMSolver(create_fvm_case()) as solver:
        solver.run(start_from=START_FROM)


if __name__ == "__main__":
    main()
