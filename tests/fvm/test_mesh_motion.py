"""Rigid mesh motion: exact kinematics, free-stream preservation, Galilean
invariance, restart and partitioned execution of the moving FVM mesh."""

from __future__ import annotations

from importlib.util import find_spec
import math
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

from openonda import fvm
from openonda.fvm.motion.kinematics import RigidPose, advance_pose, axis_rotation_matrix


def _advance(kinematics, times):
    """Compose the kinematic increments into the pose of the placed mesh."""
    pose = RigidPose()
    for start, end in zip(times[:-1], times[1:], strict=True):
        pose = advance_pose(kinematics, pose, start, end - start)
    return pose


_IRREGULAR_TIMES = np.cumsum(np.r_[0.0, 0.013 + 0.011 * np.abs(np.sin(np.arange(120)))])


@pytest.mark.parametrize(
    "kinematics, displacement, axis, angle",
    [
        (
            fvm.TranslatingMotion((0.4, -0.2, 0.1)),
            lambda t: np.array([0.4, -0.2, 0.1]) * t,
            None,
            None,
        ),
        (
            fvm.HeavingMotion(0.3, 0.7, phase=0.4, direction=(0, 2, 0)),
            lambda t: 0.3 * math.sin(2 * math.pi * 0.7 * t + 0.4) * np.array([0, 1, 0]),
            None,
            None,
        ),
        (
            fvm.LinearPeriodicMotion(0.2, 1.3, (1, 1, 0)),
            lambda t: 0.2 * math.sin(2 * math.pi * 1.3 * t) * np.array([1, 1, 0]) / math.sqrt(2),
            None,
            None,
        ),
        (
            fvm.SmoothRampMotion((1.0, 0.5, 0.0), 0.6, start_time=0.2),
            None,
            None,
            None,
        ),
        (
            fvm.RotatingMotion(
                1.5, (0, 0, 1), rotation_centre=(0.3, -0.1, 0), acceleration_time=0.5
            ),
            None,
            (0, 0, 1),
            lambda t: (
                1.5
                * (
                    0.5 * (min(t, 0.5) - 0.5 / math.pi * math.sin(math.pi * min(t, 0.5) / 0.5))
                    + max(t - 0.5, 0.0)
                )
            ),
        ),
        (
            fvm.PitchingMotion(
                12.0, 0.8, phase=0.3, rotation_axis=(0, 0, 1), rotation_centre=(0.25, 0, 0)
            ),
            None,
            (0, 0, 1),
            lambda t: math.radians(12.0) * math.sin(2 * math.pi * 0.8 * t + 0.3),
        ),
    ],
)
def test_closed_form_kinematics_move_the_mesh_exactly(kinematics, displacement, axis, angle):
    """Exact motions compose to their closed-form pose for any step sequence."""
    start, end = _IRREGULAR_TIMES[0], _IRREGULAR_TIMES[-1]
    pose = _advance(kinematics, _IRREGULAR_TIMES)
    point = np.array([[1.2, -0.7, 0.4]])
    centre = np.asarray(kinematics.rotation_centre)
    if displacement is None and angle is None:
        displacement = kinematics.displacement
    expected = point.copy()
    if angle is not None:
        rotation = axis_rotation_matrix(axis, angle(end) - angle(start))
        expected = (point - centre) @ rotation.T + centre
    if displacement is not None:
        expected = expected + displacement(end) - displacement(start)
    np.testing.assert_allclose(pose.apply(point), expected, rtol=0.0, atol=1e-12)

    # The reported velocities are the time derivatives of the same motion.
    step = 1e-6
    for time in (0.31, 0.77, 1.4):
        rotation, shift = kinematics.increment(time - step, 2 * step)
        np.testing.assert_allclose(shift / (2 * step), kinematics.get_velocity(time), atol=1e-7)
        rotation_rate = np.array([rotation[2, 1], rotation[0, 2], rotation[1, 0]]) / (2 * step)
        np.testing.assert_allclose(rotation_rate, kinematics.get_angular_velocity(time), atol=1e-7)


def test_velocity_driven_kinematics_converge_at_second_order():
    """Manoeuvre and composite motions integrate their velocities to O(dt^2)."""
    heave = fvm.HeavingMotion(0.3, 0.9, direction=(0, 1, 0))
    pitch = fvm.PitchingMotion(20.0, 0.9, phase=0.5, rotation_axis=(0, 0, 1))
    point = np.array([[1.0, 0.2, 0.0]])

    def error(kinematics, reference, n_steps):
        times = np.linspace(0.0, 1.1, n_steps + 1)
        return np.abs(_advance(kinematics, times).apply(point) - reference).max()

    maneuver = fvm.ManeuverMotion(velocity_function=heave.get_velocity)
    exact = point + heave.displacement(1.1) - heave.displacement(0.0)
    errors = [error(maneuver, exact, n) for n in (40, 80)]
    assert errors[0] / errors[1] == pytest.approx(4.0, rel=0.1)

    composite = fvm.CompositeMotion((heave, pitch))
    reference = _advance(composite, np.linspace(0.0, 1.1, 20001)).apply(point)
    errors = [error(composite, reference, n) for n in (40, 80)]
    assert errors[0] / errors[1] == pytest.approx(4.0, rel=0.15)


def _box_case(directory, boundaries, motion, *, velocity=(1.0, 0.0, 0.0), end_time=0.2):
    x = np.linspace(-1.0, 1.0, 11)
    mesh = fvm.mesher.box_mesh_3d(x, x, np.array([0.0, 0.2]), empty_spanwise=True)
    return fvm.FVMCase(
        name="moving_box",
        directory=directory,
        mesh=mesh,
        numerics=fvm.Numerics(
            transport=fvm.TransportConfig(density=1.0, kinematic_viscosity=0.01),
            schemes=fvm.DiscretizationConfig(time_scheme="backward"),
        ),
        boundaries=tuple(boundaries)
        + (fvm.BoundaryConfig.empty("zmin"), fvm.BoundaryConfig.empty("zmax")),
        initial_conditions=fvm.InitialFields(velocity=velocity),
        run=fvm.RunPlan(end_time=end_time, time_step_size=0.01, final_output=False),
        mesh_motion=motion,
    )


@pytest.mark.parametrize(
    "motion",
    [
        fvm.TranslatingMotion((0.3, -0.2, 0.0)),
        fvm.RotatingMotion(1.0, (0, 0, 1), rotation_centre=(0.1, 0.0, 0.1)),
        fvm.CompositeMotion(
            (
                fvm.HeavingMotion(0.2, 1.0, direction=(0, 1, 0)),
                fvm.PitchingMotion(10.0, 1.0, rotation_axis=(0, 0, 1)),
            )
        ),
    ],
)
def test_uniform_flow_is_preserved_on_a_moving_mesh(tmp_path, motion):
    """A rigid mesh sweeps no volume, so a uniform stream stays exactly uniform."""
    stream = (1.0, 0.0, 0.0)
    boundaries = [
        fvm.BoundaryConfig.freestream(name, stream) for name in ("inlet", "outlet", "ymin", "ymax")
    ]
    with fvm.FVMSolver(_box_case(tmp_path, boundaries, fvm.MeshMotion(motion))) as solver:
        solver.run(start_from="initial")
        n_cells = solver.mesh_data["n_cells"]
        assert solver.mesh_motion.pose.translation.any() or not np.allclose(
            solver.mesh_motion.pose.rotation, np.eye(3)
        )
        np.testing.assert_allclose(
            solver.velocity[:n_cells], np.tile(stream, (n_cells, 1)), atol=1e-12
        )
        np.testing.assert_allclose(solver.kinematic_pressure[:n_cells], 0.0, atol=1e-12)


def test_slip_planes_moving_along_their_normal_carry_the_stream(tmp_path):
    """Slip planes moving through themselves impose the mesh normal velocity."""
    stream = (1.0, 0.3, 0.0)
    boundaries = [
        fvm.BoundaryConfig.inlet("inlet", list(stream)),
        fvm.BoundaryConfig.outlet("outlet"),
        fvm.BoundaryConfig.slip("ymin"),
        fvm.BoundaryConfig.slip("ymax"),
    ]
    motion = fvm.MeshMotion(fvm.TranslatingMotion((0.0, 0.3, 0.0)))
    with fvm.FVMSolver(_box_case(tmp_path, boundaries, motion, velocity=stream)) as solver:
        solver.run(start_from="initial")
        n_cells = solver.mesh_data["n_cells"]
        np.testing.assert_allclose(
            solver.velocity[:n_cells], np.tile(stream, (n_cells, 1)), atol=1e-11
        )


def _square_cylinder_case(directory, stream, motion):
    x = np.linspace(-3.0, 6.0, 46)
    y = np.linspace(-3.0, 3.0, 31)
    mesh = fvm.mesher.box_mesh_3d(
        x, y, np.array([0.0, 0.2]), hole_box=(-0.6, 0.6, -0.6, 0.6, 0.0, 0.2), empty_spanwise=True
    )
    return fvm.FVMCase(
        name="square_cylinder",
        directory=directory,
        mesh=mesh,
        numerics=fvm.Numerics(
            transport=fvm.TransportConfig(density=1.0, kinematic_viscosity=0.012),
            schemes=fvm.DiscretizationConfig(time_scheme="backward"),
        ),
        boundaries=tuple(
            fvm.BoundaryConfig.freestream(name, stream)
            for name in ("inlet", "outlet", "ymin", "ymax")
        )
        + (
            fvm.BoundaryConfig.empty("zmin"),
            fvm.BoundaryConfig.empty("zmax"),
            fvm.BoundaryConfig.wall("cube"),
        ),
        initial_conditions=fvm.InitialFields(velocity=stream),
        run=fvm.RunPlan(end_time=10.0, time_step_size=0.02),
        mesh_motion=motion,
    )


def test_body_translating_through_still_fluid_is_galilean_invariant(tmp_path):
    """A cylinder moving at -V through still fluid equals a fixed cylinder in a stream V."""
    velocity = np.array([-1.0, 0.3, 0.0])
    fixed = fvm.FVMSolver(_square_cylinder_case(tmp_path / "fixed", tuple(-velocity), None))
    moving = fvm.FVMSolver(
        _square_cylinder_case(
            tmp_path / "moving", (0.0, 0.0, 0.0), fvm.MeshMotion(fvm.TranslatingMotion(velocity))
        )
    )
    fixed.auto_write = moving.auto_write = False
    # Start both frames from the same divergence-free state: an impulsive
    # start violates continuity, and the conservative convection term then
    # depends on the frame.
    fixed.start_from("initial")
    moving.start_from("initial")
    for _ in range(2):
        fixed.advance()
    moving.time, moving.step = fixed.time, fixed.step
    moving._n_committed_time_steps = fixed._n_committed_time_steps
    moving.mesh_motion.restore(fixed.time, np.eye(3), velocity * fixed.time)
    for name in ("velocity", "velocity_old", "velocity_older"):
        getattr(moving, name)[:] = getattr(fixed, name) + velocity
    for name in ("volumetric_face_flux", "volumetric_face_flux_old", "volumetric_face_flux_older"):
        getattr(moving, name)[:] = getattr(fixed, name) + moving.mesh_motion.mesh_face_flux
    moving.kinematic_pressure[:] = fixed.kinematic_pressure
    moving.mesh_motion.update_boundaries(moving)

    n_cells = fixed.mesh_data["n_cells"]
    for _ in range(20):
        fixed.advance()
        moving.advance()
    np.testing.assert_allclose(
        moving.velocity[:n_cells] - velocity, fixed.velocity[:n_cells], atol=1e-9
    )
    np.testing.assert_allclose(moving.kinematic_pressure, fixed.kinematic_pressure, atol=1e-9)
    assert np.abs(fixed.velocity[:n_cells] + velocity).max() > 0.3
    fixed.close()
    moving.close()


def _restart_case(directory, end_time):
    x = np.linspace(-3.0, 6.0, 37)
    y = np.linspace(-3.0, 3.0, 25)
    mesh = fvm.mesher.box_mesh_3d(
        x, y, np.array([0.0, 0.2]), hole_box=(-0.5, 0.5, -0.5, 0.5, 0.0, 0.2), empty_spanwise=True
    )
    motion = fvm.MeshMotion(
        fvm.CompositeMotion(
            (
                fvm.HeavingMotion(0.3, 0.5, direction=(0, 1, 0)),
                fvm.PitchingMotion(15.0, 0.5, rotation_axis=(0, 0, 1)),
            )
        ),
        rotation_degrees=(0.0, 0.0, 5.0),
    )
    return fvm.FVMCase(
        name="moving_restart",
        directory=directory,
        mesh=mesh,
        numerics=fvm.Numerics(
            transport=fvm.TransportConfig(density=1.0, kinematic_viscosity=0.01),
            schemes=fvm.DiscretizationConfig(time_scheme="backward"),
        ),
        boundaries=(
            fvm.BoundaryConfig.freestream("inlet", (1.0, 0.0, 0.0)),
            fvm.BoundaryConfig.freestream("outlet", (1.0, 0.0, 0.0)),
            fvm.BoundaryConfig.slip("ymin"),
            fvm.BoundaryConfig.slip("ymax"),
            fvm.BoundaryConfig.empty("zmin"),
            fvm.BoundaryConfig.empty("zmax"),
            fvm.BoundaryConfig.wall("cube"),
        ),
        initial_conditions=fvm.InitialFields(velocity=(1.0, 0.0, 0.0)),
        run=fvm.RunPlan(end_time=end_time, time_step_size=0.02, final_output=False),
        backup=fvm.BackupConfig(schedule=fvm.RunSchedule(every_time=0.2), write_at_end=True),
        mesh_motion=motion,
    )


def test_moving_mesh_restart_continues_the_same_trajectory(tmp_path):
    """A restart restores the mesh pose and continues bit for bit."""
    with fvm.FVMSolver(_restart_case(tmp_path / "continuous", 0.4)) as continuous:
        continuous.run(start_from="initial")
    with fvm.FVMSolver(_restart_case(tmp_path / "restarted", 0.2)) as first:
        first.run(start_from="initial")
    with fvm.FVMSolver(_restart_case(tmp_path / "restarted", 0.4)) as restarted:
        restarted.run(start_from="latest")
        for name in ("velocity", "kinematic_pressure", "volumetric_face_flux"):
            np.testing.assert_array_equal(getattr(restarted, name), getattr(continuous, name))
        np.testing.assert_array_equal(
            restarted.geo_data["cell_centre"], continuous.geo_data["cell_centre"]
        )


def test_partitioned_moving_mesh_matches_the_serial_run(tmp_path):
    """Each rank moves its own mesh partition; forces equal the serial run."""
    if find_spec("mpi4py") is None or find_spec("petsc4py") is None:
        pytest.skip("MPI and PETSc required")
    if not (Path(sys.executable).with_name("mpiexec").is_file() or shutil.which("mpiexec")):
        pytest.skip("mpiexec required")
    script = tmp_path / "setup.py"
    script.write_text(
        """
import sys
import numpy as np
from openonda import fvm

cores = int(sys.argv[1])
x = np.linspace(-3.0, 6.0, 37)
y = np.linspace(-3.0, 3.0, 25)
mesh = fvm.mesher.box_mesh_3d(
    x, y, np.array([0.0, 0.2]), hole_box=(-0.5, 0.5, -0.5, 0.5, 0.0, 0.2), empty_spanwise=True
)
motion = fvm.MeshMotion(fvm.CompositeMotion((
    fvm.HeavingMotion(0.3, 0.5, direction=(0, 1, 0)),
    fvm.PitchingMotion(15.0, 0.5, rotation_axis=(0, 0, 1)),
)))
case = fvm.FVMCase(
    name="moving", directory=f"run_{cores}", mesh=mesh, cores=cores,
    numerics=fvm.Numerics(
        transport=fvm.TransportConfig(density=1.0, kinematic_viscosity=0.01),
        # Converge both runs far below the comparison tolerance so the check
        # measures the partitioning, not the linear-solver stopping point.
        linear=fvm.LinearSolverConfig(momentum_tolerance=1e-13, pressure_tolerance=1e-13),
    ),
    boundaries=(
        fvm.BoundaryConfig.freestream("inlet", (1.0, 0.0, 0.0)),
        fvm.BoundaryConfig.freestream("outlet", (1.0, 0.0, 0.0)),
        fvm.BoundaryConfig.slip("ymin"),
        fvm.BoundaryConfig.slip("ymax"),
        fvm.BoundaryConfig.empty("zmin"),
        fvm.BoundaryConfig.empty("zmax"),
        fvm.BoundaryConfig.wall("cube"),
    ),
    initial_conditions=fvm.InitialFields(velocity=(1.0, 0.0, 0.0)),
    run=fvm.RunPlan(end_time=0.3, time_step_size=0.02),
    samplers=fvm.Samplers((fvm.ForceSampler(
        ["cube"], reference_velocity=1.0, reference_area=0.2, reference_length=1.0,
        moment_centre=(0.0, 0.0, 0.1), schedule=fvm.RunSchedule(every_n_steps=1),
    ),)),
    mesh_motion=motion,
)
with fvm.FVMSolver(case) as solver:
    solver.run(start_from="initial")
"""
    )
    histories = []
    for cores in (1, 2):
        subprocess.run(
            [sys.executable, str(script), str(cores)], cwd=tmp_path, check=True, timeout=600
        )
        histories.append(
            np.genfromtxt(
                tmp_path / f"run_{cores}" / "samples" / "forces_history.csv",
                delimiter=",",
                names=True,
                usecols=("total_force_x", "total_force_y", "moment_z"),
            )
        )
    serial, partitioned = histories
    for name in ("total_force_x", "total_force_y", "moment_z"):
        np.testing.assert_allclose(partitioned[name], serial[name], rtol=1e-7, atol=1e-9)
