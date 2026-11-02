"""Rigid mesh motion: exact kinematics, free-stream preservation, Galilean
invariance, the elastically mounted body, restart and partitioned execution
of the moving FVM mesh."""

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
from openonda.core.kinematics import RigidPose, advance_pose, axis_rotation_matrix
from openonda.fvm.motion.elastic import MountedBody


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
            time_step_size=0.01,
            schemes=fvm.DiscretizationConfig(time_scheme="backward"),
        ),
        boundaries=tuple(boundaries)
        + (fvm.BoundaryConfig.empty("zmin"), fvm.BoundaryConfig.empty("zmax")),
        initial_conditions=fvm.InitialFields(velocity=velocity),
        run=fvm.RunPlan(end_time=end_time, final_output=False),
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
            time_step_size=0.02,
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
        run=fvm.RunPlan(end_time=10.0),
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


_PRESCRIBED_RESTART_MOTION = fvm.CompositeMotion(
    (
        fvm.HeavingMotion(0.3, 0.5, direction=(0, 1, 0)),
        fvm.PitchingMotion(15.0, 0.5, rotation_axis=(0, 0, 1)),
    )
)
_ELASTIC_RESTART_MOTION = fvm.ElasticMount(
    mass=0.5,
    stiffness=(20.0, 30.0),
    damping=0.2,
    patch_names=("cube",),
    initial_velocity=(0.0, 0.4),
)


def _restart_case(directory, end_time, kinematics=_PRESCRIBED_RESTART_MOTION):
    x = np.linspace(-3.0, 6.0, 37)
    y = np.linspace(-3.0, 3.0, 25)
    mesh = fvm.mesher.box_mesh_3d(
        x, y, np.array([0.0, 0.2]), hole_box=(-0.5, 0.5, -0.5, 0.5, 0.0, 0.2), empty_spanwise=True
    )
    motion = fvm.MeshMotion(kinematics, rotation_degrees=(0.0, 0.0, 5.0))
    return fvm.FVMCase(
        name="moving_restart",
        directory=directory,
        mesh=mesh,
        numerics=fvm.Numerics(
            transport=fvm.TransportConfig(density=1.0, kinematic_viscosity=0.01),
            time_step_size=0.02,
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
        run=fvm.RunPlan(end_time=end_time, final_output=False),
        backup=fvm.Backup(schedule=fvm.EveryTime(0.2), write_at_end=True),
        mesh_motion=motion,
    )


@pytest.mark.parametrize(
    "kinematics",
    [_PRESCRIBED_RESTART_MOTION, _ELASTIC_RESTART_MOTION],
    ids=["prescribed", "elastic"],
)
def test_moving_mesh_restart_continues_the_same_trajectory(tmp_path, kinematics):
    """A restart restores the mesh pose, and the structural state of a mounted
    body, and continues bit for bit."""
    with fvm.FVMSolver(_restart_case(tmp_path / "continuous", 0.4, kinematics)) as continuous:
        continuous.run(start_from="initial")
        assert np.abs(continuous.mesh_motion.pose.translation).max() > 0.01
    with fvm.FVMSolver(_restart_case(tmp_path / "restarted", 0.2, kinematics)) as first:
        first.run(start_from="initial")
    with fvm.FVMSolver(_restart_case(tmp_path / "restarted", 0.4, kinematics)) as restarted:
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
        time_step_size=0.02,
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
    run=fvm.RunPlan(end_time=0.3),
    samplers=fvm.Samplers((fvm.ForceSampler(
        ["cube"], reference_velocity=1.0, reference_area=0.2, reference_length=1.0,
        moment_centre=(0.0, 0.0, 0.1), schedule=fvm.EverySteps(1),
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


def test_elastic_mount_response_is_the_exact_damped_oscillator():
    """Claim: the mount integrates ``m x'' + b x' + k x = F`` exactly for any
    interval, so its increments compose over any splitting of an interval."""
    mass, stiffness, damping, force = 2.0, 50.0, 1.2, 3.0
    mount = fvm.ElasticMount(
        mass=mass, stiffness=stiffness, damping=damping, directions=((0.0, 1.0, 0.0),)
    )
    body = MountedBody(mount, 0.0)
    body.accept(0.0, np.array([0.0, force, 0.0]))
    # Underdamped free response about the static deflection F / k
    omega = math.sqrt(stiffness / mass)
    zeta = damping / (2.0 * math.sqrt(stiffness * mass))
    damped = omega * math.sqrt(1.0 - zeta**2)
    static = force / stiffness
    for time in (0.0, 0.013, 0.37, 2.9):
        decay = math.exp(-zeta * omega * time)
        expected = static - static * decay * (
            math.cos(damped * time) + zeta * omega / damped * math.sin(damped * time)
        )
        np.testing.assert_allclose(body.increment(0.0, time)[1], [0.0, expected, 0.0], atol=1e-13)
    whole = body.increment(0.1, 0.9)[1]
    parts = sum(body.increment(0.1 + 0.09 * k, 0.09)[1] for k in range(10))
    np.testing.assert_allclose(parts, whole, atol=1e-13)
    assert body.get_angular_velocity(0.5).tolist() == [0.0, 0.0, 0.0]


def test_elastically_mounted_body_rings_at_its_natural_frequency_in_still_fluid(tmp_path):
    """Claim: a body on springs released in still fluid oscillates at the
    natural frequency lowered by its added mass, and the fluid damps it.

    The mesh is built with the spring extended by 0.2 m, so the body swings
    about the position 0.2 m below its start. A square section of side 1 m
    at mass ratio 50 has an added mass of about 1.5 rho a^2 (Blevins), which
    lowers the frequency by about 1.5 %; the viscous damping of the fluid is
    positive. The motion sampler records the response.
    """
    mass, stiffness = 50.0 * 0.2, 400.0 * 0.2  # per 0.2 m span; f_n = 1 / (2 pi) ... 0.45 Hz
    x = np.linspace(-6.0, 6.0, 49)
    mesh = fvm.mesher.box_mesh_3d(
        x, x, np.array([0.0, 0.2]), hole_box=(-0.5, 0.5, -0.5, 0.5, 0.0, 0.2), empty_spanwise=True
    )
    mount = fvm.ElasticMount(
        mass=mass,
        stiffness=stiffness,
        directions=((0.0, 1.0, 0.0),),
        patch_names=("cube",),
        initial_displacement=(0.2,),
    )
    period = 2.0 * math.pi * math.sqrt(mass / stiffness)
    case = fvm.FVMCase(
        name="ringing",
        directory=tmp_path,
        mesh=mesh,
        numerics=fvm.Numerics(
            transport=fvm.TransportConfig(density=1.0, kinematic_viscosity=0.01),
            time_step_size=period / 100.0,
            schemes=fvm.DiscretizationConfig(time_scheme="backward"),
            pimple=fvm.PimpleConfig(n_outer_correctors=3),
        ),
        boundaries=(
            fvm.BoundaryConfig.inlet("inlet", (0.0, 0.0, 0.0)),
            fvm.BoundaryConfig.inlet("outlet", (0.0, 0.0, 0.0)),
            fvm.BoundaryConfig.inlet("ymin", (0.0, 0.0, 0.0)),
            fvm.BoundaryConfig.inlet("ymax", (0.0, 0.0, 0.0)),
            fvm.BoundaryConfig.empty("zmin"),
            fvm.BoundaryConfig.empty("zmax"),
            fvm.BoundaryConfig.wall("cube"),
        ),
        initial_conditions=fvm.InitialFields(velocity=(0.0, 0.0, 0.0)),
        run=fvm.RunPlan(end_time=2.5 * period, final_output=False),
        samplers=fvm.Samplers((fvm.MotionSampler(schedule=fvm.EverySteps(1)),)),
        mesh_motion=fvm.MeshMotion(mount),
    )
    with fvm.FVMSolver(case) as solver:
        solver.auto_write = False
        solver.run(start_from="initial")
    history = np.genfromtxt(tmp_path / "samples/motion_history.csv", delimiter=",", names=True)
    time, displacement = history["time"], history["displacement_y"] + 0.2
    np.testing.assert_allclose(history["position_y"], history["displacement_y"], atol=1e-12)
    assert np.all(history["displacement_x"] == 0.0)
    assert displacement[0] == 0.2
    # Period from the downward zero crossings
    down = np.flatnonzero((displacement[:-1] > 0.0) & (displacement[1:] <= 0.0))
    crossings = time[down] - displacement[down] * np.diff(time)[down] / np.diff(displacement)[down]
    assert len(crossings) >= 2
    measured = np.diff(crossings).mean()
    assert 1.005 * period < measured < 1.05 * period, (measured, period)
    # The fluid takes energy out: the second maximum is lower than the first
    maxima = displacement[1:-1][
        (displacement[1:-1] > displacement[:-2]) & (displacement[1:-1] > displacement[2:])
    ]
    assert 0.5 * 0.2 < maxima[0] < 0.2
