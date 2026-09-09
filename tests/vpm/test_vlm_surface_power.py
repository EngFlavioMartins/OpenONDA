"""Native rotor power, surface loading, and movable checkpoint identities."""

from pathlib import Path

import numpy as np
import pytest
import taichi as ti

import openonda.vpm as vpm
from source.solvers.vpm.boundary_elements.vlm.solver.vlm_solver import VLMSolver
from tutorials.vpm.flat_plate.assets.generate_surface import create_flat_plate, save_surface


@pytest.fixture(autouse=True)
def runtime():
    ti.reset()
    ti.init(arch=ti.cpu, default_fp=ti.f64, offline_cache=False, cpu_max_num_threads=2)
    yield
    ti.reset()


def test_counter_rotating_surfaces_keep_power_when_global_torque_cancels():
    plate = create_flat_plate(
        chord=1, span=2, angle_of_attack_degrees=0, n_chordwise_panels=1, n_spanwise_panels=2
    )
    centers = [np.array([-3.0, 0, 0]), np.array([3.0, 0, 0])]
    solver = VLMSolver(
        vpm.VLMSetup(
            surfaces=tuple(
                vpm.VLMSurfaceSetup(
                    plate,
                    name=f"rotor_{i}",
                    translation=tuple(center),
                    kinematics=vpm.RotatingVLM(
                        angular_speed=(-1) ** i * 2, axis=[0, 0, 1], rotation_centre=center
                    ),
                )
                for i, center in enumerate(centers)
            ),
            dtype="f64",
        )
    )
    solver.generate_mesh()
    points = solver.lattice.bound_vortex_midpoint.to_numpy()
    forces = np.zeros_like(points)
    powers = []
    for i, center in enumerate(centers):
        indices = slice(i * 4, (i + 1) * 4)
        velocity = np.cross([0, 0, (-1) ** i * 2], points[indices] - center)
        forces[indices] = -velocity
        powers.append(float(np.sum(forces[indices] * velocity)))
    solver.lattice.panel_force.from_numpy(forces)
    solver._solved = True
    values = solver.compute_per_surface_forces(solver.density, np.array([1.0, 0.0, 0.0]))
    np.testing.assert_allclose([values[f"rotor_{i}"]["rotational_power"] for i in range(2)], powers)
    assert values["rotor_0"]["moment_z"] == pytest.approx(-values["rotor_1"]["moment_z"])
    assert sum(row["rotational_power"] for row in values.values()) < 0
    for i, center in enumerate(centers):
        np.testing.assert_allclose(
            [values[f"rotor_{i}"][f"centroid_{axis}"] for axis in "xyz"],
            center + [0.5, 0.0, 0.0],
            atol=1e-14,
        )


def test_checkpoint_identity_allows_moved_geometry_but_rejects_changed_rotation(tmp_path):
    plate = create_flat_plate(
        chord=1, span=2, angle_of_attack_degrees=5, n_chordwise_panels=1, n_spanwise_panels=2
    )
    paths = [tmp_path / "first.json", tmp_path / "moved/first.json"]
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        save_surface(plate, str(path))

    def identity(path: Path, speed):
        solver = VLMSolver(
            vpm.VLMSetup(
                surfaces=(
                    vpm.VLMSurfaceSetup(
                        str(path),
                        kinematics=vpm.RotatingVLM(angular_speed=speed, axis=[0, 0, 1]),
                    ),
                ),
                dtype="f64",
            )
        )
        solver.generate_mesh()
        return solver._restart_identity

    first = identity(paths[0], 2.0)
    assert first == identity(paths[1], 2.0)
    assert first != identity(paths[1], 3.0)


def test_rotor_ramp_angle_and_geometry_match_the_analytic_integral():
    motion = vpm.RotatingVLM(angular_speed=8.0, axis=[1, 0, 0], acceleration_time=0.2)
    assert np.linalg.norm(motion.get_angular_velocity(0.0)) == 0.0
    assert np.linalg.norm(motion.get_angular_velocity(0.1)) == pytest.approx(4.0)
    assert motion.rotation_angle(0.2) == pytest.approx(0.8)
    assert motion.rotation_angle(0.3) == pytest.approx(1.6)
    plate = create_flat_plate(
        chord=1, span=2, angle_of_attack_degrees=0, n_chordwise_panels=1, n_spanwise_panels=2
    )
    solver = VLMSolver(
        vpm.VLMSetup(surfaces=(vpm.VLMSurfaceSetup(plate, kinematics=motion),), dtype="f64")
    )
    solver.generate_mesh()
    initial = solver.lattice.get_collocation_points().copy()
    for step in range(1, 31):
        solver.advance_time(0.01, step * 0.01)
    np.testing.assert_allclose(
        solver.lattice.get_collocation_points(),
        initial @ motion._rotation_matrix(1.6).T,
        atol=2e-12,
    )
