"""Analytical qualifications for the core VPM field evaluator."""

from __future__ import annotations

import contextlib
import io

import numpy as np
from scipy.special import erf

from openonda.vpm import (
    Backup,
    DirectInduction,
    Numerics,
    TreecodeInduction,
    ViscousConfig,
    VPMCase,
    VPMSolver,
)


def _solver(tmp_path, name: str, induction, *, precision: str = "f32"):
    case = VPMCase(
        directory=tmp_path / name,
        backup=Backup(0),
        numerics=Numerics(
            compute_device="CPU",
            precision=precision,
            induction=induction,
            viscous=ViscousConfig.inviscid(particle_spacing=0.15),
            max_n_particles=1_024,
            max_evaluation_points=1_024,
            verbose=False,
        ),
    )
    with contextlib.redirect_stdout(io.StringIO()):
        return VPMSolver(case)


def _add_cloud(
    solver: VPMSolver,
    position: np.ndarray,
    vortex_strength: np.ndarray,
    core_radius: np.ndarray,
) -> None:
    count = len(position)
    solver.add_vortex_particles(
        position=position,
        velocity=np.zeros((count, 3)),
        vortex_strength=vortex_strength,
        core_radius=core_radius,
        particle_volume=np.full(count, 0.15**3),
        kinematic_viscosity=np.zeros(count),
    )


def _single_gaussian_field(
    points: np.ndarray,
    source_position: np.ndarray,
    vortex_strength: np.ndarray,
    core_radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    displacement = points - source_position
    radius = np.linalg.norm(displacement, axis=1)
    density = radius / core_radius
    q = (erf(density) - 2.0 / np.sqrt(np.pi) * density * np.exp(-(density**2))) / (4.0 * np.pi)
    scale = q / radius**3
    scale_derivative = (
        density**2 * np.exp(-(density**2)) / (np.pi**1.5 * core_radius * radius**3)
        - 3.0 * q / radius**4
    )
    cross_matrix = np.array(
        [
            [0.0, -vortex_strength[2], vortex_strength[1]],
            [vortex_strength[2], 0.0, -vortex_strength[0]],
            [-vortex_strength[1], vortex_strength[0], 0.0],
        ]
    )
    velocity = scale[:, None] * (displacement @ cross_matrix.T)
    gradient = scale[:, None, None] * cross_matrix
    gradient += (
        scale_derivative[:, None, None]
        * velocity[:, :, None]
        / scale[:, None, None]
        * displacement[:, None, :]
        / radius[:, None, None]
    )
    return velocity, gradient


def test_gaussian_biot_savart_velocity_and_gradient_match_the_closed_form(
    tmp_path, record_property
):
    """Claim: direct Gaussian fields implement the regularized Biot--Savart law.

    The oracle is the closed-form Gaussian kernel and its analytic spatial
    derivative, evaluated away from the removable source singularity.  The
    error norm is relative L2 over 12 fixed points.  CPU/f64 and seed 42 remove
    backend and stochastic uncertainty.  The kernel deliberately uses the
    Abramowitz--Stegun five-coefficient erf approximation (maximum absolute
    error about 1.5e-7), so 2e-7 is the derived implementation error budget.
    No time integration or spatial discretization is involved.
    """
    solver = _solver(tmp_path, "analytic", DirectInduction(), precision="f64")
    source_position = np.array([[0.1, -0.2, 0.05]])
    vortex_strength = np.array([[0.7, -0.3, 0.4]])
    core_radius = np.array([0.27])
    _add_cloud(solver, source_position, vortex_strength, core_radius)
    points = np.array(
        [
            [-0.8, -0.3, 0.2],
            [-0.5, 0.4, -0.1],
            [-0.2, -0.6, 0.5],
            [0.0, 0.2, 0.7],
            [0.2, -0.1, -0.5],
            [0.3, 0.5, 0.1],
            [0.5, -0.4, -0.2],
            [0.6, 0.1, 0.4],
            [0.8, -0.2, 0.3],
            [-0.4, 0.7, -0.3],
            [0.1, -0.7, -0.4],
            [0.7, 0.6, -0.1],
        ]
    )
    expected_velocity, expected_gradient = _single_gaussian_field(
        points, source_position[0], vortex_strength[0], core_radius[0]
    )
    actual_velocity = solver.compute_velocity_at_points(points)
    actual_gradient = solver.compute_velocity_gradient_at_points(points, particle_spacing=0.15)
    velocity_error = float(
        np.linalg.norm(actual_velocity - expected_velocity) / np.linalg.norm(expected_velocity)
    )
    gradient_error = float(
        np.linalg.norm(actual_gradient - expected_gradient) / np.linalg.norm(expected_gradient)
    )
    record_property("velocity_relative_l2", velocity_error)
    record_property("gradient_relative_l2", gradient_error)
    record_property("precision", "f64")
    record_property("backend", "CPU")
    assert velocity_error < 2.0e-7
    assert gradient_error < 2.0e-7


def test_treecode_converges_to_direct_summation_as_the_opening_angle_closes(
    tmp_path, record_property
):
    """Claim: tightening the tree opening angle converges toward direct summation.

    The independent oracle is the solver's pairwise direct backend for the
    same deterministic 512-particle cloud.  Relative L2 velocity error is
    measured at 64 off-particle targets for theta=(0.8, 0.4, 0.2), CPU/f32,
    multipole order two, seed 20260831.  The acceptance limits require strict
    refinement improvement and a fine-angle error below 0.5%; both are much
    larger than f32 accumulation noise and do not assume a particular tree.
    """
    rng = np.random.default_rng(20260831)
    axis = np.linspace(-0.8, 0.8, 8)
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    position = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
    vortex_strength = 0.01 * rng.normal(size=(len(position), 3))
    vortex_strength -= vortex_strength.mean(axis=0)
    core_radius = np.full(len(position), 0.15)
    targets = rng.uniform(-1.1, 1.1, size=(64, 3)) + np.array([0.013, 0.027, 0.041])

    direct = _solver(tmp_path, "direct", DirectInduction())
    _add_cloud(direct, position, vortex_strength, core_radius)
    reference = direct.compute_velocity_at_points(targets)

    errors = []
    for theta_tenths in (8, 4, 2):
        theta = theta_tenths / 10.0
        tree = _solver(
            tmp_path,
            f"tree_theta_tenths_{theta_tenths}",
            TreecodeInduction._for_testing(theta=theta, multipole_order=2),
        )
        _add_cloud(tree, position, vortex_strength, core_radius)
        approximation = tree.compute_velocity_at_points(targets)
        errors.append(float(np.linalg.norm(approximation - reference) / np.linalg.norm(reference)))

    observed_order = float(np.log(errors[0] / errors[-1]) / np.log(0.8 / 0.2))
    record_property("theta_0p8_relative_l2", errors[0])
    record_property("theta_0p4_relative_l2", errors[1])
    record_property("theta_0p2_relative_l2", errors[2])
    record_property("observed_theta_order", observed_order)
    record_property("precision", "f32")
    record_property("backend", "CPU")
    assert errors[0] > errors[1] > errors[2]
    assert errors[2] < 5.0e-3
