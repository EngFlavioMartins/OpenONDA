"""The planar cylinder seed is solenoidal, compact and uniform along the span."""

from pathlib import Path

import numpy as np

from openonda.tutorial_runner import load_case_module

CASE = Path(__file__).resolve().parents[2] / "tutorials/coupled_fvm_vpm/01_cylinder_shedding_flow"
tutorial = load_case_module(CASE)
initial_velocity = load_case_module(CASE, "assets.initial_conditions").cylinder_initial_velocity


def cylinder_initial_velocity(positions):
    return initial_velocity(
        positions,
        **tutorial.INITIAL_PERTURBATION,
        freestream_velocity=tutorial.FREESTREAM_VELOCITY,
    )


def test_perturbation_is_solenoidal_compact_and_spanwise_invariant():
    points = np.array([[0.55, 0.1, 0.0], [0.75, -0.1, 0.2], [0.9, 0.15, -0.3]])
    eps = 1e-5
    divergence = np.zeros(len(points))
    for axis in range(3):
        displacement = np.eye(3)[axis] * eps
        upper = cylinder_initial_velocity(points + displacement)
        lower = cylinder_initial_velocity(points - displacement)
        divergence += (upper[:, axis] - lower[:, axis]) / (2 * eps)
    np.testing.assert_allclose(divergence, 0, atol=2e-11)
    np.testing.assert_array_equal(cylinder_initial_velocity(points)[:, 2], 0)
    # A nonzero centreline transverse velocity excites the shedding mode even
    # on a perfectly reflection-symmetric mesh; it is independent of z.
    centreline = cylinder_initial_velocity(np.array([[0.9, 0, -0.5], [0.9, 0, 0.5]]))
    assert abs(centreline[0, 1]) > 1e-4
    assert centreline[0, 1] == centreline[1, 1]
    np.testing.assert_array_equal(
        cylinder_initial_velocity(points + [0, 0, 2]), cylinder_initial_velocity(points)
    )
    np.testing.assert_array_equal(
        cylinder_initial_velocity(np.array([[1.48, 0, 0], [-1.48, 0.2, 0.1]])),
        [[1.0, 0, 0], [1.0, 0, 0]],
    )


def test_reference_and_coupled_seed_identical_planar_startup_fields():
    reference = load_case_module(CASE / "reference_flow")
    reference_velocity = load_case_module(
        CASE / "reference_flow", "assets.initial_conditions"
    ).cylinder_initial_velocity
    points = np.array([[0.55, 0.1, -0.5], [0.75, -0.1, 0.0], [0.9, 0.15, 0.5], [2, 0, 0]])
    np.testing.assert_array_equal(
        initial_velocity(
            points,
            **tutorial.INITIAL_PERTURBATION,
            freestream_velocity=tutorial.STARTUP_FREESTREAM_VELOCITY,
        ),
        reference_velocity(
            points,
            **reference.INITIAL_PERTURBATION,
            freestream_velocity=reference.STARTUP_FREESTREAM_VELOCITY,
        ),
    )
