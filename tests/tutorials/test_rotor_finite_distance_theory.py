import numpy as np
import pytest
from scipy import special
from scipy.integrate import quad

from tests._tutorial_helpers import load_tutorial_module

_theory = load_tutorial_module("vpm/rotor_flow", "assets.finite_distance_theory")
_complete_elliptic_pi = _theory._complete_elliptic_pi
build_system = _theory.build_system
induced_velocity = _theory.induced_velocity
right_cylinder_influence = _theory.right_cylinder_influence


def test_complete_elliptic_pi_reduces_to_first_kind_when_n_is_zero():
    for parameter in (0.0, 0.2, 0.7, 0.95):
        assert np.isclose(_complete_elliptic_pi(0.0, parameter), special.ellipk(parameter))


def test_right_cylinder_influence_is_finite_away_from_the_sheet():
    inside = right_cylinder_influence(0.5, 1.0, 2.0)
    outside = right_cylinder_influence(1.5, 1.0, 2.0)
    assert np.isfinite(inside).all()
    assert np.isfinite(outside).all()
    assert not np.allclose(inside, outside)


def test_right_cylinder_influence_has_correct_leading_and_far_wake_limits():
    near_inside = right_cylinder_influence(0.5, 1.0, 1.0e-8)
    near_outside = right_cylinder_influence(1.5, 1.0, 1.0e-8)
    far_inside = right_cylinder_influence(0.5, 1.0, 1.0e8)
    far_outside = right_cylinder_influence(1.5, 1.0, 1.0e8)

    # At the leading disk the semi-infinite tangential sheet has its
    # half-jump; far downstream it reaches the full inside-cylinder value.
    assert near_inside[0] == pytest.approx(0.5)
    assert near_outside[0] == pytest.approx(0.0, abs=1.0e-7)
    assert far_inside[0] == pytest.approx(1.0)
    assert far_outside[0] == pytest.approx(0.0, abs=1.0e-12)
    # The longitudinal sheet has the corresponding half-jump outside and
    # the circulation-over-radius limit far downstream.
    assert near_inside[1] == pytest.approx(0.0, abs=1.0e-7)
    assert near_outside[1] == pytest.approx(1.0 / 3.0)
    assert far_inside[1] == pytest.approx(0.0, abs=1.0e-12)
    assert far_outside[1] == pytest.approx(2.0 / 3.0)


def test_positive_loaded_sheet_gives_positive_near_disk_axial_induction():
    system = type(
        "System",
        (),
        {
            "trailing_radii": np.array([1.0]),
            "tangential_sheet_strength": np.array([-2.0 * 7.0 * 0.2]),
            "longitudinal_sheet_strength": np.zeros(1),
        },
    )()

    result = induced_velocity(
        0.5,
        1.0e-8,
        system,
        freestream_speed=7.0,
        angular_velocity=8.0,
    )

    assert result["axial_induction"] == pytest.approx(0.2, rel=1.0e-6)


def test_axial_cylinder_formula_matches_independent_ring_quadrature():
    radius, cylinder_radius, downstream = 0.5, 1.0, 2.0

    def ring_axial(source_offset):
        scale = np.sqrt((cylinder_radius + radius) ** 2 + source_offset**2)
        parameter = 4.0 * cylinder_radius * radius / scale**2
        denominator = (cylinder_radius - radius) ** 2 + source_offset**2
        return (
            special.ellipk(parameter)
            + (cylinder_radius**2 - radius**2 - source_offset**2)
            / denominator
            * special.ellipe(parameter)
        ) / (2.0 * np.pi * scale)

    quadrature, _ = quad(
        lambda source: ring_axial(downstream - source), 0.0, np.inf, epsabs=1e-9, limit=400
    )
    analytical, _ = right_cylinder_influence(radius, cylinder_radius, downstream)
    assert np.isclose(analytical, quadrature, rtol=1e-8, atol=1e-9)


def test_bem_superposition_adds_ghost_sections_and_closes_sheet_strengths():
    bem = {
        "radial_position": np.array([1.0, 2.0, 3.0]),
        "circulation": np.array([1.2, 1.0, 0.4]),
        "tangential_induction_factor": np.array([0.08, 0.05, 0.02]),
    }
    system = build_system(
        bem,
        number_of_blades=3,
        freestream_speed=7.0,
        angular_velocity=8.0,
        hub_radius=0.5,
        rotor_radius=3.5,
    )
    assert len(system.trailing_radii) == 4
    assert len(system.tangential_sheet_strength) == 4
    assert len(system.longitudinal_sheet_strength) == 4
    assert np.isclose(system.tangential_sheet_strength.sum(), 0.0)
    assert np.isfinite(system.longitudinal_sheet_strength).all()


def test_zero_strength_system_has_zero_induced_components():
    system = type(
        "System",
        (),
        {
            "trailing_radii": np.array([1.0, 2.0]),
            "tangential_sheet_strength": np.zeros(2),
            "longitudinal_sheet_strength": np.zeros(2),
        },
    )()
    result = induced_velocity(
        0.75,
        4.0,
        system,
        freestream_speed=7.0,
        angular_velocity=8.0,
    )
    assert result["axial_velocity"] == 0.0
    assert result["tangential_velocity"] == 0.0
    assert result["axial_induction"] == 0.0
    assert result["tangential_induction"] == 0.0
