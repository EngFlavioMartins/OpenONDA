"""Independent annular momentum and dimensional scaling checks for rotor references."""

import numpy as np
import pytest

from openonda.rotor_theory import solve_blade_element_momentum


@pytest.mark.parametrize("mode", ["turbine", "propeller"])
def test_bem_closes_annular_momentum_and_scales_dimensional_loads(mode):
    r = np.linspace(0.15, 0.95, 30)
    c = np.full_like(r, 0.06)
    pitch = np.arctan2(0.65, 7 * r) - np.radians(4) if mode == "turbine" else np.radians(12 - 5 * r)
    options = {"hub_radius": 0.1, "mode": mode, "density": 1.2}
    result = solve_blade_element_momentum(
        r, c, pitch, 3, 1.0, 1.0, 7.0 if mode == "turbine" else 50.0, **options
    )
    omega = 7.0 if mode == "turbine" else 50.0
    a, ap, loss = (
        result.axial_induction_factor,
        result.tangential_induction_factor,
        result.loss_factor,
    )
    sign = 1 if mode == "propeller" else -1
    regular = np.ones(len(r), dtype=bool) if mode == "propeller" else (a <= 0.4)
    thrust = 4 * np.pi * 1.2 * r * a * (1 + sign * a) * loss
    torque = 4 * np.pi * 1.2 * omega * r**3 * ap * (1 + sign * a) * loss
    np.testing.assert_allclose(result.thrust_per_radius[regular], thrust[regular], rtol=2e-8)
    np.testing.assert_allclose(result.torque_per_radius, torque, rtol=2e-8)
    assert np.max(abs(result.residual)) < 1e-8
    doubled = solve_blade_element_momentum(r, c, pitch, 3, 1.0, 2.0, 2 * omega, **options)
    assert doubled.attrs["thrust"] == pytest.approx(4 * result.attrs["thrust"])
    assert doubled.attrs["power"] == pytest.approx(8 * result.attrs["power"])
    assert doubled.attrs["thrust_coefficient"] == pytest.approx(result.attrs["thrust_coefficient"])
    assert doubled.attrs["power_coefficient"] == pytest.approx(result.attrs["power_coefficient"])
