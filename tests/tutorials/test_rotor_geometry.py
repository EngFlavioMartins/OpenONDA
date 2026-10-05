"""Blade leading edges must face the actual rotational relative wind."""

import numpy as np
import pytest

from tests._tutorial_helpers import load_tutorial_module

create_rotor_blade = load_tutorial_module(
    "vpm/quadcopter", "assets.generate_blade"
).create_rotor_blade


@pytest.mark.parametrize("clockwise", [False, True])
def test_quadcopter_blades_face_the_flow_and_have_radial_quarter_chords(clockwise):
    blade = create_rotor_blade(
        R_hub=0.03,
        R_tip=0.15,
        chord_root=0.025,
        chord_tip=0.015,
        pitch_root_deg=12.0,
        pitch_tip_deg=6.0,
        n_chord=4,
        n_span=10,
        clockwise=clockwise,
    )
    segment = next(iter(next(iter(blade.wings.values())).segments.values()))
    a, b, c, d = (segment.vertex_position[key] for key in "abcd")
    omega = np.array([0.0, 0.0, -400.0 if clockwise else 400.0])
    for leading, trailing in ((a, d), (b, c)):
        quarter = 0.75 * leading + 0.25 * trailing
        relative = np.array([0.0, 0.0, -0.8]) - np.cross(omega, quarter)
        chord = trailing - leading
        assert relative @ chord > 0
        assert leading[2] > trailing[2]
        np.testing.assert_allclose(quarter[[0, 2]], 0, atol=1e-17)
    assert segment.normal[2] > 0
