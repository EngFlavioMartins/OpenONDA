"""Roundoff at the FVM transfer boundary must not create circulation."""

import numpy as np
import pytest

from source.coupler.vorticity_transfer import (
    _discard_unresolved_uniform_curl,
    _minimum_donor_separation,
)


def test_roundoff_floor_uses_smallest_stretched_direction() -> None:
    position = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.01]])
    assert _minimum_donor_separation(position, np.ones(3)) == pytest.approx(0.01)


@pytest.mark.parametrize("axis", range(3))
@pytest.mark.parametrize("speed", [0.1, 1.0, 100.0])
def test_uniform_flow_roundoff_is_discarded_but_resolved_weak_curl_is_kept(
    axis: int, speed: float
) -> None:
    velocity = np.zeros((8, 3))
    velocity[:, axis] = speed
    length = 0.25
    floor = 8.0 * np.finfo(np.float64).eps * speed / length
    roundoff = np.zeros_like(velocity)
    roundoff[:, (axis + 1) % 3] = floor / 2.0
    np.testing.assert_array_equal(
        _discard_unresolved_uniform_curl(roundoff, velocity, length), np.zeros_like(roundoff)
    )
    resolved = roundoff.copy()
    resolved[0, (axis + 1) % 3] = 2.0 * floor
    np.testing.assert_array_equal(
        _discard_unresolved_uniform_curl(resolved, velocity, length), resolved
    )
