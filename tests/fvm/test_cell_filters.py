"""Spectral contract of CellBoxFilter.

The fringe relaxation in the FVM-VPM coupler builds its source from the filter
RESIDUAL ``f - G*f``, so the transfer function's sign matters: a filter with a
negative grid-scale lobe makes the residual larger than the field itself, which
flips the source's sign and drives the very modes it is meant to leave alone.
These tests pin the two weightings apart so that cannot regress silently.
"""

import numpy as np
import pytest

from source.solvers.FVM.fields.filters import CellBoxFilter


def _chain(n: int = 12):
    """1-D chain of n unit-volume cells; interior faces link i to i+1."""
    mesh = {
        "n_interior_faces": n - 1,
        "owners": np.arange(n - 1),
        "neighbours": np.arange(1, n),
    }
    geo = {"element_volumes": np.ones(n)}
    return mesh, geo


def _modes(n: int):
    k = np.arange(n)
    return {
        "dc": np.ones((n, 3)),
        "nyquist": ((-1.0) ** k)[:, None] * np.ones((1, 3)),
        "smooth": np.cos(2 * np.pi * k / n)[:, None] * np.ones((1, 3)),
    }


@pytest.mark.parametrize("centre_weight", ["volume", "neighbour_sum"])
def test_constant_is_preserved(centre_weight):
    """Partition of unity: any filter must leave a uniform field untouched."""
    mesh, geo = _chain()
    f = CellBoxFilter(mesh, geo, centre_weight=centre_weight)
    const = np.full((mesh["n_interior_faces"] + 1, 3), 2.5)
    np.testing.assert_allclose(f(const), const)


def test_neighbour_sum_has_no_negative_lobe():
    """DC gain 1, grid-scale gain 0 — so the residual stays within [0, 1]*f.

    This is the property the scale-selective fringe source depends on.
    """
    n = 12
    mesh, geo = _chain(n)
    f = CellBoxFilter(mesh, geo, centre_weight="neighbour_sum")
    m = _modes(n)
    i = n // 2  # interior cell, away from the truncated ends

    assert f(m["dc"])[i, 0] == pytest.approx(1.0)
    assert f(m["nyquist"])[i, 0] / m["nyquist"][i, 0] == pytest.approx(0.0, abs=1e-12)
    # Retained fraction (what the fringe leaves unrelaxed) must not exceed 1.
    retained_nyq = (m["nyquist"] - f(m["nyquist"]))[i, 0] / m["nyquist"][i, 0]
    retained_smooth = (m["smooth"] - f(m["smooth"]))[i, 0] / m["smooth"][i, 0]
    assert retained_nyq == pytest.approx(1.0)
    assert 0.0 < retained_smooth < 0.25  # resolved scales are still relaxed


def test_volume_weighting_does_have_a_negative_lobe():
    """The classical box filter overshoots — documented, and why it is not the
    default for the fringe.  Kept as the dynamic Smagorinsky test filter."""
    n = 12
    mesh, geo = _chain(n)
    f = CellBoxFilter(mesh, geo, centre_weight="volume")
    m = _modes(n)
    i = n // 2

    assert f(m["nyquist"])[i, 0] / m["nyquist"][i, 0] == pytest.approx(-1.0 / 3.0)
    retained = (m["nyquist"] - f(m["nyquist"]))[i, 0] / m["nyquist"][i, 0]
    assert retained > 1.0  # exactly the amplification the fringe must avoid


@pytest.mark.parametrize("shape", [(12,), (12, 3), (12, 3, 3)])
def test_shape_is_preserved(shape):
    """Scalars, vectors and tensors all filter — Smagorinsky passes (n,3,3)."""
    mesh, geo = _chain(12)
    f = CellBoxFilter(mesh, geo)
    assert f(np.ones(shape)).shape == shape


def test_rejects_unknown_centre_weight():
    mesh, geo = _chain()
    with pytest.raises(ValueError, match="centre_weight"):
        CellBoxFilter(mesh, geo, centre_weight="bogus")


def test_matches_the_dynamic_smagorinsky_test_filter():
    """The lifted filter must reproduce what DynamicSmagorinsky used to compute."""
    from source.solvers.FVM.turbulence.les_models import DynamicSmagorinsky

    n = 12
    mesh, geo = _chain(n)
    rng = np.random.default_rng(0)
    field = rng.normal(size=(n, 3))

    model = DynamicSmagorinsky(mesh, geo)
    np.testing.assert_allclose(
        model._box_filter(field), CellBoxFilter(mesh, geo, centre_weight="volume")(field)
    )
