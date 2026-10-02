"""Pure NumPy algebra tests; these do not certify a GPU admission policy."""

import numpy as np
import pytest

from tests.vpm._fmm_source_remainder_prototype import singular_source_taylor
from tests.vpm._high_order_moment_reference import (
    build_compact_moments,
    direct_moments,
    moment_fields,
    storage_and_work,
    translate_moments,
)


@pytest.mark.parametrize("order", [3, 7, 9])
@pytest.mark.parametrize("units", [1e-5, 1.0, 1e5])
def test_scaled_translation_matches_direct_moments(order, units):
    rng = np.random.default_rng(103)
    position = (rng.normal(size=(32, 3)) * 0.08 + [1.0, -0.5, 0.7]) * units
    strength = rng.normal(size=(32, 3)) * units**2
    child, parent = np.array([1.0, -0.5, 0.7]) * units, np.array([0.6, -0.1, 1.1]) * units
    child_scale, parent_scale = 0.3 * units, 0.9 * units
    before = direct_moments(position, strength, child, child_scale, order)[0]
    shifted = translate_moments(before, child, child_scale, parent, parent_scale, order)
    expected, absolute = direct_moments(position, strength, parent, parent_scale, order)
    assert np.all(np.abs(shifted - expected) <= 5e-14 * np.maximum(absolute, units**2 * 1e-20))


@pytest.mark.parametrize("order", [7, 9])
def test_sparse_active_ids_need_no_fine_node_moment_storage(order):
    rng = np.random.default_rng(19)
    position, strength = rng.uniform(-0.2, 0.2, (8, 3)), rng.normal(size=(8, 3))
    centres = {900: np.zeros(3), 42: np.array([-0.05, 0, 0]), 317: np.array([0.05, 0, 0])}
    compact = build_compact_moments(
        position, strength, centres=centres, scales={900: 0.4, 42: 0.3, 317: 0.3},
        leaves={42: np.arange(4), 317: np.arange(4, 8)}, children={900: (42, 317)}, order=order,
    )
    assert compact.values.shape == (3, (120 if order == 7 else 220), 3)
    direct = direct_moments(position, strength, centres[900], 0.4, order)[0]
    np.testing.assert_allclose(compact.values[compact.slots[900]], direct, rtol=2e-13, atol=2e-16)


@pytest.mark.parametrize("order", [7, 9])
@pytest.mark.parametrize("geometry", ["random", "cancelled", "singleton", "collinear"])
def test_field_oracle_agrees_with_independent_derivative_algebra(order, geometry):
    rng = np.random.default_rng(71)
    position, strength = rng.uniform(-0.15, 0.15, (12, 3)), rng.normal(size=(12, 3))
    if geometry == "cancelled":
        strength -= np.mean(strength, axis=0)
    elif geometry == "singleton":
        position, strength = position[:1], strength[:1]
    elif geometry == "collinear":
        position[:, 1:] = 0
    centre, target = np.array([0.02, -0.03, 0.01]), np.array([1.3, 0.4, -0.5])
    values = direct_moments(position, strength, centre, 0.4, order)[0]
    actual = moment_fields(values, centre, 0.4, target, order)
    reference = singular_source_taylor(position, strength, target, centre, order)
    for value, expected in zip(actual, reference, strict=True):
        np.testing.assert_allclose(value, expected, rtol=2e-11, atol=1e-15)


def test_costs_show_compact_storage_does_not_remove_high_order_translation_cost():
    p7, p9 = (storage_and_work(order, 20000, 300000) for order in (7, 9))
    assert p7["moment_bytes"] == 28800000 and p9["moment_bytes"] == 52800000
    assert p7["m2m_vector_terms_per_binary_parent"] == 3432
    assert p9["m2m_vector_terms_per_binary_parent"] == 10010
    assert p9["m2l_vector_terms_per_pair"] == 26400
    assert not p9["runtime_qualified"] and not p9["complete_arithmetic_error_certificate"]
