"""Bounded host-only census checks; no numerical production imports."""

import numpy as np

from tests.vpm._finite_mesh_native_census import (
    correction_tail,
    count_corrections,
    finite_images,
    image_bounds,
    mesh_payload,
    reflected_positions,
    separated_boxes,
)
from tests.vpm._gaussian_broadening_reference import correction_fields


def test_full_directed_pair_counts_match_independent_dense_distances():
    rng = np.random.default_rng(746)
    x = rng.uniform([-0.7, -0.4, -0.3], [0.9, 0.5, 0.3], (23, 3))
    images = finite_images(-0.4, 0.4, 2)
    cutoffs = np.array([0.11, 0.35, 0.63, 1.31])
    counts, records = count_corrections(x, images, cutoffs)
    expected = np.zeros(len(cutoffs), np.int64)
    for (shift, odd), record in zip(images, records, strict=True):
        source = reflected_positions(x, shift, odd)
        radius = np.linalg.norm(x[:, None] - source[None], axis=-1)
        pair_counts = np.array([np.count_nonzero(radius <= value) for value in cutoffs])
        np.testing.assert_array_equal(record["near_pairs"], pair_counts)
        expected += pair_counts
        assert record["aabb_distance_lower"] <= radius.min()
    np.testing.assert_array_equal(counts, expected)
    assert len(images) == 9


def test_finite_tail_envelope_dominates_every_targets_omitted_correction():
    x = np.array([[0.01, -0.02, -0.1], [0.08, 0.03, 0.15], [-0.1, 0.02, 0.05]])
    gamma = np.array([[0.3, -0.4, 0.1], [-0.2, 0.3, 0.7], [0.4, 0.1, -0.3]])
    sigma, tau, cutoff = np.array([0.04, 0.05, 0.08]), 0.1, 0.23
    images = finite_images(-0.2, 0.2, 2)
    _, records = count_corrections(x, images, [cutoff])
    envelope = correction_tail(np.linalg.norm(gamma, axis=1).sum(), tau, cutoff, records)
    u, j = np.zeros_like(x), np.zeros((len(x), 3, 3))
    for shift, odd in images:
        source = reflected_positions(x, shift, odd)
        vectors = gamma.copy()
        if odd:
            vectors[:, :2] *= -1
        for target, q in enumerate(x):
            for p, g, s in zip(source, vectors, sigma, strict=True):
                if np.linalg.norm(q-p) > cutoff:
                    du, dj = correction_fields(q-p, g, s, tau)
                    u[target] += du
                    j[target] += dj
    assert np.max(np.linalg.norm(u, axis=1)) <= envelope["velocity"]
    assert np.max(np.linalg.norm(j, axis=(1, 2))) <= envelope["gradient_frobenius"]


def test_compact_grid_and_reflected_boxes_are_translation_safe():
    x = np.array([[0.1, 0.2, -0.375], [0.8, -0.4, 0.375]])
    before = mesh_payload(x, 0.04, 10)
    x[:, 2] += 1024
    after = mesh_payload(x, 0.04, 10)
    assert before == after
    lower, upper = x.min(axis=0), x.max(axis=0)
    a, b = image_bounds(lower, upper, 2048.8, True)
    y = reflected_positions(x, 2048.8, True)
    np.testing.assert_array_equal(a, y.min(axis=0))
    np.testing.assert_array_equal(b, y.max(axis=0))
    assert separated_boxes(lower, upper, a, b) <= np.linalg.norm(x[:, None]-y[None], axis=-1).min()
    assert before["declared_float64_payload_bytes"] == 2 * before["declared_float32_payload_bytes"]
