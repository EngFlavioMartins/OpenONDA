"""CPU-only exact geometry-cache qualification; no Taichi imports or runtime."""

import numpy as np
import pytest

from tests.vpm._gbd_body_host_oracle import legacy_host_type
from tests.vpm._gbd_geometry_cache_prototype import ExactBodyGridCache, _matching_axis


def _body(points):
    return np.sum(points**2, axis=1) < 0.22


def _thin_wall(starts, ends):
    # Fluid-fluid links can cross a thin open wall, including tangency.
    return ((starts[:, 0] <= 0.13) & (ends[:, 0] > 0.13)
            & (np.abs(starts[:, 1]) < 0.8))


def _prepare(cache, origin, shape, *, spacing=0.1, contains=_body, blocks=_thin_wall,
             revision="immutable-1", slab=None):
    return cache.prepare(origin, spacing, shape, contains=contains, blocks=blocks,
                         revision=revision, slab=slab)


def _fresh(origin, shape, **kwargs):
    return _prepare(ExactBodyGridCache(max_bytes=0), origin, shape, **kwargs)


@pytest.mark.parametrize("origin,shape", [
    ((-0.5, -0.5, -0.5), (15, 12, 11)),
    ((-0.4, -0.5, -0.5), (10, 10, 10)),
    ((-0.6, -0.5, -0.4), (14, 9, 12)),
    ((1000.03, 0.01, 0.03), (9, 5, 3)),
])
def test_changed_extents_and_origin_are_bitwise_fresh(origin, shape):
    cache = ExactBodyGridCache(qualified_bindings=((_body, _thin_wall),))
    _prepare(cache, (-0.5, -0.5, -0.5), (11, 11, 11))
    actual = _prepare(cache, origin, shape)
    expected = _fresh(origin, shape)
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])
    assert actual[2]["membership_reused"] + actual[2]["membership_queried"] == np.prod(shape)


def test_growing_same_origin_reuses_old_nodes_and_only_valid_old_edges():
    cache = ExactBodyGridCache(qualified_bindings=((_body, _thin_wall),))
    origin = (-0.5, -0.5, -0.5)
    _prepare(cache, origin, (11, 11, 11))
    mask, links, stats = _prepare(cache, origin, (13, 12, 11))
    expected = _fresh(origin, (13, 12, 11))
    np.testing.assert_array_equal(mask, expected[0])
    np.testing.assert_array_equal(links, expected[1])
    assert stats["membership_reused"] == 11**3
    assert stats["membership_queried"] == 13 * 12 * 11 - 11**3
    assert stats["links_reused"] > stats["links_queried"]


def test_outputs_do_not_alias_retained_record():
    cache = ExactBodyGridCache(qualified_bindings=((_body, _thin_wall),))
    mask, links, _ = _prepare(cache, (-0.5,) * 3, (11,) * 3)
    mask[:] = ~mask
    links[:] = 7
    actual = _prepare(cache, (-0.5,) * 3, (11,) * 3)
    expected = _fresh((-0.5,) * 3, (11,) * 3)
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])
    assert actual[2]["membership_queried"] == 0
    assert actual[2]["links_queried"] == 0


@pytest.mark.parametrize("change", ["revision", "spacing", "slab", "callback"])
def test_context_changes_invalidate_geometry(change):
    def other_body(p):
        return ~_body(p)
    cache = ExactBodyGridCache(qualified_bindings=((_body, _thin_wall), (other_body, _thin_wall)))
    _prepare(cache, (-0.5,) * 3, (11,) * 3)
    kw = {"revision": "immutable-2"} if change == "revision" else {}
    if change == "spacing":
        kw["spacing"] = np.nextafter(0.1, 1.0)
    elif change == "slab":
        kw["slab"] = (-0.48, 0.48)
    elif change == "callback":
        kw["contains"] = other_body
    actual = _prepare(cache, (-0.5,) * 3, (11,) * 3, **kw)
    expected = _fresh((-0.5,) * 3, (11,) * 3, **kw)
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])
    assert actual[2]["membership_reused"] == actual[2]["links_reused"] == 0


@pytest.mark.parametrize("revision", [[1], {"revision": 1}, ("good", []), float("nan")])
def test_untrusted_revision_fails_closed(revision):
    cache = ExactBodyGridCache(qualified_bindings=((_body, _thin_wall),))
    _prepare(cache, (-0.5,) * 3, (11,) * 3, revision=revision)
    _, _, stats = _prepare(cache, (-0.5,) * 3, (11,) * 3, revision=revision)
    assert not stats["qualified"] and stats["membership_reused"] == 0
    assert cache.record is None


def test_custom_callback_without_explicit_qualification_is_never_cached():
    cache = ExactBodyGridCache()
    _prepare(cache, (-0.5,) * 3, (11,) * 3)
    _, _, stats = _prepare(cache, (-0.5,) * 3, (11,) * 3)
    assert not stats["qualified"] and stats["retained_bytes"] == 0


def test_cap_falls_back_without_altering_results():
    cache = ExactBodyGridCache(qualified_bindings=((_body, _thin_wall),), max_bytes=100)
    actual = _prepare(cache, (-0.5,) * 3, (11,) * 3)
    expected = _fresh((-0.5,) * 3, (11,) * 3)
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])
    assert cache.record is None and actual[2]["retained_bytes"] == 0


def test_failed_query_does_not_publish_or_retain_partial_geometry():
    state = {"fail": False}

    def blocks(starts, ends):
        if state["fail"]:
            raise RuntimeError("injected query failure")
        return _thin_wall(starts, ends)

    cache = ExactBodyGridCache(qualified_bindings=((_body, blocks),))
    _prepare(cache, (-0.5,) * 3, (11,) * 3, blocks=blocks)
    state["fail"] = True
    with pytest.raises(RuntimeError, match="injected"):
        _prepare(cache, (-0.5,) * 3, (12,) * 3, blocks=blocks)
    assert cache.record is None


def test_axis_matching_does_not_equate_signed_zero_or_close_coordinates():
    previous = np.array([-0.0, 0.1, 0.2], dtype=np.float32)
    current = np.array([0.0, np.nextafter(np.float32(0.1), np.float32(1)), 0.2], dtype=np.float32)
    now, old = _matching_axis(previous, current)
    np.testing.assert_array_equal(now, [2])
    np.testing.assert_array_equal(old, [2])


def test_nominal_integer_lattice_shift_does_not_imply_coordinate_equality():
    old = np.float32(-5.221431732177734) + np.arange(40, dtype=np.int64) * 0.04
    shifted = np.float32(np.float64(np.float32(-5.221431732177734)) + 0.04)
    new = shifted + np.arange(39, dtype=np.int64) * 0.04
    assert not np.array_equal(old[1:], new)
    now, _ = _matching_axis(old, new)
    assert len(now) == 0


def test_singleton_axes_no_segments_and_empty_callback_queries():
    cache = ExactBodyGridCache(qualified_bindings=((_body, _thin_wall),))
    _prepare(cache, (1.0,) * 3, (1,) * 3)
    _, _, stats = _prepare(cache, (1.0,) * 3, (1,) * 3)
    assert stats["membership_reused"] == 1
    assert stats["links_queried"] == stats["links_reused"] == 0


@pytest.mark.parametrize("slab", [None, (-0.48, 0.48), (0.0, 0.6)])
def test_against_actual_legacy_method_bodies_including_folded_segments(slab):
    host_type = legacy_host_type()
    callback_owner = host_type((1, 1, 1), _body, _thin_wall, slab)
    contains = callback_owner._body_interior_at_particles
    blocks = callback_owner._body_blocked_segments
    cache = ExactBodyGridCache(qualified_bindings=((contains, blocks),))
    for origin, shape in [((-0.5, -0.5, -0.7), (11, 11, 16)),
                          ((-0.5, -0.5, -0.7), (13, 12, 18)),
                          ((-0.4, -0.5, -0.6), (12, 12, 17))]:
        actual = _prepare(cache, origin, shape, contains=contains, blocks=blocks, slab=slab)
        reference = host_type(shape, _body, _thin_wall, slab)
        reference._prepare_body_mask_current_grid(np.asarray(origin), 0.1, *shape)
        np.testing.assert_array_equal(actual[0], reference._body_mask_host)
        np.testing.assert_array_equal(actual[1], reference._body_links_host)
        np.testing.assert_array_equal(actual[1], reference._body_link_grid)


def test_original_input_coordinate_arithmetic_is_bitwise_preserved():
    host_type = legacy_host_type()
    observed = {"members": [], "starts": [], "ends": []}

    def contains(points):
        observed["members"].append(points.copy())
        return np.zeros(len(points), dtype=bool)

    def blocks(starts, ends):
        observed["starts"].append(starts.copy())
        observed["ends"].append(ends.copy())
        return np.zeros(len(starts), dtype=bool)

    origin, shape, spacing = (1e5 + 0.003, -5.221431732177734, -0.7), (5, 7, 9), 0.04
    host = host_type(shape, contains, blocks)
    host._prepare_body_mask_current_grid(np.asarray(origin), spacing, *shape)
    expected = {name: np.concatenate(values) for name, values in observed.items()}
    for values in observed.values():
        values.clear()
    _prepare(ExactBodyGridCache(), origin, shape, spacing=spacing, contains=contains, blocks=blocks)
    for name, values in observed.items():
        np.testing.assert_array_equal(np.concatenate(values).view(np.uint64), expected[name].view(np.uint64))
