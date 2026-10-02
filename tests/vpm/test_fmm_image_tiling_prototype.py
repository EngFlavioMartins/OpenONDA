"""Pure NumPy orchestration tests: no solver/device imports or compilation."""

from types import SimpleNamespace

import numpy as np
import pytest

from tests.vpm._fmm_image_tiling_prototype import ImageTileDispatcher


class Storage:
    def __init__(self, count, tile):
        self.count, self.bytes, self.closed = count, (count + tile) * 48, False
        self.velocity, self.gradient = np.zeros((count, 3)), np.zeros((count, 3, 3))
        self.tile_velocity, self.tile_gradient = np.zeros((tile, 3)), np.zeros((tile, 3, 3))

    def close(self):
        self.closed = True


def copy(source, target, first, offset, count):
    target[offset:offset + count] = source[first:first + count]


@pytest.mark.parametrize("capacity", [8, 32])
def test_every_target_is_scattered_once_with_nonzero_query_start_and_remainder(capacity):
    dispatcher = ImageTileDispatcher(capacity, allocate=Storage, copy=copy, synchronize=lambda: None)
    positions = np.arange(50 * 3).reshape(50, 3)
    velocity, gradient = np.full((37, 3), -99.0), np.full((37, 3, 3), -99.0)
    backend = SimpleNamespace(_fixed_source_key=True, _target_workspace=SimpleNamespace(
        max_pairs=4194304, last_diagnostics={}), _estimate_memory_bytes=lambda: 100)
    visited = []

    def original(owner, **kwargs):
        start, count = kwargs["target_start"], kwargs["target_count"]
        visited.extend(range(start, start + count))
        kwargs["target_velocity"][:count] = positions[start:start + count]
        kwargs["target_velocity_gradient"][:count] = positions[start:start + count, :, None]

    dispatcher(backend, original, source_position=None, source_vortex_strength=None,
               source_core_radius=None, source_count=2, target_position=positions,
               target_start=4, target_count=35, images=[(0, 1)],
               target_velocity=velocity, target_velocity_gradient=gradient)
    np.testing.assert_array_equal(velocity[:35], positions[4:39])
    np.testing.assert_array_equal(gradient[:35], np.repeat(positions[4:39, :, None], 3, axis=2))
    assert np.all(velocity[35:] == -99) and np.all(gradient[35:] == -99)
    assert visited == list(range(4, 39))
    assert sum(tile["target_count"] for tile in dispatcher.records[0]["tiles"]) == 35
    storage = dispatcher.storage[id(backend)]
    dispatcher.close()
    assert storage.closed


def test_failed_later_tile_never_publishes_partial_caller_fields():
    dispatcher = ImageTileDispatcher(8, allocate=Storage, copy=copy, synchronize=lambda: None)
    backend = SimpleNamespace(_fixed_source_key=True, _target_workspace=SimpleNamespace(
        max_pairs=4096, last_diagnostics={}), _estimate_memory_bytes=lambda: 100)
    velocity, gradient = np.full((16, 3), -99.0), np.full((16, 3, 3), -99.0)

    def original(owner, **kwargs):
        if kwargs["target_start"] == 8:
            raise RuntimeError("declined second tile")
        kwargs["target_velocity"][:] = 42
        kwargs["target_velocity_gradient"][:] = 42

    with pytest.raises(RuntimeError, match="second tile"):
        dispatcher(backend, original, source_position=None, source_vortex_strength=None,
                   source_core_radius=None, source_count=2, target_position=None,
                   target_start=0, target_count=16, images=[(0, 1)],
                   target_velocity=velocity, target_velocity_gradient=gradient)
    assert np.all(velocity == -99) and np.all(gradient == -99)
    assert dispatcher.records[0]["status"] == "failed"
