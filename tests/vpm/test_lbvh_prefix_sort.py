"""Active-prefix Morton sorting matches Taichi's former full-field network."""

import numpy as np
import taichi as ti
import taichi.algorithms

from source.solvers.vpm.physics.induction.treecode.lbvh import TaichiTreecode


def test_device_prefix_sort_matches_full_capacity_network_with_duplicate_keys():
    if ti.lang.impl.get_runtime().prog is None:
        ti.init(arch=ti.cpu, offline_cache=False)
    rng = np.random.default_rng(73)
    for capacity, counts in ((7, (2, 3, 7)), (19, (2, 3, 5, 11, 19)), (32, (31, 32))):
        tree = TaichiTreecode(
            max_n_particles=capacity,
            max_nodes=2 * capacity,
            hierarchy_only=True,
            device_sort_only=True,
        )
        original_keys = ti.field(ti.u32, shape=capacity)
        original_values = ti.field(ti.i32, shape=capacity)
        assert tree._sort_capacity == 1 << (capacity - 1).bit_length()
        for count in counts:
            # A small alphabet makes Morton collisions intentional.
            keys = rng.integers(0, 5, size=count, dtype=np.uint32)
            tree.morton_codes.from_numpy(np.pad(keys, (0, capacity - count)))
            tree._sort_morton(count)
            full_keys = np.pad(keys, (0, capacity - count), constant_values=0xFFFFFFFF)
            original_keys.from_numpy(full_keys)
            original_values.from_numpy(np.arange(capacity, dtype=np.int32))
            ti.algorithms.parallel_sort(keys=original_keys, values=original_values)
            got = tree.sorted_indices.to_numpy()[:count]
            expected = original_values.to_numpy()[:count]
            np.testing.assert_array_equal(got, expected)
            np.testing.assert_array_equal(keys[got], np.sort(keys))
