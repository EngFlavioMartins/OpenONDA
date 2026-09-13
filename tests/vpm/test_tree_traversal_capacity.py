"""A hierarchy must fit its traversal stack before any field can be evaluated."""

import numpy as np
import pytest
import taichi as ti

from source.solvers.vpm.physics.induction.treecode.lbvh import TaichiTreecode


def test_deep_hierarchy_is_rejected_before_children_can_be_silently_omitted():
    """A reduced stack reproduces the capacity boundary with a small real tree."""
    ti.reset()
    ti.init(arch=ti.cpu, default_fp=ti.f32, cpu_max_num_threads=2, offline_cache=False)
    try:
        tree = TaichiTreecode(max_n_particles=16, max_nodes=32, theta=0.1)
        # The allocated storage remains large enough, but traversal may use
        # only three entries. This is equivalent to reaching the real stack
        # capacity with a deeper hierarchy and avoids a million-source test.
        tree.max_stack_depth = 3
        rng = np.random.default_rng(99)
        with pytest.raises(RuntimeError, match="traversal stack"):
            tree.build(
                rng.normal(size=(16, 3)).astype(np.float32),
                rng.normal(size=(16, 3)).astype(np.float32),
                np.full(16, 0.1, np.float32),
            )
    finally:
        ti.reset()
