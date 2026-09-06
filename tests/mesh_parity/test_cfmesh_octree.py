"""Sparse addressing and distinct native box/surface refinement contracts."""

import numpy as np

import openonda.fvm.mesher as msh
from source.solvers.fvm.mesh.cartesian.cfmesh_octree import LeafLookup, object_additional_level
from source.solvers.fvm.mesh.cartesian.cfmesh_template import _additional_level


def test_exact_dyadic_object_request_uses_native_strict_bound():
    assert object_additional_level(1.0, 0.25) == 3
    assert _additional_level(1.0, 0.25) == 2
    assert object_additional_level(1.0, 0.3) == 2
    assert _additional_level(1.0, 0.3) == 2


def test_sparse_lookup_matches_dense_queries_and_morton_order():
    leaves = []
    for z in (0, 2):
        for y in (0, 2):
            for x in (0, 2):
                if (x, y, z) == (0, 0, 0):
                    for dz in (0, 1):
                        for dy in (0, 1):
                            for dx in (0, 1):
                                leaves.append((dx, dy, dz, 1, 2, 1))
                else:
                    leaves.append((x, y, z, 2, 1, 1))
    lookup = LeafLookup(leaves, 2)
    dense = np.full((4, 4, 4), -1)
    for i, (x, y, z, width, _level, _kind) in enumerate(leaves):
        dense[x : x + width, y : y + width, z : z + width] = i
    for point in np.ndindex(dense.shape):
        assert lookup.find(*point) == dense[point]
    rng = np.random.default_rng(17)
    for _ in range(50):
        lower = rng.integers(0, 4, size=3)
        upper = np.asarray([rng.integers(value, 4) for value in lower])
        expected = np.unique(
            dense[lower[0] : upper[0] + 1, lower[1] : upper[1] + 1, lower[2] : upper[2] + 1]
        )
        np.testing.assert_array_equal(lookup.in_box(lower, upper), expected)


def test_patch_refinement_is_a_public_validated_control():
    request = msh.PatchRefinement(patch="body", cell_size=0.25)
    assert request.patch == "body"
    assert request.cell_size == 0.25
