"""Sparse addressing and distinct native box/surface refinement contracts."""

from importlib.resources import files

import numpy as np
import pytest

import openonda.fvm.mesher as msh
from source.solvers.fvm.mesh.cartesian.cfmesh_octree import (
    LeafLookup,
    _balance_selection_kernel,
    balance_leaves,
    object_additional_level,
    refine_selected_leaves,
)
import source.solvers.fvm.mesh.cartesian.cfmesh_template as template
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


def _classify_leaf(x, y, z, width, level):
    # Include outside and surface-DATA leaves: they also constrain regularity.
    return (x, y, z, width, level, (x + y + z + level) % 3)


def _unbalanced_leaves(seed):
    rng = np.random.default_rng(seed)
    leaves = []

    def visit(x, y, z, width, level):
        touches_centre = all(lower <= 7 < lower + width for lower in (x, y, z))
        if level < 4 and (level == 0 or touches_centre or rng.random() < 0.25):
            child = width // 2
            for dz in (0, child):
                for dy in (0, child):
                    for dx in (0, child):
                        visit(x + dx, y + dy, z + dz, child, level + 1)
        else:
            leaves.append(_classify_leaf(x, y, z, width, level))

    visit(0, 0, 0, 16, 0)
    return leaves


def _dense_balance_selection(leaves):
    """Independent small-grid oracle, including root-boundary clipping."""
    dense = np.full((16, 16, 16), -1, dtype=int)
    for i, (x, y, z, width, _level, _kind) in enumerate(leaves):
        dense[x : x + width, y : y + width, z : z + width] = i
    assert np.all(dense >= 0)
    selected = set()
    for x, y, z, width, level, _kind in leaves:
        for offset in np.ndindex(3, 3, 3):
            query = np.asarray((x, y, z)) + (np.asarray(offset) - 1) * width
            if np.any(query < 0) or np.any(query >= 16):
                continue
            neighbour = int(dense[tuple(query)])
            if leaves[neighbour][4] + 1 < level:
                selected.add(neighbour)
    return selected


@pytest.mark.parametrize("seed", range(6))
def test_compiled_balancing_matches_dense_oracle_and_preserves_leaf_order(seed):
    original = _unbalanced_leaves(seed)
    expected = original
    passes = 0
    while True:
        selected = _dense_balance_selection(expected)
        actual = _balance_selection_kernel(np.asarray(expected, dtype=np.int64), 4)
        assert set(actual) == selected
        if not selected:
            break
        expected = refine_selected_leaves(expected, selected, 4, _classify_leaf)
        passes += 1
    assert passes > 0
    assert balance_leaves(original, 4, _classify_leaf) == expected


def test_compiled_balancing_handles_empty_and_already_balanced_inputs():
    empty = []
    assert balance_leaves(empty, 4, _classify_leaf) is empty
    root = [_classify_leaf(0, 0, 0, 1, 0)]
    assert balance_leaves(root, 0, _classify_leaf) is root


def test_compiled_balancing_does_not_pack_deep_coordinates_into_one_integer():
    # Sparse queries at depth 30 would overflow a packed x/y/z int64 key.
    width = 1 << 29
    leaves = [
        (0, 0, 0, width, 1, 0),
        (width, width, width, 1, 30, 2),
        (width + 1, width, width, 1, 30, 1),
    ]
    lookup = LeafLookup(leaves, 30)
    expected = {
        neighbour
        for leaf in leaves
        for neighbour in lookup.neighbours(leaf)
        if leaves[neighbour][4] + 1 < leaf[4]
    }
    assert expected == {0}
    assert set(_balance_selection_kernel(np.asarray(leaves, dtype=np.int64), 30)) == expected


def test_refined_cylinder_template_matches_original_python_balancing(monkeypatch):
    surface = msh.STLSurface(
        files("tutorials")
        / "coupled_fvm_vpm/01_cylinder_shedding_flow/reference_flow/assets/cylinder_long.stl",
        patch="body",
    )
    mesher = msh.CartesianMesher(
        domain=msh.BoxDomain(
            (-2.0, 2.0, -2.0, 2.0, -1.0, 1.0),
            msh.BoxPatches("xmin", "xmax", "ymin", "ymax", "zmin", "zmax"),
        ),
        surfaces=(surface,),
        max_cell_size=0.5,
        patch_refinements=(msh.PatchRefinement("body", 0.125),),
        refinements=(msh.BoxRefinement("wake", (0.0, 1.5, -0.5, 0.5, -1.0, 1.0), 0.3),),
        surface_may_cross_domain_boundary=True,
    )
    actual = mesher.build(stop_after="templateGeneration")

    def original_balance(leaves, max_level, classify):
        while True:
            lookup = LeafLookup(leaves, max_level)
            selected = {
                neighbour
                for leaf in leaves
                for neighbour in lookup.neighbours(leaf)
                if leaves[neighbour][4] + 1 < leaf[4]
            }
            if not selected:
                return leaves
            leaves = refine_selected_leaves(leaves, selected, max_level, classify)

    monkeypatch.setattr(template, "balance_leaves", original_balance)
    expected = mesher.build(stop_after="templateGeneration")
    assert actual.keys() == expected.keys()
    for key in actual:
        if isinstance(actual[key], np.ndarray):
            np.testing.assert_array_equal(actual[key], expected[key], err_msg=key)
        elif key == "faces":
            assert len(actual[key]) == len(expected[key])
            for first, second in zip(actual[key], expected[key], strict=True):
                np.testing.assert_array_equal(first, second)
        else:
            assert actual[key] == expected[key], key
