"""Explicit source-only whole-field qualification; not a particle-stage test."""

import numpy as np
import pytest

from tests.vpm._cupy_coherent_source_targets import CoherentSourceTargetsGPU


def _case():
    x = np.array([[-.082, .012, 0.], [-.042, .012, .023], [-.002, .012, .083],
                  [.038, -.012, .133], [.078, .012, .193]])
    gamma = np.array([[.3, -.2, .7], [-.7, .1, .3], [.4, -.5, -.2],
                      [-.11, .47, -.31], [.3, -.2, .7]])
    core = np.array([.035, .04, .055, .045, .038])
    wall = np.array([[.011, .019, 0.], [-.06, -.027, 0.],
                     [.011, .019, .193], [-.06, -.027, .193]])
    generic = np.array([[.012, .014, .024], [-.068, .053, .061], [.097, -.036, .04]])
    q = np.vstack((wall, generic, x[[0, 1, 4]]))
    images = [(k, odd) for k in range(-2, 3) for odd in (False, True) if k != 0 or odd]
    return x, gamma, core, q, images


def test_primary_permission_is_required_before_any_runtime_import():
    for flag in (False, None, 1, "true"):
        with pytest.raises(ValueError, match="explicit"):
            CoherentSourceTargetsGPU(None, None, None, None, include_physical_primary=flag)


def test_primary_descriptor_is_inserted_once_with_bounded_snapshot():
    owner = object.__new__(CoherentSourceTargetsGPU)
    owner.max_images, owner.zmin, owner.zmax, owner.cells = 4, 0., .193, 6
    images = [(0, True), (-1, False), (1, True)]
    assert owner._whole_descriptors(images) == ((0, False), *images)
    with pytest.raises(ValueError, match="inserted exactly once"):
        owner._whole_descriptors([(0, False)])
    visited = []

    def generator():
        for k in range(4):
            visited.append(k)
            yield k, True
        raise AssertionError("cannot consume beyond cap")

    with pytest.raises(ValueError, match="capacity"):
        owner._whole_descriptors(generator())
    assert visited == list(range(4))


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("spacing", [.035, .03])
def test_coherent_mixed_cores_walls_offgrid_and_self_j_match_direct(dtype, spacing):
    cp = pytest.importorskip("cupy")
    from tests.vpm._finite_image_mesh_reference import direct_finite_images
    from tests.vpm._finite_slab_field_mesh_reference import slab_world_images

    x, gamma, core, q, images = _case()
    originals = tuple(a.copy() for a in (x, gamma, core, q))
    world, _ = slab_world_images([(0, False), *images], 0., .193, int(np.ceil(.193/spacing)), 514)
    exact_u, exact_j, conditioning = direct_finite_images(x, gamma, core, q, world)
    with CoherentSourceTargetsGPU(x, gamma, core, q, zmin=0., zmax=.193,
                                  include_physical_primary=True, dtype=dtype,
                                  correction_dtype=dtype, spacing=spacing,
                                  stencil_backend="gpu") as owner:
        preparation = owner.prepare(images)
        u, j, diagnostics = owner.evaluate_prepared(q)
        actual_u, actual_j = cp.asnumpy(u), cp.asnumpy(j)
        assert preparation["primary_included"] and preparation["primary_core_contract"] == "source-only"
        assert diagnostics["finite_term_count"] == len(images)+1
        assert diagnostics["image_count"] == len(images)
        assert not diagnostics["particle_stage_supported"] and not diagnostics["tail_certified"]
        assert diagnostics["source_scatters"] == diagnostics["inverse_transforms"] == 0
        assert not cp.shares_memory(u, owner._compact_fields)
        # New queries cannot alter old returned allocations or owned sources.
        old_u, old_j = actual_u.copy(), actual_j.copy()
        u2, j2, _ = owner.evaluate_prepared(q[::-1])
        np.testing.assert_array_equal(cp.asnumpy(u), old_u)
        np.testing.assert_array_equal(cp.asnumpy(j), old_j)
        np.testing.assert_array_equal(cp.asnumpy(u2)[::-1], old_u)
        np.testing.assert_array_equal(cp.asnumpy(j2)[::-1], old_j)
        del u, j, u2, j2
    for actual, exact in ((actual_u, exact_u), (actual_j, exact_j)):
        assert np.linalg.norm(actual-exact)/np.linalg.norm(exact) < 1e-4
        per_point = np.linalg.norm((actual-exact).reshape(len(q), -1), axis=1)
        exact_norm = np.linalg.norm(exact.reshape(len(q), -1), axis=1)
        assert np.all(per_point <= 1e-4*exact_norm+32*np.finfo(dtype).eps)
    # Both slab planes and coincident wall sources are independently checked.
    # Finite upper normal is not identically zero: its two end images remain.
    wall_rows = np.array([0, 1, 2, 3, 7, 9])
    wall_error = np.abs(actual_u[wall_rows, 2]-exact_u[wall_rows, 2])
    wall_allowance = (32*np.finfo(dtype).eps*conditioning[wall_rows, 0]
                      +1e-4*np.abs(exact_u[wall_rows, 2]))
    assert np.all(wall_error <= wall_allowance), (wall_error, wall_allowance)
    assert np.isfinite(actual_j).all() and np.linalg.norm(actual_j[8]) > 1.
    for actual, expected in zip((x, gamma, core, q), originals, strict=True):
        np.testing.assert_array_equal(actual, expected)


def test_preserved_image_only_owner_still_rejects_physical_primary():
    pytest.importorskip("cupy")
    from tests.vpm._cupy_slab_field_mesh_fused import SlabFieldMeshFusedGPU

    x, gamma, _, q, _ = _case()
    with (SlabFieldMeshFusedGPU(x, gamma, q, zmin=0., zmax=.193) as owner,
          pytest.raises(ValueError, match="physical primary excluded")):
        owner.evaluate([(0, False)])
