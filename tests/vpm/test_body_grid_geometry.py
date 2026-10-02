"""Production exact body-grid reuse, ownership and failure qualification."""

import numpy as np
import pytest
import taichi as ti

from source.coupler.geometry import SolidBoundary, TriangulatedWall
from source.solvers.vpm.physics.diffusion.grid import _GridDiffusionMixin


@ti.data_oriented
class Harness(_GridDiffusionMixin):
    def __init__(self):
        self._init_grid_diffusion()
        self._body_mask_grid = ti.field(ti.i32, shape=(16, 16, 16))
        self._body_link_grid = ti.field(ti.i32, shape=(16, 16, 16))


@pytest.fixture(scope="module", autouse=True)
def runtime():
    owned = ti.lang.impl.get_runtime().prog is None
    if owned:
        ti.init(arch=ti.cpu, cpu_max_num_threads=1, offline_cache=False)
    yield
    if owned:
        ti.reset()


def boundary(shift=0.0):
    return SolidBoundary((TriangulatedWall.from_box(
        (-0.35 + shift, 0.35 + shift, -0.35, 0.35, -0.35, 0.35), (-3, 3, -3, 3, -3, 3)),))


def configure(h, body, *, certified=True):
    h.configure_body_classifier(body.contains, revision=body.revision,
                                query_bounds=body.bounds, blocks_segments=body.blocks_segments,
                                geometry_cache_contract=body.grid_geometry_contract() if certified else None)


def snapshot(h, origin=(-0.7, -0.7, -0.7), shape=(9, 9, 9), spacing=0.15):
    h._prepare_body_mask_current_grid(np.asarray(origin), spacing, *shape)
    slices = tuple(slice(0, n) for n in shape)
    return (h._body_mask_grid.to_numpy()[slices], h._body_link_grid.to_numpy()[slices],
            h.body_geometry_cache_diagnostics)


def test_bound_default_matches_explicit_strict_interior_and_certifies():
    body = boundary()
    points = np.array([[0, 0, 0], [0.35, 0, 0], [0.36, 0, 0]])
    np.testing.assert_array_equal(body.contains(points), body.contains(points, include_boundary=False))
    np.testing.assert_array_equal(body.contains(points), [True, False, False])
    contract = body.grid_geometry_contract()
    assert contract is not None
    assert contract.key(body.contains, body.blocks_segments, body.revision) is not None


def test_growing_grid_matches_original_fresh_device_fields_bitwise():
    body = boundary()
    candidate, original = Harness(), Harness()
    configure(candidate, body)
    configure(original, body, certified=False)
    for shape in ((9, 9, 9), (11, 10, 9), (10, 8, 9)):
        a, b, stats = snapshot(candidate, shape=shape)
        x, y, _ = snapshot(original, shape=shape)
        np.testing.assert_array_equal(a, x)
        np.testing.assert_array_equal(b, y)
        assert stats["qualified"] and stats["status"] == "complete"
    assert stats["membership_reused"] == 10 * 8 * 9
    assert stats["membership_queried"] == 0


@pytest.mark.parametrize("field", ["_body_mask_grid", "_body_link_grid"])
def test_device_field_replacement_keeps_host_evidence_but_forces_upload(field):
    h = Harness()
    configure(h, boundary())
    expected = snapshot(h)
    setattr(h, field, ti.field(ti.i32, shape=(16, 16, 16)))
    actual = snapshot(h)
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])
    assert actual[2]["status"] == "complete"
    assert actual[2]["membership_queried"] == actual[2]["links_queried"] == 0


def test_initially_absent_link_storage_publishes_the_allocated_field_identity():
    h = Harness()
    h._body_link_grid = None
    configure(h, boundary())
    expected = snapshot(h)
    assert h._body_link_grid is not None
    actual = snapshot(h)
    assert actual[2]["status"] == "resident_hit"
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])


def test_failed_partial_upload_never_publishes_residency_or_host_state():
    h = Harness()
    configure(h, boundary())
    upload = h._upload_scalar_chunk_kernel
    calls = []

    def fail_second(*args):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("injected partial link upload")
        return upload(*args)

    h._upload_scalar_chunk_kernel = fail_second
    with pytest.raises(RuntimeError, match="partial link upload"):
        snapshot(h)
    assert len(calls) == 2
    assert h._body_mask_cache_key is h._body_mask_host is h._body_links_host is None
    assert h.body_geometry_cache_diagnostics["status"] == "failed"
    del h._upload_scalar_chunk_kernel
    a, b, stats = snapshot(h)
    fresh = Harness()
    configure(fresh, boundary(), certified=False)
    expected = snapshot(fresh)
    np.testing.assert_array_equal(a, expected[0])
    np.testing.assert_array_equal(b, expected[1])
    assert stats["status"] == "complete"


@pytest.mark.parametrize("mutation", ["points", "normals", "bounds", "tolerance", "topology"])
def test_changed_static_geometry_fails_before_reusing_or_uploading(mutation):
    from vtkmodules.util.numpy_support import vtk_to_numpy

    h, body = Harness(), boundary()
    configure(h, body)
    snapshot(h)
    wall = body.bodies[0]
    if mutation == "points":
        vtk_to_numpy(wall._surface.GetPoints().GetData())[0, 0] += 0.01
    elif mutation == "normals":
        wall._normals[0, 0] += 0.1
    elif mutation == "bounds":
        wall._bounds[0] += 0.1
    elif mutation == "topology":
        # VTK may share an empty cell-array singleton. Replace it with a new
        # owned array rather than mutating GetLines() and corrupting other walls.
        from vtkmodules.vtkCommonDataModel import vtkCellArray

        lines = vtkCellArray()
        lines.InsertNextCell(2, [0, 1])
        wall._surface.SetLines(lines)
    else:
        wall.interior_tolerance *= 2
    with pytest.raises(RuntimeError, match="static body geometry changed"):
        snapshot(h)
    assert h._body_mask_cache_key is h._body_mask_host is h._body_links_host is None
    configure(h, boundary(shift=0.1))
    assert snapshot(h)[2]["status"] == "complete"


def test_harmless_vtk_modified_declines_reuse_without_claiming_geometry_changed():
    h, body = Harness(), boundary()
    configure(h, body)
    expected = snapshot(h)
    body.bodies[0]._distance.Modified()
    actual = snapshot(h)
    assert actual[2]["status"] == "fresh_fallback"
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])
    assert h._body_mask_cache_key is None


def test_changed_callbacks_and_physics_overrides_decline_incremental_reuse():
    h, body = Harness(), boundary()
    configure(h, body)
    snapshot(h)
    h._body_classifier = lambda p: np.zeros(len(p), dtype=bool)
    actual = snapshot(h, shape=(10, 9, 9))
    assert actual[2]["status"] == "fresh_fallback"
    assert not actual[0].any()
    configure(h, body)
    original = h._body_interior_at_particles
    h._body_interior_at_particles = lambda p: original(p)
    assert snapshot(h)[2]["status"] == "fresh_fallback"


def test_custom_body_and_callback_do_not_acquire_implicit_capability():
    class CustomWall:
        revision = "arbitrary"
        surface_bounds = np.array([-1, 1, -1, 1, -1, 1])

    assert SolidBoundary((CustomWall(),)).grid_geometry_contract() is None
    h = Harness()
    h.configure_body_classifier(lambda p: np.zeros(len(p), dtype=bool), revision="custom")
    assert snapshot(h)[2]["status"] == "fresh_fallback"


@pytest.mark.parametrize("change", ["slab", "spacing", "signed_zero"])
def test_slab_full_spacing_and_origin_bits_invalidate_residency(change):
    h, body = Harness(), boundary()
    configure(h, body)
    snapshot(h, origin=(0.0, -0.7, -0.7))
    kw = {"origin": (0.0, -0.7, -0.7)}
    if change == "slab":
        h._slip_slab_bounds = (-0.5, 0.5)
    elif change == "spacing":
        kw["spacing"] = np.nextafter(0.15, 1.0)
    else:
        kw["origin"] = (-0.0, -0.7, -0.7)
    result = snapshot(h, **kw)
    assert result[2]["status"] == "complete"
    if change != "signed_zero":
        assert result[2]["membership_reused"] == 0


def test_payload_cap_and_mutated_returned_host_arrays_are_safe():
    h = Harness()
    configure(h, boundary())
    snapshot(h)
    cache = h._body_grid_geometry_cache
    assert cache.record.nbytes <= cache.max_bytes
    h._body_mask_host[:] = False
    h._body_links_host[:] = 7
    result = snapshot(h, shape=(10, 9, 9))
    fresh = Harness()
    configure(fresh, boundary(), certified=False)
    expected = snapshot(fresh, shape=(10, 9, 9))
    np.testing.assert_array_equal(result[0], expected[0])
    np.testing.assert_array_equal(result[1], expected[1])
    cache.max_bytes = 0
    assert cache.record is None
    assert snapshot(h, shape=(11, 9, 9))[2]["status"] == "fresh_fallback"
