"""Unwired same-VTK/owned-thread qualification; no global SMP changes.

An explicitly prebuilt test extension is required. No runtime compilation,
solver initialization, module installation or production dispatch occurs here.
"""

from concurrent.futures import ThreadPoolExecutor
import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="module")
def native():
    path = os.environ.get("OPENONDA_TEST_NATIVE_WALL_EXTENSION")
    if not path:
        pytest.skip("separately built native wall prototype required")
    spec = importlib.util.spec_from_file_location("_native_wall_batch", Path(path).resolve(strict=True))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def wall(kind):
    from vtkmodules.vtkFiltersCore import vtkImplicitPolyDataDistance, vtkTriangleFilter
    from vtkmodules.vtkFiltersSources import vtkCubeSource, vtkPlaneSource

    if kind == "open":
        source = vtkPlaneSource()
        source.SetOrigin(0., -1., -1.)
        source.SetPoint1(0., 1., -1.)
        source.SetPoint2(0., -1., 1.)
    else:
        source = vtkCubeSource()
        source.SetBounds(-1.e-5 if kind == "thin" else -.5,
                         1.e-5 if kind == "thin" else .5, -.5, .5, -.5, .5)
    source.Update()
    triangles = vtkTriangleFilter()
    triangles.SetInputData(source.GetOutput())
    triangles.Update()
    surface = triangles.GetOutput()
    distance = vtkImplicitPolyDataDistance()
    distance.SetInput(surface)
    return surface, distance


def reference(distance, points):
    from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy

    output = numpy_to_vtk(np.empty(len(points), np.float64), deep=True)
    distance.FunctionValue(numpy_to_vtk(np.ascontiguousarray(points), deep=True), output)
    return vtk_to_numpy(output).copy()


def query(native, owner, points, *args):
    return np.frombuffer(native.evaluate(owner, points, *args), dtype=np.float64)


def exact(first, second):
    np.testing.assert_array_equal(first.view(np.uint64), second.view(np.uint64))


@pytest.mark.parametrize("kind", ["closed", "thin", "open"])
@pytest.mark.parametrize("workers", [1, 2, 4])
def test_bitwise_point_distance_and_unchanged_global_smp(native, kind, workers):
    from vtkmodules.vtkCommonCore import vtkSMPTools

    surface, distance = wall(kind)
    edge = np.array([[x, y, z] for x in (-.5, -1.e-5, -0., 0., 1.e-5, .5)
                     for y in (-.5, 0., .5) for z in (-.5, 0., .5)], np.float64)
    points = np.vstack((edge, np.nextafter(edge, np.inf), np.nextafter(edge, -np.inf),
                        np.random.default_rng(2901).uniform(-1., 1., (2000, 3))))
    before = (vtkSMPTools.GetBackend(), vtkSMPTools.GetEstimatedNumberOfThreads(),
              vtkSMPTools.GetNestedParallelism())
    owner = native.create(surface, distance, workers, len(points), 100)
    try:
        info = native.metadata(owner)
        assert info["vtk_version"] == "9.6.2"
        assert 1 <= info["workers"] <= min(workers, len(os.sched_getaffinity(0)))
        expected = reference(distance, points)
        for _ in range(2):
            result = query(native, owner, points)
            exact(result, expected)
            assert not result.flags.writeable
        assert before == (vtkSMPTools.GetBackend(), vtkSMPTools.GetEstimatedNumberOfThreads(),
                          vtkSMPTools.GetNestedParallelism())
    finally:
        native.close(owner)


def test_private_geometry_and_query_input_ownership(native):
    surface, distance = wall("closed")
    points = np.array([[.6, .1, .1], [.2, .1, .1], [-.5, .1, .1]])
    owner = native.create(surface, distance, 2, 10, 100)
    expected = query(native, owner, points)
    saved = expected.copy()
    for i in range(surface.GetNumberOfPoints()):
        x, y, z = surface.GetPoint(i)
        surface.GetPoints().SetPoint(i, x+5., y, z)
    surface.GetPoints().Modified()
    surface.Modified()
    try:
        exact(query(native, owner, points), saved)
        shifted = points.copy()
        shifted[:, 0] += 3.
        assert not np.array_equal(query(native, owner, shifted), expected)
        exact(expected, saved)  # Later work cannot overwrite prior output.
        assert points[0, 0] == .6
    finally:
        native.close(owner)


def test_failed_worker_joins_before_retry_and_publication(native):
    surface, distance = wall("closed")
    owner = native.create(surface, distance, 4, 100, 100)
    points = np.random.default_rng(31).uniform(-1., 1., (100, 3))
    expected = reference(distance, points)
    try:
        with pytest.raises(RuntimeError, match="injected worker failure"):
            native.evaluate(owner, points, 0)
        exact(query(native, owner, points), expected)
    finally:
        native.close(owner)


def test_input_limits_empty_and_idempotent_close(native):
    surface, distance = wall("closed")
    owner = native.create(surface, distance, 2, 3, 100)
    points = np.zeros((3, 3))
    for bad in (np.zeros((4, 3)), points.astype(np.float32), points[:, ::-1],
                np.array([[np.nan, 0., 0.]])):
        with pytest.raises(ValueError):
            native.evaluate(owner, bad)
    assert query(native, owner, np.empty((0, 3))).shape == (0,)
    native.close(owner)
    native.close(owner)
    assert native.metadata(owner)["closed"]
    with pytest.raises(RuntimeError, match="closed"):
        native.evaluate(owner, points)


def test_other_python_thread_cannot_query_or_close(native):
    surface, distance = wall("closed")
    owner = native.create(surface, distance, 2, 10, 100)
    try:
        with ThreadPoolExecutor(1) as pool:
            for future in (pool.submit(native.evaluate, owner, np.zeros((2, 3))),
                           pool.submit(native.close, owner)):
                with pytest.raises(RuntimeError, match="another Python thread"):
                    future.result()
        assert np.isfinite(query(native, owner, np.zeros((2, 3)))).all()
    finally:
        native.close(owner)


def test_nonstandard_transform_and_limits_decline(native):
    from vtkmodules.vtkCommonTransforms import vtkTransform

    surface, distance = wall("closed")
    for workers, points, triangles in ((0, 1, 100), (65, 1, 100), (1, 0, 100),
                                      (1, 1000001, 100), (1, 1, 1)):
        with pytest.raises(ValueError):
            native.create(surface, distance, workers, points, triangles)
    distance.SetTransform(vtkTransform())
    with pytest.raises(ValueError, match="without transform"):
        native.create(surface, distance, 1, 10, 100)
