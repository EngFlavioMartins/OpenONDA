"""Solid classification from the actual body-fitted FVM wall faces."""

from __future__ import annotations

from collections import OrderedDict
import hashlib

import numpy as np


class TriangulatedWall:
    """Signed wall distance, positive in fluid, from oriented FVM triangles.

    Native face normals point out of the fluid. Reverse wall faces for
    VTK's outward-from-solid convention. Open walls crossing a box boundary
    are only classified inside that box: no finite cap is invented for a
    spanwise-extruded cylinder. Coordinates/distances are metres; the mesh
    must be static and its wall faces consistently oriented.
    """

    def __init__(self, triangles: np.ndarray, domain_bounds: np.ndarray) -> None:
        from vtkmodules.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray
        from vtkmodules.vtkCommonCore import vtkPoints
        from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData
        from vtkmodules.vtkFiltersCore import vtkImplicitPolyDataDistance

        triangles = np.asarray(triangles, dtype=np.float64).reshape(-1, 3, 3)
        if not len(triangles) or not np.all(np.isfinite(triangles)):
            raise ValueError("Wall triangles must be nonempty and finite")
        points, connectivity = np.unique(
            triangles[:, ::-1].reshape(-1, 3), axis=0, return_inverse=True
        )
        vertices = vtkPoints()
        vertices.SetData(numpy_to_vtk(points, deep=True))
        faces = vtkCellArray()
        faces.SetData(
            numpy_to_vtkIdTypeArray(
                np.arange(0, connectivity.size + 1, 3, dtype=np.int64), deep=True
            ),
            numpy_to_vtkIdTypeArray(connectivity.astype(np.int64), deep=True),
        )
        self._surface = vtkPolyData()
        self._surface.SetPoints(vertices)
        self._surface.SetPolys(faces)
        self._distance = vtkImplicitPolyDataDistance()
        self._distance.SetInput(self._surface)
        self._bounds = np.asarray(domain_bounds, dtype=np.float64).reshape(6).copy()
        if not np.all(np.isfinite(self._bounds)) or np.any(self._bounds[1::2] <= self._bounds[::2]):
            raise ValueError("Wall domain bounds must be finite and increasing")
        self._cache: OrderedDict[bytes, np.ndarray] = OrderedDict()

    def signed_distance(self, points: np.ndarray) -> np.ndarray:
        from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy

        query = np.ascontiguousarray(points, dtype=np.float64).reshape(-1, 3)
        if not np.all(np.isfinite(query)):
            raise ValueError("Wall queries must be finite")
        key = hashlib.blake2b(query.tobytes(), digest_size=16).digest()
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key].copy()
        result = np.full(len(query), np.inf)
        inside_box = np.all((query >= self._bounds[::2]) & (query <= self._bounds[1::2]), axis=1)
        if np.any(inside_box):
            values = numpy_to_vtk(np.empty(int(inside_box.sum()), dtype=np.float64), deep=True)
            self._distance.FunctionValue(numpy_to_vtk(query[inside_box], deep=True), values)
            result[inside_box] = vtk_to_numpy(values)
        self._cache[key] = result
        while len(self._cache) > 8:
            self._cache.popitem(last=False)
        return result.copy()

    def contains(self, points: np.ndarray, *, include_boundary: bool = True) -> np.ndarray:
        distance = self.signed_distance(points)
        return distance <= 0.0 if include_boundary else distance < 0.0
