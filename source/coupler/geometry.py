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
        self.revision = hashlib.blake2b(
            np.ascontiguousarray(triangles).tobytes()
            + np.ascontiguousarray(domain_bounds, dtype=np.float64).tobytes(),
            digest_size=16,
        ).hexdigest()
        self.verified_cylinder_z = self._verify_extruded_cylinder_z(
            triangles, np.asarray(domain_bounds, dtype=np.float64)
        )
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
        self.surface_bounds = np.asarray(self._surface.GetBounds(), dtype=np.float64)
        self._distance = vtkImplicitPolyDataDistance()
        self._distance.SetInput(self._surface)
        self._bounds = np.asarray(domain_bounds, dtype=np.float64).reshape(6).copy()
        if not np.all(np.isfinite(self._bounds)) or np.any(self._bounds[1::2] <= self._bounds[::2]):
            raise ValueError("Wall domain bounds must be finite and increasing")
        body_scale = float(np.max(self.surface_bounds[1::2] - self.surface_bounds[::2]))
        self.interior_tolerance = 16.0 * np.finfo(np.float32).eps * max(body_scale, 1.0e-6)
        self._cache: OrderedDict[bytes, np.ndarray] = OrderedDict()

    @staticmethod
    def _verify_extruded_cylinder_z(triangles: np.ndarray, domain_bounds: np.ndarray):
        """Verify an open z-extruded circular wall from surface vertices.

        This is a geometric check, independent of patch names or a wall AABB.
        A candidate must cover the circumference and have no axial cap faces.
        """
        vertices = np.unique(triangles.reshape(-1, 3), axis=0)
        xy = np.unique(vertices[:, :2], axis=0)
        if len(xy) < 12 or np.ptp(vertices[:, 2]) <= 0.0:
            return None
        span = float(domain_bounds[5] - domain_bounds[4])
        axial_tolerance = max(1.0e-7, 2.0e-4 * span)
        if (
            abs(float(vertices[:, 2].min()) - float(domain_bounds[4])) > axial_tolerance
            or abs(float(vertices[:, 2].max()) - float(domain_bounds[5])) > axial_tolerance
        ):
            return None
        matrix = np.column_stack((2.0 * xy, np.ones(len(xy))))
        if np.linalg.matrix_rank(matrix) < 3:
            return None
        cx, cy, intercept = np.linalg.lstsq(matrix, np.sum(xy * xy, axis=1), rcond=None)[0]
        radius = float(np.sqrt(max(0.0, intercept + cx * cx + cy * cy)))
        if radius <= 0.0:
            return None
        radii = np.linalg.norm(xy - [cx, cy], axis=1)
        tolerance = max(1.0e-7, 0.002 * radius)
        if float(np.max(np.abs(radii - radius))) > tolerance:
            return None
        angles = np.sort(np.mod(np.arctan2(xy[:, 1] - cy, xy[:, 0] - cx), 2.0 * np.pi))
        if float(np.max(np.diff(np.r_[angles, angles[0] + 2.0 * np.pi]))) > 0.5:
            return None
        normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        magnitude = np.linalg.norm(normals, axis=1)
        valid = magnitude > 0.0
        if not np.any(valid) or np.any(np.abs(normals[valid, 2]) / magnitude[valid] > 0.01):
            return None
        return (float(cx), float(cy), radius, tolerance)

    def signed_distance(self, points: np.ndarray) -> np.ndarray:
        from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy

        query = np.ascontiguousarray(points, dtype=np.float64).reshape(-1, 3)
        if not np.all(np.isfinite(query)):
            raise ValueError("Wall queries must be finite")
        if self.verified_cylinder_z is not None:
            cx, cy, radius, _tolerance = self.verified_cylinder_z
            return np.linalg.norm(query[:, :2] - [cx, cy], axis=1) - radius
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
        return (
            distance <= self.interior_tolerance
            if include_boundary
            else distance < -self.interior_tolerance
        )
