"""Solid classification from the actual body-fitted FVM wall faces."""

from __future__ import annotations

from collections import OrderedDict
import hashlib
import inspect

import numpy as np

from source.solvers.vpm.wall_errors import WallCorrectionTooLargeError

_DISTANCE_CACHE_MAX_BYTES = 8 * 1024 * 1024


class TriangulatedWall:
    """Signed wall distance, positive in fluid, from oriented FVM triangles.

    Native face normals point out of the fluid. Reverse wall faces for
    VTK's outward-from-solid convention. Open surfaces crossing a domain
    boundary are only classified inside that domain: their continuation
    or closure is not inferred. Coordinates/distances are metres; the mesh
    must be static and its wall faces consistently oriented.
    """

    def __init__(self, triangles: np.ndarray, domain_bounds: np.ndarray) -> None:
        from vtkmodules.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray
        from vtkmodules.vtkCommonCore import vtkPoints
        from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData, vtkStaticCellLocator
        from vtkmodules.vtkFiltersCore import vtkImplicitPolyDataDistance

        triangles = np.asarray(triangles, dtype=np.float64).reshape(-1, 3, 3)
        if not len(triangles) or not np.all(np.isfinite(triangles)):
            raise ValueError("Wall triangles must be nonempty and finite")
        self.revision = hashlib.blake2b(
            np.ascontiguousarray(triangles).tobytes()
            + np.ascontiguousarray(domain_bounds, dtype=np.float64).tobytes(),
            digest_size=16,
        ).hexdigest()
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
        self._locator = vtkStaticCellLocator()
        self._locator.SetDataSet(self._surface)
        self._locator.BuildLocator()
        normals = np.cross(triangles[:, 2] - triangles[:, 0], triangles[:, 1] - triangles[:, 0])
        lengths = np.linalg.norm(normals, axis=1)
        if np.any(lengths == 0.0):
            raise ValueError("Wall triangles must have nonzero area")
        self._normals = normals / lengths[:, None]
        self._bounds = np.asarray(domain_bounds, dtype=np.float64).reshape(6).copy()
        if not np.all(np.isfinite(self._bounds)) or np.any(self._bounds[1::2] <= self._bounds[::2]):
            raise ValueError("Wall domain bounds must be finite and increasing")
        body_scale = float(np.max(self.surface_bounds[1::2] - self.surface_bounds[::2]))
        # Surface coordinates and VTK distance queries are float64. Particle
        # storage precision is handled by the projection margin, not by
        # erasing a float32-sized band from every solid interior.
        coordinate_scale = float(np.max(np.abs(triangles)))
        self.interior_tolerance = (
            16.0 * np.finfo(np.float64).eps * max(body_scale, coordinate_scale, 1.0e-6)
        )
        self._cache: OrderedDict[bytes, np.ndarray] = OrderedDict()
        self._cache_bytes = 0

    @classmethod
    def from_box(cls, bounds, domain_bounds):
        """Represent a box through the same surface-query interface as any wall."""
        lower, upper = np.asarray(bounds)[::2], np.asarray(bounds)[1::2]
        vertices = np.array(
            [
                [x, y, z]
                for x in (lower[0], upper[0])
                for y in (lower[1], upper[1])
                for z in (lower[2], upper[2])
            ]
        )
        faces = ((0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3))
        # Faces above point into the solid, like native FVM wall faces.
        triangles = np.array(
            [vertices[[a, c, b]] for a, b, c, d in faces]
            + [vertices[[a, d, c]] for a, b, c, d in faces]
        )
        return cls(triangles, domain_bounds)

    def closest_surface(self, points):
        """Return nearest surface points and normals directed into the fluid."""
        query = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        closest = np.empty_like(query)
        normals = np.empty_like(query)
        for index, point in enumerate(query):
            distance = self._distance.EvaluateFunctionAndGetClosestPoint(point, closest[index])
            delta = point - closest[index]
            length = np.linalg.norm(delta)
            if length > 1e-14:
                normals[index] = delta / length * (1.0 if distance >= 0 else -1.0)
            else:
                self._distance.EvaluateGradient(point, normals[index])
                normals[index] /= np.linalg.norm(normals[index])
        return closest, normals

    def first_intersections(self, starts, ends):
        """First entry into the solid along each segment, including thin solids.

        The surface locator tests actual triangles. Tangency and exit faces
        do not count as entries; no shape fit or sampling interval is used.
        """
        from vtkmodules.vtkCommonCore import vtkIdList, vtkPoints
        from vtkmodules.vtkCommonDataModel import vtkGenericCell

        starts = np.asarray(starts, dtype=np.float64).reshape(-1, 3)
        ends = np.asarray(ends, dtype=np.float64).reshape(-1, 3)
        delta = ends - starts
        lengths = np.linalg.norm(delta, axis=1)
        fraction = np.full(len(starts), np.inf)
        normals = np.zeros_like(starts)
        bounds = self.surface_bounds
        candidates = np.flatnonzero(
            (lengths > 0)
            & np.all(np.maximum(starts, ends) >= bounds[::2], axis=1)
            & np.all(np.minimum(starts, ends) <= bounds[1::2], axis=1)
        )
        if len(candidates):
            # Signed distance is a conservative broad phase for a segment.
            candidate_starts = starts[candidates]
            new_start = np.ones(len(candidates), dtype=bool)
            new_start[1:] = np.any(candidate_starts[1:] != candidate_starts[:-1], axis=1)
            if np.all(new_start):
                distance = self.signed_distance(candidate_starts)
            else:
                # Stencils repeat a source for each target. Evaluate each
                # adjacent run once without sorting or changing query order.
                run_index = np.cumsum(new_start, dtype=np.intp) - 1
                distance = self.signed_distance(candidate_starts[new_start])[run_index]
            candidates = candidates[
                (distance <= lengths[candidates] + self.interior_tolerance) | ~np.isfinite(distance)
            ]
        hits, cells, cell = vtkPoints(), vtkIdList(), vtkGenericCell()
        hits.SetDataTypeToDouble()
        tolerance = max(1e-12, self.interior_tolerance * 1e-3)
        for index in candidates:
            hits.Reset()
            cells.Reset()
            self._locator.IntersectWithLine(
                starts[index], ends[index], tolerance, hits, cells, cell
            )
            for hit in range(cells.GetNumberOfIds()):
                normal = self._normals[cells.GetId(hit)]
                if np.dot(delta[index], normal) >= -1e-12 * lengths[index]:
                    continue
                point = np.asarray(hits.GetPoint(hit))
                time = float(np.dot(point - starts[index], delta[index]) / lengths[index] ** 2)
                if -1e-10 <= time < fraction[index] and time <= 1.0 + 1e-10:
                    fraction[index] = np.clip(time, 0, 1)
                    normals[index] = normal
        return fraction, normals

    def signed_distance(self, points: np.ndarray) -> np.ndarray:
        from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy

        query = np.ascontiguousarray(points, dtype=np.float64).reshape(-1, 3)
        if not np.all(np.isfinite(query)):
            raise ValueError("Wall queries must be finite")
        key = hashlib.blake2b(query.view(np.uint8), digest_size=16).digest()
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key].copy()
        result = np.full(len(query), np.inf)
        inside_box = np.all((query >= self._bounds[::2]) & (query <= self._bounds[1::2]), axis=1)
        if np.any(inside_box):
            values = numpy_to_vtk(np.empty(int(inside_box.sum()), dtype=np.float64), deep=True)
            self._distance.FunctionValue(numpy_to_vtk(query[inside_box], deep=True), values)
            result[inside_box] = vtk_to_numpy(values)
        # Moving RK queries rarely repeat. Bound retained storage independently
        # of particle count, while keeping small repeated grid queries useful.
        if result.nbytes > _DISTANCE_CACHE_MAX_BYTES:
            return result
        self._cache[key] = result
        self._cache_bytes += result.nbytes
        while len(self._cache) > 8 or self._cache_bytes > _DISTANCE_CACHE_MAX_BYTES:
            _, evicted = self._cache.popitem(last=False)
            self._cache_bytes -= evicted.nbytes
        return result.copy()

    def contains(self, points: np.ndarray, *, include_boundary: bool = True) -> np.ndarray:
        distance = self.signed_distance(points)
        return (
            distance <= self.interior_tolerance
            if include_boundary
            else distance < -self.interior_tolerance
        )


class SolidBoundary:
    """One geometric description for transfer, particle motion and grid diffusion.

    Bodies supply signed distance and membership. Native triangulated walls
    also supply exact nearest points and line intersections. Distance-based
    immersed bodies use conservative advancement with their own distance
    function; ambiguous intersections fail instead of leaking through a wall.
    """

    def __init__(self, bodies):
        self.bodies = tuple(bodies)
        bounds = [getattr(body, "surface_bounds", None) for body in self.bodies]
        bounds = [
            value if value is not None else getattr(body, "solid_bounds", None)
            for body, value in zip(self.bodies, bounds, strict=True)
        ]
        self.bounds = None
        if bounds and all(value is not None for value in bounds):
            values = np.asarray(bounds, dtype=np.float64)
            self.bounds = np.empty(6)
            self.bounds[::2] = values[:, ::2].min(axis=0)
            self.bounds[1::2] = values[:, 1::2].max(axis=0)
            if not np.all(np.isfinite(self.bounds)):
                self.bounds = None  # Unbounded extrusions cannot use a finite query box.
        self.tolerance = max(
            (getattr(body, "interior_tolerance", 0.0) for body in self.bodies), default=0.0
        )

    @property
    def revision(self):
        return tuple(body.revision for body in self.bodies)

    def signed_distance(self, points):
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        result = np.full(len(points), np.inf)
        for body in self.bodies:
            result = np.minimum(result, body.signed_distance(points))
        return result

    def contains(self, points, *, include_boundary=False):
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        result = np.zeros(len(points), dtype=bool)
        for body in self.bodies:
            result |= body.contains(points, include_boundary=include_boundary)
        return result

    def grid_geometry_queries(self):
        """Verify standard static wall queries for exact diffusion-grid reuse.

        No fitted geometry or bounding-box classification is introduced. The
        signature covers the current predicate implementation and every array
        or VTK input controlling it, including mutable data behind the revision
        label. Unknown body implementations retain ordinary uncached queries.
        """
        from source.solvers.vpm.physics.diffusion.body_geometry import ImmutableBodyGeometryQueries

        geometry_signature = self._grid_geometry_signature()
        if geometry_signature is None:
            return None
        return ImmutableBodyGeometryQueries(
            self.contains,
            self.blocks_segments,
            self._grid_geometry_signature,
            geometry_signature[1],
            geometry_signature[2],
        )

    def _grid_geometry_signature(self):
        from vtkmodules.util.numpy_support import vtk_to_numpy
        from vtkmodules.vtkCommonDataModel import vtkPolyData, vtkStaticCellLocator
        from vtkmodules.vtkFiltersCore import vtkImplicitPolyDataDistance

        if (
            (SolidBoundary, TriangulatedWall) != _GRID_GEOMETRY_CLASSES
            or type(self) is not SolidBoundary
            or type(self.bodies) is not tuple
            or not self.bodies
        ):
            return None
        for instance, cls in (
            (self, SolidBoundary),
            *((body, TriangulatedWall) for body in self.bodies),
        ):
            if type(instance) is not cls:
                return None
            for name, method in _GRID_GEOMETRY_METHODS[cls].items():
                if name in vars(instance) or inspect.getattr_static(cls, name, None) is not method:
                    return None
        digest = hashlib.sha256()
        runtime = []
        for body in self.bodies:
            if (
                type(body.revision) is not str
                or type(body._surface) is not vtkPolyData
                or type(body._distance) is not vtkImplicitPolyDataDistance
                or type(body._locator) is not vtkStaticCellLocator
                or body._locator.GetDataSet() is not body._surface
                or body._distance.GetTransform() is not None
            ):
                return None
            coordinates = body._surface.GetPoints().GetData()
            topology = []
            for cells in (
                body._surface.GetVerts(),
                body._surface.GetLines(),
                body._surface.GetPolys(),
                body._surface.GetStrips(),
            ):
                topology.extend(
                    (
                        vtk_to_numpy(cells.GetOffsetsArray()),
                        vtk_to_numpy(cells.GetConnectivityArray()),
                    )
                )
            arrays = (
                vtk_to_numpy(coordinates),
                *topology,
                body._normals,
                body._bounds,
                body.surface_bounds,
            )
            for array in arrays:
                array = np.asarray(array)
                if array.dtype.kind not in "fiu" or not np.all(np.isfinite(array)):
                    return None
                digest.update(str((array.dtype.str, array.shape)).encode())
                digest.update(np.ascontiguousarray(array).tobytes())
            tolerance = float(body.interior_tolerance)
            if not np.isfinite(tolerance) or tolerance < 0:
                return None
            digest.update(np.float64(tolerance).tobytes())
            for values in (
                body._distance.GetTolerance(),
                body._distance.GetNoValue(),
                body._distance.GetNoGradient(),
                body._distance.GetNoClosestPoint(),
            ):
                digest.update(np.asarray(values, dtype=np.float64).tobytes())
            digest.update(
                str((id(body), id(body._surface), id(body._distance), id(body._locator))).encode()
            )
            runtime.append((body._distance.GetMTime(), body._locator.GetMTime()))
        return self.revision, digest.digest(), tuple(runtime)

    @staticmethod
    def _closest_surface(body, points):
        method = getattr(body, "closest_surface", None)
        if method is not None:
            return method(points)
        points = np.asarray(points, dtype=np.float64)
        distance = body.signed_distance(points)
        epsilon = 1e-6 * max(1.0, float(np.max(np.abs(points), initial=0.0)))
        normals = np.column_stack(
            [
                (
                    body.signed_distance(points + epsilon * axis)
                    - body.signed_distance(points - epsilon * axis)
                )
                / (2 * epsilon)
                for axis in np.eye(3)
            ]
        )
        magnitude = np.linalg.norm(normals, axis=1)
        ambiguous = magnitude < 1e-8
        if np.any(ambiguous):
            directions = np.concatenate((np.eye(3), -np.eye(3)))
            probes = points[ambiguous, None, :] + epsilon * directions
            values = body.signed_distance(probes.reshape(-1, 3)).reshape(-1, 6)
            normals[ambiguous] = directions[np.argmax(values, axis=1)]
            magnitude[ambiguous] = 1.0
        normals /= magnitude[:, None]
        return points - distance[:, None] * normals, normals

    def closest_surface(self, points):
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        distances = np.stack([body.signed_distance(points) for body in self.bodies])
        owners = np.argmin(distances, axis=0)
        closest, normals = np.empty_like(points), np.empty_like(points)
        for index, body in enumerate(self.bodies):
            selected = owners == index
            if np.any(selected):
                closest[selected], normals[selected] = self._closest_surface(body, points[selected])
        return closest, normals

    @classmethod
    def _distance_intersections(cls, body, starts, ends):
        delta = ends - starts
        lengths = np.linalg.norm(delta, axis=1)
        fraction = np.full(len(starts), np.inf)
        normals = np.zeros_like(starts)
        progress = np.zeros(len(starts))
        scale = max(1.0, float(np.max(np.abs(starts), initial=0.0)))
        tolerance = 64 * np.finfo(float).eps * scale
        active = np.flatnonzero(lengths > tolerance)
        for _ in range(4096):
            if not len(active):
                return fraction, normals
            query = starts[active] + progress[active, None] * delta[active]
            distance = np.asarray(body.signed_distance(query))
            near = distance <= tolerance
            if np.any(near):
                _, normal = cls._closest_surface(body, query[near])
                rows = active[near]
                entering = np.einsum("ij,ij->i", normal, delta[rows]) < -1e-12 * lengths[rows]
                fraction[rows[entering]] = progress[rows[entering]]
                normals[rows[entering]] = normal[entering]
            progress[active] += np.maximum(distance, tolerance) / lengths[active]
            active = active[(progress[active] < 1.0) & ~np.isfinite(fraction[active])]
        raise RuntimeError(
            "Solid distance cannot resolve a segment; supply triangulated wall geometry"
        )

    def first_intersections(self, starts, ends):
        starts = np.asarray(starts, dtype=np.float64).reshape(-1, 3)
        ends = np.asarray(ends, dtype=np.float64).reshape(-1, 3)
        if starts.shape != ends.shape:
            raise ValueError("Wall segments need matching start and end arrays")
        fraction = np.full(len(starts), np.inf)
        normals = np.zeros_like(starts)
        for body in self.bodies:
            method = getattr(body, "first_intersections", None)
            hit, normal = (
                method(starts, ends)
                if method is not None
                else self._distance_intersections(body, starts, ends)
            )
            earlier = hit < fraction
            fraction[earlier], normals[earlier] = hit[earlier], normal[earlier]
        return fraction, normals

    def blocks_segments(self, starts, ends):
        """Whether a segment enters solid, even with two fluid endpoints."""
        return self.first_intersections(starts, ends)[0] < 1.0 - 1e-10

    def constrain_motion(self, positions, spacing, *, starts=None):
        """Project bounded wall crossings onto fluid, without changing strength.

        Segment queries stop tunnelling through thin walls. Successive local
        projections handle corners and multiple bodies; all corrected points
        and paths are verified against the complete geometry before return.
        """
        original = np.asarray(positions)
        corrected = np.asarray(positions, dtype=np.float64).copy()
        reference = None if starts is None else np.asarray(starts, dtype=np.float64).copy()
        scale = max(spacing, float(np.max(np.abs(corrected), initial=0.0)))
        margin = max(4 * self.tolerance, 32 * np.finfo(original.dtype).eps * scale, 1e-7 * spacing)
        for _ in range(12):
            hits = np.full(len(corrected), np.inf)
            if reference is not None:
                hits, normals = self.first_intersections(reference, corrected)
                crossing = np.isfinite(hits)
                if np.any(crossing):
                    contact = reference[crossing] + hits[crossing, None] * (
                        corrected[crossing] - reference[crossing]
                    )
                    normal = normals[crossing]
                    inward = np.einsum("ij,ij->i", corrected[crossing] - contact, normal)
                    corrected[crossing] += np.maximum(margin - inward, 0)[:, None] * normal
                    reference[crossing] = contact + margin * normal
            inside = self.contains(corrected)
            if np.any(inside):
                closest, normal = self.closest_surface(corrected[inside])
                corrected[inside] = closest + margin * normal
                if reference is not None:
                    reference[inside] = corrected[inside]
            maximum = float(np.max(np.linalg.norm(corrected - original, axis=1), initial=0.0))
            if maximum > 0.25 * spacing:
                raise WallCorrectionTooLargeError(
                    "VPM wall correction exceeds a quarter of particle spacing; "
                    f"maximum={maximum:.6e} m, spacing={spacing:.6e} m. Reduce the time step."
                )
            if not np.any(inside) and not np.any(np.isfinite(hits)):
                result = corrected.astype(original.dtype)
                if np.any(self.contains(result)):
                    raise RuntimeError("Wall projection was lost at particle storage precision")
                displacement = result.astype(np.float64) - original
                selected = np.any(displacement != 0, axis=1)
                return result, selected, displacement[selected], maximum
        raise RuntimeError("Wall projection did not find fluid support at an intersecting boundary")


_GRID_GEOMETRY_CLASSES = (SolidBoundary, TriangulatedWall)
_GRID_GEOMETRY_METHODS = {
    cls: {
        name: inspect.getattr_static(cls, name)
        for name, value in vars(cls).items()
        if not name.startswith("__")
        and (callable(value) or isinstance(value, (staticmethod, classmethod, property)))
    }
    for cls in _GRID_GEOMETRY_CLASSES
}
