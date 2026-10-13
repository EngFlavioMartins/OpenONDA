"""Methods the FVM-VPM coupler calls on its native Eulerian solver.

Implements the compact coupling API on the FVM ``FVMSolver``:

Getters
    ``get_cell_centre_coordinates``, ``get_cell_volume``, ``get_velocity_field``,
    ``get_velocity_field_into``, ``get_velocity_gradient_field``,
    ``get_velocity_gradient_field_into``,
    ``get_boundary_face_centre_coordinates``,
    ``get_boundary_face_normal``, ``get_boundary_face_area``,
    ``get_boundary_face_velocity``, ``n_procs``.
Setters
    ``set_dirichlet_velocity_boundary_condition_vec``,
    ``set_normal_velocity_tangential_gradient_boundary_condition``,
    ``set_flux_consistent_pressure_boundary_condition``.
Driver
    ``solve_pimple`` / ``advance_time`` live on the ``FVMSolver`` itself.

Coupler-facing getters are collective under MPI and expose one global,
global-ID-ordered field on rank zero (with typed empty arrays elsewhere).
Setters perform the inverse operation.  Replicated solves broadcast the
rank-zero value; partitioned solves scatter only the cells or faces owned by
each rank.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..fields.mixed_velocity_boundary import (
    update_normal_velocity_tangential_gradient_boundary,
)


class CouplerInterfaceMixin:
    """Coupler-facing methods.  Expects the host to provide ``mesh_data``,
    ``geo_data``, ``boundaries``, ``velocity``, ``kinematic_pressure``, ``setup``,
    ``registered_fields``
    and ``time_step_size`` (all set up by ``FVMSolver.__init__``)."""

    parallel: Any
    mesh_data: dict[str, Any]
    geo_data: dict[str, Any]
    boundaries: list[dict[str, Any]]
    velocity: np.ndarray
    kinematic_pressure: np.ndarray
    setup: Any
    registered_fields: dict[str, np.ndarray]
    time_step_size: float
    _accepted_time_step_size: float
    _coupling_patch_ids_by_rank: dict[str, Any]

    def _velocity_gradient(self) -> np.ndarray:
        raise NotImplementedError

    def _root_view(
        self, values: np.ndarray, *, trailing_shape: tuple[int, int] | tuple[int] = ()
    ) -> np.ndarray:
        parallel = getattr(self, "parallel", None)
        if parallel is None:
            return np.ascontiguousarray(values, dtype=np.float64)
        return parallel.root_view(values, trailing_shape=trailing_shape)

    def _broadcast_root_value(self, values: np.ndarray | None) -> np.ndarray:
        parallel = getattr(self, "parallel", None)
        if parallel is None or not parallel.is_parallel:
            return values
        rank_field_data = values if parallel.is_root else None
        return parallel.bcast(rank_field_data, root=0)

    def _cached_partition_ids(
        self, cache_name: str, local_ids: np.ndarray
    ) -> list[np.ndarray] | tuple[()]:
        """Gather immutable partition IDs once and reuse the rank layout."""
        cached = getattr(self, cache_name, None)
        if cached is None:
            gathered = self.parallel.comm.gather(np.asarray(local_ids, dtype=np.int64), root=0)
            cached = gathered if self.parallel.is_root else ()
            setattr(self, cache_name, cached)
        return cached

    def _cached_patch_ids(
        self, patch_name: str, local_ids: np.ndarray
    ) -> list[np.ndarray] | tuple[()]:
        """Gather an immutable face layout once per coupling patch."""
        cache = getattr(self, "_coupling_patch_ids_by_rank", None)
        if cache is None:
            cache = {}
            self._coupling_patch_ids_by_rank = cache
        if patch_name not in cache:
            gathered = self.parallel.comm.gather(np.asarray(local_ids, dtype=np.int64), root=0)
            cache[patch_name] = gathered if self.parallel.is_root else ()
        return cache[patch_name]

    def _gather_owned_cells(
        self, values: np.ndarray, *, trailing_shape: tuple[int, int] | tuple[int] = ()
    ) -> np.ndarray:
        """Gather an owned-cell field into global cell-ID order on rank zero."""
        parallel = getattr(self, "parallel", None)
        if parallel is None or not parallel.is_partitioned:
            return self._root_view(values, trailing_shape=trailing_shape)

        partition = parallel.partition
        if partition is None:
            raise RuntimeError("Partitioned FVM context has no cell partition")
        local = np.asarray(values, dtype=np.float64)
        n_owned = len(partition.owned_global_ids)
        error = None
        if local.ndim == 0 or local.shape[0] < n_owned:
            error = (
                f"Rank {parallel.rank} cell field has shape {local.shape}; "
                f"expected at least {n_owned} leading entries"
            )
        elif local.shape[1:] != tuple(trailing_shape):
            error = (
                f"Rank {parallel.rank} cell field has trailing shape {local.shape[1:]}; "
                f"expected {tuple(trailing_shape)}"
            )
        ids_by_rank = self._cached_partition_ids(
            "_coupling_owned_cell_ids_by_rank", partition.owned_global_ids
        )
        rank_field_data = (None if error else np.ascontiguousarray(local[:n_owned]), error)
        gathered = parallel.comm.gather(rank_field_data, root=0)

        result = None
        collective_error = None
        if parallel.is_root:
            errors = [item[1] for item in gathered if item[1] is not None]
            if errors:
                collective_error = "; ".join(errors)
            else:
                result = np.empty((partition.n_global_cells, *trailing_shape), dtype=np.float64)
                seen = np.zeros(partition.n_global_cells, dtype=bool)
                for global_ids, (owned_values, _) in zip(ids_by_rank, gathered, strict=True):
                    if np.any(global_ids < 0) or np.any(global_ids >= partition.n_global_cells):
                        collective_error = "Partition contains an out-of-range global cell ID"
                        break
                    if np.any(seen[global_ids]):
                        collective_error = (
                            "Partition contains duplicate global cell rank assignment"
                        )
                        break
                    result[global_ids] = owned_values
                    seen[global_ids] = True
                if collective_error is None and not np.all(seen):
                    collective_error = "Partitioned cell gather did not cover every global cell"
        collective_error = parallel.comm.bcast(collective_error, root=0)
        if collective_error is not None:
            raise RuntimeError(collective_error)
        if parallel.is_root:
            return np.ascontiguousarray(result)
        return np.empty((0, *trailing_shape), dtype=np.float64)

    def _optional_patch(self, patch_name: str) -> dict[str, object] | None:
        """Return the local patch, or ``None`` when this rank owns no faces."""
        for boundary in self.boundaries:
            if boundary["name"] == patch_name:
                return boundary
        return None

    def _local_patch_face_ids(self, patch_name: str) -> tuple[dict | None, np.ndarray, slice]:
        boundary = self._optional_patch(patch_name)
        if boundary is None:
            return boundary, np.empty(0, dtype=np.int64), slice(0, 0)
        start, n_faces = int(boundary["start_face"]), int(boundary["n_faces"])
        face_slice = slice(start, start + n_faces)
        global_ids = np.asarray(self.mesh_data["global_face_id"][face_slice], dtype=np.int64)
        return boundary, global_ids, face_slice

    def _gather_patch_faces(
        self, patch_name: str, values: np.ndarray, *, trailing_shape: tuple[int] = ()
    ) -> np.ndarray:
        """Gather a physical patch field into global face-ID order on rank zero."""
        parallel = getattr(self, "parallel", None)
        if parallel is None or not parallel.is_partitioned:
            return self._root_view(values, trailing_shape=trailing_shape)

        _, global_ids, _ = self._local_patch_face_ids(patch_name)
        local = np.asarray(values, dtype=np.float64)
        expected = (len(global_ids), *trailing_shape)
        error = None
        if local.shape != expected:
            error = (
                f"Rank {parallel.rank} patch {patch_name!r} field has shape "
                f"{local.shape}; expected {expected}"
            )
        ids_by_rank = self._cached_patch_ids(patch_name, global_ids)
        rank_field_data = (None if error else np.ascontiguousarray(local), error)
        gathered = parallel.comm.gather(rank_field_data, root=0)

        result = None
        collective_error = None
        if parallel.is_root:
            errors = [item[1] for item in gathered if item[1] is not None]
            if errors:
                collective_error = "; ".join(errors)
            else:
                all_ids = np.concatenate(ids_by_rank)
                all_values = np.concatenate([item[0] for item in gathered], axis=0)
                if len(np.unique(all_ids)) != len(all_ids):
                    collective_error = (
                        f"Patch {patch_name!r} contains duplicate global face rank assignment"
                    )
                else:
                    order = np.argsort(all_ids, kind="stable")
                    result = np.ascontiguousarray(all_values[order])
        collective_error = parallel.comm.bcast(collective_error, root=0)
        if collective_error is not None:
            raise RuntimeError(collective_error)
        if parallel.is_root:
            return result
        return np.empty((0, *trailing_shape), dtype=np.float64)

    def _scatter_patch_values(
        self, patch_name: str, values: np.ndarray | None, *, trailing_shape: tuple[int, ...] = ()
    ) -> np.ndarray:
        """Scatter a rank-zero, global-ID-ordered patch field to local faces."""
        parallel = getattr(self, "parallel", None)
        if parallel is None or not parallel.is_partitioned:
            values = self._broadcast_root_value(values)
            return np.ascontiguousarray(
                np.asarray(values, dtype=np.float64).reshape(-1, *trailing_shape)
            )

        _, local_ids, _ = self._local_patch_face_ids(patch_name)
        ids_by_rank = self._cached_patch_ids(patch_name, local_ids)
        rank_fields = None
        error = None
        if parallel.is_root:
            all_ids = np.concatenate(ids_by_rank)
            sorted_ids = np.sort(all_ids, kind="stable")
            field = np.asarray(values, dtype=np.float64)
            expected = (len(sorted_ids), *trailing_shape)
            if len(np.unique(sorted_ids)) != len(sorted_ids):
                error = f"Patch {patch_name!r} contains duplicate global face rank assignment"
            elif field.shape != expected:
                error = (
                    f"Global patch {patch_name!r} field must have shape {expected}; "
                    f"got {field.shape}"
                )
            elif not np.all(np.isfinite(field)):
                error = f"Global patch {patch_name!r} field must contain only finite values"
            else:
                rank_fields = [
                    np.ascontiguousarray(field[np.searchsorted(sorted_ids, rank_ids)])
                    for rank_ids in ids_by_rank
                ]
        error = parallel.comm.bcast(error, root=0)
        if error is not None:
            raise ValueError(error)
        return np.ascontiguousarray(parallel.comm.scatter(rank_fields, root=0))

    # Patch lookup.
    def _patch(self, patch_name: str) -> dict[str, object]:
        """Look up a boundary patch dict by name.

        Searches ``self.boundaries`` for a patch whose ``"name"`` key
        matches *patch_name*.

        Args:
            patch_name: Name of the boundary patch (e.g. ``"inlet"``).

        Returns:
            The boundary patch dictionary.

        Raises:
            KeyError: If no patch with the given name exists.
        """
        boundary = self._optional_patch(patch_name)
        if boundary is not None:
            return boundary
        raise KeyError(f"Boundary patch '{patch_name}' not found")

    def _patch_face_slice(self, patch_name: str) -> tuple[dict[str, object], slice]:
        """Return (boundary_dict, slice) covering the patch faces.

        The slice indexes into face-centred arrays such as
        ``geo_data["face_centre"]``.

        Args:
            patch_name: Name of the boundary patch.

        Returns:
            Tuple of ``(boundary_dict, slice)`` where *slice* covers
            ``startFace`` … ``startFace + nFaces - 1``.
        """
        b = self._patch(patch_name)
        start, nf = b["start_face"], b["n_faces"]
        return b, slice(start, start + nf)

    # Getters: cell fields.
    def get_wall_surface_triangles(self) -> np.ndarray:
        """Collect body-fitted wall triangles on root, empty on other ranks.

        Triangles have shape ``(N, 3, 3)`` and normals out of the fluid.
        Each physical face is owned once in a partitioned mesh. A centroid
        fan follows the native polygon geometry, including hanging nodes.
        This getter is collective, including ranks without wall faces.
        """
        resolved = getattr(self, "_resolved_setup", self.setup)
        wall_names = {bc.name for bc in resolved.boundaries if bc.mesh_type == "wall"}
        vertices = np.asarray(self.mesh_data["vertex_position"], dtype=np.float64)
        triangles = []
        for boundary in self.boundaries:
            if boundary["name"] not in wall_names:
                continue
            start = int(boundary["start_face"])
            for face_id in range(start, start + int(boundary["n_faces"])):
                nodes = np.asarray(self.mesh_data["faces"][face_id], dtype=np.int64)
                polygon = vertices[nodes[nodes >= 0]]
                centre = polygon.mean(axis=0)
                for index in range(len(polygon)):
                    triangle = np.array(
                        [centre, polygon[index], polygon[(index + 1) % len(polygon)]]
                    )
                    if np.linalg.norm(np.cross(triangle[1] - centre, triangle[2] - centre)) > 0.0:
                        triangles.append(triangle)
        local = np.asarray(triangles, dtype=np.float64).reshape(-1, 3, 3)
        parallel = getattr(self, "parallel", None)
        if parallel is None or not parallel.is_partitioned:
            return self._root_view(local, trailing_shape=(3, 3))
        gathered = parallel.comm.gather(local, root=0)
        if parallel.is_root:
            return np.concatenate(gathered, axis=0)
        return np.empty((0, 3, 3), dtype=np.float64)

    def get_cell_centre_coordinates(self) -> np.ndarray:
        """Return cell-centre coordinates ``(nCells, 3)``, C-contiguous.

        Returns:
            Array of interior-cell centres in physical space.
        """
        n = self.mesh_data["n_cells"]
        return self._gather_owned_cells(self.geo_data["cell_centre"][:n], trailing_shape=(3,))

    def get_cell_volume(self) -> np.ndarray:
        """Return cell volumes ``(nCells,)``, C-contiguous.

        Returns:
            Array of interior-cell volumes.
        """
        return self._gather_owned_cells(self.geo_data["cell_volume"])

    def get_velocity_field(self) -> np.ndarray:
        """Return the current velocity field ``(nCells, 3)``, C-contiguous.

        Returns:
            Velocity vector at each interior cell centre.
        """
        n = self.mesh_data["n_cells"]
        return self._gather_owned_cells(self.velocity[:n], trailing_shape=(3,))

    def get_velocity_field_into(self, out: np.ndarray) -> np.ndarray:
        """Copy the velocity field into a pre-allocated buffer.

        Avoids an extra allocation when the caller already owns the
        output array (e.g. a NumPy array from the coupler's MPI buffer).

        Args:
            out: Pre-allocated ``(nCells, 3)`` float64 C-contiguous array.

        Returns:
            *out*, now filled with the velocity data.

        Raises:
            ValueError: If *out* has the wrong dtype, shape, or memory layout.
        """
        values = self.get_velocity_field()
        n = values.shape[0]
        arr = np.asarray(out)
        if arr.dtype != np.float64 or arr.shape != (n, 3) or not arr.flags.c_contiguous:
            raise ValueError("out must be a C-contiguous float64 array with shape (nCells, 3)")
        if n:
            np.copyto(arr, values)
        return arr

    def get_pressure_field(self) -> np.ndarray:
        """Return kinematic pressure at cell centres ``(nCells,)``."""
        n = self.mesh_data["n_cells"]
        return self._gather_owned_cells(np.asarray(self.kinematic_pressure, dtype=np.float64)[:n])

    def get_velocity_gradient_field(self) -> np.ndarray:
        """Return the velocity gradient at cell centres with shape ``(nCells, 3, 3)``."""
        if not self.parallel.is_partitioned and not self.parallel.is_root:
            return np.empty((0, 3, 3), dtype=np.float64)
        n = self.mesh_data["n_cells"]
        velocity_gradient = self._velocity_gradient()
        return self._gather_owned_cells(
            np.asarray(velocity_gradient, dtype=np.float64)[:n],
            trailing_shape=(3, 3),
        )

    def get_velocity_gradient_field_into(self, out: np.ndarray) -> np.ndarray:
        """Copy the velocity gradient into a C-contiguous ``(nCells, 3, 3)`` buffer."""
        values = self.get_velocity_gradient_field()
        n = values.shape[0]
        arr = np.asarray(out)
        if arr.dtype != np.float64 or arr.shape != (n, 3, 3) or not arr.flags.c_contiguous:
            raise ValueError("out must be a C-contiguous float64 array with shape (nCells, 3, 3)")
        if n:
            np.copyto(arr, values)
        return arr

    # Getters: boundary-face geometry (per patch).
    def get_boundary_face_centre_coordinates(self, patch_name: str) -> np.ndarray:
        """Return face-centre coordinates ``(nFaces, 3)`` for a patch.

        Args:
            patch_name: Name of the boundary patch.

        Returns:
            Face-centre coordinates on the patch.
        """
        if self.parallel.is_partitioned:
            _, _, face_slice = self._local_patch_face_ids(patch_name)
        else:
            _, face_slice = self._patch_face_slice(patch_name)
        return self._gather_patch_faces(
            patch_name,
            self.geo_data["face_centre"][face_slice],
            trailing_shape=(3,),
        )

    def get_boundary_face_area(self, patch_name: str) -> np.ndarray:
        """Return face area magnitudes ``(nFaces,)`` for a patch.

        Args:
            patch_name: Name of the boundary patch.

        Returns:
            Scalar face areas (‖S_f‖).
        """
        if self.parallel.is_partitioned:
            _, _, face_slice = self._local_patch_face_ids(patch_name)
        else:
            _, face_slice = self._patch_face_slice(patch_name)
        return self._gather_patch_faces(
            patch_name,
            np.linalg.norm(self.geo_data["face_area_vector"][face_slice], axis=1),
        )

    def get_boundary_face_normal(self, patch_name: str) -> np.ndarray:
        """Return unit face normals ``(nFaces, 3)`` for a patch.

        Normals point outward from the domain.

        Args:
            patch_name: Name of the boundary patch.

        Returns:
            Unit-normal vectors (‖n̂‖ = 1).
        """
        if self.parallel.is_partitioned:
            _, _, face_slice = self._local_patch_face_ids(patch_name)
        else:
            _, face_slice = self._patch_face_slice(patch_name)
        sf = self.geo_data["face_area_vector"][face_slice]
        mag = np.linalg.norm(sf, axis=1, keepdims=True)
        return self._gather_patch_faces(patch_name, sf / (mag + 1e-30), trailing_shape=(3,))

    def get_boundary_face_velocity(self, patch_name: str) -> np.ndarray:
        """Return the actual face-centred velocity stored for a boundary patch."""
        if self.parallel.is_partitioned:
            boundary, _, face_slice = self._local_patch_face_ids(patch_name)
            if boundary is None:
                local = np.empty((0, 3), dtype=np.float64)
            else:
                first = self.mesh_data["n_cells"] + (
                    face_slice.start - self.mesh_data["n_interior_faces"]
                )
                local = self.velocity[first : first + boundary["n_faces"]]
        else:
            boundary, face_slice = self._patch_face_slice(patch_name)
            first = self.mesh_data["n_cells"] + (
                face_slice.start - self.mesh_data["n_interior_faces"]
            )
            local = self.velocity[first : first + boundary["n_faces"]]
        return self._gather_patch_faces(patch_name, local, trailing_shape=(3,))

    def n_procs(self) -> int:
        """Number of ranks participating in the native FVM solve."""
        return int(getattr(getattr(self, "parallel", None), "size", 1))

    # Setters: registered volume fields (fvOptions source inputs).
    # Setters: scalar parameters.
    # Setters: velocity boundary conditions.
    def set_dirichlet_velocity_boundary_condition_vec(
        self, prescribed_velocity: np.ndarray, patch_name: str
    ) -> None:
        """Impose a Dirichlet velocity from an ``(N, 3)`` boundary array."""
        field = self._scatter_patch_values(patch_name, prescribed_velocity, trailing_shape=(3,))
        b = self._optional_patch(patch_name)
        if b is None:
            return
        if field.shape != (b["n_faces"], 3) or not np.all(np.isfinite(field)):
            raise ValueError(
                f"Dirichlet data for patch {patch_name!r} must be finite with shape "
                f"({b['n_faces']}, 3)"
            )
        b["velocity_type"] = "fixedValue"
        b["velocity_value_field"] = field
        b.pop("_fixed_freestream_outflow", None)
        b.pop("_freestream_outflow", None)
        self._write_patch_ghosts(b, field)

    def set_normal_velocity_tangential_gradient_boundary_condition(
        self, normal_velocity: np.ndarray, tangential_gradient: np.ndarray, patch_name: str
    ) -> None:
        """Prescribe normal velocity and the tangential velocity gradient per face."""
        normal_field = self._scatter_patch_values(patch_name, normal_velocity)
        gradient_field = self._scatter_patch_values(
            patch_name, tangential_gradient, trailing_shape=(3,)
        )
        b = self._optional_patch(patch_name)
        if b is None:
            return
        n_faces = b["n_faces"]
        if normal_field.shape != (n_faces,) or not np.all(np.isfinite(normal_field)):
            raise ValueError(
                f"Normal velocity for patch {patch_name!r} must be finite with shape ({n_faces},)"
            )
        if gradient_field.shape != (n_faces, 3) or not np.all(np.isfinite(gradient_field)):
            raise ValueError(
                f"Tangential gradient for patch {patch_name!r} must be finite with shape "
                f"({n_faces}, 3)"
            )

        start = b["start_face"]
        surface_vectors = np.asarray(
            self.geo_data["face_area_vector"][start : start + n_faces], dtype=np.float64
        )
        areas = np.linalg.norm(surface_vectors, axis=1)
        if np.any(areas <= 1.0e-14):
            raise ValueError(f"Patch {patch_name!r} contains a degenerate face")
        normals = surface_vectors / areas[:, np.newaxis]
        removed_normal_component = np.einsum("ij,ij->i", gradient_field, normals)
        gradient_field = gradient_field - removed_normal_component[:, np.newaxis] * normals

        b["velocity_type"] = "normalValueTangentialGradient"
        b["normal_velocity_field"] = normal_field
        b["tangential_gradient_field"] = gradient_field
        b["max_removed_tangential_gradient_normal_component"] = (
            float(np.max(np.abs(removed_normal_component))) if n_faces else 0.0
        )
        b.pop("velocity_value_field", None)
        b.pop("_fixed_freestream_outflow", None)
        b.pop("_freestream_outflow", None)
        update_normal_velocity_tangential_gradient_boundary(
            self.velocity, b, self.mesh_data, self.geo_data
        )
        self._invalidate_derived_fields()
        self._publish_state()

    def set_flux_consistent_pressure_boundary_condition(self, patch_name: str) -> None:
        """Pair a prescribed velocity flux with native ``fixedFluxPressure``.

        Reapplying the native condition retains its accepted pressure increment
        for the next momentum predictor. The pressure correction recomputes it
        from the new flux. An external replay increment or a different boundary
        condition does not supply compatible native history.
        """
        b = self._optional_patch(patch_name)
        if b is None:
            return
        retain_native_history = b.get("pressure_type") == "fixedFluxPressure" and not b.get(
            "fixed_flux_pressure_external", False
        )
        b["pressure_type"] = "fixedFluxPressure"
        b.pop("fixed_flux_pressure_external", None)
        if not retain_native_history:
            b.pop("fixed_flux_pressure_delta", None)
        b.pop("fixed_gradient_delta", None)
        invalidate = getattr(self, "_invalidate_derived_fields", None)
        if invalidate is not None:
            invalidate()

    # Helper.
    def _write_patch_ghosts(self, boundary: dict[str, object], field: np.ndarray) -> None:
        """Write a boundary-face-centred field into the velocity ghost layer.

        The FVM solver stores boundary values in a contiguous slab
        ``velocity[n_elements : n_elements + n_boundary_faces]``, indexed by
        ``(startFace - n_interior_faces)``.  This helper writes the
        per-face *field* into the correct slice.

        Args:
            boundary: Boundary patch dictionary (must contain ``startFace``,
                      ``nFaces``).
            field:    Per-face values ``(nFaces, 3)`` to impose.
        """
        n_elem = self.mesh_data["n_cells"]
        n_int = self.mesh_data["n_interior_faces"]
        start, nf = boundary["start_face"], boundary["n_faces"]
        idx = n_elem + (start - n_int)
        self.velocity[idx : idx + nf] = field
        self._invalidate_derived_fields()
        self._publish_state()
