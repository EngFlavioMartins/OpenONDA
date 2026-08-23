"""Methods the FVM-VPM coupler calls on its native Eulerian solver.

Implements the compact coupling API on the FVM ``FVMSolver``:

Getters
    ``get_cell_centre_coordinates``, ``get_cell_volume``, ``get_velocity_field``,
    ``get_velocity_field_into``, ``get_velocity_gradient_field``,
    ``get_velocity_gradient_field_into``, ``get_vorticity_field``,
    ``get_vorticity_field_into``,
    ``get_boundary_face_centre_coordinates``,
    ``get_boundary_face_normal``, ``get_boundary_face_area``, ``n_procs``.
Setters
    ``set_cell_scalar_field``, ``set_cell_vector_field``, ``set_time_step``,
    ``set_kinematic_viscosity``, ``set_dirichlet_velocity_boundary_condition_vec``,
    ``set_freestream_velocity_boundary_condition_vec``,
    ``set_normal_velocity_tangential_gradient_boundary_condition``,
    ``set_freestream_pressure_boundary_condition``,
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

    def _root_view(self, values, *, trailing_shape=()):
        parallel = getattr(self, "parallel", None)
        if parallel is None:
            return np.ascontiguousarray(values, dtype=np.float64)
        return parallel.root_view(values, trailing_shape=trailing_shape)

    def _broadcast_root_value(self, values):
        parallel = getattr(self, "parallel", None)
        if parallel is None or not parallel.is_parallel:
            return values
        payload = values if parallel.is_root else None
        return parallel.bcast(payload, root=0)

    def _cached_partition_ids(self, cache_name, local_ids):
        """Gather immutable partition IDs once and reuse the rank layout."""
        cached = getattr(self, cache_name, None)
        if cached is None:
            gathered = self.parallel.comm.gather(np.asarray(local_ids, dtype=np.int64), root=0)
            cached = gathered if self.parallel.is_root else ()
            setattr(self, cache_name, cached)
        return cached

    def _cached_patch_ids(self, patch_name, local_ids):
        """Gather an immutable face layout once per coupling patch."""
        cache = getattr(self, "_coupling_patch_ids_by_rank", None)
        if cache is None:
            cache = {}
            self._coupling_patch_ids_by_rank = cache
        if patch_name not in cache:
            gathered = self.parallel.comm.gather(np.asarray(local_ids, dtype=np.int64), root=0)
            cache[patch_name] = gathered if self.parallel.is_root else ()
        return cache[patch_name]

    def _gather_owned_cells(self, values, *, trailing_shape=()):
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
        payload = (None if error else np.ascontiguousarray(local[:n_owned]), error)
        gathered = parallel.comm.gather(payload, root=0)

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
                        collective_error = "Partition contains duplicate global cell ownership"
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

    def _scatter_cell_values(self, values, *, trailing_shape=()):
        """Scatter a rank-zero global cell field into owned-plus-halo layout."""
        parallel = getattr(self, "parallel", None)
        if parallel is None or not parallel.is_partitioned:
            values = self._broadcast_root_value(values)
            return np.ascontiguousarray(
                np.asarray(values, dtype=np.float64).reshape(-1, *trailing_shape)
            )

        partition = parallel.partition
        if partition is None:
            raise RuntimeError("Partitioned FVM context has no cell partition")
        requested_ids = self._cached_partition_ids(
            "_coupling_local_cell_ids_by_rank", partition.local_global_ids
        )
        payloads = None
        error = None
        if parallel.is_root:
            global_values = np.asarray(values, dtype=np.float64)
            expected = (partition.n_global_cells, *trailing_shape)
            if global_values.shape != expected:
                error = f"Global cell field must have shape {expected}; got {global_values.shape}"
            elif not np.all(np.isfinite(global_values)):
                error = "Global cell field must contain only finite values"
            else:
                payloads = [
                    np.ascontiguousarray(global_values[global_ids]) for global_ids in requested_ids
                ]
        error = parallel.comm.bcast(error, root=0)
        if error is not None:
            raise ValueError(error)
        return np.ascontiguousarray(parallel.comm.scatter(payloads, root=0))

    def _optional_patch(self, patch_name):
        """Return the local patch, or ``None`` when this rank owns no faces."""
        for boundary in self.boundaries:
            if boundary["name"] == patch_name:
                return boundary
        return None

    def _local_patch_face_ids(self, patch_name):
        boundary = self._optional_patch(patch_name)
        if boundary is None:
            return boundary, np.empty(0, dtype=np.int64), slice(0, 0)
        start, n_faces = int(boundary["start_face"]), int(boundary["n_faces"])
        face_slice = slice(start, start + n_faces)
        global_ids = np.asarray(self.mesh_data["global_face_id"][face_slice], dtype=np.int64)
        return boundary, global_ids, face_slice

    def _gather_patch_faces(self, patch_name, values, *, trailing_shape=()):
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
        payload = (None if error else np.ascontiguousarray(local), error)
        gathered = parallel.comm.gather(payload, root=0)

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
                        f"Patch {patch_name!r} contains duplicate global face ownership"
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

    def _scatter_patch_values(self, patch_name, values, *, trailing_shape=()):
        """Scatter a rank-zero, global-ID-ordered patch field to local faces."""
        parallel = getattr(self, "parallel", None)
        if parallel is None or not parallel.is_partitioned:
            values = self._broadcast_root_value(values)
            return np.ascontiguousarray(
                np.asarray(values, dtype=np.float64).reshape(-1, *trailing_shape)
            )

        _, local_ids, _ = self._local_patch_face_ids(patch_name)
        ids_by_rank = self._cached_patch_ids(patch_name, local_ids)
        payloads = None
        error = None
        if parallel.is_root:
            all_ids = np.concatenate(ids_by_rank)
            sorted_ids = np.sort(all_ids, kind="stable")
            field = np.asarray(values, dtype=np.float64)
            expected = (len(sorted_ids), *trailing_shape)
            if len(np.unique(sorted_ids)) != len(sorted_ids):
                error = f"Patch {patch_name!r} contains duplicate global face ownership"
            elif field.shape != expected:
                error = (
                    f"Global patch {patch_name!r} field must have shape {expected}; "
                    f"got {field.shape}"
                )
            elif not np.all(np.isfinite(field)):
                error = f"Global patch {patch_name!r} field must contain only finite values"
            else:
                payloads = [
                    np.ascontiguousarray(field[np.searchsorted(sorted_ids, rank_ids)])
                    for rank_ids in ids_by_rank
                ]
        error = parallel.comm.bcast(error, root=0)
        if error is not None:
            raise ValueError(error)
        return np.ascontiguousarray(parallel.comm.scatter(payloads, root=0))

    # ── patch lookup ─────────────────────────────────────────────────────────
    def _patch(self, patch_name):
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

    def _patch_face_slice(self, patch_name):
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

    # ── getters: cell fields ─────────────────────────────────────────────────
    def get_cell_centre_coordinates(self):
        """Return cell-centroid coordinates ``(nCells, 3)``, C-contiguous.

        Returns:
            Array of interior-cell centroids in physical space.
        """
        n = self.mesh_data["n_cells"]
        return self._gather_owned_cells(self.geo_data["cell_centre"][:n], trailing_shape=(3,))

    def get_cell_volume(self):
        """Return cell volumes ``(nCells,)``, C-contiguous.

        Returns:
            Array of interior-cell volumes.
        """
        return self._gather_owned_cells(self.geo_data["cell_volume"])

    def get_velocity_field(self):
        """Return the current velocity field ``(nCells, 3)``, C-contiguous.

        Returns:
            Velocity vector at each interior cell centre.
        """
        n = self.mesh_data["n_cells"]
        return self._gather_owned_cells(self.velocity[:n], trailing_shape=(3,))

    def get_velocity_field_into(self, out):
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

    def get_pressure_field(self):
        """Return kinematic pressure at cell centres ``(nCells,)``."""
        n = self.mesh_data["n_cells"]
        return self._gather_owned_cells(np.asarray(self.kinematic_pressure, dtype=np.float64)[:n])

    def shift_pressure_field(self, delta):
        """Add a uniform constant to the pressure field (all ranks, local).

        Legal only without a Dirichlet pressure patch. Changes no velocity and
        no closed-body force.

        Args:
            delta: Constant added to every pressure degree of freedom.

        Raises:
            RuntimeError: If a Dirichlet pressure patch already fixes the datum.
        """
        from ..utils.cavity_utils import needs_pressure_reference

        if not needs_pressure_reference(self.boundaries):
            raise RuntimeError(
                "shift_pressure_field is only valid when the pressure datum is "
                "free; this case already has a Dirichlet pressure patch."
            )
        value = float(delta)
        if not np.isfinite(value):
            raise ValueError("pressure shift must be finite")
        # Shift owned cells *and* ghosts so halo exchanges stay consistent.
        self.kinematic_pressure = np.asarray(self.kinematic_pressure, dtype=np.float64) + value
        for boundary in self.boundaries:
            if boundary.get("kinematic_pressure_value_field") is not None:
                boundary["kinematic_pressure_value_field"] = (
                    np.asarray(boundary["kinematic_pressure_value_field"]) + value
                )

    def get_velocity_gradient_field(self):
        """Return the velocity gradient at cell centres with shape ``(nCells, 3, 3)``."""
        if not self.parallel.is_partitioned and not self.parallel.is_root:
            return np.empty((0, 3, 3), dtype=np.float64)
        n = self.mesh_data["n_cells"]
        velocity_gradient = self._velocity_gradient()
        return self._gather_owned_cells(
            np.asarray(velocity_gradient, dtype=np.float64)[:n],
            trailing_shape=(3, 3),
        )

    def get_velocity_gradient_field_into(self, out):
        """Copy the velocity gradient into a C-contiguous ``(nCells, 3, 3)`` buffer."""
        values = self.get_velocity_gradient_field()
        n = values.shape[0]
        arr = np.asarray(out)
        if arr.dtype != np.float64 or arr.shape != (n, 3, 3) or not arr.flags.c_contiguous:
            raise ValueError("out must be a C-contiguous float64 array with shape (nCells, 3, 3)")
        if n:
            np.copyto(arr, values)
        return arr

    def get_vorticity_field(self):
        """Return the vorticity field, ``(nCells, 3)``.

        Computes vorticity via :func:`..fields.diagnostics.compute_vorticity`.

        Returns:
            Vorticity vector at each interior cell centre.
        """
        from ..fields import diagnostics

        if not self.parallel.is_partitioned and not self.parallel.is_root:
            return np.empty((0, 3), dtype=np.float64)
        n = self.mesh_data["n_cells"]
        vorticity = diagnostics.compute_vorticity(self.velocity, self.mesh_data, self.geo_data)
        return self._gather_owned_cells(
            np.asarray(vorticity).reshape(-1, 3)[:n], trailing_shape=(3,)
        )

    def get_vorticity_field_into(self, out):
        """Copy the vorticity field into a pre-allocated buffer.

        See :meth:`get_velocity_field_into` for the motivation.

        Args:
            out: Pre-allocated ``(nCells, 3)`` float64 C-contiguous array.

        Returns:
            *out*, now filled with the vorticity data.

        Raises:
            ValueError: If *out* has the wrong dtype, shape, or memory layout.
        """
        values = self.get_vorticity_field()
        n = values.shape[0]
        arr = np.asarray(out)
        if arr.dtype != np.float64 or arr.shape != (n, 3) or not arr.flags.c_contiguous:
            raise ValueError("out must be a C-contiguous float64 array with shape (nCells, 3)")
        if n:
            np.copyto(arr, values)
        return arr

    # ── getters: boundary-face geometry (per patch) ──────────────────────────
    def get_boundary_face_centre_coordinates(self, patch_name):
        """Return face-centroid coordinates ``(nFaces, 3)`` for a patch.

        Args:
            patch_name: Name of the boundary patch.

        Returns:
            Face-centroid coordinates on the patch.
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

    def get_boundary_face_area(self, patch_name):
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

    def get_boundary_face_normal(self, patch_name):
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

    def n_procs(self):
        """Number of ranks participating in the native FVM solve."""
        return int(getattr(getattr(self, "parallel", None), "size", 1))

    # ── setters: registered volume fields (fvOptions source inputs) ──────────
    def set_cell_scalar_field(self, name, values):
        """Register a cell-centred scalar field (e.g. an fvOption source).

        The field is stored in ``self.registered_fields[name]`` and can
        be consumed by source-term routines during the next solve.

        Args:
            name: Field identifier (e.g. ``"meanVelocity"``).
            values: 1-D array of length ``nCells``.
        """
        values = self._scatter_cell_values(values)
        expected = self.mesh_data["n_cells"]
        if values.shape != (expected,) or not np.all(np.isfinite(values)):
            raise ValueError(f"Cell scalar field {name!r} must be finite with shape ({expected},)")
        self.registered_fields[name] = values

    def set_cell_vector_field(self, name, component_x, component_y, component_z):
        """Register a cell-centred vector field from components.

        Args:
            name: Field identifier.
            component_x: x-component array ``(nCells,)``.
            component_y: y-component array ``(nCells,)``.
            component_z: z-component array ``(nCells,)``.
        """
        field = np.column_stack(
            [
                np.asarray(component_x, np.float64),
                np.asarray(component_y, np.float64),
                np.asarray(component_z, np.float64),
            ]
        )
        field = self._scatter_cell_values(field, trailing_shape=(3,))
        expected = self.mesh_data["n_cells"]
        if field.shape != (expected, 3) or not np.all(np.isfinite(field)):
            raise ValueError(
                f"Cell vector field {name!r} must be finite with shape ({expected}, 3)"
            )
        self.registered_fields[name] = field

    # ── setters: scalar parameters ───────────────────────────────────────────
    def set_time_step(self, time_step_size):
        """Override the solver time-step size.

        Args:
            time_step_size: New time-step value (seconds).
        """
        time_step_size = float(time_step_size)
        if not np.isfinite(time_step_size) or time_step_size <= 0.0:
            raise ValueError(f"Time step must be finite and positive; got {time_step_size!r}")
        self.time_step_size = time_step_size
        self._accepted_time_step_size = time_step_size
        self.setup.time.time_step_size = time_step_size

    def set_kinematic_viscosity(self, kinematic_viscosity):
        """Override the fluid kinematic viscosity.

        Args:
            kinematic_viscosity: Kinematic viscosity (m²/s).
        """
        kinematic_viscosity = float(kinematic_viscosity)
        if not np.isfinite(kinematic_viscosity) or kinematic_viscosity <= 0.0:
            raise ValueError(
                f"Kinematic viscosity must be finite and positive; got {kinematic_viscosity!r}"
            )
        self.setup.transport.kinematic_viscosity = kinematic_viscosity

    # ── setters: velocity boundary conditions ────────────────────────────────
    def set_dirichlet_velocity_boundary_condition_vec(self, prescribed_velocity, patch_name):
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
        self, normal_velocity, tangential_gradient, patch_name
    ):
        """Prescribe ``U.n`` and the tangential part of ``dU/dn`` per face."""
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

    def set_freestream_velocity_boundary_condition_vec(self, prescribed_velocity, patch_name):
        """Impose a characteristic VPM boundary condition from an ``(N, 3)`` boundary array.

        The per-face VPM boundary-condition values are applied on INFLOW faces only; outflow
        faces extrapolate the owner state (the native ``freestream`` strategy
        switches per face by the sign of the face flux, and the convection
        assembly upwinds consistently).  Pressure on the patch should use the
        matching ``freestream`` type (zero-gradient inflow / fixed outflow).
        """
        field = self._scatter_patch_values(patch_name, prescribed_velocity, trailing_shape=(3,))
        b = self._optional_patch(patch_name)
        if b is None:
            return
        if field.shape != (b["n_faces"], 3) or not np.all(np.isfinite(field)):
            raise ValueError(
                f"Characteristic VPM boundary-condition data for patch {patch_name!r} must be finite "
                f"with shape ({b['n_faces']}, 3)"
            )
        b["velocity_type"] = "freestream"
        b["velocity_value_field"] = field
        b.pop("_fixed_freestream_outflow", None)
        b.pop("_freestream_outflow", None)
        self._write_patch_ghosts(b, field)

    def set_directional_freestream_velocity_boundary_condition_vec(
        self, prescribed_velocity, patch_name, outflow_direction
    ):
        """Extrapolate velocity only on the geometrically designated outflow face.

        Unlike the ordinary ``freestream`` condition, the switching mask is
        fixed by the patch normals rather than recomputed from the local face
        flux.  This is useful for a merged FVM--VPM coupling patch: the
        downstream face is allowed to convect out, while inlet and lateral
        faces retain their complete nonuniform VPM boundary-condition trace even when local
        cross-flow points outward.
        """
        field = self._scatter_patch_values(patch_name, prescribed_velocity, trailing_shape=(3,))
        b = self._optional_patch(patch_name)
        if b is None:
            return
        if field.shape != (b["n_faces"], 3) or not np.all(np.isfinite(field)):
            raise ValueError(
                f"Directional freestream data for patch {patch_name!r} must be finite "
                f"with shape ({b['n_faces']}, 3)"
            )

        direction = np.asarray(outflow_direction, dtype=float).reshape(-1)
        magnitude = float(np.linalg.norm(direction))
        if direction.shape != (3,) or not np.all(np.isfinite(direction)) or magnitude <= 0.0:
            raise ValueError("outflow_direction must be a finite nonzero three-component vector")
        direction /= magnitude

        start, n_faces = b["start_face"], b["n_faces"]
        surface_vectors = np.asarray(
            self.geo_data["face_area_vector"][start : start + n_faces], dtype=float
        )
        normals = surface_vectors / np.linalg.norm(surface_vectors, axis=1)[:, None]
        alignment = normals @ direction
        outflow = alignment > 0.5
        if not np.any(outflow):
            raise ValueError(f"Patch {patch_name!r} has no face aligned with outflow_direction")

        b["velocity_type"] = "freestream"
        b["velocity_value_field"] = field
        b["_fixed_freestream_outflow"] = outflow
        b["_freestream_outflow"] = outflow
        self._write_patch_ghosts(b, field)

    def set_freestream_pressure_boundary_condition(self, patch_name, value=0.0):
        """Use zero-gradient pressure on inflow and fixed pressure on outflow."""
        value = float(value)
        if not np.isfinite(value):
            raise ValueError(f"Freestream pressure for patch {patch_name!r} must be finite")
        b = self._optional_patch(patch_name)
        if b is None:
            return
        b["pressure_type"] = "freestream"
        b["kinematic_pressure_value"] = value
        b.pop("_directional_fixed_flux_pressure", None)
        b.pop("fixed_flux_pressure_external", None)
        b.pop("fixed_flux_pressure_delta", None)

    def set_directional_freestream_pressure_boundary_condition(self, patch_name, value=0.0):
        """Pair fixed outflow pressure with fixed-flux pressure on VPM boundary-condition faces.

        The patch must first receive a directional freestream velocity mask.
        Pressure correction is Dirichlet on that geometric outflow face and
        homogeneous Neumann elsewhere; the absolute pressure on the remaining
        prescribed-velocity faces retains its momentum-compatible
        ``fixedFluxPressure`` increment.
        """
        value = float(value)
        if not np.isfinite(value):
            raise ValueError(f"Directional pressure for patch {patch_name!r} must be finite")
        b = self._optional_patch(patch_name)
        if b is None:
            return
        if b.get("_fixed_freestream_outflow") is None:
            raise ValueError("Directional pressure requires a directional freestream velocity mask")
        b["pressure_type"] = "freestream"
        b["kinematic_pressure_value"] = value
        b["_directional_fixed_flux_pressure"] = True
        b.pop("fixed_flux_pressure_external", None)
        b.pop("fixed_flux_pressure_delta", None)

    def set_fixed_flux_pressure_boundary_condition(self, pressure_delta, patch_name):
        """Impose an externally recorded fixed-flux pressure increment.

        ``pressure_delta`` is the face pressure minus its owner-cell pressure.
        It is used by the interface-replay verifier, where the recorded
        monolithic pressure gradient must be retained through each PIMPLE
        correction instead of being recomputed from a VPM boundary-condition velocity.
        """
        field = self._scatter_patch_values(patch_name, pressure_delta)
        b = self._optional_patch(patch_name)
        if b is None:
            return
        if field.shape != (b["n_faces"],) or not np.all(np.isfinite(field)):
            raise ValueError(
                f"Pressure increments for patch {patch_name!r} must be finite with "
                f"shape ({b['n_faces']},)"
            )
        b["pressure_type"] = "fixedFluxPressure"
        b["fixed_flux_pressure_external"] = True
        b["fixed_flux_pressure_delta"] = field
        n_cells = self.mesh_data["n_cells"]
        n_interior = self.mesh_data["n_interior_faces"]
        start, nf = b["start_face"], b["n_faces"]
        owners = self.mesh_data["owners"][start : start + nf]
        ghosts = n_cells + np.arange(start - n_interior, start - n_interior + nf)
        self.kinematic_pressure[ghosts] = self.kinematic_pressure[owners] + field

    def set_flux_consistent_pressure_boundary_condition(self, patch_name):
        """Pair a prescribed velocity flux with native ``fixedFluxPressure``."""
        b = self._optional_patch(patch_name)
        if b is None:
            return
        b["pressure_type"] = "fixedFluxPressure"
        b.pop("fixed_flux_pressure_external", None)
        b.pop("fixed_flux_pressure_delta", None)
        b.pop("fixed_gradient_delta", None)

    def set_neumann_pressure_boundary_condition(self, pressure_gradient, patch_name):
        """Impose a vector pressure gradient on a coupling patch."""
        field = self._scatter_patch_values(patch_name, pressure_gradient, trailing_shape=(3,))
        b = self._optional_patch(patch_name)
        if b is None:
            return
        nf = b["n_faces"]
        start = b["start_face"]
        sf = self.geo_data["face_area_vector"][start : start + nf]
        normals = sf / np.linalg.norm(sf, axis=1)[:, None]
        if field.shape != (nf, 3):
            raise ValueError(
                f"Pressure gradient for patch {patch_name!r} must have shape ({nf}, 3); "
                f"got {field.shape}"
            )
        gradient_normal = np.einsum("ij,ij->i", field, normals)
        if not np.all(np.isfinite(gradient_normal)):
            raise ValueError(f"Pressure gradient for patch {patch_name!r} must be finite")

        distance = np.einsum(
            "ij,ij->i",
            self.geo_data["cell_connection_vector"][start : start + nf],
            normals,
        )
        delta = gradient_normal * distance
        b["pressure_type"] = "fixedGradient"
        b["fixed_gradient_delta"] = delta
        b.pop("fixed_flux_pressure_external", None)
        b.pop("fixed_flux_pressure_delta", None)

        n_cells = self.mesh_data["n_cells"]
        n_interior = self.mesh_data["n_interior_faces"]
        owners = self.mesh_data["owners"][start : start + nf]
        ghosts = n_cells + np.arange(start - n_interior, start - n_interior + nf)
        self.kinematic_pressure[ghosts] = self.kinematic_pressure[owners] + delta

    def set_external_face_flux_boundary_condition(self, volumetric_face_flux, patch_name):
        """Prescribe replayed volumetric fluxes on a boundary patch.

        The next pressure-correction assembly uses these values directly as
        its boundary fluxes.  This is deliberately a diagnostic/replay API:
        normal coupled runs continue to derive their flux from the VPM boundary condition
        velocity condition.  Together with ``fixedFluxPressure`` it lets a
        cropped FVM solve reproduce a monolithic cut without converting a
        discrete face flux back through an interpolated velocity.
        """
        field = self._scatter_patch_values(patch_name, volumetric_face_flux)
        b = self._optional_patch(patch_name)
        if b is None:
            return
        if field.shape != (b["n_faces"],) or not np.all(np.isfinite(field)):
            raise ValueError(
                f"Face fluxes for patch {patch_name!r} must be finite with shape ({b['n_faces']},)"
            )
        b["external_face_flux"] = field

    # ── helper ───────────────────────────────────────────────────────────────
    def _write_patch_ghosts(self, boundary, field):
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
