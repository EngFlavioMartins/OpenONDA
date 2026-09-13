"""Conservative projection onto an exactly extruded spanwise-invariant state."""

from __future__ import annotations

import numpy as np

from ..mesh.partition import ownership_ranges


def spanwise_cell_groups(mesh: dict) -> np.ndarray:
    """Return stack IDs for cells sharing the same x-y footprint.

    Parameters
    ----------
    mesh : dict
        Extruded native mesh with ``vertex_position`` and compact
        ``cell_vertex_indices`` arrays.

    Returns
    -------
    numpy.ndarray
        Int32 group labels, shape ``(n_cells,)``. Each group must contain a
        uniformly sized z stack.

    Raises
    ------
    ValueError
        If the mesh is not uniformly extruded or has incomplete stacks.
    """
    points = np.asarray(mesh["vertex_position"], dtype=np.float64)
    vertices = np.asarray(mesh["cell_vertex_indices"], dtype=np.int64)
    centres_xy = points[vertices, :2].mean(axis=1)
    _, groups, counts = np.unique(
        np.round(centres_xy, decimals=12),
        axis=0,
        return_inverse=True,
        return_counts=True,
    )
    if np.unique(counts).size != 1 or counts[0] < 2:
        raise ValueError(f"Mesh is not a uniformly extruded cell stack; counts={np.unique(counts)}")
    return np.asarray(groups, dtype=np.int32)


def spanwise_face_groups(mesh: dict) -> tuple[np.ndarray, np.ndarray]:
    """Group vertical faces by x-y footprint and identify horizontal faces.

    Parameters
    ----------
    mesh : dict
        Native face mesh with vertex coordinates and polygon faces.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Int32 group IDs for every face (``-1`` for horizontal faces) and a
        boolean horizontal-face mask.

    Raises
    ------
    ValueError
        If vertical face stacks are incomplete/nonuniform.
    """
    points = np.asarray(mesh["vertex_position"], dtype=np.float64)
    faces = np.asarray(mesh["faces"], dtype=np.int64)
    vertices = points[faces]
    horizontal = np.ptp(vertices[:, :, 2], axis=1) < 1.0e-12
    vertical = ~horizontal
    xy = np.round(vertices[vertical, :, :2], decimals=12)
    order = np.lexsort((xy[:, :, 1], xy[:, :, 0]), axis=1)
    sorted_xy = np.take_along_axis(xy, order[:, :, None], axis=1)
    keys = sorted_xy.reshape(len(sorted_xy), -1)
    _, local_groups, counts = np.unique(
        keys,
        axis=0,
        return_inverse=True,
        return_counts=True,
    )
    if np.unique(counts).size != 1 or counts[0] < 2:
        raise ValueError(f"Mesh is not a uniformly extruded face stack; counts={np.unique(counts)}")
    groups = np.full(len(faces), -1, dtype=np.int32)
    groups[vertical] = local_groups
    return groups, horizontal


def build_spanwise_projection_layout(mesh: dict, n_ranks: int) -> dict[str, np.ndarray | int]:
    """Build global stack groups and deterministic face-authority metadata.

    Parameters
    ----------
    mesh : dict
        Global extruded native mesh.
    n_ranks : int
        Number of partition ranks used to assign vertical-face authority.

    Returns
    -------
    dict[str, numpy.ndarray or int]
        Cell/face group arrays, horizontal mask, face-authority ranks, and
        group counts for broadcast to all ranks.
    """
    cell_groups = spanwise_cell_groups(mesh)
    face_groups, horizontal_faces = spanwise_face_groups(mesh)
    offsets = ownership_ranges(int(mesh["n_cells"]), int(n_ranks))
    face_authority = np.searchsorted(
        offsets[1:],
        np.asarray(mesh["owners"], dtype=np.int64),
        side="right",
    ).astype(np.int32)
    return {
        "cell_groups": cell_groups,
        "face_groups": face_groups,
        "horizontal_faces": np.asarray(horizontal_faces, dtype=bool),
        "face_authority": face_authority,
        "n_cell_groups": int(cell_groups.max()) + 1,
        "n_face_groups": int(face_groups.max()) + 1,
    }


class SpanwiseInvariantProjector:
    """Project partitioned FVM fields onto an exact spanwise-invariant state.

    The projector averages cell-centred values within each x-y stack, zeros
    spanwise velocity/flux, and averages vertical face fluxes using one
    deterministic MPI authority per face group. Horizontal fluxes are set to
    zero. It is intended for quasi-2-D extruded calculations.

    Parameters
    ----------
    solver : FVMSolver
        Initialized solver with serial or partitioned parallel context.
    root_layout : dict or None
        Global layout built by :func:`build_spanwise_projection_layout` on rank
        zero. It is broadcast during construction.
    """

    def __init__(self, solver, root_layout: dict | None):
        """Build local cell/face maps from the rank-zero projection layout.

        Parameters
        ----------
        solver : FVMSolver
            Initialized serial or partitioned solver whose mesh and MPI
            context supply the local indexing.
        root_layout : dict or None
            Global layout returned by :func:`build_spanwise_projection_layout`
            on rank zero. It is broadcast to all ranks before local maps are
            formed.

        Raises
        ------
        ValueError
            If the broadcast layout is missing or contains an incomplete
            spanwise cell stack.

        Side Effects
        ------------
        Allocates host-side group/count arrays and performs MPI broadcasts and
        reductions; it does not mutate the solver's physical fields.
        """
        layout = solver.parallel.bcast(root_layout, root=0)
        if not isinstance(layout, dict):
            raise ValueError("spanwise projection layout was not provided by rank zero")
        self.parallel = solver.parallel
        self.n_owned = (
            int(self.parallel.n_owned)
            if self.parallel.is_partitioned
            else int(solver.mesh_data["n_cells"])
        )
        if self.parallel.is_partitioned:
            cell_ids = np.asarray(
                self.parallel.partition.local_global_ids[: self.n_owned], dtype=np.int64
            )
            face_ids = np.asarray(solver.mesh_data["global_face_id"], dtype=np.int64)
        else:
            cell_ids = np.arange(self.n_owned, dtype=np.int64)
            face_ids = np.arange(int(solver.mesh_data["n_faces"]), dtype=np.int64)

        self.cell_groups = np.asarray(layout["cell_groups"], dtype=np.int32)[cell_ids]
        self.n_cell_groups = int(layout["n_cell_groups"])
        local_cell_counts = np.bincount(self.cell_groups, minlength=self.n_cell_groups).astype(
            np.float64
        )
        self.cell_counts = np.asarray(self.parallel.global_sum(local_cell_counts))
        if np.any(self.cell_counts < 2.0):
            raise ValueError("spanwise cell projection contains an incomplete stack")

        all_face_groups = np.asarray(layout["face_groups"], dtype=np.int32)
        all_horizontal = np.asarray(layout["horizontal_faces"], dtype=bool)
        all_authority = np.asarray(layout["face_authority"], dtype=np.int32)
        self.face_groups = all_face_groups[face_ids]
        self.horizontal_faces = all_horizontal[face_ids]
        self.vertical_faces = ~self.horizontal_faces
        self.authoritative_vertical = self.vertical_faces & (
            all_authority[face_ids] == int(self.parallel.rank)
        )
        self.n_face_groups = int(layout["n_face_groups"])
        local_face_counts = np.bincount(
            self.face_groups[self.authoritative_vertical], minlength=self.n_face_groups
        ).astype(np.float64)
        self.face_counts = np.asarray(self.parallel.global_sum(local_face_counts))
        if np.any(self.face_counts < 2.0):
            raise ValueError("spanwise face projection contains an incomplete stack")
        self.last_removed_maximum: dict[str, float] = {}

    def _cell_mean(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        if values.ndim == 1:
            local = np.bincount(
                self.cell_groups,
                weights=values[: self.n_owned],
                minlength=self.n_cell_groups,
            )
            return np.asarray(self.parallel.global_sum(local)) / self.cell_counts
        local = np.column_stack(
            [
                np.bincount(
                    self.cell_groups,
                    weights=values[: self.n_owned, component],
                    minlength=self.n_cell_groups,
                )
                for component in range(values.shape[1])
            ]
        )
        return np.asarray(self.parallel.global_sum(local)) / self.cell_counts[:, None]

    def _face_mean(self, values: np.ndarray) -> np.ndarray:
        local = np.bincount(
            self.face_groups[self.authoritative_vertical],
            weights=np.asarray(values, dtype=np.float64)[self.authoritative_vertical],
            minlength=self.n_face_groups,
        )
        return np.asarray(self.parallel.global_sum(local)) / self.face_counts

    def __call__(self, solver) -> None:
        """Apply the projection in place and record removed maxima.

        Parameters
        ----------
        solver : FVMSolver
            Solver whose owned fields and face fluxes are modified.

        Side Effects
        ------------
        Mutates owned velocity, kinematic pressure, and volumetric face-flux
        arrays; ``last_removed_maximum`` is updated with global L-infinity
        changes for diagnostics.
        """
        velocity_before = np.array(solver.velocity[: self.n_owned], copy=True)
        pressure_before = np.array(solver.kinematic_pressure[: self.n_owned], copy=True)
        flux_before = np.array(solver.volumetric_face_flux, copy=True)

        mean_velocity = self._cell_mean(solver.velocity)
        solver.velocity[: self.n_owned] = mean_velocity[self.cell_groups]
        solver.velocity[: self.n_owned, 2] = 0.0
        mean_pressure = self._cell_mean(solver.kinematic_pressure)
        solver.kinematic_pressure[: self.n_owned] = mean_pressure[self.cell_groups]

        mean_flux = self._face_mean(solver.volumetric_face_flux)
        solver.volumetric_face_flux[self.vertical_faces] = mean_flux[
            self.face_groups[self.vertical_faces]
        ]
        solver.volumetric_face_flux[self.horizontal_faces] = 0.0

        local_removed = {
            "velocity": float(np.max(np.abs(velocity_before - solver.velocity[: self.n_owned]))),
            "kinematic_pressure": float(
                np.max(np.abs(pressure_before - solver.kinematic_pressure[: self.n_owned]))
            ),
            "volumetric_face_flux": float(
                np.max(np.abs(flux_before - solver.volumetric_face_flux))
            ),
        }
        self.last_removed_maximum = {
            name: float(self.parallel.global_max(value)) for name, value in local_removed.items()
        }
