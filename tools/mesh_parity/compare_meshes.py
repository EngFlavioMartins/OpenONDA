"""Topology-aware cfMesh/OpenONDA mesh comparison.

Raw OpenFOAM file order is not a mesh invariant.  This module first checks
numbering-independent Level-A invariants, then constructs a bounded spatial
cell correspondence and verifies dual-graph and face topology under that
correspondence.  It intentionally avoids an unconstrained graph-isomorphism
search, which is not practical for reference meshes with hundreds of thousands
of cells.
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .mesh_fingerprint import (
    MeshFingerprint,
    cell_face_counts,
    cell_neighbour_counts,
    fingerprint_differences,
    fingerprint_mesh,
    internal_face_pairs,
)
from .openfoam_poly_mesh import PolyMesh


@dataclass(frozen=True, slots=True)
class ComparisonOptions:
    """Explicit, deliberately tight tolerances for a parity comparison."""

    centroid_relative_tolerance: float = 1.0e-8
    centroid_absolute_tolerance: float = 1.0e-10
    volume_relative_tolerance: float = 1.0e-8
    face_normal_angle_tolerance_degrees: float = 1.0e-5
    candidate_limit: int = 12
    assignment_component_limit: int = 512

    def __post_init__(self) -> None:
        positive = {
            "centroid_relative_tolerance": self.centroid_relative_tolerance,
            "centroid_absolute_tolerance": self.centroid_absolute_tolerance,
            "volume_relative_tolerance": self.volume_relative_tolerance,
            "face_normal_angle_tolerance_degrees": self.face_normal_angle_tolerance_degrees,
        }
        for name, value in positive.items():
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.candidate_limit < 1:
            raise ValueError("candidate_limit must be positive")
        if self.assignment_component_limit < 1:
            raise ValueError("assignment_component_limit must be positive")

    def to_dict(self) -> dict[str, float | int]:
        """Return every numerical gate for inclusion in audit reports."""
        return asdict(self)


@dataclass(slots=True)
class _MeshTopology:
    patch_names: tuple[str, ...]
    face_counts: np.ndarray
    neighbour_counts: np.ndarray
    patch_incidence: np.ndarray
    face_patch_ids: np.ndarray
    adjacency_offsets: np.ndarray
    adjacency_indices: np.ndarray

    def neighbours_of(self, cell: int) -> np.ndarray:
        return self.adjacency_indices[
            self.adjacency_offsets[cell] : self.adjacency_offsets[cell + 1]
        ]


@dataclass(slots=True)
class _MeshGeometry:
    cell_centres: np.ndarray
    cell_volumes: np.ndarray
    face_centres: np.ndarray
    face_area_vectors: np.ndarray
    bounding_box: dict[str, list[float]]
    total_volume: float


@dataclass(frozen=True, slots=True)
class ComparisonResult:
    """JSON-ready outcome of all parity levels for one pair of meshes."""

    passed: bool
    cfmesh: MeshFingerprint
    openonda: MeshFingerprint
    invariant_differences: dict[str, dict[str, Any]]
    cell_mapping: dict[str, Any]
    topology: dict[str, Any]
    geometry: dict[str, Any]
    first_failure: str | None

    def to_dict(self) -> dict[str, Any]:
        """Return a stable report payload without raw numbering arrays."""
        return {
            "passed": self.passed,
            "cfmesh": self.cfmesh.to_dict(),
            "openonda": self.openonda.to_dict(),
            "invariant_differences": self.invariant_differences,
            "cell_mapping": self.cell_mapping,
            "topology": self.topology,
            "geometry": self.geometry,
            "first_failure": self.first_failure,
        }


def _mesh_topology(mesh: PolyMesh, patch_names: Sequence[str]) -> _MeshTopology:
    patch_names = tuple(patch_names)
    patch_index = {name: index for index, name in enumerate(patch_names)}
    face_patch_ids = np.full(mesh.n_faces, -1, dtype=np.int32)
    patch_incidence = np.zeros((mesh.n_cells, len(patch_names)), dtype=np.int32)
    for patch in mesh.boundary:
        patch_id = patch_index[patch.name]
        start = patch.start_face
        stop = start + patch.n_faces
        face_patch_ids[start:stop] = patch_id
        np.add.at(patch_incidence[:, patch_id], mesh.owner[start:stop], 1)

    pairs = internal_face_pairs(mesh)
    if len(pairs):
        cells = np.concatenate((pairs[:, 0], pairs[:, 1]))
        neighbours = np.concatenate((pairs[:, 1], pairs[:, 0]))
        order = np.argsort(cells, kind="stable")
        cells = cells[order]
        neighbours = neighbours[order]
        counts = np.bincount(cells, minlength=mesh.n_cells)
        offsets = np.concatenate((np.array([0], dtype=np.int64), np.cumsum(counts)))
    else:
        neighbours = np.empty(0, dtype=np.int64)
        offsets = np.zeros(mesh.n_cells + 1, dtype=np.int64)
    return _MeshTopology(
        patch_names=patch_names,
        face_counts=cell_face_counts(mesh),
        neighbour_counts=cell_neighbour_counts(mesh),
        patch_incidence=patch_incidence,
        face_patch_ids=face_patch_ids,
        adjacency_offsets=offsets,
        adjacency_indices=neighbours,
    )


def _mesh_geometry(mesh: PolyMesh) -> _MeshGeometry:
    """Compute closed-polyhedron centres/volumes and per-face geometry.

    Faces are assumed to be outward for their owner and inward for their
    neighbour, the OpenFOAM convention.  Each polygon is decomposed around its
    arithmetic vertex centre, making the result invariant to the starting
    vertex of a non-planar face.  A face-centre mean is used only for a
    degenerate zero-volume fallback so a malformed mesh remains diagnosable.
    """
    n_faces = mesh.n_faces
    face_centres = np.empty((n_faces, 3), dtype=np.float64)
    face_area_vectors = np.empty((n_faces, 3), dtype=np.float64)
    face_volume = np.empty(n_faces, dtype=np.float64)
    face_moment = np.empty((n_faces, 3), dtype=np.float64)
    for face_id, face in enumerate(mesh.faces):
        vertices = mesh.points[face]
        centre = vertices.mean(axis=0)
        following = np.roll(vertices, -1, axis=0)
        twice_area_vectors = np.cross(vertices - centre, following - centre)
        twice_areas = np.linalg.norm(twice_area_vectors, axis=1)
        area_sum = float(twice_areas.sum())
        face_centres[face_id] = (
            np.einsum(
                "i,ij->j",
                twice_areas,
                vertices + following + centre,
            )
            / (3.0 * area_sum)
            if area_sum > np.finfo(np.float64).tiny
            else centre
        )
        face_area_vectors[face_id] = 0.5 * twice_area_vectors.sum(axis=0)
        tetrahedra = (
            np.einsum(
                "ij,ij->i",
                vertices,
                np.cross(following, np.broadcast_to(centre, vertices.shape)),
            )
            / 6.0
        )
        face_volume[face_id] = tetrahedra.sum()
        face_moment[face_id] = np.sum(
            tetrahedra[:, None] * (vertices + following + centre) / 4.0,
            axis=0,
        )

    signed_volumes = np.zeros(mesh.n_cells, dtype=np.float64)
    moments = np.zeros((mesh.n_cells, 3), dtype=np.float64)
    face_centre_sum = np.zeros((mesh.n_cells, 3), dtype=np.float64)
    face_count = np.zeros(mesh.n_cells, dtype=np.int64)
    np.add.at(signed_volumes, mesh.owner, face_volume)
    np.add.at(moments, mesh.owner, face_moment)
    np.add.at(face_centre_sum, mesh.owner, face_centres)
    np.add.at(face_count, mesh.owner, 1)
    if mesh.n_internal_faces:
        np.add.at(signed_volumes, mesh.neighbour, -face_volume[: mesh.n_internal_faces])
        np.add.at(moments, mesh.neighbour, -face_moment[: mesh.n_internal_faces])
        np.add.at(face_centre_sum, mesh.neighbour, face_centres[: mesh.n_internal_faces])
        np.add.at(face_count, mesh.neighbour, 1)
    centres = np.empty((mesh.n_cells, 3), dtype=np.float64)
    usable = np.abs(signed_volumes) > np.finfo(np.float64).eps
    centres[usable] = moments[usable] / signed_volumes[usable, None]
    centres[~usable] = face_centre_sum[~usable] / np.maximum(face_count[~usable, None], 1)
    lower = mesh.points.min(axis=0) if len(mesh.points) else np.zeros(3)
    upper = mesh.points.max(axis=0) if len(mesh.points) else np.zeros(3)
    return _MeshGeometry(
        cell_centres=centres,
        cell_volumes=np.abs(signed_volumes),
        face_centres=face_centres,
        face_area_vectors=face_area_vectors,
        bounding_box={"min": lower.tolist(), "max": upper.tolist()},
        total_volume=float(np.abs(signed_volumes).sum()),
    )


def _candidate_sets(
    reference: _MeshGeometry,
    candidate: _MeshGeometry,
    reference_topology: _MeshTopology,
    candidate_topology: _MeshTopology,
    options: ComparisonOptions,
) -> tuple[list[tuple[int, ...]], float]:
    """Create tightly constrained nearby target-cell candidates in batches."""
    try:
        from scipy.spatial import cKDTree
    except ImportError as error:  # pragma: no cover - scipy is an FVM dependency.
        raise RuntimeError("scipy is required to build the parity cell correspondence") from error

    all_points = np.concatenate((reference.cell_centres, candidate.cell_centres), axis=0)
    span = float(np.linalg.norm(all_points.max(axis=0) - all_points.min(axis=0)))
    tolerance = max(options.centroid_absolute_tolerance, options.centroid_relative_tolerance * span)
    tree = cKDTree(candidate.cell_centres)
    results: list[tuple[int, ...]] = []
    block_size = 50_000
    for start in range(0, len(reference.cell_centres), block_size):
        stop = min(start + block_size, len(reference.cell_centres))
        distances, indices = tree.query(
            reference.cell_centres[start:stop],
            k=options.candidate_limit,
            distance_upper_bound=tolerance,
        )
        distances = np.asarray(distances)
        indices = np.asarray(indices)
        if options.candidate_limit == 1:
            distances = distances[:, None]
            indices = indices[:, None]
        for offset, (row_distances, row_indices) in enumerate(zip(distances, indices, strict=True)):
            source = start + offset
            accepted: list[int] = []
            for distance, target in zip(row_distances, row_indices, strict=True):
                target = int(target)
                if target >= len(candidate.cell_centres) or not np.isfinite(distance):
                    continue
                if reference_topology.face_counts[source] != candidate_topology.face_counts[target]:
                    continue
                if (
                    reference_topology.neighbour_counts[source]
                    != candidate_topology.neighbour_counts[target]
                ):
                    continue
                if not np.array_equal(
                    reference_topology.patch_incidence[source],
                    candidate_topology.patch_incidence[target],
                ):
                    continue
                scale = max(
                    float(reference.cell_volumes[source]),
                    float(candidate.cell_volumes[target]),
                    np.finfo(np.float64).tiny,
                )
                if (
                    abs(reference.cell_volumes[source] - candidate.cell_volumes[target]) / scale
                    > options.volume_relative_tolerance
                ):
                    continue
                accepted.append(target)
            results.append(tuple(accepted))
    return results, tolerance


def _free_candidates(
    source: int, candidates: Sequence[tuple[int, ...]], target_source: np.ndarray
) -> tuple[int, ...]:
    return tuple(target for target in candidates[source] if target_source[target] < 0)


def _assign_unique_candidates(
    candidates: Sequence[tuple[int, ...]], mapping: np.ndarray, target_source: np.ndarray
) -> int:
    """Commit one-to-one singleton candidate assignments until stable."""
    assigned = 0
    while True:
        claims: dict[int, list[int]] = defaultdict(list)
        for source in np.flatnonzero(mapping < 0):
            free = _free_candidates(int(source), candidates, target_source)
            if len(free) == 1:
                claims[free[0]].append(int(source))
        round_assignments = [
            (sources[0], target) for target, sources in claims.items() if len(sources) == 1
        ]
        if not round_assignments:
            return assigned
        for source, target in round_assignments:
            mapping[source] = target
            target_source[target] = source
        assigned += len(round_assignments)


def _assign_neighbour_evidence(
    candidates: Sequence[tuple[int, ...]],
    mapping: np.ndarray,
    target_source: np.ndarray,
    reference_topology: _MeshTopology,
    candidate_topology: _MeshTopology,
) -> int:
    """Resolve candidates whose already-mapped neighbourhood gives unique evidence."""
    proposals: dict[int, list[int]] = defaultdict(list)
    for source_value in np.flatnonzero(mapping < 0):
        source = int(source_value)
        free = _free_candidates(source, candidates, target_source)
        if len(free) == 1:
            proposals[free[0]].append(source)
            continue
        if len(free) < 2:
            continue
        mapped_neighbours = mapping[reference_topology.neighbours_of(source)]
        mapped_neighbours = mapped_neighbours[mapped_neighbours >= 0]
        if not len(mapped_neighbours):
            continue
        scores: list[int] = []
        for target in free:
            target_neighbours = candidate_topology.neighbours_of(target)
            scores.append(int(np.count_nonzero(np.isin(mapped_neighbours, target_neighbours))))
        best = max(scores)
        if best > 0 and scores.count(best) == 1:
            proposals[free[scores.index(best)]].append(source)
    assignments = [
        (sources[0], target) for target, sources in proposals.items() if len(sources) == 1
    ]
    for source, target in assignments:
        mapping[source] = target
        target_source[target] = source
    return len(assignments)


def _assign_small_components(
    candidates: Sequence[tuple[int, ...]],
    mapping: np.ndarray,
    target_source: np.ndarray,
    reference: _MeshGeometry,
    candidate: _MeshGeometry,
    options: ComparisonOptions,
) -> tuple[int, int]:
    """Use minimum-distance matching only within small residual candidate components."""
    residual: dict[int, tuple[int, ...]] = {
        int(source): _free_candidates(int(source), candidates, target_source)
        for source in np.flatnonzero(mapping < 0)
    }
    target_sources: dict[int, list[int]] = defaultdict(list)
    for source, targets in residual.items():
        for target in targets:
            target_sources[target].append(source)
    assigned = 0
    too_large = 0
    visited: set[int] = set()
    for seed in residual:
        if seed in visited or not residual[seed]:
            continue
        sources: set[int] = set()
        targets: set[int] = set()
        pending: deque[tuple[str, int]] = deque((("source", seed),))
        while pending:
            kind, item = pending.popleft()
            if kind == "source":
                if item in sources:
                    continue
                sources.add(item)
                visited.add(item)
                pending.extend(("target", target) for target in residual[item])
            else:
                if item in targets:
                    continue
                targets.add(item)
                pending.extend(("source", source) for source in target_sources[item])
        if len(sources) != len(targets) or len(sources) > options.assignment_component_limit:
            too_large += len(sources)
            continue
        try:
            from scipy.optimize import linear_sum_assignment
        except ImportError as error:  # pragma: no cover - scipy is an FVM dependency.
            raise RuntimeError("scipy is required to resolve parity candidates") from error
        source_ids = np.asarray(sorted(sources), dtype=np.int64)
        target_ids = np.asarray(sorted(targets), dtype=np.int64)
        target_column = {int(target): index for index, target in enumerate(target_ids)}
        costs = np.full((len(source_ids), len(target_ids)), np.inf, dtype=np.float64)
        for row, source in enumerate(source_ids):
            for target in residual[int(source)]:
                column = target_column[target]
                costs[row, column] = float(
                    np.linalg.norm(reference.cell_centres[source] - candidate.cell_centres[target])
                )
        rows, columns = linear_sum_assignment(costs)
        if len(rows) != len(source_ids) or not np.isfinite(costs[rows, columns]).all():
            too_large += len(sources)
            continue
        for row, column in zip(rows, columns, strict=True):
            source = int(source_ids[row])
            target = int(target_ids[column])
            mapping[source] = target
            target_source[target] = source
        assigned += len(source_ids)
    return assigned, too_large


def _build_cell_mapping(
    reference: _MeshGeometry,
    candidate: _MeshGeometry,
    reference_topology: _MeshTopology,
    candidate_topology: _MeshTopology,
    options: ComparisonOptions,
) -> tuple[np.ndarray, dict[str, Any]]:
    candidates, tolerance = _candidate_sets(
        reference, candidate, reference_topology, candidate_topology, options
    )
    mapping = np.full(len(candidates), -1, dtype=np.int64)
    target_source = np.full(len(candidate.cell_centres), -1, dtype=np.int64)
    initial_lengths = np.asarray([len(items) for items in candidates], dtype=np.int64)
    singleton_assignments = _assign_unique_candidates(candidates, mapping, target_source)
    neighbour_assignments = 0
    for _ in range(12):
        added = _assign_neighbour_evidence(
            candidates, mapping, target_source, reference_topology, candidate_topology
        )
        added += _assign_unique_candidates(candidates, mapping, target_source)
        neighbour_assignments += added
        if not added:
            break
    component_assignments, too_large = _assign_small_components(
        candidates, mapping, target_source, reference, candidate, options
    )
    unresolved = np.flatnonzero(mapping < 0)
    extra_targets = np.flatnonzero(target_source < 0)
    diagnostic = {
        "complete": bool(not len(unresolved) and not len(extra_targets)),
        "matched_cells": int(np.count_nonzero(mapping >= 0)),
        "unmatched_cfmesh_cells": int(len(unresolved)),
        "unmatched_openonda_cells": int(len(extra_targets)),
        "candidate_centroid_tolerance": tolerance,
        "candidate_count": {
            "minimum": int(initial_lengths.min()) if len(initial_lengths) else 0,
            "maximum": int(initial_lengths.max(initial=0)),
            "mean": float(initial_lengths.mean() if len(initial_lengths) else 0.0),
            "zero": int(np.count_nonzero(initial_lengths == 0)),
        },
        "assignments": {
            "unique": singleton_assignments,
            "neighbour_evidence": neighbour_assignments,
            "small_component": component_assignments,
            "components_too_large_or_non_bijective": too_large,
        },
        "first_unmatched_cfmesh_cell_centre": (
            reference.cell_centres[int(unresolved[0])].tolist() if len(unresolved) else None
        ),
    }
    return mapping, diagnostic


def _counter_mismatch(left: Counter[Any], right: Counter[Any]) -> tuple[int, list[str]]:
    only_left = left - right
    only_right = right - left
    count = max(sum(only_left.values()), sum(only_right.values()))
    examples = [
        f"cfMesh only: {item!r} x{count}" for item, count in list(only_left.items())[:3]
    ] + [f"OpenONDA only: {item!r} x{count}" for item, count in list(only_right.items())[:3]]
    return int(count), examples


def _mapped_topology(
    reference_mesh: PolyMesh,
    candidate_mesh: PolyMesh,
    reference_topology: _MeshTopology,
    candidate_topology: _MeshTopology,
    mapping: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray]:
    """Compare cell adjacency, patch incidence, and face valence after mapping."""
    reference_pairs = internal_face_pairs(reference_mesh)
    candidate_pairs = internal_face_pairs(candidate_mesh)
    mapped_pairs = np.sort(mapping[reference_pairs], axis=1)
    adjacency_count, adjacency_examples = _counter_mismatch(
        Counter(map(tuple, mapped_pairs.tolist())), Counter(map(tuple, candidate_pairs.tolist()))
    )

    reference_internal = Counter(
        (
            int(min(mapping[int(owner)], mapping[int(neighbour)])),
            int(max(mapping[int(owner)], mapping[int(neighbour)])),
            len(reference_mesh.faces[face_id]),
        )
        for face_id, (owner, neighbour) in enumerate(
            zip(
                reference_mesh.owner[: reference_mesh.n_internal_faces],
                reference_mesh.neighbour,
                strict=True,
            )
        )
    )
    candidate_internal = Counter(
        (
            int(min(owner, neighbour)),
            int(max(owner, neighbour)),
            len(candidate_mesh.faces[face_id]),
        )
        for face_id, (owner, neighbour) in enumerate(
            zip(
                candidate_mesh.owner[: candidate_mesh.n_internal_faces],
                candidate_mesh.neighbour,
                strict=True,
            )
        )
    )
    internal_face_mismatch, internal_face_examples = _counter_mismatch(
        reference_internal, candidate_internal
    )

    reference_boundary = Counter(
        (
            int(mapping[int(reference_mesh.owner[face_id])]),
            reference_topology.patch_names[int(reference_topology.face_patch_ids[face_id])],
            len(reference_mesh.faces[face_id]),
        )
        for face_id in range(reference_mesh.n_internal_faces, reference_mesh.n_faces)
    )
    candidate_boundary = Counter(
        (
            int(candidate_mesh.owner[face_id]),
            candidate_topology.patch_names[int(candidate_topology.face_patch_ids[face_id])],
            len(candidate_mesh.faces[face_id]),
        )
        for face_id in range(candidate_mesh.n_internal_faces, candidate_mesh.n_faces)
    )
    boundary_face_mismatch, boundary_face_examples = _counter_mismatch(
        reference_boundary, candidate_boundary
    )
    mapped_incidence = np.empty_like(reference_topology.patch_incidence)
    mapped_incidence[mapping] = reference_topology.patch_incidence
    patch_mismatch_cells = np.flatnonzero(
        np.any(mapped_incidence != candidate_topology.patch_incidence, axis=1)
    )

    mismatch_source_cells: set[int] = set()
    candidate_pair_set = {tuple(pair) for pair in candidate_pairs.tolist()}
    for source_first, source_second in reference_pairs:
        mapped_pair = tuple(sorted((int(mapping[source_first]), int(mapping[source_second]))))
        if mapped_pair not in candidate_pair_set:
            mismatch_source_cells.update((int(source_first), int(source_second)))
            if len(mismatch_source_cells) >= 10_000:
                break
    inverse_mapping = np.empty_like(mapping)
    inverse_mapping[mapping] = np.arange(len(mapping), dtype=np.int64)
    mismatch_source_cells.update(
        int(inverse_mapping[target]) for target in patch_mismatch_cells[:10_000]
    )
    return (
        {
            "adjacency_mismatches": adjacency_count,
            "adjacency_examples": adjacency_examples,
            "patch_incidence_mismatches": int(len(patch_mismatch_cells)),
            "internal_face_topology_mismatches": internal_face_mismatch,
            "boundary_face_topology_mismatches": boundary_face_mismatch,
            "face_topology_mismatches": internal_face_mismatch + boundary_face_mismatch,
            "face_topology_examples": internal_face_examples + boundary_face_examples,
        },
        np.asarray(sorted(mismatch_source_cells), dtype=np.int64),
    )


def _statistics(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return {"max": 0.0, "mean": 0.0, "rms": 0.0, "p95": 0.0, "p99": 0.0}
    return {
        "max": float(values.max()),
        "mean": float(values.mean()),
        "rms": float(np.sqrt(np.mean(values**2))),
        "p95": float(np.percentile(values, 95.0)),
        "p99": float(np.percentile(values, 99.0)),
    }


def _boundary_face_pairs(
    reference_mesh: PolyMesh,
    candidate_mesh: PolyMesh,
    reference_topology: _MeshTopology,
    candidate_topology: _MeshTopology,
    mapping: np.ndarray,
) -> list[tuple[int, int]]:
    """Pair boundary faces by mapped owner, patch, and face valence."""
    reference_groups: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    candidate_groups: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for face_id in range(reference_mesh.n_internal_faces, reference_mesh.n_faces):
        reference_groups[
            (
                int(mapping[reference_mesh.owner[face_id]]),
                int(reference_topology.face_patch_ids[face_id]),
                len(reference_mesh.faces[face_id]),
            )
        ].append(face_id)
    for face_id in range(candidate_mesh.n_internal_faces, candidate_mesh.n_faces):
        candidate_groups[
            (
                int(candidate_mesh.owner[face_id]),
                int(candidate_topology.face_patch_ids[face_id]),
                len(candidate_mesh.faces[face_id]),
            )
        ].append(face_id)
    pairs: list[tuple[int, int]] = []
    for key in sorted(reference_groups):
        source_faces = reference_groups[key]
        target_faces = candidate_groups.get(key, [])
        if len(source_faces) != len(target_faces):
            continue
        # A manifold FVM cell normally has at most one face of one patch.  For
        # rare repeated faces, deterministic face-id pairing remains diagnostic
        # because topology equality has already been checked above.
        pairs.extend(zip(sorted(source_faces), sorted(target_faces), strict=True))
    return pairs


def _wall_distance_statistics(
    mesh: PolyMesh,
    topology: _MeshTopology,
    surface_triangles: Mapping[str, np.ndarray],
) -> dict[str, dict[str, float]]:
    """Measure exact wall vertices against the authoritative input STL per patch."""
    if not surface_triangles:
        return {}
    try:
        from source.solvers.fvm.mesh.surface_classification import SurfaceIndex
    except ImportError as error:  # pragma: no cover - project-local import.
        raise RuntimeError(
            "OpenONDA's surface index is required for wall-distance metrics"
        ) from error
    result: dict[str, dict[str, float]] = {}
    for patch_name, triangles in surface_triangles.items():
        if patch_name not in topology.patch_names:
            continue
        patch_id = topology.patch_names.index(patch_name)
        face_ids = np.flatnonzero(topology.face_patch_ids == patch_id)
        if not len(face_ids):
            result[patch_name] = _statistics(np.empty(0))
            continue
        point_ids = np.unique(np.concatenate([mesh.faces[int(face_id)] for face_id in face_ids]))
        _closest, distances, _triangle_ids = SurfaceIndex.build(
            np.asarray(triangles, dtype=np.float64)
        ).nearest_points(mesh.points[point_ids])
        result[patch_name] = _statistics(distances)
    return result


def _geometry_comparison(
    reference_mesh: PolyMesh,
    candidate_mesh: PolyMesh,
    reference_topology: _MeshTopology,
    candidate_topology: _MeshTopology,
    reference: _MeshGeometry,
    candidate: _MeshGeometry,
    mapping: np.ndarray,
    surface_triangles: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    mapped_centres = candidate.cell_centres[mapping]
    cell_centre_distances = np.linalg.norm(reference.cell_centres - mapped_centres, axis=1)
    mapped_volumes = candidate.cell_volumes[mapping]
    volume_absolute = np.abs(reference.cell_volumes - mapped_volumes)
    volume_relative = volume_absolute / np.maximum(
        np.maximum(reference.cell_volumes, mapped_volumes), np.finfo(np.float64).tiny
    )
    face_pairs = _boundary_face_pairs(
        reference_mesh, candidate_mesh, reference_topology, candidate_topology, mapping
    )
    if face_pairs:
        source_faces = np.asarray([pair[0] for pair in face_pairs], dtype=np.int64)
        target_faces = np.asarray([pair[1] for pair in face_pairs], dtype=np.int64)
        face_centre_distances = np.linalg.norm(
            reference.face_centres[source_faces] - candidate.face_centres[target_faces], axis=1
        )
        source_normals = reference.face_area_vectors[source_faces]
        target_normals = candidate.face_area_vectors[target_faces]
        source_lengths = np.linalg.norm(source_normals, axis=1)
        target_lengths = np.linalg.norm(target_normals, axis=1)
        valid = (source_lengths > 0.0) & (target_lengths > 0.0)
        normal_angles = np.full(len(source_faces), 180.0, dtype=np.float64)
        dot = np.einsum("ij,ij->i", source_normals[valid], target_normals[valid])
        dot /= source_lengths[valid] * target_lengths[valid]
        normal_angles[valid] = np.degrees(np.arccos(np.clip(dot, -1.0, 1.0)))
    else:
        face_centre_distances = np.empty(0, dtype=np.float64)
        normal_angles = np.empty(0, dtype=np.float64)
    return {
        "cell_centre_distance": _statistics(cell_centre_distances),
        "cell_volume_absolute_error": _statistics(volume_absolute),
        "cell_volume_relative_error": _statistics(volume_relative),
        "boundary_face_centre_distance": _statistics(face_centre_distances),
        "boundary_face_normal_angle_degrees": _statistics(normal_angles),
        "boundary_face_pairs_compared": len(face_pairs),
        "cfmesh_total_fluid_volume": reference.total_volume,
        "openonda_total_fluid_volume": candidate.total_volume,
        "total_fluid_volume_absolute_error": abs(reference.total_volume - candidate.total_volume),
        "cfmesh_bounding_box": reference.bounding_box,
        "openonda_bounding_box": candidate.bounding_box,
        "wall_point_distance_to_stl": {
            "cfmesh": _wall_distance_statistics(
                reference_mesh, reference_topology, surface_triangles
            ),
            "openonda": _wall_distance_statistics(
                candidate_mesh, candidate_topology, surface_triangles
            ),
        },
    }


def _first_bad_region(
    reference: _MeshGeometry, source_cells: np.ndarray
) -> dict[str, list[float]] | None:
    if not len(source_cells):
        return None
    centres = reference.cell_centres[source_cells]
    return {"min": centres.min(axis=0).tolist(), "max": centres.max(axis=0).tolist()}


def compare_meshes(
    cfmesh: PolyMesh,
    openonda: PolyMesh,
    *,
    surface_triangles: Mapping[str, np.ndarray] | None = None,
    options: ComparisonOptions | None = None,
) -> ComparisonResult:
    """Compare two equivalent meshes up to point, face, and cell relabelling.

    ``cfmesh`` is the reference only by convention; no path in this module
    invokes cfMesh.  The early Level-A exit is intentional: graph matching
    cannot prove equivalence when mandatory global invariants already differ.
    """
    options = options or ComparisonOptions()
    surface_triangles = surface_triangles or {}
    cfmesh_fingerprint = fingerprint_mesh(cfmesh)
    openonda_fingerprint = fingerprint_mesh(openonda)
    invariant_differences = fingerprint_differences(cfmesh_fingerprint, openonda_fingerprint)
    if invariant_differences:
        return ComparisonResult(
            passed=False,
            cfmesh=cfmesh_fingerprint,
            openonda=openonda_fingerprint,
            invariant_differences=invariant_differences,
            cell_mapping={"complete": False, "reason": "level_a_invariants_differ"},
            topology={
                "adjacency_mismatches": None,
                "patch_incidence_mismatches": None,
                "face_topology_mismatches": None,
                "first_differing_spatial_region": None,
            },
            geometry={},
            first_failure="level_a_invariants",
        )

    patch_names = tuple(sorted(patch.name for patch in cfmesh.boundary))
    cfmesh_topology = _mesh_topology(cfmesh, patch_names)
    openonda_topology = _mesh_topology(openonda, patch_names)
    cfmesh_geometry = _mesh_geometry(cfmesh)
    openonda_geometry = _mesh_geometry(openonda)
    mapping, mapping_report = _build_cell_mapping(
        cfmesh_geometry,
        openonda_geometry,
        cfmesh_topology,
        openonda_topology,
        options,
    )
    unmatched = np.flatnonzero(mapping < 0)
    if len(unmatched):
        return ComparisonResult(
            passed=False,
            cfmesh=cfmesh_fingerprint,
            openonda=openonda_fingerprint,
            invariant_differences={},
            cell_mapping=mapping_report,
            topology={
                "adjacency_mismatches": None,
                "patch_incidence_mismatches": None,
                "face_topology_mismatches": None,
                "first_differing_spatial_region": _first_bad_region(cfmesh_geometry, unmatched),
            },
            geometry={},
            first_failure="cell_mapping",
        )

    topology, mismatch_source_cells = _mapped_topology(
        cfmesh,
        openonda,
        cfmesh_topology,
        openonda_topology,
        mapping,
    )
    topology["first_differing_spatial_region"] = _first_bad_region(
        cfmesh_geometry, mismatch_source_cells
    )
    geometry = _geometry_comparison(
        cfmesh,
        openonda,
        cfmesh_topology,
        openonda_topology,
        cfmesh_geometry,
        openonda_geometry,
        mapping,
        surface_triangles,
    )
    topology_failed = any(
        topology[key] != 0
        for key in (
            "adjacency_mismatches",
            "patch_incidence_mismatches",
            "face_topology_mismatches",
        )
    )
    geometry_failed = (
        geometry["cell_centre_distance"]["max"] > mapping_report["candidate_centroid_tolerance"]
        or geometry["cell_volume_relative_error"]["max"] > options.volume_relative_tolerance
        or geometry["boundary_face_centre_distance"]["max"]
        > mapping_report["candidate_centroid_tolerance"]
        or geometry["boundary_face_normal_angle_degrees"]["max"]
        > options.face_normal_angle_tolerance_degrees
    )
    return ComparisonResult(
        passed=not topology_failed and not geometry_failed,
        cfmesh=cfmesh_fingerprint,
        openonda=openonda_fingerprint,
        invariant_differences={},
        cell_mapping=mapping_report,
        topology=topology,
        geometry=geometry,
        first_failure=(
            "cell_adjacency" if topology_failed else "geometry" if geometry_failed else None
        ),
    )


__all__ = ["ComparisonOptions", "ComparisonResult", "compare_meshes"]
