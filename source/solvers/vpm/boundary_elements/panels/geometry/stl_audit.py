"""Fail-fast STL mesh-audit and topological-orientation checks for panel bodies."""

from __future__ import annotations

import json
import logging

import numpy as np

from .stl_io import _unique_face_vertex_ids_from_triangles

logger = logging.getLogger("vpm")


class StlAuditError(ValueError):
    """Raised when an STL mesh fails preflight validation for the panel solver.

    The validation functions in this module raise this exception (through
    the :func:`_require` helper) when the mesh cannot safely enter the panel
    solver as a body. The error message includes the specific check that
    failed and the offending counts.
    """


def _require(condition, message):
    if not condition:
        raise StlAuditError(message)


def _directed_edges_for_panel(vertex_ids_row: np.ndarray) -> tuple:
    a, b, c = (int(v) for v in vertex_ids_row)
    return ((a, b), (b, c), (c, a))


def _build_directed_edge_map(vertex_ids: np.ndarray) -> dict:
    """Map each undirected edge to the (panel, directed_edge) pairs that touch it."""
    edge_map: dict[tuple[int, int], list[tuple[int, tuple[int, int]]]] = {}
    for panel in range(vertex_ids.shape[0]):
        for directed in _directed_edges_for_panel(vertex_ids[panel]):
            key = (
                (directed[0], directed[1])
                if directed[0] < directed[1]
                else (directed[1], directed[0])
            )
            edge_map.setdefault(key, []).append((panel, directed))
    return edge_map


def _label_connected_components(edge_map: dict, panel_count: int) -> np.ndarray:
    """Union-find over shared edges; returns a component label per panel."""
    parent = list(range(panel_count))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        root_i, root_j = find(i), find(j)
        if root_i != root_j:
            parent[root_i] = root_j

    for occurrences in edge_map.values():
        panels = [panel for panel, _ in occurrences]
        for other in panels[1:]:
            union(panels[0], other)

    roots = [find(i) for i in range(panel_count)]
    root_to_label = {root: label for label, root in enumerate(sorted(set(roots)))}
    return np.array([root_to_label[root] for root in roots], dtype=np.int64)


def _check_edge_topology(edge_map: dict) -> tuple[int, int, int]:
    """Return ``(n_open_edges, n_nonmanifold_edges, n_inconsistent_winding_edges)``."""
    n_open = 0
    n_nonmanifold = 0
    n_inconsistent = 0
    for occurrences in edge_map.values():
        count = len(occurrences)
        if count == 1:
            n_open += 1
        elif count == 2:
            (_, first_directed), (_, second_directed) = occurrences
            if first_directed == second_directed:
                n_inconsistent += 1
        else:
            n_nonmanifold += 1
    return n_open, n_nonmanifold, n_inconsistent


def signed_volume(triangles: np.ndarray) -> float:
    """Signed enclosed volume of a triangle set via the divergence theorem."""
    v0, v1, v2 = triangles[:, 0, :], triangles[:, 1, :], triangles[:, 2, :]
    return float(np.sum(np.einsum("ij,ij->i", v0, np.cross(v1, v2))) / 6.0)


def orient_components_by_signed_volume(triangles: np.ndarray) -> np.ndarray:
    """Orient each closed connected component to outward-pointing winding.

    Uses the divergence-theorem signed volume of each connected component, a
    global topological property, instead of a per-panel direction test
    against the body's geometric centroid. The centroid test misorients
    concave bodies whenever the centroid is not uniformly "inside" relative
    to every panel; signed volume is correct for any closed, consistently
    wound component regardless of concavity.
    """
    triangles = np.asarray(triangles, dtype=np.float64)
    vertex_ids = _unique_face_vertex_ids_from_triangles(triangles)
    edge_map = _build_directed_edge_map(vertex_ids)
    labels = _label_connected_components(edge_map, triangles.shape[0])

    oriented = triangles.copy()
    for label in np.unique(labels):
        mask = labels == label
        if signed_volume(oriented[mask]) < 0.0:
            flipped = oriented[mask].copy()
            flipped[:, [1, 2], :] = flipped[:, [2, 1], :]
            oriented[mask] = flipped
    return oriented


def _flag_candidate_self_intersections(
    triangles: np.ndarray, vertex_ids: np.ndarray, *, proximity_factor: float = 0.25
) -> list[tuple[int, int]]:
    """Approximate, non-exhaustive proximity heuristic for likely self-intersections.

    Flags non-adjacent triangle pairs whose centroids are closer than
    ``proximity_factor`` times the sum of their circumradii, using a
    ``scipy.spatial.cKDTree`` broad phase. This is a proximity heuristic, not
    an exact triangle-triangle intersection test: it is meant to surface
    clearly overlapping/interpenetrating geometry for a human to review, per
    the "report, do not silently repair" policy, not to certify a mesh free
    of self-intersections.
    """
    from scipy.spatial import cKDTree

    n_triangles = triangles.shape[0]
    if n_triangles < 2:
        return []

    centroid = triangles.mean(axis=1)
    circumradius = np.max(np.linalg.norm(triangles - centroid[:, None, :], axis=2), axis=1)
    max_radius = float(np.max(circumradius))
    if max_radius <= 0.0:
        return []

    tree = cKDTree(centroid)
    pairs = tree.query_pairs(r=2.0 * max_radius, output_type="ndarray")

    flagged: list[tuple[int, int]] = []
    for panel_i, panel_j in pairs:
        panel_i, panel_j = int(panel_i), int(panel_j)
        if set(vertex_ids[panel_i]) & set(vertex_ids[panel_j]):
            continue  # shares a vertex: adjacency, not a candidate intersection
        threshold = proximity_factor * (circumradius[panel_i] + circumradius[panel_j])
        if np.linalg.norm(centroid[panel_i] - centroid[panel_j]) < threshold:
            flagged.append((panel_i, panel_j))
    return flagged


def validate_finite(triangles: np.ndarray) -> None:
    _require(
        bool(np.all(np.isfinite(triangles))),
        "STL mesh contains non-finite (NaN/Inf) vertex coordinates",
    )


def validate_no_degenerate_or_duplicate_triangles(
    triangles: np.ndarray, vertex_ids: np.ndarray
) -> tuple[int, int]:
    """Reject zero-area and duplicate triangles; return their counts."""
    edge1 = triangles[:, 1, :] - triangles[:, 0, :]
    edge2 = triangles[:, 2, :] - triangles[:, 0, :]
    area = 0.5 * np.linalg.norm(np.cross(edge1, edge2), axis=1)
    flat = triangles.reshape(-1, 3)
    bounding_box_scale = max(float(np.linalg.norm(flat.max(axis=0) - flat.min(axis=0))), 1.0)
    area_tolerance = (bounding_box_scale**2) * 1.0e-12
    n_degenerate = int(np.count_nonzero(area <= area_tolerance))
    _require(
        n_degenerate == 0, f"STL mesh contains {n_degenerate} zero-area/degenerate triangle(s)"
    )

    sorted_ids = np.sort(vertex_ids, axis=1)
    _, counts = np.unique(sorted_ids, axis=0, return_counts=True)
    n_duplicate = int(np.sum(counts[counts > 1] - 1))
    _require(n_duplicate == 0, f"STL mesh contains {n_duplicate} duplicate triangle(s)")
    return n_degenerate, n_duplicate


def validate_watertight_topology(edge_map: dict) -> tuple[int, int, int]:
    """Reject open edges, non-manifold edges, and inconsistent winding."""
    n_open, n_nonmanifold, n_inconsistent = _check_edge_topology(edge_map)
    _require(n_open == 0, f"STL mesh has {n_open} open edge(s); the body is not watertight")
    _require(n_nonmanifold == 0, f"STL mesh has {n_nonmanifold} non-manifold edge(s)")
    _require(
        n_inconsistent == 0,
        f"STL mesh has {n_inconsistent} edge(s) with inconsistent triangle winding",
    )
    return n_open, n_nonmanifold, n_inconsistent


def audit_stl_mesh(
    vertex_position: np.ndarray,
    *,
    max_panels: int | None = None,
    expected_components: int | None = None,
) -> dict:
    """Run the full panel-solver STL preflight and return a machine-readable report.

    Raises :class:`StlAuditError` on any condition that must not silently
    reach the panel solver: non-finite coordinates, zero-area/duplicate
    triangles, open or non-manifold edges, inconsistent winding, a panel
    count above ``max_panels``, or a disconnected-component count other than
    ``expected_components`` (or ``1`` when ``expected_components`` is not
    given, so a multi-body STL must be explicitly acknowledged rather than
    silently treated as one body). Aspect ratio, scale, and candidate
    self-intersections are reported, never rejected.

    Parameters
    ----------
    vertex_position : np.ndarray
        Triangle vertex_position with shape ``(N, 3, 3)``.
    max_panels : int | None
        Reject meshes with more triangles than this, before any allocation.
    expected_components : int | None
        Required number of disconnected watertight components. ``None``
        requires exactly one (a single closed body).

    Returns
    -------
    dict
        Audit report: ``n_triangles``, ``bounding_box``, ``component_count``,
        ``component_signed_volumes``, ``area_min/max/mean``,
        ``aspect_ratio_max``, ``n_open_edges``, ``n_nonmanifold_edges``,
        ``n_inconsistent_winding_edges``, ``n_degenerate_triangles``,
        ``n_duplicate_triangles``, ``candidate_self_intersections``,
        ``warnings``, ``disposition`` (``"pass"`` or ``"warn"``).

    Examples
    --------
    >>> report = audit_stl_mesh(vertex_position)
    >>> report["disposition"]
    'pass'
    """
    triangles = np.asarray(vertex_position, dtype=np.float64)
    _require(
        triangles.ndim == 3 and triangles.shape[1:] == (3, 3),
        "vertex_position must have shape (N, 3, 3)",
    )

    n_triangles = int(triangles.shape[0])
    if max_panels is not None:
        _require(
            n_triangles <= max_panels,
            f"STL mesh has {n_triangles} triangles, exceeding max_panels={max_panels}",
        )

    validate_finite(triangles)

    vertex_ids = _unique_face_vertex_ids_from_triangles(triangles)
    n_degenerate, n_duplicate = validate_no_degenerate_or_duplicate_triangles(triangles, vertex_ids)

    edge_map = _build_directed_edge_map(vertex_ids)
    n_open_edges, n_nonmanifold_edges, n_inconsistent_winding = validate_watertight_topology(
        edge_map
    )

    labels = _label_connected_components(edge_map, n_triangles)
    component_count = int(len(np.unique(labels))) if n_triangles else 0
    required_components = 1 if expected_components is None else expected_components
    _require(
        component_count == required_components,
        f"STL mesh has {component_count} disconnected component(s); "
        f"expected {required_components}"
        + (
            ""
            if expected_components is not None
            else " (pass expected_components to declare multiple bodies)"
        ),
    )

    component_signed_volumes = [
        signed_volume(triangles[labels == label]) for label in range(component_count)
    ]

    flat = triangles.reshape(-1, 3)
    bounding_box = (flat.min(axis=0).tolist(), flat.max(axis=0).tolist())

    edge1 = triangles[:, 1, :] - triangles[:, 0, :]
    edge2 = triangles[:, 2, :] - triangles[:, 0, :]
    edge3 = triangles[:, 2, :] - triangles[:, 1, :]
    area = 0.5 * np.linalg.norm(np.cross(edge1, edge2), axis=1)
    edge_lengths = np.stack(
        [
            np.linalg.norm(edge1, axis=1),
            np.linalg.norm(edge2, axis=1),
            np.linalg.norm(edge3, axis=1),
        ],
        axis=1,
    )
    aspect_ratio = edge_lengths.max(axis=1) / np.clip(edge_lengths.min(axis=1), 1.0e-300, None)
    max_aspect_ratio = float(np.max(aspect_ratio)) if n_triangles else 0.0

    warnings: list[str] = []
    if max_aspect_ratio > 100.0:
        warnings.append(f"maximum panel aspect ratio {max_aspect_ratio:.1f} is poor (>100)")

    candidate_self_intersections = _flag_candidate_self_intersections(triangles, vertex_ids)
    if candidate_self_intersections:
        warnings.append(
            f"{len(candidate_self_intersections)} triangle pair(s) are close enough to be "
            "candidate self-intersections (approximate proximity heuristic, not exhaustive)"
        )

    return {
        "n_triangles": n_triangles,
        "bounding_box": bounding_box,
        "component_count": component_count,
        "component_signed_volumes": component_signed_volumes,
        "area_min": float(np.min(area)) if n_triangles else 0.0,
        "area_max": float(np.max(area)) if n_triangles else 0.0,
        "area_mean": float(np.mean(area)) if n_triangles else 0.0,
        "aspect_ratio_max": max_aspect_ratio,
        "n_open_edges": n_open_edges,
        "n_nonmanifold_edges": n_nonmanifold_edges,
        "n_inconsistent_winding_edges": n_inconsistent_winding,
        "n_degenerate_triangles": n_degenerate,
        "n_duplicate_triangles": n_duplicate,
        "candidate_self_intersections": candidate_self_intersections[:20],
        "warnings": warnings,
        "disposition": "warn" if warnings else "pass",
    }


def write_audit_report_json(report: dict, path: str) -> None:
    """Write an ``audit_stl_mesh`` report to ``path`` as machine-readable JSON."""
    with open(path, "w") as handle:
        json.dump(report, handle, indent=2)
    logger.debug(f"Wrote STL audit report '{path}'.")
