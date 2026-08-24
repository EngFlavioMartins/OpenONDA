"""
VLM mesh generation: builds panel lattices from wing geometry and computes panel
panel_corner_position, topology, and bilinear coefficients.

Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Date: January 2026

Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

import numpy as np
import taichi as ti

from ....config.constants import VLM_SMALL_VELOCITY
from ..geometry.aircraft import Aircraft, WingSegment


def _stitch_symmetry_neighbors(aircraft: Aircraft, neigh_np: np.ndarray) -> None:
    """Connect wing root panels across the symmetry plane by updating neigh_np in-place."""
    current_idx = 0
    for _wing_uid, wing in aircraft.wings.items():
        if wing.symmetry > 0:
            for _seg_uid, segment in wing.segments.items():
                nc = segment.n_chordwise_panels
                ns = segment.n_spanwise_panels
                n_panels_seg = nc * ns
                start_orig = current_idx
                current_idx += n_panels_seg
                start_mirror = current_idx
                current_idx += n_panels_seg
                for i in range(nc):
                    idx_orig = start_orig + i
                    idx_mirror = start_mirror + i
                    neigh_np[idx_orig, 0] = idx_mirror
                    neigh_np[idx_mirror, 0] = idx_orig
        else:
            for seg in wing.segments.values():
                current_idx += seg.n_chordwise_panels * seg.n_spanwise_panels


def _stitch_segment_neighbors(blocks: list[dict], neigh_np: np.ndarray) -> None:
    """Connect contiguous spanwise segments belonging to the same wing half.

    OpenVSP imports commonly represent one blade/wing as many consecutive
    spanwise ``WingSegment`` objects.  Each segment has local structured
    neighbours, but without this stitch every segment boundary is treated as an
    artificial tip by the wake-shedding kernel.
    """
    by_half: dict[tuple[int, bool], list[dict]] = {}
    for block in blocks:
        by_half.setdefault((block["wing_id"], block["is_mirrored"]), []).append(block)

    for half_blocks in by_half.values():
        half_blocks.sort(key=lambda item: item["segment_order"])
        for lower, upper in zip(half_blocks[:-1], half_blocks[1:], strict=False):
            nc = min(lower["nc"], upper["nc"])
            lower_start = lower["start"] + (lower["ns"] - 1) * lower["nc"]
            upper_start = upper["start"]
            for i in range(nc):
                lower_idx = lower_start + i
                upper_idx = upper_start + i
                neigh_np[lower_idx, 1] = upper_idx
                neigh_np[upper_idx, 0] = lower_idx


def generate_vlm_mesh(
    aircraft: Aircraft,
    lattice,
    trailing_edge_infty: float = 10000.0,
    spanwise_spacing: str = "uniform",
    spanwise_spacing_ratio: float = 1.0,
    spanwise_spacing_region: str = "both",
) -> None:
    """
    Generate a VLM mesh from aircraft geometry.

    GPU-optimized: Collects all mesh data in numpy arrays, then does a single
    batch transfer to GPU. This avoids thousands of small CPU→GPU transfers
    that cause hangs on Vulkan backend.

    Args:
        aircraft: Aircraft geometry definition
        lattice: VLMLattice object to populate. Mesh arrays use its configured
            precision (``float32`` or ``float64``).
        trailing_edge_infty: Distance for "infinity" trailing legs (chord lengths)
        spanwise_spacing: Panel distribution method ('uniform' or 'geometric')
        spanwise_spacing_ratio: Concentration ratio for geometric spacing
        spanwise_spacing_region: Refinement region ('start', 'end', 'both')
    """
    # Pre-allocate numpy arrays for all mesh data
    max_n_panels = lattice.max_n_panels

    np_dtype = lattice.np_dtype
    corners_np = np.zeros((max_n_panels, 4, 3), dtype=np_dtype)
    vortex_np = np.zeros((max_n_panels, 4, 3), dtype=np_dtype)
    colloc_np = np.zeros((max_n_panels, 3), dtype=np_dtype)
    normals_np = np.zeros((max_n_panels, 3), dtype=np_dtype)
    areas_np = np.zeros(max_n_panels, dtype=np_dtype)
    bound_np = np.zeros((max_n_panels, 3), dtype=np_dtype)
    trail_np = np.zeros((max_n_panels, 2, 3), dtype=np_dtype)
    wing_id_np = np.zeros(max_n_panels, dtype=np.int32)
    seg_id_np = np.zeros(max_n_panels, dtype=np.int32)
    mirror_np = np.zeros(max_n_panels, dtype=np.int32)

    # Topology arrays
    te_np = np.zeros(max_n_panels, dtype=np.int32)
    neigh_np = np.full((max_n_panels, 4), -1, dtype=np.int32)
    te_idx_np = np.full(max_n_panels, -1, dtype=np.int32)

    panel_idx = 0
    wing_id = 0
    segment_blocks: list[dict] = []

    for _wing_uid, wing in aircraft.wings.items():
        total_ns = sum(seg.n_spanwise_panels for seg in wing.segments.values())
        global_edges = _compute_spanwise_edges(
            total_ns, spanwise_spacing, spanwise_spacing_ratio, spanwise_spacing_region
        )

        global_span_idx = 0

        for segment_id, (_seg_uid, segment) in enumerate(wing.segments.items()):
            ns_segment = segment.n_spanwise_panels

            segment_start = global_span_idx
            segment_end = global_span_idx + ns_segment
            segment_edges = global_edges[segment_start : segment_end + 1]

            s_start = segment_edges[0]
            s_end = segment_edges[-1]
            if s_end > s_start:
                local_edges = (segment_edges - s_start) / (s_end - s_start)
            else:
                local_edges = np.linspace(0, 1, ns_segment + 1)

            # Generate to numpy arrays (no GPU transfer yet)
            start_orig = panel_idx
            panel_idx = _generate_segment_to_numpy(
                segment,
                panel_idx,
                wing_id,
                segment_id,
                local_edges,
                corners_np,
                vortex_np,
                colloc_np,
                normals_np,
                areas_np,
                bound_np,
                trail_np,
                wing_id_np,
                seg_id_np,
                mirror_np,
                te_np,
                neigh_np,
                te_idx_np,
                is_mirrored=False,
                trailing_edge_infty=trailing_edge_infty,
            )
            segment_blocks.append(
                {
                    "wing_id": wing_id,
                    "segment_order": segment_id,
                    "is_mirrored": False,
                    "start": start_orig,
                    "nc": segment.n_chordwise_panels,
                    "ns": segment.n_spanwise_panels,
                }
            )

            if wing.symmetry > 0:
                # Use SAME edges as original (no reversal needed since we don't swap vertex_position)
                mirrored_edges = local_edges
                start_mirror = panel_idx
                panel_idx = _generate_segment_to_numpy(
                    segment,
                    panel_idx,
                    wing_id,
                    segment_id,
                    mirrored_edges,
                    corners_np,
                    vortex_np,
                    colloc_np,
                    normals_np,
                    areas_np,
                    bound_np,
                    trail_np,
                    wing_id_np,
                    seg_id_np,
                    mirror_np,
                    te_np,
                    neigh_np,
                    te_idx_np,
                    is_mirrored=True,
                    symmetry_plane=wing.symmetry,
                    trailing_edge_infty=trailing_edge_infty,
                )
                segment_blocks.append(
                    {
                        "wing_id": wing_id,
                        "segment_order": segment_id,
                        "is_mirrored": True,
                        "start": start_mirror,
                        "nc": segment.n_chordwise_panels,
                        "ns": segment.n_spanwise_panels,
                    }
                )

            global_span_idx += ns_segment
        wing_id += 1

    _stitch_symmetry_neighbors(aircraft, neigh_np)
    _stitch_segment_neighbors(segment_blocks, neigh_np)

    lattice.panel_corner_position.from_numpy(corners_np)
    lattice.vortex_point_position.from_numpy(vortex_np)
    lattice.collocation_point.from_numpy(colloc_np)
    lattice.normal.from_numpy(normals_np)
    lattice.area.from_numpy(areas_np)
    lattice.bound_vortex_midpoint.from_numpy(bound_np)
    lattice.trailing_direction.from_numpy(trail_np)
    lattice.wing_id.from_numpy(wing_id_np)
    lattice.segment_id.from_numpy(seg_id_np)
    lattice.is_mirrored.from_numpy(mirror_np)
    lattice.is_trailing_edge.from_numpy(te_np)
    lattice.neighbor_indices.from_numpy(neigh_np)
    lattice.trailing_edge_index.from_numpy(te_idx_np)

    lattice.n_panels = panel_idx

    # Mark LE panels (static topology; must run after neighbor_indices is uploaded)
    lattice.mark_le_panels()

    print(f"Generated VLM mesh: {lattice.n_panels} panels from {wing_id} wings", flush=True)


def _fill_panel_geometry(
    alpha,
    s_min,
    s_max,
    c_min,
    c_max,
    dc,
    idx,
    panel_corner_position,
    colloc,
    normal_out,
    area_out,
    trail,
    vortex,
    bound,
    is_mirrored,
):
    """Fill geometric arrays for one panel at *idx*."""
    P = _bilinear_interp(alpha, s_min, c_min)
    Q = _bilinear_interp(alpha, s_max, c_min)
    R = _bilinear_interp(alpha, s_max, c_max)
    S = _bilinear_interp(alpha, s_min, c_max)
    panel_corner_position[idx, 0], panel_corner_position[idx, 1] = P, Q
    panel_corner_position[idx, 2], panel_corner_position[idx, 3] = R, S
    s_coll = 0.5 * (s_min + s_max)
    c_coll = 0.5 * (c_min + c_max) + 0.25 * dc
    colloc[idx] = _bilinear_interp(alpha, s_coll, c_coll)
    diag1, diag2 = R - P, Q - S
    normal_raw = np.cross(diag1, diag2)
    panel_area = 0.5 * np.linalg.norm(normal_raw)
    panel_normal = (
        normal_raw / (2 * panel_area) if panel_area > 1e-12 else np.array([0.0, 0.0, 1.0])
    )
    if is_mirrored:
        panel_normal = -panel_normal
    normal_out[idx] = panel_normal
    area_out[idx] = panel_area
    trail_dir = 0.5 * (R + S) - 0.5 * (P + Q)
    trail_mag = np.linalg.norm(trail_dir)
    trail_dir = trail_dir / trail_mag if trail_mag > 1e-12 else np.array([1.0, 0.0, 0.0])
    trail[idx, 0], trail[idx, 1] = trail_dir, trail_dir
    c_vort_front = 0.5 * (c_min + c_max) - 0.25 * dc
    V2 = _bilinear_interp(alpha, s_min, c_vort_front)
    V3 = _bilinear_interp(alpha, s_max, c_vort_front)
    V1 = V2 + trail_dir * 1000.0
    V4 = V3 + trail_dir * 1000.0
    vortex[idx, 0], vortex[idx, 1] = V1, V2
    vortex[idx, 2], vortex[idx, 3] = V3, V4
    bound[idx] = 0.5 * (V2 + V3)


def _set_panel_topology(idx, i, j, nc, ns, te_flags, neighbors, te_idx):
    """Set trailing-edge flag and structured connectivity for one panel."""
    te_flags[idx] = 1 if i == nc - 1 else 0
    # TE panel of this panel's chordwise strip (same span station j).
    # Strip starts at idx - i, TE panel is at chordwise position nc - 1.
    te_idx[idx] = (idx - i) + (nc - 1)
    if i > 0:
        neighbors[idx, 2] = idx - 1
    if i < nc - 1:
        neighbors[idx, 3] = idx + 1
    if j > 0:
        neighbors[idx, 0] = idx - nc
    if j < ns - 1:
        neighbors[idx, 1] = idx + nc


def _generate_segment_to_numpy(
    segment: WingSegment,
    start_idx: int,
    wing_id: int,
    segment_id: int,
    s_edges: np.ndarray,
    panel_corner_position,
    vortex,
    colloc,
    normal,
    area,
    bound,
    trail,
    w_ids,
    s_ids,
    mirror,
    te_flags,
    neighbors,
    te_idx,
    is_mirrored: bool = False,
    symmetry_plane: int = 0,
    trailing_edge_infty: float = 10000.0,
) -> int:
    """
    Generate segment mesh data directly into numpy arrays.

    GPU-optimized helper that writes to pre-allocated numpy arrays,
    enabling single batch transfer after all segments are processed.
    """
    nc = segment.n_chordwise_panels
    ns = segment.n_spanwise_panels
    a, b, c, d = (
        segment.vertex_position["a"],
        segment.vertex_position["b"],
        segment.vertex_position["c"],
        segment.vertex_position["d"],
    )
    if is_mirrored:
        a, b, c, d = _mirror_vertices(a, b, c, d, symmetry_plane)
    alpha = _compute_bilinear_coefficients(a, b, c, d)
    dc = 1.0 / nc
    idx = start_idx
    for j in range(ns):
        for i in range(nc):
            s_min, s_max = s_edges[j], s_edges[j + 1]
            c_min, c_max = i * dc, (i + 1) * dc
            _fill_panel_geometry(
                alpha,
                s_min,
                s_max,
                c_min,
                c_max,
                dc,
                idx,
                panel_corner_position,
                colloc,
                normal,
                area,
                trail,
                vortex,
                bound,
                is_mirrored,
            )
            w_ids[idx], s_ids[idx] = wing_id, segment_id
            mirror[idx] = int(is_mirrored)
            _set_panel_topology(idx, i, j, nc, ns, te_flags, neighbors, te_idx)
            idx += 1
    return idx


def _compute_bilinear_coefficients(a, b, c, d):
    """Compute bilinear interpolation coefficients for segment vertex_position."""
    alpha = np.zeros((3, 4))  # [x,y,z] x [1, s, c, s*c]
    for dim in range(3):
        alpha[dim, 0] = a[dim]  # 1
        alpha[dim, 1] = b[dim] - a[dim]  # s
        alpha[dim, 2] = d[dim] - a[dim]  # c
        alpha[dim, 3] = a[dim] - b[dim] + c[dim] - d[dim]  # s*c
    return alpha


def _compute_panel_corners(alpha, s_min, s_max, c_min, c_max):
    """Compute panel corner position using bilinear interpolation."""
    P = _bilinear_interp(alpha, s_min, c_min)
    Q = _bilinear_interp(alpha, s_max, c_min)
    R = _bilinear_interp(alpha, s_max, c_max)
    S = _bilinear_interp(alpha, s_min, c_max)
    return P, Q, R, S


def _compute_panel_geometry(P, Q, R, S, is_mirrored, symmetry_plane):
    """Compute panel normal, area, and geometric properties."""
    # Horseshoe geometry (bound at 25% chord, collocation_point at 75%)
    V2 = 0.75 * P + 0.25 * S
    V3 = 0.75 * Q + 0.25 * R
    collocation_point = 0.125 * (P + Q) + 0.375 * (R + S)

    # Panel normal
    diag1 = R - P
    diag2 = S - Q
    normal = np.cross(diag2, diag1)
    normal_mag = np.linalg.norm(normal)
    normal = normal / normal_mag if normal_mag > 1e-10 else np.array([0.0, 0.0, 1.0])

    # Panel area
    area1 = 0.5 * np.linalg.norm(np.cross(Q - P, R - P))
    area2 = 0.5 * np.linalg.norm(np.cross(R - P, S - P))
    area = area1 + area2

    return V2, V3, collocation_point, normal, area


def _compute_trailing_geometry(V2, V3, trailing_edge_infty):
    """Compute trailing edge vortex points."""
    trail_dir = np.array([1.0, 0.0, 0.0])
    V1 = V2 + trail_dir * trailing_edge_infty
    V4 = V3 + trail_dir * trailing_edge_infty
    bound_midpoint = 0.5 * (V2 + V3)
    return V1, V2, V3, V4, bound_midpoint, trail_dir


def _store_panel_data(
    lattice,
    panel_idx,
    P,
    Q,
    R,
    S,
    V1,
    V2,
    V3,
    V4,
    collocation_point,
    normal,
    area,
    bound_midpoint,
    trail_dir,
    wing_id,
    segment_id,
    is_mirrored,
):
    """Store all panel data into lattice arrays."""
    # Corners
    for k in range(3):
        lattice.panel_corner_position[panel_idx, 0][k] = P[k]
        lattice.panel_corner_position[panel_idx, 1][k] = Q[k]
        lattice.panel_corner_position[panel_idx, 2][k] = R[k]
        lattice.panel_corner_position[panel_idx, 3][k] = S[k]

    # Vortex points
    for k in range(3):
        lattice.vortex_point_position[panel_idx, 0][k] = V1[k]
        lattice.vortex_point_position[panel_idx, 1][k] = V2[k]
        lattice.vortex_point_position[panel_idx, 2][k] = V3[k]
        lattice.vortex_point_position[panel_idx, 3][k] = V4[k]

    # Collocation, normal, area
    for k in range(3):
        lattice.collocation_point[panel_idx][k] = collocation_point[k]
        lattice.normal[panel_idx][k] = normal[k]
    lattice.area[panel_idx] = area

    # Bound midpoint
    for k in range(3):
        lattice.bound_vortex_midpoint[panel_idx][k] = bound_midpoint[k]

    # Trailing directions
    for k in range(3):
        lattice.trailing_direction[panel_idx, 0][k] = trail_dir[k]
        lattice.trailing_direction[panel_idx, 1][k] = trail_dir[k]

    # Bookkeeping
    lattice.wing_id[panel_idx] = wing_id
    lattice.segment_id[panel_idx] = segment_id
    lattice.is_mirrored[panel_idx] = 1 if is_mirrored else 0


def _generate_segment_mesh(
    segment: WingSegment,
    lattice,
    start_panel_idx: int,
    wing_id: int,
    segment_id: int,
    is_mirrored: bool = False,
    symmetry_plane: int = 0,
    trailing_edge_infty: float = 10000.0,
    spanwise_spacing: str = "uniform",
    spanwise_spacing_ratio: float = 1.0,
    spanwise_spacing_region: str = "both",
) -> int:
    """
    Generate panels for a single wing segment.

    Args:
        segment: Wing segment geometry
        lattice: VLMLattice to populate
        start_panel_idx: Starting panel index
        wing_id: Wing identifier
        segment_id: Segment identifier
        is_mirrored: Whether to mirror the segment
        symmetry_plane: Symmetry plane (1=XY, 2=XZ, 3=YZ)
        trailing_edge_infty: Distance for trailing legs
        spanwise_spacing: Panel distribution method ('uniform' or 'geometric')
        spanwise_spacing_ratio: Concentration ratio for geometric spacing
        spanwise_spacing_direction: Clustering direction ('+y', '-y', 'y')

    Returns:
        Next available panel index
    """
    nc = segment.n_chordwise_panels
    ns = segment.n_spanwise_panels

    # Get segment vertex_position
    a = segment.vertex_position["a"]
    b = segment.vertex_position["b"]
    c = segment.vertex_position["c"]
    d = segment.vertex_position["d"]

    # Apply symmetry transformation if needed
    if is_mirrored:
        a, b, c, d = _mirror_vertices(a, b, c, d, symmetry_plane)

    # Compute bilinear interpolation coefficients
    alpha = _compute_bilinear_coefficients(a, b, c, d)

    # Panel spacing
    dc = 1.0 / nc  # Chordwise always uniform
    s_edges = _compute_spanwise_edges(
        ns, spanwise_spacing, spanwise_spacing_ratio, spanwise_spacing_region
    )

    panel_idx = start_panel_idx

    # Generate panels
    for j in range(ns):  # Spanwise
        for i in range(nc):  # Chordwise
            # Panel bounds in (s, c) space
            s_min, s_max = s_edges[j], s_edges[j + 1]
            c_min, c_max = i * dc, (i + 1) * dc

            # Compute panel geometry
            P, Q, R, S = _compute_panel_corners(alpha, s_min, s_max, c_min, c_max)
            V2, V3, collocation_point, normal, area = _compute_panel_geometry(
                P, Q, R, S, is_mirrored, symmetry_plane
            )
            V1, V2, V3, V4, bound_midpoint, trail_dir = _compute_trailing_geometry(
                V2, V3, trailing_edge_infty
            )

            # Store in lattice
            _store_panel_data(
                lattice,
                panel_idx,
                P,
                Q,
                R,
                S,
                V1,
                V2,
                V3,
                V4,
                collocation_point,
                normal,
                area,
                bound_midpoint,
                trail_dir,
                wing_id,
                segment_id,
                is_mirrored,
            )

            panel_idx += 1

    return panel_idx


def _generate_segment_mesh_with_edges(
    segment: WingSegment,
    lattice,
    start_panel_idx: int,
    wing_id: int,
    segment_id: int,
    s_edges: np.ndarray,
    is_mirrored: bool = False,
    symmetry_plane: int = 0,
    trailing_edge_infty: float = 10000.0,
) -> int:
    """
    Generate panels for a single wing segment using pre-computed spanwise edges.

    This version takes explicit spanwise edge locations, enabling refinement
    to be applied across multiple segments of a wing.

    Args:
        segment: Wing segment geometry
        lattice: VLMLattice to populate
        start_panel_idx: Starting panel index
        wing_id: Wing identifier
        segment_id: Segment identifier
        s_edges: Pre-computed spanwise edge locations in [0, 1] range
        is_mirrored: Whether to mirror the segment
        symmetry_plane: Symmetry plane (1=XY, 2=XZ, 3=YZ)
        trailing_edge_infty: Distance for trailing legs

    Returns:
        Next available panel index
    """
    nc = segment.n_chordwise_panels
    ns = segment.n_spanwise_panels

    # Get segment vertex_position
    a = segment.vertex_position["a"]
    b = segment.vertex_position["b"]
    c = segment.vertex_position["c"]
    d = segment.vertex_position["d"]

    # Apply symmetry transformation if needed
    if is_mirrored:
        a, b, c, d = _mirror_vertices(a, b, c, d, symmetry_plane)

    # Compute bilinear interpolation coefficients
    alpha = _compute_bilinear_coefficients(a, b, c, d)

    # Panel spacing
    dc = 1.0 / nc
    panel_idx = start_panel_idx

    # Generate panels
    for j in range(ns):
        for i in range(nc):
            # Panel bounds in (s, c) space
            s_min, s_max = s_edges[j], s_edges[j + 1]
            c_min, c_max = i * dc, (i + 1) * dc

            # Compute panel panel_corner_position
            P, Q, R, S = _compute_panel_corners(alpha, s_min, s_max, c_min, c_max)

            # Collocation at 75% chord, center span
            s_coll = 0.5 * (s_min + s_max)
            c_coll = 0.5 * (c_min + c_max) + 0.25 * dc
            collocation_point = _bilinear_interp(alpha, s_coll, c_coll)

            # Normal and area
            diag1, diag2 = R - P, Q - S
            normal_raw = np.cross(diag1, diag2)
            area = 0.5 * np.linalg.norm(normal_raw)
            normal = normal_raw / (2 * area) if area > 1e-12 else np.array([0.0, 0.0, 1.0])

            # Horseshoe vortex points at 25% chord
            c_vort = 0.5 * (c_min + c_max) - 0.25 * dc
            V1 = V2 = _bilinear_interp(alpha, s_min, c_vort)
            V3 = V4 = _bilinear_interp(alpha, s_max, c_vort)

            # Trailing direction
            trail_dir = 0.5 * (R + S) - 0.5 * (P + Q)
            trail_mag = np.linalg.norm(trail_dir)
            trail_dir = trail_dir / trail_mag if trail_mag > 1e-12 else np.array([1.0, 0.0, 0.0])

            # Bound midpoint
            bound_midpoint = 0.5 * (V2 + V3)

            # Store in lattice
            _store_panel_data(
                lattice,
                panel_idx,
                P,
                Q,
                R,
                S,
                V1,
                V2,
                V3,
                V4,
                collocation_point,
                normal,
                area,
                bound_midpoint,
                trail_dir,
                wing_id,
                segment_id,
                is_mirrored,
            )

            panel_idx += 1

    return panel_idx


def _bilinear_interp(alpha: np.ndarray, s: float, c: float) -> np.ndarray:
    """
    Bilinear interpolation within unit square.

    Position(s, c) = alpha[0] + alpha[1]*s + alpha[2]*c + alpha[3]*s*c

    Args:
        alpha: Interpolation coefficients (3 x 4) for [x, y, z]
        s: Spanwise coordinate [0, 1]
        c: Chordwise coordinate [0, 1]

    Returns:
        3D position
    """
    pos = np.zeros(3)
    for dim in range(3):
        pos[dim] = alpha[dim, 0] + alpha[dim, 1] * s + alpha[dim, 2] * c + alpha[dim, 3] * s * c
    return pos


def _mirror_vertices(a, b, c, d, symmetry_plane: int) -> tuple[np.ndarray, ...]:
    """
    Mirror vertex_position across symmetry plane.

    NOTE: This function NO LONGER swaps vertex order. The normal negation
    for mirrored panels is now handled in _generate_segment_to_numpy.
    This preserves consistent spanwise ordering (root→tip) for both
    original and mirrored panels, which is critical for wake shedding.

    Args:
        a, b, c, d: Original vertex_position
        symmetry_plane: 1 (XY/Z-mirror), 2 (XZ/Y-mirror), 3 (YZ/X-mirror)

    Returns:
        Mirrored vertex_position in SAME ORDER (a_m, b_m, c_m, d_m)
    """
    a_m = a.copy()
    b_m = b.copy()
    c_m = c.copy()
    d_m = d.copy()

    if symmetry_plane == 1:  # Mirror in Z (XY plane)
        a_m[2] = -a[2]
        b_m[2] = -b[2]
        c_m[2] = -c[2]
        d_m[2] = -d[2]
        # NO swap - return in original order, normal will be negated elsewhere
        return a_m, b_m, c_m, d_m

    elif symmetry_plane == 2:  # Mirror in Y (XZ plane)
        a_m[1] = -a[1]
        b_m[1] = -b[1]
        c_m[1] = -c[1]
        d_m[1] = -d[1]
        return a_m, b_m, c_m, d_m

    elif symmetry_plane == 3:  # Mirror in X (YZ plane)
        a_m[0] = -a[0]
        b_m[0] = -b[0]
        c_m[0] = -c[0]
        d_m[0] = -d[0]
        return a_m, b_m, c_m, d_m

    return a, b, c, d


# =========================================================
# TAICHI KERNELS FOR TRAILING DIRECTION UPDATES
# =========================================================


@ti.kernel
def _update_trailing_uniform_kernel(
    trailing_direction: ti.template(),
    vortex_point_position: ti.template(),
    normal: ti.template(),
    n_panels: ti.i32,
    trail_dir_x: float,
    trail_dir_y: float,
    trail_dir_z: float,
    trailing_infty: float,
):
    """Taichi kernel for uniform trailing direction update."""
    trail_dir = ti.Vector([trail_dir_x, trail_dir_y, trail_dir_z])

    for i in range(n_panels):
        # Set trailing directions
        trailing_direction[i, 0] = trail_dir
        trailing_direction[i, 1] = trail_dir

        # Recompute far trailing points (downstream). The far wake lies in the
        # local wing (tangent) plane: we keep the streamwise wake direction but
        # remove its component normal to the panel. Slanting the far legs along
        # the freestream (which carries an angle-of-attack vertical component)
        # lifts the wake out of the wing plane and suppresses the tip downwash,
        # erasing the spanwise taper expected from finite-wing theory. The
        # trailing_direction field (used for VPM particle shedding) is left along
        # the relative velocity on purpose.
        n = normal[i]
        d_plane = trail_dir - trail_dir.dot(n) * n
        d_mag = d_plane.norm()
        d_far = trail_dir
        if d_mag > VLM_SMALL_VELOCITY:
            d_far = d_plane / d_mag
        V2 = vortex_point_position[i, 1]
        V3 = vortex_point_position[i, 2]

        vortex_point_position[i, 0] = V2 + d_far * trailing_infty
        vortex_point_position[i, 3] = V3 + d_far * trailing_infty


@ti.kernel
def _update_trailing_local_kernel(
    trailing_direction: ti.template(),
    vortex_point_position: ti.template(),
    normal: ti.template(),
    V_local: ti.template(),
    n_panels: ti.i32,
    trailing_infty: float,
    epsilon: float,
):
    """Taichi kernel for local (per-panel) trailing direction update."""
    for i in range(n_panels):
        # Get local velocity for this panel
        V = V_local[i]
        V_mag = V.norm()

        # Compute trailing direction
        trail_dir = ti.Vector([1.0, 0.0, 0.0])  # Default
        if V_mag > epsilon:
            trail_dir = V / V_mag

        # Set trailing directions
        trailing_direction[i, ti.i32(0)] = trail_dir
        trailing_direction[i, ti.i32(1)] = trail_dir

        # Recompute far trailing points (downstream). The far wake lies in the
        # local wing (tangent) plane; see _update_trailing_uniform_kernel.
        n = normal[i]
        d_plane = trail_dir - trail_dir.dot(n) * n
        d_mag = d_plane.norm()
        d_far = trail_dir
        if d_mag > VLM_SMALL_VELOCITY:
            d_far = d_plane / d_mag
        V2 = vortex_point_position[i, ti.i32(1)]
        V3 = vortex_point_position[i, ti.i32(2)]

        vortex_point_position[i, ti.i32(0)] = V2 + d_far * trailing_infty
        vortex_point_position[i, ti.i32(3)] = V3 + d_far * trailing_infty


def update_trailing_edge_directions(lattice, freestream_velocity: np.ndarray):
    """
    Update trailing edge direction vectors based on freestream.

    Args:
        lattice: VLMLattice object
        freestream_velocity: Freestream velocity vector (3,)
    """
    freestream_speed = np.linalg.norm(freestream_velocity)
    if freestream_speed < VLM_SMALL_VELOCITY:
        return

    trail_dir = freestream_velocity / freestream_speed

    # Use Taichi kernel for performance
    _update_trailing_uniform_kernel(
        lattice.trailing_direction,
        lattice.vortex_point_position,
        lattice.normal,
        lattice.n_panels,
        float(trail_dir[0]),
        float(trail_dir[1]),
        float(trail_dir[2]),
        10000.0,
    )


def _prepare_trailing_temp(lattice, freestream_velocity: np.ndarray):
    """Allocate/reuse Taichi temp field and upload freestream_velocity for trailing direction update."""
    if not hasattr(lattice, "_trailing_temp") or lattice._trailing_temp is None:
        lattice._trailing_temp = ti.Vector.field(
            3, dtype=lattice.external_velocity.dtype, shape=lattice.external_velocity.shape[0]
        )
    else:
        existing = lattice._trailing_temp
        if (
            existing.shape[0] != lattice.external_velocity.shape[0]
            or existing.dtype != lattice.external_velocity.dtype
        ):
            lattice._trailing_temp = ti.Vector.field(
                3,
                dtype=lattice.external_velocity.dtype,
                shape=lattice.external_velocity.shape[0],
            )
    temp_field = lattice._trailing_temp
    dtype_np = np.float32 if lattice.external_velocity.dtype == ti.f32 else np.float64
    if (
        not hasattr(lattice, "_trailing_temp_np")
        or lattice._trailing_temp_np is None
        or lattice._trailing_temp_np.shape[0] < temp_field.shape[0]
    ):
        lattice._trailing_temp_np = np.zeros((temp_field.shape[0], 3), dtype=dtype_np)
    V_full = lattice._trailing_temp_np
    V_full[: lattice.n_panels] = freestream_velocity
    temp_field.from_numpy(V_full)
    return temp_field


def update_trailing_directions_local(
    lattice, freestream_velocity: np.ndarray, trailing_infty: float = 10000.0
):
    """
    Update trailing edge direction vectors based on local velocity field.

    This is a performance-critical function that uses a Taichi kernel
    to avoid O(N) GPU-CPU data transfers.

    Args:
        lattice: VLMLattice object
        freestream_velocity: Velocity field (N, 3) where N = n_panels,
               representing local transport velocity at each panel.
               Typically external_velocity - V_kinematic.
        trailing_infty: Distance to far trailing points
    """
    # Check input shape
    is_field = freestream_velocity.ndim == 2 and freestream_velocity.shape[0] >= lattice.n_panels
    is_vector = freestream_velocity.ndim == 1 and freestream_velocity.shape[0] == 3

    if is_vector:
        # Uniform velocity - use simpler kernel
        freestream_speed = np.linalg.norm(freestream_velocity)
        if freestream_speed < VLM_SMALL_VELOCITY:
            trail_dir = np.array([1.0, 0.0, 0.0])
        else:
            trail_dir = freestream_velocity / freestream_speed

        _update_trailing_uniform_kernel(
            lattice.trailing_direction,
            lattice.vortex_point_position,
            lattice.normal,
            lattice.n_panels,
            float(trail_dir[0]),
            float(trail_dir[1]),
            float(trail_dir[2]),
            trailing_infty,
        )
    elif is_field:
        temp_field = _prepare_trailing_temp(lattice, freestream_velocity)
        _update_trailing_local_kernel(
            lattice.trailing_direction,
            lattice.vortex_point_position,
            lattice.normal,
            temp_field,
            lattice.n_panels,
            trailing_infty,
            VLM_SMALL_VELOCITY,
        )
    else:
        raise ValueError(
            f"freestream_velocity must be (3,) vector or (N, 3) field, got shape {freestream_velocity.shape}"
        )


def _geometric_spacing_single_end(n: int, ratio: float, refine_start: bool) -> np.ndarray:
    """
    Compute geometric spacing refined at one end.

    Args:
        n: Number of panels
        ratio: Concentration ratio (d_max / d_min)
        refine_start: If True, refine at start (s=0); if False, refine at end (s=1)

    Returns:
        Array of n+1 edge locations
    """
    if refine_start:
        r = ratio ** (1.0 / (n - 1))
        d = np.zeros(n)
        d[0] = 1.0  # Smallest at start
        for i in range(1, n):
            d[i] = d[i - 1] * r  # Growing
    else:
        r = (1.0 / ratio) ** (1.0 / (n - 1))
        d = np.zeros(n)
        d[0] = 1.0  # Largest at start
        for i in range(1, n):
            d[i] = d[i - 1] * r  # Decreasing

    d = d * (1.0 / np.sum(d))
    edges = np.zeros(n + 1)
    for i in range(n):
        edges[i + 1] = edges[i] + d[i]
    return edges


def _geometric_spacing_both_ends(n: int, ratio: float) -> np.ndarray:
    """
    Compute geometric spacing refined at both ends.

    Args:
        n: Number of panels
        ratio: Concentration ratio (d_max / d_min)

    Returns:
        Array of n+1 edge locations
    """
    n1 = n // 2
    n2 = n - n1

    # First half: Smallest at 0, largest at center
    if n1 > 1:
        r = ratio ** (1.0 / (n1 - 1))
        d = np.zeros(n1)
        d[0] = 1.0
        for i in range(1, n1):
            d[i] = d[i - 1] * r
        d = d * (0.5 / np.sum(d))

        edges1 = np.zeros(n1 + 1)
        for i in range(n1):
            edges1[i + 1] = edges1[i] + d[i]
    else:
        edges1 = np.linspace(0, 0.5, n1 + 1)

    # Second half: Largest at center, smallest at end
    if n2 > 1:
        r = ratio ** (1.0 / (n2 - 1))
        d = np.zeros(n2)
        d[0] = 1.0
        for i in range(1, n2):
            d[i] = d[i - 1] * r
        d = d[::-1]
        d = d * (0.5 / np.sum(d))

        edges2 = np.zeros(n2 + 1)
        edges2[0] = 0.5
        for i in range(n2):
            edges2[i + 1] = edges2[i] + d[i]
    else:
        edges2 = np.linspace(0.5, 1.0, n2 + 1)

    return np.concatenate([edges1, edges2[1:]])


def _compute_spanwise_edges(
    n: int, spacing_type: str = "uniform", ratio: float = 1.0, region: str = "both"
) -> np.ndarray:
    """
    Compute spanwise panel edge locations in [0, 1].

    Args:
        n: Number of panels
        spacing_type: 'uniform', 'cosine', 'geometric'
        ratio: Concentration ratio (d_max / d_min) for geometric spacing
        region: Where to apply refinement for geometric spacing:
                'start': Refine at s=0 (smallest panels at start)
                'end': Refine at s=1 (smallest panels at end)
                'both': Refine at both ends (smallest at 0 and 1, largest at center)

    Returns:
        Array of n+1 edge locations starting at 0 and ending at 1
    """
    if spacing_type == "cosine":
        return np.array([0.5 * (1 - np.cos(np.pi * j / n)) for j in range(n + 1)])

    elif spacing_type == "geometric":
        if n < 2 or ratio < 1.0 + 1e-9:
            return np.linspace(0, 1, n + 1)

        if region == "both":
            return _geometric_spacing_both_ends(n, ratio)
        elif region == "end":
            return _geometric_spacing_single_end(n, ratio, refine_start=False)
        elif region == "start":
            return _geometric_spacing_single_end(n, ratio, refine_start=True)
        else:
            return np.linspace(0, 1, n + 1)

    else:  # uniform
        return np.linspace(0, 1, n + 1)
