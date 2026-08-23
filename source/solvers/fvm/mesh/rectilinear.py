"""Rectilinear hex meshes with an optional carved (body-fitted) box body.

Single home for the tensor-grid mesh generation shared by the FVM solver,
the FVM–VPM coupler, and the cube_flow tutorial:

* graded 1-D node distributions — :func:`stretched`, :func:`wall_refined_axis`;
* the 3-D generator :func:`box_mesh_3d` with six named outer patches; and
* :func:`coupling_box_mesh`, the coupler-facing variant whose outer sides form
  one merged coupling patch (optionally leaving the spanwise pair ``empty``).

Both mesh functions carve the optional ``hole_box`` out of the grid: the cells
inside it are removed and the exposed faces become a body-fitted ``wall``
patch, so no fluid cells exist inside the body and the wall is an explicit
boundary of mesh faces (no immersed-boundary or cut-cell treatment).
"""

from __future__ import annotations

import numpy as np

OUTER_PATCH_NAMES = ("inlet", "outlet", "ymin", "ymax", "zmin", "zmax")


def stretched(start: float, end: float, h0: float, ratio: float, h_max: float | None = None):
    """Node positions from ``start`` to ``end`` with sizes growing
    geometrically from ``h0*ratio`` (capped at ``h_max``); the last sliver is
    merged into the final cell.  ``end`` may be below ``start`` (leftward run).
    Returns the nodes EXCLUDING ``start``, in the direction of travel."""
    direction = 1.0 if end >= start else -1.0
    span = abs(end - start)
    sizes = []
    size = h0
    pos = 0.0
    while span - pos > 1e-12:
        size *= ratio
        if h_max is not None:
            size = min(size, h_max)
        remaining = span - pos
        if remaining < 1.35 * size:  # absorb the sliver into one final cell
            size = remaining
        sizes.append(size)
        pos += size
    return start + direction * np.cumsum(np.asarray(sizes, dtype=np.float64))


def _grade_segment(a: float, b: float, h_wall: float, h_far: float, ratio: float, wall: str):
    """Interior node coordinates of one graded segment (a, b), EXCLUDING both
    endpoints.  The cell adjacent to the ``wall`` end ('lo'→a, 'hi'→b) is
    ``h_wall``; sizes grow geometrically by ``ratio`` toward the other end,
    capped at ``h_far``.  The final sliver is merged so the endpoints are hit
    exactly (same scheme as ``stretched``)."""
    length = float(b) - float(a)
    sizes = []
    size = float(h_wall)
    pos = 0.0
    while length - pos > 1e-12:
        remaining = length - pos
        if remaining < 1.5 * size:  # absorb the sliver into the last cell
            sizes.append(remaining)
            break
        sizes.append(size)
        pos += size
        size = min(size * ratio, h_far)
    sizes = np.asarray(sizes, dtype=np.float64)  # small→large, from the wall end
    # 'lo': fine cell adjacent to a; 'hi': fine cell adjacent to b (sizes run large→small from a)
    nodes = a + np.cumsum(sizes if wall == "lo" else sizes[::-1])
    return nodes[:-1]  # drop the far endpoint (interior nodes only)


def wall_refined_axis(
    lo: float,
    hi: float,
    wall_lo: float,
    wall_hi: float,
    h_wall: float,
    h_far: float,
    ratio: float = 1.25,
) -> np.ndarray:
    """1D node array on [lo, hi] with cells of size ~``h_wall`` adjacent to the
    two body faces ``wall_lo``/``wall_hi`` (e.g. ±0.5), coarsening geometrically
    to ``h_far`` toward ``lo``/``hi`` and through the carved body interior.

    Breakpoints land EXACTLY on lo, wall_lo, wall_hi, hi so the cube carve and
    the box faces sit on mesh planes.  Shared by the coupled box and the
    reference core so their common region is identical cell-for-cell.
    """
    left = _grade_segment(lo, wall_lo, h_wall, h_far, ratio, wall="hi")
    # Body interior: grade OUTWARD from each wall to the body midpoint (two
    # clean half-segments).  Never union two overlapping node sets — the
    # interleaved nodes create near-zero-width slabs that, on a tensor grid,
    # cut through the entire domain (measured min cell 4e-4 at h_wall=0.025,
    # collapsing the CFL time step and blowing up both solvers).
    mid = 0.5 * (wall_lo + wall_hi)
    body = np.concatenate(
        [
            _grade_segment(wall_lo, mid, h_wall, h_far, ratio, wall="lo"),
            [mid],
            _grade_segment(mid, wall_hi, h_wall, h_far, ratio, wall="hi"),
        ]
    )
    right = _grade_segment(wall_hi, hi, h_wall, h_far, ratio, wall="lo")
    nodes = np.concatenate([[lo], left, [wall_lo], body, [wall_hi], right, [hi]])
    nodes = np.unique(np.round(nodes, 12))
    d = np.diff(nodes)
    if d.min() < 0.4 * h_wall:
        raise ValueError(
            f"wall_refined_axis produced a degenerate cell ({d.min():.3e} < "
            f"0.4*h_wall={0.4 * h_wall:.3e}) — refusing to build a broken mesh."
        )
    return nodes


def _cells_along(lo: float, hi: float, spacing: float, axis: str) -> int:
    """Number of cells along one axis; the box must be an integer number of
    cells wide (the hand-off lattice and the blending zone assume a uniform,
    box-conforming grid)."""
    length = float(hi) - float(lo)
    if length <= 0.0:
        raise ValueError(f"fvm_box has non-positive extent along {axis}: {length}")
    n = length / float(spacing)
    n_int = int(round(n))
    if n_int < 1 or abs(n - n_int) > 1e-6 * max(n, 1.0):
        raise ValueError(
            f"fvm_box extent along {axis} ({length:g}) is not an integer "
            f"multiple of grid_spacing ({spacing:g}): {n:g} cells."
        )
    return n_int


def _plane_index(coords: np.ndarray, value: float, what: str) -> int:
    """Index of the grid plane a body face lies on; it must lie ON a plane."""
    idx = int(np.argmin(np.abs(coords - value)))
    if abs(coords[idx] - value) > 1e-6 * max(abs(value), 1.0):
        raise ValueError(f"{what} at {value:g} is not on a mesh plane (closest {coords[idx]:g})")
    return idx


def box_mesh_3d(
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    hole_box: tuple[float, float, float, float, float, float] | None = None,
    wall_patch_name: str = "cube",
    merge_outer_patch: str | None = None,
    empty_spanwise: bool = False,
    separate_outer: tuple[str, ...] = (),
) -> dict:
    """Face-based ``mesh_data`` dict for a rectilinear grid with a box hole.

    Patch order: inlet (xmin), outlet (xmax), ymin, ymax, zmin, zmax, then the
    ``wall_patch_name`` hole faces (type ``wall``). With ``merge_outer_patch``
    the outer sides become one boundary patch of that name (the coupler's
    coupling-patch layout). If ``empty_spanwise`` is true, zmin/zmax remain
    separate ``empty`` patches and only the four in-plane sides are merged.

    ``separate_outer`` names outer faces kept out of the merge, each with its
    own patch. Merged families come first, so every patch stays contiguous.
    """
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    zs = np.asarray(zs, dtype=np.float64)
    nx, ny, nz = len(xs) - 1, len(ys) - 1, len(zs) - 1
    npx, npy = nx + 1, ny + 1

    Z, Y, X = np.meshgrid(zs, ys, xs, indexing="ij")
    points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()]).astype(np.float64)

    def pid(i, j, k):
        return i + npx * (j + npy * k)

    def cid(i, j, k):
        return i + nx * (j + ny * k)

    def oriented_block(quads, desired_dir):
        """Flip the whole (M, 4) family if its uniform winding opposes desired_dir."""
        quads = np.ascontiguousarray(quads, dtype=np.int32)
        if quads.shape[0]:
            p = points[quads[0]]
            if np.dot(np.cross(p[1] - p[0], p[2] - p[0]), desired_dir) < 0:
                quads = np.ascontiguousarray(quads[:, ::-1])
        return quads

    def x_family(i_arr):
        i, k, j = np.meshgrid(i_arr, np.arange(nz), np.arange(ny), indexing="ij")
        i, k, j = i.ravel(), k.ravel(), j.ravel()
        quads = np.stack(
            [pid(i, j, k), pid(i, j + 1, k), pid(i, j + 1, k + 1), pid(i, j, k + 1)], axis=1
        )
        return quads, i, j, k

    def y_family(j_arr):
        j, k, i = np.meshgrid(j_arr, np.arange(nz), np.arange(nx), indexing="ij")
        i, k, j = i.ravel(), k.ravel(), j.ravel()
        quads = np.stack(
            [pid(i, j, k), pid(i + 1, j, k), pid(i + 1, j, k + 1), pid(i, j, k + 1)], axis=1
        )
        return quads, i, j, k

    def z_family(k_arr):
        k, j, i = np.meshgrid(k_arr, np.arange(ny), np.arange(nx), indexing="ij")
        i, k, j = i.ravel(), k.ravel(), j.ravel()
        quads = np.stack(
            [pid(i, j, k), pid(i + 1, j, k), pid(i + 1, j + 1, k), pid(i, j + 1, k)], axis=1
        )
        return quads, i, j, k

    ex, ey, ez = np.eye(3)

    # Interior families (owner = lower cell, neighbour = upper cell).
    qx, ix, jx, kx = x_family(np.arange(1, nx))
    qy, iy, jy, ky = y_family(np.arange(1, ny))
    qz, iz, jz, kz = z_family(np.arange(1, nz))
    interior_quads = np.vstack(
        [oriented_block(qx, ex), oriented_block(qy, ey), oriented_block(qz, ez)]
    )
    interior_owners = np.concatenate(
        [cid(ix - 1, jx, kx), cid(iy, jy - 1, ky), cid(iz, jz, kz - 1)]
    )
    interior_neighbours = np.concatenate([cid(ix, jx, kx), cid(iy, jy, ky), cid(iz, jz, kz)])

    # Outer boundary families in the fixed order xmin, xmax, ymin, ymax, zmin,
    # zmax (owner = adjacent cell).
    q_x0, _, jb, kb = x_family(np.array([0]))
    o_x0 = cid(np.zeros_like(jb), jb, kb)
    q_x1, _, jb, kb = x_family(np.array([nx]))
    o_x1 = cid(np.full_like(jb, nx - 1), jb, kb)
    q_y0, ib, _, kb = y_family(np.array([0]))
    o_y0 = cid(ib, np.zeros_like(ib), kb)
    q_y1, ib, _, kb = y_family(np.array([ny]))
    o_y1 = cid(ib, np.full_like(ib, ny - 1), kb)
    q_z0, ib, jb, _ = z_family(np.array([0]))
    o_z0 = cid(ib, jb, np.zeros_like(ib))
    q_z1, ib, jb, _ = z_family(np.array([nz]))
    o_z1 = cid(ib, jb, np.full_like(ib, nz - 1))

    outer = [
        (OUTER_PATCH_NAMES[0], oriented_block(q_x0, -ex), o_x0),
        (OUTER_PATCH_NAMES[1], oriented_block(q_x1, ex), o_x1),
        (OUTER_PATCH_NAMES[2], oriented_block(q_y0, -ey), o_y0),
        (OUTER_PATCH_NAMES[3], oriented_block(q_y1, ey), o_y1),
        (OUTER_PATCH_NAMES[4], oriented_block(q_z0, -ez), o_z0),
        (OUTER_PATCH_NAMES[5], oriented_block(q_z1, ez), o_z1),
    ]

    # ── Optional body: carve the hole and expose its faces as a wall patch ──
    wall_quads = np.empty((0, 4), dtype=np.int32)
    wall_owners = np.empty(0, dtype=np.int64)
    keep = np.ones(nx * ny * nz, dtype=bool)
    if hole_box is not None:
        i0 = _plane_index(xs, hole_box[0], f"{wall_patch_name} x-min face")
        i1 = _plane_index(xs, hole_box[1], f"{wall_patch_name} x-max face")
        j0 = _plane_index(ys, hole_box[2], f"{wall_patch_name} y-min face")
        j1 = _plane_index(ys, hole_box[3], f"{wall_patch_name} y-max face")
        k0 = _plane_index(zs, hole_box[4], f"{wall_patch_name} z-min face")
        k1 = _plane_index(zs, hole_box[5], f"{wall_patch_name} z-max face")
        if i0 >= i1 or j0 >= j1 or k0 >= k1:
            raise ValueError(f"hole_box {hole_box} has empty extent on the grid.")

        ci, cj, ck = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
        in_hole = (ci >= i0) & (ci < i1) & (cj >= j0) & (cj < j1) & (ck >= k0) & (ck < k1)
        keep[cid(ci.ravel(), cj.ravel(), ck.ravel())] = ~in_hole.ravel()

        own_kept = keep[interior_owners]
        nb_kept = keep[interior_neighbours]
        both = own_kept & nb_kept
        # Wall faces sit between a fluid cell and a removed cell.  Interior
        # winding points from owner (lower cell) to neighbour (upper cell), so
        # it already points out of the fluid when the owner survives; when the
        # neighbour survives it becomes the owner and the ring is reversed.
        low_side = own_kept & ~nb_kept
        up_side = ~own_kept & nb_kept
        wall_quads = np.vstack([interior_quads[low_side], interior_quads[up_side][:, ::-1]]).astype(
            np.int32
        )
        wall_owners = np.concatenate([interior_owners[low_side], interior_neighbours[up_side]])
        interior_quads = interior_quads[both]
        interior_owners = interior_owners[both]
        interior_neighbours = interior_neighbours[both]
        # Outer faces owned by removed cells vanish with them (only possible
        # when the hole touches an outer boundary).
        outer = [(name, quads[keep[own]], own[keep[own]]) for name, quads, own in outer]

    outer_names = [name for name, _, _ in outer]
    unknown = [name for name in separate_outer if name not in outer_names]
    if unknown:
        raise ValueError(
            f"separate_outer names {unknown} are not outer faces; expected any of {outer_names}"
        )

    def _is_standalone(name: str) -> bool:
        return name in separate_outer or (empty_spanwise and name in {"zmin", "zmax"})

    if merge_outer_patch is not None:
        # Merged families first, so every patch is a contiguous range.
        outer = [entry for entry in outer if not _is_standalone(entry[0])] + [
            entry for entry in outer if _is_standalone(entry[0])
        ]

    all_quads = [interior_quads] + [quads for _, quads, _ in outer] + [wall_quads]
    all_owners = [interior_owners] + [own for _, _, own in outer] + [wall_owners]
    n_interior = interior_quads.shape[0]

    boundary = []
    start = n_interior
    if merge_outer_patch is not None:
        merged = [entry for entry in outer if not _is_standalone(entry[0])]
        standalone = [entry for entry in outer if _is_standalone(entry[0])]
        n_outer = sum(quads.shape[0] for _, quads, _ in merged)
        if n_outer:
            boundary.append(
                {
                    "name": merge_outer_patch,
                    "start_face": start,
                    "n_faces": n_outer,
                    "type": "patch",
                }
            )
            start += n_outer
        for name, quads, _ in standalone:
            boundary.append(
                {
                    "name": name,
                    "start_face": start,
                    "n_faces": quads.shape[0],
                    "type": "empty" if empty_spanwise and name in {"zmin", "zmax"} else "patch",
                }
            )
            start += quads.shape[0]
    else:
        for name, quads, _ in outer:
            boundary.append(
                {
                    "name": name,
                    "start_face": start,
                    "n_faces": quads.shape[0],
                    "type": "empty" if empty_spanwise and name in {"zmin", "zmax"} else "patch",
                }
            )
            start += quads.shape[0]
    if hole_box is not None:
        boundary.append(
            {
                "name": wall_patch_name,
                "start_face": start,
                "n_faces": wall_quads.shape[0],
                "type": "wall",
            }
        )

    all_quads = np.vstack(all_quads)
    owners = np.concatenate(all_owners)

    # Compact cell numbering after the carve (identity when there is no hole).
    new_id = np.cumsum(keep) - 1
    owners = new_id[owners]
    neighbours = new_id[interior_neighbours]

    # A rectilinear mesh has fixed-width quad faces and hex cells.  Keeping
    # that information as contiguous arrays is materially cheaper than a
    # Python list containing one tiny ndarray per face (the reference cube
    # case has several million faces).  ``faces`` remains indexable exactly
    # as before, so generic FVM operators and mesh readers stay compatible.
    cell_ids = np.flatnonzero(keep)
    cell_i = cell_ids % nx
    cell_j = (cell_ids // nx) % ny
    cell_k = cell_ids // (nx * ny)
    cell_vertex_indices = np.column_stack(
        (
            pid(cell_i, cell_j, cell_k),
            pid(cell_i + 1, cell_j, cell_k),
            pid(cell_i + 1, cell_j + 1, cell_k),
            pid(cell_i, cell_j + 1, cell_k),
            pid(cell_i, cell_j, cell_k + 1),
            pid(cell_i + 1, cell_j, cell_k + 1),
            pid(cell_i + 1, cell_j + 1, cell_k + 1),
            pid(cell_i, cell_j + 1, cell_k + 1),
        )
    ).astype(np.int32, copy=False)

    return {
        "vertex_position": points,
        "faces": np.ascontiguousarray(all_quads, dtype=np.int32),
        "owners": np.asarray(owners, dtype=np.int32),
        "neighbours": np.asarray(neighbours, dtype=np.int32),
        "boundary": boundary,
        "n_cells": int(keep.sum()),
        "n_faces": all_quads.shape[0],
        "n_interior_faces": n_interior,
        "n_points": points.shape[0],
        # Gmsh element type 5 = 8-node hexahedron.  The explicit vertices let
        # VTK write compact native hex cells instead of reconstructing every
        # cell as a general polyhedron.
        "cell_vertex_indices": np.ascontiguousarray(cell_vertex_indices),
        "cell_type_code": np.full(int(keep.sum()), 5, dtype=np.int32),
    }


def coupling_box_mesh(
    fvm_box: tuple[float, float, float, float, float, float],
    spacing: float,
    patch_name: str = "numericalBoundary",
    hole_box: tuple[float, float, float, float, float, float] | None = None,
    wall_patch_name: str = "cube",
    nodes: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    empty_spanwise: bool = False,
    separate_outer: tuple[str, ...] = (),
) -> dict:
    """Return a hex mesh whose six sides form one coupling patch.

    ``nodes=(xs, ys, zs)`` supplies explicit per-axis grid lines (e.g. graded,
    wall-refined via :func:`wall_refined_axis`); otherwise a uniform grid at
    ``spacing`` is built. With ``empty_spanwise``, the zmin/zmax faces remain
    separate ``empty`` patches while the four in-plane faces form the coupling
    patch. With ``hole_box``, the cells inside it are removed
    (body-fitted): the exposed faces become a
    second boundary patch ``wall_patch_name`` of type ``wall``.  The hole faces
    must lie exactly on mesh planes.

    A fully meshed reference must pass ``separate_outer=("outlet",)`` plus a
    :meth:`BoundaryConfig.outlet`, else its outlet is clamped to the freestream.
    """
    x0, x1, y0, y1, z0, z1 = (float(v) for v in fvm_box)
    if nodes is not None:
        xs, ys, zs = (np.asarray(a, dtype=np.float64) for a in nodes)
        if not (
            abs(xs[0] - x0) < 1e-9
            and abs(xs[-1] - x1) < 1e-9
            and abs(ys[0] - y0) < 1e-9
            and abs(ys[-1] - y1) < 1e-9
            and abs(zs[0] - z0) < 1e-9
            and abs(zs[-1] - z1) < 1e-9
        ):
            raise ValueError("nodes endpoints must equal fvm_box bounds")
    else:
        nx = _cells_along(x0, x1, spacing, "x")
        ny = _cells_along(y0, y1, spacing, "y")
        nz = _cells_along(z0, z1, spacing, "z")
        dx, dy, dz = (x1 - x0) / nx, (y1 - y0) / ny, (z1 - z0) / nz
        xs = x0 + dx * np.arange(nx + 1)
        ys = y0 + dy * np.arange(ny + 1)
        zs = z0 + dz * np.arange(nz + 1)
    return box_mesh_3d(
        xs,
        ys,
        zs,
        hole_box=hole_box,
        wall_patch_name=wall_patch_name,
        merge_outer_patch=patch_name,
        empty_spanwise=empty_spanwise,
        separate_outer=separate_outer,
    )


def periodic_square_mesh(n: int, length: float = 2.0 * np.pi) -> dict:
    """Return an ``n`` by ``n`` single-cell-thick periodic square mesh.

    The four in-plane patches are paired later by cyclic boundary conditions;
    the two spanwise patches are marked ``empty``.  Keeping this generator in
    the installed FVM package lets tutorials and benchmarks run without adding
    the repository or its tutorial tree to :mod:`sys.path`.
    """
    if n < 2:
        raise ValueError("n must be at least 2")
    if length <= 0.0:
        raise ValueError("length must be positive")

    spacing = float(length) / int(n)
    axis = np.linspace(0.0, float(length), int(n) + 1)
    mesh = box_mesh_3d(axis, axis, np.asarray([0.0, spacing], dtype=np.float64))
    for patch in mesh["boundary"]:
        if patch["name"] == "inlet":
            patch["name"] = "xmin"
        elif patch["name"] == "outlet":
            patch["name"] = "xmax"
        if str(patch["name"]).startswith("z"):
            patch["type"] = "empty"
    return mesh
