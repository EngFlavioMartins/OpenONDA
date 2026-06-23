"""Structured orthogonal hexahedral box mesh generator for verification tests.

Produces a ``mesh_data`` dict in the same polyMesh-style format the FVM solver
consumes (see ``conftest.hand_built_3d_mesh``), but for an arbitrary
``nx × ny × nz`` grid on ``[0,lx] × [0,ly] × [0,lz]``.

Why this exists: order-of-accuracy verification (MMS, Taylor–Green) must isolate
the *scheme* order from *mesh-quality* effects.  On gmsh tetrahedra the nominally
2nd-order convection degrades to ~1st order because of non-orthogonality and the
explicit gradient-based corrections.  On a structured orthogonal hex mesh the
non-orthogonal correction is identically zero and the base scheme can reach its
design order, so a clean convergence rate is recoverable.

Face winding is made robust by computing each face normal and reversing the node
order when it does not point owner→neighbour (interior) / outward (boundary) —
the sign convention ``geometry.compute_geometry`` relies on (it derives
``face_sf`` from the winding alone and uses ``sign·Sf·Cf`` for the volume).

Boundary patches are emitted contiguously in the order
``xmin, xmax, ymin, ymax, zmin, zmax`` so ``startFace``/``nFaces`` address them.
"""

from __future__ import annotations

import numpy as np


def structured_box(nx, ny, nz, lx=1.0, ly=1.0, lz=1.0):
    """Return ``mesh_data`` for an ``nx×ny×nz`` orthogonal hex box.

    Periodic-friendly: cell centres are at ``(i+0.5)·dx`` etc., so a field
    sampled at the centroids of opposite faces matches under translation.
    """
    dx, dy, dz = lx / nx, ly / ny, lz / nz
    npx, npy, npz = nx + 1, ny + 1, nz + 1

    # --- points -----------------------------------------------------------
    pid = lambda i, j, k: i + npx * (j + npy * k)  # noqa: E731
    points = np.zeros((npx * npy * npz, 3), dtype=np.float64)
    for k in range(npz):
        for j in range(npy):
            for i in range(npx):
                points[pid(i, j, k)] = (i * dx, j * dy, k * dz)

    cid = lambda i, j, k: i + nx * (j + ny * k)  # noqa: E731

    def oriented(quad, desired_dir):
        """Reverse the node ring if its right-hand normal opposes desired_dir."""
        q = np.asarray(quad, dtype=np.int32)
        p = points[q]
        n = np.cross(p[1] - p[0], p[2] - p[0])
        if np.dot(n, desired_dir) < 0:
            q = q[::-1].copy()
        return q

    interior_faces, interior_owners, interior_neighbours = [], [], []
    patches = {name: [] for name in ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")}
    patch_owners = {name: [] for name in patches}

    ex, ey, ez = np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0, 0, 1.0])

    # --- x-perpendicular faces (at x = i·dx) ------------------------------
    for i in range(nx + 1):
        for k in range(nz):
            for j in range(ny):
                quad = [pid(i, j, k), pid(i, j + 1, k), pid(i, j + 1, k + 1), pid(i, j, k + 1)]
                if i == 0:
                    patches["xmin"].append(oriented(quad, -ex))
                    patch_owners["xmin"].append(cid(0, j, k))
                elif i == nx:
                    patches["xmax"].append(oriented(quad, ex))
                    patch_owners["xmax"].append(cid(nx - 1, j, k))
                else:
                    interior_faces.append(oriented(quad, ex))
                    interior_owners.append(cid(i - 1, j, k))
                    interior_neighbours.append(cid(i, j, k))

    # --- y-perpendicular faces (at y = j·dy) ------------------------------
    for j in range(ny + 1):
        for k in range(nz):
            for i in range(nx):
                quad = [pid(i, j, k), pid(i + 1, j, k), pid(i + 1, j, k + 1), pid(i, j, k + 1)]
                if j == 0:
                    patches["ymin"].append(oriented(quad, -ey))
                    patch_owners["ymin"].append(cid(i, 0, k))
                elif j == ny:
                    patches["ymax"].append(oriented(quad, ey))
                    patch_owners["ymax"].append(cid(i, ny - 1, k))
                else:
                    interior_faces.append(oriented(quad, ey))
                    interior_owners.append(cid(i, j - 1, k))
                    interior_neighbours.append(cid(i, j, k))

    # --- z-perpendicular faces (at z = k·dz) ------------------------------
    for k in range(nz + 1):
        for j in range(ny):
            for i in range(nx):
                quad = [pid(i, j, k), pid(i + 1, j, k), pid(i + 1, j + 1, k), pid(i, j + 1, k)]
                if k == 0:
                    patches["zmin"].append(oriented(quad, -ez))
                    patch_owners["zmin"].append(cid(i, j, 0))
                elif k == nz:
                    patches["zmax"].append(oriented(quad, ez))
                    patch_owners["zmax"].append(cid(i, j, nz - 1))
                else:
                    interior_faces.append(oriented(quad, ez))
                    interior_owners.append(cid(i, j, k - 1))
                    interior_neighbours.append(cid(i, j, k))

    # --- assemble: interior first, then patches in fixed order ------------
    faces = list(interior_faces)
    owners = list(interior_owners)
    n_interior = len(interior_faces)

    boundary = []
    start = n_interior
    for name in ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax"):
        pf, po = patches[name], patch_owners[name]
        boundary.append({"name": name, "startFace": start, "nFaces": len(pf), "type": "patch"})
        faces.extend(pf)
        owners.extend(po)
        start += len(pf)

    n_elements = nx * ny * nz
    n_faces = len(faces)

    return {
        "points": points,
        "faces": [np.asarray(f, dtype=np.int32) for f in faces],
        "owners": np.asarray(owners, dtype=np.int32),
        "neighbours": np.asarray(interior_neighbours, dtype=np.int32),
        "boundary": boundary,
        "n_elements": n_elements,
        "n_faces": n_faces,
        "n_interior_faces": n_interior,
        "n_points": points.shape[0],
    }
