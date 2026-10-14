"""Small native mesh fixtures for solver consistency checks."""

import numpy as np


def structured_box(
    nx: int,
    ny: int,
    nz: int,
    lx: float = 1.0,
    ly: float = 1.0,
    lz: float = 1.0,
) -> dict:
    """Build a small native rectilinear box for FVM consistency checks."""
    if min(nx, ny, nz) < 1 or min(lx, ly, lz) <= 0.0:
        raise ValueError("Cell counts and box lengths must be positive")
    from openonda.fvm.mesh.rectilinear import box_mesh_3d

    mesh = box_mesh_3d(
        np.linspace(0.0, lx, nx + 1),
        np.linspace(0.0, ly, ny + 1),
        np.linspace(0.0, lz, nz + 1),
    )
    names = ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")
    for patch, name in zip(mesh["boundary"], names, strict=True):
        patch["name"] = name
    return mesh


def cylinder_o_grid(
    diameter: float,
    outer_diameter: float,
    n_circumference: int,
    n_radial: int,
    wall_spacing: float,
    depth: float = 1.0,
) -> dict:
    """Build a one-cell-thick polar O-grid around a cylinder at the origin.

    The radial spacing starts at ``wall_spacing`` and grows geometrically to
    the circular far field. Patches: ``cylinder`` (wall), ``farfield``,
    ``front``/``back`` (empty span planes).
    """
    inner, outer = 0.5 * diameter, 0.5 * outer_diameter
    low, high = 1.0, 2.0
    while wall_spacing * (high**n_radial - 1.0) / (high - 1.0) < outer - inner:
        high *= 2.0
    for _ in range(200):
        ratio = 0.5 * (low + high)
        if wall_spacing * (ratio**n_radial - 1.0) / (ratio - 1.0) < outer - inner:
            low = ratio
        else:
            high = ratio
    radii = inner + np.concatenate(([0.0], np.cumsum(wall_spacing * ratio ** np.arange(n_radial))))
    radii[-1] = outer
    angle = 2.0 * np.pi * np.arange(n_circumference) / n_circumference
    radius, theta = np.meshgrid(radii, angle, indexing="ij")
    ring = np.column_stack(
        (radius.ravel() * np.cos(theta.ravel()), radius.ravel() * np.sin(theta.ravel()))
    )
    n_ring = len(ring)
    points = np.vstack(
        (np.column_stack((ring, np.zeros(n_ring))), np.column_stack((ring, np.full(n_ring, depth))))
    )

    i, j = (
        index.ravel()
        for index in np.meshgrid(np.arange(n_radial), np.arange(n_circumference), indexing="ij")
    )
    jp = (j + 1) % n_circumference

    def point(k, ii, jj):
        return k * n_ring + ii * n_circumference + jj

    def cell(ii, jj):
        return ii * n_circumference + jj

    hexes = np.column_stack(
        (
            point(0, i, j),
            point(0, i + 1, j),
            point(0, i + 1, jp),
            point(0, i, jp),
            point(1, i, j),
            point(1, i + 1, j),
            point(1, i + 1, jp),
            point(1, i, jp),
        )
    )
    inner_ring = i < n_radial - 1
    interior = np.vstack(
        (
            np.column_stack(
                (point(0, i, jp), point(0, i + 1, jp), point(1, i + 1, jp), point(1, i, jp))
            ),
            np.column_stack(
                (point(0, i + 1, j), point(1, i + 1, j), point(1, i + 1, jp), point(0, i + 1, jp))
            )[inner_ring],
        )
    )
    first = np.concatenate((cell(i, j), cell(i, j)[inner_ring]))
    second = np.concatenate((cell(i, jp), cell(i + 1, j)[inner_ring]))
    neighbours = np.maximum(first, second)
    wall, rim = j[i == 0], j[i == n_radial - 1]
    patches = (
        ("cylinder", hexes[i == 0][:, [0, 3, 7, 4]], cell(0, wall), "wall"),
        ("farfield", hexes[i == n_radial - 1][:, [1, 5, 6, 2]], cell(n_radial - 1, rim), "patch"),
        ("front", hexes[:, [0, 3, 2, 1]], cell(i, j), "empty"),
        ("back", hexes[:, [4, 5, 6, 7]], cell(i, j), "empty"),
    )
    faces, owners, boundary, start = [interior], [np.minimum(first, second)], [], len(interior)
    for name, quads, patch_owners, patch_type in patches:
        faces.append(quads)
        owners.append(patch_owners)
        boundary.append(
            {"name": name, "start_face": start, "n_faces": len(quads), "type": patch_type}
        )
        start += len(quads)
    faces = np.vstack(faces).astype(np.int32)
    owners = np.concatenate(owners).astype(np.int32)

    # Orient every face from its owner towards the neighbour (or outwards).
    centres = points[hexes].mean(axis=1)
    corners = points[faces]
    normal = np.cross(corners[:, 2] - corners[:, 0], corners[:, 3] - corners[:, 1])
    target = corners.mean(axis=1)
    target[: len(neighbours)] = centres[neighbours]
    reverse = np.einsum("ij,ij->i", normal, target - centres[owners]) < 0.0
    faces[reverse] = faces[reverse][:, ::-1]
    return {
        "vertex_position": points,
        "faces": faces,
        "owners": owners,
        "neighbours": neighbours.astype(np.int32),
        "boundary": boundary,
        "n_cells": len(hexes),
        "n_faces": len(faces),
        "n_interior_faces": len(neighbours),
        "n_points": len(points),
    }
