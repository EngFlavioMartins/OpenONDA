#!/usr/bin/env python3
"""Body-fitted polar O-grid around a circular cylinder (one cell thick).

The grid is built directly as the face-based ``mesh_data`` dictionary the FVM
solver reads. Radial lines are evenly spaced in angle; the radial spacing
starts at ``wall_spacing`` on the cylinder and grows geometrically to the
circular far-field boundary.

Patches: ``cylinder`` (wall), ``inlet`` and ``outlet`` (upstream, x < 0, and
downstream halves of the outer circle), ``front``/``back`` (``empty`` span
planes at z = 0 and z = depth).
"""

import numpy as np


def radial_nodes(inner_radius, outer_radius, n_radial, wall_spacing):
    """Radii from the wall to the far field with geometric growth of the spacing."""
    low, high = 1.0, 2.0
    while wall_spacing * (high**n_radial - 1.0) / (high - 1.0) < outer_radius - inner_radius:
        high *= 2.0
    for _ in range(200):
        ratio = 0.5 * (low + high)
        length = wall_spacing * (ratio**n_radial - 1.0) / (ratio - 1.0)
        if length < outer_radius - inner_radius:
            low = ratio
        else:
            high = ratio
    spacing = wall_spacing * ratio ** np.arange(n_radial)
    radii = inner_radius + np.concatenate(([0.0], np.cumsum(spacing)))
    radii[-1] = outer_radius
    return radii


def cylinder_o_grid(diameter, outer_diameter, n_circumference, n_radial, wall_spacing, depth):
    """Return the native mesh of a cylinder centred at the origin, axis along z."""
    radii = radial_nodes(0.5 * diameter, 0.5 * outer_diameter, n_radial, wall_spacing)
    angle = 2.0 * np.pi * np.arange(n_circumference) / n_circumference

    # Points (k, i, j): span level k, radius i, angle j
    n_ring = (n_radial + 1) * n_circumference
    radius, theta = np.meshgrid(radii, angle, indexing="ij")
    ring = np.column_stack(
        (radius.ravel() * np.cos(theta.ravel()), radius.ravel() * np.sin(theta.ravel()))
    )
    points = np.vstack(
        (np.column_stack((ring, np.zeros(n_ring))), np.column_stack((ring, np.full(n_ring, depth))))
    )

    i, j = np.meshgrid(np.arange(n_radial), np.arange(n_circumference), indexing="ij")
    i, j = i.ravel(), j.ravel()
    jp = (j + 1) % n_circumference

    def point(k, ii, jj):
        return k * n_ring + ii * n_circumference + jj

    def cell(ii, jj):
        return ii * n_circumference + jj

    # Hexahedra: bottom quad counter-clockwise about +z, then the top quad
    cell_vertices = np.column_stack(
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

    # Interior faces along radial lines (between angles j and j + 1)
    angular_faces = np.column_stack(
        (point(0, i, jp), point(0, i + 1, jp), point(1, i + 1, jp), point(1, i, jp))
    )
    angular_owner, angular_neighbour = cell(i, j), cell(i, jp)

    # Interior faces along circles (between radii i and i + 1)
    inner = i < n_radial - 1
    ring_faces = np.column_stack(
        (point(0, i + 1, j), point(1, i + 1, j), point(1, i + 1, jp), point(0, i + 1, jp))
    )[inner]
    ring_owner, ring_neighbour = cell(i, j)[inner], cell(i + 1, j)[inner]

    interior_faces = np.vstack((angular_faces, ring_faces))
    first = np.concatenate((angular_owner, ring_owner))
    second = np.concatenate((angular_neighbour, ring_neighbour))
    owners = np.minimum(first, second)
    neighbours = np.maximum(first, second)

    def far_field(sector):
        quads = np.column_stack(
            (
                point(0, n_radial, sector),
                point(1, n_radial, sector),
                point(1, n_radial, (sector + 1) % n_circumference),
                point(0, n_radial, (sector + 1) % n_circumference),
            )
        )
        return quads, cell(n_radial - 1, sector), "patch"

    wall = j[i == 0]
    sector = np.arange(n_circumference)
    upstream = np.cos(2.0 * np.pi * (sector + 0.5) / n_circumference) < 0.0
    patches = {
        "cylinder": (
            np.column_stack(
                (
                    point(0, 0, wall),
                    point(0, 0, (wall + 1) % n_circumference),
                    point(1, 0, (wall + 1) % n_circumference),
                    point(1, 0, wall),
                )
            ),
            cell(0, wall),
            "wall",
        ),
        "inlet": far_field(sector[upstream]),
        "outlet": far_field(sector[~upstream]),
        "front": (cell_vertices[:, [0, 3, 2, 1]], cell(i, j), "empty"),
        "back": (cell_vertices[:, [4, 5, 6, 7]], cell(i, j), "empty"),
    }

    # Orient every face from its owner towards the neighbour (or outwards)
    centres = points[cell_vertices].mean(axis=1)
    faces = [interior_faces]
    all_owners = [owners]
    boundary = []
    start = len(interior_faces)
    for name, (quads, patch_owners, patch_type) in patches.items():
        faces.append(quads)
        all_owners.append(patch_owners)
        boundary.append(
            {"name": name, "start_face": start, "n_faces": len(quads), "type": patch_type}
        )
        start += len(quads)
    faces = np.vstack(faces).astype(np.int32)
    all_owners = np.concatenate(all_owners)

    corners = points[faces]
    normal = np.cross(corners[:, 2] - corners[:, 0], corners[:, 3] - corners[:, 1])
    face_centre = corners.mean(axis=1)
    target = face_centre.copy()
    target[: len(neighbours)] = centres[neighbours]
    reverse = np.einsum("ij,ij->i", normal, target - centres[all_owners]) < 0.0
    faces[reverse] = faces[reverse][:, ::-1]

    return {
        "vertex_position": points,
        "faces": faces,
        "owners": all_owners.astype(np.int32),
        "neighbours": neighbours.astype(np.int32),
        "boundary": boundary,
        "n_cells": len(cell_vertices),
        "n_faces": len(faces),
        "n_interior_faces": len(neighbours),
        "n_points": len(points),
        "cell_vertex_indices": cell_vertices.astype(np.int32),
        "cell_type_code": np.full(len(cell_vertices), 5, dtype=np.int32),
    }
