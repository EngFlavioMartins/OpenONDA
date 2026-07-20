"""Validated coupled-boundary topology for finite-volume operators."""

from __future__ import annotations

import numpy as np

from .validation import MeshValidationError


def _patch_faces(patch: dict) -> np.ndarray:
    start = int(patch["startFace"])
    return np.arange(start, start + int(patch["nFaces"]), dtype=np.int32)


def configure_cyclic_boundaries(mesh_data: dict, geo_data: dict) -> None:
    """Pair translational cyclic patches and expose their coupled-cell topology.

    OpenFOAM keeps each side of a cyclic interface as a boundary patch.  The
    solver retains that representation for field I/O, while this function adds
    the paired owner-cell column and periodic centre vector needed by discrete
    operators.  Rotational cyclic transforms are rejected explicitly.
    """
    n_faces = int(mesh_data["n_faces"])
    paired_faces = np.full(n_faces, -1, dtype=np.int32)
    paired_cells = np.full(n_faces, -1, dtype=np.int32)
    translations = np.zeros((n_faces, 3), dtype=np.float64)
    patches = {patch["name"]: patch for patch in mesh_data["boundary"]}
    cyclic = {
        name: patch
        for name, patch in patches.items()
        if patch.get("bc_type_U") == "cyclic" or patch.get("bc_type_p") == "cyclic"
    }

    for name, patch in cyclic.items():
        if patch.get("bc_type_U") != "cyclic" or patch.get("bc_type_p") != "cyclic":
            raise MeshValidationError(
                f"Cyclic patch {name!r} must use cyclic for both velocity and pressure"
            )
        transform = str(patch.get("transform", "translational")).lower()
        if transform not in {"translational", "none"}:
            raise MeshValidationError(
                f"Cyclic patch {name!r} uses unsupported transform {transform!r}; "
                "only translational coupling is implemented"
            )
        neighbour_name = patch.get("neighbourPatch") or patch.get("neighbour_patch")
        if not neighbour_name:
            raise MeshValidationError(f"Cyclic patch {name!r} is missing neighbourPatch")
        neighbour = patches.get(str(neighbour_name))
        if neighbour is None:
            raise MeshValidationError(
                f"Cyclic patch {name!r} references unknown neighbourPatch {neighbour_name!r}"
            )
        if neighbour.get("bc_type_U") != "cyclic" or neighbour.get("bc_type_p") != "cyclic":
            raise MeshValidationError(
                f"Cyclic neighbour {neighbour_name!r} must use cyclic for velocity and pressure"
            )
        reciprocal = neighbour.get("neighbourPatch") or neighbour.get("neighbour_patch")
        if reciprocal != name:
            raise MeshValidationError(
                f"Cyclic patches {name!r} and {neighbour_name!r} are not reciprocal"
            )

        faces = _patch_faces(patch)
        other_faces = _patch_faces(neighbour)
        if len(faces) != len(other_faces):
            raise MeshValidationError(
                f"Cyclic patches {name!r} and {neighbour_name!r} have different face counts"
            )
        if len(faces) == 0:
            continue

        face_centres = geo_data["face_centroids"]
        shift = np.mean(face_centres[faces], axis=0) - np.mean(face_centres[other_faces], axis=0)
        delta = face_centres[faces, None, :] - (face_centres[other_faces][None, :, :] + shift)
        distance = np.linalg.norm(delta, axis=2)
        match = np.argmin(distance, axis=1)
        scale = max(float(np.ptp(mesh_data["points"], axis=0).max()), 1.0)
        tolerance = max(1e-10, 1e-8 * scale)
        if len(np.unique(match)) != len(match) or np.any(
            distance[np.arange(len(faces)), match] > tolerance
        ):
            raise MeshValidationError(
                f"Cyclic faces on {name!r} and {neighbour_name!r} do not match by translation"
            )

        matched_faces = other_faces[match]
        area = geo_data["face_areas"]
        if not np.allclose(area[faces], area[matched_faces], rtol=1e-8, atol=tolerance**2):
            raise MeshValidationError(f"Cyclic face areas differ on patch pair {name!r}")
        normals = geo_data["face_sf"]
        if not np.allclose(normals[faces], -normals[matched_faces], rtol=1e-8, atol=tolerance**2):
            raise MeshValidationError(
                f"Cyclic face normals are not opposite on patch pair {name!r}"
            )

        owners = mesh_data["owners"]
        paired_faces[faces] = matched_faces
        paired_cells[faces] = owners[matched_faces]
        translations[faces] = shift
        patch["_paired_cells"] = paired_cells[faces]
        patch["_paired_faces"] = paired_faces[faces]

    coupled = paired_cells >= 0
    mesh_data["boundary_pair_faces"] = paired_faces
    mesh_data["boundary_neighbours"] = paired_cells
    geo_data["boundary_translations"] = translations
    if np.any(coupled):
        owners = mesh_data["owners"]
        centres = geo_data["element_centroids"]
        face_centres = geo_data["face_centroids"]
        image_centres = centres[paired_cells[coupled]] + translations[coupled]
        vectors = image_centres - centres[owners[coupled]]
        if np.any(np.linalg.norm(vectors, axis=1) <= 1e-30):
            raise MeshValidationError("Cyclic coupling produced a zero centre-to-centre distance")
        geo_data["face_cf_vector"][coupled] = vectors
        geo_data["face_cf"][coupled] = face_centres[coupled] - centres[owners[coupled]]
        geo_data["face_ff"][coupled] = face_centres[coupled] - image_centres
        normal = geo_data["face_sf"][coupled] / geo_data["face_areas"][coupled, None]
        owner_distance = np.sum(geo_data["face_cf"][coupled] * normal, axis=1)
        total_distance = np.sum(vectors * normal, axis=1)
        weights = owner_distance / total_distance
        if np.any((weights <= 0.0) | (weights >= 1.0)):
            raise MeshValidationError(
                "Cyclic interpolation weights must lie strictly inside (0, 1)"
            )
        geo_data["face_weights"][coupled] = weights

    # Cyclic pairing adds off-diagonal boundary couplings.  Invalidate the
    # mesh-owned sparse pattern so assembly sees the completed topology.
    mesh_data.pop("_fvm_csr_patterns", None)
