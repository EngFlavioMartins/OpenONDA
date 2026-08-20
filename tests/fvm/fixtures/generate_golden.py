from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "source"))

from source.solvers.FVM.assemble.diffusion import assemble_diffusion_term
from source.solvers.FVM.assemble.matrix_assembly import assemble_matrix_from_fluxes_vectorized
from source.solvers.FVM.fields.gradients import compute_gauss_gradient
from source.solvers.FVM.mesh.geometry import compute_mesh_geometry


def hand_built_3d_mesh():
    points = np.array(
        [
            [-1.0, -1.0, -1.0],
            [0.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 0.0, -1.0],
            [0.0, 0.0, -1.0],
            [1.0, 0.0, -1.0],
            [-1.0, 1.0, -1.0],
            [0.0, 1.0, -1.0],
            [1.0, 1.0, -1.0],
            [-1.0, -1.0, 0.0],
            [0.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [-1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [-1.0, -1.0, 1.0],
            [0.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
            [-1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [-1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )

    faces = [
        np.array([4, 13, 10, 1], dtype=np.int32),
        np.array([7, 16, 13, 4], dtype=np.int32),
        np.array([13, 22, 19, 10], dtype=np.int32),
        np.array([16, 25, 22, 13], dtype=np.int32),
        np.array([3, 12, 13, 4], dtype=np.int32),
        np.array([12, 21, 22, 13], dtype=np.int32),
        np.array([4, 13, 14, 5], dtype=np.int32),
        np.array([13, 22, 23, 14], dtype=np.int32),
        np.array([13, 12, 9, 10], dtype=np.int32),
        np.array([14, 13, 10, 11], dtype=np.int32),
        np.array([16, 15, 12, 13], dtype=np.int32),
        np.array([17, 16, 13, 14], dtype=np.int32),
        np.array([0, 9, 12, 3], dtype=np.int32),
        np.array([3, 12, 15, 6], dtype=np.int32),
        np.array([9, 18, 21, 12], dtype=np.int32),
        np.array([12, 21, 24, 15], dtype=np.int32),
        np.array([5, 14, 11, 2], dtype=np.int32),
        np.array([8, 17, 14, 5], dtype=np.int32),
        np.array([14, 23, 20, 11], dtype=np.int32),
        np.array([17, 26, 23, 14], dtype=np.int32),
        np.array([1, 10, 9, 0], dtype=np.int32),
        np.array([2, 11, 10, 1], dtype=np.int32),
        np.array([10, 19, 18, 9], dtype=np.int32),
        np.array([11, 20, 19, 10], dtype=np.int32),
        np.array([6, 15, 16, 7], dtype=np.int32),
        np.array([7, 16, 17, 8], dtype=np.int32),
        np.array([15, 24, 25, 16], dtype=np.int32),
        np.array([16, 25, 26, 17], dtype=np.int32),
        np.array([1, 0, 3, 4], dtype=np.int32),
        np.array([2, 1, 4, 5], dtype=np.int32),
        np.array([4, 3, 6, 7], dtype=np.int32),
        np.array([5, 4, 7, 8], dtype=np.int32),
        np.array([22, 21, 18, 19], dtype=np.int32),
        np.array([23, 22, 19, 20], dtype=np.int32),
        np.array([25, 24, 21, 22], dtype=np.int32),
        np.array([26, 25, 22, 23], dtype=np.int32),
    ]

    owners = np.array(
        [
            0,
            2,
            4,
            6,
            0,
            4,
            1,
            5,
            0,
            1,
            2,
            3,
            0,
            2,
            4,
            6,
            1,
            3,
            5,
            7,
            0,
            1,
            4,
            5,
            2,
            3,
            6,
            7,
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
        ],
        dtype=np.int32,
    )
    neighbours = np.array([1, 3, 5, 7, 2, 6, 3, 7, 4, 5, 6, 7], dtype=np.int32)

    boundary = [
        {"name": "xmin", "start_face": 12, "n_faces": 4, "type": "patch"},
        {"name": "xmax", "start_face": 16, "n_faces": 4, "type": "patch"},
        {"name": "ymin", "start_face": 20, "n_faces": 4, "type": "patch"},
        {"name": "ymax", "start_face": 24, "n_faces": 4, "type": "patch"},
        {"name": "zmin", "start_face": 28, "n_faces": 4, "type": "patch"},
        {"name": "zmax", "start_face": 32, "n_faces": 4, "type": "patch"},
    ]

    mesh_data = {
        "points": points,
        "faces": faces,
        "owners": owners,
        "neighbours": neighbours,
        "boundary": boundary,
        "n_cells": 8,
        "n_faces": 36,
        "n_interior_faces": 12,
        "n_points": 27,
    }
    return mesh_data


def main():
    output_dir = Path(__file__).resolve().parent
    mesh_data = hand_built_3d_mesh()
    geo_data = compute_mesh_geometry(mesh_data)

    volumes = geo_data["cell_volumes"]
    print("Element volumes:", volumes)
    assert np.allclose(volumes, 1.0), f"Volumes not all 1.0: {volumes}"

    centroids = geo_data["cell_centroids"]
    print("Element centroids:\n", centroids)

    n_cells = mesh_data["n_cells"]
    n_interior = mesh_data["n_interior_faces"]

    phi_elem = centroids[:, 0] + centroids[:, 1] + centroids[:, 2]

    face_centroids = geo_data["face_centroids"]
    phi_b = (
        face_centroids[n_interior:, 0]
        + face_centroids[n_interior:, 1]
        + face_centroids[n_interior:, 2]
    )
    face_flux = np.concatenate([phi_elem, phi_b])

    grad_phi = compute_gauss_gradient(face_flux, mesh_data, geo_data)
    grad_elem = grad_phi[:n_cells]
    print("Gradient phi (element average):\n", grad_elem.mean(axis=0))
    expected = np.array([1.0, 1.0, 1.0])
    assert np.allclose(grad_elem.mean(axis=0), expected, atol=1e-10), (
        f"Gradient not [1,1,1]: {grad_elem.mean(axis=0)}"
    )

    gamma = np.ones(n_cells, dtype=np.float64)
    boundaries = mesh_data["boundary"]
    flux_data = assemble_diffusion_term(
        face_flux, grad_elem, gamma, mesh_data, geo_data, boundaries
    )
    A = assemble_matrix_from_fluxes_vectorized(flux_data, mesh_data)
    A_dense = A.toarray()
    diag = A_dense.diagonal()
    row_sum = np.array(A_dense.sum(axis=1)).flatten()

    output_path = output_dir / "golden_reference.npz"
    np.savez(
        output_path,
        cell_volumes=volumes,
        cell_centroids=centroids,
        face_sf=geo_data["face_sf"],
        face_areas=geo_data["face_areas"],
        face_weights=geo_data["face_weights"],
        grad_phi=grad_elem,
        matrix_diagonal=diag,
        matrix_row_sum=row_sum,
    )
    print(f"Golden reference saved to {output_path}")
    print("All checks passed.")


if __name__ == "__main__":
    main()
