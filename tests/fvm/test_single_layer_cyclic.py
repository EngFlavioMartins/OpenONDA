"""Unit-span periodic extrusion preserves planar fields without dense pairing."""

from importlib.util import find_spec
import json
import os
from pathlib import Path
import subprocess
import sys
import tracemalloc

import numpy as np
import pytest

from source.solvers.fvm.assemble.diffusion import assemble_diffusion_term
from source.solvers.fvm.assemble.matrix_assembly import MatrixAssemblyWorkspace
from source.solvers.fvm.fields.gradients import compute_lsq_geometry, compute_lsq_gradient
from source.solvers.fvm.mesh.coupled import configure_cyclic_boundaries
from source.solvers.fvm.mesh.geometry import compute_mesh_geometry
from source.solvers.fvm.mesh.validation import MeshValidationError
from tests.support.fvm_mesh import structured_box


def periodic_mesh(nx=6, ny=5):
    mesh = structured_box(nx, ny, 1)
    for patch in mesh["boundary"]:
        if patch["name"] in {"zmin", "zmax"}:
            patch.update(
                velocity_type="cyclic",
                pressure_type="cyclic",
                neighbour_patch="zmax" if patch["name"] == "zmin" else "zmin",
            )
        else:
            patch.update(velocity_type="fixedValue", pressure_type="zeroGradient")
    geo = compute_mesh_geometry(mesh, gradient_scheme="lsq", compute_lsq=False)
    return mesh, geo


def test_one_layer_periodic_self_neighbours_have_unit_image_distance_and_zero_spanwise_flux():
    mesh, geo = periodic_mesh()
    configure_cyclic_boundaries(mesh, geo)
    faces = np.flatnonzero(mesh["boundary_neighbour_cell"] >= 0)
    np.testing.assert_array_equal(mesh["boundary_neighbour_cell"][faces], mesh["owners"][faces])
    np.testing.assert_allclose(np.abs(geo["cell_connection_vector"][faces, 2]), 1)
    np.testing.assert_allclose(geo["cell_connection_vector"][faces, :2], 0, atol=1e-14)
    np.testing.assert_allclose(geo["face_interpolation_weight"][faces], 0.5)
    geo.update(compute_lsq_geometry(mesh, geo))
    n, interior = mesh["n_cells"], mesh["n_interior_faces"]
    points = np.concatenate((geo["cell_centre"][:n], geo["face_centre"][interior:]))
    field = 1 + 0.3 * points[:, 0] - 0.7 * points[:, 1]
    gradient = compute_lsq_gradient(field, mesh, geo)
    np.testing.assert_allclose(gradient[:n, :, 0], np.tile([0.3, -0.7, 0], (n, 1)), atol=2e-14)
    cyclic = [p for p in mesh["boundary"] if p.get("velocity_type") == "cyclic"]
    flux = assemble_diffusion_term(field, gradient, 0.02, mesh, geo, cyclic)
    np.testing.assert_allclose(flux["flux_tf"][faces], 0, atol=1e-15)
    # Retain only spanwise coefficients. Both sides map onto the same CSR diagonal.
    for values in flux.values():
        values[:interior] = 0
    matrix = MatrixAssemblyWorkspace.create(mesh).update(flux, mesh)
    np.testing.assert_allclose(matrix.data, 0, atol=1e-15)


def test_periodic_pairing_matches_permuted_face_storage():
    mesh, geo = periodic_mesh()
    patch = next(p for p in mesh["boundary"] if p["name"] == "zmax")
    faces = np.arange(patch["start_face"], patch["start_face"] + patch["n_faces"])
    shuffled = np.random.default_rng(24).permutation(len(faces))
    for name in ("face_centre", "face_area", "face_area_vector"):
        geo[name][faces] = geo[name][faces[shuffled]]
    mesh["owners"][faces] = mesh["owners"][faces[shuffled]]
    configure_cyclic_boundaries(mesh, geo)
    paired = mesh["boundary_pair_face"][faces]
    np.testing.assert_allclose(geo["face_centre"][faces, :2], geo["face_centre"][paired, :2])
    np.testing.assert_array_equal(mesh["boundary_neighbour_cell"][faces], mesh["owners"][faces])


@pytest.mark.parametrize("defect", ["duplicate", "translation", "area", "normal", "reciprocal"])
def test_periodic_pairing_keeps_geometric_validation_checks(defect):
    mesh, geo = periodic_mesh()
    patch = next(p for p in mesh["boundary"] if p["name"] == "zmax")
    face = patch["start_face"]
    if defect == "duplicate":
        geo["face_centre"][face] = geo["face_centre"][face + 1]
    elif defect == "translation":
        geo["face_centre"][face, 0] += 0.02
    elif defect == "area":
        geo["face_area"][face] *= 1.1
    elif defect == "normal":
        geo["face_area_vector"][face] *= -1
    else:
        patch["neighbour_patch"] = "xmin"
    with pytest.raises(MeshValidationError):
        configure_cyclic_boundaries(mesh, geo)


def test_periodic_pairing_scratch_memory_is_bounded_for_broad_spanwise_patches():
    mesh, geo = periodic_mesh(40, 40)
    tracemalloc.start()
    try:
        configure_cyclic_boundaries(mesh, geo)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    # The former dense1600x1600x3 displacement array alone required61 MB.
    assert peak < 8 * 1024**2


@pytest.mark.integration
@pytest.mark.parametrize("cores", [1, 4])
def test_native_one_layer_periodic_flow_serial_and_mpi(tmp_path, cores):
    if cores > 1 and (find_spec("mpi4py") is None or find_spec("petsc4py") is None):
        pytest.skip("MPI and PETSc are required")
    script = Path(__file__).with_name("_single_layer_cyclic_probe.py")
    environment = dict(os.environ)
    environment.pop("_OPENONDA_MPI_CHILD", None)
    environment.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", NUMBA_NUM_THREADS="1")
    completed = subprocess.run(
        [sys.executable, str(script), str(tmp_path), str(cores)],
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stdout[-5000:] + completed.stderr[-5000:]
    for rank in range(cores):
        report = json.loads((tmp_path / f"rank-{rank}.json").read_text())
        assert report["status"] == "passed"
        assert report["cells"] == 30
        assert report["velocity_error"] < 1e-9
        assert report["continuity_error"] < 1e-9
