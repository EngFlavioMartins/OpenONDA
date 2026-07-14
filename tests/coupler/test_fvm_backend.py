"""Coupling contract tests for the native FVM backend."""

from __future__ import annotations

import numpy as np
import pytest

from source.coupler.config.types import CouplerSetup

CONTRACT_METHODS = [
    "get_cell_center_coordinates",
    "get_cell_volumes",
    "get_velocity_field",
    "get_velocity_field_into",
    "get_vorticity_field",
    "get_vorticity_field_into",
    "get_boundary_face_center_coordinates",
    "get_boundary_face_normals",
    "get_boundary_face_areas",
    "n_procs",
    "set_cell_scalar_field",
    "set_cell_vector_field",
    "set_time_step",
    "set_kinematic_viscosity",
    "set_dirichlet_velocity_boundary_condition_vec",
    "set_robin_velocity_boundary_condition",
    "solve_pimple",
    "advance_time",
]

BOX = (-0.5, 0.5, -0.5, 0.5, -0.5, 0.5)


def _fvm_setup(tmp_path, **overrides):
    kwargs = {
        "backend": "fvm",
        "u_inf": [1.0, 0.0, 0.0],
        "nu": 0.01,
        "dt": 0.05,
        "t_end": 0.3,
        "fvm_box": BOX,
        "grid_spacing": 0.25,
        "h": 0.25,
        "buffer_thickness": 0.25,
        "dead_zone_h": 0.5,
        "wall_patch_name": None,
        "case_dir": str(tmp_path),
    }
    kwargs.update(overrides)
    return CouplerSetup(**kwargs)


def test_coupler_imports_without_ofw():
    from source.coupler import FVMVPMCoupler  # noqa: F401

    assert FVMVPMCoupler is not None


def test_backend_field_validation():
    assert CouplerSetup().backend == "ofw"
    assert CouplerSetup(backend="fvm").backend == "fvm"
    with pytest.raises(ValueError, match="backend"):
        CouplerSetup(backend="openfoam")


def test_backend_reaches_to_dict():
    d = CouplerSetup(backend="fvm").to_dict()
    assert d["fvm_solver"]["backend"] == "fvm"


def test_prepare_case_fvm_needs_no_openfoam_case(tmp_path):
    from source.coupler.core.solver import FVMVPMCoupler

    setup = _fvm_setup(tmp_path)
    FVMVPMCoupler.prepare_case(setup)  # no constant/polyMesh, no 0.orig
    assert (tmp_path / "solution").is_dir()
    # No OpenFOAM artifacts were created.
    assert not (tmp_path / "system").exists()
    assert not (tmp_path / "0").exists()


def test_prepare_case_fvm_restart_not_implemented(tmp_path):
    from source.coupler.core.solver import FVMVPMCoupler

    setup = _fvm_setup(tmp_path)
    with pytest.raises(NotImplementedError, match="restart"):
        FVMVPMCoupler.prepare_case(setup, restart=True)


def test_coupling_box_mesh_single_merged_patch():
    from source.coupler.core.helpers.fvm_backend import coupling_box_mesh

    mesh = coupling_box_mesh(BOX, 0.25, patch_name="numericalBoundary")
    assert mesh["n_elements"] == 4**3
    assert len(mesh["boundary"]) == 1
    patch = mesh["boundary"][0]
    assert patch["name"] == "numericalBoundary"
    assert patch["nFaces"] == 6 * 4 * 4
    assert patch["startFace"] == mesh["n_interior_faces"]
    assert mesh["n_faces"] == mesh["n_interior_faces"] + patch["nFaces"]
    # Points span the physical box, not [0, L]³.
    assert np.allclose(mesh["points"].min(axis=0), [-0.5, -0.5, -0.5])
    assert np.allclose(mesh["points"].max(axis=0), [0.5, 0.5, 0.5])


def test_coupling_box_mesh_rejects_non_conforming_spacing():
    from source.coupler.core.helpers.fvm_backend import coupling_box_mesh

    with pytest.raises(ValueError, match="integer"):
        coupling_box_mesh(BOX, 0.3)


@pytest.fixture(scope="module")
def built_backend(tmp_path_factory):
    from source.coupler.core.helpers.fvm_backend import build_fvm_backend

    tmp = tmp_path_factory.mktemp("fvm_backend")
    setup = _fvm_setup(tmp)
    return setup, build_fvm_backend(setup, quiet=True)


def test_build_rejects_ofw_setup(tmp_path):
    from source.coupler.core.helpers.fvm_backend import build_fvm_backend

    with pytest.raises(ValueError, match="backend"):
        build_fvm_backend(_fvm_setup(tmp_path, backend="ofw"))


def test_contract_methods_present(built_backend):
    _, fvm = built_backend
    missing = [m for m in CONTRACT_METHODS if not callable(getattr(fvm, m, None))]
    assert not missing, f"missing OFW-contract methods: {missing}"


def test_boundary_geometry_matches_coupler_expectations(built_backend):
    setup, fvm = built_backend
    fc = np.asarray(fvm.get_boundary_face_center_coordinates(setup.patch_name))
    fn = np.asarray(fvm.get_boundary_face_normals(setup.patch_name))
    fa = np.asarray(fvm.get_boundary_face_areas(setup.patch_name))

    n_faces = 6 * 4 * 4
    assert fc.shape == (n_faces, 3)
    assert fn.shape == (n_faces, 3)
    assert fa.shape == (n_faces,)
    # Unit outward normals (box centred at the origin → n·x_f > 0).
    assert np.allclose(np.linalg.norm(fn, axis=1), 1.0)
    assert (np.einsum("ij,ij->i", fn, fc) > 0).all()
    # Closed box: total area and Σ S_f = 0.
    assert np.isclose(fa.sum(), 6.0)
    assert np.allclose((fn * fa[:, None]).sum(axis=0), 0.0, atol=1e-12)
    # Every face centre lies on the box surface.
    on_surf = np.zeros(n_faces, dtype=bool)
    for ax in range(3):
        lo, hi = setup.fvm_box[2 * ax], setup.fvm_box[2 * ax + 1]
        on_surf |= np.isclose(fc[:, ax], lo) | np.isclose(fc[:, ax], hi)
    assert on_surf.all()


def test_dirichlet_bc_and_driver_split_produce_finite_flow(built_backend):
    import contextlib
    import io

    setup, fvm = built_backend
    fc = np.asarray(fvm.get_boundary_face_center_coordinates(setup.patch_name))
    u_bc = np.tile(setup.U_inf, (fc.shape[0], 1))
    fvm.set_dirichlet_velocity_boundary_condition_vec(u_bc, setup.patch_name)
    with contextlib.redirect_stdout(io.StringIO()):
        fvm.solve_pimple()
        fvm.advance_time()
    U = np.asarray(fvm.get_velocity_field())
    assert np.isfinite(U).all()
    # Uniform inflow through an empty box stays uniform.
    assert np.allclose(U.mean(axis=0), setup.U_inf, atol=1e-8)


def test_coupler_runtime_setters_apply(built_backend):
    _, fvm = built_backend
    # What FVMVPMCoupler.initialize() stamps on the injected backend.
    fvm.set_time_step(0.025)
    fvm.set_kinematic_viscosity(0.02)
    assert fvm.dt == 0.025
    assert fvm.config.transport.nu == 0.02
    assert fvm.n_procs() == 1
    # Coupled runs must not let the FVM adapt its own dt.
    assert fvm.config.time.adjust_timestep is False


def test_initialize_rejects_serial_backend_under_mpi(tmp_path, monkeypatch):
    import source.coupler.core.solver as coupler_mod

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(coupler_mod, "_mpi4py_comm", None)
    monkeypatch.setenv("OMPI_COMM_WORLD_SIZE", "2")

    class _SerialBackend:
        def n_procs(self):
            return 1

    setup = _fvm_setup(tmp_path)
    coupler = coupler_mod.FVMVPMCoupler(object(), _SerialBackend(), setup)
    with pytest.raises(RuntimeError, match="serial"):
        coupler.initialize()
