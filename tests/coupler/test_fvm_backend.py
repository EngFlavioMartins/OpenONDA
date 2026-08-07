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
    "get_velocity_gradient_field",
    "get_velocity_gradient_field_into",
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
    "solve_pimple",
    "advance_time",
]

BOX = (-0.5, 0.5, -0.5, 0.5, -0.5, 0.5)


def _fvm_setup(tmp_path, **overrides):
    """Coupling-only setup: physics/time/mesh live on the injected solvers."""
    kwargs = {
        "backend": "fvm",
        "u_inf": [1.0, 0.0, 0.0],
        "h": 0.25,
        "buffer_thickness": 0.25,
        "dead_zone_h": 0.5,
        "wall_patch_name": None,
        "case_dir": str(tmp_path),
    }
    kwargs.update(overrides)
    return CouplerSetup(**kwargs)


def _build_backend(tmp_path, spacing=0.25, box=BOX, hole_box=None, wall_patch_name=None):
    from source.coupler.core.helpers.fvm_backend import build_fvm_backend, coupling_box_mesh

    return build_fvm_backend(
        mesh_data=coupling_box_mesh(
            box, spacing, hole_box=hole_box, wall_patch_name=wall_patch_name or "cube"
        ),
        case_dir=str(tmp_path),
        dt=0.05,
        t_end=0.3,
        nu=0.01,
        u_inf=[1.0, 0.0, 0.0],
        wall_patch_name=wall_patch_name,
        quiet=True,
    )


def test_coupler_imports_without_ofw():
    from source.coupler import FVMVPMCoupler  # noqa: F401

    assert FVMVPMCoupler is not None


def test_backend_field_validation():
    assert CouplerSetup().backend == "fvm"
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


def test_prepare_case_fvm_restart_requires_complete_checkpoint(tmp_path):
    from source.coupler.core.solver import FVMVPMCoupler

    setup = _fvm_setup(tmp_path)
    with pytest.raises(FileNotFoundError, match="coupled checkpoint"):
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
    tmp = tmp_path_factory.mktemp("fvm_backend")
    setup = _fvm_setup(tmp)
    return setup, _build_backend(tmp)


def test_ofw_setup_requires_case_fields(tmp_path):
    """The OFW backend writes the case from CouplerSetup, so the Eulerian-case
    values are required there — and only there (the native backend leaves them
    to the injected solvers)."""
    from source.coupler.core.solver import FVMVPMCoupler

    with pytest.raises(ValueError, match="requires"):
        FVMVPMCoupler.prepare_case(CouplerSetup(backend="ofw", case_dir=str(tmp_path)))


def test_fvm_setup_rejects_contradicting_physics(tmp_path):
    """A CouplerSetup value that contradicts the injected FVM solver's own
    configuration raises instead of being silently reconciled."""
    from source.coupler.core.solver import FVMVPMCoupler

    fvm = _build_backend(tmp_path)  # dt=0.05, nu=0.01
    setup = _fvm_setup(tmp_path, dt=0.99)
    coupler = FVMVPMCoupler(object(), fvm, setup)
    coupler.ofw = fvm
    with pytest.raises(ValueError, match="owns this value"):
        coupler._resolve_eulerian_ownership()


def test_fvm_setup_adopts_solver_values_and_derives_box(tmp_path):
    """With coupling-only CouplerSetup the coupler adopts dt/t_end/nu from the
    injected solver and derives fvm_box from the coupling-patch geometry."""
    from source.coupler.core.solver import FVMVPMCoupler

    fvm = _build_backend(tmp_path)
    setup = _fvm_setup(tmp_path)
    assert setup.dt is None and setup.nu is None and setup.fvm_box is None
    coupler = FVMVPMCoupler(object(), fvm, setup)
    coupler.ofw = fvm
    coupler._resolve_eulerian_ownership()
    assert coupler.dt_fvm == 0.05
    assert coupler.t_end == 0.3
    assert setup.nu == 0.01
    assert np.allclose(setup.fvm_box, BOX, atol=1e-12)


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
        lo, hi = BOX[2 * ax], BOX[2 * ax + 1]
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


def test_velocity_gradient_contract_uses_configured_reconstruction(built_backend):
    _, fvm = built_backend
    gradient = np.asarray(fvm.get_velocity_gradient_field())
    assert gradient.shape == (fvm.mesh_data["n_elements"], 3, 3)
    assert np.isfinite(gradient).all()
    out = np.empty_like(gradient)
    assert fvm.get_velocity_gradient_field_into(out) is out
    np.testing.assert_array_equal(out, gradient)


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


# ---------------------------------------------------------------------------
# Body-fitted mesh: carved hole + wall patch (OpenFOAM-like topology)
# ---------------------------------------------------------------------------

HOLE = (-0.5, 0.5, -0.5, 0.5, -0.5, 0.5)


def test_carved_mesh_hole_and_wall_patch():
    from source.coupler.core.helpers.fvm_backend import coupling_box_mesh

    mesh = coupling_box_mesh(
        (-1.5, 1.5, -1.5, 1.5, -1.5, 1.5), 0.125, hole_box=HOLE, wall_patch_name="cube"
    )
    n_side, n_cube = 24, 8
    assert mesh["n_elements"] == n_side**3 - n_cube**3
    names = [b["name"] for b in mesh["boundary"]]
    assert names == ["numericalBoundary", "cube"]
    wall = mesh["boundary"][1]
    assert wall["type"] == "wall"
    assert wall["nFaces"] == 6 * n_cube**2
    # Patches tile the boundary contiguously after the interior faces.
    coupling = mesh["boundary"][0]
    assert wall["startFace"] == coupling["startFace"] + coupling["nFaces"]
    assert mesh["n_faces"] == mesh["n_interior_faces"] + coupling["nFaces"] + wall["nFaces"]
    # Owner/neighbour indices are compact after the carve.
    assert mesh["owners"].max() < mesh["n_elements"]
    assert mesh["neighbours"].max() < mesh["n_elements"]

    # Wall faces lie on the cube surface with outward-of-fluid normals
    # (pointing into the hole, i.e. toward the cube centre at the origin).
    pts = mesh["points"]
    sl = slice(wall["startFace"], wall["startFace"] + wall["nFaces"])
    for face in mesh["faces"][sl.start : sl.stop]:
        p = pts[face]
        c = p.mean(axis=0)
        assert np.isclose(np.abs(c), 0.5).any()
        n = np.cross(p[1] - p[0], p[2] - p[0])
        assert np.dot(n, -c) > 0.0


def test_carved_mesh_misaligned_hole_raises():
    from source.coupler.core.helpers.fvm_backend import coupling_box_mesh

    with pytest.raises(ValueError, match="mesh plane"):
        coupling_box_mesh((-1.5, 1.5, -1.5, 1.5, -1.5, 1.5), 0.075, hole_box=HOLE)


def test_builder_body_fitted_cube(tmp_path):
    import contextlib
    import io

    setup = _fvm_setup(
        tmp_path,
        h=0.125,
        wall_patch_name="cube",
        buffer_thickness=0.375,
    )
    fvm = _build_backend(
        tmp_path,
        spacing=0.125,
        box=(-1.5, 1.5, -1.5, 1.5, -1.5, 1.5),
        hole_box=HOLE,
        wall_patch_name="cube",
    )

    # No fluid cells inside the body — the coupling can never inject
    # fictitious interior vorticity into the VPM.
    cc = np.asarray(fvm.get_cell_center_coordinates())
    assert np.all(np.abs(cc) < 0.5, axis=1).sum() == 0

    # Wall patch reachable through the coupler contract; no-slip configured;
    # wall-force logging on (as an explicit ForceSampler).
    wf = np.asarray(fvm.get_boundary_face_center_coordinates("cube"))
    assert wf.shape[0] == 6 * 8**2
    force_sampler = next(s for s in fvm.config.samplers if s.name == "forces_history")
    assert force_sampler.patch_names == ["cube"]
    wall_cfg = next(b for b in fvm.config.boundaries if b.name == "cube")
    assert wall_cfg.type_U == "fixedValue" and wall_cfg.value_U == [0.0, 0.0, 0.0]

    # One PIMPLE step stays finite and generates wall vorticity.
    fc = np.asarray(fvm.get_boundary_face_center_coordinates(setup.patch_name))
    u_bc = np.tile(setup.U_inf, (fc.shape[0], 1))
    fvm.set_dirichlet_velocity_boundary_condition_vec(u_bc, setup.patch_name)
    with contextlib.redirect_stdout(io.StringIO()):
        fvm.solve_pimple()
        fvm.advance_time()
    U = np.asarray(fvm.get_velocity_field())
    assert np.isfinite(U).all()
    w = np.linalg.norm(np.asarray(fvm.get_vorticity_field()), axis=1)
    assert w.max() > 1.0
