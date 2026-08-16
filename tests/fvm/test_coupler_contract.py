"""Coupler-contract conformance for the native FVM solver.

The coupler (``source/coupler/``) drives the native FVM through a fixed
duck-typed API. These tests guard that public contract:
every method exists with the right shapes, the driver splits into
``solve_pimple`` (no time advance) + ``advance_time``, the per-face Dirichlet
donor BC applies, and the fvOptions-equivalent fringe source
``S = λ(Utarget − U)`` is wired from the registered fields to the momentum
operator.
"""

import contextlib
import io

import numpy as np
import pytest

from source.solvers.FVM import (
    BoundaryConfig,
    FVMSetup,
    LinearSolverConfig,
    PimpleControl,
    SchemesConfig,
    Solver,
    TimeConfig,
    TransportConfig,
)
from source.solvers.FVM.assemble.convection import compute_volumetric_face_flux
from source.solvers.FVM.assemble.momentum import assemble_momentum_equation
from source.solvers.FVM.core.parallel import ParallelContext
from source.solvers.FVM.mesh.geometry import compute_mesh_geometry
from source.solvers.FVM.solve.linear_interface import solve_linear_system

from ._structured_mesh import structured_box

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
    "set_freestream_velocity_boundary_condition_vec",
    "set_directional_freestream_velocity_boundary_condition_vec",
    "set_freestream_pressure_boundary_condition",
    "set_directional_freestream_pressure_boundary_condition",
    "solve_pimple",
    "advance_time",
]


def _make_solver(init=(0.0, 0.0, 0.0)):
    mesh = structured_box(6, 6, 6)
    sp_schemes = SchemesConfig(convection_scheme="central")
    sp_linear = LinearSolverConfig(linear_solver="spsolve")
    sp_pimple = PimpleControl(n_correctors=2)
    bnds = [BoundaryConfig.wall(n) for n in ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")]
    cfg = FVMSetup(
        case_name="coupler-contract",
        time=TimeConfig(delta_t=0.1, end_time=1.0, write_interval=999),
        schemes=sp_schemes,
        linear=sp_linear,
        pimple=sp_pimple,
        transport=TransportConfig(density=1.0, nu=0.05),
        boundaries=bnds,
        initial_U=list(init),
    )
    with contextlib.redirect_stdout(io.StringIO()):
        s = Solver(cfg, case_dir="/tmp/openonda-coupler-contract", mesh_data=mesh)
        s.auto_write = False
    return s, mesh


def test_all_contract_methods_present():
    s, _ = _make_solver()
    missing = [m for m in CONTRACT_METHODS if not callable(getattr(s, m, None))]
    assert not missing, f"missing coupler-contract methods: {missing}"


def test_getter_shapes_and_normals():
    s, mesh = _make_solver()
    n = mesh["n_elements"]
    assert s.get_cell_center_coordinates().shape == (n, 3)
    assert s.get_cell_volumes().shape == (n,)
    assert s.get_velocity_field().shape == (n, 3)
    assert s.get_vorticity_field().shape == (n, 3)
    nf = next(b["nFaces"] for b in mesh["boundary"] if b["name"] == "xmax")
    assert s.get_boundary_face_center_coordinates("xmax").shape == (nf, 3)
    normals = s.get_boundary_face_normals("xmax")
    assert np.allclose(np.linalg.norm(normals, axis=1), 1.0)
    assert s.get_boundary_face_areas("xmax").shape == (nf,)
    assert s.n_procs() == 1


def test_field_getters_fill_existing_buffers():
    s, mesh = _make_solver()
    n = mesh["n_elements"]
    u_out = np.empty((n, 3), dtype=np.float64)
    w_out = np.empty((n, 3), dtype=np.float64)

    assert s.get_velocity_field_into(u_out) is u_out
    assert s.get_vorticity_field_into(w_out) is w_out
    np.testing.assert_allclose(u_out, s.get_velocity_field())
    np.testing.assert_allclose(w_out, s.get_vorticity_field())


def test_field_getters_reject_wrong_buffer_shape():
    s, mesh = _make_solver()
    bad = np.empty((mesh["n_elements"], 2), dtype=np.float64)
    with pytest.raises(ValueError):
        s.get_velocity_field_into(bad)


def test_replicated_parallel_getters_expose_cells_on_root_only():
    root, mesh = _make_solver()
    root.parallel = ParallelContext(mode="petsc_replicated", rank=0, size=2)
    assert root.n_procs() == 2
    assert root.get_velocity_field().shape == (mesh["n_elements"], 3)

    worker, _ = _make_solver()
    worker.parallel = ParallelContext(mode="petsc_replicated", rank=1, size=2)
    assert worker.n_procs() == 2
    assert worker.get_cell_center_coordinates().shape == (0, 3)
    assert worker.get_cell_volumes().shape == (0,)
    assert worker.get_velocity_field().shape == (0, 3)
    assert worker.get_vorticity_field().shape == (0, 3)
    assert worker.get_boundary_face_center_coordinates("xmax").shape == (0, 3)
    assert worker.get_boundary_face_normals("xmax").shape == (0, 3)
    assert worker.get_boundary_face_areas("xmax").shape == (0,)

    u_out = np.empty((0, 3), dtype=np.float64)
    w_out = np.empty((0, 3), dtype=np.float64)
    assert worker.get_velocity_field_into(u_out) is u_out
    assert worker.get_vorticity_field_into(w_out) is w_out


def test_scalar_param_setters():
    s, _ = _make_solver()
    s.set_time_step(0.05)
    s.set_kinematic_viscosity(0.02)
    assert s.dt == 0.05 and s.config.transport.nu == 0.02


def test_registered_fields_build_fringe_source():
    s, mesh = _make_solver()
    n = mesh["n_elements"]
    lam = np.full(n, 3.0)
    ut = np.tile([2.0, -1.0, 0.5], (n, 1))
    s.set_cell_scalar_field("lambdaRelax", lam)
    s.set_cell_vector_field("Utarget", ut[:, 0], ut[:, 1], ut[:, 2])
    su, sp = s._fringe_source()
    assert np.allclose(sp, lam)  # Sp = λ
    assert np.allclose(su, lam[:, None] * ut)  # Su = λ·Utarget


def test_driver_split_solve_then_advance():
    s, _ = _make_solver()
    t0, step0, nc0 = s.flow_time, s.time_step, s._n_committed
    with contextlib.redirect_stdout(io.StringIO()):
        s.set_time_step(0.1)
        s.solve_pimple()
        assert s.flow_time == t0 and s.time_step == step0, "solve_pimple must not advance time"
        s.advance_time()
    assert abs(s.flow_time - (t0 + 0.1)) < 1e-12
    assert s.time_step == step0 + 1 and s._n_committed == nc0 + 1


def test_vector_dirichlet_writes_ghosts():
    s, mesh = _make_solver()
    b = next(b for b in mesh["boundary"] if b["name"] == "xmin")
    nf = b["nFaces"]
    values = np.column_stack(
        [
            np.full(nf, 3.0),
            np.linspace(0.0, 1.0, nf),
            np.linspace(1.0, 2.0, nf),
        ]
    )
    s.set_dirichlet_velocity_boundary_condition_vec(values, "xmin")
    idx = mesh["n_elements"] + (b["startFace"] - mesh["n_interior_faces"])
    assert np.allclose(s.U[idx : idx + nf], values)
    assert b["bc_type_U"] == "fixedValue"


def test_fringe_source_relaxes_velocity_to_target():
    """At the momentum-operator level a strong fringe drives U → Utarget
    (isolated from the continuity projection that suppresses net flow in a
    closed box)."""
    mesh = structured_box(6, 6, 6)
    geo = compute_mesh_geometry(mesh)
    n = mesh["n_elements"]
    nb = mesh["n_faces"] - mesh["n_interior_faces"]
    for b in mesh["boundary"]:
        b["bc_type_U"] = "zeroGradient"
    U = np.zeros((n + nb, 3))
    p = np.zeros(n + nb)
    phi = compute_volumetric_face_flux(U, mesh, geo)
    target = np.tile([2.0, -1.0, 0.5], (n, 1))
    lam = np.full(n, 1.0e4)
    mom = assemble_momentum_equation(
        U,
        p,
        phi,
        1.0,
        0.05,
        mesh,
        geo,
        mesh["boundary"],
        convection_scheme="central",
        dt=0.1,
        U_old=U,
        source_explicit=lam[:, None] * target,
        source_implicit=lam,
    )
    sol = np.zeros((n, 3))
    for i, c in enumerate("xyz"):
        sol[:, i] = solve_linear_system(
            mom[c]["A"], mom[c]["b"], method="spsolve", equation_type="momentum"
        )
    assert np.max(np.abs(sol - target)) < 1e-2, "strong fringe should pull U to Utarget"
