"""Tests for the discrete direct-forcing IBM (immersed_boundary/).

Covers the transfer-operator properties the method relies on (Pinelli et al.
2010; Constant et al., docs/literature/Constant2016.pdf):

1. the Roma kernel satisfies discrete partition of unity;
2. interpolation reproduces constant and linear fields;
3. the Pinelli quadrature makes spread∘interpolate reproduce constants
   (``A ε = 1``) and smooth fields to O(h²);
4. one forcing application drives the marker slip to ~0 in a model problem;
5. an end-to-end coarse cylinder step produces a finite, positive drag and
   reduces the no-slip error vs. the unforced predictor.
"""

import numpy as np
import pytest

from tests.fvm._structured_mesh import structured_box

from source.solvers.FVM.immersed_boundary import ImmersedBody, IBMForcing
from source.solvers.FVM.immersed_boundary.forcing import roma_delta_1d
from source.solvers.FVM.mesh import geometry


def _mesh_2d(nx=48, ny=48, lx=3.0, ly=3.0, lz=0.1):
    m = structured_box(nx, ny, 1, lx, ly, lz)
    for b in m["boundary"]:
        if b["name"] in ("zmin", "zmax"):
            b["type"] = "empty"
            b["bc_type_U"] = "empty"
    geo = geometry.compute_mesh_geometry(m, gradient_scheme="gauss")
    return m, geo


@pytest.fixture(scope="module")
def cylinder_setup():
    m, geo = _mesh_2d()
    h = 3.0 / 48
    body = ImmersedBody.cylinder_z(centre=[1.5, 1.5, 0.05], diameter=0.75, h=h)
    ibm = IBMForcing(m, geo, [body], h=h)
    return m, geo, ibm


def test_roma_kernel_partition_of_unity():
    # Sum of phi over the integer-shifted grid equals 1 for any offset.
    for offset in np.linspace(-0.5, 0.5, 11):
        pts = offset + np.arange(-3, 4)
        assert np.sum(roma_delta_1d(pts)) == pytest.approx(1.0, abs=1e-12)


def test_interpolation_reproduces_constant_and_linear(cylinder_setup):
    m, geo, ibm = cylinder_setup
    n_tot = m["n_elements"] + m["n_faces"] - m["n_interior_faces"]
    cc = geo["element_centroids"]

    const = np.full((n_tot, 3), 2.5)
    err_const = np.abs(ibm.interpolate(const) - 2.5).max()
    assert err_const < 1e-12  # exact by row normalisation

    lin = np.zeros((n_tot, 3))
    lin[: m["n_elements"], 0] = 1.0 + 0.7 * cc[:, 0] - 0.3 * cc[:, 1]
    expect = 1.0 + 0.7 * ibm.X[:, 0] - 0.3 * ibm.X[:, 1]
    err_lin = np.abs(ibm.interpolate(lin)[:, 0] - expect).max()
    assert err_lin < 5e-3  # O(h^2) with h = 1/16 of D


def test_pinelli_quadrature_consistency(cylinder_setup):
    _, _, ibm = cylinder_setup
    # A eps = 1 solved exactly.
    assert ibm.diagnostics()["quadrature_residual"] < 1e-10
    # Round trip: interpolate(spread(F)) ~ F for constant F.
    F = np.tile([1.0, -2.0, 0.5], (ibm.X.shape[0], 1))
    round_trip = ibm.interpolate(ibm.spread(F))
    assert np.abs(round_trip - F).max() < 1e-8


def test_forcing_kills_slip_in_one_application(cylinder_setup):
    m, geo, ibm = cylinder_setup
    n_tot = m["n_elements"] + m["n_faces"] - m["n_interior_faces"]
    # Uniform flow through the (fixed) body: slip = |U_inf| initially.
    U = np.zeros((n_tot, 3))
    U[:, 0] = 1.0
    dt = 0.01
    slip0 = ibm.slip_error(U)
    assert slip0 == pytest.approx(1.0, rel=1e-6)

    # Explicit model update u += dt * f (what the momentum solve does to
    # leading order at the forced cells).
    f = ibm.compute_force(U, dt)
    U[: m["n_elements"]] += dt * f
    slip1 = ibm.slip_error(U)
    assert slip1 < 0.05 * slip0  # >20x reduction


def test_multidirect_forcing_converges_slip(cylinder_setup):
    m, geo, ibm = cylinder_setup
    n_tot = m["n_elements"] + m["n_faces"] - m["n_interior_faces"]
    U = np.zeros((n_tot, 3))
    U[:, 0] = 1.0
    dt = 0.01
    ibm.compute_force(U, dt)  # initialise last_F
    F_before = ibm.last_F.copy()
    ibm.multidirect_correct(U, dt, n_iter=3)
    # Slip driven far below the single-application level (test above: < 0.05).
    assert ibm.last_slip < 5e-3
    # Increments were accumulated into the logged Lagrangian force.
    assert not np.allclose(ibm.last_F, F_before)


def test_body_force_matches_eulerian_integral(cylinder_setup):
    m, geo, ibm = cylinder_setup
    n_tot = m["n_elements"] + m["n_faces"] - m["n_interior_faces"]
    U = np.zeros((n_tot, 3))
    U[:, 0] = 1.0
    f = ibm.compute_force(U, dt=0.02)
    vol = geo["element_volumes"]
    F_euler = -np.sum(f * vol[:, np.newaxis], axis=0)
    F_body = ibm.body_forces(rho=1.0)["cylinder"]
    assert np.allclose(F_body, F_euler, rtol=1e-10, atol=1e-12)
    # The forcing decelerates the fluid, so the reaction on the body is a
    # positive-x drag.
    assert F_body[0] > 0.0


def test_cylinder_step_integration():
    """End-to-end: a few PIMPLE steps with IBM produce finite fields, positive
    drag, and a small no-slip error at the markers."""
    from source.solvers.FVM import (
        BoundaryConfig,
        FVMConfig,
        Solver,
        SolverParams,
        TimeConfig,
        TransportConfig,
    )

    m, _ = _mesh_2d(nx=60, ny=40, lx=6.0, ly=4.0)
    config = FVMConfig(
        case_name="ibm_smoke",
        time=TimeConfig(delta_t=0.02, start_time=0.0, end_time=1.0, write_interval=1000),
        solver=SolverParams.pimple(
            n_correctors=2, n_outer=1, linear_solver="spsolve",
            convection_scheme="upwind", gradient_scheme="gauss",
        ),
        transport=TransportConfig(density=1.0, nu=0.025),  # Re = 40 on D=1
        boundaries=[
            BoundaryConfig.inlet("xmin", [1.0, 0.0, 0.0]),
            BoundaryConfig.outlet("xmax", p=0.0),
            BoundaryConfig.freestream("ymin", [1.0, 0.0, 0.0]),
            BoundaryConfig.freestream("ymax", [1.0, 0.0, 0.0]),
            BoundaryConfig.empty("zmin"),
            BoundaryConfig.empty("zmax"),
        ],
        initial_U=[1.0, 0.0, 0.0],
        initial_p=0.0,
    )
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        solver = Solver(config, case_dir=tmp, mesh_data=m)
        solver.auto_write = False
        h = 6.0 / 60
        body = ImmersedBody.cylinder_z(centre=[2.0, 2.0, 0.05], diameter=1.0, h=h)
        ibm = solver.set_immersed_bodies(body, h=h)

        for _ in range(10):
            solver.evolve()

        n = m["n_elements"]
        assert np.all(np.isfinite(solver.U[:n]))
        # No-slip enforced at markers to a small fraction of U_inf.
        assert ibm.slip_error(solver.U) < 0.05
        # Drag is positive (force on body along +x).
        Fb = ibm.body_forces(rho=1.0)["cylinder"]
        assert Fb[0] > 0.0
        # Wake deficit exists: velocity behind the body below U_inf.
        cc = solver.geo_data["element_centroids"][:n]
        wake = (np.abs(cc[:, 1] - 2.0) < 0.2) & (cc[:, 0] > 2.6) & (cc[:, 0] < 3.5)
        assert solver.U[:n][wake, 0].mean() < 0.8
