"""Transfer identities and integrated reference gates for direct-forcing IBM."""

import numpy as np
import pytest

from source.solvers.FVM.immersed_boundary import IBMForcing, ImmersedBody
from source.solvers.FVM.immersed_boundary.forcing import roma_delta_1d
from source.solvers.FVM.mesh import geometry

from ._structured_mesh import structured_box


def _mesh_2d(nx=48, ny=48, lx=3.0, ly=3.0, lz=0.1):
    m = structured_box(nx, ny, 1, lx, ly, lz)
    for b in m["boundary"]:
        if b["name"] in ("zmin", "zmax"):
            b["type"] = "empty"
            b["velocity_type"] = "empty"
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


def test_immersed_body_rejects_invalid_marker_state():
    with pytest.raises(ValueError, match="non-empty"):
        ImmersedBody.from_points([[0.0, 0.0, 0.0]], name="")
    with pytest.raises(ValueError, match="markers must be finite"):
        ImmersedBody.from_points([[np.nan, 0.0, 0.0]])
    with pytest.raises(ValueError, match="target velocity must be finite"):
        ImmersedBody.from_points([[0.0, 0.0, 0.0]], U_target=[np.inf, 0.0, 0.0])
    with pytest.raises(ValueError, match="diameter"):
        ImmersedBody.sphere([0.0, 0.0, 0.0], diameter=0.0, h=0.1)
    with pytest.raises(ValueError, match="finite and positive"):
        ImmersedBody.rectangle_z([0.0, 0.0, 0.0], 1.0, 1.0, h=-0.1)


def test_rectangle_marker_factory_has_unique_perimeter_points():
    body = ImmersedBody.rectangle_z([1.0, 2.0, 0.05], 1.0, 0.5, h=0.1)
    assert len(np.unique(body.X, axis=0)) == body.n_markers
    assert np.allclose(body.X[:, 2], 0.05)
    offsets = np.abs(body.X[:, :2] - [1.0, 2.0])
    assert np.all((np.isclose(offsets[:, 0], 0.5)) | (np.isclose(offsets[:, 1], 0.25)))


def test_immersed_body_exact_solid_geometry():
    cylinder = ImmersedBody.cylinder_z([0.0, 0.0, 0.05], diameter=1.0, h=0.1)
    query = np.array(
        [
            [0.0, 0.0, 0.05],
            [0.49, 0.0, 0.05],
            [0.5, 0.0, 0.05],
            [0.51, 0.0, 0.05],
        ]
    )
    np.testing.assert_array_equal(cylinder.contains(query), [True, True, False, False])
    np.testing.assert_array_equal(
        cylinder.contains(query, include_boundary=True), [True, True, True, False]
    )

    polygon = ImmersedBody.extruded_polygon_z(
        [[-0.5, -0.25], [0.5, -0.25], [0.5, 0.25], [-0.5, 0.25]],
        [-0.5, 0.5],
        h=0.1,
        name="foil",
    )
    assert polygon.has_solid_geometry
    assert polygon.n_markers > 0
    side_levels = int(np.ceil(1.0 / 0.1)) + 1
    assert np.count_nonzero(np.isclose(polygon.X[:, 2], 0.0)) == 30
    assert polygon.n_markers >= 30 * side_levels
    np.testing.assert_array_equal(
        polygon.contains(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.5],
                [0.0, 0.3, 0.0],
            ]
        ),
        [True, False, False],
    )
    assert polygon.contains([[0.0, 0.0, 0.5]], include_boundary=True)[0]


def test_extruded_cylinder_factory_has_exact_solid_geometry():
    body = ImmersedBody.extruded_cylinder_z(
        centre=[0.0, 0.0, 0.0],
        diameter=1.0,
        z_bounds=[-1.0, 1.0],
        h=0.2,
        caps=True,
    )
    query = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.51, 0.0, 0.0],
            [0.0, 0.0, 1.01],
        ]
    )
    assert body.X[:, 2].min() == pytest.approx(-1.0)
    assert body.X[:, 2].max() == pytest.approx(1.0)
    assert body.contains(query).tolist() == [True, False, False, False]
    assert body.contains(query, include_boundary=True).tolist() == [True, True, False, False]

    through_domain = ImmersedBody.extruded_cylinder_z(
        centre=[0.0, 0.0, 0.0],
        diameter=1.0,
        z_bounds=[-1.0, 1.0],
        h=0.2,
        caps=False,
    )
    assert through_domain.contains([[0.0, 0.0, 100.0]])[0]


def test_interpolation_reproduces_constant_and_linear(cylinder_setup):
    m, geo, ibm = cylinder_setup
    n_tot = m["n_cells"] + m["n_faces"] - m["n_interior_faces"]
    cc = geo["cell_centroids"]

    const = np.full((n_tot, 3), 2.5)
    err_const = np.abs(ibm.interpolate(const) - 2.5).max()
    assert err_const < 1e-12  # exact by row normalisation

    lin = np.zeros((n_tot, 3))
    lin[: m["n_cells"], 0] = 1.0 + 0.7 * cc[:, 0] - 0.3 * cc[:, 1]
    expect = 1.0 + 0.7 * ibm.X[:, 0] - 0.3 * ibm.X[:, 1]
    err_lin = np.abs(ibm.interpolate(lin)[:, 0] - expect).max()
    assert err_lin < 5e-3  # O(h^2) with h = 1/16 of D


def test_pinelli_quadrature_consistency(cylinder_setup):
    _, _, ibm = cylinder_setup
    # A eps = 1 solved exactly.
    diagnostics = ibm.diagnostics()
    assert diagnostics["quadrature_residual"] < 1e-10
    assert diagnostics["eps_min"] >= 0.0
    assert diagnostics["eps_max"] > 0.0
    # Round trip: interpolate(spread(F)) ~ F for constant F.
    F = np.tile([1.0, -2.0, 0.5], (ibm.X.shape[0], 1))
    round_trip = ibm.interpolate(ibm.spread(F))
    assert np.abs(round_trip - F).max() < 1e-8


def test_forcing_kills_slip_in_one_application(cylinder_setup):
    m, geo, ibm = cylinder_setup
    n_tot = m["n_cells"] + m["n_faces"] - m["n_interior_faces"]
    # Uniform flow through the (fixed) body: slip = |U_inf| initially.
    velocity = np.zeros((n_tot, 3))
    velocity[:, 0] = 1.0
    time_step_size = 0.01
    slip0 = ibm.slip_error(velocity)
    assert slip0 == pytest.approx(1.0, rel=1e-6)

    # Explicit model update u += dt * f (what the momentum solve does to
    # leading order at the forced cells).
    f = ibm.compute_force(velocity, time_step_size)
    velocity[: m["n_cells"]] += time_step_size * f
    slip1 = ibm.slip_error(velocity)
    assert slip1 < 0.05 * slip0  # >20x reduction


def test_multidirect_forcing_converges_slip(cylinder_setup):
    m, geo, ibm = cylinder_setup
    n_tot = m["n_cells"] + m["n_faces"] - m["n_interior_faces"]
    velocity = np.zeros((n_tot, 3))
    velocity[:, 0] = 1.0
    time_step_size = 0.01
    ibm.compute_force(velocity, time_step_size)  # initialise last_F
    F_before = ibm.last_F.copy()
    ibm.multidirect_correct(velocity, time_step_size, n_iter=3)
    # Slip driven far below the single-application level (test above: < 0.05).
    assert ibm.last_slip < 5e-3
    # Increments were accumulated into the logged Lagrangian force.
    assert not np.allclose(ibm.last_F, F_before)


def test_body_force_matches_eulerian_integral(cylinder_setup):
    m, geo, ibm = cylinder_setup
    n_tot = m["n_cells"] + m["n_faces"] - m["n_interior_faces"]
    velocity = np.zeros((n_tot, 3))
    velocity[:, 0] = 1.0
    f = ibm.compute_force(velocity, time_step_size=0.02)
    vol = geo["cell_volumes"]
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
        DiscretizationConfig,
        FVMSetup,
        FVMSolver,
        LinearSolverConfig,
        PimpleControl,
        TimeConfig,
        TransportConfig,
    )

    m, _ = _mesh_2d(nx=60, ny=40, lx=6.0, ly=4.0)
    config = FVMSetup(
        case_name="ibm_smoke",
        time=TimeConfig(
            time_step_size=0.02, start_time=0.0, end_time=1.0, output_interval_steps=1000
        ),
        schemes=DiscretizationConfig(convection_scheme="upwind", gradient_scheme="gauss"),
        linear=LinearSolverConfig(linear_solver="spsolve"),
        pimple=PimpleControl(n_correctors=2, n_outer_correctors=1),
        transport=TransportConfig(density=1.0, kinematic_viscosity=0.025),  # Re = 40 on D=1
        boundaries=[
            BoundaryConfig.inlet("xmin", [1.0, 0.0, 0.0]),
            BoundaryConfig.outlet("xmax", kinematic_pressure=0.0),
            BoundaryConfig.freestream("ymin", [1.0, 0.0, 0.0]),
            BoundaryConfig.freestream("ymax", [1.0, 0.0, 0.0]),
            BoundaryConfig.empty("zmin"),
            BoundaryConfig.empty("zmax"),
        ],
        initial_velocity=[1.0, 0.0, 0.0],
        initial_kinematic_pressure=0.0,
    )
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        solver = FVMSolver(config, case_dir=tmp, mesh_data=m)
        solver.auto_write = False
        h = 6.0 / 60
        body = ImmersedBody.cylinder_z(centre=[2.0, 2.0, 0.05], diameter=1.0, h=h)
        ibm = solver.set_immersed_bodies(body, h=h)

        for _ in range(10):
            solver.advance()

        n = m["n_cells"]
        assert np.all(np.isfinite(solver.velocity[:n]))
        # No-slip enforced at markers to a small fraction of U_inf.
        assert ibm.slip_error(solver.velocity) < 0.05
        # Drag is positive (force on body along +x).
        Fb = ibm.body_forces(rho=1.0)["cylinder"]
        assert Fb[0] > 0.0
        # Wake deficit exists: velocity behind the body below U_inf.
        cc = solver.geo_data["cell_centroids"][:n]
        wake = (np.abs(cc[:, 1] - 2.0) < 0.2) & (cc[:, 0] > 2.6) & (cc[:, 0] < 3.5)
        assert solver.velocity[:n][wake, 0].mean() < 0.8


def test_solver_rejects_unqualified_moving_body_support(tmp_path):
    from source.solvers.FVM import (
        BoundaryConfig,
        DiscretizationConfig,
        FVMSetup,
        FVMSolver,
        LinearSolverConfig,
        PimpleControl,
        TimeConfig,
        TransportConfig,
    )

    mesh, _ = _mesh_2d(nx=20, ny=20)
    config = FVMSetup(
        case_name="moving_ibm_rejected",
        time=TimeConfig.transient(time_step_size=0.01, duration=0.01),
        schemes=DiscretizationConfig(),
        linear=LinearSolverConfig(linear_solver="spsolve"),
        pimple=PimpleControl(),
        transport=TransportConfig(kinematic_viscosity=0.01),
        boundaries=[
            BoundaryConfig.inlet("xmin", [1.0, 0.0, 0.0]),
            BoundaryConfig.outlet("xmax"),
            BoundaryConfig.wall("ymin"),
            BoundaryConfig.wall("ymax"),
            BoundaryConfig.empty("zmin"),
            BoundaryConfig.empty("zmax"),
        ],
    )
    solver = FVMSolver(config, case_dir=str(tmp_path), mesh_data=mesh)
    moving = ImmersedBody.from_points([[1.5, 1.5, 0.05]], U_target=[0.1, 0.0, 0.0], name="moving")
    with pytest.raises(NotImplementedError, match="energy accounting"):
        solver.set_immersed_bodies(moving, h=0.15)


@pytest.mark.slow
@pytest.mark.parametrize("h", (0.25, 0.125))
def test_ibm_square_force_and_wake_match_body_fitted_reference(tmp_path, h):
    from source.solvers.FVM import (
        BoundaryConfig,
        DiscretizationConfig,
        FVMSetup,
        FVMSolver,
        LinearSolverConfig,
        PimpleControl,
        TimeConfig,
        TransportConfig,
    )
    from source.solvers.FVM.mesh.rectilinear import coupling_box_mesh

    domain = (0.0, 6.0, 0.0, 4.0, 0.0, h)

    def boundaries(with_square):
        values = [
            BoundaryConfig.inlet("xmin", [1.0, 0.0, 0.0]),
            BoundaryConfig.outlet("xmax", 0.0),
            BoundaryConfig.freestream("ymin", [1.0, 0.0, 0.0]),
            BoundaryConfig.freestream("ymax", [1.0, 0.0, 0.0]),
            BoundaryConfig.empty("zmin"),
            BoundaryConfig.empty("zmax"),
        ]
        if with_square:
            values.append(BoundaryConfig.wall("square"))
        return values

    def config(with_square):
        schemes = DiscretizationConfig(convection_scheme="upwind", gradient_scheme="gauss")
        linear = LinearSolverConfig(linear_solver="spsolve")
        pimple = PimpleControl(n_correctors=2, n_outer_correctors=1)
        samplers = []
        if with_square:
            from source.solvers.FVM.sampling.base import SamplingSchedule
            from source.solvers.FVM.sampling.forces import ForceSampler

            samplers = [
                ForceSampler(
                    patch_names=["square"],
                    ref_velocity=1.0,
                    ref_area=h,
                    schedule=SamplingSchedule(every_n_steps=1),
                )
            ]
        return FVMSetup(
            case_name="body-fitted" if with_square else "ibm",
            time=TimeConfig.transient(
                time_step_size=0.02, duration=0.16, output_interval_steps=1000
            ),
            schemes=schemes,
            linear=linear,
            pimple=pimple,
            samplers=samplers,
            transport=TransportConfig(density=1.0, kinematic_viscosity=0.05),
            boundaries=boundaries(with_square),
            initial_velocity=[1.0, 0.0, 0.0],
            initial_kinematic_pressure=0.0,
        )

    fitted_mesh = coupling_box_mesh(
        domain,
        h,
        hole_box=(1.5, 2.5, 1.5, 2.5, 0.0, h),
        wall_patch_name="square",
    )
    nx = round(6.0 / h)
    ny = round(4.0 / h)
    body_cells = round(1.0 / h)
    z_faces = nx * ny - body_cells**2
    counts = (ny, ny, nx, nx, z_faces, z_faces)
    start = fitted_mesh["n_interior_faces"]
    patches = []
    for name, count in zip(("xmin", "xmax", "ymin", "ymax", "zmin", "zmax"), counts, strict=True):
        patches.append({"name": name, "start_face": start, "n_faces": count, "type": "patch"})
        start += count
    patches.append({**fitted_mesh["boundary"][-1], "start_face": start})
    fitted_mesh["boundary"] = patches
    ibm_mesh = structured_box(nx, ny, 1, 6.0, 4.0, h)

    fitted = FVMSolver(config(True), str(tmp_path / "fitted"), mesh_data=fitted_mesh)
    immersed = FVMSolver(config(False), str(tmp_path / "immersed"), mesh_data=ibm_mesh)
    fitted.auto_write = False
    immersed.auto_write = False
    body = ImmersedBody.rectangle_z([2.0, 2.0, 0.5 * h], 1.0, 1.0, h=h, name="square")
    ibm = immersed.set_immersed_bodies(body, h=h)
    for _ in range(8):
        fitted.advance()
        immersed.advance()

    fitted_drag = fitted.last_forces["square"]["Ftot"][0]
    immersed_drag = ibm.body_forces(rho=1.0)["square"][0]

    def wake_velocity(solver):
        centres = solver.geo_data["cell_centroids"]
        wake = (np.abs(centres[:, 1] - 2.0) < 0.21) & (centres[:, 0] > 2.7) & (centres[:, 0] < 3.5)
        return float(np.mean(solver.velocity[: solver.mesh_data["n_cells"]][wake, 0]))

    fitted_wake = wake_velocity(fitted)
    immersed_wake = wake_velocity(immersed)
    assert fitted_drag > 0.0 and immersed_drag > 0.0
    assert fitted_wake < 1.0 and immersed_wake < 1.0
    force_error = abs(immersed_drag - fitted_drag) / fitted_drag
    wake_error = abs(immersed_wake - fitted_wake)
    limits = {
        0.25: {"force": 0.70, "wake": 0.12, "slip": 0.23},
        0.125: {"force": 0.03, "wake": 0.07, "slip": 0.21},
    }[h]
    assert force_error < limits["force"]
    assert wake_error < limits["wake"]
    assert ibm.last_slip < limits["slip"]
