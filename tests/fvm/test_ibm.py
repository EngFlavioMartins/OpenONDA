"""Transfer identities and integrated reference gates for direct-forcing IBM."""

import numpy as np
import pytest

from source.solvers.fvm.immersed_boundary import IBMForcing, ImmersedBody
from source.solvers.fvm.immersed_boundary.forcing import roma_delta_1d
from source.solvers.fvm.mesh import geometry

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
    body = ImmersedBody.cylinder_z(centre=[1.5, 1.5, 0.05], diameter=0.75, grid_spacing=h)
    ibm = IBMForcing(m, geo, [body], grid_spacing=h)
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
        ImmersedBody.from_points([[0.0, 0.0, 0.0]], prescribed_velocity=[np.inf, 0.0, 0.0])
    with pytest.raises(ValueError, match="diameter"):
        ImmersedBody.sphere([0.0, 0.0, 0.0], diameter=0.0, grid_spacing=0.1)
    with pytest.raises(ValueError, match="finite and positive"):
        ImmersedBody.rectangle_z([0.0, 0.0, 0.0], 1.0, 1.0, grid_spacing=-0.1)


def test_rectangle_marker_factory_has_unique_perimeter_points():
    body = ImmersedBody.rectangle_z([1.0, 2.0, 0.05], 1.0, 0.5, grid_spacing=0.1)
    assert len(np.unique(body.position, axis=0)) == body.n_markers
    assert np.allclose(body.position[:, 2], 0.05)
    offsets = np.abs(body.position[:, :2] - [1.0, 2.0])
    assert np.all((np.isclose(offsets[:, 0], 0.5)) | (np.isclose(offsets[:, 1], 0.25)))


def test_immersed_body_exact_solid_geometry():
    cylinder = ImmersedBody.cylinder_z([0.0, 0.0, 0.05], diameter=1.0, grid_spacing=0.1)
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
        grid_spacing=0.1,
        name="foil",
    )
    assert polygon.has_solid_geometry
    assert polygon.n_markers > 0
    side_levels = int(np.ceil(1.0 / 0.1)) + 1
    assert np.count_nonzero(np.isclose(polygon.position[:, 2], 0.0)) == 30
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
        grid_spacing=0.2,
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
    assert body.position[:, 2].min() == pytest.approx(-1.0)
    assert body.position[:, 2].max() == pytest.approx(1.0)
    assert body.contains(query).tolist() == [True, False, False, False]
    assert body.contains(query, include_boundary=True).tolist() == [True, True, False, False]

    through_domain = ImmersedBody.extruded_cylinder_z(
        centre=[0.0, 0.0, 0.0],
        diameter=1.0,
        z_bounds=[-1.0, 1.0],
        grid_spacing=0.2,
        caps=False,
    )
    assert through_domain.contains([[0.0, 0.0, 100.0]])[0]


def test_interpolation_reproduces_constant_and_linear(cylinder_setup):
    m, geo, ibm = cylinder_setup
    n_tot = m["n_cells"] + m["n_faces"] - m["n_interior_faces"]
    cc = geo["cell_centre"]

    const = np.full((n_tot, 3), 2.5)
    err_const = np.abs(ibm.interpolate(const) - 2.5).max()
    assert err_const < 1e-12  # exact by row normalisation

    lin = np.zeros((n_tot, 3))
    lin[: m["n_cells"], 0] = 1.0 + 0.7 * cc[:, 0] - 0.3 * cc[:, 1]
    expect = 1.0 + 0.7 * ibm.marker_position[:, 0] - 0.3 * ibm.marker_position[:, 1]
    err_lin = np.abs(ibm.interpolate(lin)[:, 0] - expect).max()
    assert err_lin < 5e-3  # O(h^2) with h = 1/16 of D


def test_pinelli_quadrature_consistency(cylinder_setup):
    _, _, ibm = cylinder_setup
    # A eps = 1 solved exactly.
    diagnostics = ibm.diagnostics()
    assert diagnostics["quadrature_residual"] < 1e-10
    assert diagnostics["min_quadrature_weight"] >= 0.0
    assert diagnostics["max_quadrature_weight"] > 0.0
    # Round trip: interpolate(spread(F)) ~ F for constant F.
    F = np.tile([1.0, -2.0, 0.5], (ibm.marker_position.shape[0], 1))
    round_trip = ibm.interpolate(ibm.spread(F))
    assert np.abs(round_trip - F).max() < 1e-8


def test_forcing_kills_slip_in_one_application(cylinder_setup):
    m, geo, ibm = cylinder_setup
    n_tot = m["n_cells"] + m["n_faces"] - m["n_interior_faces"]
    # Uniform flow through the fixed body: slip initially equals freestream speed.
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
    ibm.compute_force(velocity, time_step_size)  # initialise last_marker_acceleration
    F_before = ibm.last_marker_acceleration.copy()
    ibm.multidirect_correct(velocity, time_step_size, n_iterations=3)
    # Slip driven far below the single-application level (test above: < 0.05).
    assert ibm.last_slip < 5e-3
    # Increments were accumulated into the logged Lagrangian force.
    assert not np.allclose(ibm.last_marker_acceleration, F_before)


def test_body_force_matches_eulerian_integral(cylinder_setup):
    m, geo, ibm = cylinder_setup
    n_tot = m["n_cells"] + m["n_faces"] - m["n_interior_faces"]
    velocity = np.zeros((n_tot, 3))
    velocity[:, 0] = 1.0
    f = ibm.compute_force(velocity, time_step_size=0.02)
    vol = geo["cell_volume"]
    F_euler = -np.sum(f * vol[:, np.newaxis], axis=0)
    F_body = ibm.body_forces(density=1.0)["cylinder"]
    assert np.allclose(F_body, F_euler, rtol=1e-10, atol=1e-12)
    # The forcing decelerates the fluid, so the reaction on the body is a
    # positive-x drag.
    assert F_body[0] > 0.0


def test_body_force_removes_fictitious_solid_fluid_momentum():
    mesh, geo = _mesh_2d(nx=20, ny=16, lx=5.0, ly=4.0)
    body = ImmersedBody.rectangle_z([2.0, 2.0, 0.05], 1.0, 1.0, grid_spacing=0.25, name="square")
    ibm = IBMForcing(mesh, geo, body, grid_spacing=0.25)
    n_total = mesh["n_cells"] + mesh["n_faces"] - mesh["n_interior_faces"]
    velocity_old = np.zeros((n_total, 3))
    velocity = np.zeros_like(velocity_old)
    velocity_old[:, 0] = 1.0
    mask = body.contains(geo["cell_centre"])
    expected_rate = -np.sum(geo["cell_volume"][mask]) / 0.5

    ibm.update_fictitious_fluid_momentum_rate(velocity, velocity_old, 0.5)

    force = ibm.body_forces(density=1.0)["square"]
    np.testing.assert_allclose(force, [expected_rate, 0.0, 0.0], atol=1e-14)


def test_cylinder_step_integration():
    """End-to-end: a few PIMPLE steps with IBM produce finite fields, positive
    drag, and a small no-slip error at the markers."""
    from source.solvers.fvm import (
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
        # n_outer_correctors=1 does not converge the pressure-velocity coupling
        # for this impulsively-started bluff-body/IBM case: slip error and drag
        # sign are still evolving step-to-step at 1 outer corrector (CFL grows
        # past 1.4 by step 10 and Fx is negative). 8 outer correctors converges
        # both (slip 0.090->0.002, Fx -2.55->+0.24); see
        # docs/PROJECT_COMPLETION_TODO.md §E for the corrector sweep evidence.
        # This is a numerical-convergence fix, not a calibration of IBM
        # physics/coefficients against the expected result.
        pimple=PimpleControl(n_correctors=2, n_outer_correctors=8),
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
        body = ImmersedBody.cylinder_z(centre=[2.0, 2.0, 0.05], diameter=1.0, grid_spacing=h)
        ibm = solver.set_immersed_bodies(body, grid_spacing=h)

        for _ in range(10):
            solver.advance()

        n = m["n_cells"]
        assert np.all(np.isfinite(solver.velocity[:n]))
        # No-slip enforced at markers to a small fraction of freestream speed.
        assert ibm.slip_error(solver.velocity) < 0.05
        # Drag is positive (force on body along +x).
        Fb = ibm.body_forces(density=1.0)["cylinder"]
        assert Fb[0] > 0.0
        # Wake deficit exists: velocity behind the body is below freestream speed.
        cc = solver.geo_data["cell_centre"][:n]
        wake = (np.abs(cc[:, 1] - 2.0) < 0.2) & (cc[:, 0] > 2.6) & (cc[:, 0] < 3.5)
        assert solver.velocity[:n][wake, 0].mean() < 0.8


def test_solver_rejects_unqualified_moving_body_support(tmp_path):
    from source.solvers.fvm import (
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
    moving = ImmersedBody.from_points(
        [[1.5, 1.5, 0.05]], prescribed_velocity=[0.1, 0.0, 0.0], name="moving"
    )
    with pytest.raises(NotImplementedError, match="energy accounting"):
        solver.set_immersed_bodies(moving, grid_spacing=0.15)


@pytest.mark.slow
@pytest.mark.parametrize("h", (0.25, 0.125, 0.0625))
def test_ibm_square_force_and_wake_match_body_fitted_reference(tmp_path, h):
    from source.solvers.fvm import (
        BoundaryConfig,
        DiscretizationConfig,
        FVMSetup,
        FVMSolver,
        LinearSolverConfig,
        PimpleControl,
        TimeConfig,
        TransportConfig,
    )
    from source.solvers.fvm.mesh.rectilinear import coupling_box_mesh

    domain = (0.0, 6.0, 0.0, 4.0, 0.0, h)
    time_step_size = min(0.02, 0.16 * h)

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
        # See the n_outer_correctors note in test_cylinder_step_integration:
        # the same non-convergence pattern makes the body-fitted reference's
        # own transient unreliable at 1 outer corrector (up to ~50x too-large,
        # sign-flipping forces in the first ~20 steps); 8 converges it.
        pimple = PimpleControl(n_correctors=2, n_outer_correctors=8)
        samplers = []
        if with_square:
            from source.solvers.fvm.sampling.base import SamplingSchedule
            from source.solvers.fvm.sampling.forces import ForceSampler

            samplers = [
                ForceSampler(
                    patch_names=["square"],
                    reference_velocity=1.0,
                    reference_area=h,
                    schedule=SamplingSchedule(every_n_steps=1),
                )
            ]
        return FVMSetup(
            case_name="body-fitted" if with_square else "ibm",
            time=TimeConfig.transient(
                time_step_size=time_step_size, duration=0.16, output_interval_steps=1000
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
    body = ImmersedBody.rectangle_z([2.0, 2.0, 0.5 * h], 1.0, 1.0, grid_spacing=h, name="square")
    ibm = immersed.set_immersed_bodies(body, grid_spacing=h)
    for _ in range(round(0.16 / time_step_size)):
        fitted.advance()
        immersed.advance()

    fitted_drag = fitted.last_forces["square"]["total_force"][0]
    immersed_drag = ibm.body_forces(density=1.0)["square"][0]
    uncorrected_drag = ibm.forcing_reaction_forces(density=1.0)["square"][0]

    def wake_velocity(solver):
        centres = solver.geo_data["cell_centre"]
        wake = (np.abs(centres[:, 1] - 2.0) < 0.21) & (centres[:, 0] > 2.7) & (centres[:, 0] < 3.5)
        return float(np.mean(solver.velocity[: solver.mesh_data["n_cells"]][wake, 0]))

    fitted_wake = wake_velocity(fitted)
    immersed_wake = wake_velocity(immersed)
    assert fitted_drag > 0.0 and immersed_drag > 0.0
    assert fitted_wake < 1.0 and immersed_wake < 1.0
    force_error = abs(immersed_drag - fitted_drag) / fitted_drag
    uncorrected_force_error = abs(uncorrected_drag - fitted_drag) / fitted_drag
    wake_error = abs(immersed_wake - fitted_wake)
    limits = {
        0.25: {"force": 0.70, "wake": 0.12, "slip": 0.23},
        0.125: {"force": 0.50, "wake": 0.07, "slip": 0.21},
        0.0625: {"force": 0.50, "wake": 0.04, "slip": 0.20},
    }[h]
    # Instantaneous force at t=0.16 is still inside the impulsive-start added-
    # mass transient, so pointwise grid convergence is not a valid steady-drag
    # study.  The first-principles requirement is that removing the interior
    # fictitious-fluid momentum moves the IBM load toward the surface-traction
    # reference at every resolution; wake and marker slip remain independently
    # tightened under refinement.
    assert force_error < uncorrected_force_error
    assert force_error < limits["force"]
    assert wake_error < limits["wake"]
    assert ibm.last_slip < limits["slip"]
