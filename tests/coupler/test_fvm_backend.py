"""Coupling contract tests for the native FVM backend."""

from __future__ import annotations

import contextlib
import io

import numpy as np
import pytest

from source.coupler.boundary import apply_fvm_boundary
from source.coupler.config.types import CouplerSetup
from source.solvers.FVM import (
    BoundaryConfig,
    ComputeConfig,
    DiscretizationConfig,
    FVMSetup,
    FVMSolver,
    LinearSolverConfig,
    PimpleControl,
    TimeConfig,
    TransportConfig,
)
from source.solvers.FVM.mesh.rectilinear import coupling_box_mesh
from source.solvers.FVM.sampling.base import SamplingSchedule
from source.solvers.FVM.sampling.forces import ForceSampler

CONTRACT_METHODS = [
    "get_cell_centre_coordinates",
    "get_cell_volumes",
    "get_velocity_field",
    "get_velocity_field_into",
    "get_velocity_gradient_field",
    "get_velocity_gradient_field_into",
    "get_vorticity_field",
    "get_vorticity_field_into",
    "get_boundary_face_centre_coordinates",
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
    "set_normal_velocity_tangential_gradient_boundary_condition",
    "set_freestream_pressure_boundary_condition",
    "set_directional_freestream_pressure_boundary_condition",
    "set_flux_consistent_pressure_boundary_condition",
    "solve_pimple",
    "advance_time",
]

BOX = (-0.5, 0.5, -0.5, 0.5, -0.5, 0.5)


def _fvm_setup(tmp_path, **overrides):
    """Coupling-only setup: physics/time/mesh live on the injected solvers."""
    kwargs = {
        "freestream_velocity": [1.0, 0.0, 0.0],
        "vpm_particle_spacing": 0.25,
        "authority_ramp_width": 0.25,
        "vpm_only_width": 0.125,
    }
    kwargs.update(overrides)
    return CouplerSetup(**kwargs)


class _FakeStepLogger:
    def step_begin(self, step, time, time_step_size):
        pass

    def courant_info(self, maximum, target=None):
        pass

    def step_end(self, elapsed):
        pass


class _FakeFVMTimeConfig:
    adjust_time_step = False
    max_cfl = None


class _FakeFVMConfig:
    time = _FakeFVMTimeConfig()


def test_coupler_setup_validates_and_serializes_vpm_bc_mode(tmp_path):
    setup = _fvm_setup(tmp_path, boundary_condition_mode="characteristic")
    assert setup.to_dict()["coupler"]["boundary_condition_mode"] == "characteristic"
    directional = _fvm_setup(tmp_path, boundary_condition_mode="directional_outflow")
    assert directional.to_dict()["coupler"]["boundary_condition_mode"] == "directional_outflow"
    pressure = _fvm_setup(tmp_path, boundary_condition_mode="pressure_gradient")
    assert pressure.to_dict()["coupler"]["boundary_condition_mode"] == "pressure_gradient"
    mixed = _fvm_setup(tmp_path, boundary_condition_mode="vorticity_mixed")
    assert mixed.to_dict()["coupler"]["boundary_condition_mode"] == "vorticity_mixed"
    with pytest.raises(ValueError, match="boundary_condition_mode"):
        _fvm_setup(tmp_path, boundary_condition_mode="unsupported")


def test_coupler_setup_validates_separate_transfer_region_box(tmp_path):
    outer = (-1.5, 1.5, -1.5, 1.5, -1.5, 1.5)
    inner = (-1.2, 1.2, -1.2, 1.2, -1.2, 1.2)
    setup = _fvm_setup(tmp_path, transfer_region_bounds=inner)
    assert setup.to_dict()["coupler"]["transfer_region_bounds"]["xmax"] == 1.2
    setup.validate_transfer_region_box(outer)
    invalid = _fvm_setup(tmp_path, transfer_region_bounds=(-1.6, 1.2, *inner[2:]))
    with pytest.raises(ValueError, match="contained within the FVM domain"):
        invalid.validate_transfer_region_box(outer)


def _build_backend(tmp_path, spacing=0.25, box=BOX, hole_box=None, wall_patch_name=None):
    mesh = coupling_box_mesh(
        box, spacing, hole_box=hole_box, wall_patch_name=wall_patch_name or "cube"
    )
    boundaries = [
        BoundaryConfig(
            name="numericalBoundary",
            velocity_type="fixedValue",
            velocity_value=[1.0, 0.0, 0.0],
            pressure_type="fixedFluxPressure",
        )
    ]
    samplers = []
    if wall_patch_name is not None:
        boundaries.append(BoundaryConfig.wall(wall_patch_name))
        body = np.asarray(hole_box, dtype=float)
        samplers.append(
            ForceSampler(
                patch_names=[wall_patch_name],
                ref_velocity=1.0,
                ref_area=float((body[3] - body[2]) * (body[5] - body[4])),
                ref_length=float(body[1] - body[0]),
                schedule=SamplingSchedule(every_n_steps=1),
            )
        )
    config = FVMSetup(
        case_name="coupled_test",
        execution=ComputeConfig(),
        time=TimeConfig(time_step_size=0.05, end_time=0.3, output_interval_steps=10**9),
        schemes=DiscretizationConfig(
            convection_scheme="LUST", gradient_scheme="lsq", time_scheme="backward"
        ),
        linear=LinearSolverConfig(
            linear_solver="bicgstab",
            pressure_solver="amg",
            pressure_tolerance=1e-8,
            momentum_max_iterations=2000,
            ilu_drop_tolerance=1e-4,
            ilu_fill_factor=10.0,
            ilu_reuse_tolerance=0.05,
        ),
        pimple=PimpleControl(n_correctors=2, n_outer_correctors=2),
        samplers=tuple(samplers),
        transport=TransportConfig(density=1.0, kinematic_viscosity=0.01),
        boundaries=boundaries,
        initial_velocity=[1.0, 0.0, 0.0],
    )
    with contextlib.redirect_stdout(io.StringIO()):
        solver = FVMSolver(config, case_dir=str(tmp_path), mesh_data=mesh)
    solver.auto_write = False
    return solver


def test_coupling_box_mesh_single_merged_patch():
    mesh = coupling_box_mesh(BOX, 0.25, patch_name="numericalBoundary")
    assert mesh["n_cells"] == 4**3
    assert len(mesh["boundary"]) == 1
    patch = mesh["boundary"][0]
    assert patch["name"] == "numericalBoundary"
    assert patch["n_faces"] == 6 * 4 * 4
    assert patch["start_face"] == mesh["n_interior_faces"]
    assert mesh["n_faces"] == mesh["n_interior_faces"] + patch["n_faces"]
    # Points span the physical box, not [0, L]³.
    assert np.allclose(mesh["points"].min(axis=0), [-0.5, -0.5, -0.5])
    assert np.allclose(mesh["points"].max(axis=0), [0.5, 0.5, 0.5])


def test_coupling_box_mesh_can_keep_spanwise_faces_empty():
    mesh = coupling_box_mesh(
        BOX,
        0.25,
        patch_name="numericalBoundary",
        empty_spanwise=True,
    )

    assert [patch["name"] for patch in mesh["boundary"]] == [
        "numericalBoundary",
        "zmin",
        "zmax",
    ]
    coupling, zmin, zmax = mesh["boundary"]
    assert coupling["n_faces"] == 4 * 4 * 4
    assert coupling["start_face"] == mesh["n_interior_faces"]
    assert zmin["type"] == zmax["type"] == "empty"
    assert zmin["n_faces"] == zmax["n_faces"] == 4 * 4
    assert zmin["start_face"] == coupling["start_face"] + coupling["n_faces"]
    assert zmax["start_face"] == zmin["start_face"] + zmin["n_faces"]
    assert mesh["n_faces"] == zmax["start_face"] + zmax["n_faces"]


def test_coupling_box_mesh_rejects_non_conforming_spacing():
    with pytest.raises(ValueError, match="integer"):
        coupling_box_mesh(BOX, 0.3)


@pytest.fixture(scope="module")
def built_backend(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("fvm_backend")
    setup = _fvm_setup(tmp)
    return setup, _build_backend(tmp)


def test_fvm_setup_adopts_solver_values_and_derives_box(tmp_path):
    """The coupler reads FVM-owned values without mutating CouplerSetup."""
    from source.coupler.solver import FVMVPMCoupler

    fvm = _build_backend(tmp_path)
    setup = _fvm_setup(tmp_path)
    coupler = FVMVPMCoupler(fvm, object(), setup)
    coupler.fvm_solver = fvm
    coupler._read_fvm_state()
    assert coupler.fvm_time_step_size == 0.05
    assert coupler.end_time == 0.3
    assert coupler.kinematic_viscosity == 0.01
    assert np.allclose(coupler.fvm_box, BOX, atol=1e-12)
    assert "nu" not in setup.to_dict()["coupler"]


def test_contract_methods_present(built_backend):
    _, fvm = built_backend
    missing = [m for m in CONTRACT_METHODS if not callable(getattr(fvm, m, None))]
    assert not missing, f"missing coupler-contract methods: {missing}"


def test_boundary_geometry_matches_coupler_expectations(built_backend):
    setup, fvm = built_backend
    fc = np.asarray(fvm.get_boundary_face_centre_coordinates(setup.coupling_patch))
    fn = np.asarray(fvm.get_boundary_face_normals(setup.coupling_patch))
    fa = np.asarray(fvm.get_boundary_face_areas(setup.coupling_patch))

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
    fc = np.asarray(fvm.get_boundary_face_centre_coordinates(setup.coupling_patch))
    u_bc = np.tile(setup.freestream_velocity_vector, (fc.shape[0], 1))
    fvm.set_dirichlet_velocity_boundary_condition_vec(u_bc, setup.coupling_patch)
    with contextlib.redirect_stdout(io.StringIO()):
        fvm.solve_pimple()
        fvm.advance_time()
    velocity = np.asarray(fvm.get_velocity_field())
    assert np.isfinite(velocity).all()
    # Uniform inflow through an empty box stays uniform.
    assert np.allclose(velocity.mean(axis=0), setup.freestream_velocity_vector, atol=1e-8)


def test_mixed_bc_preserves_prescribed_normal_flux_through_pimple(built_backend):
    import contextlib
    import io

    setup, fvm = built_backend
    normals = np.asarray(fvm.get_boundary_face_normals(setup.coupling_patch))
    areas = np.asarray(fvm.get_boundary_face_areas(setup.coupling_patch))
    normal_velocity = normals @ setup.freestream_velocity_vector
    tangential_gradient = np.zeros_like(normals)
    fvm.set_normal_velocity_tangential_gradient_boundary_condition(
        normal_velocity, tangential_gradient, setup.coupling_patch
    )
    fvm.set_flux_consistent_pressure_boundary_condition(setup.coupling_patch)
    with contextlib.redirect_stdout(io.StringIO()):
        fvm.solve_pimple()

    patch = next(b for b in fvm.boundaries if b["name"] == setup.coupling_patch)
    start = patch["start_face"]
    actual_flux = fvm.face_flux[start : start + patch["n_faces"]]
    np.testing.assert_allclose(actual_flux, normal_velocity * areas, atol=1.0e-11)
    assert patch["velocity_type"] == "normalValueTangentialGradient"
    assert patch["pressure_type"] == "fixedFluxPressure"
    assert "value_velocity_field" not in patch


def test_characteristic_vpm_bc_sets_matching_velocity_and_pressure(built_backend):
    setup, fvm = built_backend
    fc = np.asarray(fvm.get_boundary_face_centre_coordinates(setup.coupling_patch))
    u_bc = np.tile(setup.freestream_velocity_vector, (fc.shape[0], 1))
    fvm.set_freestream_velocity_boundary_condition_vec(u_bc, setup.coupling_patch)
    fvm.set_freestream_pressure_boundary_condition(setup.coupling_patch, value=0.0)
    patch = next(b for b in fvm.mesh_data["boundary"] if b["name"] == setup.coupling_patch)
    assert patch["velocity_type"] == "freestream"
    assert patch["pressure_type"] == "freestream"
    assert patch["kinematic_pressure_value"] == 0.0


def test_directional_outflow_vpm_bc_fixes_only_downstream_switch(tmp_path):
    import contextlib
    import io

    setup = _fvm_setup(tmp_path)
    fvm = _build_backend(tmp_path)
    fc = np.asarray(fvm.get_boundary_face_centre_coordinates(setup.coupling_patch))
    u_bc = np.tile(setup.freestream_velocity_vector, (fc.shape[0], 1))
    fvm.set_directional_freestream_velocity_boundary_condition_vec(
        u_bc, setup.coupling_patch, setup.freestream_velocity_vector
    )
    fvm.set_directional_freestream_pressure_boundary_condition(setup.coupling_patch, value=0.0)
    patch = next(b for b in fvm.mesh_data["boundary"] if b["name"] == setup.coupling_patch)
    outflow = patch["_freestream_outflow"]
    local_centres = np.asarray(fvm.get_boundary_face_centre_coordinates(setup.coupling_patch))

    assert patch["velocity_type"] == "freestream"
    assert patch["pressure_type"] == "freestream"
    assert patch["_directional_fixed_flux_pressure"] is True
    assert np.count_nonzero(outflow) == 4 * 4
    assert np.allclose(local_centres[outflow, 0], BOX[1])
    with contextlib.redirect_stdout(io.StringIO()):
        fvm.solve_pimple()
    np.testing.assert_array_equal(patch["_freestream_outflow"], outflow)
    delta = np.asarray(patch["fixed_flux_pressure_delta"])
    assert np.isfinite(delta).all()
    n_cells = fvm.mesh_data["n_cells"]
    n_interior = fvm.mesh_data["n_interior_faces"]
    start = patch["start_face"]
    ghosts = n_cells + np.arange(start - n_interior, start - n_interior + patch["n_faces"])
    owners = fvm.mesh_data["owners"][start : start + patch["n_faces"]]
    np.testing.assert_allclose(fvm.kinematic_pressure[ghosts[outflow]], 0.0)
    np.testing.assert_allclose(
        fvm.kinematic_pressure[ghosts[~outflow]],
        fvm.kinematic_pressure[owners[~outflow]] + delta[~outflow],
    )


def test_coupler_characteristic_step_uses_matching_velocity_and_pressure(tmp_path):
    from source.coupler.solver import FVMVPMCoupler

    class FakeFVM:
        last_yplus = None
        time_step_size = 0.01
        step = 0
        time = 0.0
        cfl_max = 0.0
        logger = _FakeStepLogger()
        setup = _FakeFVMConfig()

        def __init__(self):
            self.calls = []

        def set_freestream_velocity_boundary_condition_vec(self, values, patch):
            self.calls.append(("velocity", patch, np.asarray(values).copy()))

        def set_freestream_pressure_boundary_condition(self, patch, value=0.0):
            self.calls.append(("pressure", patch, value))

        def solve_pimple(self):
            self.calls.append(("solve",))

        def advance_time(self):
            self.calls.append(("advance",))

    coupler = object.__new__(FVMVPMCoupler)
    coupler.fvm_solver = FakeFVM()
    coupler.setup = _fvm_setup(tmp_path, boundary_condition_mode="characteristic")
    target = np.array([[1.0, 0.1, 0.0], [0.8, -0.1, 0.0]])
    apply_fvm_boundary(coupler, "numericalBoundary", target)

    assert [call[0] for call in coupler.fvm_solver.calls] == [
        "velocity",
        "pressure",
        "solve",
        "advance",
    ]
    np.testing.assert_array_equal(coupler.fvm_solver.calls[0][2], target)


def test_coupler_directional_outflow_step_passes_freestream_direction(tmp_path):
    from source.coupler.solver import FVMVPMCoupler

    class FakeFVM:
        last_yplus = None
        time_step_size = 0.01
        step = 0
        time = 0.0
        cfl_max = 0.0
        logger = _FakeStepLogger()
        setup = _FakeFVMConfig()

        def __init__(self):
            self.calls = []

        def set_directional_freestream_velocity_boundary_condition_vec(
            self, values, patch, direction
        ):
            self.calls.append(
                ("velocity", patch, np.asarray(values).copy(), np.asarray(direction).copy())
            )

        def set_directional_freestream_pressure_boundary_condition(self, patch, value=0.0):
            self.calls.append(("pressure", patch, value))

        def solve_pimple(self):
            self.calls.append(("solve",))

        def advance_time(self):
            self.calls.append(("advance",))

    coupler = object.__new__(FVMVPMCoupler)
    coupler.fvm_solver = FakeFVM()
    coupler.setup = _fvm_setup(tmp_path, boundary_condition_mode="directional_outflow")
    target = np.array([[1.0, 0.1, 0.0], [0.8, -0.1, 0.0]])
    apply_fvm_boundary(coupler, "numericalBoundary", target)

    assert [call[0] for call in coupler.fvm_solver.calls] == [
        "velocity",
        "pressure",
        "solve",
        "advance",
    ]
    np.testing.assert_array_equal(coupler.fvm_solver.calls[0][2], target)
    np.testing.assert_array_equal(
        coupler.fvm_solver.calls[0][3], coupler.setup.freestream_velocity_vector
    )


def test_coupler_pressure_gradient_step_sets_velocity_and_pressure(tmp_path):
    from source.coupler.solver import FVMVPMCoupler

    class FakeFVM:
        last_yplus = None
        time_step_size = 0.01
        step = 0
        time = 0.0
        cfl_max = 0.0
        logger = _FakeStepLogger()
        setup = _FakeFVMConfig()

        def __init__(self):
            self.calls = []

        def set_dirichlet_velocity_boundary_condition_vec(self, values, patch):
            self.calls.append(("velocity", patch, np.asarray(values).copy()))

        def set_neumann_pressure_boundary_condition(self, values, patch):
            self.calls.append(("pressure", patch, np.asarray(values).copy()))

        def solve_pimple(self):
            self.calls.append(("solve",))

        def advance_time(self):
            self.calls.append(("advance",))

    coupler = object.__new__(FVMVPMCoupler)
    coupler.fvm_solver = FakeFVM()
    coupler.setup = _fvm_setup(tmp_path, boundary_condition_mode="pressure_gradient")
    target = np.array([[1.0, 0.1, 0.0], [0.8, -0.1, 0.0]])
    gradient = np.array([[0.2, 0.0, 0.0], [-0.1, 0.3, 0.0]])
    apply_fvm_boundary(coupler, "numericalBoundary", target, gradient)

    assert [call[0] for call in coupler.fvm_solver.calls] == [
        "velocity",
        "pressure",
        "solve",
        "advance",
    ]
    np.testing.assert_array_equal(coupler.fvm_solver.calls[0][2], target)
    np.testing.assert_array_equal(coupler.fvm_solver.calls[1][2], gradient)


def test_coupler_vorticity_mixed_step_sets_directional_trace_and_pressure(tmp_path):
    from source.coupler.solver import FVMVPMCoupler

    class FakeFVM:
        last_yplus = None
        time_step_size = 0.01
        step = 0
        time = 0.0
        cfl_max = 0.0
        logger = _FakeStepLogger()
        setup = _FakeFVMConfig()

        def __init__(self):
            self.calls = []

        def set_normal_velocity_tangential_gradient_boundary_condition(
            self, normal_velocity, tangential_gradient, patch
        ):
            self.calls.append(
                (
                    "velocity",
                    patch,
                    np.asarray(normal_velocity).copy(),
                    np.asarray(tangential_gradient).copy(),
                )
            )

        def set_flux_consistent_pressure_boundary_condition(self, patch):
            self.calls.append(("pressure", patch))

        def solve_pimple(self):
            self.calls.append(("solve",))

        def advance_time(self):
            self.calls.append(("advance",))

    coupler = object.__new__(FVMVPMCoupler)
    coupler.fvm_solver = FakeFVM()
    coupler.setup = _fvm_setup(tmp_path, boundary_condition_mode="vorticity_mixed")
    target = np.array([[1.0, 0.1, 0.0], [0.8, -0.1, 0.0]])
    normal_velocity = np.array([-1.0, 0.8])
    tangential_gradient = np.array([[0.0, 0.2, 0.3], [0.0, -0.1, 0.4]])

    apply_fvm_boundary(
        coupler,
        "numericalBoundary",
        target,
        normal_velocity=normal_velocity,
        tangential_gradient=tangential_gradient,
    )

    assert [call[0] for call in coupler.fvm_solver.calls] == [
        "velocity",
        "pressure",
        "solve",
        "advance",
    ]
    np.testing.assert_array_equal(coupler.fvm_solver.calls[0][2], normal_velocity)
    np.testing.assert_array_equal(coupler.fvm_solver.calls[0][3], tangential_gradient)


def test_velocity_gradient_contract_uses_configured_reconstruction(built_backend):
    _, fvm = built_backend
    gradient = np.asarray(fvm.get_velocity_gradient_field())
    assert gradient.shape == (fvm.mesh_data["n_cells"], 3, 3)
    assert np.isfinite(gradient).all()
    out = np.empty_like(gradient)
    assert fvm.get_velocity_gradient_field_into(out) is out
    np.testing.assert_array_equal(out, gradient)


def test_coupler_runtime_setters_apply(built_backend):
    _, fvm = built_backend
    # What FVMVPMCoupler.initialize() stamps on the injected backend.
    fvm.set_time_step(0.025)
    fvm.set_kinematic_viscosity(0.02)
    assert fvm.time_step_size == 0.025
    assert fvm.setup.transport.kinematic_viscosity == 0.02
    assert fvm.n_procs() == 1
    # Coupled runs must not let the FVM adapt its own dt.
    assert fvm.setup.time.adjust_time_step is False


def test_initialize_rejects_serial_backend_under_mpi(tmp_path, monkeypatch):
    import source.coupler.solver as coupler_mod

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(coupler_mod, "_mpi4py_comm", None)
    monkeypatch.setenv("OMPI_COMM_WORLD_SIZE", "2")

    class _SerialBackend:
        case_dir = str(tmp_path)

        def n_procs(self):
            return 1

    setup = _fvm_setup(tmp_path)
    coupler = coupler_mod.FVMVPMCoupler(_SerialBackend(), object(), setup)
    with pytest.raises(RuntimeError, match="serial"):
        coupler.initialize()


# ---------------------------------------------------------------------------
# Body-fitted mesh: carved hole plus wall patch.
# ---------------------------------------------------------------------------

HOLE = (-0.5, 0.5, -0.5, 0.5, -0.5, 0.5)


def test_carved_mesh_hole_and_wall_patch():
    from source.solvers.FVM.mesh.rectilinear import coupling_box_mesh

    mesh = coupling_box_mesh(
        (-1.5, 1.5, -1.5, 1.5, -1.5, 1.5), 0.125, hole_box=HOLE, wall_patch_name="cube"
    )
    n_side, n_cube = 24, 8
    assert mesh["n_cells"] == n_side**3 - n_cube**3
    names = [b["name"] for b in mesh["boundary"]]
    assert names == ["numericalBoundary", "cube"]
    wall = mesh["boundary"][1]
    assert wall["type"] == "wall"
    assert wall["n_faces"] == 6 * n_cube**2
    # Patches tile the boundary contiguously after the interior faces.
    coupling = mesh["boundary"][0]
    assert wall["start_face"] == coupling["start_face"] + coupling["n_faces"]
    assert mesh["n_faces"] == mesh["n_interior_faces"] + coupling["n_faces"] + wall["n_faces"]
    # Owner/neighbour indices are compact after the carve.
    assert mesh["owners"].max() < mesh["n_cells"]
    assert mesh["neighbours"].max() < mesh["n_cells"]

    # Wall faces lie on the cube surface with outward-of-fluid normals
    # (pointing into the hole, i.e. toward the cube centre at the origin).
    pts = mesh["points"]
    sl = slice(wall["start_face"], wall["start_face"] + wall["n_faces"])
    for face in mesh["faces"][sl.start : sl.stop]:
        p = pts[face]
        c = p.mean(axis=0)
        assert np.isclose(np.abs(c), 0.5).any()
        n = np.cross(p[1] - p[0], p[2] - p[0])
        assert np.dot(n, -c) > 0.0


def test_carved_mesh_misaligned_hole_raises():
    from source.solvers.FVM.mesh.rectilinear import coupling_box_mesh

    with pytest.raises(ValueError, match="mesh plane"):
        coupling_box_mesh((-1.5, 1.5, -1.5, 1.5, -1.5, 1.5), 0.075, hole_box=HOLE)


def test_builder_body_fitted_cube(tmp_path):
    import contextlib
    import io

    setup = _fvm_setup(
        tmp_path,
        vpm_particle_spacing=0.125,
        authority_ramp_width=0.375,
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
    cc = np.asarray(fvm.get_cell_centre_coordinates())
    assert np.all(np.abs(cc) < 0.5, axis=1).sum() == 0

    # Wall patch reachable through the coupler contract; no-slip configured;
    # wall-force logging on (as an explicit ForceSampler).
    wf = np.asarray(fvm.get_boundary_face_centre_coordinates("cube"))
    assert wf.shape[0] == 6 * 8**2
    force_sampler = next(s for s in fvm.setup.samplers if s.name == "forces_history")
    assert force_sampler.patch_names == ["cube"]
    wall_cfg = next(b for b in fvm.setup.boundaries if b.name == "cube")
    assert wall_cfg.velocity_type == "fixedValue" and wall_cfg.velocity_value == [0.0, 0.0, 0.0]

    # One PIMPLE step stays finite and generates wall vorticity.
    fc = np.asarray(fvm.get_boundary_face_centre_coordinates(setup.coupling_patch))
    u_bc = np.tile(setup.freestream_velocity_vector, (fc.shape[0], 1))
    fvm.set_dirichlet_velocity_boundary_condition_vec(u_bc, setup.coupling_patch)
    with contextlib.redirect_stdout(io.StringIO()):
        fvm.solve_pimple()
        fvm.advance_time()
    velocity = np.asarray(fvm.get_velocity_field())
    assert np.isfinite(velocity).all()
    w = np.linalg.norm(np.asarray(fvm.get_vorticity_field()), axis=1)
    assert w.max() > 1.0
