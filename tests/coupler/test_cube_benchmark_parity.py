"""Physics and discretisation parity for the hybrid/reference cube cases."""

from __future__ import annotations

import contextlib
import importlib.util
import io
from pathlib import Path
import sys

import numpy as np
import pytest

_CASE = Path(__file__).parents[2] / "tutorials/coupled_FVM_VPM/cube_flow/cubeFlow_setup.py"
_REFERENCE = (
    Path(__file__).parents[2]
    / "tutorials/coupled_FVM_VPM/cube_flow/reference_flow/referenceFlow_setup.py"
)
_CUBE_ROOT = Path(__file__).parents[2] / "tutorials/coupled_FVM_VPM/cube_flow"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    search_paths = [str(path.parent), str(path.parent.parent)]
    sys.path[:0] = search_paths
    try:
        spec.loader.exec_module(module)
    finally:
        del sys.path[: len(search_paths)]
    return module


def _small_coupled_mesh(core_box):
    """A coarse gmsh box-minus-cube mesh (numericalBoundary + cube), like the
    hybrid case's mesher but small enough to build inside a test."""
    gmsh = pytest.importorskip("gmsh", reason="hybrid parity solver needs Gmsh")
    from source.solvers.FVM.mesh.gmsh_importer import GmshImporter

    x0, x1, y0, y1, z0, z1 = core_box
    gmsh.initialize()
    try:
        gmsh.model.add("hybrid_parity")
        box = gmsh.model.occ.addBox(x0, y0, z0, x1 - x0, y1 - y0, z1 - z0)
        cube = gmsh.model.occ.addBox(-0.5, -0.5, -0.5, 1.0, 1.0, 1.0)
        fluid = gmsh.model.occ.cut([(3, box)], [(3, cube)])[0][0][1]
        gmsh.model.occ.synchronize()
        outer, wall = [], []
        for dim, tag in gmsh.model.getBoundary([(3, fluid)], oriented=False):
            cx, cy, cz = gmsh.model.occ.getCenterOfMass(dim, tag)
            on_outer = (
                abs(cx - x0) < 1e-6
                or abs(cx - x1) < 1e-6
                or abs(cy - y0) < 1e-6
                or abs(cy - y1) < 1e-6
                or abs(cz - z0) < 1e-6
                or abs(cz - z1) < 1e-6
            )
            (outer if on_outer else wall).append(tag)
        gmsh.model.setPhysicalName(2, gmsh.model.addPhysicalGroup(2, outer), "numericalBoundary")
        gmsh.model.setPhysicalName(2, gmsh.model.addPhysicalGroup(2, wall), "cube")
        gmsh.model.addPhysicalGroup(3, [fluid])
        gmsh.option.setNumber("Mesh.MeshSizeMax", 0.5)
        gmsh.model.mesh.generate(3)
        importer = GmshImporter()
        try:
            return importer.get_mesh_data()
        finally:
            importer.finalize()
    finally:
        if gmsh.isInitialized():
            gmsh.finalize()


@pytest.fixture(scope="module")
def bench():
    return _load("hybrid_cube_setup", _CASE)


@pytest.fixture(scope="module")
def reference():
    return _load("reference_cube_setup", _REFERENCE)


@pytest.fixture(scope="module")
def vpm(bench):
    return bench.VPM_SETUP


@pytest.fixture(scope="module")
def hybrid_solver(bench, tmp_path_factory):
    from source.solvers.FVM import FVMSolver

    case_dir = tmp_path_factory.mktemp("hybrid_parity")
    with contextlib.redirect_stdout(io.StringIO()):
        return FVMSolver(
            bench.FVM_SETUP, case_dir=str(case_dir), mesh_data=_small_coupled_mesh(bench.FVM_BOX)
        )


def test_wall_boundary_identical(bench, reference):
    def wall(setup):
        (boundary,) = [item for item in setup.boundaries if item.name == "cube"]
        return boundary

    assert wall(bench.FVM_SETUP) == wall(reference.FVM_SETUP)


def test_cube_case_has_no_smoke_mode() -> None:
    source = _CASE.read_text(encoding="utf-8")
    readme = (_CUBE_ROOT / "README.md").read_text(encoding="utf-8")
    assert "OPENONDA_SMOKE" not in source
    assert "OPENONDA_SMOKE" not in readme
    assert "os.environ" not in source


def test_sampling_cadence_matches_reference(bench, reference):
    def schedules(samplers):
        return {
            sampler.__class__.__name__: (
                sampler.schedule.every_n_steps,
                sampler.schedule.every_time,
            )
            for sampler in samplers
        }

    hybrid_fvm = schedules(bench.FVM_SAMPLERS)
    reference_fvm = schedules(reference.SAMPLERS)
    assert hybrid_fvm["ForceSampler"] == reference_fvm["ForceSampler"]
    assert hybrid_fvm["LineSampler"] == reference_fvm["LineSampler"]
    assert hybrid_fvm["SurfaceSampler"] == reference_fvm["SurfaceSampler"]
    assert hybrid_fvm["ForceSampler"] == (bench.LINE_SAMPLE_EVERY_FVM_STEPS, None)
    assert hybrid_fvm["LineSampler"] == (bench.LINE_SAMPLE_EVERY_FVM_STEPS, None)
    assert hybrid_fvm["SurfaceSampler"] == (bench.SLICE_SAMPLE_EVERY_FVM_STEPS, None)

    for sampler in bench.VPM_SAMPLERS:
        expected = (
            bench.LINE_SAMPLE_EVERY_COUPLING_STEPS
            if sampler.__class__.__name__ == "LineSampler"
            else bench.SLICE_SAMPLE_EVERY_COUPLING_STEPS
        )
        assert sampler.schedule.every_n_steps == expected
        assert sampler.schedule.every_time is None


def test_backup_schedule_uses_shared_accepted_steps(bench):
    assert bench.FVM_SETUP.time.output_interval_steps == bench.BACKUP_EVERY_FVM_STEPS
    assert bench.FVM_SETUP.time.output_interval_time is None
    assert bench.VPM_SETUP.checkpoint_interval_steps == bench.BACKUP_EVERY_COUPLING_STEPS
    assert bench.VPM_SETUP.checkpoint_interval_time is None
    assert bench.COUPLER_SETUP.checkpoint_interval_steps == bench.BACKUP_EVERY_COUPLING_STEPS
    assert bench.COUPLER_SETUP.checkpoint_interval_time is None


def test_shared_cadence_lands_on_same_solver_states(bench, reference):
    assert bench.FVM_STEPS_PER_COUPLING_STEP == 3
    assert bench.LINE_SAMPLE_EVERY_FVM_STEPS == (
        bench.LINE_SAMPLE_EVERY_COUPLING_STEPS * bench.FVM_STEPS_PER_COUPLING_STEP
    )
    assert bench.SLICE_SAMPLE_EVERY_FVM_STEPS == (
        bench.SLICE_SAMPLE_EVERY_COUPLING_STEPS * bench.FVM_STEPS_PER_COUPLING_STEP
    )
    assert bench.BACKUP_EVERY_FVM_STEPS == (
        bench.BACKUP_EVERY_COUPLING_STEPS * bench.FVM_STEPS_PER_COUPLING_STEP
    )
    assert reference.FVM_SETUP.time.output_interval_steps == bench.BACKUP_EVERY_FVM_STEPS


def test_requested_end_uses_nearest_complete_coupling_step(bench):
    assert pytest.approx(bench.COUPLING_STEPS * bench.VPM_TIME_STEP_SIZE) == bench.END_TIME
    assert abs(bench.END_TIME - bench.REQUESTED_END_TIME) <= bench.VPM_TIME_STEP_SIZE / 2


def test_production_particle_lattice_is_cube_wall_commensurate(bench):
    assert pytest.approx(0.03125) == bench.VPM_PARTICLE_SPACING
    assert pytest.approx(32.0) == bench.CUBE_SIDE / bench.VPM_PARTICLE_SPACING
    assert pytest.approx(0.03) == bench.VPM_TIME_STEP_SIZE
    assert pytest.approx(0.01) == bench.FVM_TIME_STEP_SIZE


def test_cube_vpm_uses_common_stage_advection_stretching(bench):
    assert bench.VPM_SETUP.time_integration == "COUPLED"
    assert bench.VPM_SETUP.advection.scheme == bench.VPM_SETUP.stretching.scheme == "RK2"
    assert bench.VPM_SETUP.coupled_max_strain_increment is None
    assert bench.VPM_SETUP.coupled_max_advection_fraction is None


def test_hybrid_pressure_correction_matches_reference(bench, reference):
    hybrid = bench.FVM_SETUP.pimple
    monolithic = reference.FVM_SETUP.pimple
    assert hybrid.n_correctors == monolithic.n_correctors
    assert hybrid.n_outer_correctors == monolithic.n_outer_correctors
    assert hybrid.n_orthogonal_correctors == monolithic.n_orthogonal_correctors


def test_legitimate_differences_are_the_only_differences(bench, reference):
    hybrid_names = {boundary.name for boundary in bench.FVM_SETUP.boundaries}
    reference_names = {boundary.name for boundary in reference.FVM_SETUP.boundaries}
    assert hybrid_names == {"numericalBoundary", "cube"}
    assert reference_names == {"inlet", "outlet", "ymin", "ymax", "zmin", "zmax", "cube"}


def test_coupler_setup_owns_no_solver_physics(bench):
    setup = bench.COUPLER_SETUP
    for name in (
        "nu",
        "rho",
        "dt",
        "t_end",
        "fvm_box",
        "grid_spacing",
        "initial_velocity",
        "surface",
        "wall_patch_name",
    ):
        assert not hasattr(setup, name)
    assert setup.vpm_only_width == 0.0


def test_vpm_setup_compatible(bench, vpm):
    assert type(vpm).__name__ == "VPMSetup"
    assert vpm.viscous.kinematic_viscosity == pytest.approx(bench.KINEMATIC_VISCOSITY)
    assert tuple(vpm.freestream_velocity) == tuple(bench.FREESTREAM_VELOCITY)
    ratio = vpm.time_step_size / bench.FVM_TIME_STEP_SIZE
    assert ratio == pytest.approx(round(ratio))
    domain = np.asarray(vpm.domain_bounds, dtype=float)
    box = np.asarray(bench.FVM_BOX, dtype=float)
    assert np.all(domain[::2] <= box[::2]) and np.all(domain[1::2] >= box[1::2])
    assert vpm.compute_device == "AUTO"
    assert vpm.precision == "f32"
    assert vpm.panel_solver.bc_type == "NEUMANN"
    assert vpm.panel_solver.coupling_scope == "vpm_bc"
    assert vpm.turbulence.flow_model == "LES"
    recovered_ck = (vpm.turbulence.c_s**2 * vpm.turbulence.c_e**0.5) ** (2.0 / 3.0)
    assert recovered_ck == pytest.approx(0.094)


def test_mesh_domain_uses_case_setting(bench, vpm):
    from source.solvers.FVM.mesh.triangulated_surface import TriangulatedSurface

    resolved = np.asarray(bench.FVM_MESH.domain)
    requested = np.asarray(bench.FVM_BOX)
    assert np.all(resolved[::2] <= requested[::2])
    assert np.all(resolved[1::2] >= requested[1::2])
    assert np.max(np.abs(resolved - requested)) < bench.FVM_MESH.requested_max_cell_size
    assert bench.FVM_MESH.requested_max_cell_size == pytest.approx(bench.SURFACE_CELL_SIZE * 4)
    assert bench.FVM_MESH.max_cell_size <= (bench.SURFACE_CELL_SIZE * 4)
    assert bench.FVM_MESH.surface_cell_size == pytest.approx(bench.SURFACE_CELL_SIZE)
    assert bench.FVM_MESH.surface_file == str(bench.CUBE_STL.resolve())
    surface = TriangulatedSurface.from_stl(bench.CUBE_STL)
    assert surface.bounds == (-0.5, 0.5, -0.5, 0.5, -0.5, 0.5)
    assert vpm.panel_solver.max_panels >= len(surface.triangles)


def test_cube_main_is_rank_agnostic(bench, monkeypatch):
    class FakeFVM:
        def write_vtk(self):
            pass

    class FakeCoupler:
        def run(self):
            pass

    fake_vpm_solver = object()
    received = {}

    monkeypatch.setattr(
        bench.fvm,
        "create_fvm_solver",
        lambda *args, **kwargs: FakeFVM(),
    )

    def fake_create_vpm_solver(setup, **kwargs):
        received["vpm_setup"] = setup
        received["vpm_kwargs"] = kwargs
        return fake_vpm_solver

    monkeypatch.setattr(
        bench.vpm,
        "create_vpm_solver",
        fake_create_vpm_solver,
    )

    def fake_create_coupler(fvm_solver, vpm_solver, setup):
        received["coupled_vpm"] = vpm_solver
        received["coupler_setup"] = setup
        return FakeCoupler()

    monkeypatch.setattr(
        bench.coupling,
        "create_coupler",
        fake_create_coupler,
    )

    bench.main()

    assert received["vpm_setup"] is bench.VPM_SETUP
    assert received["vpm_kwargs"]["case_dir"] == bench.CASE_DIR
    assert received["coupled_vpm"] is fake_vpm_solver
    assert received["coupler_setup"] is bench.COUPLER_SETUP


def test_coupler_adopts_and_validates_hybrid_solver(bench, hybrid_solver, tmp_path):
    from source.coupler import FVMVPMCoupler

    setup = bench.COUPLER_SETUP
    coupler = FVMVPMCoupler(hybrid_solver, object(), setup)
    coupler.fvm_solver = hybrid_solver
    coupler._read_fvm_state()
    assert coupler.fvm_time_step_size == pytest.approx(bench.FVM_TIME_STEP_SIZE)
    assert coupler.end_time == pytest.approx(bench.END_TIME)
    assert coupler.kinematic_viscosity == pytest.approx(bench.KINEMATIC_VISCOSITY)
    assert np.allclose(coupler.fvm_box, bench.FVM_BOX, atol=1e-12)


def test_pressure_anchor_selection_caches_nonmaster_empty_view():
    from source.coupler.pressure_reference import PressureReference

    class FakeFVM:
        def __init__(self):
            self.calls = 0

        def get_cell_centre_coordinates(self):
            self.calls += 1
            return np.empty((0, 3))

    fvm = FakeFVM()
    pressure = PressureReference(
        fvm,
        fvm_box=np.array([-1, 1, -1, 1, -1, 1]),
        freestream_velocity=np.array([1.0, 0.0, 0.0]),
        particle_spacing=0.1,
        boundary_mode="dirichlet",
        enabled=True,
        is_master=False,
    )
    assert pressure._selection() is None
    assert pressure.cell_indices is not None
    assert pressure._selection() is None
    assert fvm.calls == 1


def test_incompatible_vpm_viscosity_raises(bench, tmp_path):
    from source.coupler import FVMVPMCoupler

    class _FakeViscous:
        kinematic_viscosity = 10 * bench.KINEMATIC_VISCOSITY
        scheme = "CS"
        core_radius_ratio = 1.0

    class _FakeVPMConfig:
        viscous = _FakeViscous()
        domain_bounds = None

    class _FakeVPM:
        setup = _FakeVPMConfig()
        time_step_size = bench.VPM_TIME_STEP_SIZE
        freestream_velocity = bench.FREESTREAM_VELOCITY

    with pytest.raises(ValueError, match="viscosity"):
        FVMVPMCoupler._validate_vpm(
            _FakeVPM(), bench.COUPLER_SETUP, np.asarray(bench.FVM_BOX), bench.KINEMATIC_VISCOSITY
        )


def test_incompatible_vpm_freestream_raises(bench, tmp_path):
    from source.coupler import FVMVPMCoupler

    class _FakeVPM:
        freestream_velocity = [0.5, 0.0, 0.0]
        time_step_size = bench.VPM_TIME_STEP_SIZE
        setup = type(
            "Config",
            (),
            {
                "domain_bounds": None,
                "viscous": type(
                    "Viscous",
                    (),
                    {
                        "scheme": "CS",
                        "kinematic_viscosity": bench.KINEMATIC_VISCOSITY,
                        "core_radius_ratio": 1.0,
                    },
                )(),
            },
        )()

    with pytest.raises(ValueError, match="freestream"):
        FVMVPMCoupler._validate_vpm(
            _FakeVPM(), bench.COUPLER_SETUP, np.asarray(bench.FVM_BOX), bench.KINEMATIC_VISCOSITY
        )


@pytest.mark.parametrize(
    ("scheme", "attr", "mode"),
    [
        ("GBD", "gbd_threshold_mode", "relative_max"),
        ("GBD", "gbd_threshold_mode", "budget"),
        ("GBD", "gbd_threshold_mode", "absolute"),
        ("GBD", "gbd_threshold_mode", "relative_local"),
        ("DVH", "dvh_threshold_mode", "relative_max"),
        ("DVH", "dvh_threshold_mode", "budget"),
        ("DVH", "dvh_threshold_mode", "absolute"),
        ("DVH", "dvh_threshold_mode", "relative_local"),
    ],
)
def test_coupling_accepts_configured_regen_threshold_mode(bench, scheme, attr, mode):
    from source.coupler import FVMVPMCoupler

    nu = float(bench.FVM_SETUP.transport.kinematic_viscosity)

    class _FakeViscous:
        kinematic_viscosity = nu
        core_radius_ratio = bench.COUPLER_SETUP.vpm_core_radius_ratio

    class _FakeVPMSetup:
        viscous = _FakeViscous()
        domain_bounds = None

    class _FakeVPM:
        setup = _FakeVPMSetup()
        time_step_size = bench.VPM_TIME_STEP_SIZE
        freestream_velocity = bench.FREESTREAM_VELOCITY

    _FakeViscous.scheme = scheme
    setattr(_FakeViscous, attr, mode)

    FVMVPMCoupler._validate_vpm(_FakeVPM(), bench.COUPLER_SETUP, np.asarray(bench.FVM_BOX), nu)
