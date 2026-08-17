"""Physics and discretisation parity for the hybrid/reference cube cases."""

from __future__ import annotations

import contextlib
from dataclasses import replace
import importlib.util
import io
import math
from pathlib import Path
import sys

import numpy as np
import pytest

_CASE = Path(__file__).parents[2] / "tutorials/coupled_FVM_VPM/cubeFlow/cubeFlow_setup.py"
_REFERENCE = (
    Path(__file__).parents[2]
    / "tutorials/coupled_FVM_VPM/cubeFlow/referenceFlow/referenceFlow_setup.py"
)
_CUBE_ROOT = Path(__file__).parents[2] / "tutorials/coupled_FVM_VPM/cubeFlow"


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
    return bench.make_vpm_setup()


@pytest.fixture(scope="module")
def hybrid_solver(bench, tmp_path_factory):
    from source.solvers.FVM import Solver

    case_dir = tmp_path_factory.mktemp("hybrid_parity")
    serial_setup = replace(bench.FVM_SETUP, cores=1)
    with contextlib.redirect_stdout(io.StringIO()):
        return Solver(
            serial_setup, case_dir=str(case_dir), mesh_data=_small_coupled_mesh(bench.FVM_BOX)
        )


def test_common_fvm_settings_identical(bench, reference):
    hybrid = bench.FVM_SETUP
    fully_meshed = reference.FVM_SETUP

    assert hybrid.cores == fully_meshed.cores == 4
    assert hybrid.schemes == fully_meshed.schemes
    assert hybrid.linear == fully_meshed.linear
    assert fully_meshed.linear.momentum_tol <= 1.0e-6
    assert hybrid.pimple == fully_meshed.pimple
    assert fully_meshed.pimple.alpha_u == pytest.approx(0.7)
    assert fully_meshed.pimple.alpha_p == pytest.approx(0.3)
    assert hybrid.transport == fully_meshed.transport

    # Wall-load integration is an explicit sample; both setups carry an
    # equivalent wall ForceSampler (their other, differently-named field
    # samplers are intentionally not identical).
    def force_sampler(setup):
        return next(s for s in setup.samplers if s.name == "forces_history")

    assert force_sampler(hybrid) == force_sampler(fully_meshed)
    assert hybrid.execution == fully_meshed.execution
    assert hybrid.output == fully_meshed.output
    assert hybrid.time.delta_t == fully_meshed.time.delta_t
    assert fully_meshed.time.delta_t == pytest.approx(0.01)
    assert hybrid.time.start_time == fully_meshed.time.start_time
    assert hybrid.time.end_time == pytest.approx(6.0)
    assert fully_meshed.time.end_time == pytest.approx(20.0)
    assert hybrid.time.adjust_timestep is False
    assert fully_meshed.time.adjust_timestep is False
    assert hybrid.initial_U == fully_meshed.initial_U
    assert hybrid.turbulence == fully_meshed.turbulence
    assert fully_meshed.turbulence.model == "EquilibriumSmagorinsky"
    assert fully_meshed.turbulence.Ck == pytest.approx(0.094)
    assert fully_meshed.turbulence.Ce == pytest.approx(1.048)


def test_wall_boundary_identical(bench, reference):
    def wall(setup):
        (boundary,) = [item for item in setup.boundaries if item.name == "cube"]
        return boundary

    assert wall(bench.FVM_SETUP) == wall(reference.FVM_SETUP)


def test_legitimate_differences_are_the_only_differences(bench, reference):
    hybrid_names = {boundary.name for boundary in bench.FVM_SETUP.boundaries}
    reference_names = {boundary.name for boundary in reference.FVM_SETUP.boundaries}
    assert hybrid_names == {"numericalBoundary", "cube"}
    assert reference_names == {"inlet", "outlet", "ymin", "ymax", "zmin", "zmax", "cube"}


def test_coupler_setup_owns_no_solver_physics(bench):
    setup = bench.COUPLER_SETUP
    for name in ("nu", "rho", "dt", "t_end", "fvm_box", "grid_spacing", "initial_U"):
        assert getattr(setup, name) is None
    assert not setup.surface
    assert setup.dead_zone_h == 0.0


def test_vpm_setup_compatible(bench, vpm):
    assert type(vpm).__name__ == "VPMSetup"
    assert vpm.viscous.viscosity == pytest.approx(bench.NU)
    assert tuple(vpm.background_velocity) == tuple(bench.U_INF)
    ratio = vpm.time_step_size / bench.DT_FVM
    assert ratio == pytest.approx(round(ratio))
    domain = np.asarray(vpm.vpm_domain_bounds, dtype=float)
    box = np.asarray(bench.FVM_BOX, dtype=float)
    assert np.all(domain[::2] <= box[::2]) and np.all(domain[1::2] >= box[1::2])
    assert vpm.processing_unit == "AUTO"
    assert vpm.precision == "f32"
    assert vpm.panel_solver.bc_type == "NEUMANN"
    assert vpm.panel_solver.coupling_scope == "vpm_bc"
    assert vpm.turbulence.flow_model == "LES"
    recovered_ck = (vpm.turbulence.cs**2 * vpm.turbulence.ce**0.5) ** (2.0 / 3.0)
    assert recovered_ck == pytest.approx(0.094)


def test_mesh_domain_uses_case_setting(bench, vpm):
    from source.solvers.FVM.mesh.triangulated_surface import TriangulatedSurface

    assert bench.FVM_MESH.domain == bench.FVM_BOX
    assert bench.FVM_MESH.requested_max_cell_size == pytest.approx(bench.FVM_CELL_SIZE)
    assert bench.FVM_MESH.max_cell_size <= bench.FVM_CELL_SIZE
    assert bench.FVM_MESH.surface_cell_size == pytest.approx(0.015)
    assert bench.FVM_MESH.surface_file == str(bench.CUBE_STL.resolve())
    surface = TriangulatedSurface.from_stl(bench.CUBE_STL)
    assert surface.bounds == (-0.5, 0.5, -0.5, 0.5, -0.5, 0.5)
    assert vpm.panel_solver.max_panels >= len(surface.triangles)


def test_production_case_keeps_the_validated_cost_limits(bench, vpm):
    assert bench.HANDOFF_BOX == (-1.5, 3.2, -1.5, 1.5, -1.5, 1.5)
    assert bench.FVM_BOX == (-1.5, 3.5, -1.5, 1.5, -1.5, 1.5)
    assert bench.FVM_WAKE_BOX == (-1.25, 3.2, -1.25, 1.25, -1.25, 1.25)
    assert pytest.approx(0.06) == bench.FVM_CELL_SIZE
    assert pytest.approx(0.03) == bench.FVM_WAKE_CELL_SIZE
    assert pytest.approx(0.04) == bench.PARTICLE_SPACING
    assert bench.FVM_MESH.effective_cell_size(bench.FVM_MESH.surface_cell_size) == pytest.approx(
        bench.FVM_MESH.max_cell_size / 4.0
    )
    assert bench.FVM_MESH.effective_cell_size(bench.FVM_WAKE_CELL_SIZE) == pytest.approx(
        bench.FVM_MESH.max_cell_size / 2.0
    )
    assert bench.PARTICLE_LIMIT == 1_500_000
    assert vpm.viscous.gbd_max_nodes == bench.PARTICLE_LIMIT
    assert vpm.max_particles == bench.PARTICLE_LIMIT
    assert bench.COUPLER_SETUP.handoff_max_particles == bench.PARTICLE_LIMIT
    assert vpm.viscous.gbd_grid_spacing == pytest.approx(bench.PARTICLE_SPACING)
    assert bench.COUPLER_SETUP.h == pytest.approx(bench.PARTICLE_SPACING)
    assert bench.COUPLER_SETUP.buffer_thickness == pytest.approx(6 * bench.PARTICLE_SPACING)
    assert bench.COUPLER_SETUP.dead_zone_h == 0.0
    assert bench.COUPLER_SETUP.prune_vorticity_min == pytest.approx(0.005)
    assert bench.COUPLER_SETUP.boundary_prune_multiplier == pytest.approx(10.0)
    assert bench.COUPLER_SETUP.transfer_amplification_cap == pytest.approx(
        bench.TRANSFER_AMPLIFICATION_CAP
    )
    assert bench.COUPLER_SETUP.resync_vpm_bc_after_handoff is False


def test_output_names_and_cadence_match_allplot_contract(bench, reference, vpm):
    assert bench.FVM_SETUP.case_name.startswith("coupled_")
    assert reference.FVM_SETUP.case_name == "referenceFlow"
    assert vpm.backup_file_name == ""
    assert Path(vpm.backup_directory) == bench.CASE_DIR / "solution"
    assert vpm.backup_frequency == 0
    assert pytest.approx(bench.CHECKPOINT_INTERVAL) == bench.BACKUP_PERIOD * bench.DT_VPM
    assert pytest.approx(bench.DIAGNOSTIC_INTERVAL) == bench.VPM_LOG_PERIOD * bench.DT_VPM
    assert vpm.logging_frequency == bench.VPM_LOG_PERIOD
    assert bench.FVM_SETUP.time.write_interval_time == pytest.approx(bench.FVM_VOLUME_INTERVAL)
    hybrid_steps = round(bench.FORCE_INTERVAL / bench.DT_FVM)
    reference_steps = round(reference.WRITE_INTERVAL / reference.DT_FVM)
    assert hybrid_steps * bench.DT_FVM == pytest.approx(bench.FORCE_INTERVAL)
    assert reference_steps * reference.DT_FVM == pytest.approx(reference.WRITE_INTERVAL)
    common_time = math.lcm(hybrid_steps, reference_steps) * bench.DT_FVM
    assert common_time <= bench.T_END


def test_cube_main_builds_vpm_on_master_only(bench, monkeypatch):
    class FakeFVM:
        def write_vtk(self):
            pass

    class FakeCoupler:
        def run(self):
            pass

    received = []
    monkeypatch.setattr(bench, "setup_fvm_solver", lambda *args, **kwargs: FakeFVM())
    monkeypatch.setattr(bench.FVMVPMCoupler, "is_master_rank", staticmethod(lambda: False))
    monkeypatch.setattr(
        bench,
        "make_vpm_setup",
        lambda: pytest.fail("worker rank must not construct the GPU VPM setup"),
    )
    monkeypatch.setattr(
        bench,
        "setup_coupler",
        lambda vpm, fvm, setup: received.append(vpm) or FakeCoupler(),
    )

    bench.main()

    assert received == [None]


def test_coupler_adopts_and_validates_hybrid_solver(bench, hybrid_solver, tmp_path):
    from source.coupler import FVMVPMCoupler

    setup = replace(bench.COUPLER_SETUP, case_dir=str(tmp_path))
    coupler = FVMVPMCoupler(object(), hybrid_solver, setup)
    coupler.fvm = hybrid_solver
    coupler._resolve_eulerian_ownership()
    assert coupler.dt_fvm == pytest.approx(bench.DT_FVM)
    assert coupler.t_end == pytest.approx(bench.T_END)
    assert setup.nu == pytest.approx(bench.NU)
    assert np.allclose(setup.fvm_box, bench.FVM_BOX, atol=1e-12)

    bad = replace(
        bench.COUPLER_SETUP,
        case_dir=str(tmp_path),
        nu=2 * bench.NU,
    )
    coupler_bad = FVMVPMCoupler(object(), hybrid_solver, bad)
    coupler_bad.fvm = hybrid_solver
    with pytest.raises(ValueError, match="owns this value"):
        coupler_bad._resolve_eulerian_ownership()


def test_pressure_anchor_selection_caches_nonmaster_empty_view():
    from source.coupler import FVMVPMCoupler

    class FakeFVM:
        def __init__(self):
            self.calls = 0

        def get_cell_center_coordinates(self):
            self.calls += 1
            return np.empty((0, 3))

    coupler = object.__new__(FVMVPMCoupler)
    coupler._pressure_anchor_cells = None
    coupler.fvm = FakeFVM()
    coupler.config = type("Config", (), {"fvm_box": (-1, 1, -1, 1, -1, 1)})()
    coupler.u_inf = np.array([1.0, 0.0, 0.0])

    assert coupler._pressure_anchor_selection() is None
    assert coupler._pressure_anchor_cells is not None
    assert coupler._pressure_anchor_selection() is None
    assert coupler.fvm.calls == 1


def test_incompatible_vpm_viscosity_raises(bench, tmp_path):
    from source.coupler import FVMVPMCoupler

    class _FakeViscous:
        viscosity = 10 * bench.NU

    class _FakeVPMConfig:
        viscous = _FakeViscous()
        vpm_domain_bounds = None

    class _FakeVPM:
        config = _FakeVPMConfig()
        time_step_size = bench.DT_VPM

    setup = replace(bench.COUPLER_SETUP, case_dir=str(tmp_path))
    with pytest.raises(ValueError, match="viscosity"):
        FVMVPMCoupler._validate_injected_vpm(_FakeVPM(), setup, bench.FVM_BOX, bench.NU)


def test_incompatible_vpm_freestream_raises(bench, tmp_path):
    from source.coupler import FVMVPMCoupler

    class _FakeVPM:
        background_velocity = [0.5, 0.0, 0.0]
        time_step_size = bench.DT_VPM

    setup = replace(bench.COUPLER_SETUP, case_dir=str(tmp_path))
    with pytest.raises(ValueError, match="freestream"):
        FVMVPMCoupler._validate_injected_vpm(_FakeVPM(), setup, bench.FVM_BOX, bench.NU)


@pytest.mark.parametrize(
    ("scheme", "attr", "mode", "rejected"),
    [
        ("GBD", "gbd_threshold_mode", "relative_max", True),
        ("GBD", "gbd_threshold_mode", "budget", True),
        ("GBD", "gbd_threshold_mode", "absolute", True),
        ("GBD", "gbd_threshold_mode", "relative_local", False),
        ("DVH", "dvh_threshold_mode", "relative_max", True),
        ("DVH", "dvh_threshold_mode", "relative_local", False),
    ],
)
def test_coupling_requires_local_regen_threshold(bench, tmp_path, scheme, attr, mode, rejected):
    from source.coupler import FVMVPMCoupler

    class _FakeViscous:
        viscosity = bench.NU
        regen_radius_ratio = bench.COUPLER_SETUP.overlap_radius_ratio

    class _FakeVPMConfig:
        viscous = _FakeViscous()
        vpm_domain_bounds = None

    class _FakeVPM:
        config = _FakeVPMConfig()
        time_step_size = bench.DT_VPM

    _FakeViscous.scheme = scheme
    setattr(_FakeViscous, attr, mode)

    setup = replace(bench.COUPLER_SETUP, case_dir=str(tmp_path))
    if rejected:
        with pytest.raises(ValueError, match="relative_local"):
            FVMVPMCoupler._validate_injected_vpm(_FakeVPM(), setup, bench.FVM_BOX, bench.NU)
    else:
        FVMVPMCoupler._validate_injected_vpm(_FakeVPM(), setup, bench.FVM_BOX, bench.NU)
