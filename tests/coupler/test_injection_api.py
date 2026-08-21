"""Construction and native-solver contract tests for FVMVPMCoupler.

Exercises the wiring — solver adoption, dt/sub-cycle reconciliation from the
injected VPM step, master/non-master rank gating, and injected-solver
validation — with lightweight fakes so no GPU/Taichi runtime is needed.
The full end-to-end coupling loop is covered by the cubeFlow acceptance run.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from source.coupler import CouplerSetup, FVMVPMCoupler, create_coupler
from source.coupler.boundary import outflow_axis_sign


class _FakeParticles:
    n_particles = 0


class _FakeVPM:
    def __init__(
        self,
        time_step_size,
        domain_bounds=None,
        freestream_velocity=(1.0, 0.0, 0.0),
    ):
        self.time_step_size = float(time_step_size)
        self.particles = _FakeParticles()
        self._bg = None
        self._override = None
        self._freestream_velocity = np.asarray(freestream_velocity, dtype=float)
        self.setup = SimpleNamespace(
            particle_kernel="GAUSSIAN",
            domain_bounds=domain_bounds,
            viscous=SimpleNamespace(
                scheme="CS",
                kinematic_viscosity=1e-3,
                core_radius_ratio=1.0,
                dvh_threshold_mode="relative_local",
                gbd_threshold_mode="relative_local",
            ),
        )

    @property
    def freestream_velocity(self):
        return self._freestream_velocity

    def set_freestream_velocity(self, u):
        self._bg = u

    def set_velocity_override(self, fn):
        self._override = fn


class _FakeFVM:
    """Minimal native coupler-contract fake with no volume cells."""

    def __init__(self):
        from pathlib import Path

        self.case_dir = str(Path.cwd())
        self.ibm = None
        self.boundaries = []
        self.setup = SimpleNamespace(
            time=SimpleNamespace(time_step_size=0.02, end_time=1.0),
            transport=SimpleNamespace(kinematic_viscosity=1e-3, density=1.0),
            boundaries=[],
        )

    def n_procs(self):
        return 1

    def get_cell_centre_coordinates(self):
        return np.zeros((0, 3))

    def get_cell_volumes(self):
        return np.zeros((0,))

    def get_velocity_field(self):
        return np.zeros((0, 3))

    def set_cell_scalar_field(self, name, values):
        pass

    def set_cell_vector_field(self, name, vx, vy, vz):
        pass

    def get_boundary_face_centre_coordinates(self, patch):
        return np.array(
            [
                [-1.5, 0.0, 0.0],
                [1.5, 0.0, 0.0],
                [0.0, -1.5, 0.0],
                [0.0, 1.5, 0.0],
                [0.0, 0.0, -1.5],
                [0.0, 0.0, 1.5],
            ]
        )

    def get_boundary_face_normals(self, patch):
        return np.array(
            [
                [-1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, -1.0],
                [0.0, 0.0, 1.0],
            ]
        )

    def get_boundary_face_areas(self, patch):
        return np.ones(6)


@pytest.fixture(autouse=True)
def _stub_solver_info(monkeypatch):
    import source.coupler.solver as solver_mod

    monkeypatch.setattr(solver_mod, "_vpm_solver_info", lambda vpm: "")


def _make_config(**over):
    base = {
        "freestream_velocity": [1.0, 0.0, 0.0],
        "vpm_particle_spacing": 0.05,
        "authority_ramp_width": 0.3,
        "vpm_only_width": 0.2,
        "pressure_anchor_to_freestream": False,
    }
    base.update(over)
    return CouplerSetup(**base)


def test_use_injected_flag_and_adoption(monkeypatch, tmp_path):
    monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "0")
    monkeypatch.chdir(tmp_path)
    cfg = _make_config()
    vpm = _FakeVPM(time_step_size=0.1)
    fvm = _FakeFVM()

    c = FVMVPMCoupler(fvm, vpm, cfg)
    assert c._injected_fvm is fvm and c._injected_vpm is vpm

    c.initialize()
    # Solvers adopted, not rebuilt.
    assert c.fvm_solver is fvm
    assert c.vpm_solver is vpm
    # The native FVM configuration owns its sub-step and fluid properties.
    assert c.fvm_time_step_size == pytest.approx(0.02)
    assert c.vpm_time_step_size == pytest.approx(0.1)
    assert c.kinematic_viscosity == pytest.approx(1e-3)


def test_substep_count_derived_from_vpm_step(monkeypatch, tmp_path):
    """The FVM substep count follows the native VPM/FVM time-step ratio."""
    monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "0")
    monkeypatch.chdir(tmp_path)
    cfg = _make_config()
    c = FVMVPMCoupler(_FakeFVM(), _FakeVPM(time_step_size=0.06), cfg)
    c.initialize()
    assert c.fvm_time_step_size == pytest.approx(0.02)
    assert c.vpm_time_step_size == pytest.approx(0.06)
    assert c.fvm_substeps == 3


def test_master_requires_vpm(monkeypatch, tmp_path):
    monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "0")
    monkeypatch.chdir(tmp_path)
    c = FVMVPMCoupler(_FakeFVM(), None, _make_config())
    with pytest.raises(ValueError, match="vpm_solver is None on the master"):
        c.initialize()


def test_non_master_tolerates_none_vpm(monkeypatch, tmp_path):
    monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "1")
    monkeypatch.chdir(tmp_path)
    c = FVMVPMCoupler(_FakeFVM(), None, _make_config())
    assert c._is_master is False
    c.initialize()
    assert c.vpm_solver is None
    assert c.fvm_solver is not None


def test_physics_first_coupler_factory(monkeypatch):
    import source.coupler.solver as solver_mod

    setup = _make_config()
    fvm = object()
    vpm = object()
    result = object()
    captured = []
    monkeypatch.setattr(
        solver_mod,
        "FVMVPMCoupler",
        lambda fvm_solver, vpm_solver, value: (
            captured.append((fvm_solver, vpm_solver, value)) or result
        ),
    )

    assert create_coupler(fvm, vpm, setup) is result
    assert captured == [(fvm, vpm, setup)]


def test_coupler_uses_native_case_directory(monkeypatch, tmp_path):
    from source.solvers.FVM import FVMSetup

    class _NativeFVM:
        setup = FVMSetup(case_name="native")
        case_dir = str(tmp_path)

    coupler = create_coupler(_NativeFVM(), object(), CouplerSetup())
    assert coupler.case_dir == tmp_path


def test_vpm_factory_hides_distributed_root_ownership(
    monkeypatch,
    tmp_path,
):
    from source.solvers.VPM import VPMSetup, create_vpm_solver
    import source.solvers.VPM.factory as vpm_factory

    monkeypatch.setattr(
        vpm_factory,
        "_rank_and_size",
        lambda: (2, 4),
    )
    handle = create_vpm_solver(
        VPMSetup(log_mode="console"),
        case_dir=tmp_path,
    )

    assert handle is not None
    assert handle._openonda_inactive_rank
    assert handle.case_dir == tmp_path.resolve()


def test_positional_solver_setup_constructor(monkeypatch, tmp_path):
    """Preferred public API: vpm_solver + fvm_solver + CouplerSetup."""
    monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "0")
    monkeypatch.chdir(tmp_path)
    setup = CouplerSetup(
        freestream_velocity=[1.0, 0.0, 0.0],
        vpm_particle_spacing=0.05,
        authority_ramp_width=0.3,
        vpm_only_width=0.2,
        pressure_anchor_to_freestream=False,
    )
    vpm = _FakeVPM(time_step_size=0.1)
    fvm = _FakeFVM()

    c = FVMVPMCoupler(fvm, vpm, setup)

    assert c.setup is setup
    assert c._injected_fvm is fvm
    assert c._injected_vpm is vpm


def test_initialize_is_idempotent_and_solve_guard(monkeypatch, tmp_path):
    """The explicit build→initialize→solve flow: initialize() twice is a no-op,
    and solve() before initialize() fails loudly."""
    monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "0")
    monkeypatch.chdir(tmp_path)
    c = FVMVPMCoupler(_FakeFVM(), _FakeVPM(0.1), _make_config())
    with pytest.raises(RuntimeError, match="before initialize"):
        c.solve()

    c.initialize()
    transfer = c.vorticity_transfer
    c.initialize()  # second call: no-op
    assert c.vorticity_transfer is transfer


def test_injected_vpm_domain_must_contain_box(monkeypatch, tmp_path):
    """Safety validation: a VPM removal domain that does not enclose the FVM
    box would cull injected near-body particles — fail fast at initialize()."""
    monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "0")
    monkeypatch.chdir(tmp_path)
    cfg = _make_config()
    # Domain too small in x (only ±1.0): does not contain the ±1.5 box.
    bad_vpm = _FakeVPM(0.1, domain_bounds=(-1.0, 1.0, -2.0, 2.0, -2.0, 2.0))
    c = FVMVPMCoupler(_FakeFVM(), bad_vpm, cfg)
    with pytest.raises(ValueError, match="does not contain the FVM box"):
        c.initialize()

    # A domain that encloses the box passes.
    good_vpm = _FakeVPM(0.1, domain_bounds=(-2.0, 15.0, -2.0, 2.0, -2.0, 2.0))
    c2 = FVMVPMCoupler(_FakeFVM(), good_vpm, cfg)
    c2.initialize()  # no raise


def test_outflow_axis_sign_direction_agnostic(monkeypatch, tmp_path):
    """The outflow-face diagnostic follows freestream_velocity, not a hard-wired +x."""
    monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "0")
    monkeypatch.chdir(tmp_path)
    for u, expect in [
        ([1.0, 0, 0], (0, +1.0)),
        ([-2.0, 0, 0], (0, -1.0)),
        ([0, 3.0, 0], (1, +1.0)),
        ([0, 0, -1.0], (2, -1.0)),
    ]:
        assert outflow_axis_sign(np.asarray(u, dtype=np.float64)) == expect


def test_missing_fvm_solver_raises(monkeypatch, tmp_path):
    """Injection-only: constructing without an FVM solver fails fast."""
    monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "0")
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="requires an injected fvm_solver"):
        FVMVPMCoupler(None, _FakeVPM(0.1), _make_config())
