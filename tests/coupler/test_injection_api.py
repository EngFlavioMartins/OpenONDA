"""Dependency-injection construction of FVMVPMCoupler (from_solvers).

Exercises the wiring — solver adoption, dt/sub-cycle reconciliation from the
injected VPM step, master/non-master rank gating, and injected-solver
validation — with lightweight fakes so no GPU/Taichi or OpenFOAM is needed.
The full end-to-end coupling loop is covered by the cubeFlow acceptance run.
"""

from __future__ import annotations

import numpy as np
import pytest

from source.coupler import CouplerSetup, FVMVPMCoupler


# ---------------------------------------------------------------------------
# Lightweight fakes satisfying just the contract initialize() touches
# ---------------------------------------------------------------------------
class _FakeParticles:
    number_of_particles = 0
    _np_float_dtype = np.float32


class _FakeVPM:
    def __init__(self, time_step_size, vpm_domain_bounds=None, background_velocity=None):
        self.time_step_size = float(time_step_size)
        self.particles = _FakeParticles()
        self._bg = None
        self._override = None
        # Optional attributes the injection validator inspects.
        if vpm_domain_bounds is not None:
            self.vpm_domain_bounds = vpm_domain_bounds
        if background_velocity is not None:
            self.background_velocity = background_velocity

    def set_background_velocity(self, u):
        self._bg = u

    def set_velocity_override(self, fn):
        self._override = fn


class _FakeFVM:
    """Minimal OFW-contract fake: geometry getters return empty (serial, no
    cells) so injector/fringe setup runs without a mesh."""

    def __init__(self):
        self._dt = None
        self._nu = None

    def n_procs(self):
        return 1

    def set_time_step(self, dt):
        self._dt = dt

    def set_kinematic_viscosity(self, nu):
        self._nu = nu

    def get_cell_center_coordinates(self):
        return np.zeros((0, 3))

    def get_cell_volumes(self):
        return np.zeros((0,))

    def get_velocity_field(self):
        return np.zeros((0, 3))

    def set_cell_scalar_field(self, name, values):
        pass

    def set_cell_vector_field(self, name, vx, vy, vz):
        pass


@pytest.fixture(autouse=True)
def _stub_solver_info(monkeypatch):
    import source.coupler.core.solver as solver_mod

    monkeypatch.setattr(solver_mod, "_vpm_solver_info", lambda vpm: "")


def _make_config(**over):
    base = {
        "u_inf": [1.0, 0.0, 0.0],
        "nu": 1e-3,
        "dt": 0.02,
        "t_end": 1.0,
        "fvm_box": (-1.5, 1.5, -1.5, 1.5, -1.5, 1.5),
        "h": 0.05,
        "buffer_thickness": 0.3,
        "dead_zone_h": 4.0,
        "overlap_velocity_forcing": False,
    }
    base.update(over)
    return CouplerSetup(**base)


# ---------------------------------------------------------------------------
def test_use_injected_flag_and_adoption(monkeypatch, tmp_path):
    monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "0")
    monkeypatch.chdir(tmp_path)
    cfg = _make_config()
    vpm = _FakeVPM(time_step_size=0.1)
    fvm = _FakeFVM()

    c = FVMVPMCoupler.from_solvers(cfg, fvm_solver=fvm, vpm_solver=vpm)
    assert c._injected_fvm is fvm and c._injected_vpm is vpm

    c.initialize()
    # Solvers adopted, not rebuilt.
    assert c.ofw is fvm
    assert c.vpm is vpm
    # Runtime setters stamped the reconciled FVM SUB-step + viscosity.
    # (cfg.dt is mutated to the coupling step dt_vpm during initialize; the
    # FVM marches on dt_fvm = 0.02.)
    assert fvm._dt == pytest.approx(0.02)  # dt_fvm
    assert c.dt_vpm == pytest.approx(0.1)
    assert fvm._nu == pytest.approx(1e-3)


def test_substep_count_derived_from_vpm_step(monkeypatch, tmp_path):
    """period_multiplier is derived from the injected VPM time_step_size / dt_fvm."""
    monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "0")
    monkeypatch.chdir(tmp_path)
    cfg = _make_config(dt=0.02)
    c = FVMVPMCoupler.from_solvers(
        cfg, fvm_solver=_FakeFVM(), vpm_solver=_FakeVPM(time_step_size=0.06)
    )
    c.initialize()
    assert c.dt_fvm == pytest.approx(0.02)
    assert c.dt_vpm == pytest.approx(0.06)
    assert c.period_multiplier == 3  # round(0.06 / 0.02)


def test_master_requires_vpm(monkeypatch, tmp_path):
    monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "0")
    monkeypatch.chdir(tmp_path)
    c = FVMVPMCoupler.from_solvers(_make_config(), fvm_solver=_FakeFVM(), vpm_solver=None)
    with pytest.raises(ValueError, match="vpm_solver is None on the master"):
        c.initialize()


def test_non_master_tolerates_none_vpm(monkeypatch, tmp_path):
    monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "1")
    monkeypatch.chdir(tmp_path)
    c = FVMVPMCoupler.from_solvers(_make_config(), fvm_solver=_FakeFVM(), vpm_solver=None)
    assert c._is_master is False
    c.initialize()
    assert c.vpm is None
    assert c.ofw is not None


def test_is_master_rank(monkeypatch):
    monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "0")
    assert FVMVPMCoupler.is_master_rank() is True
    monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "3")
    assert FVMVPMCoupler.is_master_rank() is False


def test_config_has_no_vpm_physics_fields():
    """Clean break: the coupling config must not carry VPM solver-build params
    (they live in the injected SolverConfig now)."""
    cfg = _make_config()
    for gone in (
        "viscous_scheme",
        "stretching_scheme",
        "advection_scheme",
        "les_smagorinsky_cs",
        "treecode_theta",
        "max_particles",
        "vpm_domain",
        "particles_kernel",
        "precision",
        "stabilization",
        "samplers",
        "eulerian_backend",
        "period_multiplier",
    ):
        assert not hasattr(cfg, gone), f"{gone} should have moved to SolverConfig"


def test_public_api_exposes_coupler_setup_not_vpm_configs():
    """The coupler package should not be a convenience export point for VPM
    solver-build classes; those belong to source.solvers.VPM.config.types."""
    import source.coupler as coupler_pkg

    assert not hasattr(coupler_pkg, "ViscousConfig")
    assert not hasattr(coupler_pkg, "StabilizationConfig")


def test_positional_solver_setup_constructor(monkeypatch, tmp_path):
    """Preferred public API: vpm_solver + fvm_solver + CouplerSetup."""
    monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "0")
    monkeypatch.chdir(tmp_path)
    setup = CouplerSetup(
        u_inf=[1.0, 0.0, 0.0],
        nu=1e-3,
        dt=0.02,
        t_end=1.0,
        fvm_box=(-1.5, 1.5, -1.5, 1.5, -1.5, 1.5),
        h=0.05,
        buffer_thickness=0.3,
        dead_zone_h=4.0,
        overlap_velocity_forcing=False,
    )
    vpm = _FakeVPM(time_step_size=0.1)
    fvm = _FakeFVM()

    c = FVMVPMCoupler(vpm, fvm, setup)

    assert c.coupler_setup is setup
    assert c._injected_fvm is fvm
    assert c._injected_vpm is vpm


def test_initialize_is_idempotent_and_solve_guard(monkeypatch, tmp_path):
    """The explicit build→initialize→solve flow: initialize() twice is a no-op,
    and solve() before initialize() fails loudly."""
    monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "0")
    monkeypatch.chdir(tmp_path)
    c = FVMVPMCoupler.from_solvers(_make_config(), fvm_solver=_FakeFVM(), vpm_solver=_FakeVPM(0.1))
    with pytest.raises(RuntimeError, match="before initialize"):
        c.solve()

    c.initialize()
    inj = c.injector
    c.initialize()  # second call: no-op
    assert c.injector is inj  # not rebuilt


def test_injected_vpm_domain_must_contain_box(monkeypatch, tmp_path):
    """Safety validation: a VPM removal domain that does not enclose the FVM
    box would cull injected near-body particles — fail fast at initialize()."""
    monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "0")
    monkeypatch.chdir(tmp_path)
    cfg = _make_config(fvm_box=(-1.5, 1.5, -1.5, 1.5, -1.5, 1.5))
    # Domain too small in x (only ±1.0): does not contain the ±1.5 box.
    bad_vpm = _FakeVPM(0.1, vpm_domain_bounds=(-1.0, 1.0, -2.0, 2.0, -2.0, 2.0))
    c = FVMVPMCoupler.from_solvers(cfg, fvm_solver=_FakeFVM(), vpm_solver=bad_vpm)
    with pytest.raises(ValueError, match="does not contain the FVM box"):
        c.initialize()

    # A domain that encloses the box passes.
    good_vpm = _FakeVPM(0.1, vpm_domain_bounds=(-2.0, 15.0, -2.0, 2.0, -2.0, 2.0))
    c2 = FVMVPMCoupler.from_solvers(cfg, fvm_solver=_FakeFVM(), vpm_solver=good_vpm)
    c2.initialize()  # no raise


def test_outflow_axis_sign_direction_agnostic(monkeypatch, tmp_path):
    """The outflow-face diagnostic follows u_inf, not a hard-wired +x."""
    monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "0")
    monkeypatch.chdir(tmp_path)
    for u, expect in [
        ([1.0, 0, 0], (0, +1.0)),
        ([-2.0, 0, 0], (0, -1.0)),
        ([0, 3.0, 0], (1, +1.0)),
        ([0, 0, -1.0], (2, -1.0)),
    ]:
        c = FVMVPMCoupler.from_solvers(
            _make_config(u_inf=u), fvm_solver=_FakeFVM(), vpm_solver=_FakeVPM(0.1)
        )
        assert c._outflow_axis_sign() == expect


def test_missing_fvm_solver_raises(monkeypatch, tmp_path):
    """Injection-only: constructing without an FVM solver fails fast."""
    monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "0")
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="requires an injected fvm_solver"):
        FVMVPMCoupler(_FakeVPM(0.1), None, _make_config())


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
