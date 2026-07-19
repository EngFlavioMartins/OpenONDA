"""Configuration parity and ownership for the cubeFlow benchmark (M5+M6).

Guards the single-source property of ``cube_benchmark_config``:

* every setting the two mathematical problems share is IDENTICAL between the
  hybrid FVM configuration and the reference FVM configuration — the cases
  may differ only in domain extension, far-field/coupling boundary
  conditions, and run horizon;
* the coupling setup carries NO solver-owned physics (density, viscosity,
  time stepping, mesh settings are None — owned by the injected solvers);
* the standalone VPM configuration is compatible with the FVM one (same
  fluid, same freestream, integer sub-cycling);
* the coupler raises on incompatible injected solvers instead of silently
  reconciling them.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
from pathlib import Path

import numpy as np
import pytest

_BENCH = Path(__file__).parents[2] / "tutorials/coupled_FVM_VPM/cubeFlow/cube_benchmark_config.py"


@pytest.fixture(scope="module")
def bench():
    spec = importlib.util.spec_from_file_location("cube_benchmark_config", _BENCH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def hybrid_solver(bench, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("hybrid_parity")
    with contextlib.redirect_stdout(io.StringIO()):
        return bench.make_hybrid_fvm_solver(tmp, quiet=True)


def test_common_fvm_settings_identical(bench, hybrid_solver):
    """Schemes, linear solvers, PIMPLE controls, transport, force reporting,
    dt, write cadence, execution backend, and the initial field are common
    settings and must be byte-identical between the two cases."""
    hybrid = hybrid_solver.config
    reference = bench.make_reference_fvm_config()

    assert hybrid.schemes == reference.schemes
    assert hybrid.linear == reference.linear
    assert hybrid.pimple == reference.pimple
    assert hybrid.transport == reference.transport
    assert hybrid.forces == reference.forces
    assert hybrid.execution == reference.execution
    assert hybrid.time.delta_t == reference.time.delta_t
    assert hybrid.time.write_interval_time == reference.time.write_interval_time
    assert hybrid.time.adjust_timestep is False
    assert reference.time.adjust_timestep is False
    assert hybrid.initial_U == reference.initial_U
    assert hybrid.turbulence == reference.turbulence


def test_wall_boundary_identical(bench, hybrid_solver):
    """The cube wall BC is part of the common problem."""

    def wall(cfg):
        (b,) = [b for b in cfg.boundaries if b.name == bench.WALL_PATCH]
        return b

    assert wall(hybrid_solver.config) == wall(bench.make_reference_fvm_config())


def test_legitimate_differences_are_the_only_differences(bench, hybrid_solver):
    """Beyond the wall patch, the hybrid has exactly one coupling patch and
    the reference exactly the six far-field patches; the run horizons cover
    the same comparison window."""
    hybrid_names = {b.name for b in hybrid_solver.config.boundaries}
    reference_names = {b.name for b in bench.make_reference_fvm_config().boundaries}
    assert hybrid_names == {"numericalBoundary", bench.WALL_PATCH}
    assert reference_names == {"inlet", "outlet", "ymin", "ymax", "zmin", "zmax", bench.WALL_PATCH}
    assert bench.T_END_REFERENCE >= bench.T_END_HYBRID  # superset window


def test_coupler_setup_owns_no_solver_physics(bench, tmp_path):
    """CouplerSetup must not duplicate solver-owned values (spec Phase 5)."""
    setup = bench.make_coupler_setup(tmp_path)
    for name in ("nu", "rho", "dt", "t_end", "fvm_box", "grid_spacing", "initial_U"):
        assert getattr(setup, name) is None, f"CouplerSetup.{name} duplicates solver config"
    assert not setup.surface, "body geometry is owned by the mesh, not the coupling setup"


def test_vpm_config_compatible(bench, tmp_path):
    vpm = bench.make_vpm_config(tmp_path)
    assert vpm.viscous.viscosity == pytest.approx(bench.NU)
    assert tuple(vpm.background_velocity) == tuple(bench.U_INF)
    ratio = vpm.time_step_size / bench.DT
    assert ratio == pytest.approx(round(ratio)), "VPM step must be an integer multiple of DT"
    # The VPM domain must contain the hybrid FVM box (injection survival).
    dom = np.asarray(vpm.vpm_domain_bounds, dtype=float)
    box = np.asarray(bench.CORE_BOX, dtype=float)
    assert np.all(dom[::2] <= box[::2]) and np.all(dom[1::2] >= box[1::2])


def test_coupler_adopts_and_validates_hybrid_solver(bench, hybrid_solver, tmp_path):
    """End-to-end ownership: with the benchmark's coupling-only setup the
    coupler adopts dt/t_end/nu from the FVM solver and derives the box from
    the coupling patch; a contradicting setup raises."""
    from source.coupler import FVMVPMCoupler

    setup = bench.make_coupler_setup(tmp_path)
    coupler = FVMVPMCoupler(object(), hybrid_solver, setup)
    coupler.ofw = hybrid_solver
    coupler._resolve_eulerian_ownership()
    assert coupler.dt_fvm == pytest.approx(bench.DT)
    assert coupler.t_end == pytest.approx(bench.T_END_HYBRID)
    assert setup.nu == pytest.approx(bench.NU)
    assert np.allclose(setup.fvm_box, bench.CORE_BOX, atol=1e-12)

    bad = bench.make_coupler_setup(tmp_path)
    bad.nu = 2 * bench.NU
    coupler_bad = FVMVPMCoupler(object(), hybrid_solver, bad)
    coupler_bad.ofw = hybrid_solver
    with pytest.raises(ValueError, match="owns this value"):
        coupler_bad._resolve_eulerian_ownership()


def test_incompatible_vpm_viscosity_raises(bench, hybrid_solver, tmp_path):
    """Shared physical properties are compatibility-checked at coupler
    construction (spec Phase 5): mismatched viscosity is a hard error."""
    from source.coupler import FVMVPMCoupler

    class _FakeViscous:
        viscosity = 10 * bench.NU

    class _FakeVPMConfig:
        viscous = _FakeViscous()
        vpm_domain_bounds = None

    class _FakeVPM:
        config = _FakeVPMConfig()
        time_step_size = bench.DT_VPM

    setup = bench.make_coupler_setup(tmp_path)
    with pytest.raises(ValueError, match="viscosity"):
        FVMVPMCoupler._validate_injected_vpm(_FakeVPM(), setup, bench.CORE_BOX, bench.NU)


def test_incompatible_vpm_freestream_raises(bench, tmp_path):
    from source.coupler import FVMVPMCoupler

    class _FakeVPM:
        background_velocity = [0.5, 0.0, 0.0]  # != U_INF
        time_step_size = bench.DT_VPM

    setup = bench.make_coupler_setup(tmp_path)
    with pytest.raises(ValueError, match="freestream"):
        FVMVPMCoupler._validate_injected_vpm(_FakeVPM(), setup, bench.CORE_BOX, bench.NU)
