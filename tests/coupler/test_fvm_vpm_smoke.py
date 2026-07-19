"""End-to-end native FVM/VPM smoke test on a uniform freestream."""

from __future__ import annotations

import numpy as np
import pytest

taichi = pytest.importorskip("taichi", reason="VPM requires taichi")

DT_FVM = 0.05
DT_VPM = 0.15  # period_multiplier = 3
H = 0.125


@pytest.mark.slow
def test_coupled_fvm_vpm_two_steps(tmp_path, monkeypatch):
    # The coupler roots itself at the CWD (case_dir = Path(".").absolute()).
    monkeypatch.chdir(tmp_path)

    from source.coupler import CouplerSetup, FVMVPMCoupler
    from source.coupler.core.helpers.fvm_backend import build_fvm_backend, coupling_box_mesh
    from source.solvers.VPM import Solver as VPM_Solver
    from source.solvers.VPM import SolverConfig

    # Coupling-only setup: physics/time/mesh are owned by the injected solvers.
    setup = CouplerSetup(
        backend="fvm",
        u_inf=[1.0, 0.0, 0.0],
        h=H,
        buffer_thickness=2 * H,
        dead_zone_h=1.0,
        wall_patch_name=None,
        case_dir=".",
    )

    vpm = VPM_Solver(
        SolverConfig(
            time_step_size=DT_VPM,
            processing_unit="CPU",
            max_particles=50_000,
            vpm_domain_bounds=[-1.0, 1.0, -1.0, 1.0, -1.0, 1.0],
            background_velocity=[1.0, 0.0, 0.0],
        )
    )

    def make_fvm():
        return build_fvm_backend(
            mesh_data=coupling_box_mesh((-0.5, 0.5, -0.5, 0.5, -0.5, 0.5), H),
            case_dir=".",
            dt=DT_FVM,
            t_end=2 * DT_VPM,  # two coupling steps
            nu=0.01,
            u_inf=setup.u_inf,
            quiet=True,
        )

    FVMVPMCoupler.prepare_case(setup, vpm_solver=vpm)
    fvm = make_fvm()
    coupler = FVMVPMCoupler(vpm, fvm, setup)
    coupler.run()

    # Sub-cycling derived from the two native solver time steps.
    assert coupler.period_multiplier == 3
    assert coupler.dt == pytest.approx(DT_VPM)

    # Uniform inflow, no body → the FVM field stays finite and uniform.
    U = np.asarray(fvm.get_velocity_field())
    assert np.isfinite(U).all()
    assert np.allclose(U.mean(axis=0), setup.U_inf, atol=1e-6)
    # Two coupling steps × 3 sub-steps were committed.
    assert fvm.time_step == 6
    assert fvm.flow_time == pytest.approx(2 * DT_VPM)

    # Impulsive start with zero interior vorticity: hand-off ran, no particles.
    assert vpm.particles.number_of_particles == 0

    # Backend-named logs: fvm.log (not ofw.log) + the coupler diagnostics.
    sol = tmp_path / "solution"
    assert (sol / "fvm.log").exists()
    assert not (sol / "ofw.log").exists()
    coupler_log = (sol / "coupler.log").read_text()
    assert "FVM-VPM COUPLED SOLVER" in coupler_log
    assert coupler_log.count("[Inject]") == 2
    assert "period_multiplier=3" in coupler_log
    checkpoint = sol / "coupled_checkpoint"
    assert (checkpoint / "manifest.json").is_file()
    assert (checkpoint / "fvm.npz").is_file()
    assert (checkpoint / "vpm_latest.h5").is_file()
    assert (checkpoint / "donor_state.npz").is_file()

    expected_u = fvm.U.copy()
    expected_p = fvm.p.copy()
    expected_phi = fvm.phi.copy()
    restored_vpm = VPM_Solver(
        SolverConfig(
            time_step_size=DT_VPM,
            processing_unit="CPU",
            max_particles=50_000,
            vpm_domain_bounds=[-1.0, 1.0, -1.0, 1.0, -1.0, 1.0],
            background_velocity=[1.0, 0.0, 0.0],
        )
    )
    restored_fvm = make_fvm()
    restored = FVMVPMCoupler(restored_vpm, restored_fvm, setup)
    restored.initialize()
    restored_step = restored.load_state(checkpoint)

    assert restored_step == 2
    assert restored_fvm.time_step == 6
    assert restored_vpm.time_step == 2
    np.testing.assert_allclose(restored_fvm.U, expected_u, rtol=0.0, atol=1e-13)
    np.testing.assert_allclose(restored_fvm.p, expected_p, rtol=0.0, atol=1e-13)
    np.testing.assert_allclose(restored_fvm.phi, expected_phi, rtol=0.0, atol=1e-13)
    # No OpenFOAM case artifacts were created anywhere.
    assert not (tmp_path / "constant").exists()
    assert not (tmp_path / "system").exists()
