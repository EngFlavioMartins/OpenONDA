"""End-to-end native FVM/VPM smoke test on a uniform freestream."""

from __future__ import annotations

import json

import numpy as np
import pytest

taichi = pytest.importorskip("taichi", reason="VPM requires taichi")

DT_FVM = 0.05
DT_VPM = 0.15
H = 0.125


@pytest.mark.slow
def test_coupled_fvm_vpm_two_steps(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    from source.coupler import CouplerSetup, FVMVPMCoupler
    from source.solvers.FVM import (
        BoundaryConfig,
        FVMSetup,
        TimeConfig,
        TransportConfig,
    )
    from source.solvers.FVM import (
        Solver as FVM_Solver,
    )
    from source.solvers.FVM.mesh.rectilinear import coupling_box_mesh
    from source.solvers.VPM import Solver as VPM_Solver
    from source.solvers.VPM import VPMSetup

    # Coupling-only setup: physics/time/mesh are owned by the injected solvers.
    setup = CouplerSetup(
        freestream_velocity=[1.0, 0.0, 0.0],
        vpm_particle_spacing=H,
        overlap_zone_ramp_width=2 * H,
        overlap_zone_dead_zone_width=H,
    )

    vpm = VPM_Solver(
        VPMSetup(
            time_step_size=DT_VPM,
            processing_unit="CPU",
            max_particles=50_000,
            vpm_domain_bounds=[-1.0, 1.0, -1.0, 1.0, -1.0, 1.0],
            freestream_velocity=[1.0, 0.0, 0.0],
        )
    )

    def make_fvm():
        config = FVMSetup(
            case_name="coupled_smoke",
            time=TimeConfig(delta_t=DT_FVM, end_time=2 * DT_VPM),
            transport=TransportConfig(nu=0.01),
            boundaries=[
                BoundaryConfig(
                    name="numericalBoundary",
                    type_velocity="fixedValue",
                    value_velocity=setup.freestream_velocity,
                    type_p="fixedFluxPressure",
                )
            ],
            initial_velocity=setup.freestream_velocity,
        )
        return FVM_Solver(
            config,
            case_dir=".",
            mesh_data=coupling_box_mesh((-0.5, 0.5, -0.5, 0.5, -0.5, 0.5), H),
        )

    fvm = make_fvm()
    coupler = FVMVPMCoupler(vpm, fvm, setup)
    coupler.run()

    # Sub-cycling derived from the two native solver time steps.
    assert coupler.n_fvm_substeps == 3
    assert coupler.dt_vpm == pytest.approx(DT_VPM)

    # Uniform inflow, no body → the FVM field stays finite and uniform.
    U = np.asarray(fvm.get_velocity_field())
    assert np.isfinite(U).all()
    assert np.allclose(U.mean(axis=0), setup.freestream_velocity_vector, atol=1e-6)
    # Two coupling steps × 3 sub-steps were committed.
    assert fvm.time_step == 6
    assert fvm.flow_time == pytest.approx(2 * DT_VPM)

    # Impulsive start with zero interior vorticity: hand-off ran, no particles.
    assert vpm.particles.number_of_particles == 0

    # Native FVM and coupler diagnostics are written independently.
    sol = tmp_path / "solution"
    assert (sol / "fvm.log").exists()
    coupler_log = (sol / "coupler.log").read_text()
    assert "FVM-VPM COUPLED SOLVER" in coupler_log
    assert coupler_log.count("[Transfer]") >= 2
    assert "n_fvm_substeps=3" in coupler_log
    checkpoint = sol / "checkpoint"
    manifest = json.loads((checkpoint / "manifest.json").read_text())
    assert manifest["format_version"] == 4
    assert manifest["kind"] == "openonda.coupled_checkpoint"
    assert all((checkpoint / name).is_file() for name in manifest["artifacts"].values())
    assert manifest["artifacts"] == {
        "fvm": "fvm_000002.npz",
        "vpm": "vpm_000002.h5",
        "vpm_bc": "vpm_bc_000002.npz",
    }
    assert not list(checkpoint.glob("*_000001*"))

    expected_u = fvm.U.copy()
    expected_p = fvm.p.copy()
    expected_phi = fvm.phi.copy()
    restored_vpm = VPM_Solver(
        VPMSetup(
            time_step_size=DT_VPM,
            processing_unit="CPU",
            max_particles=50_000,
            vpm_domain_bounds=[-1.0, 1.0, -1.0, 1.0, -1.0, 1.0],
            freestream_velocity=[1.0, 0.0, 0.0],
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
    # No external solver case artifacts were created anywhere.
    assert not (tmp_path / "constant").exists()
    assert not (tmp_path / "system").exists()
