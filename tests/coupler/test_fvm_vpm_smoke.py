"""End-to-end native FVM/VPM smoke test on a uniform freestream."""

from __future__ import annotations

import json

import numpy as np
import pytest

taichi = pytest.importorskip("taichi", reason="VPM requires taichi")

FVM_TIME_STEP_SIZE = 0.05
VPM_TIME_STEP_SIZE = 0.15
H = 0.125


@pytest.mark.slow
def test_coupled_fvm_vpm_two_steps(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    from source.coupler import CouplerSetup, FVMVPMCoupler
    from source.solvers.FVM import (
        BoundaryConfig,
        FVMSetup,
        FVMSolver,
        TimeConfig,
        TransportConfig,
    )
    from source.solvers.FVM.mesh.rectilinear import coupling_box_mesh
    from source.solvers.VPM import VPMSetup, VPMSolver

    # Coupling-only setup: physics/time/mesh are owned by the injected solvers.
    setup = CouplerSetup(
        freestream_velocity=[1.0, 0.0, 0.0],
        vpm_particle_spacing=H,
        overlap_zone_ramp_width=2 * H,
        overlap_zone_dead_zone_width=H,
    )

    vpm = VPMSolver(
        VPMSetup(
            time_step_size=VPM_TIME_STEP_SIZE,
            processing_unit="CPU",
            max_particles=50_000,
            vpm_domain_bounds=[-1.0, 1.0, -1.0, 1.0, -1.0, 1.0],
            freestream_velocity=[1.0, 0.0, 0.0],
        )
    )

    def make_fvm():
        config = FVMSetup(
            case_name="coupled_smoke",
            time=TimeConfig(time_step_size=FVM_TIME_STEP_SIZE, end_time=2 * VPM_TIME_STEP_SIZE),
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
        return FVMSolver(
            config,
            case_dir=".",
            mesh_data=coupling_box_mesh((-0.5, 0.5, -0.5, 0.5, -0.5, 0.5), H),
        )

    fvm = make_fvm()
    coupler = FVMVPMCoupler(fvm, vpm, setup)
    coupler.run()

    # Sub-cycling derived from the two native solver time steps.
    assert coupler.fvm_substeps == 3
    assert coupler.vpm_time_step_size == pytest.approx(VPM_TIME_STEP_SIZE)

    # Uniform inflow, no body → the FVM field stays finite and uniform.
    U = np.asarray(fvm.get_velocity_field())
    assert np.isfinite(U).all()
    assert np.allclose(U.mean(axis=0), setup.freestream_velocity_vector, atol=1e-6)
    # Two coupling steps × 3 sub-steps were committed.
    assert fvm.step == 6
    assert fvm.time == pytest.approx(2 * VPM_TIME_STEP_SIZE)

    # Impulsive start with zero interior vorticity: hand-off ran, no particles.
    assert vpm.particles.number_of_particles == 0

    # Native FVM and coupler diagnostics are written independently.
    sol = tmp_path / "solution"
    assert (sol / "fvm.log").exists()
    coupler_log = (sol / "coupler.log").read_text()
    assert "FVM-VPM COUPLED SOLVER" in coupler_log
    assert coupler_log.count("[Transfer]") >= 2
    assert "fvm_substeps=3" in coupler_log
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
    restored_vpm = VPMSolver(
        VPMSetup(
            time_step_size=VPM_TIME_STEP_SIZE,
            processing_unit="CPU",
            max_particles=50_000,
            vpm_domain_bounds=[-1.0, 1.0, -1.0, 1.0, -1.0, 1.0],
            freestream_velocity=[1.0, 0.0, 0.0],
        )
    )
    restored_fvm = make_fvm()
    restored = FVMVPMCoupler(restored_fvm, restored_vpm, setup)
    restored.initialize()
    restored_step = restored.load_state(checkpoint)

    assert restored_step == 2
    assert restored_fvm.step == 6
    assert restored_vpm.step == 2
    np.testing.assert_allclose(restored_fvm.U, expected_u, rtol=0.0, atol=1e-13)
    np.testing.assert_allclose(restored_fvm.p, expected_p, rtol=0.0, atol=1e-13)
    np.testing.assert_allclose(restored_fvm.phi, expected_phi, rtol=0.0, atol=1e-13)
    # No external solver case artifacts were created anywhere.
    assert not (tmp_path / "constant").exists()
    assert not (tmp_path / "system").exists()
