"""End-to-end native FVM/VPM smoke test on a uniform freestream."""

from __future__ import annotations

import json
import shutil

import numpy as np
import pytest

FVM_TIME_STEP_SIZE = 0.05
VPM_TIME_STEP_SIZE = 0.15
H = 0.125


def test_coupled_fvm_vpm_two_steps(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    from source.coupler import CouplerSetup, FVMVPMCoupler
    from source.solvers.fvm import (
        BoundaryConfig,
        FVMSetup,
        FVMSolver,
        TimeConfig,
        TransportConfig,
    )
    from source.solvers.fvm.mesh.rectilinear import coupling_box_mesh
    from source.solvers.vpm import Numerics, ViscousConfig, VPMCase, VPMSolver

    # Coupling-only setup: physics/time/mesh are owned by the injected solvers.
    setup = CouplerSetup(
        freestream_velocity=[1.0, 0.0, 0.0],
        eta_blend_width=0.0,
        backup_interval_steps=2,
    )

    def make_vpm():
        numerics = Numerics(
            time_step_size=VPM_TIME_STEP_SIZE,
            compute_device="CPU",
            max_n_particles=50_000,
            domain_bounds=(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0),
            freestream_velocity=(1.0, 0.0, 0.0),
            viscous=ViscousConfig.cs(kinematic_viscosity=0.01, particle_spacing=H),
        )
        return VPMSolver(VPMCase(numerics=numerics, directory=tmp_path)), numerics

    def make_fvm(case_dir="."):
        config = FVMSetup(
            case_name="coupled_smoke",
            time=TimeConfig(time_step_size=FVM_TIME_STEP_SIZE, end_time=2 * VPM_TIME_STEP_SIZE),
            transport=TransportConfig(kinematic_viscosity=0.01),
            boundaries=[
                BoundaryConfig(
                    name="numericalBoundary",
                    velocity_type="fixedValue",
                    velocity_value=setup.freestream_velocity,
                    pressure_type="fixedFluxPressure",
                )
            ],
            initial_velocity=setup.freestream_velocity,
        )
        solver = FVMSolver(
            config,
            case_dir=case_dir,
            mesh_data=coupling_box_mesh((-0.5, 0.5, -0.5, 0.5, -0.5, 0.5), H),
        )
        return solver, config

    vpm, vpm_setup = make_vpm()
    fvm, fvm_config = make_fvm()
    coupler = FVMVPMCoupler(fvm, vpm, setup)

    # Solver/coupler owners store their *Setup object as ``self.setup``.
    assert fvm.setup is fvm_config
    assert vpm.numerics is vpm_setup
    assert coupler.setup is setup
    assert not hasattr(coupler, "config")

    first_stop = coupler.run(max_coupling_steps=1, backup_at_stop=True)
    assert first_stop == 1
    first_backup = tmp_path / "solution" / "backups"
    first_manifest = json.loads((first_backup / "manifest.json").read_text())
    assert first_manifest["coupling_step"] == 1
    assert all("000001" in name for name in first_manifest["artifacts"].values())
    seed_backup = tmp_path / "seed_backup"
    shutil.copytree(first_backup, seed_backup)
    first_metadata = json.loads((tmp_path / "solution" / "run_metadata.json").read_text())
    assert first_metadata["execution"] == {
        "start_coupling_step": 0,
        "stop_coupling_step": 1,
        "configured_end_coupling_step": 2,
        "start_time": 0.0,
        "stop_time": pytest.approx(VPM_TIME_STEP_SIZE),
        "is_limited": True,
    }

    second_stop = coupler.solve(
        start_step=first_stop,
        max_coupling_steps=1,
        backup_at_stop=True,
    )
    assert second_stop == 2

    # Sub-cycling derived from the two native solver time steps.
    assert coupler.n_fvm_substeps == 3
    assert coupler.vpm_time_step_size == pytest.approx(VPM_TIME_STEP_SIZE)

    # Uniform inflow, no body → the FVM field stays finite and uniform.
    velocity = np.asarray(fvm.get_velocity_field())
    assert np.isfinite(velocity).all()
    assert np.allclose(velocity.mean(axis=0), setup.freestream_velocity_vector, atol=1e-6)
    # Two coupling steps × 3 sub-steps were committed.
    assert fvm.step == 6
    assert fvm.time == pytest.approx(2 * VPM_TIME_STEP_SIZE)

    # Impulsive start with zero interior vorticity: hand-off ran, no particles.
    assert vpm.particles.n_particles_total == 0

    # Native FVM and coupler diagnostics are written independently.
    sol = tmp_path / "solution"
    assert (sol / "fvm.log").exists()
    vpm_log = (sol / "vpm.log").read_text()
    assert "fvm      step" not in vpm_log
    assert vpm_log.count("VPM TIME STEP 1") == 1
    # The VPM startup report is written once, not repeated by the coupler.
    assert vpm_log.count("VPM SOLVER CONFIGURATION") == 1
    coupler_log = (sol / "coupler.log").read_text()
    assert "coupler  run" in coupler_log
    # Initial synchronization plus one absolute replacement after each FVM interval.
    assert coupler_log.count("coupler  state replacement") == 3
    assert "fvm substeps per coupling step" in coupler_log
    backup = sol / "backups"
    manifest = json.loads((backup / "manifest.json").read_text())
    assert manifest["format_version"] == 11
    assert manifest["kind"] == "openonda.coupled_backup"
    assert all((backup / name).is_file() for name in manifest["artifacts"].values())
    assert manifest["artifacts"] == {
        "fvm": "fvm_000002.npz",
        "vpm": "vpm_000002.h5",
        "vpm_xdmf": "vpm_000002.xdmf",
        "vpm_boundary_condition": "vpm_boundary_condition_000002.npz",
    }
    assert set(manifest["artifact_sha256"]) == set(manifest["artifacts"])
    assert not list(backup.glob("*_000001*"))
    assert sorted(path.name for path in sol.glob("vpm_*.h5")) == [
        "vpm_000001.h5",
        "vpm_000002.h5",
    ]
    assert sorted(path.name for path in sol.glob("vpm_*.xdmf")) == [
        "vpm_000001.xdmf",
        "vpm_000002.xdmf",
    ]

    expected_u = fvm.velocity.copy()
    expected_p = fvm.kinematic_pressure.copy()
    expected_flux = fvm.volumetric_face_flux.copy()
    restart_case = tmp_path / "restart"
    restored_vpm, _ = make_vpm()
    restored_fvm, _ = make_fvm(restart_case)
    restored = FVMVPMCoupler(restored_fvm, restored_vpm, setup)
    restored_step = restored.run(
        restart_from=seed_backup,
        max_coupling_steps=1,
        backup_at_stop=True,
    )

    assert restored_step == 2
    assert restored.vorticity_transfer is not None
    # Initial synchronization is transfer 1; the saved step-1 state is transfer
    # 2, and the first resumed replacement must therefore be transfer 3.
    assert restored.vorticity_transfer.step == restored_step + 1 == 3
    assert restored_fvm.step == 6
    assert restored_vpm.step == 2
    assert restored_vpm.time == pytest.approx(vpm.time)
    assert restored_vpm.particles.n_particles_total == vpm.particles.n_particles_total
    np.testing.assert_allclose(restored_fvm.velocity, expected_u, rtol=0.0, atol=1e-13)
    np.testing.assert_allclose(restored_fvm.kinematic_pressure, expected_p, rtol=0.0, atol=1e-13)
    np.testing.assert_allclose(
        restored_fvm.volumetric_face_flux, expected_flux, rtol=0.0, atol=1e-13
    )
    np.testing.assert_allclose(
        restored._velocity_boundary_condition_old,
        coupler._velocity_boundary_condition_old,
        rtol=0.0,
        atol=1e-13,
    )
    restarted_metadata = json.loads((restart_case / "solution" / "run_metadata.json").read_text())
    assert restarted_metadata["execution"] == {
        "start_coupling_step": 1,
        "stop_coupling_step": 2,
        "configured_end_coupling_step": 2,
        "start_time": pytest.approx(VPM_TIME_STEP_SIZE),
        "stop_time": pytest.approx(2 * VPM_TIME_STEP_SIZE),
        "is_limited": False,
    }
    # No external solver case artifacts were created anywhere.
    assert not (tmp_path / "constant").exists()
    assert not (tmp_path / "system").exists()


@pytest.mark.parametrize("value", [True, False, 0, -1, 1.5, "2"])
def test_coupling_step_limit_rejects_invalid_values(value):
    from source.coupler import FVMVPMCoupler

    error = TypeError if isinstance(value, bool | float | str) else ValueError
    with pytest.raises(error):
        FVMVPMCoupler._validate_step_limit(value)
