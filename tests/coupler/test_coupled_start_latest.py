"""Automatic coupled restoration uses the committed bundle, including step zero."""

import json

import numpy as np
import pytest

from source.coupler import CouplerSetup, FVMVPMCoupler
from source.solvers.fvm import BoundaryConfig, FVMSetup, FVMSolver, TimeConfig, TransportConfig
from source.solvers.fvm.mesh.rectilinear import coupling_box_mesh
from source.solvers.vpm import DirectInduction, Numerics, ViscousConfig, VPMCase, VPMSolver


def _coupler(directory):
    velocity = [1.0, 0.0, 0.0]
    setup = CouplerSetup(
        freestream_velocity=velocity,
        eta_blend_width=0.0,
        backup_interval_steps=2,
    )
    fvm = FVMSolver(
        FVMSetup(
            case_name="continue",
            time=TimeConfig(time_step_size=0.01, end_time=0.06),
            transport=TransportConfig(kinematic_viscosity=0.01),
            boundaries=[
                BoundaryConfig(
                    name="numericalBoundary",
                    velocity_type="fixedValue",
                    velocity_value=velocity,
                    pressure_type="fixedFluxPressure",
                )
            ],
            initial_velocity=velocity,
        ),
        case_dir=directory,
        mesh_data=coupling_box_mesh((-0.5, 0.5, -0.5, 0.5, -0.5, 0.5), 0.25),
    )
    vpm = VPMSolver(
        VPMCase(
            directory=directory,
            numerics=Numerics(
                time_step_size=0.02,
                compute_device="CPU",
                max_n_particles=4000,
                domain_bounds=(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0),
                freestream_velocity=velocity,
                induction=DirectInduction(),
                viscous=ViscousConfig.gbd(
                    kinematic_viscosity=0.01,
                    particle_spacing=0.25,
                    threshold=1e-5,
                    threshold_mode="absolute",
                ),
            ),
        )
    )
    return FVMVPMCoupler(fvm, vpm, setup)


def test_coupled_latest_replays_tail_and_stops_at_configured_end(tmp_path):
    with _coupler(tmp_path) as first:
        assert first.run(start_from="latest", max_coupling_steps=2) == 2
        first.solve(start_step=2)
        expected = first.fvm_solver.velocity.copy()
    # The tail at step three has samples but no scheduled atomic checkpoint.
    manifest_path = tmp_path / "solution/backups/manifest.json"
    assert json.loads(manifest_path.read_text())["coupling_step"] == 2
    with _coupler(tmp_path) as resumed:
        assert resumed.run(start_from="latest") == 3
        np.testing.assert_allclose(resumed.fvm_solver.velocity, expected, atol=1e-13, rtol=0)
    assert json.loads(manifest_path.read_text())["coupling_step"] == 3
    history = tmp_path / "solution/coupler_diagnostics.jsonl"
    assert [json.loads(row)["step"] for row in history.read_text().splitlines()] == [1, 2, 3]
    before = history.read_bytes()
    with _coupler(tmp_path) as completed:
        assert completed.run(start_from="latest") == 3
    assert history.read_bytes() == before


def test_coupled_zero_backup_does_not_repeat_initial_transfer(tmp_path):
    with _coupler(tmp_path) as first:
        first.initialize()
        first.solve(max_coupling_steps=1, backup_at_start=True)
    manifest = tmp_path / "solution/backups/manifest.json"
    assert json.loads(manifest.read_text())["coupling_step"] == 0
    with _coupler(tmp_path) as resumed:
        resumed.run(start_from="latest", max_coupling_steps=1)
        # One initialization transfer, then one accepted transfer.
        assert resumed.vorticity_transfer.step == 2


@pytest.mark.parametrize("corrupt_manifest", [False, True])
def test_coupled_initial_replaces_prior_run_and_latest_continues_new_branch(
    tmp_path, corrupt_manifest
):
    case = tmp_path / "restarted"
    with _coupler(case) as previous:
        assert previous.run(start_from="latest", max_coupling_steps=2) == 2

    manifest = case / "solution/backups/manifest.json"
    assert json.loads(manifest.read_text())["coupling_step"] == 2
    # This frame is intentionally invalid. A fresh run must remove it from
    # native discovery, even though the coupled manifest owns the restart.
    old_frame = case / "solution/vpm/vpm_999999.h5"
    old_frame.parent.mkdir(parents=True, exist_ok=True)
    old_frame.write_bytes(b"stale VPM frame from the previous run")
    if corrupt_manifest:
        manifest.write_text("{invalid prior manifest")

    with _coupler(case) as fresh:
        assert fresh.run(start_from="initial", max_coupling_steps=1) == 1
        assert fresh.fvm_solver.step == fresh.n_fvm_substeps
        assert fresh.vpm_solver.step == 1
    assert not old_frame.exists()
    assert json.loads(manifest.read_text())["coupling_step"] == 1

    with _coupler(case) as resumed:
        assert resumed.run(start_from="latest") == 3
        actual_velocity = resumed.fvm_solver.velocity.copy()
        actual_pressure = resumed.fvm_solver.get_pressure_field().copy()
    assert json.loads(manifest.read_text())["coupling_step"] == 3
    history = case / "solution/coupler_diagnostics.jsonl"
    assert [json.loads(row)["step"] for row in history.read_text().splitlines()] == [1, 2, 3]
    before = history.read_bytes()
    with _coupler(case) as completed:
        assert completed.run(start_from="latest") == 3
    assert history.read_bytes() == before

    with _coupler(tmp_path / "reference") as reference:
        assert reference.run(start_from="initial") == 3
        np.testing.assert_allclose(
            actual_velocity, reference.fvm_solver.velocity, rtol=0, atol=1e-12
        )
        np.testing.assert_allclose(
            actual_pressure, reference.fvm_solver.get_pressure_field(), rtol=0, atol=1e-12
        )


def test_latest_without_bundle_replaces_prior_output_history(tmp_path):
    import shutil

    with _coupler(tmp_path) as previous:
        previous.run(start_from="latest", max_coupling_steps=2)
    shutil.rmtree(tmp_path / "solution/backups")
    with _coupler(tmp_path) as fresh:
        assert fresh.run(start_from="latest", max_coupling_steps=1) == 1
    history = tmp_path / "solution/coupler_diagnostics.jsonl"
    assert [json.loads(row)["step"] for row in history.read_text().splitlines()] == [1]
