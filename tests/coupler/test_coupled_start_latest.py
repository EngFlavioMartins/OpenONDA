"""Automatic coupled restoration uses the committed bundle, including step zero."""

import json

import numpy as np

from source.coupler import CouplerSetup, FVMVPMCoupler
from source.solvers.fvm import BoundaryConfig, FVMSetup, FVMSolver, TimeConfig, TransportConfig
from source.solvers.fvm.mesh.rectilinear import coupling_box_mesh
from source.solvers.vpm import DirectInduction, Numerics, ViscousConfig, VPMCase, VPMSolver


def _coupler(directory):
    velocity = [1.0, 0.0, 0.0]
    setup = CouplerSetup(freestream_velocity=velocity, eta_blend_width=0.0, backup_interval_steps=2, transfer_discretization_error_limit=1.0)
    fvm = FVMSolver(FVMSetup(
        case_name="continue", time=TimeConfig(time_step_size=0.01, end_time=0.06),
        transport=TransportConfig(kinematic_viscosity=0.01),
        boundaries=[BoundaryConfig(name="numericalBoundary", velocity_type="fixedValue",
                                   velocity_value=velocity, pressure_type="fixedFluxPressure")],
        initial_velocity=velocity,
    ), case_dir=directory, mesh_data=coupling_box_mesh((-.5,.5,-.5,.5,-.5,.5), .25))
    vpm = VPMSolver(VPMCase(directory=directory, numerics=Numerics(
        time_step_size=0.02, compute_device="CPU", max_n_particles=4000,
        domain_bounds=(-1.,1.,-1.,1.,-1.,1.), freestream_velocity=velocity,
        induction=DirectInduction(),
        viscous=ViscousConfig.cs(kinematic_viscosity=0.01, particle_spacing=.25),
    )))
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
