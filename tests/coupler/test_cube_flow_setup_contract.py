"""Cube-flow timing and output contracts."""

from dataclasses import replace
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from source.coupler.solver import _validate_gbd_moment_recovery
from source.coupler.stable_renewal import (
    build_stable_renewal_lattice,
    required_buffer_length,
)
from source.coupler.vorticity_transfer import VorticityTransfer

CASE_DIR = Path(__file__).resolve().parents[2] / "tutorials" / "coupled_fvm_vpm" / "cube_flow"


def _valid_gbd_recovery(**overrides):
    recovery = {
        "applied": True,
        "nonzero_node_count": 100,
        "retained_node_count": 90,
        "pruned_node_count": 10,
        "support_augmented_node_count": 2,
        "correction_fraction": 0.01,
        "normalized_vortex_strength_residual": 1.0e-8,
        "normalized_linear_impulse_residual": 2.0e-8,
        "normalized_angular_impulse_residual": 3.0e-8,
    }
    recovery.update(overrides)
    return recovery


def test_gbd_moment_recovery_runtime_guard_accepts_a_closed_prune():
    _validate_gbd_moment_recovery(_valid_gbd_recovery(), 0.08)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"applied": False}, "without conservative moment recovery"),
        ({"correction_fraction": np.nan}, "is non-finite"),
        ({"normalized_linear_impulse_residual": np.inf}, "is non-finite"),
        ({"normalized_angular_impulse_residual": 1.1e-5}, "residual tolerance"),
        ({"correction_fraction": 0.081}, "excessive particle-strength correction"),
    ],
)
def test_gbd_moment_recovery_runtime_guard_stops_conclusive_failures(overrides, message):
    with pytest.raises(RuntimeError, match=message):
        _validate_gbd_moment_recovery(_valid_gbd_recovery(**overrides), 0.08)


def _load_setup(path: Path, module_name: str):
    spec = spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_trial(monkeypatch, module_name: str):
    setup = _load_setup(CASE_DIR / "cube_flow_setup.py", f"{module_name}_setup")
    monkeypatch.setitem(sys.modules, "cube_flow_setup", setup)
    return _load_setup(CASE_DIR / "assets" / "run_trial.py", module_name)


def test_cube_flow_uses_one_exact_sampling_cadence_and_native_substeps():
    assert not (CASE_DIR / "cube_flow_timing.py").exists()
    setup = _load_setup(CASE_DIR / "cube_flow_setup.py", "cube_flow_setup_contract")

    assert pytest.approx(0.010) == setup.FVM_TIME_STEP_SIZE
    assert pytest.approx(0.050) == setup.VPM_TIME_STEP_SIZE
    assert setup.VPM_TIME_STEP_SIZE / setup.FVM_TIME_STEP_SIZE == 5
    assert pytest.approx(0.5) == setup.WRITE_SOLUTION_BACKUP
    assert setup.FVM_WRITE_SOLUTION_BACKUP_INTERVAL_STEPS == 50
    assert setup.VPM_WRITE_SOLUTION_BACKUP_INTERVAL_STEPS == 10
    assert setup.FVM_SAMPLING_INTERVAL_STEPS == 5
    assert setup.VPM_SAMPLING_INTERVAL_STEPS == 1
    assert (
        pytest.approx(setup.WRITE_SOLUTION_BACKUP)
        == setup.FVM_WRITE_SOLUTION_BACKUP_INTERVAL_STEPS * setup.FVM_TIME_STEP_SIZE
    )
    assert (
        pytest.approx(setup.WRITE_SOLUTION_BACKUP)
        == setup.VPM_WRITE_SOLUTION_BACKUP_INTERVAL_STEPS * setup.VPM_TIME_STEP_SIZE
    )
    assert (
        pytest.approx(setup.SAMPLING_INTERVAL_TIME)
        == setup.FVM_SAMPLING_INTERVAL_STEPS * setup.FVM_TIME_STEP_SIZE
    )
    assert (
        pytest.approx(setup.SAMPLING_INTERVAL_TIME)
        == setup.VPM_SAMPLING_INTERVAL_STEPS * setup.VPM_TIME_STEP_SIZE
    )
    assert setup.END_TIME / setup.VPM_TIME_STEP_SIZE == 400

    assert all(
        sampler.schedule.every_n_steps == setup.FVM_SAMPLING_INTERVAL_STEPS
        and sampler.schedule.every_time is None
        for sampler in setup.FVM_SAMPLERS
    )
    assert all(
        sampler.schedule.every_n_steps == setup.VPM_SAMPLING_INTERVAL_STEPS
        for sampler in setup.VPM_SAMPLERS
    )
    assert (
        setup.FVM_SETUP.time.output_interval_steps == setup.FVM_WRITE_SOLUTION_BACKUP_INTERVAL_STEPS
    )
    assert setup.FVM_SETUP.time.output_interval_time is None
    assert setup.VPM_SETUP.checkpoint_interval_steps == 0
    assert setup.VPM_SETUP.write_precision == "f32"
    assert not setup.VPM_SETUP.checkpoint_store_velocity_gradient
    assert (
        setup.COUPLER_SETUP.checkpoint_interval_steps
        == setup.VPM_WRITE_SOLUTION_BACKUP_INTERVAL_STEPS
    )
    assert not hasattr(setup.COUPLER_SETUP, "vpm_particle_spacing")
    assert not hasattr(setup.COUPLER_SETUP, "vpm_core_radius_ratio")
    assert not hasattr(
        setup.COUPLER_SETUP,
        "is_boundary_condition_resynchronized_after_transfer",
    )
    assert setup.COUPLER_SETUP.transfer_method == "buffered_m4_renewal"
    assert pytest.approx(6.0 * setup.VPM_PARTICLE_SPACING) == setup.ETA_BLEND_WIDTH
    assert pytest.approx(setup.ETA_BLEND_WIDTH) == setup.COUPLER_SETUP.eta_blend_width
    assert setup.COUPLER_SETUP.fvm_consistency_width == pytest.approx(0.25)
    assert setup.COUPLER_SETUP.transfer_vorticity_cutoff == pytest.approx(0.05)
    assert setup.COUPLER_SETUP.transfer_boundary_prune_multiplier == pytest.approx(10.0)
    assert setup.COUPLER_SETUP.transfer_amplification_cap == pytest.approx(1.8)
    assert setup.VPM_SETUP.viscous.scheme == "GBD"


@pytest.mark.parametrize("scheme", ["CS", "RWM", "DVH", "GBD", "NONE"])
def test_cube_flow_viscous_config_factory_builds_each_supported_scheme(scheme):
    setup = _load_setup(CASE_DIR / "cube_flow_setup.py", f"cube_flow_viscous_{scheme.lower()}")
    viscous = setup.make_vpm_viscous_config(scheme)

    assert viscous.scheme == scheme
    assert viscous.particle_spacing == pytest.approx(setup.VPM_PARTICLE_SPACING)
    assert viscous.core_radius_ratio == pytest.approx(setup.VPM_CORE_RADIUS_RATIO)
    if scheme != "NONE":
        assert viscous.kinematic_viscosity == pytest.approx(setup.KINEMATIC_VISCOSITY)
    replace(setup.VPM_SETUP, viscous=viscous)._validate_config()


@pytest.mark.parametrize("scheme", ["CS", "RWM", "DVH", "NONE"])
def test_buffered_cube_transfer_explicitly_requires_gbd(scheme):
    setup = _load_setup(CASE_DIR / "cube_flow_setup.py", f"cube_flow_rejects_{scheme.lower()}")
    coupler = SimpleNamespace(
        setup=setup.COUPLER_SETUP,
        kinematic_viscosity=setup.KINEMATIC_VISCOSITY,
        fvm_box=np.asarray(setup.FVM_BOX, dtype=np.float64),
        vpm_solver=SimpleNamespace(viscous_scheme=scheme),
    )

    with pytest.raises(ValueError, match="currently requires the GBD viscous scheme"):
        VorticityTransfer(coupler)


def test_cube_interface_uses_the_production_half_grid_release_geometry():
    setup = _load_setup(CASE_DIR / "cube_flow_setup.py", "cube_flow_m4_buffer_contract")
    h = setup.VPM_PARTICLE_SPACING
    downstream_face = setup.TRANSFER_REGION_BOX[1]
    buffer = required_buffer_length(
        np.linalg.norm(setup.FREESTREAM_VELOCITY),
        setup.VPM_TIME_STEP_SIZE,
        h,
    )
    lattice_anchor = np.full(3, -0.5 * setup.CUBE_SIDE - 0.5 * h)
    lattice = build_stable_renewal_lattice(
        setup.TRANSFER_REGION_BOX,
        h,
        buffer_length=buffer,
        authority_ramp_width=setup.ETA_BLEND_WIDTH,
        lattice_anchor=lattice_anchor,
    )
    x_planes = np.unique(lattice.positions[:, 0])
    authority_by_plane = np.array(
        [lattice.fvm_authority[lattice.positions[:, 0] == x].max() for x in x_planes]
    )
    downstream_authority = x_planes[(x_planes > 0.0) & (authority_by_plane > 0.0)]
    renewal_planes = x_planes[x_planes <= lattice.renewal_bounds[1]]
    persistent_planes = x_planes[x_planes > lattice.renewal_bounds[1]]

    assert buffer == pytest.approx(0.1375)
    assert lattice.renewal_bounds[1] == pytest.approx(1.3875)
    assert lattice.origin[0] == pytest.approx(-1.453125)
    assert lattice.shape == (94, 94, 94)
    assert not np.any(np.isclose(x_planes, downstream_face, rtol=0.0, atol=1.0e-14))
    assert downstream_authority[-1] == pytest.approx(1.234375)
    assert x_planes[x_planes > downstream_face][0] == pytest.approx(1.265625)
    assert renewal_planes[-1] == pytest.approx(1.359375)
    assert persistent_planes[0] == pytest.approx(1.390625)
    assert x_planes[-1] == pytest.approx(1.453125)


def test_cube_flow_timing_resolver_adjusts_steps_without_shifting_outputs():
    setup = _load_setup(CASE_DIR / "cube_flow_setup.py", "cube_flow_timing_resolver")

    fvm_step, vpm_step, fvm_backup, vpm_backup, fvm_sample, vpm_sample = setup.resolve_case_timing(
        0.005, 3, 0.5, 0.05
    )

    assert vpm_step / fvm_step == pytest.approx(3)
    assert fvm_backup * fvm_step == pytest.approx(0.5)
    assert vpm_backup * vpm_step == pytest.approx(0.5)
    assert fvm_sample * fvm_step == pytest.approx(0.05)
    assert vpm_sample * vpm_step == pytest.approx(0.05)


def test_trial_restart_accepts_the_matching_legacy_physical_horizon(tmp_path, monkeypatch):
    trial = _load_trial(monkeypatch, "cube_flow_restart_end_time_contract")
    assert "vpm.viscous.gbd_threshold" in trial.TRANSFER_RESTART_ALLOWLIST
    assert "panel.coupling_scope" in trial.TRANSFER_RESTART_ALLOWLIST
    captured = {}
    monkeypatch.setattr(trial.case, "main", lambda **kwargs: captured.update(kwargs))
    restart = tmp_path / "seed"
    output = tmp_path / "restart"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_trial.py",
            "--end-time",
            "2.5",
            "--case-directory",
            str(output),
            "--restart-from",
            str(restart),
        ],
    )

    trial.main()

    assert trial.case.FVM_SETUP.time.end_time == pytest.approx(2.5)
    assert captured["restart_from"] == restart.resolve()
    assert captured["max_coupling_steps"] is None
    assert captured["checkpoint_at_stop"] is True


def test_trial_restart_step_limit_keeps_the_production_horizon(tmp_path, monkeypatch):
    trial = _load_trial(monkeypatch, "cube_flow_restart_step_limit_contract")
    captured = {}
    monkeypatch.setattr(trial.case, "main", lambda **kwargs: captured.update(kwargs))
    restart = tmp_path / "seed"
    output = tmp_path / "restart"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_trial.py",
            "--coupling-steps",
            "5",
            "--case-directory",
            str(output),
            "--restart-from",
            str(restart),
        ],
    )

    trial.main()

    assert trial.case.FVM_SETUP.time.end_time == pytest.approx(trial.case.END_TIME)
    assert captured["restart_from"] == restart.resolve()
    assert captured["max_coupling_steps"] == 5
    assert captured["checkpoint_at_stop"] is True


def test_cube_acceptance_rejects_an_excessive_renewal_closure(monkeypatch):
    monkeypatch.syspath_prepend(str(CASE_DIR / "assets"))
    check = _load_setup(
        CASE_DIR / "assets" / "check_run.py",
        "cube_flow_check_renewal_closure_contract",
    )
    transfer = {
        "transfer_method": "buffered_m4_renewal",
        "n_particles_before": 10,
        "n_particles_removed": 4,
        "n_particles_injected": 5,
        "n_particles_after": 11,
        "population_pruned_particles": 0,
        "state_change_vortex_strength_net_x": 0.0,
        "state_change_vortex_strength_net_y": 0.0,
        "state_change_vortex_strength_net_z": 0.0,
        "renewal_raw_vortex_strength_error": 0.02,
        "renewal_applied_vortex_strength_correction": 0.02,
        "renewal_conservation_error": 1.0e-14,
        "renewal_vortex_strength_tolerance": 1.0e-10,
        "renewal_raw_linear_impulse_error": 0.03,
        "renewal_applied_linear_impulse_correction": 0.03,
        "renewal_linear_impulse_error": 2.0e-14,
        "renewal_linear_impulse_tolerance": 1.0e-10,
        "renewal_applied_particle_strength_fraction": 0.07,
    }
    record = {
        "gbd_moment_recovery": {
            "applied": True,
            "nonzero_node_count": 100,
            "retained_node_count": 90,
            "pruned_node_count": 10,
            "support_augmented_node_count": 2,
            "correction_fraction": 0.01,
            "normalized_vortex_strength_residual": 1.0e-8,
            "normalized_linear_impulse_residual": 2.0e-8,
            "normalized_angular_impulse_residual": 3.0e-8,
        },
        "vpm_boundary_condition_flux": {
            "corrected_mismatch": 0.0,
            "raw_relative": 0.0,
            "acceptance_limit": 1.0e-8,
        },
        "transfer": transfer,
    }

    check._check_coupling_history([record], closure_correction_limit=0.08)

    record["gbd_moment_recovery"]["applied"] = False
    with pytest.raises(SystemExit, match="without conservative moment recovery"):
        check._check_coupling_history([record], closure_correction_limit=0.08)
    record["gbd_moment_recovery"]["applied"] = True

    record["gbd_moment_recovery"]["support_augmented_node_count"] = -1
    with pytest.raises(SystemExit, match="negative GBD moment-recovery diagnostic"):
        check._check_coupling_history([record], closure_correction_limit=0.08)
    record["gbd_moment_recovery"]["support_augmented_node_count"] = 2

    transfer["renewal_applied_particle_strength_fraction"] = 0.09
    with pytest.raises(SystemExit, match="excessive particle-strength correction"):
        check._check_coupling_history([record], closure_correction_limit=0.08)


def test_cube_acceptance_requires_contiguous_coupler_diagnostics(monkeypatch):
    monkeypatch.syspath_prepend(str(CASE_DIR / "assets"))
    check = _load_setup(
        CASE_DIR / "assets" / "check_run.py",
        "cube_flow_check_coupling_coverage_contract",
    )
    metadata = {
        "vpm_time_step_size": 0.01,
        "execution": {
            "start_coupling_step": 2,
            "stop_coupling_step": 5,
            "stop_time": 0.05,
        },
    }
    records = [{"step": step, "time": 0.01 * step} for step in range(3, 6)]

    check._check_coupling_coverage(records, metadata)

    with pytest.raises(SystemExit, match="complete execution segment"):
        check._check_coupling_coverage([records[0], records[2]], metadata)
    wrong_time = [dict(record) for record in records]
    wrong_time[-1]["time"] = 0.049
    with pytest.raises(SystemExit, match="times do not match"):
        check._check_coupling_coverage(wrong_time, metadata)


def test_cube_acceptance_horizon_supports_short_runs_and_defaults_to_two_seconds(monkeypatch):
    monkeypatch.syspath_prepend(str(CASE_DIR / "assets"))
    check = _load_setup(
        CASE_DIR / "assets" / "check_run.py",
        "cube_flow_check_acceptance_horizon_contract",
    )

    assert check._resolve_acceptance_horizon(0.05, None) == pytest.approx(0.05)
    assert check._resolve_acceptance_horizon(0.10, None) == pytest.approx(0.10)
    assert check._resolve_acceptance_horizon(20.0, None) == pytest.approx(2.0)
    assert check._resolve_acceptance_horizon(20.0, 1.5) == pytest.approx(1.5)
    with pytest.raises(ValueError, match="acceptance horizon"):
        check._resolve_acceptance_horizon(0.05, 0.10)


def test_cube_reference_gate_requires_every_profile_at_the_acceptance_horizon(
    tmp_path,
    monkeypatch,
):
    monkeypatch.syspath_prepend(str(CASE_DIR / "assets"))
    check = _load_setup(
        CASE_DIR / "assets" / "check_run.py",
        "cube_flow_check_reference_coverage_contract",
    )
    candidate = tmp_path / "candidate" / "samples"
    reference = tmp_path / "reference" / "samples"
    candidate.mkdir(parents=True)
    reference.mkdir(parents=True)
    times = (0.05, 0.10)

    force_text = "time,drag_coefficient\n" + "".join(f"{time},2.0\n" for time in times)
    profile_text = "time,position_x,velocity_x\n" + "".join(
        f"{time},-1.0,1.0\n{time},1.0,1.0\n" for time in times
    )
    (candidate / "forces_history.csv").write_text(force_text, encoding="utf-8")
    (reference / "forces_history.csv").write_text(force_text, encoding="utf-8")
    for name in ("centreline", "offaxis_y075"):
        (reference / f"{name}.csv").write_text(profile_text, encoding="utf-8")
        for source in ("fvm", "vpm"):
            (candidate / f"{source}_{name}.csv").write_text(profile_text, encoding="utf-8")

    summary = check._check_reference_accuracy(
        candidate.parent,
        reference.parent,
        0.05,
        0.10,
    )
    assert "through t=0.1" in summary

    short_profile = "time,position_x,velocity_x\n0.05,-1.0,1.0\n0.05,1.0,1.0\n"
    (reference / "offaxis_y075.csv").write_text(short_profile, encoding="utf-8")
    with pytest.raises(SystemExit, match="does not cover the acceptance horizon"):
        check._check_reference_accuracy(
            candidate.parent,
            reference.parent,
            0.05,
            0.10,
        )


def test_cube_reference_gate_uses_spatial_mean_profile_error(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(CASE_DIR / "assets"))
    check = _load_setup(
        CASE_DIR / "assets" / "check_run.py",
        "cube_flow_check_profile_mean_contract",
    )
    candidate = tmp_path / "candidate" / "samples"
    reference = tmp_path / "reference" / "samples"
    candidate.mkdir(parents=True)
    reference.mkdir(parents=True)
    (candidate / "forces_history.csv").write_text(
        "time,drag_coefficient\n0.1,1.0\n",
        encoding="utf-8",
    )
    (reference / "forces_history.csv").write_text(
        "time,drag_coefficient\n0.1,1.0\n",
        encoding="utf-8",
    )

    positions = [1.0 + 0.01 * index for index in range(100)]
    reference_profile = "time,position_x,velocity_x\n" + "".join(
        f"0.1,{position},1.0\n" for position in positions
    )
    localized_profile = "time,position_x,velocity_x\n" + "".join(
        f"0.1,{position},{1.5 if index == 0 else 1.0}\n" for index, position in enumerate(positions)
    )
    for name in ("centreline", "offaxis_y075"):
        (reference / f"{name}.csv").write_text(reference_profile, encoding="utf-8")
        for source in ("fvm", "vpm"):
            (candidate / f"{source}_{name}.csv").write_text(
                localized_profile,
                encoding="utf-8",
            )

    summary = check._check_reference_accuracy(
        candidate.parent,
        reference.parent,
        0.07,
        0.10,
    )
    assert "worst reference error" in summary

    uniform_error_profile = "time,position_x,velocity_x\n" + "".join(
        f"0.1,{position},1.08\n" for position in positions
    )
    (candidate / "fvm_offaxis_y075.csv").write_text(
        uniform_error_profile,
        encoding="utf-8",
    )
    with pytest.raises(SystemExit, match="reference errors exceed"):
        check._check_reference_accuracy(
            candidate.parent,
            reference.parent,
            0.07,
            0.10,
        )


def test_reference_flow_uses_the_same_sampling_and_checkpoint_cadence():
    coupled = _load_setup(CASE_DIR / "cube_flow_setup.py", "coupled_flow_reference_contract")
    reference = _load_setup(
        CASE_DIR / "reference_flow" / "reference_flow_setup.py",
        "reference_flow_setup_contract",
    )

    assert pytest.approx(coupled.FVM_TIME_STEP_SIZE) == reference.FVM_TIME_STEP_SIZE
    assert pytest.approx(0.050) == reference.SAMPLING_INTERVAL_TIME
    assert all(
        sampler.schedule.every_time == reference.SAMPLING_INTERVAL_TIME
        for sampler in reference.SAMPLERS
    )
    assert reference.FVM_SETUP.time.output_interval_time == reference.CHECKPOINT_INTERVAL_TIME
