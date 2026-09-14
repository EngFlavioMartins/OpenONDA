"""Cube-flow timing and output tests."""

from dataclasses import replace
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from source.coupler.vorticity_transfer import VorticityTransfer

CASE_DIR = Path(__file__).resolve().parents[2] / "tutorials" / "coupled_fvm_vpm" / "02_cube_flow"


def _load_setup(path: Path, module_name: str):
    spec = spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_trial(monkeypatch, module_name: str):
    setup = _load_setup(CASE_DIR / "setup.py", f"{module_name}_setup")
    monkeypatch.setitem(sys.modules, "setup", setup)
    return _load_setup(CASE_DIR / "assets" / "run_trial.py", module_name)


def test_cube_flow_schedules_share_physical_time():
    setup = _load_setup(CASE_DIR / "setup.py", "cube_flow_schedule")
    for dt, backup, sample in (
        (
            setup.FVM_TIME_STEP_SIZE,
            setup.FVM_WRITE_SOLUTION_BACKUP_INTERVAL_STEPS,
            setup.FVM_SAMPLING_INTERVAL_STEPS,
        ),
        (
            setup.VPM_TIME_STEP_SIZE,
            setup.VPM_WRITE_SOLUTION_BACKUP_INTERVAL_STEPS,
            setup.VPM_SAMPLING_INTERVAL_STEPS,
        ),
    ):
        assert dt * backup == pytest.approx(setup.WRITE_SOLUTION_BACKUP)
        assert dt * sample == pytest.approx(setup.SAMPLING_INTERVAL_TIME)
    ratio = setup.VPM_TIME_STEP_SIZE / setup.FVM_TIME_STEP_SIZE
    assert ratio == pytest.approx(round(ratio))


def test_cube_recommended_formulation_preserves_resolution_and_small_domain():
    setup = _load_setup(CASE_DIR / "setup.py", "cube_recommended")
    assert setup.REFERENCE_FINE_DX == 0.06
    assert setup.FVM_MESH.max_cell_size == pytest.approx(0.72)
    assert setup.FVM_MESH.refinements[0].name == "nearBody"
    assert setup.FVM_MESH.effective_cell_size(0.06, strict=True) == pytest.approx(0.045)
    assert setup.VPM_CASE.numerics.viscous.particle_spacing == pytest.approx(0.06)
    assert setup.FVM_BOX == (-1.5, 1.5, -1.5, 1.5, -1.5, 1.5)
    assert setup.FVM_MESH.patch_refinements[0].cell_size == 0.06
    assert setup.COUPLER_SETUP.interface_iterations == 12
    assert setup.COUPLER_SETUP.fvm_consistency_width == 0
    assert setup.COUPLER_SETUP.boundary_condition_mode == "vorticity_mixed"
    assert setup.VPM_PANEL_SOLVER.coupling_scope == "fvm_vpm"
    assert setup.VPM_CASE.numerics.viscous.scheme == "GBD"
    assert isinstance(setup.VPM_CASE.numerics.induction, setup.vpm.TreecodeInduction)
    assert setup.VPM_CASE.numerics.compute_device == "AUTO"


@pytest.mark.parametrize("is_root", [False, True])
def test_cube_run_owns_vpm_only_on_root(monkeypatch, is_root):
    setup = _load_setup(CASE_DIR / "setup.py", f"cube_run_root_{is_root}")
    closed = []
    fvm_solver = SimpleNamespace(
        parallel=SimpleNamespace(is_root=is_root),
        close=lambda: closed.append("fvm"),
    )
    vpm_solver = SimpleNamespace(close=lambda: closed.append("vpm"))
    coupled = {}
    monkeypatch.setattr(
        setup, "RunConfig", lambda **kwargs: SimpleNamespace(ensure_runtime=lambda _: None)
    )
    monkeypatch.setattr(setup.msh, "CachedMesh", lambda *args: object())
    monkeypatch.setattr(setup.fvm, "create_fvm_solver", lambda *args, **kwargs: fvm_solver)
    monkeypatch.setattr(
        setup.vpm,
        "VPMSolver",
        lambda *args: vpm_solver if is_root else pytest.fail("VPM created on worker"),
    )
    def create_coupler(fvm, vpm, config):
        coupled["vpm"] = vpm
        return SimpleNamespace(run=lambda: None)

    monkeypatch.setattr(setup.coupling, "create_coupler", create_coupler)

    setup.main()

    assert coupled["vpm"] is (vpm_solver if is_root else None)
    assert closed == (["fvm", "vpm"] if is_root else ["fvm"])


def test_cube_flow_viscous_config_factory_rejects_rwm_for_les():
    setup = _load_setup(CASE_DIR / "setup.py", "cube_flow_viscous_rwm_rejected")

    # The reusable Numerics constructor owns this physical compatibility rule.
    with pytest.raises(ValueError, match="RWM.*GBD.*LES"):
        replace(
            setup.VPM_CASE.numerics,
            viscous=setup.vpm.ViscousConfig.rwm(
                kinematic_viscosity=setup.KINEMATIC_VISCOSITY,
                particle_spacing=setup.VPM_PARTICLE_SPACING,
            ),
        )


def test_cube_flow_viscous_config_factory_rejects_dvh_for_les():
    setup = _load_setup(CASE_DIR / "setup.py", "cube_flow_viscous_dvh_rejected")

    with pytest.raises(ValueError, match="DVH.*GBD.*LES"):
        replace(
            setup.VPM_CASE.numerics,
            viscous=setup.vpm.ViscousConfig.dvh(
                kinematic_viscosity=setup.KINEMATIC_VISCOSITY,
                particle_spacing=setup.VPM_PARTICLE_SPACING,
            ),
        )


@pytest.mark.parametrize("scheme", ["CS", "RWM", "DVH", "NONE"])
def test_buffered_cube_transfer_explicitly_requires_gbd(scheme):
    setup = _load_setup(CASE_DIR / "setup.py", f"cube_flow_rejects_{scheme.lower()}")
    coupler = SimpleNamespace(
        setup=setup.COUPLER_SETUP,
        kinematic_viscosity=setup.KINEMATIC_VISCOSITY,
        fvm_box=np.asarray(setup.FVM_BOX, dtype=np.float64),
        vpm_solver=SimpleNamespace(viscous_scheme=scheme),
    )

    with pytest.raises(ValueError, match="currently requires the GBD viscous scheme"):
        VorticityTransfer(coupler)


def test_trial_restart_step_limit_keeps_the_production_horizon(tmp_path, monkeypatch):
    trial = _load_trial(monkeypatch, "cube_flow_restart_step_limit_test")
    captured = {}
    monkeypatch.setattr(trial, "_run_case", lambda **kwargs: captured.update(kwargs))
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


def test_cube_acceptance_rejects_an_excessive_renewal_closure(monkeypatch):
    monkeypatch.syspath_prepend(str(CASE_DIR / "assets"))
    check = _load_setup(
        CASE_DIR / "assets" / "check_run.py",
        "cube_flow_check_renewal_closure_test",
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
        "cube_flow_check_coupling_coverage_test",
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
        "cube_flow_check_acceptance_horizon_test",
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
        "cube_flow_check_reference_coverage_test",
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
        "cube_flow_check_profile_mean_test",
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


def test_reference_flow_declares_its_sampling_and_backup_cadence(monkeypatch):
    coupled = _load_setup(CASE_DIR / "setup.py", "coupled_flow_reference_test")
    reference = _load_setup(
        CASE_DIR / "reference_flow" / "setup.py",
        "reference_flow_setup_test",
    )
    captured = {}

    def create(config, **_kwargs):
        captured["config"] = config
        return object()

    monkeypatch.setattr(reference.fvm, "create_fvm_solver", create)
    reference.create_solver("coarse", 0.125)
    config = captured["config"]

    assert config.time.time_step_size == pytest.approx(coupled.FVM_TIME_STEP_SIZE)
    schedules = {sampler.name: sampler.schedule.every_time for sampler in config.samplers}
    assert schedules == {
        "forces_history": 0.05,
        "centreline": 0.25,
        "offaxis_y075": 0.25,
    }
    assert config.time.output_schedule.every_time == 1.0
    assert config.backup.schedule.every_time == 1.0
