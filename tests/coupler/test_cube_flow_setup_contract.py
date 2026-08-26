"""Cube-flow timing and output contracts."""

from dataclasses import replace
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pytest

from source.coupler.lattice_transfer import map_cell_circulation_to_lattice

CASE_DIR = Path(__file__).resolve().parents[2] / "tutorials" / "coupled_fvm_vpm" / "cube_flow"


def _load_setup(path: Path, module_name: str):
    spec = spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cube_flow_uses_one_exact_sampling_cadence_and_native_substeps():
    assert not (CASE_DIR / "cube_flow_timing.py").exists()
    setup = _load_setup(CASE_DIR / "cube_flow_setup.py", "cube_flow_setup_contract")

    assert pytest.approx(0.005) == setup.FVM_TIME_STEP_SIZE
    assert pytest.approx(0.010) == setup.VPM_TIME_STEP_SIZE
    assert setup.VPM_TIME_STEP_SIZE / setup.FVM_TIME_STEP_SIZE == 2
    assert pytest.approx(0.5) == setup.WRITE_SOLUTION_BACKUP
    assert setup.FVM_WRITE_SOLUTION_BACKUP_INTERVAL_STEPS == 100
    assert setup.VPM_WRITE_SOLUTION_BACKUP_INTERVAL_STEPS == 50
    assert setup.FVM_SAMPLING_INTERVAL_STEPS == 10
    assert setup.VPM_SAMPLING_INTERVAL_STEPS == 5
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
    assert setup.END_TIME / setup.VPM_TIME_STEP_SIZE == 2000

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
    assert setup.COUPLER_SETUP.transfer_method == "common_lattice"
    assert pytest.approx(3.0 * setup.VPM_PARTICLE_SPACING) == setup.ETA_BLEND_WIDTH
    assert pytest.approx(setup.ETA_BLEND_WIDTH) == setup.COUPLER_SETUP.eta_blend_width


@pytest.mark.parametrize("scheme", ["CS", "RWM", "DVH", "GBD", "NONE"])
def test_cube_flow_can_select_any_supported_vpm_viscous_scheme(scheme):
    setup = _load_setup(CASE_DIR / "cube_flow_setup.py", f"cube_flow_viscous_{scheme.lower()}")
    viscous = setup.make_vpm_viscous_config(scheme)

    assert viscous.scheme == scheme
    assert viscous.particle_spacing == pytest.approx(setup.VPM_PARTICLE_SPACING)
    assert viscous.core_radius_ratio == pytest.approx(setup.VPM_CORE_RADIUS_RATIO)
    if scheme != "NONE":
        assert viscous.kinematic_viscosity == pytest.approx(setup.KINEMATIC_VISCOSITY)
    replace(setup.VPM_SETUP, viscous=viscous)._validate_config()


def test_cube_interface_m4_support_requires_one_outer_renewal_plane():
    setup = _load_setup(CASE_DIR / "cube_flow_setup.py", "cube_flow_m4_buffer_contract")
    h = setup.VPM_PARTICLE_SPACING
    downstream_face = setup.TRANSFER_REGION_BOX[1]
    last_donor_centre = downstream_face - 0.5 * h

    # The cube interface is a regular VPM plane while its last FVM donor is at
    # the half-grid phase.  Complete M4' support therefore reaches one full
    # VPM plane into persistent authority; renewal must own that support plane
    # while forming the new absolute state.
    assert downstream_face / h == pytest.approx(round(downstream_face / h))
    mapped = map_cell_circulation_to_lattice(
        np.array([[last_donor_centre, 0.0, 0.0]]),
        np.array([h**3]),
        np.array([[0.0, 1.0, 0.0]]),
        lattice_anchor=np.zeros(3),
        spacing=h,
    )
    active = np.linalg.norm(mapped.vortex_strength, axis=1) > 0.0
    active_x_planes = np.unique(mapped.position[active, 0])

    np.testing.assert_allclose(
        active_x_planes,
        downstream_face + h * np.array([-2.0, -1.0, 0.0, 1.0]),
        rtol=0.0,
        atol=2.0e-15,
    )
    assert active_x_planes[-1] - downstream_face == pytest.approx(h)


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
