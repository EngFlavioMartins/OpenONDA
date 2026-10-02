"""Resolved cylinder study controls before solver allocation."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


def _setup_module():
    path = (
        Path(__file__).resolve().parents[2]
        / "tutorials/coupled_fvm_vpm/01_cylinder_shedding_flow/setup.py"
    )
    spec = importlib.util.spec_from_file_location("coupled_cylinder_setup", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_builder_keeps_default_span_force_area_and_interface_iterations():
    module = _setup_module()
    fvm, vpm, coupler, mesh = module.build_case()
    assert fvm.samplers[0].reference_area == pytest.approx(0.96)
    assert coupler.interface_iterations == 3
    assert coupler.interface_acceleration == "none"
    assert vpm.numerics.induction.z_min == pytest.approx(-0.48)
    assert vpm.numerics.viscous.particle_spacing == pytest.approx(0.04)
    assert isinstance(mesh, module.msh.ExtrudedCartesianMesher)
    assert len(mesh.levels) - 1 == 24
    assert vpm.numerics.compute_device == "AUTO"
    assert vpm.numerics.induction.gaussian_mesh_policy == module.GAUSSIAN_MESH_POLICY


def test_builder_acceleration_is_explicit_and_restart_identified():
    from source.coupler import CouplerSetup

    module = _setup_module()
    _fvm, _vpm, accelerated, _mesh = module.build_case(
        overrides={"interface_acceleration": "aitken"}
    )
    assert accelerated.interface_iterations == 3
    assert accelerated.interface_acceleration == "aitken"
    assert accelerated.interface_normal_tolerance == pytest.approx(1.0e-5)
    assert accelerated.interface_gradient_tolerance == pytest.approx(1.0e-5)
    assert accelerated.to_dict()["coupler"]["interface_acceleration"] == "aitken"
    assert CouplerSetup().to_dict()["coupler"]["interface_acceleration"] == "none"

    with pytest.raises(ValueError, match="interface_acceleration"):
        module.build_case(overrides={"interface_acceleration": "anderson"})
    for sweeps in (1, 2):
        with pytest.raises(ValueError, match="at least three"):
            CouplerSetup(interface_acceleration="aitken", interface_iterations=sweeps)


def test_builder_resolves_independent_span_dz_and_particle_spacing():
    module = _setup_module()
    fvm, vpm, coupler, mesh = module.build_case(
        gaussian_mesh_policy=None,
        overrides={
            "hxy": 0.08,
            "span": 0.48,
            "dz": 0.06,
            "particle_spacing_ratio": 1.25,
            "core_radius_ratio": 1.2,
            "blend_width_ratio": 8.0,
            "release_width_ratio": 2.0,
            "exchange_dt": 0.08,
            "cores": 2,
            "compute_device": "CPU",
            "particle_limit": 50000,
        }
    )
    assert fvm.cores == 2
    assert fvm.samplers[0].reference_area == pytest.approx(0.48)
    span_samples = {
        sample.file_name: sample for sample in fvm.samplers if sample.file_name.startswith("span_")
    }
    assert set(span_samples) == {"span_lower", "span_middle", "span_upper"}
    assert [
        span_samples[name].start[2] for name in ("span_lower", "span_middle", "span_upper")
    ] == pytest.approx([-0.12, 0.0, 0.12])
    assert len(mesh.levels) - 1 == 8
    np.testing.assert_allclose(np.diff(mesh.levels), 0.06)
    assert vpm.numerics.viscous.particle_spacing == pytest.approx(0.08)
    assert vpm.numerics.viscous.core_radius_ratio == pytest.approx(1.2)
    assert vpm.numerics.max_n_particles == 50000
    assert vpm.numerics.time_step_size == pytest.approx(0.08)
    # Fast force/phase records cannot be finer than one accepted exchange.
    assert vpm.samplers.samples[0].schedule.interval == 1
    assert fvm.samplers[0].schedule.every_n_steps == 10
    profile = next(sample for sample in vpm.samplers.samples
                   if sample.file_name == "vpm_transverse_x2")
    assert profile.schedule.interval == 2
    assert vpm.run.steps % vpm.samplers.samples[0].schedule.interval == 0
    assert (vpm.numerics.induction.z_min, vpm.numerics.induction.z_max) == (-0.24, 0.24)
    assert coupler.eta_blend_width == pytest.approx(0.64)
    assert coupler.vpm_only_width == pytest.approx(0.16)
    assert coupler.interface_iterations == 3


def test_builder_rejects_exchange_clock_and_release_width_errors():
    module = _setup_module()
    with pytest.raises(ValueError, match="integer multiple"):
        module.build_case(overrides={"exchange_dt": 0.041})
    with pytest.raises(ValueError, match="release_width_ratio"):
        module.build_case(overrides={"blend_width_ratio": 2.0, "release_width_ratio": 2.0})
    with pytest.raises(ValueError, match="authority ramp begins inside"):
        module.build_case(overrides={"hxy": 0.08, "span": 0.96, "blend_width_ratio": 10.0})


def test_overridden_mesh_keeps_exact_slip_planes_when_dz_equals_hxy():
    module = _setup_module()
    _fvm, vpm, _coupler, mesh = module.build_case(overrides={"hxy": 0.064, "span": 0.96})
    assert isinstance(mesh, module.msh.ExtrudedCartesianMesher)
    assert mesh.domain.bounds[:4] == pytest.approx(mesh.source.domain.bounds[:4])
    assert mesh.domain.bounds[:4] == pytest.approx((-1.6, 1.6, -1.6, 1.6))
    assert mesh.levels[0] == pytest.approx(-0.48)
    assert mesh.levels[-1] == pytest.approx(0.48)
    assert vpm.numerics.induction.z_min == pytest.approx(-0.48)
    assert vpm.numerics.induction.z_max == pytest.approx(0.48)


@pytest.mark.parametrize("hxy", [0.1, 0.08, 0.064])
def test_grid_family_has_one_resolved_outer_xy_box(hxy):
    module = _setup_module()
    _fvm, _vpm, _coupler, mesh = module.build_case(overrides={"hxy": hxy})
    assert mesh.source.domain.bounds[:4] == pytest.approx((-1.6, 1.6, -1.6, 1.6))
    assert mesh.domain.bounds[:4] == pytest.approx((-1.6, 1.6, -1.6, 1.6))
