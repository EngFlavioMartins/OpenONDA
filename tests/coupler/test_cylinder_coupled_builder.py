"""Resolved cylinder study controls before solver allocation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from openonda.tutorial_runner import load_case_module


def _setup_module():
    path = (
        Path(__file__).resolve().parents[2]
        / "tutorials/coupled_fvm_vpm/01_cylinder_shedding_flow/setup.py"
    )
    return load_case_module(path.parent)


def test_builder_keeps_default_span_force_area_and_interface_iterations():
    module = _setup_module()
    fvm, vpm, coupler, mesh = module.build_case()
    assert fvm.samplers[0].reference_area == pytest.approx(1.0)
    assert coupler.interface_iterations == module.INTERFACE_ITERATIONS
    assert coupler.interface_normal_tolerance == pytest.approx(1e-5)
    assert coupler.interface_gradient_tolerance == pytest.approx(1e-5)
    assert coupler.freestream_velocity == [1.0, 0.1, 0.0]
    assert fvm.samplers[0].reference_velocity == pytest.approx(1.0)
    assert fvm.transport.kinematic_viscosity == pytest.approx(1 / 150)
    assert coupler.transfer_region_bounds[:4] == pytest.approx((-1.25, 2.05, -1.25, 1.25))
    assert isinstance(vpm.numerics.induction, module.vpm.SlipSlabInduction)
    assert vpm.numerics.induction.z_max - vpm.numerics.induction.z_min == pytest.approx(1.0)
    assert vpm.numerics.viscous.particle_spacing == pytest.approx(0.04)
    assert isinstance(mesh, module.msh.ExtrudedCartesianMesher)
    assert mesh.levels == (-0.5, 0.5)
    periodic = {patch.name: patch for patch in fvm.boundaries if patch.velocity_type == "cyclic"}
    assert set(periodic) == {"zmin", "zmax"}
    assert periodic["zmin"].neighbour_patch == "zmax"
    assert periodic["zmax"].neighbour_patch == "zmin"
    assert all(patch.pressure_type == "cyclic" for patch in periodic.values())
    assert vpm.numerics.compute_device == "AUTO"
    assert vpm.numerics.viscous.gbd_threshold == pytest.approx(0.01 * 0.04**3)


def test_builder_resolves_independent_span_and_in_plane_particle_spacing():
    module = _setup_module()
    fvm, vpm, coupler, mesh = module.build_case(
        overrides={
            "hxy": 0.08,
            "span": 0.48,
            "particle_spacing_ratio": 1.25,
            "core_radius_ratio": 1.2,
            "blend_width_ratio": 6.0,
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
    assert set(span_samples) == {"span_middle"}
    assert span_samples["span_middle"].start[2] == 0.0
    assert len(mesh.levels) - 1 == 1
    np.testing.assert_allclose(np.diff(mesh.levels), 0.48)
    assert vpm.numerics.viscous.particle_spacing == pytest.approx(0.1)
    assert vpm.numerics.viscous.core_radius_ratio == pytest.approx(1.2)
    assert vpm.numerics.max_n_particles == 50000
    assert vpm.numerics.time_step_size == pytest.approx(0.08)
    # Fast force/phase records cannot be finer than one accepted exchange.
    assert vpm.samplers.samples[0].schedule.interval == 1
    assert fvm.samplers[0].schedule.every_n_steps == 10
    profile = next(
        sample for sample in vpm.samplers.samples if sample.file_name == "vpm_transverse_x2"
    )
    assert profile.schedule.interval == 2
    assert vpm.run.steps % vpm.samplers.samples[0].schedule.interval == 0
    assert vpm.numerics.induction.z_max - vpm.numerics.induction.z_min == pytest.approx(0.48)
    assert vpm.numerics.viscous.gbd_threshold == pytest.approx(0.01 * 0.1**3)
    assert coupler.eta_blend_width == pytest.approx(0.6)
    assert coupler.vpm_only_width == pytest.approx(0.2)
    assert coupler.interface_iterations == module.INTERFACE_ITERATIONS


def test_overridden_mesh_keeps_one_periodic_layer_independent_of_xy_spacing():
    module = _setup_module()
    _fvm, vpm, _coupler, mesh = module.build_case(overrides={"hxy": 0.064, "span": 1.0})
    assert isinstance(mesh, module.msh.ExtrudedCartesianMesher)
    assert mesh.domain.bounds[:4] == pytest.approx(mesh.source.domain.bounds[:4])
    # Anchored Cartesian boxes expand to cell planes: 2.4 / .064 = 37.5.
    assert mesh.source.requested_domain.bounds[:4] == pytest.approx((-1.6, 2.4, -1.6, 1.6))
    assert mesh.domain.bounds[:4] == pytest.approx((-1.6, 2.432, -1.6, 1.6))
    assert mesh.levels == (-0.5, 0.5)
    assert vpm.numerics.induction.z_max - vpm.numerics.induction.z_min == pytest.approx(1.0)
    assert vpm.numerics.viscous.particle_spacing == pytest.approx(0.064)


@pytest.mark.parametrize("hxy, downstream", [(0.1, 2.4), (0.08, 2.4), (0.064, 2.432), (0.04, 2.4)])
def test_grid_family_reports_anchored_outer_xy_box(hxy, downstream):
    module = _setup_module()
    _fvm, _vpm, _coupler, mesh = module.build_case(overrides={"hxy": hxy})
    assert mesh.source.requested_domain.bounds[:4] == pytest.approx((-1.6, 2.4, -1.6, 1.6))
    assert mesh.source.domain.bounds[:4] == pytest.approx((-1.6, downstream, -1.6, 1.6))
    assert mesh.domain.bounds[:4] == pytest.approx((-1.6, downstream, -1.6, 1.6))
